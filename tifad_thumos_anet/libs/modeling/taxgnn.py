"""TaxGNN -- Stage-2 inductive taxonomy GNN (see STAGE1_LITERATURE_AND_
STAGE2_DESIGN.md, Stage 2).

Graph over the CURRENT class set (train classes in train mode, test classes
at eval): action leaves, taxonomy families, attribute values (TTB assets),
and per-action phase nodes (ensemble sentences 1..K-1). Edges are typed;
DGP-style direct leaf->ancestor edges collapse the phase->action->family
2-hop into an extra direct relation. Each relation type carries a learned
scalar attention-logit bias.

Node features: frozen CLIP text embedding of the node text (encoded ONCE
lazily, cached, no_grad, via TextFeatures.extract_text_emb) concat a
projected seen-class visual prototype (zero for nodes without one; the
projector has NO bias so "no prototype" contributes exactly nothing) concat
a has-prototype indicator bit, then a Linear to d_model.

Aggregator: 2 transformer layers (TrGCN-style trained aggregator), weights
SHARED across all nodes -- one multi-head attention per layer over the
whole node set with an additive attention mask that (a) restricts each
query to its graph neighbors + self and (b) adds the per-relation learned
scalar to the logit. (Attention is hand-rolled, not nn.MultiheadAttention:
see _TaxGNNLayer for the torch-2.0.1 float-mask corruption it avoids.)

Output: zero-init Linear d_model -> text_emb_dim producing RESIDUALS for
the action leaves only; the caller adds them to the (post-h19) class text
embeddings. Zero-init => exact init-equality with the h19 baseline.

Leaf-holdout (decisive trick vs failure mode (d)): during training, each
action node's prototype slot + indicator are zeroed with prob `holdout`,
so the aggregator is optimized for the exact text-only insert-new-node
operation used on unseen classes at test time.

Env wiring lives in libs/modeling/text.py (TIFAD_GNN=1, TIFAD_GNN_PROTOS,
TIFAD_GNN_HOLDOUT; graph assets reuse TIFAD_HTCA_ATTRS / TIFAD_HTCA_FAMS).
Inert when TIFAD_GNN is unset: this module is not even imported.
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# relation vocabulary (directed: row = query/receiver, col = key/sender)
REL_NAMES = [
    'self',                # 0  every node attends to itself
    'act<-fam',            # 1  action reads its family (ancestor)
    'fam<-act',            # 2  family reads its member actions
    'act<-attr',           # 3  action reads its attribute values
    'attr<-act',           # 4  attribute reads the actions carrying it
    'act<-phase',          # 5  action reads its phase children
    'phase<-act',          # 6  phase reads its action parent
    'phase<-fam_anc',      # 7  DGP direct: phase reads its distance-2
                           #    ancestor family (skip edge)
    'fam<-phase_desc',     # 8  DGP direct reverse: family reads its
                           #    distance-2 descendant phases
]
NUM_REL = len(REL_NAMES)


class _TaxGraph:
    """Static (per class-set) node/edge tables. Plain data, no parameters."""

    def __init__(self, cls_names, ensemble_prompts, attr_map, fam_map):
        self.cls_names = list(cls_names)
        node_texts, node_kind = [], []

        # --- action leaves (row i == cls_names[i]) ---
        for c in self.cls_names:
            if ensemble_prompts is not None and c in ensemble_prompts:
                node_texts.append(ensemble_prompts[c][0])   # class prompt
            else:
                node_texts.append(f'a video of action {c}')
            node_kind.append('action')
        self.action_rows = list(range(len(self.cls_names)))

        # --- family nodes (only families present in this class set) ---
        fams = sorted({fam_map[c] for c in self.cls_names if c in fam_map})
        fam_row = {}
        for f in fams:
            fam_row[f] = len(node_texts)
            node_texts.append(f'a video of action {f}')
            node_kind.append('family')

        # --- attribute-value nodes ---
        attrs = sorted({a for c in self.cls_names
                        for a in attr_map.get(c, [])})
        attr_row = {}
        for a in attrs:
            attr_row[a] = len(node_texts)
            node_texts.append(
                'an action characterized by ' + a.replace('_', ' '))
            node_kind.append('attribute')

        # --- phase nodes: ensemble sentences 1..K-1, children of the action
        phase_rows = {}                       # action row -> [phase rows]
        if ensemble_prompts is not None:
            for i, c in enumerate(self.cls_names):
                sents = ensemble_prompts.get(c, [])
                rows = []
                for s in sents[1:]:
                    rows.append(len(node_texts))
                    node_texts.append(s)
                    node_kind.append('phase')
                phase_rows[i] = rows

        self.node_texts = node_texts
        self.node_kind = node_kind
        N = len(node_texts)

        # --- typed adjacency: rel_mat[q, k] = relation id, -1 = no edge ---
        rel = torch.full((N, N), -1, dtype=torch.long)
        rel[torch.arange(N), torch.arange(N)] = 0            # self
        for i, c in enumerate(self.cls_names):
            fr = fam_row.get(fam_map.get(c))
            if fr is not None:
                rel[i, fr] = 1                               # act <- fam
                rel[fr, i] = 2                               # fam <- act
            for a in attr_map.get(c, []):
                ar = attr_row[a]
                rel[i, ar] = 3                               # act <- attr
                rel[ar, i] = 4                               # attr <- act
            for pr in phase_rows.get(i, []):
                rel[i, pr] = 5                               # act <- phase
                rel[pr, i] = 6                               # phase <- act
                if fr is not None:                           # DGP skip edges
                    rel[pr, fr] = 7                          # phase <- fam
                    rel[fr, pr] = 8                          # fam <- phase
        self.rel_mat = rel                                    # (N, N) cpu


class _TaxGNNLayer(nn.Module):
    """Shared-weight transformer aggregation over the neighbor set.

    NOTE: attention is hand-rolled (same qkv/out parameter structure as
    nn.MultiheadAttention) because torch 2.0.1's eval-mode fast path
    (_native_multi_head_attention) converts FLOAT attn_masks to BOOL
    (nonzero => hard-masked), which would corrupt the learned per-relation
    additive logit biases exactly at eval time. Explicit math guarantees
    additive semantics in every mode / torch version.
    """

    def __init__(self, d_model, n_heads=4, ffn_mult=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x, attn_bias):
        # x: (N, d); attn_bias: (N, N) float additive mask (-inf = no edge)
        N, d = x.shape
        H = self.n_heads
        hd = d // H
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(N, H, hd).transpose(0, 1)          # (H, N, hd)
        k = k.view(N, H, hd).transpose(0, 1)
        v = v.view(N, H, hd).transpose(0, 1)
        logits = q @ k.transpose(-1, -2) / hd ** 0.5  # (H, N, N)
        logits = logits + attn_bias.unsqueeze(0)
        attn = logits.softmax(dim=-1)
        h = (attn @ v).transpose(0, 1).reshape(N, d)  # (N, d)
        x = self.norm1(x + self.out(h))
        x = self.norm2(x + self.ffn(x))
        return x


class TaxGNN(nn.Module):
    def __init__(self, text_emb_dim, attrs_file, fams_file,
                 ensemble_prompts=None, protos_file='none', holdout=0.3,
                 d_model=None, n_layers=2, n_heads=4):
        super().__init__()
        d_model = d_model or text_emb_dim
        self.text_emb_dim = text_emb_dim
        self.holdout = float(holdout)
        self.ensemble_prompts = ensemble_prompts
        self.attr_map = json.load(open(attrs_file))
        self.fam_map = json.load(open(fams_file))

        # ---- seen visual prototypes (collect_protos.py output) ----
        self.protos = {}                       # class name -> (Dfeat,) cpu
        proto_dim = text_emb_dim               # fallback when no file
        if protos_file and protos_file != 'none':
            blob = torch.load(protos_file, map_location='cpu')
            P = blob['protos_seen'].float()    # (S, Dfeat)
            names = blob['seen_classes']
            cnt = blob.get('proto_counts',
                           torch.ones(len(names), dtype=torch.long))
            for i, nme in enumerate(names):
                if int(cnt[i]) > 0:
                    self.protos[nme] = P[i]
            proto_dim = P.shape[1]
            print(f"[TaxGNN] prototypes: {len(self.protos)} classes, "
                  f"dim {proto_dim}, from {protos_file}", flush=True)
        else:
            print("[TaxGNN] no prototypes (text-only nodes)", flush=True)
        self.proto_dim = proto_dim

        # prototype -> node feature space; bias=False so a zeroed prototype
        # slot contributes exactly nothing (heldout leaf == unseen leaf)
        self.proto_proj = nn.Linear(proto_dim, text_emb_dim, bias=False)
        # [text D | projected proto D | indicator 1] -> d_model
        self.in_proj = nn.Linear(2 * text_emb_dim + 1, d_model)
        self.layers = nn.ModuleList(
            [_TaxGNNLayer(d_model, n_heads) for _ in range(n_layers)])
        # per-relation learned scalar attention-logit biases (init 0)
        self.rel_weight = nn.Parameter(torch.zeros(NUM_REL))
        # ZERO-INIT residual head => init-equality with the h19 baseline.
        # Keep the zero init if you modify this block.
        self.out_head = nn.Linear(d_model, text_emb_dim)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

        # lazy per-class-set caches (plain dicts: rebuilt on demand, not
        # part of state_dict)
        self._graphs = {}                      # key -> _TaxGraph
        self._text_cache = {}                  # key -> (N, D) cpu tensor

    # ------------------------------------------------------------------
    def _graph(self, cls_names):
        key = tuple(cls_names)
        if key not in self._graphs:
            g = _TaxGraph(cls_names, self.ensemble_prompts,
                          self.attr_map, self.fam_map)
            self._graphs[key] = g
            kinds = g.node_kind
            print(f"[TaxGNN] graph({len(cls_names)} classes): "
                  f"{kinds.count('action')} actions, "
                  f"{kinds.count('family')} families, "
                  f"{kinds.count('attribute')} attributes, "
                  f"{kinds.count('phase')} phases; "
                  f"{int((g.rel_mat >= 0).sum())} typed edges", flush=True)
        return self._graphs[key]

    def _node_text_emb(self, graph, encode_fn, key):
        """Frozen CLIP embedding per node text; encoded once, cached (cpu)."""
        if key not in self._text_cache:
            embs = []
            with torch.no_grad():
                for i in range(0, len(graph.node_texts), 64):
                    chunk = graph.node_texts[i:i + 64]
                    embs.append(encode_fn(chunk).detach().float().cpu())
            self._text_cache[key] = torch.cat(embs, dim=0)   # (N, D)
        return self._text_cache[key]

    # ------------------------------------------------------------------
    def forward(self, cls_names, encode_fn, training=False):
        """Residuals (A, text_emb_dim) for the action leaves of cls_names.

        encode_fn: callable(list[str]) -> (n, D) CLIP text embeddings
                   (TextFeatures.extract_text_emb). Used only on the first
                   call per class set (lazy no_grad cache).
        training:  enables leaf-holdout prototype masking.
        """
        dev = self.rel_weight.device
        key = tuple(cls_names)
        g = self._graph(cls_names)
        N = len(g.node_texts)

        txt = self._node_text_emb(g, encode_fn, key).to(dev)     # (N, D)

        # prototype slab + indicator (actions with prototypes only)
        proto = txt.new_zeros(N, self.proto_dim)
        ind = txt.new_zeros(N, 1)
        for i, c in enumerate(cls_names):
            p = self.protos.get(c)
            if p is not None:
                proto[i] = p.to(dev)
                ind[i] = 1.0
        # leaf-holdout: simulate test-time insertion of a text-only leaf
        if training and self.holdout > 0:
            drop = (torch.rand(len(cls_names), device=dev) < self.holdout)
            drop = drop & (ind[:len(cls_names), 0] > 0)
            proto[:len(cls_names)][drop] = 0.0
            ind[:len(cls_names)][drop] = 0.0

        x = self.in_proj(
            torch.cat([txt, self.proto_proj(proto), ind], dim=-1))  # (N, d)

        # additive attention mask: -inf off-graph, learned scalar per
        # relation on-graph (diagonal always on => no all-masked rows)
        rel = g.rel_mat.to(dev)                                     # (N, N)
        bias = torch.where(
            rel >= 0,
            self.rel_weight[rel.clamp(min=0)],
            torch.tensor(float('-inf'), device=dev),
        )

        for layer in self.layers:
            x = layer(x, bias)

        resid = self.out_head(x[g.action_rows])          # (A, text_emb_dim)
        return resid


# ======================================================================
# TaxGNN v3 (TIFAD_GNN3=1) -- instance-level graph + trained embeddings +
# per-class / per-snippet gating. See STAGE1_LITERATURE_AND_STAGE2_DESIGN.md
# "TaxGNN v3". Differences vs TaxGNN (v1/v2) above:
#   (i)  INSTANCE nodes: the IN-BATCH GT segments (live level-0 features
#        mean-pooled inside each segment -- gradients flow to the backbone),
#        joined to their action leaf by label edges. At most K per class per
#        forward; per-CLASS instance dropout p (drop ALL of a class's
#        instances) trains the exact text-only insertion condition unseen
#        classes face at eval (where no instance nodes exist at all).
#   (ii) TRAINED embeddings: a zero-init residual adapter Linear on top of
#        the cached (no_grad) CLIP node-text embeddings -- identity at step
#        0, trainable thereafter (the sanctioned cheaper alternative to
#        unfreezing the CLIP text tower).
#   (iii) TWO residual outputs, BOTH zero-init and gated:
#        - class-embedding residuals for the action leaves (out_head), and
#        - a snippet-feature residual via graph-conditioned cross-attention
#          (snip_attn over the TEXT-node outputs; gradients reach the
#          backbone through the queries AND through the instance nodes).
#        Gates: sigmoid(Linear on [cos, scaled-dot, scaled-key-norm]
#        similarity features of (text-emb/snippet, graph-emb) pairs);
#        zero weight + bias=logit(0.1) => gate == 0.1 exactly at init.
#        Residual heads are zero-init, so init-equality holds regardless.
# Wiring lives in meta_archs.py (constructed on TiFAD so instance features
# are reachable). Inert when TIFAD_GNN3 unset.
# DOC-AMBIG decisions (task spec left open):
#   - snippet cross-attention keys/values = TEXT-node outputs only (not
#     instance nodes), keeping the eval-time key set identical to train;
#     instances influence those keys via message passing.
#   - instance "phase description" feature omitted (phase nodes already sit
#     in the graph one hop away); instance features = [segment feature |
#     adapted label text emb | start/end normalized timestamps].
# ======================================================================

REL3_NAMES = REL_NAMES + [
    'act<-inst',           # 9   action reads its instance clips
    'inst<-act',           # 10  instance reads its action leaf
]
NUM_REL3 = len(REL3_NAMES)


class _TaxGNN3Layer(nn.Module):
    """HGT-style relation-aware transformer aggregation (v3 upgrade over
    _TaxGNNLayer's per-relation scalars): every edge TYPE carries a
    learnable embedding (shared across layers, owned by TaxGNN3) that
    PARTICIPATES IN THE ATTENTION SCORE through a per-layer projection into
    per-head key space, plus a per-layer per-relation per-head scalar bias:

        score(i,j) = q_i.k_j / sqrt(hd)
                   + q_i . W_rel e_{rel(i,j)} / sqrt(hd)   (relation-aware
                                                            key term)
                   + b[rel(i,j), h]                        (scalar bias)

    computed as einsum('hnd,rhd->hnr') then a gather over the (N, N)
    relation-id matrix -- O(N*R*d) extra work, never materializing per-pair
    relation keys. Off-graph pairs (rel id -1) are hard-masked to -inf; the
    diagonal always carries the 'self' relation so no row is fully masked.
    """

    def __init__(self, d_model, n_rel, n_heads=4, ffn_mult=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        # per-layer projection of the shared relation embeddings into
        # per-head key space + per-layer per-relation per-head scalar bias
        self.rel_proj = nn.Linear(d_model, d_model, bias=False)
        self.rel_bias = nn.Parameter(torch.zeros(n_rel, n_heads))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x, rel_mat, rel_emb):
        # x: (N, d); rel_mat: (N, N) long, -1 = no edge; rel_emb: (R, d)
        N, d = x.shape
        H = self.n_heads
        hd = d // H
        R = rel_emb.size(0)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(N, H, hd).transpose(0, 1)          # (H, N, hd)
        k = k.view(N, H, hd).transpose(0, 1)
        v = v.view(N, H, hd).transpose(0, 1)
        logits = q @ k.transpose(-1, -2) / hd ** 0.5  # (H, N, N)

        safe = rel_mat.clamp(min=0)                   # (N, N)
        rk = self.rel_proj(rel_emb).view(R, H, hd)    # (R, H, hd)
        rel_scores = torch.einsum('hnd,rhd->hnr', q, rk) / hd ** 0.5
        idx = safe.unsqueeze(0).expand(H, -1, -1)     # (H, N, N)
        logits = logits + rel_scores.gather(-1, idx)  # relation-aware keys
        logits = logits + self.rel_bias[safe].permute(2, 0, 1)
        logits = logits.masked_fill(
            (rel_mat < 0).unsqueeze(0), float('-inf'))

        attn = logits.softmax(dim=-1)
        h = (attn @ v).transpose(0, 1).reshape(N, d)  # (N, d)
        x = self.norm1(x + self.out(h))
        x = self.norm2(x + self.ffn(x))
        return x


class TaxGNN3(nn.Module):
    def __init__(self, emb_dim, attrs_file, fams_file, ensemble_prompts=None,
                 d_model=None, n_layers=3, n_heads=4,
                 inst_per_class=2, inst_drop=0.3):
        super().__init__()
        d_model = d_model or emb_dim
        assert d_model == emb_dim, \
            "[TaxGNN3] d_model must equal emb_dim (snippet queries attend " \
            "in the updated_feat space)"
        self.emb_dim = emb_dim
        self.inst_per_class = int(inst_per_class)
        self.inst_drop = float(inst_drop)
        self.ensemble_prompts = ensemble_prompts
        self.attr_map = json.load(open(attrs_file))
        self.fam_map = json.load(open(fams_file))

        # trained embeddings: zero-init residual adapter over the cached
        # CLIP node-text embeddings (identity at init)
        self.text_adapter = nn.Linear(emb_dim, emb_dim)
        nn.init.zeros_(self.text_adapter.weight)
        nn.init.zeros_(self.text_adapter.bias)

        self.in_proj = nn.Linear(emb_dim, d_model)
        # instance node: [seg feature D | adapted label text emb D |
        #                 start | end (normalized)] -> d_model
        self.inst_proj = nn.Linear(2 * emb_dim + 2, d_model)
        # HGT-style relation-aware attention: one learnable embedding per
        # edge TYPE, shared across the 3 layers (each layer projects it into
        # per-head key space; see _TaxGNN3Layer). 3 layers = 3 hops (covers
        # sibling-via-parent 2-hop chains and cross-level action-family-
        # attribute chains) while the DGP-style direct ancestor skip edges
        # from _TaxGraph are kept. N stays ~300 even with in-batch instance
        # nodes, so 3 layers is cheap -- the 2-layer + explicit-sibling-edge
        # fallback was not needed.
        self.rel_emb = nn.Embedding(NUM_REL3, d_model)
        nn.init.normal_(self.rel_emb.weight, std=0.02)
        self.layers = nn.ModuleList(
            [_TaxGNN3Layer(d_model, NUM_REL3, n_heads)
             for _ in range(n_layers)])

        _gate_bias = -2.1972245773362196          # logit(0.1)
        # (i) class-embedding residual head + per-class gate
        self.out_head = nn.Linear(d_model, emb_dim)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)
        self.cls_gate = nn.Linear(3, 1)
        nn.init.zeros_(self.cls_gate.weight)
        nn.init.constant_(self.cls_gate.bias, _gate_bias)
        # (ii) snippet residual: graph-conditioned cross-attention (standard
        # init -- the ZERO init lives in snip_head so the context stays
        # informative for the gate) + zero-init head + per-snippet gate
        self.snip_attn = nn.MultiheadAttention(d_model, n_heads,
                                               batch_first=True)
        self.snip_head = nn.Linear(d_model, emb_dim)
        nn.init.zeros_(self.snip_head.weight)
        nn.init.zeros_(self.snip_head.bias)
        self.snip_gate = nn.Linear(3, 1)
        nn.init.zeros_(self.snip_gate.weight)
        nn.init.constant_(self.snip_gate.bias, _gate_bias)

        # lazy per-class-set caches (not part of state_dict)
        self._graphs = {}
        self._text_cache = {}

    # ------------------------------------------------------------------
    def _graph(self, cls_names):
        key = tuple(cls_names)
        if key not in self._graphs:
            g = _TaxGraph(cls_names, self.ensemble_prompts,
                          self.attr_map, self.fam_map)
            self._graphs[key] = g
            kinds = g.node_kind
            print(f"[TaxGNN3] graph({len(cls_names)} classes): "
                  f"{kinds.count('action')} actions, "
                  f"{kinds.count('family')} families, "
                  f"{kinds.count('attribute')} attributes, "
                  f"{kinds.count('phase')} phases; "
                  f"{int((g.rel_mat >= 0).sum())} typed edges", flush=True)
        return self._graphs[key]

    def _node_text_emb(self, graph, encode_fn, key):
        """Cached no_grad CLIP embedding per node text (cpu); the trainable
        text_adapter is applied ON TOP of this cache in forward()."""
        if key not in self._text_cache:
            embs = []
            with torch.no_grad():
                for i in range(0, len(graph.node_texts), 64):
                    chunk = graph.node_texts[i:i + 64]
                    embs.append(encode_fn(chunk).detach().float().cpu())
            self._text_cache[key] = torch.cat(embs, dim=0)   # (N, D)
        return self._text_cache[key]

    @staticmethod
    def gate_feats(a, b):
        """Similarity features between paired embeddings a, b (..., D):
        [cosine, dot/sqrt(D), ||b||/sqrt(D)] -> (..., 3)."""
        d = a.size(-1)
        cos = F.cosine_similarity(a, b, dim=-1)
        dot = (a * b).sum(-1) / d ** 0.5
        nb = b.norm(dim=-1) / d ** 0.5
        return torch.stack([cos, dot, nb], dim=-1)

    # ------------------------------------------------------------------
    def forward(self, cls_names, encode_fn,
                inst_feats=None, inst_labels=None, inst_se=None,
                training=False):
        """Run the graph. Returns (act_nodes (A, d), text_nodes (N0, d),
        cls_resid (A, emb_dim)).

        inst_feats:  (M, D) LIVE segment features (train only; None at eval
                     -- test classes join text-only, the trained insertion
                     condition).
        inst_labels: (M,) long, index into cls_names.
        inst_se:     (M, 2) float, start/end normalized to [0, 1].
        """
        dev = self.rel_emb.weight.device
        key = tuple(cls_names)
        g = self._graph(cls_names)
        N0 = len(g.node_texts)

        txt = self._node_text_emb(g, encode_fn, key).to(dev)     # (N0, D)
        txt = txt + self.text_adapter(txt)      # trained embs (id at init)
        x = self.in_proj(txt)                                    # (N0, d)

        rel = g.rel_mat.to(dev)                                  # (N0, N0)
        M = 0 if inst_feats is None else inst_feats.size(0)
        if M > 0:
            lab = inst_labels.to(dev)
            xi = self.inst_proj(torch.cat(
                [inst_feats, txt[lab], inst_se.to(txt.dtype)], dim=-1))
            x = torch.cat([x, xi], dim=0)                        # (N0+M, d)
            N = N0 + M
            rel_full = torch.full((N, N), -1, dtype=torch.long, device=dev)
            rel_full[:N0, :N0] = rel
            ar = torch.arange(M, device=dev)
            rel_full[N0 + ar, N0 + ar] = 0                       # self
            rel_full[lab, N0 + ar] = 9                           # act<-inst
            rel_full[N0 + ar, lab] = 10                          # inst<-act
            rel = rel_full

        for layer in self.layers:
            x = layer(x, rel, self.rel_emb.weight)

        text_nodes = x[:N0]
        act_nodes = x[g.action_rows]                     # (A, d)
        cls_resid = self.out_head(act_nodes)             # (A, emb_dim)
        return act_nodes, text_nodes, cls_resid
