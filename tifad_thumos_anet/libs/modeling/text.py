import torch
import torch.nn as nn
from transformers import CLIPTokenizer  # pip install transformers==4.19.2
from .clip import CLIPModel

def data_split_dir(file, data_split, mode, split_num):
        if mode == 'train':
            file_dir = file.format(mode='train',r1=data_split, r2=100-data_split, r3=split_num)
        else:
            file_dir = file.format(mode='test',r1=data_split, r2=100-data_split, r3=split_num)
        return file_dir

class TextFeatures(nn.Module):
    def __init__(
        self,
        model_path,
        subset_file,
        data_split,
        emb_dim,
        split_num,
        freeze_txt_model=True,
    ):
        super().__init__()
        self.train_classes = self._load_classes(subset_file, data_split, 'train', split_num)
        self.test_classes = self._load_classes(subset_file, data_split, 'test', split_num)

        self.tokenizer = CLIPTokenizer.from_pretrained(model_path)
        self.txt_model = CLIPModel.from_pretrained(model_path, ignore_mismatched_sizes=True).float()

        if isinstance(self.txt_model.text_projection, nn.Linear):
            self.txt_model.text_projection = nn.Linear(self.txt_model.text_embed_dim, emb_dim, bias=False)
            nn.init.normal_(self.txt_model.text_projection.weight, std=self.txt_model.text_embed_dim ** -0.5)
        else:
            self.txt_model.text_projection = nn.Parameter(torch.empty(self.txt_model.text_embed_dim, emb_dim))
            nn.init.normal_(self.txt_model.text_projection, std=self.txt_model.text_embed_dim ** -0.5)


        if freeze_txt_model:
            self.txt_model.requires_grad_(False)
            self.txt_model.text_projection.requires_grad_(True)

        # ---------------- h19 (four text-side switches) ----------------
        # H4a  TIFAD_PROMPT_ENSEMBLE : per-class sentence ensemble
        # H4b  TIFAD_ATTR_ATTN=1     : zero-init class->description cross-attention
        # h14  TIFAD_PROMPT_DROPOUT  : KV dropout on description tokens (train only)
        # h18  TIFAD_SEM_DEGRADE     : replace a class's text by its PARENT STEM, p per iter
        import os as _os, json as _json
        self.ensemble_prompts = None
        self.attr_attn = None
        self.prompt_drop = float(_os.environ.get('TIFAD_PROMPT_DROPOUT', '0'))
        self.sem_degrade = float(_os.environ.get('TIFAD_SEM_DEGRADE', '0'))
        self._stems = None                      # lazy: parent stems for h18
        _ep = _os.environ.get('TIFAD_PROMPT_ENSEMBLE', '')
        if _ep:
            self.ensemble_prompts = _json.load(open(_ep))
            n_sent = len(next(iter(self.ensemble_prompts.values())))
            print(f"[H4a] prompt ensembles active: {len(self.ensemble_prompts)} classes "
                  f"x {n_sent} sentences")
            if _os.environ.get('TIFAD_ATTR_ATTN', '') == '1':
                self.attr_attn = nn.MultiheadAttention(emb_dim, 4, batch_first=True)
                # ZERO-INIT out-proj: at step 0 the attention contributes exactly 0,
                # so the model is bit-identical to using sentence[0] alone. This is the
                # init-equality invariant -- keep it if you modify this block.
                nn.init.zeros_(self.attr_attn.out_proj.weight)
                nn.init.zeros_(self.attr_attn.out_proj.bias)
                print("[H4b] zero-init class->description cross-attention active")
        # FAMTIE (senone-style tying): trained zero-init per-family residual,
        # registered here so the optimizer sees it. Zero-init => init-equality.
        if int(_os.environ.get('TIFAD_FAMTIE', '0') or 0):
            _fmap = _json.load(open(_os.environ['TIFAD_FAMTIE_MAP']))
            _fams = sorted(set(_fmap.values()))
            self._famtie_fid = {f: i for i, f in enumerate(_fams)}
            self._famtie_map = _fmap
            self.famtie_emb = nn.Embedding(len(_fams), emb_dim)
            nn.init.zeros_(self.famtie_emb.weight)
            print(f"[FAMTIE] trained family residuals: {len(_fams)} families",
                  flush=True)

        # ---- HTCA / HAF hierarchy anchor banks (TTB assets): trained
        # embeddings over the attribute-VALUE vocabulary and the taxonomy
        # FAMILY vocabulary, shared across classes so unseen classes inherit
        # them. SMALL-RANDOM init (std 0.02): these are attention keys (HTCA)
        # / cosine anchors (HAF) and need signal; init-equality is guaranteed
        # downstream instead (HTCA: zero-init attention out_proj; HAF: loss-
        # only pathway). Registered here as nn.Embedding so make_optimizer
        # sees them (FAMTIE lesson). Envs: TIFAD_HTCA=1 and/or TIFAD_HAF=<w>,
        # plus TIFAD_HTCA_ATTRS=<json {class: [attr value names]}> and
        # TIFAD_HTCA_FAMS=<json {class: family}>. Inert when unset.
        self.htca_attr_emb = None
        self.htca_fam_emb = None
        _htca_on = _os.environ.get('TIFAD_HTCA', '') == '1'
        _haf_w = float(_os.environ.get('TIFAD_HAF', '0') or 0)
        if _htca_on or _haf_w != 0:
            _attrs = _json.load(open(_os.environ['TIFAD_HTCA_ATTRS']))
            _hfmap = _json.load(open(_os.environ['TIFAD_HTCA_FAMS']))
            _vocab = sorted({a for vs in _attrs.values() for a in vs})
            _hfams = sorted(set(_hfmap.values()))
            _vid = {a: i for i, a in enumerate(_vocab)}
            _hfid = {f: i for i, f in enumerate(_hfams)}
            # per-class token index lists into the two banks
            self._htca_attr_of = {c: [_vid[a] for a in vs]
                                  for c, vs in _attrs.items()}
            self._htca_fam_of = {c: _hfid[f] for c, f in _hfmap.items()}
            self.htca_attr_emb = nn.Embedding(len(_vocab), emb_dim)
            self.htca_fam_emb = nn.Embedding(len(_hfams), emb_dim)
            nn.init.normal_(self.htca_attr_emb.weight, std=0.02)
            nn.init.normal_(self.htca_fam_emb.weight, std=0.02)
            print(f"[HTCA/HAF] hierarchy banks: V={len(_vocab)} attribute "
                  f"values, F={len(_hfams)} families (std-0.02 init)",
                  flush=True)

        # ---- TaxGNN (Stage-2, see STAGE1_LITERATURE_AND_STAGE2_DESIGN.md):
        # inductive 2-layer transformer-aggregator GNN over the taxonomy
        # graph (actions / families / attributes / phases), producing
        # ZERO-INIT residuals added to the (post-h19) class embeddings in
        # forward(). Constructed HERE so make_optimizer sees the params
        # (FAMTIE lesson; TextFeatures is a submodule of TiFAD). Envs:
        # TIFAD_GNN=1, TIFAD_GNN_PROTOS=<collect_protos .pt | 'none'>,
        # TIFAD_GNN_HOLDOUT=<mask prob, default 0.3>; graph assets reuse
        # TIFAD_HTCA_ATTRS / TIFAD_HTCA_FAMS. Inert (byte-identical) when
        # TIFAD_GNN is unset -- taxgnn is not even imported.
        self.taxgnn = None
        if _os.environ.get('TIFAD_GNN', '') == '1':
            from .taxgnn import TaxGNN
            self.taxgnn = TaxGNN(
                text_emb_dim=emb_dim,
                attrs_file=_os.environ['TIFAD_HTCA_ATTRS'],
                fams_file=_os.environ['TIFAD_HTCA_FAMS'],
                ensemble_prompts=self.ensemble_prompts,
                protos_file=_os.environ.get('TIFAD_GNN_PROTOS', 'none'),
                holdout=float(_os.environ.get('TIFAD_GNN_HOLDOUT', '0.3')),
            )
            print(f"[TaxGNN] active: holdout="
                  f"{self.taxgnn.holdout}, d_model={emb_dim}", flush=True)

        # ---- IDEAS-1 (three-part program; all inert when envs unset) ----
        # PART 1  TIFAD_NODEATTR=action|action_clean|all : attributes as NODE
        #   FEATURES -- per-class mean CLIP text emb of the class's attribute-
        #   value names (cached no_grad), injected via a ZERO-INIT Linear
        #   residual. 'action'/'action_clean' inject into the final class
        #   embedding only ('action_clean' is code-identical; the runner pairs
        #   it with configs/thumos14_qwen_ensemble_noattr.json whose sentence 0
        #   is the attr-free parent stem -- with plain 'action' the sentence-0
        #   attr clauses intentionally STAY). 'all' injects the same residual
        #   uniformly into per[:, 0] AND per[:, 1:] pre-H4b (documented
        #   fallback for the activity level: the h18 degrade path replaces the
        #   SENTENCES inside `per` by the parent stem, so the uniform pre-H4b
        #   injection covers that representation too).
        # PART 2  TIFAD_ATTNDESIGN=ownphase|phasechain|allphase|act2act|
        #   cascade : attention designs around H4b, every new module has a
        #   ZERO-INIT out_proj so init-equality with stock H4b holds exactly.
        #   'ownphase' = control alias (no code path change).
        # PART 3  TIFAD_LMA=1 : Laban Movement Analysis components (configs/
        #   thumos14_lma.json) encoded as mean CLIP text emb of
        #   'movement with <value> ...' phrases, injected via a second
        #   ZERO-INIT Linear residual into the class embedding (shares the
        #   PART-1 mechanism; composes with TIFAD_NODEATTR).
        self.nodeattr = _os.environ.get('TIFAD_NODEATTR', '')
        self.lma_on = _os.environ.get('TIFAD_LMA', '') == '1'
        self.attn_design = _os.environ.get('TIFAD_ATTNDESIGN', '')
        if self.attn_design == 'ownphase':      # control alias == stock H4b
            self.attn_design = ''
        if self.attn_design:
            assert self.attn_design in ('phasechain', 'allphase', 'act2act',
                                        'cascade'), self.attn_design
            assert self.attr_attn is not None, \
                "[ATTNDESIGN] needs TIFAD_PROMPT_ENSEMBLE + TIFAD_ATTR_ATTN=1"
            if self.attn_design in ('phasechain', 'cascade'):
                self.chain_attn = nn.MultiheadAttention(emb_dim, 4,
                                                        batch_first=True)
                nn.init.zeros_(self.chain_attn.out_proj.weight)
                nn.init.zeros_(self.chain_attn.out_proj.bias)
            if self.attn_design in ('act2act', 'cascade'):
                self.act_attn = nn.MultiheadAttention(emb_dim, 4,
                                                      batch_first=True)
                nn.init.zeros_(self.act_attn.out_proj.weight)
                nn.init.zeros_(self.act_attn.out_proj.bias)
                _fmap2 = _json.load(open(_os.environ.get(
                    'TIFAD_FAM_MAP', 'configs/thumos14_fam_map.json')))
                self._activities = sorted(set(_fmap2.values()))
                self._act_cache = None
            print(f"[ATTNDESIGN] '{self.attn_design}' active "
                  f"(zero-init residuals)", flush=True)

        # ---- SUB-ENSEMBLE (TIFAD_SUB_ENSEMBLE=<json {class: [[sub_1..
        # sub_Kp] x K]}>): a SECOND level of sub-actions ("sub-phases")
        # below the K phase sentences of TIFAD_PROMPT_ENSEMBLE -- the same
        # ASA block one level down. Per phase k of class c the Kp sub-phase
        # sentences are embedded (same CLIP forward path, same h18 degrade
        # rule: a degraded class's sub-sentences become its parent stem
        # too), refined by banded neighbour self-attention among the Kp
        # sub-phases (`chain_attn_sub`, only with TIFAD_ATTNDESIGN=
        # phasechain|cascade, |i-j| <= 1), then the phase embedding
        # per[c, 1+k] (query) attends over [phase; refined sub-phases]
        # (`attr_attn_sub`, 1+Kp keys; h14 TIFAD_PROMPT_DROPOUT drops
        # sub-phase keys in train mode, never the query). The result
        # REPLACES per[:, 1:] BEFORE the existing phasechain / H4b code
        # runs unchanged (see forward: `_sub_refine`). Both out_proj's are
        # ZERO-INIT => the refined phases equal the plain ones bitwise at
        # step 0 (init-equality invariant -- keep it). Module names carry
        # the upper-level substrings ('attr_attn', 'chain_attn') so
        # TIFAD_FREEZE_EXCEPT=chain_attn / attr_attn select both levels.
        # Requires TIFAD_PROMPT_ENSEMBLE + TIFAD_ATTR_ATTN=1. Byte-inert
        # when unset (no modules, forward untouched).
        self.sub_prompts = None
        self.attr_attn_sub = None
        self.chain_attn_sub = None
        self._sub_kp = 0
        _se = _os.environ.get('TIFAD_SUB_ENSEMBLE', '')
        if _se:
            assert self.attr_attn is not None, \
                "[SUB] needs TIFAD_PROMPT_ENSEMBLE + TIFAD_ATTR_ATTN=1"
            self.sub_prompts = _json.load(open(_se))
            _se_k = len(next(iter(self.ensemble_prompts.values()))) - 1
            _se_missing = sorted(set(self.ensemble_prompts)
                                 - set(self.sub_prompts))
            assert not _se_missing, \
                f"[SUB] classes missing from {_se}: {_se_missing[:5]}"
            _se_kps = set()
            for _se_c in self.ensemble_prompts:
                _se_lists = self.sub_prompts[_se_c]
                assert len(_se_lists) == _se_k, \
                    (f"[SUB] {_se_c}: {len(_se_lists)} sub-phase lists "
                     f"!= K={_se_k} phases")
                _se_kps.update(len(_l) for _l in _se_lists)
            assert len(_se_kps) == 1, \
                f"[SUB] Kp must be uniform, got {sorted(_se_kps)}"
            self._sub_kp = _se_kps.pop()
            assert self._sub_kp >= 1, "[SUB] Kp must be >= 1"
            self.attr_attn_sub = nn.MultiheadAttention(emb_dim, 4,
                                                       batch_first=True)
            nn.init.zeros_(self.attr_attn_sub.out_proj.weight)
            nn.init.zeros_(self.attr_attn_sub.out_proj.bias)
            if self.attn_design in ('phasechain', 'cascade'):
                self.chain_attn_sub = nn.MultiheadAttention(emb_dim, 4,
                                                            batch_first=True)
                nn.init.zeros_(self.chain_attn_sub.out_proj.weight)
                nn.init.zeros_(self.chain_attn_sub.out_proj.bias)
            print(f"[SUB] sub-phase ensemble active: "
                  f"{len(self.sub_prompts)} classes x {_se_k} phases x "
                  f"Kp={self._sub_kp} sub-phases; attr_attn_sub (phase -> "
                  f"sub-phases, zero-init)"
                  + (", chain_attn_sub (banded sub-phase neighbours, "
                     "zero-init)" if self.chain_attn_sub is not None
                     else ", no chain_attn_sub (TIFAD_ATTNDESIGN is not "
                          "phasechain)"), flush=True)

        # ---- ORDER CONTROLS (does ORDERED composition
        # of the K child sentences matter?). Both byte-inert when unset.
        # (1) TIFAD_SHUFFLE_CHILDREN=eval|train|both : permute the K child
        #     sentences per[:, 1:] of every class along the phase axis
        #     (root per[:, 0] stays at index 0) BEFORE anything reads the
        #     bank (chain_attn / attr_attn / the ordered phase stashes).
        #     'eval' : a FIXED permutation per class drawn once at
        #              construction from torch.Generator seeded with
        #              TIFAD_SHUFFLE_SEED (default 0), in sorted class-name
        #              order -> deterministic across forwards AND across
        #              constructions; applied when mode != 'train'.
        #     'train': a fresh torch.randperm per class per forward when
        #              mode == 'train' (global torch RNG).
        #     'both' : both.
        #     With TIFAD_SUB_ENSEMBLE the permutation is applied AFTER
        #     _sub_refine, i.e. to the sub-refined phases: every phase moves
        #     together with its own Kp sub-children (consistent phase axis).
        #     Zero-init chain/attr out_projs => the shuffle cannot change the
        #     output (the root sentence alone determines it). Incompatible
        #     with TIFAD_LEAFLOSS=1 in the train modes (its phase targets
        #     index the ORDERED bank) -- asserted.
        # (2) TIFAD_CHAIN_FULL=1 : the 'phasechain'/'cascade' chain_attn
        #     runs with NO attention mask (every child attends to every
        #     child) in train and eval; the sub-level chain_attn_sub and
        #     everything else are unchanged. Honoured wherever chain_attn
        #     runs (training included), so the fair mask ablation retrains
        #     stage 2 with this env; evaluating a band-trained checkpoint
        #     under it is a train/test mismatch by design.
        self.shuffle_children = _os.environ.get('TIFAD_SHUFFLE_CHILDREN', '')
        self._shuffle_perm = None
        if self.shuffle_children:
            assert self.shuffle_children in ('eval', 'train', 'both'), \
                f"[SHUFFLE] TIFAD_SHUFFLE_CHILDREN={self.shuffle_children!r}"
            assert self.ensemble_prompts is not None, \
                "[SHUFFLE] needs TIFAD_PROMPT_ENSEMBLE (child sentences)"
            _shk = len(next(iter(self.ensemble_prompts.values()))) - 1
            assert _shk >= 2, "[SHUFFLE] ensemble needs >= 2 children/class"
            if self.shuffle_children != 'eval':
                assert _os.environ.get('TIFAD_LEAFLOSS', '') != '1', \
                    "[SHUFFLE] train-mode shuffle is incompatible with " \
                    "TIFAD_LEAFLOSS=1 (ordered phase targets)"
            _shseed = int(_os.environ.get('TIFAD_SHUFFLE_SEED', '0') or 0)
            _shmoved = 0
            if self.shuffle_children != 'train':
                # fixed per-class eval permutation ('eval' / 'both' only)
                _shg = torch.Generator().manual_seed(_shseed)
                self._shuffle_perm = {
                    c: torch.randperm(_shk, generator=_shg)
                    for c in sorted(self.ensemble_prompts)}
                _shmoved = sum(int((p != torch.arange(_shk)).sum())
                               for p in self._shuffle_perm.values())
            print(f"[SHUFFLE] child-order control '{self.shuffle_children}': "
                  f"K={_shk} children/class"
                  + (f"; fixed eval permutation per class (seed {_shseed}, "
                     f"{_shmoved}/{_shk * len(self._shuffle_perm)} children "
                     f"displaced)" if self.shuffle_children != 'train'
                     else "")
                  + ("; fresh random permutation per class per train "
                     "forward" if self.shuffle_children != 'eval' else ""),
                  flush=True)
        self.chain_full = _os.environ.get('TIFAD_CHAIN_FULL', '') == '1'
        if self.chain_full:
            assert self.attn_design in ('phasechain', 'cascade'), \
                "[CHAIN-FULL] needs TIFAD_ATTNDESIGN=phasechain|cascade"
            print("[CHAIN-FULL] chain_attn runs with NO neighbour mask "
                  "(full child-to-child attention, train and eval)",
                  flush=True)

        # ---- TXT_DEANISO (TIFAD_TXT_DEANISO=mean|pca<k>|whiten): remove
        # the anisotropic (common) component of the CLIP PHASE-sentence
        # embeddings before they enter the 'phasechain'/'cascade' sub-action
        # self-attention (Eq. order). The phase embeddings are nearly
        # collinear across phases AND classes (leafloss: text-cosine phase
        # alignment at chance), so the banded self-attention has little
        # within-action contrast to work with. Parameter-free statistics,
        # recomputed every forward from the stacked phases of ALL classes
        # in the current vocabulary (detached):
        #   mean    : subtract the mean phase embedding
        #   pca<k>  : mean, then also remove the projection onto the top-k
        #             principal components of the centred stack (SVD)
        #   whiten  : mean, then divide by the per-dimension std
        # then every phase embedding is rescaled back to its ORIGINAL L2
        # norm. Applied identically at train and eval; the action sentence
        # p(c) (bank index 0) is untouched. TIFAD_TXT_DEANISO_SCOPE (default
        # 'attn'): 'attn' feeds the transformed phases ONLY to the
        # self-attention's Q/K/V -- its zero-init residual is added to the
        # ORIGINAL phases, so init-equality with the stage-1 checkpoint and
        # the frozen H4b cross-attention's input distribution both hold
        # exactly; 'all' replaces the phases downstream as well (H4b keys +
        # the LEAFLOSS/VSA stashes). Byte-inert when unset. Once per mode
        # the mean pairwise phase cosine (within a class / across classes)
        # is printed before and after the transform (see _deaniso).
        self.txt_deaniso = _os.environ.get('TIFAD_TXT_DEANISO', '')
        self.txt_deaniso_k = 0
        self.txt_deaniso_scope = _os.environ.get('TIFAD_TXT_DEANISO_SCOPE',
                                                 'attn')
        self._deaniso_printed = set()
        if self.txt_deaniso:
            assert self.attn_design in ('phasechain', 'cascade'), \
                "[DEANISO] needs TIFAD_ATTNDESIGN=phasechain|cascade"
            assert self.txt_deaniso_scope in ('attn', 'all'), \
                self.txt_deaniso_scope
            if self.txt_deaniso.startswith('pca'):
                self.txt_deaniso_k = int(self.txt_deaniso[3:])
                assert self.txt_deaniso_k >= 1, self.txt_deaniso
            else:
                assert self.txt_deaniso in ('mean', 'whiten'), \
                    self.txt_deaniso
            print(f"[DEANISO] '{self.txt_deaniso}' on the phase bank before "
                  f"the sub-action self-attention (scope="
                  f"{self.txt_deaniso_scope}, norm-preserving, no params)",
                  flush=True)

        # ---- LEVELATTN (TIFAD_LEVELATTN=1): unified level-aware attention
        # in the H4b block (tree positional embeddings, arXiv:2507.04003).
        # (a) LEARNED level embeddings added to the (A, K, D) bank tokens
        # pre-attention: level 0 = class sentence (index 0), level 1 =
        # phases (indices 1..K-1) -- nn.Embedding(2, D) ZERO-INIT; (b) a
        # learned phase-ORDER embedding nn.Embedding(K-1, D) ZERO-INIT on
        # indices 1..K-1; (c) a NEW banded self-attention pass over the
        # whole bank (phase_i attends {phase_i-1, phase_i, phase_i+1,
        # class 0}; class token 0 attends all) with ZERO-INIT out_proj,
        # THEN the existing H4b merger runs unchanged on the refined bank.
        # Zero-init embeddings + zero-init out_proj => exact init-equality;
        # byte-inert when the env is unset. Mutually exclusive with
        # TIFAD_ATTNDESIGN (asserted) to keep interpretation clean.
        self.levelattn = _os.environ.get('TIFAD_LEVELATTN', '') == '1'
        if self.levelattn:
            assert self.attr_attn is not None, \
                "[LEVELATTN] needs TIFAD_PROMPT_ENSEMBLE + TIFAD_ATTR_ATTN=1"
            assert not self.attn_design, \
                "[LEVELATTN] mutually exclusive with TIFAD_ATTNDESIGN"
            _la_k = len(next(iter(self.ensemble_prompts.values()))) - 1
            assert _la_k >= 1, \
                "[LEVELATTN] ensemble needs >= 2 sentences/class"
            self.level_emb = nn.Embedding(2, emb_dim)
            self.order_emb = nn.Embedding(_la_k, emb_dim)
            nn.init.zeros_(self.level_emb.weight)
            nn.init.zeros_(self.order_emb.weight)
            self.level_attn = nn.MultiheadAttention(emb_dim, 4,
                                                    batch_first=True)
            nn.init.zeros_(self.level_attn.out_proj.weight)
            nn.init.zeros_(self.level_attn.out_proj.bias)
            print("[LEVELATTN] active (level+order embeddings, banded "
                  "self-attn, zero-init)", flush=True)
        if self.nodeattr:
            assert self.nodeattr in ('action', 'action_clean', 'all'), \
                self.nodeattr
            if self.nodeattr == 'all' and self.ensemble_prompts is None:
                print("[NODEATTR] 'all' needs an ensemble; falling back to "
                      "'action'", flush=True)
                self.nodeattr = 'action'
            self._nodeattr_vals = _json.load(open(_os.environ.get(
                'TIFAD_NODEATTR_ATTRS', 'configs/thumos14_ttb_attrs.json')))
            self.nodeattr_proj = nn.Linear(emb_dim, emb_dim)
            nn.init.zeros_(self.nodeattr_proj.weight)
            nn.init.zeros_(self.nodeattr_proj.bias)
            self._nodeattr_cache = {}
            print(f"[NODEATTR] '{self.nodeattr}' active: zero-init Linear "
                  f"residual of cached attr features "
                  f"({len(self._nodeattr_vals)} classes)", flush=True)
            # ---- NODEATTR_GATE (TIFAD_NODEATTR_GATE=1; 'action'/'all'
            # modes): multiply the injected residual by tanh(alpha_c), alpha
            # a per-class learnable vector init ZERO -- tanh(0)=0 kills the
            # whole residual at init, so init-equality is trivial (doubly so
            # with the zero-init Linear). cls_name varies per split, so
            # alphas are keyed by class NAME over the FIXED sorted list of
            # the attrs-JSON keys; forward maps cls_name -> index.
            # Constructed HERE so make_optimizer sees it (FAMTIE lesson);
            # plain nn.Parameter name -> decay bucket. Byte-inert when the
            # env is unset (alpha stays None, no new params).
            self.nodeattr_gate_alpha = None
            if _os.environ.get('TIFAD_NODEATTR_GATE', '') == '1':
                self._nodeattr_gate_idx = {
                    c: i for i, c in enumerate(sorted(self._nodeattr_vals))}
                self.nodeattr_gate_alpha = nn.Parameter(
                    torch.zeros(len(self._nodeattr_gate_idx)))
                print("[NODEATTR] per-class gate active (zero-init alpha)",
                      flush=True)
        if self.lma_on:
            self._lma_vals = {k: v for k, v in _json.load(open(_os.environ.get(
                'TIFAD_LMA_FILE', 'configs/thumos14_lma.json'))).items()
                if not k.startswith('_')}
            self.lma_proj = nn.Linear(emb_dim, emb_dim)
            nn.init.zeros_(self.lma_proj.weight)
            nn.init.zeros_(self.lma_proj.bias)
            self._lma_cache = {}
            print(f"[LMA] active: zero-init Linear residual of cached LMA "
                  f"features ({len(self._lma_vals)} classes)", flush=True)

        # ---- ATTRCODE (TIFAD_ATTRCODE=1): BINARY attribute coding. Each
        # class gets a FIXED 0/1 code over the sorted vocabulary of ALL
        # unique attribute values in the attrs json (V values); a ZERO-INIT
        # nn.Linear(V, D) ("code_proj") maps the code to a residual injected
        # PRE-ATTENTION uniformly into the ensemble bank (exact NODEATTR-
        # 'all' mechanism; falls back to final-emb injection when no
        # ensemble -- noted in the marker). Independent of TIFAD_NODEATTR:
        # both residuals compose. The 0/1 codes are plain CPU tensors in a
        # dict (rebuilt into a batch tensor per call) -- NOT parameters.
        # Constructed HERE so make_optimizer sees code_proj (FAMTIE lesson).
        # Byte-inert when TIFAD_ATTRCODE is unset; zero-init proj => exact
        # init-equality when set.
        self.code_proj = None
        if _os.environ.get('TIFAD_ATTRCODE', '') == '1':
            _ac_vals = _json.load(open(_os.environ.get(
                'TIFAD_ATTRCODE_ATTRS', 'configs/thumos14_ttb_attrs.json')))
            _ac_vocab = sorted({v for vs in _ac_vals.values() for v in vs})
            _ac_vid = {v: i for i, v in enumerate(_ac_vocab)}
            self._attrcode = {}
            for _ac_c, _ac_vs in _ac_vals.items():
                _code = torch.zeros(len(_ac_vocab))
                _code[[_ac_vid[v] for v in _ac_vs]] = 1.0
                self._attrcode[_ac_c] = _code
            # ---- ATTRCODE_WHITEN (TIFAD_ATTRCODE_WHITEN=1, modifier for
            # TIFAD_ATTRCODE=1): replace the raw 0/1 codes by their ESZSL-
            # style centered + decorrelated version. S (C x V over ALL
            # attrs-JSON classes) -> column-center -> D = (S^T S / C +
            # lambda I)^(-1/2) with lambda=0.1 (symmetric eigh inverse
            # sqrt, torch only, no scipy) -> code_c = (s_c - mean) @ D,
            # computed ONCE here; mean/D registered as buffers. The SAME
            # zero-init code_proj consumes the whitened codes, so exact
            # init-equality is untouched. Byte-inert when unset (raw 0/1
            # codes, no buffers).
            _ac_wnote = ''
            if _os.environ.get('TIFAD_ATTRCODE_WHITEN', '') == '1':
                _aw_cls = sorted(_ac_vals.keys())
                _aw_S = torch.stack([self._attrcode[c] for c in _aw_cls])
                _aw_mu = _aw_S.mean(0)                            # (V,)
                _aw_Sc = _aw_S - _aw_mu
                _aw_G = ((_aw_Sc.t() @ _aw_Sc) / _aw_S.size(0)
                         + 0.1 * torch.eye(_aw_S.size(1)))
                _aw_w, _aw_V = torch.linalg.eigh(_aw_G)
                _aw_D = (_aw_V @ torch.diag(
                    _aw_w.clamp(min=1e-12).rsqrt()) @ _aw_V.t())
                self.register_buffer('attrcode_whiten_mu', _aw_mu)
                self.register_buffer('attrcode_whiten_D', _aw_D)
                for _aw_c in _aw_cls:
                    self._attrcode[_aw_c] = (
                        self._attrcode[_aw_c] - _aw_mu) @ _aw_D
                _ac_wnote = '; whitened (ESZSL)'
            self.code_proj = nn.Linear(len(_ac_vocab), emb_dim)
            nn.init.zeros_(self.code_proj.weight)
            nn.init.zeros_(self.code_proj.bias)
            _ac_note = ('' if self.ensemble_prompts is not None
                        else '; no ensemble -> final-emb injection')
            print(f"[ATTRCODE] active (V={len(_ac_vocab)} binary attribute "
                  f"code, zero-init proj{_ac_note}{_ac_wnote})", flush=True)

        # ---- ATTRTOK (TIFAD_ATTRTOK=1): attributes as TOKENS in the H4b
        # attention. Each class's PRESENT attribute-value CLIP embeddings
        # (cached no_grad, ONE embedding per value the class HAS -- variable
        # count per class, padded via the attention key_padding_mask) are
        # appended as EXTRA TOKENS to the KEY/VALUE set of the existing
        # attr_attn call in forward(); the query (class sentence) is
        # unchanged and `per` itself is NOT modified. Init-equality rides on
        # H4b's zero-init out_proj (the attention residual is exactly 0
        # regardless of the key set); NO new parameters. Byte-inert when
        # unset. Requires TIFAD_PROMPT_ENSEMBLE + TIFAD_ATTR_ATTN=1;
        # incompatible with TIFAD_ATTNDESIGN=allphase (that branch swaps in
        # its own shared key pool). Env: TIFAD_ATTRTOK=1,
        # TIFAD_ATTRTOK_ATTRS (default configs/thumos14_ttb_attrs.json).
        self.attrtok = _os.environ.get('TIFAD_ATTRTOK', '') == '1'
        if self.attrtok:
            assert self.attr_attn is not None, \
                "[ATTRTOK] needs TIFAD_PROMPT_ENSEMBLE + TIFAD_ATTR_ATTN=1"
            assert self.attn_design != 'allphase', \
                "[ATTRTOK] incompatible with TIFAD_ATTNDESIGN=allphase"
            self._attrtok_vals = _json.load(open(_os.environ.get(
                'TIFAD_ATTRTOK_ATTRS', 'configs/thumos14_ttb_attrs.json')))
            self._attrtok_cache = {}          # value name -> CPU CLIP emb
            _at_max = max(len(v) for v in self._attrtok_vals.values())
            print(f"[ATTRTOK] active (present-attr tokens in H4b keys, "
                  f"max {_at_max} per class)", flush=True)

        # ---- ATTRVQ (TIFAD_ATTRVQ=<int Q>): CODEBOOK attribute encoding.
        # A learned codebook nn.Embedding(Q, D) (std-0.02 init from a
        # DEDICATED RNG generator, so the global stream sees only the
        # module-construction draws that happen after all baseline init --
        # VPW precedent); each class's fixed 0/1 code over the V attribute
        # values is mapped by a ZERO-INIT Linear(V, Q) ("vq_route") to
        # softmax codebook weights (uniform at init), and the class attr
        # representation weights @ codebook is injected PRE-ATTENTION
        # uniformly into the ensemble bank via a SECOND ZERO-INIT
        # Linear(D, D) ("vq_inject") -- exact ATTRCODE mechanism, incl. the
        # final-emb fallback when no ensemble. DOUBLE zero-init: routing
        # zero => uniform softmax => the codebook MEAN (a fixed vector),
        # AND the injection Linear zero => the residual is exactly 0 at
        # init. Byte-inert when unset. Envs: TIFAD_ATTRVQ=<Q>,
        # TIFAD_ATTRVQ_ATTRS (default configs/thumos14_ttb_attrs.json).
        self.vq_codebook = None
        _vq_env = _os.environ.get('TIFAD_ATTRVQ', '')
        if _vq_env:
            _vq_q = int(_vq_env)
            assert _vq_q >= 1, "[ATTRVQ] TIFAD_ATTRVQ must be a positive int"
            _vq_vals = _json.load(open(_os.environ.get(
                'TIFAD_ATTRVQ_ATTRS', 'configs/thumos14_ttb_attrs.json')))
            _vq_vocab = sorted({v for vs in _vq_vals.values() for v in vs})
            _vq_vid = {v: i for i, v in enumerate(_vq_vocab)}
            self._vqcode = {}
            for _vq_c, _vq_vs in _vq_vals.items():
                _vqc = torch.zeros(len(_vq_vocab))
                _vqc[[_vq_vid[v] for v in _vq_vs]] = 1.0
                self._vqcode[_vq_c] = _vqc
            self.vq_codebook = nn.Embedding(_vq_q, emb_dim)
            with torch.no_grad():
                _vq_g = torch.Generator().manual_seed(20260906)
                self.vq_codebook.weight.normal_(0.0, 0.02, generator=_vq_g)
            self.vq_route = nn.Linear(len(_vq_vocab), _vq_q)
            nn.init.zeros_(self.vq_route.weight)
            nn.init.zeros_(self.vq_route.bias)
            self.vq_inject = nn.Linear(emb_dim, emb_dim)
            nn.init.zeros_(self.vq_inject.weight)
            nn.init.zeros_(self.vq_inject.bias)
            print(f"[ATTRVQ] active (Q={_vq_q} codebook, zero-init "
                  f"routing+injection)", flush=True)

        # ---- PDA (TIFAD_PDA=1, arXiv 2603.24030 "Decompose and Transfer"):
        # CSD text side. Four PHASE-SPECIFIC trained projections Phi_txt^p
        # (Linear 512->D, no bias, init like the fork's text_projection) over
        # the frozen CLIP pooled embeddings of the LLM phase descriptions
        # (TIFAD_PDA_DESC json, default configs/thumos14_pda_desc.json). In
        # forward() the GLOBAL phase embedding t^glob replaces the label-level
        # text_embed handed to the backbone (its TVCA stays Ti-FAD's), and
        # all four banks are stashed at self._pda_phase (WITH graph) for the
        # TIF/APA consumer in meta_archs. The stock text_projection is then
        # unused. Registered HERE so make_optimizer sees the params. Byte-
        # inert when unset (pda_modules is not even imported).
        self.pda_text = None
        if _os.environ.get('TIFAD_PDA', '') == '1':
            assert self.ensemble_prompts is None, \
                "[PDA] incompatible with TIFAD_PROMPT_ENSEMBLE"
            from .pda_modules import PDATextEncoder
            _pda_desc = _os.environ.get('TIFAD_PDA_DESC',
                                        'configs/thumos14_pda_desc.json')
            self.pda_text = PDATextEncoder(
                _pda_desc, self.txt_model.text_embed_dim, emb_dim)
            print(f"[PDA] CSD text encoder active: {len(self.pda_text.desc)} "
                  f"classes x 4 phases from {_pda_desc}; four Phi_txt "
                  f"projections {self.txt_model.text_embed_dim}->{emb_dim}",
                  flush=True)


    def extract_text_emb(self, cls_name, is_prompt=False):
        if is_prompt:
            train_prompt = self.get_prompt(cls_name)
        else:
            train_prompt = cls_name 
        device = next(iter(self.txt_model.parameters())).device
        texts = self.tokenizer(train_prompt, padding=True, return_tensors="pt").to(device)

        if hasattr(self.txt_model, 'get_text_features'):
            text_emb = self.txt_model.get_text_features(**texts)
        else:
            text_emb = self.txt_model.encode_text(train_prompt)
            
        return text_emb
        
    def txt_read(self, file_path, sort=False):
        with open(file_path, 'r') as f:
            cls_name = [cls_name.strip('\n') for cls_name in f.readlines()]

        if sort:
            cls_name = sorted(cls_name)
        
        split_dict = {cls_name: i for i, cls_name in enumerate(cls_name)}

        return split_dict

    def get_prompt(self, cls_name):
        prompt_cls_name = [f'a video of action {c}' for c in cls_name]
        
        return prompt_cls_name
    
    def _load_classes(self, subset_file, data_split, mode, split_num):
        file_path = data_split_dir(subset_file, data_split, mode, split_num)
        try:
            with open(file_path, 'r') as f:
                cls_name = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            raise ValueError(f"No class file: {file_path}")

        return cls_name

    # ---- IDEAS-1 helpers (cached no_grad CLIP feature banks) ------------
    def _mean_text_feat(self, texts):
        with torch.no_grad():
            return self.extract_text_emb(
                texts, is_prompt=False).mean(0).detach().cpu()

    def _bank_feats(self, cls_name, cache, texts_of, device):
        for c in cls_name:
            if c not in cache:
                cache[c] = self._mean_text_feat(texts_of(c))
        return torch.stack([cache[c] for c in cls_name]).to(device)

    def _nodeattr_feats(self, cls_name, device):
        # attribute FEATURE vector: mean CLIP emb of the class's attribute-
        # value names (underscores -> spaces), e.g. 'motion_throw_release'
        feats = self._bank_feats(
            cls_name, self._nodeattr_cache,
            lambda c: [v.replace('_', ' ') for v in self._nodeattr_vals[c]],
            device)
        # TIFAD_NODEATTR_CENTER=1: subtract the bank mean computed over the
        # FULL attrs-JSON vocabulary (split-independent, so train and eval
        # see the same centering; diagnosis 2026-09-05: bank mutual cos 0.92,
        # the common mode compresses intra-family geometry -- centering
        # keeps only the class-specific component).
        import os as _osnc
        if _osnc.environ.get('TIFAD_NODEATTR_CENTER', '') == '1':
            if not hasattr(self, '_nodeattr_center_mu'):
                allc = sorted(self._nodeattr_vals.keys())
                mu = self._bank_feats(
                    allc, self._nodeattr_cache,
                    lambda c: [v.replace('_', ' ')
                               for v in self._nodeattr_vals[c]],
                    device).mean(0, keepdim=True).detach()
                self._nodeattr_center_mu = mu.cpu()
                print("[NODEATTR] bank centering active (full-vocab mean)",
                      flush=True)
            feats = feats - self._nodeattr_center_mu.to(device)
        return feats

    def _nodeattr_gate(self, cls_name, device):
        # (A,) per-class tanh gate for the NODEATTR residual
        # (TIFAD_NODEATTR_GATE=1): index by NAME into the fixed
        # sorted-attrs-JSON alpha vector; zero-init alpha => gate == 0 at
        # init, residual vanishes exactly.
        idx = torch.as_tensor(
            [self._nodeattr_gate_idx[c] for c in cls_name], device=device)
        return torch.tanh(self.nodeattr_gate_alpha[idx])

    def _lma_feats(self, cls_name, device):
        def _texts(c):
            e = self._lma_vals[c]
            t = [f"movement with {e['effort'][f]} {f} effort"
                 for f in ('space', 'weight', 'time', 'flow')]
            t += [f"movement with a {s} shape quality" for s in e['shape']]
            t += [f"movement mainly using the {b}" for b in e['body']]
            return t
        return self._bank_feats(cls_name, self._lma_cache, _texts, device)

    def _attrtok_bank(self, cls_name, like):
        # ATTRTOK: (A, M, D) padded bank of each class's PRESENT attribute-
        # value CLIP embeddings (cached no_grad; underscores -> spaces) plus
        # the (A, M) key_padding_mask (True = PAD). Classes absent from the
        # attrs json contribute zero real tokens (fully padded row -- safe:
        # the H4b key set always also contains the unpadded ensemble bank).
        need = sorted({v for c in cls_name
                       for v in self._attrtok_vals.get(c, [])
                       if v not in self._attrtok_cache})
        if need:
            with torch.no_grad():
                _emb = self.extract_text_emb(
                    [v.replace('_', ' ') for v in need],
                    is_prompt=False).detach().cpu()
            for _v, _e in zip(need, _emb):
                self._attrtok_cache[_v] = _e
        rows = [[self._attrtok_cache[v]
                 for v in self._attrtok_vals.get(c, [])] for c in cls_name]
        A = len(cls_name)
        M = max(1, max(len(r) for r in rows))
        bank = like.new_zeros(A, M, like.size(-1))
        pad = torch.ones(A, M, dtype=torch.bool, device=like.device)
        for i, r in enumerate(rows):
            if r:
                bank[i, :len(r)] = torch.stack(r).to(
                    device=like.device, dtype=like.dtype)
                pad[i, :len(r)] = False
        return bank, pad

    def _act_keys(self, device):
        # the 9 activity stem embeddings (taxonomy families), cached no_grad
        if self._act_cache is None:
            with torch.no_grad():
                self._act_cache = self.extract_text_emb(
                    [f'a video of a sports action: {a}'
                     for a in self._activities],
                    is_prompt=False).detach().cpu()
        return self._act_cache.to(device)

    @staticmethod
    def _band_mask(n, device):
        # bool attn_mask, True = MASKED; allows self + sequential neighbors
        # within window 1
        i = torch.arange(n, device=device)
        return (i[:, None] - i[None, :]).abs() > 1

    @staticmethod
    def _phase_cos_stats(ph):
        # ph (A, P, D): mean pairwise cosine among the P phases of one class
        # (averaged over classes) and among phases of DIFFERENT classes.
        A, P, _ = ph.shape
        z = torch.nn.functional.normalize(ph.detach().float(), dim=-1)
        z = z.reshape(A * P, -1)
        cos = z @ z.t()                                          # (AP, AP)
        cid = torch.arange(A, device=z.device).repeat_interleave(P)
        same = cid[:, None] == cid[None, :]
        off = ~torch.eye(A * P, dtype=torch.bool, device=z.device)
        within = cos[same & off].mean().item() if P > 1 else float('nan')
        across = cos[~same].mean().item() if A > 1 else float('nan')
        return within, across

    def _deaniso(self, ph, mode):
        """TIFAD_TXT_DEANISO: ph (A, P, D) phase bank -> de-anisotropised
        bank of identical shape and per-row L2 norms. Statistics are taken
        over the A*P stacked phases of the current vocabulary, detached
        (no parameters; gradients flow only through the identity path)."""
        A, P, D = ph.shape
        x = ph.reshape(A * P, D)
        orig_norm = x.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            xs = x.detach().float()
            mu = xs.mean(0, keepdim=True)                        # (1, D)
            xc = xs - mu
            sd = None
            basis = None
            if self.txt_deaniso == 'whiten':
                sd = xc.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
            elif self.txt_deaniso_k > 0:
                # top-k principal directions of the centred stack (rows of
                # Vh); SVD is deterministic (pca_lowrank is randomised)
                _, _, vh = torch.linalg.svd(xc, full_matrices=False)
                basis = vh[:min(self.txt_deaniso_k, vh.size(0))]  # (k, D)
        y = x - mu.to(x.dtype)
        if sd is not None:
            y = y / sd.to(x.dtype)
        if basis is not None:
            b = basis.to(x.dtype)
            y = y - (y @ b.t()) @ b
        y = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-8) * orig_norm
        y = y.view(A, P, D)
        if mode not in self._deaniso_printed:
            self._deaniso_printed.add(mode)
            w0, a0 = self._phase_cos_stats(ph)
            w1, a1 = self._phase_cos_stats(y)
            print(f"[DEANISO] {self.txt_deaniso} ({mode}, {A} classes x {P} "
                  f"phases): mean pairwise phase cosine within-class "
                  f"{w0:.3f} -> {w1:.3f}, across-class {a0:.3f} -> {a1:.3f}",
                  flush=True)
        return y

    def _sub_refine(self, per, cls_name, degrade, mode):
        """TIFAD_SUB_ENSEMBLE: lower-level ASA. per (A, K+1, D) projected
        ensemble bank (index 0 = class sentence, 1..K = phases); degrade =
        the h18 set of classes whose sentences were replaced by the parent
        stem this iteration (their sub-sentences are replaced too).
        Returns per with per[:, 1:] replaced by the sub-phase-refined phase
        embeddings (bitwise equal to the input while both out_proj's are
        zero)."""
        A, K1, D = per.shape
        K, Kp = K1 - 1, self._sub_kp
        flat = []
        for c in cls_name:
            if c in degrade:
                flat.extend([self._stems.get(c, c)] * (K * Kp))
            else:
                for lst in self.sub_prompts[c]:
                    flat.extend(lst)
        sub = self.extract_text_emb(flat, is_prompt=False)        # (A*K*Kp, D)
        sub = sub.view(A * K, Kp, D)
        # banded neighbour self-attention among the Kp sub-phases of ONE
        # phase (every (class, phase) is an independent sequence)
        if self.chain_attn_sub is not None and Kp > 1:
            _sm = self._band_mask(Kp, sub.device)
            _su, _ = self.chain_attn_sub(sub, sub, sub, attn_mask=_sm,
                                         need_weights=False)
            sub = sub + _su
        # phase query attends over [phase; refined sub-phases]
        q = per[:, 1:].reshape(A * K, 1, D)
        keys = torch.cat([q, sub], dim=1)                          # (A*K, 1+Kp, D)
        kpm = None
        if self.prompt_drop > 0 and mode == 'train':
            kpm = torch.rand(A * K, 1 + Kp, device=per.device) < self.prompt_drop
            kpm[:, 0] = False                    # never drop the phase query
        upd, _ = self.attr_attn_sub(q, keys, keys, key_padding_mask=kpm,
                                    need_weights=False)
        ph = (q + upd).reshape(A, K, D)
        return torch.cat([per[:, :1], ph], dim=1)

    def _shuffle_bank(self, per, cls_name, mode):
        """TIFAD_SHUFFLE_CHILDREN: per (A, K+1, D) -> the same bank with the
        K children per[:, 1:] permuted along dim 1 per class (root index 0
        untouched). Fixed per-class permutation (mode != 'train', 'eval' /
        'both') or a fresh torch.randperm per class (mode == 'train',
        'train' / 'both'); identity otherwise (see __init__)."""
        is_train = mode == 'train'
        if is_train and self.shuffle_children == 'eval':
            return per
        if not is_train and self.shuffle_children == 'train':
            return per
        A, K1, D = per.shape
        K = K1 - 1
        if K < 2:
            return per
        if is_train:
            idx = torch.stack([torch.randperm(K) for _ in cls_name])
        else:
            idx = torch.stack([self._shuffle_perm[c] for c in cls_name])
        idx = idx.to(per.device).unsqueeze(-1).expand(-1, -1, D)   # (A, K, D)
        return torch.cat([per[:, :1], per[:, 1:].gather(1, idx)], dim=1)

    def forward(self, batch_size, mode):

        cls_name = self.train_classes if mode == 'train' else self.test_classes

        # ---- FINEGYM LEVEL COMPOSE (per-level hierarchy eval, EVAL ONLY):
        # expand each eval ACTIVITY class into [activity] + its member
        # ELEMENT classes (VARIABLE-size groups, unlike TIFAD_SUBCLASS's
        # uniform S), BEFORE the embedding so every pseudo-class flows
        # through the full h19 path (ensemble + attention) and the backbone
        # text update. meta_archs max-pools the logit groups back to C after
        # the matmul (see "LEVEL COMPOSE max-pool" there). Group order per
        # class c: [c, members(c)...]; classes missing from the map form
        # singleton groups. Envs: FINEGYM_LEVEL_COMPOSE=1,
        # FINEGYM_LEVEL_MAP=<json {activity: [member element names]}>
        # (default configs/finegym_set_members.json). NOTE: with the h19
        # ensemble active, every member element must be an ensemble key --
        # use configs/finegym_setmix_ensemble.json. Inert (byte-identical)
        # when unset or in train mode.
        import os as _oslc
        self._lvl_group_sizes = None
        if (_oslc.environ.get('FINEGYM_LEVEL_COMPOSE', '') == '1'
                and mode != 'train'):
            assert not _oslc.environ.get('TIFAD_SUBCLASS', ''), \
                "[LVL] FINEGYM_LEVEL_COMPOSE is incompatible with " \
                "TIFAD_SUBCLASS"
            if not hasattr(self, '_lvl_members'):
                import json as _json
                self._lvl_members = _json.load(open(_oslc.environ.get(
                    'FINEGYM_LEVEL_MAP',
                    'configs/finegym_set_members.json')))
                print(f"[LVL] compose-mode expansion active: "
                      f"{len(self._lvl_members)} activities in map",
                      flush=True)
            _exp, _sizes = [], []
            for _c in cls_name:
                _grp = [_c] + list(self._lvl_members.get(_c, []))
                _exp.extend(_grp)
                _sizes.append(len(_grp))
            cls_name = _exp
            self._lvl_group_sizes = _sizes

        _htca_per = None      # per-sentence embs (A, K, D) for the HTCA bank
        if self.pda_text is not None:
            # ---- PDA (TIFAD_PDA=1): phase banks {p: (C, D)}; the GLOBAL
            # description embedding plays the label-level text's role in
            # the backbone; all four are stashed for TIF/APA (meta_archs).
            _pda = self.pda_text(self, cls_name)
            self._pda_phase = _pda
            text_emb = _pda['glob']
        elif self.ensemble_prompts is None:
            # stock Ti-FAD path (one sentence per class)
            text_emb = self.extract_text_emb(cls_name, None)
        else:
            # ---- h18: semantic degradation to the parent stem (TRAIN ONLY) ----
            degrade = set()
            if self.sem_degrade > 0 and mode == 'train':
                import random as _r, json as _json
                if self._stems is None:
                    import os as _os
                    _ar = _json.load(open(_os.environ.get(
                        'TIFAD_ARITH', 'configs/thumos14_arith.json')))
                    self._stems = {c: e['parent'] for c, e in _ar.items()}
                degrade = {c for c in cls_name if _r.random() < self.sem_degrade}

            # flatten the ensemble; one CLIP forward for all sentences
            flat = []
            for c in cls_name:
                sents = self.ensemble_prompts[c]
                if c in degrade:
                    sents = [self._stems.get(c, c)] * len(sents)
                flat.extend(sents)
            all_emb = self.extract_text_emb(flat, is_prompt=False)      # (A*K, D)

            A = len(cls_name)
            K = len(self.ensemble_prompts[cls_name[0]])                 # uniform K
            per = all_emb.view(A, K, -1)                                # (A, K, D)
            # ---- SUB-ENSEMBLE (TIFAD_SUB_ENSEMBLE): lower-level ASA over
            # the Kp sub-phase sentences of every phase; REPLACES per[:, 1:]
            # here so every stash / injection / attention below sees the
            # refined phases (exactly per at init: zero-init out_proj's).
            if self.sub_prompts is not None and K > 1:
                per = self._sub_refine(per, cls_name, degrade, mode)
            # ---- ORDER CONTROL (TIFAD_SHUFFLE_CHILDREN): permute the K-1
            # children per class BEFORE every stash / injection / attention
            # below reads the bank (after _sub_refine so sub-children move
            # with their phase). Identity when unset (see __init__).
            if self.shuffle_children:
                per = self._shuffle_bank(per, cls_name, mode)
            _htca_per = per

            # ---- V1 VIDPHASE stash (TIFAD_VIDPHASE=<w>, TRAIN ONLY): keep
            # the per-sentence embeddings (WITH graph -- gradients reach
            # text_projection) so meta_archs can build the 3-phase text bank
            # for the video-third InfoNCE. Cleared by the consumer each
            # step. Inert when unset.
            import os as _osvp
            if (mode == 'train'
                    and float(_osvp.environ.get('TIFAD_VIDPHASE', '0')
                              or 0) != 0):
                self._vidphase_per = per

            # ---- VPW stash (TIFAD_VPW=1, TRAIN AND EVAL): keep the
            # per-sentence embeddings (WITH graph -- gradients reach
            # text_projection) so meta_archs' video-conditioned phase
            # weighting can score every snippet against each phase
            # embedding per[:, 1:]. Cleared by the consumer each forward
            # (VIDPHASE precedent). Inert when TIFAD_VPW is unset.
            if _osvp.environ.get('TIFAD_VPW', '') == '1':
                self._vpw_per = per

            # ---- PHASELOGIT stash (TIFAD_PHASELOGIT=1, TRAIN AND EVAL):
            # keep the per-sentence embeddings (WITH graph -- gradients
            # reach text_projection) so meta_archs' score-space phase
            # composition (CHiLS, arXiv:2302.02551, adapted to score space)
            # can score every snippet against each phase embedding
            # per[:, 1:]. Independent of _vpw_per (own stash, own
            # consumer; both may be set). Cleared by the consumer each
            # forward (VIDPHASE precedent). Inert when TIFAD_PHASELOGIT is
            # unset.
            if _osvp.environ.get('TIFAD_PHASELOGIT', '') == '1':
                self._phaselogit_per = per

            # ---- PHASEPROF stash (TIFAD_DUMP_PHASEPROF=<dir>, EVAL
            # analysis): keep the per-sentence embeddings so meta_archs can
            # dump per-timestep phase-vs-video score profiles for offline
            # boundary analysis. Detached (no training use). Inert unset.
            # (also stashed for TIFAD_SUPPORT -- the eval-time pre-NMS
            # completeness rescoring consumes the same 11-sentence bank)
            if (_osvp.environ.get('TIFAD_DUMP_PHASEPROF', '')
                    or float(_osvp.environ.get('TIFAD_SUPPORT', '0')
                             or 0) > 0):
                self._dumpphase_per = per.detach()

            # ---- PART 1 'all': zero-init attr-feature residual injected
            # UNIFORMLY at every level pre-H4b: per[:, 0] (class sentence) and
            # per[:, 1:] (10 phase sentences). The activity representation of
            # the h18 degrade path lives inside `per` as the parent-stem
            # sentence embeddings, so this uniform injection covers it
            # (documented fallback -- see __init__ comment).
            if self.nodeattr == 'all':
                _naf = self._nodeattr_feats(cls_name, per.device)       # (A, D)
                _nar = self.nodeattr_proj(_naf).unsqueeze(1)            # (A, 1, D)
                # NODEATTR_GATE: scale by per-class tanh(alpha) (0 at init)
                if self.nodeattr_gate_alpha is not None:
                    _nar = _nar * self._nodeattr_gate(
                        cls_name, per.device).view(-1, 1, 1)
                per = per + _nar

            # ---- ATTRCODE (TIFAD_ATTRCODE=1): zero-init projection of the
            # class's fixed 0/1 attribute-value code injected PRE-ATTENTION
            # uniformly into the ensemble bank (per[:, 0] AND per[:, 1:]),
            # exactly the NODEATTR-'all' mechanism above. Independent of
            # TIFAD_NODEATTR -- both residuals compose additively.
            if self.code_proj is not None:
                _acc = torch.stack(
                    [self._attrcode[c] for c in cls_name]).to(per.device)
                per = per + self.code_proj(_acc).unsqueeze(1)    # (A, 1, D)

            # ---- ATTRVQ (TIFAD_ATTRVQ=<Q>): codebook attribute encoding.
            # Soft codebook weights from the class's fixed 0/1 code
            # (zero-init routing => UNIFORM softmax at init), class attr
            # representation = weights @ codebook, injected PRE-ATTENTION
            # uniformly into the bank via the zero-init vq_inject Linear
            # (exact ATTRCODE mechanism above; composes additively).
            if self.vq_codebook is not None:
                _vqb = torch.stack(
                    [self._vqcode[c] for c in cls_name]).to(per.device)
                _vqw = torch.softmax(self.vq_route(_vqb), dim=-1)  # (A, Q)
                per = per + self.vq_inject(
                    _vqw @ self.vq_codebook.weight).unsqueeze(1)  # (A, 1, D)

            if self.attr_attn is None:
                text_emb = per.mean(1)                                  # H4a: mean
            else:
                # ---- LEVELATTN (TIFAD_LEVELATTN=1): zero-init level +
                # phase-order embeddings added to the bank tokens, then ONE
                # banded self-attention pass over the whole bank (band
                # window 1 over phases, class token 0 fully connected both
                # ways) BEFORE the H4b merger below, which then consumes
                # the refined bank unchanged. All additive and exactly 0 at
                # init (init-equality invariant -- keep it).
                if self.levelattn and K > 1:
                    _la_lvl = torch.ones(K, dtype=torch.long,
                                         device=per.device)
                    _la_lvl[0] = 0
                    _la_ord = torch.cat([
                        self.order_emb.weight.new_zeros(1, per.size(-1)),
                        self.order_emb(torch.arange(K - 1,
                                                    device=per.device)),
                    ], dim=0)                                       # (K, D)
                    per = per + (self.level_emb(_la_lvl)
                                 + _la_ord).unsqueeze(0)
                    _la_i = torch.arange(K, device=per.device)
                    _la_m = (_la_i[:, None] - _la_i[None, :]).abs() > 1
                    _la_m[0, :] = False     # class query attends to all
                    _la_m[:, 0] = False     # every phase sees class 0
                    _la_u, _ = self.level_attn(per, per, per,
                                               attn_mask=_la_m,
                                               need_weights=False)
                    per = per + _la_u
                # ---- H4b (+ h14 dropout): sentence[0] attends over its own set ----
                q = per[:, :1]                                          # (A, 1, D)
                # ---- PART 2 'act2act'/'cascade': class query attends over the
                # activity stem embeddings (zero-init residual) BEFORE H4b.
                if self.attn_design in ('act2act', 'cascade'):
                    _ak = self._act_keys(per.device).unsqueeze(0).expand(
                        A, -1, -1)                                      # (A, F, D)
                    _au, _ = self.act_attn(q, _ak, _ak, need_weights=False)
                    q = q + _au
                keys = per
                # ---- PART 2 'phasechain'/'cascade': phase sentences attend to
                # their SEQUENTIAL neighbors (band mask, window 1), zero-init
                # residual, then H4b consumes the refined phases as usual.
                if self.attn_design in ('phasechain', 'cascade') and K > 1:
                    ph = per[:, 1:]                                     # (A, K-1, D)
                    # ORDER CONTROL (TIFAD_CHAIN_FULL=1): no mask -> every
                    # child attends to every child (see __init__)
                    _bm = (None if self.chain_full
                           else self._band_mask(K - 1, ph.device))
                    # ---- TXT_DEANISO: the self-attention reads the
                    # de-anisotropised phases (see __init__ / _deaniso);
                    # scope 'attn' keeps the residual on the ORIGINAL
                    # phases (init-equality), scope 'all' replaces them.
                    _phin = ph
                    if self.txt_deaniso:
                        _phin = self._deaniso(ph, mode)
                        if self.txt_deaniso_scope == 'all':
                            ph = _phin
                    _pu, _ = self.chain_attn(_phin, _phin, _phin,
                                             attn_mask=_bm,
                                             need_weights=False)
                    ph = ph + _pu
                    keys = torch.cat([q, ph], dim=1)
                elif self.attn_design == 'act2act':
                    keys = torch.cat([q, per[:, 1:]], dim=1)
                kpm = None
                if self.prompt_drop > 0 and mode == 'train':
                    kpm = torch.rand(A, K, device=per.device) < self.prompt_drop
                    kpm[:, 0] = False                    # never drop the class prompt
                # ---- ATTRTOK (TIFAD_ATTRTOK=1): append the class's PRESENT
                # attribute-value CLIP embeddings as EXTRA KEY/VALUE tokens
                # of the H4b call below (query unchanged, `per` untouched);
                # variable counts are padded and masked True in the
                # key_padding_mask. Zero-init out_proj => the H4b residual
                # is exactly 0 at init with or without the extra keys
                # (init-equality invariant -- keep it). 'allphase' is
                # excluded by the __init__ assert.
                if self.attrtok:
                    _at_bank, _at_pad = self._attrtok_bank(cls_name, per)
                    if kpm is None:
                        kpm = torch.zeros(A, keys.size(1), dtype=torch.bool,
                                          device=per.device)
                    keys = torch.cat([keys, _at_bank], dim=1)
                    kpm = torch.cat([kpm, _at_pad], dim=1)
                if self.attn_design == 'allphase':
                    # ---- PART 2 'allphase': class query attends over ALL
                    # classes' phase embeddings (cross-class phase transfer);
                    # keys = concat of every class's K-1 phases, shared.
                    pool = per[:, 1:].reshape(1, A * (K - 1), -1).expand(
                        A, -1, -1)                                      # (A, A*(K-1), D)
                    ppm = None
                    if self.prompt_drop > 0 and mode == 'train':
                        ppm = torch.rand(A, A * (K - 1),
                                         device=per.device) < self.prompt_drop
                    upd, _ = self.attr_attn(q, pool, pool,
                                            key_padding_mask=ppm,
                                            need_weights=False)
                else:
                    upd, _ = self.attr_attn(q, keys, keys,
                                            key_padding_mask=kpm,
                                            need_weights=False)
                text_emb = (q + upd).squeeze(1)                         # (A, D)

            # ---- LEAFLOSS stash (TIFAD_LEAFLOSS=1, TRAIN ONLY): keep the
            # ten ORDERED phase-sentence embeddings of every class (WITH
            # graph -- gradients reach CLIP text_projection / the phase
            # refinement) so meta_archs' leaf-level phase loss can score the
            # positive detector locations against the phases of their GT
            # class. Taken AFTER the 'phasechain'/'cascade' neighbour
            # attention when that design is active (refined phases live in
            # `keys[:, 1:]`), else the plain projected bank per[:, 1:]. Own
            # stash (VPW / PHASELOGIT precedent), cleared by the consumer
            # each forward. Inert when TIFAD_LEAFLOSS is unset.
            if (mode == 'train'
                    and _osvp.environ.get('TIFAD_LEAFLOSS', '') == '1'):
                if (self.attr_attn is not None
                        and self.attn_design in ('phasechain', 'cascade')
                        and K > 1):
                    self._leaf_phase_embs = keys[:, 1:K]            # (A, K-1, D)
                else:
                    self._leaf_phase_embs = per[:, 1:]              # (A, K-1, D)

            # ---- VSA-XM stash (TIFAD_VSA_XM=1, TRAIN AND EVAL): the SAME
            # ordered (refined) phase embeddings as the LEAFLOSS stash, for
            # the current vocabulary (train classes in train mode, the
            # unseen test classes in eval -- rows in cls_name order, i.e.
            # the class-logit column order), WITH graph. Consumed (cleared)
            # by meta_archs' VSA hook each forward. Inert when unset.
            # QH v4 (TIFAD_QH_PHASEMAP=1) reads the SAME stash for its
            # ordered phase-similarity map (meta_archs QH phase bank).
            # (PH v5 TIFAD_PH_SRC=text reads it too: centered phase cosine)
            if (_osvp.environ.get('TIFAD_VSA_XM', '') == '1'
                    or _osvp.environ.get('TIFAD_QH_PHASEMAP', '') == '1'
                    or _osvp.environ.get('TIFAD_PH_SRC', '') == 'text'):
                if (self.attr_attn is not None
                        and self.attn_design in ('phasechain', 'cascade')
                        and K > 1):
                    self._vsa_phase_embs = keys[:, 1:K]             # (A, K-1, D)
                else:
                    self._vsa_phase_embs = per[:, 1:]               # (A, K-1, D)

        # ---- ATTRCODE fallback (TIFAD_ATTRCODE=1, NO ensemble): the
        # pre-attention bank does not exist, so add the zero-init code
        # residual to the final class embedding instead (NODEATTR-'action'
        # style; this fallback is noted in the __init__ marker).
        if self.code_proj is not None and self.ensemble_prompts is None:
            _acc = torch.stack(
                [self._attrcode[c] for c in cls_name]).to(text_emb.device)
            text_emb = text_emb + self.code_proj(_acc)

        # ---- ATTRVQ fallback (TIFAD_ATTRVQ=<Q>, NO ensemble): mirror the
        # ATTRCODE fallback above -- inject the (doubly) zero-init codebook
        # residual into the final class embedding instead of the
        # nonexistent pre-attention bank.
        if self.vq_codebook is not None and self.ensemble_prompts is None:
            _vqb = torch.stack(
                [self._vqcode[c] for c in cls_name]).to(text_emb.device)
            _vqw = torch.softmax(self.vq_route(_vqb), dim=-1)
            text_emb = text_emb + self.vq_inject(
                _vqw @ self.vq_codebook.weight)

        # ---- PART 1 'action'/'action_clean' + PART 3 LMA: ZERO-INIT Linear
        # residuals of cached no_grad CLIP feature banks injected into the
        # class embeddings (composable: LMA uses its own zero-init Linear).
        if self.nodeattr in ('action', 'action_clean'):
            _nar = self.nodeattr_proj(
                self._nodeattr_feats(cls_name, text_emb.device))        # (A, D)
            # NODEATTR_GATE: scale by per-class tanh(alpha) (0 at init)
            if self.nodeattr_gate_alpha is not None:
                _nar = _nar * self._nodeattr_gate(
                    cls_name, text_emb.device).unsqueeze(-1)
            text_emb = text_emb + _nar
        if self.lma_on:
            text_emb = text_emb + self.lma_proj(
                self._lma_feats(cls_name, text_emb.device))

        # ---- TaxGNN residuals (TIFAD_GNN=1): run the taxonomy GNN over the
        # graph restricted to the CURRENT class set and add its zero-init
        # residuals to the post-h19 class embeddings. At eval the test-class
        # action nodes carry no prototypes -- exactly the leaf-holdout
        # condition trained for. Holdout randomness only when genuinely
        # training (mode=='train' AND module.training; PROTO_COLLECT runs
        # eval-mode with mode=='train' and must stay deterministic).
        if self.taxgnn is not None:
            _resid = self.taxgnn(
                cls_name, self.extract_text_emb,
                training=(mode == 'train' and self.training))
            text_emb = text_emb + _resid

        # ---- FAMTIE (senone-style hierarchical parameter tying): a TRAINED
        # zero-init residual per taxonomy family, added to every member class
        # embedding (train AND eval). Unseen classes inherit the residual their
        # seen siblings trained. Env: TIFAD_FAMTIE=1, TIFAD_FAMTIE_MAP=<json>.
        if hasattr(self, 'famtie_emb'):
            _fidx = torch.as_tensor(
                [self._famtie_fid[self._famtie_map[c]] for c in cls_name],
                device=text_emb.device)
            text_emb = text_emb + self.famtie_emb(_fidx)

        # ---- HTCA/HAF per-class token bank: for class c the token rows are
        # [text_emb[c] (post-H4b/FAMTIE), per-sentence phase embs per[c, 1:K],
        #  E_attr rows of attrs(c), E_fam[fam(c)]], padded to the max N over
        # classes with a key_padding_mask (True = PAD). Stashed on self as
        # self._htca_bank (C, N, D) / self._htca_mask (C, N). NOTE: the HTCA
        # attention in meta_archs consumes the CLASS-AGNOSTIC anchor sub-bank
        # (see HTCA-NOTE there); this full per-class bank is exposed for
        # analysis / extensions.
        if self.htca_attr_emb is not None:
            _dev = text_emb.device
            _toks = []
            for _i, _c in enumerate(cls_name):
                _rows = [text_emb[_i].unsqueeze(0)]
                if _htca_per is not None and _htca_per.size(1) > 1:
                    _rows.append(_htca_per[_i, 1:])
                _ai = self._htca_attr_of.get(_c, [])
                if _ai:
                    _rows.append(self.htca_attr_emb(
                        torch.as_tensor(_ai, dtype=torch.long, device=_dev)))
                _fi = self._htca_fam_of.get(_c)
                if _fi is not None:
                    _rows.append(self.htca_fam_emb(
                        torch.as_tensor([_fi], dtype=torch.long, device=_dev)))
                _toks.append(torch.cat(_rows, dim=0))
            _N = max(t.size(0) for t in _toks)
            _bank = text_emb.new_zeros(len(_toks), _N, text_emb.size(-1))
            _pad = torch.ones(len(_toks), _N, dtype=torch.bool, device=_dev)
            for _i, _t in enumerate(_toks):
                _bank[_i, :_t.size(0)] = _t
                _pad[_i, :_t.size(0)] = False
            self._htca_bank, self._htca_mask = _bank, _pad

        # ---- SUBCLASS EXPANSION (CHiLS downward, EVAL ONLY): append S subclass
        # sentence embeddings per class as pseudo-classes, interleaved
        # [c0, c0s1..c0sS, c1, ...]; the meta-arch max-pools logits back to C
        # (TIFAD_SUBCLASS_S must match). Env: TIFAD_SUBCLASS=<json>.
        import os as _osx
        _sc = _osx.environ.get('TIFAD_SUBCLASS', '')
        if _sc and mode != 'train':
            if not hasattr(self, '_sub_prompts'):
                import json as _json
                self._sub_prompts = _json.load(open(_sc))
                _S = len(next(iter(self._sub_prompts.values())))
                print(f"[SUBCLS] expansion active: S={_S} subclasses/class",
                      flush=True)
            _S = len(next(iter(self._sub_prompts.values())))
            _flat = [s for c in cls_name for s in self._sub_prompts[c]]
            _sub = self.extract_text_emb(_flat, is_prompt=False)   # (C*S, D)
            _C, _D = text_emb.size(0), text_emb.size(-1)
            _sub = _sub.view(_C, _S, _D)
            text_emb = torch.cat([text_emb.unsqueeze(1), _sub], dim=1
                                 ).reshape(_C * (1 + _S), _D)

        # ---- FAMCOMP (CHiLS-style, EVAL ONLY): blend each class embedding with
        # its taxonomy-family mean. Env: TIFAD_FAMCOMP=<beta>, TIFAD_FAMCOMP_MAP=
        # <json {class: family_id}>. Inert when unset or in train mode.
        import os as _osf
        _fam_b = float(_osf.environ.get('TIFAD_FAMCOMP', '0') or 0)
        if _fam_b > 0 and mode != 'train':
            if not hasattr(self, '_famcomp_map'):
                import json as _json
                self._famcomp_map = _json.load(
                    open(_osf.environ['TIFAD_FAMCOMP_MAP']))
                print(f"[FAMCOMP] beta={_fam_b}, "
                      f"{len(set(self._famcomp_map.values()))} families",
                      flush=True)
            fam_of = [self._famcomp_map.get(c) for c in cls_name]
            import collections as _coll
            groups = _coll.defaultdict(list)
            for i, f in enumerate(fam_of):
                if f is not None:
                    groups[f].append(i)
            blended = text_emb.clone()
            for f, idxs in groups.items():
                if len(idxs) < 2:
                    continue
                fam_mean = text_emb[idxs].mean(0, keepdim=True)
                blended[idxs] = (1.0 - _fam_b) * text_emb[idxs] + _fam_b * fam_mean
            text_emb = blended

        if len(text_emb.size()) == 2:
            text_emb = text_emb.expand(batch_size, -1, -1)

        split_num_cls = text_emb.size(1)

        return text_emb, split_num_cls