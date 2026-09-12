# FineGym Table-3 three-system comparison: shared frozen-CLIP-TEXT pathway.
#
# Systems (all share this pathway on top of the ActionFormerWithCLIP visual
# backbone; the closed-set flat classification head is REPLACED by scaled
# cosine similarity between the cls-head trunk features and per-class text
# embeddings):
#   1. base : prompt = "a video of action {class name}" (single sentence).
#   2. tifad: base + per-FPN-level text<-video cross-attention (TiCA/TVCA
#             pattern from the tifad fork, blocks.py TransformerBlock +
#             tv_attn.py TVCA) + foreground head weighting (fg branch of
#             Ti-FAD's decode score sqrt(sig(cls)*sig(cn))*sig(fg); we have
#             no cn head, so score = sig(cls)*sig(fg) — documented deviation).
#   3. champ: base pathway + 4 components
#             (a) HP  11-sentence Qwen ensemble + phase-sentence dropout
#                 (h14, p=0.3) + parent-stem degradation (h18, p=0.3)
#             (b) PA  zero-init class->phase MultiheadAttention
#                 (tifad text.py H4b: q = sentence0, keys = all 11)
#             (c) AS  14-way phrase-level focal CE + 2-way Kendall UW
#                 (tifad meta_archs FAMCE-UW; lives in meta_archs.losses)
#             (d) DP  eval-time log-normal duration prior (lives in
#                 meta_archs.inference; stats from gen_finegym_duration_prior)
#
# Init-equality invariant (verified by scripts/verify_t3_systems.py):
# with PA's out_proj zero-initialized, the champion's class embedding is
# exactly proj(CLIP(sentence_0)); if sentence_0 equals the baseline prompt,
# champion logits are byte-equal to baseline logits at init. The REAL
# ensemble's sentence 0 ("a video of a gymnastics element: {name}, ...")
# intentionally differs from the baseline prompt — that prompt difference is
# part of the HP component and is the one unavoidable deviation from strict
# init-equality (documented in the verification script).
#
# Frozen-CLIP caching: everything before CLIP's text_projection output is
# frozen, so the raw 512-d sentence embeddings are computed ONCE (no_grad,
# at init, disk-cached) and only the trainable Linear(512 -> head_dim)
# projection (+ PA attention) runs per forward. This is mathematically
# identical to the tifad fork, which re-runs the frozen CLIP text tower each
# step with a trainable text_projection at the end.

import hashlib
import json
import math
import os

import torch
from torch import nn
from torch.nn import functional as F

CLIP_NAME = 'openai/clip-vit-base-patch32'   # matches the visual side
CLIP_TEXT_DIM = 512

# gym99 apparatus codes -> natural words, for stem sentences.
_APPARATUS = {'VT': 'vault', 'FX': 'floor exercise',
              'BB': 'balance beam', 'UB': 'uneven bars'}


def load_verbalizer_actions(verbalizer_path):
    """Return ordered per-class dicts: [{id, name, phrase, phrase_name, activity}]."""
    with open(verbalizer_path) as f:
        verb = json.load(f)
    actions = verb['actions']
    phrases = verb['phrases']
    # apparatus (top level): short code -> index in the verbalizer's order
    act_idx = {v['short']: int(k[1:]) for k, v in verb['activities'].items()}
    out = []
    for cid in range(len(actions)):
        a = actions[f'c{cid:02d}']
        out.append({
            'id': cid,
            'name': a['name'],
            'phrase': a['phrase'],
            'phrase_name': phrases[a['phrase']]['name'],
            'activity': a['activity'],
            'phrase_idx': int(a['phrase'][1:]),
            'activity_idx': act_idx[a['activity']],
        })
    return out


def humanize_phrase(phrase_name):
    """'BB_flight_salto' -> 'balance beam flight salto' (parent stem, h18)."""
    parts = phrase_name.split('_')
    if parts[0] in _APPARATUS:
        return (_APPARATUS[parts[0]] + ' ' + ' '.join(parts[1:])).strip()
    return ' '.join(parts)


def resolve_ensemble_keys(actions, ensemble):
    """Map class id -> ensemble key.

    Ensemble keys are "({activity}) {name}". gym99 has ONE duplicated display
    name — "(BB) salto backward tucked" (c58 phrase BB_flight_salto, c70
    phrase BB_dismounts) — which the ensemble disambiguates as "... v1" /
    "... v2". We resolve duplicates by matching phrase-derived keywords
    ('dismount' etc.) against the candidate keys' sentences; asserts a
    perfect 1-1 cover of all 99 classes.
    """
    keys = dict()          # cid -> ensemble key
    used = set()
    # group classes by base key
    by_base = {}
    for a in actions:
        base = f"({a['activity']}) {a['name']}"
        by_base.setdefault(base, []).append(a)
    for base, group in by_base.items():
        if len(group) == 1 and base in ensemble:
            keys[group[0]['id']] = base
            used.add(base)
            continue
        # duplicate names (or missing base): candidates share the base prefix
        candidates = [k for k in ensemble
                      if k == base or k.startswith(base + ' v')]
        assert len(candidates) == len(group), (
            f"[t3-text] ensemble key mismatch for '{base}': "
            f"{len(group)} classes vs candidates {candidates}")
        remaining = list(candidates)
        for a in sorted(group, key=lambda x: x['id']):
            # keyword score: phrase-name words appearing in candidate sentences
            kws = [w for w in humanize_phrase(a['phrase_name']).split()
                   if w not in ('balance', 'beam', 'floor', 'exercise',
                                'uneven', 'bars', 'vault')]
            def score(k):
                text = ' '.join(ensemble[k]).lower()
                return sum(text.count(w.lower()) for w in kws)
            remaining.sort(key=score, reverse=True)
            pick = remaining.pop(0)
            keys[a['id']] = pick
            used.add(pick)
    assert len(keys) == len(actions), "[t3-text] failed to cover all classes"
    assert len(set(keys.values())) == len(actions), "[t3-text] duplicate key use"
    return keys


def _embed_sentences(sentences, cache_dir='./cache_text_emb'):
    """Frozen CLIP text embeddings (N, 512) for a list of sentences.

    no_grad + eval; disk-cached on the md5 of the sentence list so repeated
    constructions (smoke tests, eval re-runs) don't reload CLIP.
    """
    h = hashlib.md5('\x00'.join(sentences).encode('utf-8')).hexdigest()
    cache_path = os.path.join(cache_dir, f'clip_text_{h}.pt')
    if os.path.isfile(cache_path):
        emb = torch.load(cache_path, map_location='cpu')
        assert emb.shape == (len(sentences), CLIP_TEXT_DIM)
        return emb
    from transformers import CLIPTokenizer, CLIPTextModelWithProjection
    # fork_rng: HF from_pretrained randomly initializes modules before
    # loading weights, consuming global RNG. Isolate it so whether the disk
    # cache hits or misses cannot change later trainable-module init draws
    # (init-equality invariant across the three systems).
    with torch.random.fork_rng(devices=[]):
        tok = CLIPTokenizer.from_pretrained(CLIP_NAME)
        model = CLIPTextModelWithProjection.from_pretrained(CLIP_NAME)
        model.eval()
        embs = []
        with torch.no_grad():
            for i in range(0, len(sentences), 64):
                batch = sentences[i:i + 64]
                inputs = tok(batch, padding=True, truncation=True,
                             max_length=77, return_tensors='pt')
                embs.append(model(**inputs).text_embeds.float())
        emb = torch.cat(embs, dim=0)
        del model
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(emb, cache_path)
    return emb


class TextClsPathway(nn.Module):
    """Shared frozen-CLIP-text classification pathway.

    Buffers (built once, no_grad):
      text_feats (C, K, 512) raw CLIP sentence embeddings
                              (K=1 baseline prompt, K=11 champion ensemble)
      stem_feats (C, 512)     parent-stem sentence embeddings (h18 degrade)
    Trainable:
      proj        Linear(512 -> head_dim)   [built FIRST: RNG-order matters
                                             for the init-equality check]
      attr_attn   zero-init MultiheadAttention (PA; champion only)
      logit_scale log-parameterized cosine scale (init log 20)
      logit_bias  scalar bias, init to the focal prior -log((1-p)/p)
    """

    def __init__(self, verbalizer_path, head_dim, prior_prob=0.01,
                 ensemble_path='', phase_attention=False,
                 phase_neighbor_attention=False,
                 sentence_dropout=0.0, stem_degrade=0.0,
                 prompt_template='a video of action {}',
                 cache_dir='./cache_text_emb'):
        super().__init__()
        actions = load_verbalizer_actions(verbalizer_path)
        self.num_classes = len(actions)
        self.sentence_dropout = float(sentence_dropout)
        self.stem_degrade = float(stem_degrade)
        self.phase_attention = bool(phase_attention)
        self.phase_neighbor_attention = bool(phase_neighbor_attention)
        # action -> phrase index (also used by the AS loss in meta_archs)
        self.register_buffer(
            'action_to_phrase',
            torch.tensor([a['phrase_idx'] for a in actions], dtype=torch.long),
            persistent=False)
        # action -> apparatus index (4-level hierarchy: the AS loss can also
        # supervise this top level, see activity_align_levels in meta_archs)
        self.register_buffer(
            'action_to_activity',
            torch.tensor([a['activity_idx'] for a in actions], dtype=torch.long),
            persistent=False)

        # env override so the verification script can force a synthetic
        # ensemble (all sentences == the baseline prompt) without new configs
        ensemble_path = os.environ.get('FG3_TEXT_ENSEMBLE', ensemble_path)

        if ensemble_path:
            with open(ensemble_path) as f:
                ensemble = json.load(f)
            key_of = resolve_ensemble_keys(actions, ensemble)
            K = len(next(iter(ensemble.values())))
            sents = []
            for a in actions:
                cls_sents = ensemble[key_of[a['id']]]
                assert len(cls_sents) == K
                sents.extend(cls_sents)
            flat = _embed_sentences(sents, cache_dir)
            text_feats = flat.view(self.num_classes, K, CLIP_TEXT_DIM)
            print(f"[t3-text] ensemble pathway: {self.num_classes} classes x "
                  f"{K} sentences from {ensemble_path}")
        else:
            prompts = [prompt_template.format(a['name']) for a in actions]
            text_feats = _embed_sentences(prompts, cache_dir).unsqueeze(1)
            print(f"[t3-text] name-only pathway: {self.num_classes} prompts "
                  f"'{prompt_template}'")
        self.register_buffer('text_feats', text_feats)   # (C, K, 512)

        # parent stems (h18): raw stem string, mirroring tifad's
        # `sents = [stems[c]] * K` (no template around the stem).
        stems = [humanize_phrase(a['phrase_name']) for a in actions]
        self.register_buffer('stem_feats', _embed_sentences(stems, cache_dir))

        # ---- trainable modules; ORDER MATTERS (init-equality RNG) ----
        self.proj = nn.Linear(CLIP_TEXT_DIM, head_dim)
        self.attr_attn = None
        if self.phase_attention:
            # exact tifad text.py H4b pattern (zero-init out_proj => the
            # attention contributes exactly 0 at step 0)
            self.attr_attn = nn.MultiheadAttention(head_dim, 4, batch_first=True)
            nn.init.zeros_(self.attr_attn.out_proj.weight)
            nn.init.zeros_(self.attr_attn.out_proj.bias)
            print("[t3-text] PA: zero-init class->phase attention active")
        # Phase-level attention (paper Eq. 3): the K-1 phase sentences attend
        # to themselves and their immediate temporal neighbours only (banded
        # additive mask, 0 on |i-j|<=1, -inf elsewhere); zero-init out_proj so
        # the class embeddings are unchanged at step 0. Trained in a SECOND
        # stage (train_shard.py --init_from <stage-1 ckpt> --freeze_except
        # text_pathway.phase_attn), exactly as on THUMOS14/ActivityNet.
        self.phase_attn = None
        if self.phase_neighbor_attention:
            assert self.attr_attn is not None, \
                "phase_neighbor_attention requires phase_attention (needs an ensemble)"
            K = self.text_feats.shape[1]
            self.phase_attn = nn.MultiheadAttention(head_dim, 4, batch_first=True)
            nn.init.zeros_(self.phase_attn.out_proj.weight)
            nn.init.zeros_(self.phase_attn.out_proj.bias)
            idx = torch.arange(K - 1)
            band = torch.zeros(K - 1, K - 1)
            band[(idx[:, None] - idx[None, :]).abs() > 1] = float('-inf')
            self.register_buffer('phase_band_mask', band, persistent=False)
            print(f"[t3-text] APA phase-level attention active: {K - 1} phases, "
                  "|i-j|<=1 band, zero-init out_proj")
        # scaled-cosine parameters (no RNG). Names chosen so make_optimizer's
        # no-decay pass catches them ('logit_scale' explicit, '*_bias').
        self.logit_scale = nn.Parameter(torch.tensor(math.log(20.0)))
        bias_init = -math.log((1.0 - prior_prob) / prior_prob)
        self.cls_logit_bias = nn.Parameter(torch.tensor(bias_init))

    def class_embeddings(self):
        """Return (C, head_dim) class embeddings (train-time stochastic)."""
        feats = self.text_feats                            # (C, K, 512)
        C, K, _ = feats.shape
        if self.training and self.stem_degrade > 0:
            # h18: replace ALL K sentences of a degraded class by its stem
            deg = (torch.rand(C, device=feats.device) < self.stem_degrade)
            stems = self.stem_feats.to(feats.dtype).unsqueeze(1).expand(-1, K, -1)
            feats = torch.where(deg.view(C, 1, 1), stems, feats)
        per = self.proj(feats)                             # (C, K, D)
        if self.phase_attn is not None:
            ph = per[:, 1:]                                # (C, K-1, D) phases only
            upd, _ = self.phase_attn(ph, ph, ph,
                                     attn_mask=self.phase_band_mask.to(ph.dtype),
                                     need_weights=False)
            per = torch.cat([per[:, :1], ph + upd], dim=1)
        if self.attr_attn is not None:
            q = per[:, :1]                                 # (C, 1, D) sentence 0
            kpm = None
            if self.training and self.sentence_dropout > 0:
                # h14: KV dropout on phase sentences, never the class prompt
                kpm = torch.rand(C, K, device=per.device) < self.sentence_dropout
                kpm[:, 0] = False
            upd, _ = self.attr_attn(q, per, per, key_padding_mask=kpm,
                                    need_weights=False)
            emb = (q + upd).squeeze(1)                     # (C, D)
        elif K == 1:
            emb = per[:, 0]
        else:
            emb = per.mean(1)                              # H4a mean (unused by t3)
        return emb

    def cosine_logits(self, feat, cls_emb):
        """feat (B, D, T) trunk features, cls_emb (C, D) or (B, C, D).

        Returns (B, C, T) scaled-cosine logits.
        """
        f = F.normalize(feat, dim=1)
        e = F.normalize(cls_emb, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        if e.dim() == 2:
            logits = torch.einsum('bdt,cd->bct', f, e.to(f.dtype))
        else:
            logits = torch.einsum('bdt,bcd->bct', f, e.to(f.dtype))
        return logits * scale.to(f.dtype) + self.cls_logit_bias.to(f.dtype)


class TextVideoCrossAttnLevel(nn.Module):
    """Ti-FAD TiCA re-implementation for ONE FPN level (system 2).

    Mirrors the tifad fork's TransformerBlock TiCA stage (blocks.py ~775-878):
      (1) class score map from the current text embeddings,
      (2) dynamic top-k salient points (ratio 0.1 of level length),
      (3) Salient Attentive Mask: merged-center gaussians with per-position
          predicted sigma (SigmaLayer: conv1d + sigmoid),
      (4) TVCA cross-attention with the SAM as importance weights, then an
          FFN — here only the TEXT stream is updated.
    Documented deviations from the fork (faithful-but-minimal):
      - Applied to the cls-head TRUNK features at the FPN outputs instead of
        interleaved inside every branch transformer block (keeps the
        ActionFormerWithCLIP backbone and the localization heads untouched).
      - Only the text->video attention direction is kept (the fork's TVCA
        also updates the video stream, which feeds its reg head; our reg
        head must stay untouched, so the video update is dropped). The SAM
        importance therefore weights the VIDEO KEYS of the text update
        (log-additive attention bias) — the natural transposition of the
        fork's weighting of the v<-t direction.
      - No separate text self-attention (attn_t) — the shared pathway's
        class embeddings are already per-class pooled sentence vectors.
    """

    def __init__(self, dim, n_head=4, max_len=288, topk_ratio=0.1,
                 threshold=10, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.sigma_conv = nn.Conv1d(dim, 1, kernel_size=3, padding=1)
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout,
                                          batch_first=True)
        self.norm_t = nn.LayerNorm(dim)
        self.mlp_t = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout))
        self.dropout_t = nn.Dropout(dropout)
        self.base_topk_ratio = topk_ratio
        self.base_threshold = threshold
        self.initial_length = max(1, max_len // 2)

    def _sam_weights(self, scores, sigma, valid):
        """Salient Attentive Mask, replicating gaussian_kernels_with_threshold.

        scores (B, T) max-over-class score map, sigma (B, T) in (0,1),
        valid (B, T) bool. Returns (B, T) importance weights in [0, 1].
        """
        B, T = scores.shape
        k = max(1, int(T * self.base_topk_ratio))
        thr = self.base_threshold * (T / self.initial_length)
        masked = scores.masked_fill(~valid, float('-inf'))
        _, top_idx = masked.topk(min(k, T), dim=1)
        top_idx, _ = torch.sort(top_idx, dim=1)
        time_steps = torch.arange(T, device=scores.device).float()
        rows = []
        for b in range(B):
            # merge overlapping centers (threshold-based averaging)
            merged = []
            for idx in top_idx[b].tolist():
                if not merged or abs(idx - merged[-1]) > thr:
                    merged.append(float(idx))
                else:
                    merged[-1] = (merged[-1] + idx) / 2.0
            kernels = []
            for center in merged:
                # per-center sigma keeps GRADIENT to the sigma conv (as in
                # the fork: only the top-k CENTER selection is non-diff)
                s = sigma[b, int(center)].float()
                kernels.append(torch.exp(
                    -0.5 * (time_steps - center) ** 2 / (s ** 2 + 1e-8)))
            rows.append(torch.stack(kernels, dim=0).sum(dim=0))
        weights = torch.stack(rows, dim=0)                               # (B,T)
        weights = weights / (weights.max(dim=1, keepdim=True)[0].detach() + 1e-5)
        return weights

    def forward(self, feat, mask, txt, prelim_logits):
        """feat (B, D, T); mask (B, 1, T) bool; txt (B, C, D);
        prelim_logits (B, C, T). Returns updated txt (B, C, D)."""
        B, D, T = feat.shape
        valid = mask.squeeze(1).bool()
        # run the conv in the module's own dtype (bf16 under deepspeed bf16 —
        # feat.float() vs bf16 weights raises a dtype mismatch), then do the
        # sigmoid/gaussian math in float32
        sigma = torch.sigmoid(self.sigma_conv(feat).float()).squeeze(1)   # (B,T)
        # score map only picks the top-k centers (non-differentiable in the
        # fork too); sigma keeps its gradient through the gaussian kernels
        scores = prelim_logits.float().max(dim=1).values.detach()        # (B,T)
        sam = self._sam_weights(scores, sigma, valid)
        v = feat.permute(0, 2, 1)                                        # (B,T,D)
        # additive attention bias over video keys: log SAM + invalid -> -inf
        bias = torch.log(sam.to(v.dtype) + 1e-4)
        bias = bias.masked_fill(~valid, float('-inf'))
        n_head = self.attn.num_heads
        attn_mask = bias.unsqueeze(1).expand(B, txt.shape[1], T)
        attn_mask = attn_mask.repeat_interleave(n_head, dim=0)           # (B*h,C,T)
        upd, _ = self.attn(txt, v, v, attn_mask=attn_mask, need_weights=False)
        txt = txt + self.dropout_t(upd)
        txt = txt + self.mlp_t(self.norm_t(txt))
        return txt


class DurationPrior(nn.Module):
    """Eval-time log-normal duration prior (champion component d).

    Stats JSON from scripts/gen_finegym_duration_prior.py: per SEEN class a
    log-normal (mu, sigma) fitted on Dec16 train durations; per HELD-OUT
    class a moment-matched Jaccard-weighted mixture over seen classes using
    the 48-dim attribute rows. score *= exp(-gamma/2 * z^2),
    z = (log dur_sec - mu_c) / sigma_c; gamma env-tunable via FG3_DP_GAMMA.
    """

    def __init__(self, stats_path, gamma=0.3):
        super().__init__()
        with open(stats_path) as f:
            stats = json.load(f)
        C = len(stats['classes'])
        mu = torch.zeros(C)
        sigma = torch.ones(C)
        for cid_s, entry in stats['classes'].items():
            cid = int(cid_s)
            mu[cid] = float(entry['mu'])
            sigma[cid] = max(float(entry['sigma']), 1e-3)
        self.register_buffer('mu', mu, persistent=False)
        self.register_buffer('sigma', sigma, persistent=False)
        self.gamma = float(os.environ.get('FG3_DP_GAMMA', gamma))
        print(f"[t3-DP] duration prior loaded ({C} classes, "
              f"gamma={self.gamma}) from {stats_path}")

    @torch.no_grad()
    def rescore(self, scores, labels, dur_sec):
        """scores (N,), labels (N,) long, dur_sec (N,) seconds -> new scores."""
        if self.gamma == 0 or scores.numel() == 0:
            return scores
        mu = self.mu.to(scores.device)[labels]
        sigma = self.sigma.to(scores.device)[labels]
        z = (dur_sec.clamp(min=1e-3).log() - mu) / sigma
        return scores * torch.exp(-0.5 * self.gamma * z * z)
