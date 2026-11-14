# ---------------------------------------------------------------------------
# PDA: "Decompose and Transfer: CoT-Prompting Enhanced Alignment for
# Open-Vocabulary Temporal Action Detection" (Zhu et al., CVPR 2026,
# arXiv 2603.24030), re-implemented on top of Ti-FAD (this fork).
#
# Paper recap (Sec. 3):
#   CSD  (Eq. 1)   d_c^p for p in {start, middle, end, glob} authored offline by
#                  an LLM (configs/thumos14_pda_desc.json, scripts/
#                  gen_pda_desc.py), wrapped as "a video of people's motion
#                  that {d}", encoded by the FROZEN CLIP text encoder
#                  s_c^p = f_text(d_c^p) (512-d pooled EOT state, BEFORE the
#                  fork's text_projection) and mapped by PHASE-SPECIFIC trained
#                  projections t_c^p = Phi_txt^p(s_c^p) in R^D. Phi_txt^p is a
#                  Linear(512 -> D, no bias) initialised exactly like the fork's
#                  text_projection -- "four of them" (PDATextEncoder).
#   TIF  (Eq. 2-3) S_fg^p = Softmax_T( max_C( F_v F_t^{pT} ) ), binarised at
#                  its temporal mean, hard-gates F_v -> F_v^p (pda_tif).
#   APA  (Eq. 4-7) per phase: Fbar^p = Softmax(Q(F_v^p) K(F_t^p)^T / sqrt(D))
#                  V(F_t^p)  [NO residual in the paper; TIFAD_PDA_RESID=1 adds
#                  F_v^p + attn], C_cls^p = Fbar^p F_t^{pT} (T x C); adaptive
#                  weights omega from the supplementary's weighting network
#                  (feature replicated into 4 virtual tokens + 4 learnable
#                  phase embeddings -> one 4-head MHSA layer -> Linear(D,1024)
#                  -ReLU-Linear(1024,1) per token -> softmax over phases);
#                  C_cls = sum_p omega_p C_cls^p.
#   LOC  (Eq. 9)   F_hat = L([F^start; F^mid; F^end; F^glob]) with L a
#                  3-layer MLP (hidden 1024); F_hat REPLACES F_v as input of
#                  the foreground-aware / regression (/ centerness) heads.
#   Loss (Eq. 10)  L = L_cls + L_fg + L_loc -- the fork's existing losses are
#                  reused on C_cls (sigmoid focal, NOT the paper's CE; see the
#                  calibration note in PDAPhaseAlign.forward).
#
# Everything is gated by env TIFAD_PDA=1 (+ TIFAD_PDA_DESC=<json>); when
# unset this file is never imported and the fork is byte-identical.
# ---------------------------------------------------------------------------
import json
import math

import torch
import os as _pda_os
_PDA_TIF = _pda_os.environ.get('TIFAD_PDA_TIF', 'hard')
_PDA_LOCRES = _pda_os.environ.get('TIFAD_PDA_LOCRES', '') == '1'
_PDA_WSIG = _pda_os.environ.get('TIFAD_PDA_WSIG', '') == '1'
import torch.nn as nn
import torch.nn.functional as F

PDA_PHASES = ('start', 'middle', 'end', 'glob')
# paper: 'a video of people's motion that [Description]', Description being
# "The person would {text}." -> lower-cased lead-in re-attached here
PDA_PROMPT = "a video of people's motion that the person would {}"


class PDATextEncoder(nn.Module):
    """CSD text side (Eq. 1). Owned by TextFeatures (text.py) so the four
    Phi_txt^p projections sit next to the fork's own text_projection."""

    _ALIASES = {'start': ('start',), 'middle': ('middle', 'mid'),
                'end': ('end',), 'glob': ('glob', 'global')}

    def __init__(self, desc_file, clip_dim, emb_dim):
        super().__init__()
        with open(desc_file, 'r') as f:
            self.desc = json.load(f)
        self.proj = nn.ModuleDict(
            {p: nn.Linear(clip_dim, emb_dim, bias=False) for p in PDA_PHASES})
        for p in PDA_PHASES:      # same init as the fork's text_projection
            nn.init.normal_(self.proj[p].weight, std=clip_dim ** -0.5)
        self._cache = {}          # tuple(cls_names) -> (4, C, clip_dim) CPU
        self.max_tokens = 0       # longest prompt seen (CLIP context is 77)

    def _text(self, c, p):
        e = self.desc.get(c)
        if e is None:
            raise KeyError(f"[PDA] class '{c}' missing from TIFAD_PDA_DESC")
        for k in self._ALIASES[p]:
            if k in e:
                return e[k]
        raise KeyError(f"[PDA] class '{c}' has no '{p}' description")

    @torch.no_grad()
    def _encode(self, text_features, cls_names):
        """Frozen CLIP: pooled EOT state BEFORE text_projection -> (4, C, 512)."""
        prompts = [PDA_PROMPT.format(self._text(c, p).rstrip('.'))
                   for p in PDA_PHASES for c in cls_names]
        tok, model = text_features.tokenizer, text_features.txt_model
        dev = next(model.parameters()).device
        toks = tok(prompts, padding=True, truncation=True, max_length=77,
                   return_tensors='pt')
        n_tok = int(toks['attention_mask'].sum(-1).max())
        self.max_tokens = max(self.max_tokens, n_tok)
        toks = toks.to(dev)
        pooled = model.text_model(input_ids=toks['input_ids'],
                                  attention_mask=toks['attention_mask'])[1]
        return pooled.detach().float().cpu().view(
            len(PDA_PHASES), len(cls_names), -1)

    def forward(self, text_features, cls_names):
        """-> {phase: (C, D)} t_c^p = Phi_txt^p(s_c^p), un-normalised."""
        key = tuple(cls_names)
        if key not in self._cache:
            self._cache[key] = self._encode(text_features, cls_names)
        dev = self.proj['glob'].weight.device
        s = self._cache[key].to(dev)                              # (4, C, 512)
        return {p: self.proj[p](s[i]) for i, p in enumerate(PDA_PHASES)}


def pda_tif(x, t_p, valid):
    """TIF (Eq. 2-3) for one phase. x (B,T,D) F_v; t_p (C,D) unit-norm phase
    texts; valid (B,T) bool. Returns (F_v^p (B,T,D), gate (B,T)).
    Softmax runs over the VALID temporal positions only (padding excluded);
    the binarisation threshold is the mean over valid positions. The gate is
    a hard, non-differentiable mask exactly as written (no straight-through
    estimator)."""
    sim = x @ t_p.t()                                             # (B,T,C)
    score = sim.max(dim=-1).values.masked_fill(~valid, float('-inf'))
    score = score.softmax(dim=-1)                                 # Eq. 2
    n_valid = valid.sum(dim=-1, keepdim=True).clamp(min=1)
    thr = score.sum(dim=-1, keepdim=True) / n_valid.to(score.dtype)
    # Reproduction variants (TIFAD_PDA_TIF): 'hard' = paper as written;
    # 'soft' = differentiable gate min(score/thr, 1); 'off' = no filtering.
    if _PDA_TIF == 'soft':
        gate = (score / thr.clamp(min=1e-12)).clamp(max=1.0) * valid.to(x.dtype)
    elif _PDA_TIF == 'off':
        gate = valid.to(x.dtype)
    else:
        gate = torch.logical_and(score >= thr, valid).to(x.dtype) # \hat S_fg^p
    return x * gate.unsqueeze(-1), gate                           # Eq. 3


class PDAPhaseAlign(nn.Module):
    """TIF + APA + localisation fusion (Sec. 3.3-3.4), shared over FPN levels."""

    def __init__(self, emb_dim, resid=False, n_head=4, w_hidden=1024,
                 loc_hidden=1024):
        super().__init__()
        self.emb_dim = emb_dim
        self.resid = resid
        # Eq. 4: one Q/K/V set (the paper writes the same Q, K, V for every
        # phase -> shared across phases)
        self.q = nn.Linear(emb_dim, emb_dim)
        self.k = nn.Linear(emb_dim, emb_dim)
        self.v = nn.Linear(emb_dim, emb_dim)
        # weighting network W (supplementary Sec. 7): 4 virtual tokens +
        # learnable phase embeddings -> 4-head MHSA -> MLP(D,1024,1) -> softmax
        self.phase_emb = nn.Parameter(torch.zeros(len(PDA_PHASES), emb_dim))
        nn.init.normal_(self.phase_emb, std=0.02)
        self.w_norm = nn.LayerNorm(emb_dim)
        self.w_attn = nn.MultiheadAttention(emb_dim, n_head, batch_first=True)
        self.w_mlp = nn.Sequential(nn.Linear(emb_dim, w_hidden), nn.ReLU(),
                                   nn.Linear(w_hidden, 1))
        # Eq. 9: L = three-layer MLP, hidden 1024, 4D -> D
        self.loc = nn.Sequential(
            nn.Linear(len(PDA_PHASES) * emb_dim, loc_hidden), nn.ReLU(),
            nn.Linear(loc_hidden, loc_hidden), nn.ReLU(),
            nn.Linear(loc_hidden, emb_dim))

    def phase_weights(self, x):
        """x (B,T,D) -> omega (B,T,4), softmax over phases."""
        B, T, D = x.shape
        tok = x.unsqueeze(2) + self.phase_emb.view(1, 1, -1, D)   # (B,T,4,D)
        tok = tok.reshape(B * T, len(PDA_PHASES), D)
        h = self.w_norm(tok)
        tok = tok + self.w_attn(h, h, h, need_weights=False)[0]
        w = self.w_mlp(tok).squeeze(-1)                           # (B*T, 4)
        if _PDA_WSIG:   # main-text Eq. 6: per-phase sigmoid instead of softmax
            return torch.sigmoid(w).view(B, T, len(PDA_PHASES))
        return w.softmax(dim=-1).view(B, T, len(PDA_PHASES))

    def forward(self, fpn_feats, fpn_masks, phase_texts, logit_scale,
                bias_lang, bias0):
        """
        fpn_feats   : list of (B, D, T_l) backbone pyramid features F_v
        fpn_masks   : list of (B, 1, T_l) bool
        phase_texts : {phase: (C, D)} t_c^p (un-normalised, with graph)
        Returns (cls_logits: list of (B, T_l, C), fused: list of (B, D, T_l),
                 stats dict)
        """
        # L2-normalise the phase text banks (mirrors the stock matmul path,
        # which normalises the class text before scoring)
        tn = {p: F.normalize(phase_texts[p], dim=-1) for p in PDA_PHASES}
        # sigmoid-focal calibration (fork loss, not the paper's CE): the same
        # text-derived prior bias and learned temperature the stock matmul
        # applies, computed from the global phase text
        cls_bias = tn['glob'] @ bias_lang + bias0                 # (C,)
        scale = 1.0 / math.sqrt(self.emb_dim)
        k = {p: self.k(tn[p]) for p in PDA_PHASES}
        v = {p: self.v(tn[p]) for p in PDA_PHASES}

        out_logits, fused, keep = [], [], []
        for feat, mask in zip(fpn_feats, fpn_masks):
            x = feat.transpose(1, 2)                              # (B,T,D)
            m = mask.squeeze(1).bool()                            # (B,T)
            omega = self.phase_weights(x)                         # (B,T,4)
            logits, streams = None, []
            for i, p in enumerate(PDA_PHASES):
                xp, gate = pda_tif(x, tn[p], m)                   # Eq. 2-3
                keep.append(gate.sum() / m.sum().clamp(min=1))
                attn = torch.softmax(
                    (self.q(xp) @ k[p].t()) * scale, dim=-1)      # (B,T,C)
                fbar = attn @ v[p]                                # Eq. 4
                if self.resid:
                    fbar = xp + fbar
                c_p = fbar @ tn[p].t()                            # Eq. 5
                c_p = omega[..., i:i + 1] * c_p                   # Eq. 7
                logits = c_p if logits is None else logits + c_p
                streams.append(xp)
            logits = logits / logit_scale.exp() + cls_bias.view(1, 1, -1)
            out_logits.append(logits)
            fh = self.loc(torch.cat(streams, dim=-1))             # Eq. 9
            if _PDA_LOCRES:                                       # variant: add to F_v
                fh = x + fh
            fh = fh * m.unsqueeze(-1).to(fh.dtype)
            fused.append(fh.transpose(1, 2))                      # (B,D,T)
        stats = {'tif_keep': float(torch.stack(keep).mean())}
        return out_logits, fused, stats
