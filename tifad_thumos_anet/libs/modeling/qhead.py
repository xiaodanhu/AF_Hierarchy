"""QH (TIFAD_QH=1): CLASS-AGNOSTIC LOCALIZATION-QUALITY HEAD, VSA v4.

Review: /data3/xiaodan8/actionformer6/autoresearch/vsa_review_v4.md.
Literature: TadTR actionness head (RoIAlign over the 1.5x-expanded
candidate, target = tIoU with the best GT, score = sqrt(p_cls * q)), SSN
completeness (pre/in/post context, incomplete negatives = >=80% of the
proposal inside a GT but tIoU < 0.3), BREM (quality modules trained
SEPARATELY on frozen features for offline-feature detectors), IoU-Net
(NMS ordered by the quality-aware score).

Per location n on pyramid level l with its DETACHED decoded span [s, e]
(level-0 grid units, L = e - s), the head reads (all stop-gradient):
  * f_n            : the location's own pre-VSA cls-trunk feature (D)
  * 12 span pools  : from cls_feat[0] (pre-VSA, level 0) via masked
                     integral features -- pre-context [s-0.5L, s), K=10
                     ordered in-span chunks, post-context (e, e+0.5L]
                     (SSN 50% extension == TadTR 1.5x RoI). Regions
                     outside the valid video pool to ZERO (explicit
                     "no context", COMPL convention).
  * phase map      : (TIFAD_QH_PHASEMAP=1) cos(chunk_j, phase_k(c*)),
                     j over the 12 pools, k over the K=10 CALIBRATED text
                     phase embeddings (text.py `_vsa_phase_embs` through
                     text_proj(normalize(x)/2), i.e. the base matmul
                     space) of the class c* (GT class at positives in
                     training, argmax class otherwise, the candidate's
                     own class at inference); flattened, 120 dims.
  * structure     : (TIFAD_QH_STRUCT=1) CLASS-AGNOSTIC structure of the
                     expanded span: the 12 x 12 chunk cosine-similarity
                     matrix (66 off-diagonal entries: is the pre-context
                     similar to the in-span content -> under-extension /
                     fragment, or different -> real boundary), the 12
                     cosines chunk-vs-f_n, and the frozen FG head's
                     actionness profile sigmoid(fg) pooled over the 12
                     chunks (SSN's completeness cue read from the
                     class-agnostic foreground head). TIFAD_QH_RAW=0 drops
                     the raw D-blocks so the head sees ONLY structure
                     (+ phase map): the anti-memorisation variant (smoke:
  * level/duration : (TIFAD_QH_LVLFEAT=1, R4) pyramid-level one-hot (F)
                     and the normalized log-duration log(1+L)/log(1+Tmax)
                     of the decoded span, so the head can learn that an
                     over-extended coarse-level candidate is low quality.
Each D-block is LayerNormed; MLP (in -> 256 -> 256 -> 1, ZERO-init last
layer => q == 0.5 everywhere at init; TIFAD_QH_DROPOUT on the input,
default 0) -> sigmoid q in (0, 1).

Training (meta_archs, TRAIN ONLY, seen-class GT only): soft-label BCE of
q against the tIoU between the decoded span and its best GT (TadTR/GFL;
TIFAD_QH_LOSS=l1 alternative); spans with tIoU < 0.1 against every GT
are target 0. Sampled set = every valid location with tIoU >= 0.1 plus
TIFAD_QH_BG_RATIO (default 1.0) x as many random background locations.
Plus an SSN-style pairwise margin ranking loss (TIFAD_QH_RANK_W, default
0.5; margin TIFAD_QH_MARGIN, default 0.2): per GT, full candidates
inside the GT, tIoU < 0.3), up to 16 x 16 random pairs per GT.
Diagnostics: qh_pair_acc, qh_pearson, qh_spearman, qh_n, qh_npairs.

Inference (meta_archs.inference_single_video): q for every candidate
that survives the champion's own pre-NMS selection; pre-NMS score =
score^(1 - beta) * q^beta (TIFAD_QH_BETA, default 0.5; beta == 0 is the
byte-identical champion path); soft-NMS then runs on the combined score,
so it is ORDERED by quality. Byte-inert when TIFAD_QH is unset (module
never constructed). All params live under the `qhead` attribute so
TIFAD_FREEZE_EXCEPT=qhead selects exactly them.

R1 (TIFAD_BR=1, `brhead`): BOUNDARY REFINEMENT HEAD (BREM / AFSD style),
sequential on the frozen champion. SAME detached inputs as the quality
head (f_n + the 12 context pools of the 1.5x-expanded decoded span) PLUS
the level one-hot and normalized log-duration; MLP -> 2 outputs ->
corrections (ds, de) = tanh(.) * TIFAD_BR_RANGE (default 0.5) * L per
side, ZERO-init last layer => corrections exactly 0 at init. Train
(candidates with tIoU >= 0.5 with a seen GT only): L1 on the relative
corrections against the clipped true relative offsets to that GT +
DIoU of the corrected span vs the GT (both summed / loss_normalizer,
times TIFAD_BR_W, default 1.0). Eval: TIFAD_BR_APPLY=1 applies the
corrections to the decoded spans BEFORE soft-NMS (scores unchanged; the
quality head, if on, scores the ORIGINAL decoded span it was trained
on); TIFAD_BR_ITERS=2 re-pools on the refined span once more.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


def lvl_dur_feat(lvl, spans, n_levels, max_len):
    """lvl: (N,) long pyramid level; spans: (N, 2) level-0 units.
    Returns (N, n_levels + 1): one-hot level + log(1+L)/log(1+max_len)."""
    oh = F.one_hot(lvl.long().clamp(0, n_levels - 1), n_levels).float()
    L = (spans[:, 1] - spans[:, 0]).clamp(min=0).float()
    ld = torch.log1p(L) / math.log1p(float(max_len))
    return torch.cat([oh, ld.unsqueeze(-1)], dim=1)


def pool_span_chunks(x0, mask0, spans, K=10, ctx=0.5):
    """Masked integral-feature pooling of K in-span chunks plus one
    pre-context and one post-context pool per span (vsa._pool_chunks
    generalised to float spans on the level-0 grid with context).
    x0: (B, D, T0) level-0 features; mask0: (B, T0) bool valid;
    spans: (B, N, 2) float [s, e] in level-0 grid units.
    Returns (B, N, K + 2, D): [pre, in_1 .. in_K, post]. Pools whose
    integer range is empty or fully outside [0, T0) are ZERO."""
    B, D, T0 = x0.shape
    N = spans.size(1)
    m = mask0.to(x0.dtype).unsqueeze(1)                          # (B, 1, T0)
    cs = F.pad((x0 * m).cumsum(dim=-1), (1, 0)).transpose(1, 2)  # (B, T0+1, D)
    cm = F.pad(m.cumsum(dim=-1), (1, 0))[:, 0, :]                # (B, T0+1)
    s = spans[..., 0]
    e = torch.maximum(spans[..., 1], s + 1e-3)
    L = e - s                                                    # (B, N)
    k = torch.arange(K + 1, device=x0.device, dtype=s.dtype)
    bnd = s.unsqueeze(-1) + L.unsqueeze(-1) * k / K              # (B, N, K+1)
    lo_in = bnd[..., :-1].floor()
    hi_in = torch.maximum(bnd[..., 1:].ceil(), lo_in + 1)        # >= 1 unit
    lo_pre = (s - ctx * L).floor().unsqueeze(-1)
    hi_pre = s.floor().unsqueeze(-1)                             # [s-cL, s)
    lo_post = e.ceil().unsqueeze(-1)
    hi_post = (e + ctx * L).ceil().unsqueeze(-1)                 # (e, e+cL]
    lo = torch.cat([lo_pre, lo_in, lo_post], dim=-1).clamp(0, T0).long()
    hi = torch.cat([hi_pre, hi_in, hi_post], dim=-1).clamp(0, T0).long()
    hi = torch.maximum(hi, lo)                                   # empty ok
    lo = lo.reshape(B, N * (K + 2))
    hi = hi.reshape(B, N * (K + 2))
    bidx = torch.arange(B, device=x0.device).unsqueeze(-1)
    num = cs[bidx, hi] - cs[bidx, lo]                            # (B, N(K+2), D)
    cnt = cm[bidx, hi] - cm[bidx, lo]                            # (B, N(K+2))
    Z = num / cnt.clamp(min=1.0).unsqueeze(-1)
    Z = Z * (cnt > 0).to(Z.dtype).unsqueeze(-1)
    return Z.reshape(B, N, K + 2, D)


def seg_tiou(spans, gts):
    """spans: (N, 2), gts: (G, 2) -> tIoU (N, G), inside fraction
    |span & gt| / |span| (N, G)."""
    s = spans[:, None, 0]
    e = spans[:, None, 1]
    gs = gts[None, :, 0]
    ge = gts[None, :, 1]
    inter = (torch.minimum(e, ge) - torch.maximum(s, gs)).clamp(min=0)
    union = (e - s).clamp(min=1e-6) + (ge - gs).clamp(min=1e-6) - inter
    return inter / union.clamp(min=1e-6), inter / (e - s).clamp(min=1e-6)


def _rank_corr(a, b):
    """Pearson and Spearman correlation of two 1-D tensors (float)."""
    a = a.float()
    b = b.float()
    if a.numel() < 3 or a.std() < 1e-8 or b.std() < 1e-8:
        z = a.new_zeros(())
        return z, z

    def _pear(u, v):
        u = u - u.mean()
        v = v - v.mean()
        return (u * v).sum() / (u.norm() * v.norm()).clamp(min=1e-8)

    def _ranks(u):
        r = torch.empty_like(u)
        r[u.argsort()] = torch.arange(u.numel(), device=u.device,
                                      dtype=u.dtype)
        return r
    return _pear(a, b), _pear(_ranks(a), _ranks(b))


class QualityHead(nn.Module):

    def __init__(self, dim, K=10, phasemap=False, hidden=256, ctx=0.5,
                 loss='bce', rank_w=0.5, margin=0.2, bg_ratio=1.0,
                 beta=0.5, raw=True, struct=False, dropout=0.0,
                 lvlfeat=False, n_levels=6):
        super().__init__()
        assert K >= 1, "[QH] K must be >= 1"
        assert loss in ('bce', 'l1'), f"[QH] unknown loss '{loss}'"
        assert raw or struct or phasemap, "[QH] no input selected"
        self.lvlfeat = bool(lvlfeat)
        self.n_levels = int(n_levels)
        self.K = int(K)
        self.ctx = float(ctx)
        self.phasemap = bool(phasemap)
        self.raw = bool(raw)
        self.struct = bool(struct)
        self.loss_type = loss
        self.rank_w = float(rank_w)
        self.margin = float(margin)
        self.bg_ratio = float(bg_ratio)
        self.beta = float(beta)
        self.n_blocks = 1 + self.K + 2                     # f_n + pools
        self.norm = nn.LayerNorm(dim)
        P = self.K + 2
        in_dim = (self.n_blocks * dim if self.raw else 0) + (
            P * self.K if self.phasemap else 0) + (
            (P * (P - 1)) // 2 + P + P if self.struct else 0) + (
            self.n_levels + 1 if self.lvlfeat else 0)
        self.in_dim = in_dim
        _iu = torch.triu_indices(P, P, offset=1)
        self.register_buffer('triu_i', _iu[0], persistent=False)
        self.register_buffer('triu_j', _iu[1], persistent=False)
        self.mlp = nn.Sequential(
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    # ------------------------------------------------------------------
    def logits(self, f_n, pools, phase=None, fgp=None, ld=None):
        """f_n: (N, D) location features; pools: (N, K+2, D); phase:
        (N, K, D) calibrated phase bank of each candidate's class (needed
        iff phasemap); fgp: (N, K+2) FG-head logits pooled over the same
        chunks (needed iff struct); ld: (N, n_levels+1) level one-hot +
        normalized log-duration (needed iff lvlfeat). Returns quality
        logits (N,). Inputs are DETACHED here: the head never sends
        gradient into the trunk / text / fg head."""
        f_n = f_n.detach()
        pools = pools.detach()
        parts = []
        dt = f_n.dtype
        if self.raw:
            blocks = torch.cat([f_n.unsqueeze(1), pools], dim=1)  # (N, 1+K+2, D)
            parts.append(self.norm(blocks).flatten(1))
        pn = F.normalize(pools, dim=-1)                          # (N, K+2, D)
        if self.struct:
            assert fgp is not None, "[QH-STRUCT] fg profile missing"
            sim = torch.einsum('nid,njd->nij', pn, pn)           # (N, P, P)
            parts.append(sim[:, self.triu_i, self.triu_j].to(dt))
            parts.append(torch.einsum('njd,nd->nj', pn,
                                      F.normalize(f_n, dim=-1)).to(dt))
            parts.append(torch.sigmoid(fgp.detach().float()).to(dt))
        if self.phasemap:
            assert phase is not None, "[QH] phase bank missing"
            pm = torch.einsum('njd,nkd->njk', pn,
                              F.normalize(phase.detach(), dim=-1))  # (N, K+2, K)
            parts.append(pm.flatten(1).to(dt))
        if self.lvlfeat:
            assert ld is not None, "[QH-LVLFEAT] level/duration feat missing"
            parts.append(ld.detach().to(dt))
        x = torch.cat(parts, dim=1)
        return self.mlp(x).squeeze(-1)

    def forward(self, f_n, pools, phase=None, fgp=None, ld=None):
        return torch.sigmoid(
            self.logits(f_n, pools, phase, fgp, ld).float())

    # ------------------------------------------------------------------
    def train_losses(self, q_logit, spans, gts, valid, generator=None):
        """q_logit: (N,) head logits for the N locations of ONE item;
        spans: (N, 2) decoded spans (level-0 units); gts: (G, 2) seen GT
        spans of the item; valid: (N,) bool. Returns a dict with tensors
        'qual' (sum of per-sample losses), 'n', 'rank' (sum over pairs),
        'npairs', 'pair_correct', and the sampled (q, target) for the
        correlation diagnostics ('q_s', 't_s')."""
        dev = q_logit.device
        out = {'qual': q_logit.new_zeros(()), 'n': 0,
               'rank': q_logit.new_zeros(()), 'npairs': 0,
               'pair_correct': 0, 'q_s': None, 't_s': None}
        if gts is None or gts.numel() == 0 or not bool(valid.any()):
            return out
        with torch.no_grad():
            tiou, inside = seg_tiou(spans, gts)                # (N, G)
            best, _ = tiou.max(dim=1)
            target = torch.where(best >= 0.1, best, torch.zeros_like(best))
            pool = valid & (best >= 0.1)
            bg = valid & (best < 0.1)
            n_pool = int(pool.sum())
            n_bg_keep = int(min(int(bg.sum()),
                                max(int(round(self.bg_ratio * n_pool)), 16)))
            sel = pool.clone()
            if n_bg_keep > 0:
                bg_idx = bg.nonzero(as_tuple=True)[0]
                pick = torch.randperm(bg_idx.numel(), device=dev,
                                      generator=generator)[:n_bg_keep]
                sel[bg_idx[pick]] = True
        if not bool(sel.any()):
            return out
        q_sel = q_logit[sel].float()
        t_sel = target[sel].float()
        if self.loss_type == 'bce':
            qual = F.binary_cross_entropy_with_logits(
                q_sel, t_sel, reduction='sum')
        else:
            qual = (torch.sigmoid(q_sel) - t_sel).abs().sum()
        out.update(qual=qual, n=int(sel.sum()),
                   q_s=torch.sigmoid(q_sel).detach(), t_s=t_sel)
        # ---- SSN-style fragment pairs, per GT
        if self.rank_w == 0:
            return out
        q_all = torch.sigmoid(q_logit.float())
        rank = q_logit.new_zeros(())
        npairs = 0
        correct = 0
        with torch.no_grad():
            full_m = valid.unsqueeze(1) & (tiou >= 0.6)         # (N, G)
            frag_m = valid.unsqueeze(1) & (inside >= 0.8) & (tiou < 0.3)
        for g in range(gts.size(0)):
            fi = full_m[:, g].nonzero(as_tuple=True)[0]
            ri = frag_m[:, g].nonzero(as_tuple=True)[0]
            if fi.numel() == 0 or ri.numel() == 0:
                continue
            if fi.numel() > 16:
                fi = fi[torch.randperm(fi.numel(), device=dev,
                                       generator=generator)[:16]]
            if ri.numel() > 16:
                ri = ri[torch.randperm(ri.numel(), device=dev,
                                       generator=generator)[:16]]
            diff = q_all[fi].unsqueeze(1) - q_all[ri].unsqueeze(0)
            rank = rank + F.relu(self.margin - diff).sum()
            npairs += diff.numel()
            correct += int((diff > 0).sum())
        out.update(rank=rank, npairs=npairs, pair_correct=correct)
        return out


def qh_corr(q_s, t_s):
    return _rank_corr(q_s, t_s)


class BoundaryHead(nn.Module):
    """R1 boundary refinement head (see module docstring)."""

    def __init__(self, dim, K=10, hidden=256, ctx=0.5, rng=0.5, w=1.0,
                 n_levels=6, iters=1):
        super().__init__()
        self.K = int(K)
        self.ctx = float(ctx)
        self.rng = float(rng)
        self.w = float(w)
        self.n_levels = int(n_levels)
        self.iters = int(iters)
        self.norm = nn.LayerNorm(dim)
        self.in_dim = (1 + self.K + 2) * dim + self.n_levels + 1
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def rel(self, f_n, pools, ld):
        """Relative corrections (N, 2) in (-rng, rng) units of L."""
        blocks = torch.cat([f_n.detach().unsqueeze(1), pools.detach()], dim=1)
        x = torch.cat([self.norm(blocks).flatten(1),
                       ld.detach().to(f_n.dtype)], dim=1)
        return torch.tanh(self.mlp(x).float()) * self.rng

    def apply(self, spans, rel):
        """spans (N, 2) level-0 units -> corrected spans (N, 2)."""
        L = (spans[:, 1] - spans[:, 0]).clamp(min=1e-3).float()
        s = spans[:, 0].float() + rel[:, 0] * L
        e = spans[:, 1].float() + rel[:, 1] * L
        e = torch.maximum(e, s + 1e-3)
        return torch.stack([s, e], dim=1).to(spans.dtype)

    def train_losses(self, rel, spans, gts, valid, diou_fn):
        """rel (N, 2) predicted relative corrections of ONE item; spans
        (N, 2) decoded; gts (G, 2) seen GT; valid (N,) bool. Returns
        (l1_sum, diou_sum, n_matched, tiou_before_sum, tiou_after_sum)."""
        z = rel.new_zeros(())
        if gts is None or gts.numel() == 0 or not bool(valid.any()):
            return z, z, 0, 0.0, 0.0
        with torch.no_grad():
            tiou, _ = seg_tiou(spans.float(), gts.float())
            best, gi = tiou.max(dim=1)
            m = valid & (best >= 0.5)
        if not bool(m.any()):
            return z, z, 0, 0.0, 0.0
        sp = spans[m].float()
        g = gts[gi[m]].float()
        L = (sp[:, 1] - sp[:, 0]).clamp(min=1e-3)
        tgt = torch.stack([(g[:, 0] - sp[:, 0]) / L,
                           (g[:, 1] - sp[:, 1]) / L], dim=1)
        tgt = tgt.clamp(-self.rng, self.rng)
        r = rel[m]
        l1 = (r - tgt).abs().sum()
        c = 0.5 * (sp[:, 0] + sp[:, 1])
        cs = self.apply(sp, r)
        po = torch.stack([c - cs[:, 0], cs[:, 1] - c], dim=1).clamp(min=0)
        go = torch.stack([c - g[:, 0], g[:, 1] - c], dim=1).clamp(min=0)
        diou = diou_fn(po, go, reduction='sum')
        with torch.no_grad():
            t_after, _ = seg_tiou(cs, gts.float())
            t_after = t_after.gather(1, gi[m].unsqueeze(1)).squeeze(1)
        return (l1, diou, int(m.sum()), float(best[m].sum()),
                float(t_after.sum()))
