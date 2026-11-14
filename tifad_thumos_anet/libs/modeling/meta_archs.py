import math
import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns

from . import make_backbone, make_generator, make_neck

from .models import register_meta_arch
from .blocks import LayerNorm, MaskedConv1D, Scale
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss
from libs.utils.nms import batched_nms
# zeroshot
from .text import TextFeatures


def tfcade_ccr_prior(out_cls_logits, fpn_masks, updated_feat, updated_text_lv,
                     sigma=1.0, subclass_s=0):
    """TF-CADE (CVPR'26) ACA-style internal foreground video-prior, eval-only.

    Paper: "TF-CADE: Foreground-Concentrated Text-Video Alignment for
    Zero-Shot Temporal Action Detection" (Lee, Kim, Lee), Sec. 4.2-4.3.
    Computes the foreground-based video-level similarity S_fg (Eq. 3-7) and
    returns softmax(S_fg) -- the per-class video-level prior consumed by CCR
    (Eq. 8) inside inference_single_video.

    Args (all per-FPN-level lists, eval-time shapes):
        out_cls_logits : F x (B, T_l, C)  video-text confidence map. DOC-AMBIG:
            the paper's Eq. 2 P_cls is a raw v.c^T similarity; this codebase's
            logits are (feat^T @ text)/logit_scale + bias. We use the model's
            own logits as P_cls -- max/softmax over them is shift/scale
            tolerant and they are exactly what sigmoid() consumes in Eq. 8.
        fpn_masks      : F x (B, T_l)   bool validity masks (already squeezed).
        updated_feat   : F x (B, D, T_l) cross-modal-adapted video features
            (the paper's v^(l), Eq. 6).
        updated_text_lv: F x (B, C', D) projected per-level text embeddings
            (the paper's c_n^(l), Eq. 7). C' may be C*(1+S) under
            TIFAD_SUBCLASS_S; pass subclass_s to collapse group-wise.
        sigma          : Gaussian std in feature bins (paper Table 7d: sigma=1
            best on THUMOS 50/50; env TIFAD_CCR_SIGMA).
    Returns:
        (B, C) softmax prior over classes.
    """
    L = len(out_cls_logits)
    device = out_cls_logits[0].device
    dtype = out_cls_logits[0].dtype
    # fixed (non-learnable) Gaussian kernel. Radius 3*sigma covers >99.7% of mass.
    rad = max(1, int(round(3.0 * float(sigma))))
    xs = torch.arange(-rad, rad + 1, device=device, dtype=dtype)
    gk = torch.exp(-0.5 * (xs / float(sigma)) ** 2)
    gk = (gk / gk.sum()).view(1, 1, -1)

    s_fg = None
    for lg, msk, feat, text in zip(out_cls_logits, fpn_masks,
                                   updated_feat, updated_text_lv):
        vmask = msk.to(torch.bool)                       # (B, T_l)
        # Eq. 3: m_max = softmax_T( max_C(P_cls) ), invalid steps excluded
        peak = lg.max(dim=-1).values                     # (B, T_l)
        peak = peak.masked_fill(~vmask, float('-inf'))
        m_max = torch.softmax(peak, dim=-1)              # (B, T_l), 0 on pad
        # Eq. 4: m_filter = m_max (*) G(sigma). DOC-AMBIG: padding mode is not
        # specified in the paper; zero-padding is used, consistent with
        # treating outside-video/padded steps as zero certainty (m_max is
        # already exactly 0 there).
        m_filt = F.conv1d(m_max.unsqueeze(1), gk, padding=rad).squeeze(1)
        # Eq. 5 + "normalize along the temporal dimension by its sum"
        m = (m_max + m_filt) * vmask.to(dtype)
        m = m / m.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        # Eq. 6: v_fg = sum_t m_t * v_t
        v_fg = (feat * m.unsqueeze(1)).sum(dim=-1)       # (B, D)
        # Eq. 7 (per level): cosine similarity with each class text embedding
        v_n = F.normalize(v_fg, dim=-1)
        t_n = F.normalize(text, dim=-1)                  # (B, C', D)
        s_l = torch.bmm(t_n, v_n.unsqueeze(-1)).squeeze(-1)   # (B, C')
        if subclass_s > 0:
            # collapse C*(1+S) pseudo-class rows group-wise (mirror of the
            # TIFAD_SUBCLASS max-pool on the logits) so the prior matches C
            g = 1 + subclass_s
            s_l = s_l.view(s_l.size(0), s_l.size(1) // g, g).amax(-1)
        s_fg = s_l if s_fg is None else s_fg + s_l
    s_fg = s_fg / L                                      # Eq. 7: mean over levels
    # Eq. 8 prerequisite: softmax over classes = likelihood each class is
    # present in the video
    return torch.softmax(s_fg, dim=-1)                   # (B, C)


class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = []
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))

            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        # fpn_masks remains the same
        return out_logits


def _aspec_attr_family(name):
    """ASPEC attribute -> family: prefix families impl_/motion_/force_/env_,
    everything else (standalone values like has_runup) pooled into 'misc'."""
    _p = name.split('_', 1)[0]
    return _p if _p in ('impl', 'motion', 'force', 'env') else 'misc'


def _aspec_compose_sibsoft(term, lg, groups, sibsoft, parent_beta):
    """SIBSOFT / PARENT transform of the ASPEC composition term (EVAL ONLY;
    called from TiFAD.forward). `term` is the already-alpha-weighted
    composition (..., C); `lg` the PRE-composition class logits (..., C);
    `groups` the multi-member family column-index tensors.

    sibsoft: within each multi-member family subtract the family logsumexp of
    `term` (= within-family softmax normalization in log space), restricting
    the attribute discrimination to siblings; singleton-family classes keep
    the absolute term. parent_beta: additionally add beta * family_logit,
    family_logit = logsumexp of the family members' PRE-composition class
    logits (singleton family -> the class's own logit)."""
    if sibsoft:
        term = term.clone()
        for _mem in groups:
            term[..., _mem] = term[..., _mem] - torch.logsumexp(
                term[..., _mem], dim=-1, keepdim=True)
    if parent_beta != 0:
        fam_lg = lg.clone()
        for _mem in groups:
            fam_lg[..., _mem] = torch.logsumexp(
                lg[..., _mem], dim=-1, keepdim=True)
        term = term + parent_beta * fam_lg
    return term


def _level_compose_pool(logits, group_sizes):
    """FINEGYM_LEVEL_COMPOSE max-pool (EVAL ONLY; pairs with the class-list
    expansion in text.py). `logits` (..., sum(group_sizes)) over the expanded
    pseudo-classes [act0, act0_members..., act1, ...] -> (..., len(group_sizes))
    via amax within each VARIABLE-size group: an unseen activity scores by
    max(own-name logit, member-element logits) -- the element level acts as
    auxiliary support for the activity decision."""
    assert logits.size(-1) == sum(group_sizes), \
        f"[LVL] logit cols {logits.size(-1)} != sum(groups) {sum(group_sizes)}"
    outs, off = [], 0
    for sz in group_sizes:
        outs.append(logits[..., off:off + sz].amax(dim=-1))
        off += sz
    return torch.stack(outs, dim=-1)


def _pool_gt_segment_feats(feat0, gt_segments, gt_labels, n_chunks):
    """Mean-pool level-0 features inside each GT segment, split into
    `n_chunks` contiguous temporal chunks (n_chunks=3: phase-like thirds for
    VIDPHASE; n_chunks=1: whole-segment features for VIDACT). Segment
    coordinates are level-0 feature-grid units (same convention as the
    TaxGNN3 instance pooling above). Returns ((M, n_chunks, D), (M,) long
    labels) or (None, None) when the batch has no usable segments."""
    B, D, T0 = feat0.shape
    fs, ls = [], []
    for bi in range(B):
        segs, labs = gt_segments[bi], gt_labels[bi]
        if segs is None or segs.numel() == 0:
            continue
        for si in range(segs.size(0)):
            s0 = min(max(int(segs[si, 0].item()), 0), T0 - 1)
            e0 = min(max(int(math.ceil(segs[si, 1].item())), s0 + 1), T0)
            chunks = []
            for t in range(n_chunks):
                a = s0 + ((e0 - s0) * t) // n_chunks
                b = max(s0 + ((e0 - s0) * (t + 1)) // n_chunks, a + 1)
                b = min(b, T0)
                a = min(a, b - 1)
                chunks.append(feat0[bi, :, a:b].mean(dim=-1))
            fs.append(torch.stack(chunks, dim=0))
            ls.append(int(labs[si]))
    if not fs:
        return None, None
    return (torch.stack(fs, dim=0),
            torch.tensor(ls, dtype=torch.long, device=feat0.device))


def _compl_pool(feat0, spans):
    """COMPL structured pooling (SSN-style, see PtTransformer.__init__).
    feat0: (D, T) LEVEL-0 features for ONE item (already sliced to the valid
    length); spans: (N, 2) in level-0 feature-grid units. For each span
    [s, e) with L = e - s, mean-pool 5 regions -- pre [s-0.25L, s), the 3
    interior thirds of [s, e), post [e, e+0.25L) -- via an integral
    (cumsum) table. Regions falling entirely outside [0, T] (video edges,
    out-of-range decodes) pool to ZEROS: an explicit "no context" signal
    rather than borrowing in-span bins. Returns (N, 5*D) float32."""
    D, T = feat0.shape
    cs = torch.cat([feat0.new_zeros(D, 1).float(),
                    feat0.float().cumsum(dim=1)], dim=1)       # (D, T+1)

    def region(a, b):
        ai = a.floor().clamp(0, T).long()
        bi = b.ceil().clamp(0, T).long()
        n = (bi - ai).clamp(min=0)
        m = cs[:, bi] - cs[:, ai]                              # (D, N)
        m = m / n.clamp(min=1).to(m.dtype)
        return (m * (n > 0).to(m.dtype)).transpose(0, 1)       # (N, D)

    s, e = spans[:, 0], spans[:, 1]
    L = (e - s).clamp(min=1e-3)
    parts = [region(s - 0.25 * L, s)]
    for t in range(3):
        parts.append(region(s + L * (t / 3.0), s + L * ((t + 1) / 3.0)))
    parts.append(region(e, e + 0.25 * L))
    return torch.cat(parts, dim=1)


def _compl_mine(segs, t_valid, min_len=4.0):
    """COMPL train-time candidate mining (SSN-style, see
    PtTransformer.__init__). For each GT span [s, e) in LEVEL-0 grid units
    (length L >= min_len bins; shorter GTs are skipped -- their fragments
    would collapse to single bins):
      * 4 positives: both endpoints jittered by U(-0.1L, +0.1L) -> IoU with
        the GT >= 0.8 worst case ((L-0.2L)/L), comfortably above the 0.7
        spec floor even before the [0, t_valid] clamp (which only trims
        <= 0.1L per side).
      * 4 fragment negatives, fully INSIDE the GT (100% >= the 70% spec
        floor) with length r*L, r ~ U(0.25, 0.45) -> IoU = r < 0.5:
        start-anchored, end-anchored, and 2 uniformly-placed center crops.
      * 2 shifted negatives: same length L, offset d ~ U(0.55, 0.8)*L left
        and right -> IoU = (L-d)/(L+d) in [0.11, 0.29] (spec band 0.1-0.3).
        Emitted ONLY when fully inside [0, t_valid] (clamping would shrink
        the union and push the IoU past the band); tries the mirrored
        direction before dropping.
    Returns (cands (M, 2) float32, labels (M,) float32 1/0) on segs.device,
    or (None, None) when the item has no usable GT."""
    if segs is None or segs.numel() == 0:
        return None, None
    dev = segs.device
    cands, labels = [], []

    def U(a, b):
        return a + (b - a) * float(torch.rand((), device=dev))

    for si in range(segs.size(0)):
        s = max(0.0, float(segs[si, 0]))
        e = min(float(t_valid), float(segs[si, 1]))
        L = e - s
        if L < min_len:
            continue
        # 4 jittered positives (IoU >= 0.8 with the GT)
        for _ in range(4):
            a = max(0.0, s + U(-0.1, 0.1) * L)
            b = min(float(t_valid), e + U(-0.1, 0.1) * L)
            cands.append((a, b))
            labels.append(1.0)
        # 2 boundary-anchored fragments (inside, IoU = r < 0.5)
        r = U(0.25, 0.45)
        cands.append((s, s + r * L))
        labels.append(0.0)
        r = U(0.25, 0.45)
        cands.append((e - r * L, e))
        labels.append(0.0)
        # 2 center crops (uniform placement inside the GT)
        for _ in range(2):
            r = U(0.25, 0.45)
            h = 0.5 * r * L
            c = s + h + U(0.0, 1.0) * (L - 2.0 * h)
            cands.append((c - h, c + h))
            labels.append(0.0)
        # 2 shifted negatives (full length, IoU in [0.11, 0.29])
        for sign in (-1.0, 1.0):
            d = U(0.55, 0.8) * L
            for sg in (sign, -sign):        # mirrored fallback
                a, b = s + sg * d, e + sg * d
                if a >= 0.0 and b <= float(t_valid):
                    cands.append((a, b))
                    labels.append(0.0)
                    break
    if not cands:
        return None, None
    return (torch.tensor(cands, dtype=torch.float32, device=dev),
            torch.tensor(labels, dtype=torch.float32, device=dev))


def _vidphase_phase_bank(per_sent, n_phases=3):
    """V1 VIDPHASE text-side phase bank: per-sentence embeddings (C, K, D)
    from the h19 ensemble -> (C, n_phases, D) L2-normalized. Sentence 0 is
    the class prompt; sentences 1..K-1 are the temporally ORDERED phase
    descriptions, split into n_phases contiguous chunks (early/middle/late)
    and mean-pooled -- the text-side structure the video thirds align to."""
    C, K, D = per_sent.shape
    n = K - 1
    if n < n_phases:
        raise RuntimeError(
            f"[VIDPHASE] need >= {n_phases + 1} sentences/class, got {K}")
    base, rem = divmod(n, n_phases)
    banks, off = [], 1
    for t in range(n_phases):
        sz = base + (1 if t < rem else 0)
        banks.append(per_sent[:, off:off + sz].mean(dim=1))
        off += sz
    return F.normalize(torch.stack(banks, dim=1), dim=-1)


def _vidphase_infonce(third_feats, seg_labels, phase_bank, tau=0.1):
    """V1 VIDPHASE auxiliary InfoNCE (TRAIN ONLY): video thirds (M, P, D)
    of GT segments vs the (C, P, D) normalized phase-text bank. Query
    (segment m of class c, third t) has positive key (c, t); negatives are
    ALL other (class, phase) keys -- both cross-phase (within-class temporal
    structure) and cross-class negatives. Returns the MEAN cross-entropy
    over the M*P queries (scale-stable wrt batch segment count)."""
    M, P, D = third_feats.shape
    C = phase_bank.size(0)
    q = F.normalize(third_feats, dim=-1).reshape(M * P, D)
    k = phase_bank.reshape(C * P, D)
    logits = q @ k.t() / tau                                   # (M*P, C*P)
    tgt = (seg_labels.unsqueeze(1) * P
           + torch.arange(P, device=seg_labels.device)).reshape(-1)
    return F.cross_entropy(logits, tgt, reduction='mean')


class ASPECSplitHead(nn.Module):
    """Per-attribute-FAMILY specialist heads (TIFAD_ASPEC_SPLIT=1): one small
    2-layer conv head per attribute family (same PtTransformerClsHead
    template as the shared ASPEC head, output = that family's value count).
    forward() concatenates the family logits and reorders them back to the
    sorted-vocab attribute order, so ALL downstream ASPEC plumbing (train
    loss / eval composition) consumes the same (B, A, T_l) tensors as the
    shared head and is unchanged."""
    def __init__(self, attr_names, input_dim, feat_dim, kernel_size,
                 prior_prob, with_ln):
        super().__init__()
        _byfam = {}
        for _a in attr_names:               # attr_names is sorted vocab order
            _byfam.setdefault(_aspec_attr_family(_a), []).append(_a)
        self.family_names = sorted(_byfam)
        self.family_sizes = [len(_byfam[f]) for f in self.family_names]
        _concat_order = [a for f in self.family_names for a in _byfam[f]]
        # vocab_perm[i] = column of vocab attr i in the concatenated output
        self.register_buffer(
            'vocab_perm',
            torch.tensor([_concat_order.index(a) for a in attr_names],
                         dtype=torch.long),
            persistent=False)
        self.heads = nn.ModuleList(
            PtTransformerClsHead(
                input_dim, feat_dim, len(_byfam[f]),
                kernel_size=kernel_size,
                prior_prob=prior_prob,
                with_ln=with_ln,
                num_layers=2,
            ) for f in self.family_names)

    def forward(self, fpn_feats, fpn_masks):
        # H x F x (B, A_f, T_l)
        per_head = [h(fpn_feats, fpn_masks) for h in self.heads]
        # concat family blocks per level, then reorder to vocab order
        return tuple(
            torch.cat([ph[l] for ph in per_head], dim=1)
                 .index_select(1, self.vocab_perm)
            for l in range(len(fpn_feats)))


class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        dfl_ranges=None,
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # ---- DFL (TIFAD_DFL=<nbins>, R2 2026-09-17): BOUNDARY-AS-
        # DISTRIBUTION regression (GFL, arXiv 2006.04388; TriDet's
        # relative-bin boundary head). Each side's offset is a softmax
        # distribution over `nbins` linearly spaced bins on [0, R_l]
        # (R_l = min(level regression-range top, max_seq_len / stride_l),
        # stride-normalized units); the DECODED offset is the expectation,
        # so every consumer (DIoU reg loss, VSA / QH / inference decode)
        # is unchanged. The per-level Scale multiplies the bin LOGITS
        # (a learned temperature). Parameter budget: a kernel-1 projection
        # feat_dim -> 2*nbins (8.2k at nbins=16) replaces the kernel-3
        # feat_dim -> 2 conv (1.5k) on top of the shared ~393k conv stack,
        # i.e. +1.7 % regression parameters. Logits are stashed per
        # forward in `dfl_logits` (F x (B, T_l, 2, nbins)) for the DFL
        # loss in losses(). Byte-inert when unset (stock kernel-3 head).
        import os as _osdf
        self.dfl_bins = int(_osdf.environ.get('TIFAD_DFL', '0') or 0)
        self.dfl_logits = None
        if self.dfl_bins > 0:
            assert dfl_ranges is not None and len(dfl_ranges) == fpn_levels
            self.offset_head = MaskedConv1D(
                feat_dim, 2 * self.dfl_bins, 1, stride=1, padding=0)
            for l, R in enumerate(dfl_ranges):
                self.register_buffer(
                    f'dfl_bins_{l}',
                    torch.linspace(0.0, float(R), self.dfl_bins),
                    persistent=False)
            print(f"[DFL] active (distributional offsets: {self.dfl_bins} "
                  f"bins/side, ranges={[round(float(r), 1) for r in dfl_ranges]})",
                  flush=True)
        else:
            # segment regression
            self.offset_head = MaskedConv1D(
                    feat_dim, 2, kernel_size,
                    stride=1, padding=kernel_size//2
                )

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        dfl_logits = []
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
                
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            # cur_offsets = torch.clamp(cur_offsets, min=-50, max=50)   # 추가
            if self.dfl_bins > 0:
                B_, _, T_ = cur_offsets.shape
                lg = self.scale[l](cur_offsets).reshape(
                    B_, 2, self.dfl_bins, T_)                     # (B, 2, n, T)
                bins = getattr(self, f'dfl_bins_{l}')
                prob = torch.softmax(lg, dim=2)
                out_offsets += ((prob * bins.view(1, 1, -1, 1)).sum(2), )
                dfl_logits.append(lg.permute(0, 3, 1, 2))         # (B, T, 2, n)
                continue
            out_offsets += (F.softplus(self.scale[l](cur_offsets)), )
            # out_offsets += (F.relu(self.scale[l](cur_offsets)), ) 
            
        self.dfl_logits = dfl_logits if self.dfl_bins > 0 else None

        # fpn_masks remains the same
        return out_offsets


class LevelAxisScaleAttn(nn.Module):
    """LEVELAXIS (TIFAD_LEVELAXIS=1): Dynamic-Head-style scale attention
    across the FPN levels (Dai et al., CVPR'21 "Dynamic Head", pi_L),
    adapted to 1-D. The heads consume each pyramid level INDEPENDENTLY;
    this block is the cheapest falsifier of the cross-level hypothesis:
    every level's features are nearest-upsampled to the level-0 grid,
    stacked into L tokens per (batch, timestep), passed through ONE
    multi-head attention over the L axis (B*T0 folded into batch, so the
    attention is over just L=6 tokens), then re-downsampled to each
    level's own grid (strided slicing by scale_factor**l -- exactly how
    the pyramid was built, T_l == ceil(T0 / s**l); nearest interpolation
    fallback if a backbone variant ever floors instead) and added back
    as a per-level residual.

    Init-equality invariant: the attention out_proj AND the learned level
    embeddings are ZERO-INIT, so the residual is exactly 0 at step 0
    (byte-identical outputs; keep this if you modify the block).

    Mask handling: the attention mixes levels AT one timestep, so padding
    is inherited from the per-level masks -- deltas are zeroed wherever
    fpn_masks[l] is False before the residual add."""

    def __init__(self, dim, num_levels, n_head=4, scale_factor=2):
        super().__init__()
        self.num_levels = num_levels
        self.scale_factor = scale_factor
        self.lvl_emb = nn.Embedding(num_levels, dim)
        nn.init.zeros_(self.lvl_emb.weight)
        self.attn = nn.MultiheadAttention(dim, n_head, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, feats, fpn_masks):
        # feats: F x (B, C, T_l); fpn_masks: F x (B, 1, T_l)
        L = len(feats)
        assert L == self.num_levels, \
            f"[LEVELAXIS] built for {self.num_levels} levels, got {L}"
        B, C, T0 = feats[0].shape
        # (a) nearest upsample every level to the level-0 grid
        up = [feats[0]] + [F.interpolate(f, size=T0, mode='nearest')
                           for f in feats[1:]]
        x = torch.stack(up, dim=-1)                      # (B, C, T0, L)
        x = x.permute(0, 2, 3, 1).reshape(B * T0, L, C)  # (B*T0, L, C)
        # (b) learned level embeddings (zero-init) + attention over the
        # L axis; residual r == 0 at init (zero-init out_proj)
        x = x + self.lvl_emb.weight.unsqueeze(0)
        r, _ = self.attn(x, x, x, need_weights=False)    # (B*T0, L, C)
        r = r.reshape(B, T0, L, C).permute(0, 3, 1, 2)   # (B, C, T0, L)
        # (c) re-downsample each level's delta to its own grid + residual
        out = []
        for l, (f, m) in enumerate(zip(feats, fpn_masks)):
            T_l = f.size(-1)
            d = r[..., l]                                # (B, C, T0)
            stride = self.scale_factor ** l
            ds = d[..., ::stride]
            if ds.size(-1) != T_l:
                # defensive fallback (floor-length pyramid variants)
                ds = F.interpolate(d, size=T_l, mode='nearest')
            out.append(f + ds * m.to(ds.dtype))
        return out


class TopDownSemNeck(nn.Module):
    """TDNECK (TIFAD_TDNECK=1): CLTDR-style top-down semantic fusion with
    task decoupling. Top-down pass l = L-2..0:
        sem[L-1] = feat[L-1]
        sem[l]   = feat[l] + gate_l(upsample(sem[l+1]))
    where each level has its OWN 1x1 Conv1d gate (num_levels-1 convs),
    ZERO-INIT (weight AND bias) => sem == feat exactly at step 0
    (init-equality invariant; keep it if you modify the block). The sem
    pyramid feeds ONLY the classification side (text matmul + the
    score-space logit blocks); the regression / foreground / centerness
    heads keep the ORIGINAL features (task decoupling).

    Mask handling: the gated top-down delta is zeroed wherever
    fpn_masks[l] is False before the residual add."""

    def __init__(self, dim, num_levels, content_gate=False):
        super().__init__()
        self.num_levels = num_levels
        self.gates = nn.ModuleList(
            [nn.Conv1d(dim, dim, 1) for _ in range(num_levels - 1)])
        for g in self.gates:
            nn.init.zeros_(g.weight)
            nn.init.zeros_(g.bias)
        # TIFAD_TDNECK_GATE=1 (tdneck_v2): per-timestep scalar gate
        # conditioned on [fine feature; upsampled coarse], so injection can
        # be DECLINED where fine features are already discriminative
        # (H-indiscriminate-gate fix; s1-collapse diagnosis 2026-09-07).
        # The 1x1 conv path stays zero-init, so init-equality holds for
        # any gate value; the gate conv itself is zero-init => sigmoid=0.5
        # everywhere at step 0 (a constant, absorbed by the zero conv).
        self.content_gate = content_gate
        if content_gate:
            self.gate_cond = nn.ModuleList(
                [nn.Conv1d(2 * dim, 1, 1) for _ in range(num_levels - 1)])
            for g in self.gate_cond:
                nn.init.zeros_(g.weight)
                nn.init.zeros_(g.bias)

    def forward(self, feats, fpn_masks):
        # feats: F x (B, C, T_l); fpn_masks: F x (B, 1, T_l)
        L = len(feats)
        assert L == self.num_levels, \
            f"[TDNECK] built for {self.num_levels} levels, got {L}"
        sem = [None] * L
        sem[L - 1] = feats[L - 1]
        for l in range(L - 2, -1, -1):
            up = F.interpolate(sem[l + 1], size=feats[l].size(-1),
                               mode='nearest')
            delta = self.gates[l](up)
            if self.content_gate:
                w = torch.sigmoid(
                    self.gate_cond[l](torch.cat([feats[l], up], dim=1)))
                delta = delta * w
            sem[l] = feats[l] + delta * fpn_masks[l].to(up.dtype)
        return sem


class SemCtxAttn(nn.Module):
    """CTXATTN (TIFAD_CTXATTN=1): hierarchical ACTIVITY-CONTEXT attention.
    The hierarchy is SEMANTIC (activity families -> action classes), not
    temporal: every snippet (query) cross-attends a token bank of
    (a) ACTIVITY tokens -- cached no_grad CLIP embeddings of
        "a video of <family> activity" (one per fam-map family), and
    (b) ACTION tokens -- the CURRENT class list's text embeddings
    (banks built/cached by the caller; CONFGATE caching precedent), both
    pushed into the trunk feature space by ONE shared trainable
    Linear(text_dim -> C) and tagged with a learned 2-level embedding
    (activity vs action). Complementary to LEVELAXIS: that block mixes
    across TEMPORAL pyramid levels, this one mixes across semantic
    CONTEXT levels.

    Init-equality invariant: the attention out_proj is ZERO-INIT, so the
    residual is exactly 0 at step 0 (byte-identical outputs; keep this if
    you modify the block). The bank projection / in_proj keep normal init
    -- they only feed K/V behind the zero out_proj. The LEVEL embedding is
    ZERO-INIT .

    Mask handling: queries are per-timestep; deltas are zeroed wherever
    fpn_masks[l] is False before the residual add (keys are text tokens,
    never padded)."""

    def __init__(self, text_dim, dim, n_head=4):
        super().__init__()
        # text pathway output dim == embd_dim here (text.py replaces CLIP's
        # text_projection with Linear(text_embed_dim -> embd_dim), and
        # get_text_features applies it), asserted again at forward time.
        self.proj = nn.Linear(text_dim, dim)
        self.lvl_emb = nn.Embedding(2, dim)
        nn.init.zeros_(self.lvl_emb.weight)
        self.attn = nn.MultiheadAttention(dim, n_head, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, feats, fpn_masks, bank, n_activity):
        # feats: F x (B, C, T_l); bank: (N, text_dim) raw cached text embs
        # (first n_activity rows = activity tokens, rest = action tokens)
        assert bank.size(-1) == self.proj.in_features, (
            f"[CTXATTN] bank dim {bank.size(-1)} != expected text dim "
            f"{self.proj.in_features}")
        kv = self.proj(bank)                              # (N, C)
        n_act = bank.size(0) - n_activity
        lvl = torch.cat(
            [self.lvl_emb.weight[0].unsqueeze(0).expand(n_activity, -1),
             self.lvl_emb.weight[1].unsqueeze(0).expand(n_act, -1)], dim=0)
        kv = (kv + lvl).unsqueeze(0)                      # (1, N, C)
        out = []
        for f, m in zip(feats, fpn_masks):
            q = f.transpose(1, 2)                         # (B, T_l, C)
            k = kv.expand(q.size(0), -1, -1)
            r, _ = self.attn(q, k, k, need_weights=False)
            # masked residual; exactly 0 at init (zero-init out_proj)
            q = q + r * m.transpose(1, 2).to(r.dtype)
            out.append(q.transpose(1, 2))
        return out


class TemporalSlotAttn(nn.Module):
    """VIDSLOT (TIFAD_VIDSLOT=<K>): temporal slot discovery. K learned slot
    queries gather MULTI-SECOND video context from the level-0 features
    (slots attend feats: MHA-1, key_padding_mask hides padding), pass one
    small self-interaction (residual Linear+ReLU), then EVERY pyramid
    level's features attend BACK to the slots (MHA-2) and receive a
    residual. Detection-loss-only -- no clustering / contrastive aux
    (house law). Independent of and composable with
    CTXATTN / LEVELAXIS / TDNECK.

    Init-equality invariant: MHA-2's out_proj is ZERO-INIT, so the return
    path residual is exactly 0 at step 0 (byte-identical outputs; keep
    this if you modify the block). Slots (std-0.02), MHA-1 and the
    self-interaction Linear keep normal init -- they only feed the second
    stage behind the zero out_proj.

    Mask handling: MHA-1 masks padded level-0 keys via key_padding_mask;
    MHA-2 deltas are zeroed wherever fpn_masks[l] is False before the
    residual add (assumes >= 1 valid timestep per video, which
    preprocessing() guarantees)."""

    def __init__(self, dim, num_slots, n_head=4):
        super().__init__()
        self.num_slots = num_slots
        self.slots = nn.Parameter(torch.empty(num_slots, dim))
        nn.init.normal_(self.slots, std=0.02)
        self.gather = nn.MultiheadAttention(dim, n_head, batch_first=True)
        self.slot_ff = nn.Linear(dim, dim)
        self.scatter = nn.MultiheadAttention(dim, n_head, batch_first=True)
        nn.init.zeros_(self.scatter.out_proj.weight)
        nn.init.zeros_(self.scatter.out_proj.bias)

    def forward(self, feats, fpn_masks):
        # feats: F x (B, C, T_l); fpn_masks: F x (B, 1, T_l)
        B = feats[0].size(0)
        f0 = feats[0].transpose(1, 2)                     # (B, T0, C)
        kpm = ~fpn_masks[0].squeeze(1)                    # (B, T0) True=pad
        s = self.slots.unsqueeze(0).expand(B, -1, -1)     # (B, K, C)
        g, _ = self.gather(s, f0, f0, key_padding_mask=kpm,
                           need_weights=False)
        s = s + g
        s = s + F.relu(self.slot_ff(s))                   # self-interaction
        out = []
        for f, m in zip(feats, fpn_masks):
            q = f.transpose(1, 2)                         # (B, T_l, C)
            kv = s
            r, _ = self.scatter(q, kv, kv, need_weights=False)
            # masked residual; exactly 0 at init (zero-init out_proj)
            q = q + r * m.transpose(1, 2).to(r.dtype)
            out.append(q.transpose(1, 2))
        return out


@register_meta_arch("TiFAD")
class TiFAD(nn.Module):
    """
        Transformer based model for single stage action localization
    """
    def __init__(
        self,
        split_num,             # file number of text data split  (0~9)
        subset_file,           # data split path
        data_split,            # 50 for 50%-50% setting and 75 for 75%-25% setting
        fpn_type,              # a string defines which fpn we use
        backbone_arch,         # a tuple defines # layers in embed / stem / branch
        scale_factor,          # scale factor between branch layers
        input_dim,             # input feat dim
        max_seq_len,           # max sequence length (used for training)
        max_buffer_len_factor, # max buffer size (defined a factor of max_seq_len)
        n_head,                # number of heads for self-attention in transformer
        n_mha_win_size,        # window size for self attention; -1 to use full seq
        embd_kernel_size,      # kernel size of the embedding network
        embd_dim,              # output feat channel of the embedding network
        embd_with_ln,          # attach layernorm to embedding network
        fpn_dim,               # feature dim on FPN
        fpn_with_ln,           # if to apply layer norm at the end of fpn
        fpn_start_level,       # start level of fpn
        head_dim,              # feature dim for head
        regression_range,      # regression range on each level of FPN
        head_num_layers,       # number of layers in the head (including the classifier)
        head_kernel_size,      # kernel size for reg/cls heads
        head_with_ln,          # attache layernorm to reg/cls heads
        use_abs_pe,            # if to use abs position encoding
        use_rel_pe,            # if to use rel position encoding
        text_prior_prob,       # text prior probability
        topk_ratio,            # topk ratio of salient attentive mask 
        threshold,             # threshold of salient attentive mask 
        text_model,            # text model (e.g., CLIP)
        train_cfg,             # other cfg for training
        test_cfg,              # other cfg for testing
    ):
        super().__init__()

        # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor**i for i in range(
            fpn_start_level, backbone_arch[-1]+1
        )]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)

        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size]*(1 + backbone_arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-1])
            self.mha_win_size = n_mha_win_size
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']

        # test time config
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']
        
        # backbone
        self.backbone = make_backbone(
                'convTransformer',
                **{
                    'n_in' : input_dim,
                    'n_embd' : embd_dim, 
                    'n_head': n_head,    
                    'n_embd_ks': embd_kernel_size,
                    'max_len': max_seq_len,
                    'arch' : backbone_arch,    
                    'mha_win_size': self.mha_win_size,
                    'scale_factor' : scale_factor,
                    'with_ln' : embd_with_ln,          
                    'attn_pdrop' : 0.0,
                    'proj_pdrop' : self.train_dropout,
                    'path_pdrop' : self.train_droppath,
                    'use_abs_pe' : use_abs_pe,
                    'use_rel_pe' : use_rel_pe,
                    'topk_ratio' : topk_ratio,
                    'threshold' : threshold,
                }
            )
        
        # fpn network: convs
        # assert fpn_type in ['fpn', 'identity']
        # self.neck = make_neck(
        #     fpn_type,
        #     **{
        #         'in_channels' : [embd_dim] * (backbone_arch[-1] + 1),
        #         'out_channel' : fpn_dim,
        #         'scale_factor' : scale_factor,
        #         'start_level' : fpn_start_level,
        #         'with_ln' : fpn_with_ln
        #     }
        # )

        # location generator: points
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len' : max_seq_len * max_buffer_len_factor,
                'fpn_strides' : self.fpn_strides,
                'regression_range' : self.reg_range
            }
        )

        # regerssion head
        import os as _osdf_w
        self.reg_head = PtTransformerRegHead(
            fpn_dim, head_dim, len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln,
            # DFL (TIFAD_DFL): per-level bin ranges in stride-normalized
            # units -- the regression-range top capped by the level length
            dfl_ranges=[min(float(r[1]), float(self.max_seq_len) / s)
                        for r, s in zip(self.reg_range, self.fpn_strides)],
        )
        self.dfl_w = float(_osdf_w.environ.get('TIFAD_DFL_W', '0.25') or 0)

        # foreground head
        self.fg_head = PtTransformerClsHead(
            fpn_dim, head_dim, 1,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob, 
            with_ln=head_with_ln,
            num_layers=head_num_layers,   
            empty_cls=train_cfg['head_empty_cls']
        )

        # centerness head
        self.cn_head = PtTransformerClsHead(
            fpn_dim, head_dim, 1,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob, 
            with_ln=head_with_ln,
            num_layers=head_num_layers,   
            empty_cls=train_cfg['head_empty_cls']
        )

        # ---- IOUQ (TIFAD_IOUQ=1): IoU-aware quality scoring. Retrains the
        # centerness branch (cn_head, unchanged architecture -- no new
        # params) so its POSITIVE-point target is the actual tIoU between
        # the segment the regressor DECODES at that point and the matched
        # GT segment, instead of the point-centerness gaussian computed
        # from GT offsets alone. Motivation (measured): for ~30% of covered
        # GTs a same-class near-miss fragment outranks the correct-extent
        # detection 3-5x because the decode score
        # sqrt(sigmoid(cls)*sigmoid(cn))*sigmoid(fg) never evaluates the
        # quality of the DECODED extent. Targets-only change; eval decode
        # plumbing untouched. Inert (byte-identical) when unset.
        # TIFAD_IOUQ_MINLVL=<int>: apply the IoU targets ONLY to positive
        # points on FPN levels >= MINLVL (long actions decode from coarse
        # levels; measured s1 collapse showed short actions on fine levels
        # get harshly low IoU targets from small boundary errors). Positives
        # on finer levels keep the original gaussian centerness targets.
        # Default 0 == all levels (plain IOUQ).
        import os as _osiq
        self.iouq_on = _osiq.environ.get('TIFAD_IOUQ', '') == '1'
        self.iouq_minlvl = int(
            _osiq.environ.get('TIFAD_IOUQ_MINLVL', '0') or 0)
        if self.iouq_on:
            _sfx = (f", levels >= {self.iouq_minlvl}"
                    if self.iouq_minlvl > 0 else "")
            print("[IOUQ] active (centerness -> decoded-segment IoU targets"
                  f"{_sfx})", flush=True)

        # ---- COMPL (TIFAD_COMPL=1): SSN-style TRAINED completeness head
        # (Zhao et al., ICCV'17 Structured Segment Networks). Motivation
        # (measured, fragment root-cause analysis): fragments covering an
        # action's climax outrank the correct full extent 3-5x; the
        # centerness slot is dead (sigmoid ~0.5 always); the eval-time
        # text-curve completeness (TIFAD_SUPPORT) failed at 10 splits.
        # Mechanism: a SEPARATE MLP scores completeness from the VISUAL
        # level-0 features of {pre-span, 3 in-span thirds, post-span}
        # (_compl_pool: 5*fpn_dim -> 256 -> 1), trained with BCE on
        # candidates mined around every GT (_compl_mine: 4 jittered
        # positives, 4 inside-fragment + up to 2 shifted negatives) on
        # DETACHED cls_feat -- the head must not reshape the trunk
        # (gradients reach ONLY compl_head; interference-free params).
        # Eval: every decoded candidate's score is multiplied by
        # (sigmoid(compl) / per-video-median)^beta pre-soft-NMS.
        # INIT-EQUALITY (documented choice): the final Linear is ZERO-INIT
        # -> logit exactly 0 -> sigmoid exactly 0.5 -> ratio-to-median
        # exactly 1.0 -> the multiplier is exactly 1 at init. Chosen over
        # the raw sigmoid^beta variant (constant 0.5^beta at init), which
        # is only RANK-neutral and would still scale scores across the
        # fixed test_min_score / pre_nms thresholds. Envs: TIFAD_COMPL=1,
        # TIFAD_COMPL_W (train BCE weight, default 0.5), TIFAD_COMPL_BETA
        # (eval blend exponent, default 0.5). Byte-inert when unset.
        import os as _oscp
        self.compl_head = None
        if _oscp.environ.get('TIFAD_COMPL', '') == '1':
            self.compl_w = float(
                _oscp.environ.get('TIFAD_COMPL_W', '0.5') or 0.5)
            self.compl_beta = float(
                _oscp.environ.get('TIFAD_COMPL_BETA', '0.5') or 0.5)
            self.compl_head = nn.Sequential(
                nn.Linear(5 * fpn_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
            )
            torch.nn.init.zeros_(self.compl_head[2].weight)
            torch.nn.init.zeros_(self.compl_head[2].bias)
            print(f"[COMPL] active (SSN completeness head, "
                  f"W={self.compl_w}, beta={self.compl_beta})", flush=True)

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

        # text setting
        self.text_model = text_model
        self.text_proj = nn.Linear(embd_dim, embd_dim)
        self.bias_lang = nn.Parameter(torch.zeros(embd_dim), requires_grad=True)
        bias_value = -math.log((1 - text_prior_prob) / text_prior_prob)
        self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)
        self.logit_scale = nn.Parameter(torch.zeros([]))
        self.subset_file = subset_file
        self.data_split = data_split

        self.text_features = TextFeatures(
            model_path=self.text_model,
            subset_file=self.subset_file,
            data_split=self.data_split,
            emb_dim=embd_dim,
            split_num=split_num,
        )

        # ---- DT ("Decompose and Transfer", Zhu et al. CVPR'26): CSD phase
        # texts + TIF foreground filtering + APA phase-wise alignment.
        # Env: TIFAD_DT=1, TIFAD_DT_PHASES=<json {class: {start,middle,end,
        # global}}>. Inert (byte-identical) when unset -- dt_modules is not
        # even imported.
        import os as _osdt
        self.dt_module = None
        if _osdt.environ.get('TIFAD_DT', '') == '1':
            from .dt_modules import DTPhaseDecomposeAlign
            self.dt_module = DTPhaseDecomposeAlign(
                emb_dim=embd_dim,
                phases_file=_osdt.environ['TIFAD_DT_PHASES'],
            )
            print(f"[DT] PDA active: phases json = "
                  f"{_osdt.environ['TIFAD_DT_PHASES']}", flush=True)

        # ---- PDA (TIFAD_PDA=1; arXiv 2603.24030 "Decompose and Transfer",
        # faithful re-implementation, independent of the older TIFAD_DT
        # attempt above -- mutually exclusive). Video side: TIF phase-wise
        # foreground filtering + APA cross-attention alignment + adaptive
        # phase weighting (supplementary weighting network) + Eq. 9
        # localisation MLP, applied per FPN level with SHARED weights (see
        # libs/modeling/pda_modules.py). The CSD text side (four Phi_txt^p)
        # lives on self.text_features.pda_text. TIFAD_PDA_RESID=1 adds the
        # residual F_v^p + attn in Eq. 4 (the paper writes none). Inert
        # (byte-identical) when unset -- pda_modules is not even imported.
        import os as _ospda
        self.pda = None
        if _ospda.environ.get('TIFAD_PDA', '') == '1':
            assert self.dt_module is None, "[PDA] mutually exclusive with TIFAD_DT"
            from .pda_modules import PDAPhaseAlign
            _pda_resid = _ospda.environ.get('TIFAD_PDA_RESID', '') == '1'
            self.pda = PDAPhaseAlign(embd_dim, resid=_pda_resid)
            print(f"[PDA] TIF+APA+loc active (D={embd_dim}, Eq.4 residual="
                  f"{_pda_resid}, weighting net: 4 tokens/4-head MHSA/"
                  f"MLP-1024 softmax, loc MLP 4D->1024->1024->D, shared over "
                  f"FPN levels; cls loss = fork's sigmoid focal)", flush=True)

        # ---- HTCA (Hierarchical Text-Concept Attention): snippet features
        # cross-attend over the trained hierarchy anchor banks that live on
        # self.text_features (built there when TIFAD_HTCA=1 / TIFAD_HAF set;
        # see text.py). ZERO-INIT out_proj => the residual is exactly 0 at
        # step 0 (init-equality invariant -- keep it if you modify this).
        # Constructed HERE (not lazily) so make_optimizer sees it, mirroring
        # TIFAD_DT above. Env: TIFAD_HTCA=1, plus TIFAD_HTCA_ATTRS /
        # TIFAD_HTCA_FAMS jsons. Inert (byte-identical) when unset.
        self.htca_attn = None
        if _osdt.environ.get('TIFAD_HTCA', '') == '1':
            assert self.text_features.htca_attr_emb is not None, \
                "[HTCA] banks missing -- set TIFAD_HTCA_ATTRS/TIFAD_HTCA_FAMS"
            self.htca_attn = nn.MultiheadAttention(embd_dim, 4,
                                                   batch_first=True)
            nn.init.zeros_(self.htca_attn.out_proj.weight)
            nn.init.zeros_(self.htca_attn.out_proj.bias)
            print("[HTCA] zero-init snippet->hierarchy-anchor cross-attention "
                  "active", flush=True)

        # ---- HAF (Hierarchy-Anchor Foreground): auxiliary TRAIN-TIME
        # foreground loss from snippet<->anchor cosine similarities (shares
        # the banks above). Env: TIFAD_HAF=<float weight> (+ same jsons);
        # TIFAD_HAF_EVAL=<float> optionally blends the logit into out_fg at
        # eval (default 0 = train-only, output-identical at eval).
        self.haf_w = float(_osdt.environ.get('TIFAD_HAF', '0') or 0)
        if self.haf_w != 0:
            print(f"[HAF] aux hierarchy-anchor foreground loss active, "
                  f"weight={self.haf_w}", flush=True)

        # ---- ASPEC (attribute specialists on the VIDEO side): a shared
        # 2-layer MaskedConv1D head predicts per-snippet ATTRIBUTE logits on
        # every FPN level of updated_feat. Train: multi-label sigmoid focal
        # loss against the GT class's attribute set (losses()). Eval: unseen
        # class scores can be COMPOSED from the attribute logits
        # (TIFAD_ASPEC_ALPHA, forward()). Envs:
        #   TIFAD_ASPEC=<train loss weight>
        #   TIFAD_ASPEC_ATTRS=<json {class: [attr value names]}> (gates the
        #     head construction so ckpts round-trip strict=True)
        #   TIFAD_ASPEC_ALPHA=<eval compose weight, default 0>
        # Inert (byte-identical) when TIFAD_ASPEC_ATTRS is unset.
        self.aspec_head = None
        self.aspec_w = float(_osdt.environ.get('TIFAD_ASPEC', '0') or 0)
        _aspec_attrs = _osdt.environ.get('TIFAD_ASPEC_ATTRS', '')
        if _aspec_attrs:
            import json as _ajson
            self.aspec_class2attrs = _ajson.load(open(_aspec_attrs))
            self.aspec_attr_names = sorted(
                {a for v in self.aspec_class2attrs.values() for a in v})
            if _osdt.environ.get('TIFAD_ASPEC_SPLIT', '') == '1':
                # ---- ASPEC_SPLIT (TIFAD_ASPEC_SPLIT=1): one small 2-layer
                # head PER ATTRIBUTE FAMILY (impl/motion/force/env prefixes +
                # 'misc' for standalone values) instead of the single shared
                # head. Loss / eval composition plumbing unchanged -- the
                # module concatenates back to the vocab-order attr vector.
                # NOTE: must also be set at EVAL time so split-trained ckpts
                # round-trip strict=True.
                self.aspec_head = ASPECSplitHead(
                    self.aspec_attr_names, fpn_dim, head_dim,
                    kernel_size=head_kernel_size,
                    prior_prob=self.train_cls_prior_prob,
                    with_ln=head_with_ln)
                print(f"[ASPEC-SPLIT] {len(self.aspec_head.family_names)} "
                      f"family heads "
                      f"{dict(zip(self.aspec_head.family_names, self.aspec_head.family_sizes))}"
                      f" over {len(self.aspec_attr_names)} attributes",
                      flush=True)
            else:
                # shared 2-layer head: 1 hidden MaskedConv1D + attr
                # classifier, same template as the fg/cn heads
                # (PtTransformerClsHead)
                self.aspec_head = PtTransformerClsHead(
                    fpn_dim, head_dim, len(self.aspec_attr_names),
                    kernel_size=head_kernel_size,
                    prior_prob=self.train_cls_prior_prob,
                    with_ln=head_with_ln,
                    num_layers=2,
                )
                print(f"[ASPEC] {len(self.aspec_attr_names)} attribute "
                      f"specialists active", flush=True)

        # ---- FAMCE_UW (Kendall & Gal uncertainty weighting over {main,
        # famce}): two learned log-variances s = log sigma^2, combined in
        # losses() as exp(-s0)*main + s0/2 + exp(-s1)*famce + s1/2. Init 0
        # => weights (1, 1) == the plain famce sum at step 0. Registered
        # here so make_optimizer sees it (FAMTIE lesson). Env:
        # TIFAD_FAMCE_UW=1 (pairs with TIFAD_FAMCE). Inert when unset.
        self.famce_uw_logvar = None
        if _osdt.environ.get('TIFAD_FAMCE_UW', '') == '1':
            self.famce_uw_logvar = nn.Parameter(torch.zeros(2))
            print("[FAMCE-UW] Kendall uncertainty weighting over "
                  "{main, famce} active (init s=0 -> weights 1,1)",
                  flush=True)

        # ---- FAMCE_UW3 (TIFAD_FAMCE_UW3=1): extend the Kendall weighting
        # to a THIRD task {main, famce, vattr}: logvar becomes a 3-vector
        # (third entry init 0 => weight 1), and in losses() the vattr aux
        # enters as exp(-s2)*vattr_loss + s2/2 INSTEAD of the fixed direct
        # add (vattr_w stays as the inner scale inside vattr_loss).
        # Requires BOTH parents: TIFAD_FAMCE_UW=1 and TIFAD_VATTR=1.
        # Byte-inert when unset (2-vector logvar untouched).
        self.famce_uw3 = False
        if _osdt.environ.get('TIFAD_FAMCE_UW3', '') == '1':
            assert self.famce_uw_logvar is not None, \
                "[FAMCE-UW3] requires TIFAD_FAMCE_UW=1"
            assert _osdt.environ.get('TIFAD_VATTR', '') == '1', \
                "[FAMCE-UW3] requires TIFAD_VATTR=1"
            self.famce_uw_logvar = nn.Parameter(torch.zeros(3))
            self.famce_uw3 = True
            print("[FAMCE-UW3] Kendall over {main, famce, vattr} "
                  "(3-vector logvar, init s=0 -> weights 1,1,1)",
                  flush=True)

        # ---- LEAFLOSS (TIFAD_LEAFLOSS=1, TRAIN ONLY): leaf-level (temporal
        # PHASE) supervision on the SAME detection trunk -- the mirror image
        # of FAMCE below the detected action. For every positive detector
        # location assigned to GT segment j of class c, the 10 ordered
        # phase-sentence embeddings of c (text.py stash _leaf_phase_embs,
        # (C, 10, D)) are scored EXACTLY like the class embeddings in the
        # base cls matmul (L2-normalize, text_proj(x/2), feat . e /
        # logit_scale.exp(); bias omitted -- per-class constant across the
        # 10 phases) and trained with softmax CE against the pseudo phase
        # index k in 0..9 whose Stage-A interval contains the location's
        # time (dataset: item['leaf_cuts']; forward: _leaf_targets). Sum
        # over labeled positives / loss_normalizer (same as cls_loss),
        # weight TIFAD_LEAFLOSS_W (default 0.3, lambda_act precedent).
        # Combination: with TIFAD_FAMCE_UW=1 the Kendall logvar is EXTENDED
        # to 3 scalars {main, famce, leaf} (exp(-s2)*w*L_leaf + s2/2),
        # else the weighted loss is added directly. The 3-vector is created
        # ONLY under TIFAD_LEAFLOSS=1 so existing checkpoints load
        # unchanged. Mutually exclusive with TIFAD_FAMCE_UW3 (slot 2 taken
        # by vattr). Byte-inert when unset.
        self.leafloss = _osdt.environ.get('TIFAD_LEAFLOSS', '') == '1'
        self.leafloss_w = 0.0
        if self.leafloss:
            assert self.text_features.ensemble_prompts is not None, \
                "[LEAFLOSS] requires TIFAD_PROMPT_ENSEMBLE (phase sentences)"
            assert not getattr(self, 'famce_uw3', False), \
                "[LEAFLOSS] incompatible with TIFAD_FAMCE_UW3 (logvar slot 2)"
            self.leafloss_k = len(next(iter(
                self.text_features.ensemble_prompts.values()))) - 1
            assert self.leafloss_k == 10, \
                f"[LEAFLOSS] ensemble needs 1 class + 10 phase sentences " \
                f"(got K-1={self.leafloss_k})"
            self.leafloss_w = float(
                _osdt.environ.get('TIFAD_LEAFLOSS_W', '0.3') or 0.3)
            self._leaf_assign = []
            self._leaf_dbg_left = int(_osdt.environ.get(
                'TIFAD_LEAFLOSS_DEBUG', '0') or 0)
            # TIFAD_LEAFLOSS_HEAD=1: predict the phase index with a small
            # head on the trunk feature (class-agnostic, 10-way) instead of
            # the cosine to the phase-sentence embeddings (which are nearly
            # collinear in CLIP space and stay at chance accuracy).
            self.leaf_head = None
            if _osdt.environ.get('TIFAD_LEAFLOSS_HEAD', '') == '1':
                # fork the RNG so creating the head does not shift the
                # initialisation of every later module (keeps the detection
                # trajectory identical to the reference when the head is
                # detached)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(20260917)
                    self.leaf_head = nn.Sequential(
                        nn.Linear(embd_dim, embd_dim), nn.ReLU(inplace=True),
                        nn.Linear(embd_dim, 10))
                print("[LEAFLOSS] trunk phase head active (MLP D->D->10)",
                      flush=True)
            if self.famce_uw_logvar is not None:
                self.famce_uw_logvar = nn.Parameter(torch.zeros(3))
                print("[LEAFLOSS] phase-level CE on the trunk active, "
                      f"w={self.leafloss_w}, K=10; Kendall logvar extended "
                      "to {main, famce, leaf} (init s=0 -> weights 1,1,1)",
                      flush=True)
            else:
                print("[LEAFLOSS] phase-level CE on the trunk active, "
                      f"w={self.leafloss_w}, K=10 (direct add)", flush=True)

        # ---- V1 VIDPHASE (video-side PHASE structure; see
        # VIDEO_HIERARCHY_DESIGN.md): TRAIN-ONLY auxiliary InfoNCE aligning
        # the 3 temporal thirds of every GT segment (mean-pooled level-0
        # features -- phase proxies) to the 3-chunk phase-TEXT bank built
        # from the h19 ensemble's ordered phase sentences (stashed by
        # text.py as _vidphase_per). Loss-only pathway: forward outputs are
        # untouched, no new parameters => trivially init-equal and
        # byte-inert when TIFAD_VIDPHASE is unset.
        # Env: TIFAD_VIDPHASE=<float weight> (requires
        # TIFAD_PROMPT_ENSEMBLE).
        self.vidphase_w = float(_osdt.environ.get('TIFAD_VIDPHASE', '0')
                                or 0)
        if self.vidphase_w != 0:
            assert self.text_features.ensemble_prompts is not None, \
                "[VIDPHASE] requires TIFAD_PROMPT_ENSEMBLE (phase sentences)"
            print(f"[VIDPHASE] video-third <-> phase-text InfoNCE active, "
                  f"weight={self.vidphase_w}, 3 thirds, tau=0.1", flush=True)

        # ---- VIDPHASE variants (V1 regression fixes; ALL inert unless
        # TIFAD_VIDPHASE is set, and byte-inert when their own env is unset):
        # * TIFAD_VIDPHASE_W=<float> overrides the aux-loss weight (CLAB-style
        #   tiny 0.005-scale weights without code edits; default keeps the
        #   TIFAD_VIDPHASE value).
        # * TIFAD_VIDPHASE_DETACH=1 computes the InfoNCE on STOP-GRADIENT
        #   trunk features passed through a NEW trainable Linear(D, D) head
        #   (normal init OK -- it sits behind the detach), so the shared
        #   trunk receives ZERO gradient from the aux loss (V1 backpropped
        #   into the trunk and lost -2.74 mAP).
        # * TIFAD_VIDPHASE_DECAY=1 scales the weight by
        #   max(0, 1 - step/decay_steps), i.e. linear decay reaching 0 at
        #   decay_steps = 0.5 * max_steps. No global step/epoch count reaches
        #   losses() in this codebase, so `step` counts training forwards
        #   (self._vp_step) and decay_steps comes from
        #   TIFAD_VIDPHASE_DECAY_STEPS=<int> (required).
        # New params constructed HERE so make_optimizer sees them (FAMTIE
        # lesson): detach head weight->decay bucket, bias->no_decay.
        self.vidphase_detach_proj = None
        self.vidphase_decay_steps = 0
        self._vp_step = 0
        if self.vidphase_w != 0:
            _vp_w_env = _osdt.environ.get('TIFAD_VIDPHASE_W', '')
            if _vp_w_env:
                self.vidphase_w = float(_vp_w_env)
                print(f"[VIDPHASE] weight override: TIFAD_VIDPHASE_W="
                      f"{self.vidphase_w}", flush=True)
            if _osdt.environ.get('TIFAD_VIDPHASE_DETACH', '') == '1':
                self.vidphase_detach_proj = nn.Linear(embd_dim, embd_dim)
                print("[VIDPHASE] detach mode (aux grads isolated)",
                      flush=True)
            if _osdt.environ.get('TIFAD_VIDPHASE_DECAY', '') == '1':
                self.vidphase_decay_steps = int(float(_osdt.environ.get(
                    'TIFAD_VIDPHASE_DECAY_STEPS', '0') or 0))
                assert self.vidphase_decay_steps > 0, (
                    "[VIDPHASE] TIFAD_VIDPHASE_DECAY=1 requires "
                    "TIFAD_VIDPHASE_DECAY_STEPS=<step where weight hits 0; "
                    "set to 0.5*max_steps> -- no global step is visible to "
                    "losses() here")
                print(f"[VIDPHASE] linear decay active: w0={self.vidphase_w}"
                      f" -> 0 at step {self.vidphase_decay_steps}",
                      flush=True)

        # ---- VPW (TIFAD_VPW=1; PDA/APA "adaptive phase weighting" adapted
        # to the text-side phase ensemble): keep the K phase embeddings
        # per[:, 1:] (stashed by text.py as _vpw_per) available at scoring
        # time; per snippet, phase scores s_k = <feat, proj(phase_k)> at the
        # SAME calibration as the base matmul (normalize + text_proj(x/2) +
        # /logit_scale.exp()), and a zero-init Linear gate (weight AND bias)
        # gives w = softmax(gate(feat)) over phases (uniform at init). The
        # weighted phase logit is blended residually:
        #   logit = base_logit + vpw_alpha * sum_k w_k s_k
        # with vpw_alpha a LEARNED scalar init 0 => EXACT init-equality;
        # byte-inert when TIFAD_VPW is unset. Gate + alpha registered HERE
        # so make_optimizer sees them (FAMTIE lesson).
        self.vpw_gate = None
        self.vpw_alpha = None
        if _osdt.environ.get('TIFAD_VPW', '') == '1':
            assert self.text_features.ensemble_prompts is not None, \
                "[VPW] requires TIFAD_PROMPT_ENSEMBLE (phase sentences)"
            self.vpw_k = len(next(iter(
                self.text_features.ensemble_prompts.values()))) - 1
            assert self.vpw_k >= 1, \
                "[VPW] ensemble needs >= 2 sentences/class (class + phases)"
            self.vpw_gate = nn.Linear(embd_dim, self.vpw_k)
            nn.init.zeros_(self.vpw_gate.weight)
            nn.init.zeros_(self.vpw_gate.bias)
            self.vpw_alpha = nn.Parameter(torch.zeros([]))
            print(f"[VPW] active (K={self.vpw_k}, zero-init gate)",
                  flush=True)

        # ---- PHASELOGIT (TIFAD_PHASELOGIT=1; CHiLS arXiv:2302.02551
        # adapted to SCORE-SPACE phase composition): score every snippet
        # against the K phase embeddings per[:, 1:] (stashed by text.py as
        # _phaselogit_per -- own stash, independent of VPW's), pool the
        # per-class phase scores with a temperature softmax OVER PHASES
        #   pooled_c = sum_k softmax_k(s_c/tau) * s_ck
        # (tau from TIFAD_PHASELOGIT_TAU, default 0.5) and blend residually
        #   logit_c += tanh(lambda) * pooled_c
        # with lambda a scalar nn.Parameter init 0 (tanh(0)=0 => EXACT
        # init-equality). Phase embeddings get the SAME calibration as the
        # base matmul (VPW recipe: L2-normalize, text_proj(x/2),
        # /logit_scale.exp(); no bias -- the residual rides on the biased
        # base logit). Byte-inert when unset; lambda registered HERE so
        # make_optimizer sees it (FAMTIE lesson).
        self.phaselogit_lambda = None
        if _osdt.environ.get('TIFAD_PHASELOGIT', '') == '1':
            assert self.text_features.ensemble_prompts is not None, \
                "[PHASELOGIT] requires TIFAD_PROMPT_ENSEMBLE (phase sentences)"
            self.phaselogit_k = len(next(iter(
                self.text_features.ensemble_prompts.values()))) - 1
            assert self.phaselogit_k >= 1, \
                "[PHASELOGIT] ensemble needs >= 2 sentences/class (class + phases)"
            self.phaselogit_tau = float(_osdt.environ.get(
                'TIFAD_PHASELOGIT_TAU', '0.5') or 0.5)
            assert self.phaselogit_tau > 0, \
                "[PHASELOGIT] TIFAD_PHASELOGIT_TAU must be > 0"
            self.phaselogit_lambda = nn.Parameter(torch.zeros([]))
            print(f"[PHASELOGIT] active (score-space phase composition, "
                  f"tau={self.phaselogit_tau}, zero-init gate)", flush=True)

        # ---- ATTRVIDATTN (TIFAD_ATTRVIDATTN=1): attribute-embedding vs
        # VIDEO scoring, SCORE-SPACE present-vs-absent contrast (PHASELOGIT
        # pattern). The V-value attribute bank (raw CLIP embs of the
        # attrs-json value names, cached no_grad on CPU in forward --
        # CONFGATE precedent) is calibrated EXACTLY like class embeddings
        # (L2-norm, text_proj(x/2), /logit_scale.exp() -- the VPW recipe);
        # per FPN level the snippet features yield attr scores (B, T, V);
        # per class ONLY its present attributes' scores minus the absent
        # ones' are pooled:
        #   s_c = mean(scores[..., present_c]) - mean(scores[..., absent_c])
        # and blended residually: logit_c += tanh(gamma) * s_c, gamma a
        # scalar nn.Parameter init 0 (tanh(0)=0 => EXACT init-equality).
        # Classes missing from the attrs json get an all-zero contrast
        # column (skipped). gamma registered HERE so make_optimizer sees it
        # (FAMTIE lesson); byte-inert when unset. Envs: TIFAD_ATTRVIDATTN=1,
        # TIFAD_ATTRVIDATTN_ATTRS (default configs/thumos14_ttb_attrs.json).
        self.attrvid_gamma = None
        if _osdt.environ.get('TIFAD_ATTRVIDATTN', '') == '1':
            import json as _avjson
            _av_vals = _avjson.load(open(_osdt.environ.get(
                'TIFAD_ATTRVIDATTN_ATTRS',
                'configs/thumos14_ttb_attrs.json')))
            self._attrvid_vocab = sorted(
                {v for vs in _av_vals.values() for v in vs})
            _av_vid = {v: i for i, v in enumerate(self._attrvid_vocab)}
            self._attrvid_present = {c: [_av_vid[v] for v in vs]
                                     for c, vs in _av_vals.items()}
            self.attrvid_gamma = nn.Parameter(torch.zeros([]))
            self._attrvid_val_cache = None    # (V, D) raw CLIP embs, CPU
            self._attrvid_w_cache = {}        # class-list -> (V, C) contrast
            print("[ATTRVIDATTN] active (present-vs-absent attr score "
                  "contrast, zero-init gate)", flush=True)

        # ---- CONFGATE (TIFAD_CONFGATE=<float alpha_init, normally 0>):
        # confusability-adaptive FAMILY-axis scoring. Family text embeddings
        # ("a video of <family> activity", families from the fam map; raw
        # CLIP embs cached no_grad on CPU) are calibrated EXACTLY like class
        # embeddings (L2-normalize, text_proj(x/2), /logit_scale.exp() --
        # the VPW recipe) and scored against the video features per level.
        # Per-class confusability conf_c = cosine of class c's raw CLIP name
        # embedding to its NEAREST other class in the CURRENT list, min-max
        # normalized to [0,1] (cached per class-list). Blend:
        #   logit_c += tanh(gate_alpha) * conf_c * fam_logit_{fam(c)}
        # gate_alpha a scalar nn.Parameter initialized from the env value
        # (0 => tanh(0)=0 => EXACT init-equality). Classes missing from the
        # fam map get conf weight 0 (skipped). Constructed HERE so
        # make_optimizer sees gate_alpha (FAMTIE lesson). Byte-inert when
        # TIFAD_CONFGATE is unset. Envs: TIFAD_CONFGATE=<float>,
        # TIFAD_CONFGATE_MAP=<json {class: family}> (falls back to
        # TIFAD_FAMCE_MAP, then configs/thumos14_fam_map.json).
        self.confgate_alpha = None
        _cg_env = _osdt.environ.get('TIFAD_CONFGATE', '')
        if _cg_env != '':
            import json as _cgjson
            _cg_map_path = (_osdt.environ.get('TIFAD_CONFGATE_MAP', '')
                            or _osdt.environ.get('TIFAD_FAMCE_MAP', '')
                            or 'configs/thumos14_fam_map.json')
            self._confgate_map = _cgjson.load(open(_cg_map_path))
            self._confgate_fams = sorted(set(self._confgate_map.values()))
            self._confgate_fid = {f: i
                                  for i, f in enumerate(self._confgate_fams)}
            self.confgate_alpha = nn.Parameter(torch.tensor(float(_cg_env)))
            self._confgate_fam_cache = None   # (F, D) raw CLIP fam embs, CPU
            self._confgate_conf_cache = {}    # class-list -> (conf, fidx, msk)
            print(f"[CONFGATE] active (family-axis blend, zero-init gate, "
                  f"F={len(self._confgate_fams)})", flush=True)

        # ---- V2 VIDACT (video-side ACTIVITY prototypes; see
        # VIDEO_HIERARCHY_DESIGN.md): EMA prototype per ACTIVITY (taxonomy
        # family) built from GT-segment mean-pooled level-0 features at
        # TRAIN time, plus a TRAIN-ONLY cosine pull of each segment feature
        # toward its activity prototype, weighted by a LEARNED Kendall
        # log-variance (exact famce_uw mechanism: exp(-s)*w*L + s/2, s init
        # 0 => plain w*L at step 0). Prototypes/ready-flags are BUFFERS and
        # s is a PARAMETER (registered here so make_optimizer sees it --
        # FAMTIE lesson), so checkpoints round-trip strict=True ONLY when
        # TIFAD_VIDACT is also set at eval/resume time (ASPEC precedent).
        # Eval forward is untouched. Byte-inert when unset.
        # Envs: TIFAD_VIDACT=<float weight>,
        #       TIFAD_VIDACT_MAP=<json {class: activity}> (falls back to
        #         TIFAD_FAMCE_MAP),
        #       TIFAD_VIDACT_M=<EMA momentum, default 0.99>.
        # NOTE: EMA buffer updates require the single-device DataParallel
        # path (len(devices)==1, the repo default) -- replicated buffers
        # would be discarded.
        self.vidact_w = float(_osdt.environ.get('TIFAD_VIDACT', '0') or 0)
        self.vidact_uw_logvar = None
        if self.vidact_w != 0:
            import json as _vjson
            _vmap_path = (_osdt.environ.get('TIFAD_VIDACT_MAP', '')
                          or _osdt.environ.get('TIFAD_FAMCE_MAP', ''))
            assert _vmap_path, \
                "[VIDACT] set TIFAD_VIDACT_MAP (or TIFAD_FAMCE_MAP)"
            self._vidact_map = _vjson.load(open(_vmap_path))
            self._vidact_fams = sorted(set(self._vidact_map.values()))
            self._vidact_fid = {f: i
                                for i, f in enumerate(self._vidact_fams)}
            self.vidact_momentum = float(
                _osdt.environ.get('TIFAD_VIDACT_M', '0.99') or 0.99)
            self.register_buffer(
                'vidact_protos',
                torch.zeros(len(self._vidact_fams), embd_dim))
            self.register_buffer(
                'vidact_proto_ready',
                torch.zeros(len(self._vidact_fams)))
            self.vidact_uw_logvar = nn.Parameter(torch.zeros(1))
            print(f"[VIDACT] EMA activity-prototype pull active: "
                  f"F={len(self._vidact_fams)} activities from {_vmap_path},"
                  f" w={self.vidact_w}, m={self.vidact_momentum}, learned"
                  f" UW logvar (init 0)", flush=True)

        # ---- TaxGNN v3 (TIFAD_GNN3=1; see STAGE1_LITERATURE_AND_STAGE2_
        # DESIGN.md "TaxGNN v3"): instance-level taxonomy graph with LIVE
        # in-batch GT-segment nodes (gradients to backbone), trainable text
        # adapter, and TWO zero-init gated residuals -- (i) per-class text-
        # embedding residual, (ii) per-snippet graph-conditioned cross-
        # attention residual on updated_feat. Constructed HERE (not in
        # TextFeatures) because instance nodes need post-backbone video
        # features that only exist in this module's forward. Graph assets
        # reuse TIFAD_HTCA_ATTRS / TIFAD_HTCA_FAMS (+ the h19 ensemble for
        # phase nodes). Inert (byte-identical) when TIFAD_GNN3 unset --
        # taxgnn is not even imported.
        self.taxgnn3 = None
        if _osdt.environ.get('TIFAD_GNN3', '') == '1':
            from .taxgnn import TaxGNN3
            self.taxgnn3 = TaxGNN3(
                emb_dim=embd_dim,
                attrs_file=_osdt.environ['TIFAD_HTCA_ATTRS'],
                fams_file=_osdt.environ['TIFAD_HTCA_FAMS'],
                ensemble_prompts=self.text_features.ensemble_prompts,
                inst_per_class=int(
                    _osdt.environ.get('TIFAD_GNN3_K', '2') or 2),
                inst_drop=float(
                    _osdt.environ.get('TIFAD_GNN3_DROP', '0.3') or 0.3),
            )
            print(f"[TaxGNN3] active: K={self.taxgnn3.inst_per_class} "
                  f"instances/class, inst_drop={self.taxgnn3.inst_drop}",
                  flush=True)

        # ---- VATTR (TIFAD_VATTR=1): VIDEO-side attribute space. A linear
        # head vattr_head: (D -> V) reads per-snippet ATTRIBUTE-VALUE
        # evidence off the trunk features; V = #unique values in the attrs
        # json (sorted vocab; 40 for configs/thumos14_ttb_attrs_v2.json --
        # NOT hardcoded). Three pieces:
        #   (1) TRAIN aux BCE (in losses(), on the focal loss's positive
        #       points, DETACHED trunk features -- VIDPHASE_DETACH lesson):
        #       weak labels = the GT class's binary attribute code row.
        #   (2) TRAIN orthogonality penalty on the head rows:
        #       ||W_n W_n^T - I||_F^2 / V^2, W_n row-L2-normalized.
        #   (3) SCORE-SPACE blend (train AND eval forward): per level,
        #       a = sigmoid(head(feat)) (B,T,V); class attr score
        #       s_c = <a, code_c - colmean(codes)> / ||code_c - colmean||_1
        #       over the CURRENT class list; logit_c += tanh(vattr_gate)*s_c
        #       with vattr_gate a scalar nn.Parameter init 0 (tanh(0)=0 =>
        #       EXACT init-equality; head init is normal -- it feeds only
        #       the aux loss and this zero-gated path).
        # Head + gate constructed HERE so make_optimizer sees them (FAMTIE
        # lesson: weight->decay, bias/gate handled by the name rules). The
        # code matrix is a BUFFER, so ckpts round-trip strict=True ONLY when
        # TIFAD_VATTR is also set at eval/resume (ASPEC/VIDACT precedent).
        # Byte-inert when TIFAD_VATTR is unset.
        # Env: TIFAD_VATTR=1,
        #      TIFAD_VATTR_ATTRS=<json {class: [values]}>
        #        (default configs/thumos14_ttb_attrs_v2.json),
        #      TIFAD_VATTR_W=<aux BCE weight, default 0.05>,
        #      TIFAD_VATTR_ORTH=<orth weight, default 0.1>,
        #      TIFAD_VATTR_DECAY_STEPS=<step where the aux weight hits 0;
        #        0/unset = no decay> (VIDPHASE decay pattern: no global step
        #        reaches losses(), so a per-forward counter is used).
        self.vattr_head = None
        self.vattr_gate = None
        if _osdt.environ.get('TIFAD_VATTR', '') == '1':
            import json as _vajson
            _va_path = _osdt.environ.get(
                'TIFAD_VATTR_ATTRS', 'configs/thumos14_ttb_attrs_v2.json')
            _va_map = _vajson.load(open(_va_path))
            # sorted class / value lists (python attrs -- tensors below are
            # the buffers; buffers cannot hold strings)
            self._vattr_classes = sorted(_va_map)
            self._vattr_values = sorted(
                {v for vs in _va_map.values() for v in vs})
            _va_vid = {v: i for i, v in enumerate(self._vattr_values)}
            _va_V = len(self._vattr_values)
            _va_code = torch.zeros(len(self._vattr_classes), _va_V)
            for _va_ci, _va_c in enumerate(self._vattr_classes):
                for _va_v in _va_map[_va_c]:
                    _va_code[_va_ci, _va_vid[_va_v]] = 1.0
            self.register_buffer('vattr_code', _va_code)     # (C_json, V)
            self._vattr_cid = {c: i
                               for i, c in enumerate(self._vattr_classes)}
            self.vattr_head = nn.Linear(embd_dim, _va_V)
            self.vattr_gate = nn.Parameter(torch.zeros([]))
            self.vattr_w = float(
                _osdt.environ.get('TIFAD_VATTR_W', '0.05') or 0.05)
            self.vattr_orth = float(
                _osdt.environ.get('TIFAD_VATTR_ORTH', '0.1') or 0.1)
            self.vattr_decay_steps = int(float(_osdt.environ.get(
                'TIFAD_VATTR_DECAY_STEPS', '0') or 0))
            self._vattr_step = 0
            self._vattr_code_cache = {}  # class-list -> (code (C,V), present (C,))
            self._vattr_score_cache = {}  # class-list -> (V, C) centered/L1 weights
            _va_msg = (f"[VATTR] active (V={_va_V}, detached BCE "
                       f"w={self.vattr_w}, orth={self.vattr_orth}, "
                       f"zero-init gate)")
            if self.vattr_decay_steps > 0:
                _va_msg += f" +decay to 0 at {self.vattr_decay_steps}"
            print(_va_msg, flush=True)

        # ---- LEVELAXIS (TIFAD_LEVELAXIS=1): Dynamic-Head-style scale
        # attention across the pyramid levels (see LevelAxisScaleAttn).
        # Plugs in AFTER the trunk (+ GNN3/DT/HTCA residuals), BEFORE the
        # heads; BOTH the cls and reg sides see the refined features (v1
        # simplicity). Constructed HERE so make_optimizer sees the params
        # (HTCA/VATTR precedent). Zero-init out_proj + zero-init level
        # embeddings => EXACT init-equality; byte-inert when unset.
        self.levelaxis = None
        if _osdt.environ.get('TIFAD_LEVELAXIS', '') == '1':
            self.levelaxis = LevelAxisScaleAttn(
                fpn_dim, len(self.fpn_strides), n_head=4,
                scale_factor=scale_factor)
            print("[LEVELAXIS] active (6-level scale attention, zero-init)",
                  flush=True)

        # ---- TDNECK (TIFAD_TDNECK=1): CLTDR-style top-down semantic
        # fusion with task decoupling (see TopDownSemNeck). The sem pyramid
        # feeds ONLY the classification side (text matmul + score-space
        # logit blocks); reg/fg/cn heads keep the raw features. Composable
        # with LEVELAXIS (levelaxis first, tdneck on its output).
        # Constructed HERE so make_optimizer sees the params. Zero-init 1x1
        # conv gates (weight+bias) => EXACT init-equality; byte-inert when
        # unset.
        self.tdneck = None
        if _osdt.environ.get('TIFAD_TDNECK', '') == '1':
            _cg = _osdt.environ.get('TIFAD_TDNECK_GATE', '') == '1'
            self.tdneck = TopDownSemNeck(fpn_dim, len(self.fpn_strides),
                                         content_gate=_cg)
            print("[TDNECK] active (top-down semantic pyramid -> cls, "
                  "raw -> reg, zero-init gates"
                  + (", content-gated v2" if _cg else "") + ")", flush=True)

        # ---- CTXATTN (TIFAD_CTXATTN=1): hierarchical ACTIVITY-CONTEXT
        # attention (see SemCtxAttn) -- snippets cross-attend a semantic
        # token bank of activity-family + current-class text embeddings.
        # DIMS: text_features.extract_text_emb returns embd_dim-d vectors
        # (text.py swaps CLIP's text_projection for a trainable
        # Linear(text_embed_dim -> embd_dim) that get_text_features
        # applies), so the bank projection is Linear(embd_dim -> fpn_dim).
        # Banks: activity bank cached ONCE (no_grad, CPU -- CONFGATE
        # precedent); action bank cached PER CLASS-LIST tuple (train and
        # test lists differ). Runs in forward alongside LEVELAXIS --
        # BEFORE the TDNECK split -- so under TDNECK the sem (cls) pyramid
        # is built from the refined features while this is cls-flavored
        # evidence reaching both sides in v1 (documented at the call
        # site). Constructed HERE (kept LAST with VIDSLOT so base weights
        # draw identical RNG whether or not the env is set) so
        # make_optimizer sees the params. Zero-init out_proj + zero-init
        # level embedding => EXACT init-equality; byte-inert when unset.
        # Envs: TIFAD_CTXATTN=1, TIFAD_CTXATTN_MAP=<json {class: family}>
        # (falls back to TIFAD_FAMCE_MAP, then
        # configs/thumos14_fam_map.json).
        self.ctxattn = None
        if _osdt.environ.get('TIFAD_CTXATTN', '') == '1':
            import json as _cxjson
            _cx_map_path = (_osdt.environ.get('TIFAD_CTXATTN_MAP', '')
                            or _osdt.environ.get('TIFAD_FAMCE_MAP', '')
                            or 'configs/thumos14_fam_map.json')
            self._ctxattn_fams = sorted(set(
                _cxjson.load(open(_cx_map_path)).values()))
            self.ctxattn = SemCtxAttn(embd_dim, fpn_dim, n_head=4)
            self._ctxattn_fam_cache = None   # (F, D) raw text embs, CPU
            self._ctxattn_act_cache = {}     # class-list -> (C, D) CPU
            print(f"[CTXATTN] active (semantic-hierarchy context "
                  f"attention: {len(self._ctxattn_fams)} activities + "
                  f"{len(self.text_features.train_classes)}(train)/"
                  f"{len(self.text_features.test_classes)}(test) actions, "
                  f"zero-init)", flush=True)

        # ---- VIDSLOT (TIFAD_VIDSLOT=<K int>): K temporal slot queries
        # gather multi-second context from level-0 and scatter it back to
        # every pyramid level (see TemporalSlotAttn). Detection-loss-only.
        # Runs in forward after CTXATTN, BEFORE the TDNECK split (both cls
        # and reg sides see it, v1 simplicity). Constructed HERE (kept
        # LAST -- see CTXATTN note on RNG) so make_optimizer sees the
        # params. Zero-init MHA-2 out_proj => EXACT init-equality;
        # byte-inert when unset.
        self.vidslot = None
        _vs_env = _osdt.environ.get('TIFAD_VIDSLOT', '')
        if _vs_env not in ('', '0'):
            _vs_k = int(_vs_env)
            assert _vs_k > 0, "[VIDSLOT] TIFAD_VIDSLOT must be a positive int"
            self.vidslot = TemporalSlotAttn(fpn_dim, _vs_k, n_head=4)
            print(f"[VIDSLOT] active (K={_vs_k} temporal slots, zero-init "
                  f"return path)", flush=True)

        # ---- POSEEV (TIFAD_POSEEV=1): pose-kinematics EVIDENCE BRANCH.
        # Consumes item['pose_feats'] (234 x T: 110-d 2D + 124-d 3D
        # kinematics on the I3D grid, loaded by the dataset -- thumos14.py
        # [POSEEV]). Three pieces:
        #   (1) encoder: Conv1d(234->D,k3)+ReLU+Conv1d(D->D,k3) over the
        #       full-resolution grid, then per-FPN-level strided max-pool
        #       (kernel=stride=fpn_strides[l]) to each pyramid length;
        #       attribute head Linear(D -> V) over the SAME attrs vocab as
        #       VATTR. Encoder/head init is NORMAL -- they feed only this
        #       branch, never the trunk.
        #   (2) TRAIN aux BCE (in losses()) of the pose attribute head at
        #       the focal loss's positive points, restricted to the FACET
        #       SUBSET columns (motion/pose attributes). NO detach: the
        #       pose branch shares ZERO parameters with the detection trunk
        #       (its input is the raw pose tensor, not trunk features), so
        #       this aux loss cannot push gradients into the trunk -- the
        #       VIDPHASE_DETACH concern does not arise.
        #   (3) SCORING (train AND eval, VATTR recipe): a = sigmoid(head)
        #       over the facet columns; s_c = <a, code_c centered over the
        #       CURRENT class list, facet columns only> / L1; logit_c +=
        #       tanh(poseev_gate) * s_c, gate a scalar nn.Parameter init 0
        #       => EXACT init-equality on the scoring path. This is cls
        #       evidence (cls_feat routing convention): it lands on
        #       out_cls_logits alongside the VATTR blend, BEFORE the
        #       SUBCLASS / LEVEL-COMPOSE pools.
        # Constructed HERE (LAST -- see CTXATTN note: base weights draw
        # identical RNG whether or not the env is set) so make_optimizer
        # sees the params. Byte-inert when TIFAD_POSEEV unset.
        # Env: TIFAD_POSEEV=1,
        #      TIFAD_POSEEV_ATTRS=<json {class: [values]}>
        #        (default configs/thumos14_ttb_attrs_v3.json),
        #      TIFAD_POSEEV_W=<aux BCE weight, default 0.1>,
        #      TIFAD_POSEEV_FACETS=<comma list of attribute-name prefixes
        #        that select the supervised+scored columns; default
        #        "motion_,force_,arm_,two_handed,rotational_windup,
        #        has_runup,runup_">. Non-facet logits exist but are neither
        #        supervised nor scored.
        self.poseev_enc = None
        self.poseev_head = None
        self.poseev_gate = None
        if _osdt.environ.get('TIFAD_POSEEV', '') == '1':
            import json as _pejson
            _pe_path = _osdt.environ.get(
                'TIFAD_POSEEV_ATTRS', 'configs/thumos14_ttb_attrs_v3.json')
            _pe_map = _pejson.load(open(_pe_path))
            self._poseev_classes = sorted(_pe_map)
            self._poseev_values = sorted(
                {v for vs in _pe_map.values() for v in vs})
            _pe_vid = {v: i for i, v in enumerate(self._poseev_values)}
            _pe_V = len(self._poseev_values)
            _pe_code = torch.zeros(len(self._poseev_classes), _pe_V)
            for _pe_ci, _pe_c in enumerate(self._poseev_classes):
                for _pe_v in _pe_map[_pe_c]:
                    _pe_code[_pe_ci, _pe_vid[_pe_v]] = 1.0
            self.register_buffer('poseev_code', _pe_code)    # (C_json, V)
            self._poseev_cid = {c: i
                                for i, c in enumerate(self._poseev_classes)}
            _pe_facets = [s for s in _osdt.environ.get(
                'TIFAD_POSEEV_FACETS',
                'motion_,force_,arm_,two_handed,rotational_windup,'
                'has_runup,runup_').split(',') if s]
            self._poseev_fidx = torch.tensor(
                [i for i, v in enumerate(self._poseev_values)
                 if any(v.startswith(p) for p in _pe_facets)],
                dtype=torch.long)                            # CPU, derived
            assert self._poseev_fidx.numel() > 0, \
                "[POSEEV] TIFAD_POSEEV_FACETS matches no attribute values"
            self._poseev_dim = 234
            self.poseev_enc = nn.Sequential(
                nn.Conv1d(self._poseev_dim, embd_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(embd_dim, embd_dim, 3, padding=1),
            )
            self.poseev_head = nn.Linear(embd_dim, _pe_V)
            self.poseev_gate = nn.Parameter(torch.zeros([]))
            self.poseev_w = float(
                _osdt.environ.get('TIFAD_POSEEV_W', '0.1') or 0.1)
            self._poseev_code_cache = {}   # class-list -> (code, present)
            self._poseev_score_cache = {}  # class-list -> (Vf, C) weights
            self._poseev_warned_nokey = False
            print(f"[POSEEV] active (pose kinematics branch, "
                  f"D={self._poseev_dim}, V={_pe_V}, "
                  f"facet={self._poseev_fidx.numel()} attrs, "
                  f"zero-init gate)", flush=True)

        # ---- PHASEHEAD (TIFAD_PHASEHEAD=1): Stage-B DETACHED phase
        # predictor trained on the Stage-A pseudo labels the dataset
        # attaches as item['phase_feats'] (768 x T vmae_grid) and
        # item['phase_targets'] ((T,) long, -1 = unlabeled, else phase
        # 0..9 for K1-passing classes only). Encoder Conv1d(768->256,k3)
        # + ReLU + Conv1d(256->256,k3), head Linear(256->10). The branch
        # reads the RAW vmae_grid tensor and shares ZERO parameters with
        # the detection trunk, and (unlike POSEEV) it NEVER touches
        # out_cls_logits: v1 is pure evidence infrastructure -- the
        # scoring path is byte-identical with the env set, so no
        # zero-init gate is needed and no detach is needed (backprop
        # from its losses reaches only phasehead_* params; interference
        # is structurally impossible). TRAIN: CE over grid steps with
        # target >= 0 (* TIFAD_PHASEHEAD_W, default 1.0 -- shares no
        # params, so the weight only trades off against nothing) +
        # monotonicity hinge on E[phase|t] within labeled runs
        # (* TIFAD_PHASEHEAD_MONO, default 0.3, margin 0.0). EVAL:
        # TIFAD_PHASEHEAD_DUMP=<dir> stashes per-video (T, 10) logits
        # for Stage-C offline work. Params in __init__ (optimizer sees
        # them). Byte-inert when TIFAD_PHASEHEAD unset.
        self.phasehead_enc = None
        self.phasehead_head = None
        if _osdt.environ.get('TIFAD_PHASEHEAD', '') == '1':
            self._phasehead_dim = 768
            self._phasehead_K = 10
            self.phasehead_enc = nn.Sequential(
                nn.Conv1d(self._phasehead_dim, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, 256, 3, padding=1),
            )
            self.phasehead_head = nn.Linear(256, self._phasehead_K)
            self.phasehead_w = float(
                _osdt.environ.get('TIFAD_PHASEHEAD_W', '1.0') or 1.0)
            self.phasehead_mono = float(
                _osdt.environ.get('TIFAD_PHASEHEAD_MONO', '0.3') or 0.3)
            self.phasehead_dump = _osdt.environ.get(
                'TIFAD_PHASEHEAD_DUMP', '')
            self._phasehead_warned_nokey = False
            print(f"[PHASEHEAD] active (VMAE-grid phase predictor, "
                  f"K1 classes, mono={self.phasehead_mono:g})", flush=True)

        # ---- VSA (TIFAD_VSA=1): VIDEO-SIDE SUBTREE ATTENTION, the mirror
        # of the text-side phasechain + H4b composition per candidate
        # segment (see libs/modeling/vsa.py). Runs in forward AFTER the
        # reg head (it reads the DETACHED offsets) and replaces cls_feat --
        # the pyramid the class cosine reads -- on every level / location
        # (train AND eval); reg/fg/cn keep the raw features. Constructed
        # HERE (kept LAST -- CTXATTN/VIDSLOT RNG note) so make_optimizer
        # sees the params; the attribute is named `vsa` so
        # TIFAD_FREEZE_EXCEPT=vsa selects exactly `vsa.*`. Both out_proj's
        # are zero-init => EXACT init-equality; byte-inert when unset.
        # Envs: TIFAD_VSA_K (default 10), TIFAD_VSA_HEADS (default 4).
        # VERSION 2 (see vsa.py docstring): TIFAD_VSA_XM=1 cross-modal
        # ordered phase matching (identity-init adaptors, zero-init position
        # weights and term weight lam; text.py stash `_vsa_phase_embs` in
        # BOTH modes; the term is added to the cls logits after the base
        # matmul in base-cosine units), TIFAD_VSA_XM_ONLY=1 (matching term
        # without the v1 clip attention), TIFAD_VSA_SOFTPHASE=1 (posterior-
        # weighted chunk pooling from the LEAFLOSS trunk phase head,
        # blended with the equal chunks by TIFAD_VSA_SOFTPHASE_MIX, default
        # 0.5; requires TIFAD_LEAFLOSS=1 TIFAD_LEAFLOSS_HEAD=1). All three
        # keep exact init-equality and are byte-inert when unset.
        self.vsa = None
        if _osdt.environ.get('TIFAD_VSA', '') == '1':
            from .vsa import VideoSubtreeAttn
            _vsa_k = int(_osdt.environ.get('TIFAD_VSA_K', '10') or 10)
            _vsa_h = int(_osdt.environ.get('TIFAD_VSA_HEADS', '4') or 4)
            _vsa_xm = _osdt.environ.get('TIFAD_VSA_XM', '') == '1'
            _vsa_xmo = _osdt.environ.get('TIFAD_VSA_XM_ONLY', '') == '1'
            _vsa_sp = _osdt.environ.get('TIFAD_VSA_SOFTPHASE', '') == '1'
            _vsa_mix = float(_osdt.environ.get(
                'TIFAD_VSA_SOFTPHASE_MIX', '0.5') or 0.5)
            # (D) TIFAD_VSA_POSEMB=1 phase-index token embeddings; (E)
            # TIFAD_VSA_ALIGN_W=<w> phase-alignment loss (train only,
            # losses(): align_loss / align_acc; needs XM's adaptors)
            _vsa_pe = _osdt.environ.get('TIFAD_VSA_POSEMB', '') == '1'
            self.vsa_align_w = float(
                _osdt.environ.get('TIFAD_VSA_ALIGN_W', '0') or 0)
            if self.vsa_align_w != 0:
                assert _vsa_xm, "[VSA-ALIGN] requires TIFAD_VSA_XM=1"
            # v3: NOSELF / OUTINIT / MINLVL (vsa.py), ALIGN_UW (4th Kendall
            # slot for the alignment loss; requires LEAFLOSS's 3-slot
            # logvar + a nonzero ALIGN_W), XM_RESCORE (eval-only candidate
            # rescoring by the raw ordered match, inference_single_video)
            _vsa_noself = _osdt.environ.get('TIFAD_VSA_NOSELF', '') == '1'
            _vsa_oinit = float(_osdt.environ.get('TIFAD_VSA_OUTINIT', '0') or 0)
            _vsa_minl = int(_osdt.environ.get('TIFAD_VSA_MINLVL', '0') or 0)
            self.vsa_align_uw = _osdt.environ.get('TIFAD_VSA_ALIGN_UW', '') == '1'
            if self.vsa_align_uw:
                assert self.vsa_align_w != 0, \
                    "[VSA-ALIGN-UW] requires TIFAD_VSA_ALIGN_W != 0"
                assert (self.famce_uw_logvar is not None
                        and self.famce_uw_logvar.numel() == 3), \
                    ("[VSA-ALIGN-UW] requires TIFAD_FAMCE_UW=1 + "
                     "TIFAD_LEAFLOSS=1 (3-slot logvar to extend)")
                self.famce_uw_logvar = nn.Parameter(torch.zeros(4))
                print("[VSA-ALIGN-UW] Kendall logvar extended to {main, "
                      "famce, leaf, align} (init s=0 -> weights 1,1,1,1)",
                      flush=True)
            self.vsa_xm_rescore = float(
                _osdt.environ.get('TIFAD_VSA_XM_RESCORE', '0') or 0)
            if self.vsa_xm_rescore != 0:
                assert _vsa_xm, "[VSA-XM-RESCORE] requires TIFAD_VSA_XM=1"
            if _vsa_xm:
                assert self.text_features.ensemble_prompts is not None, \
                    "[VSA-XM] requires TIFAD_PROMPT_ENSEMBLE (phase sentences)"
                _vsa_kt = len(next(iter(
                    self.text_features.ensemble_prompts.values()))) - 1
                assert _vsa_kt == _vsa_k, \
                    f"[VSA-XM] text phases ({_vsa_kt}) != TIFAD_VSA_K ({_vsa_k})"
            if _vsa_sp:
                assert getattr(self, 'leaf_head', None) is not None, \
                    ("[VSA-SOFTPHASE] requires the trunk phase head: "
                     "TIFAD_LEAFLOSS=1 TIFAD_LEAFLOSS_HEAD=1")
                assert _vsa_k == 10, "[VSA-SOFTPHASE] head is 10-way: K=10"
            # ---- NESTED (TIFAD_VSA_NESTED=1): two-level video subtree
            # attention (K action chunks x Kp = TIFAD_VSA_KP leaf chunks,
            # default 3; see vsa.py VideoSubtreeAttnNested). Same attribute
            # `vsa` (TIFAD_FREEZE_EXCEPT=vsa), same call / return
            # convention as v1 (match / align are None); all four
            # out_proj's zero-init => exact init-equality. The v2/v3 flags
            # are unsupported and asserted off. Byte-inert when unset.
            _vsa_nested = _osdt.environ.get('TIFAD_VSA_NESTED', '') == '1'
            if _vsa_nested:
                from .vsa import VideoSubtreeAttnNested
                _vsa_kp = int(_osdt.environ.get('TIFAD_VSA_KP', '3') or 3)
                assert not (_vsa_xm or _vsa_xmo or _vsa_sp or _vsa_pe
                            or _vsa_noself or _vsa_oinit != 0
                            or _vsa_minl != 0 or self.vsa_align_w != 0
                            or self.vsa_xm_rescore != 0), \
                    ("[VSA-nested] TIFAD_VSA_XM / XM_ONLY / SOFTPHASE / "
                     "POSEMB / NOSELF / OUTINIT / MINLVL / ALIGN_W / "
                     "XM_RESCORE must be off with TIFAD_VSA_NESTED=1")
                self.vsa = VideoSubtreeAttnNested(
                    fpn_dim, K=_vsa_k, Kp=_vsa_kp, heads=_vsa_h)
                print(f"[VSA-nested] active (two-level video-side subtree "
                      f"attention: K={_vsa_k} action chunks x Kp={_vsa_kp} "
                      f"leaf chunks, heads={_vsa_h}, four zero-init "
                      f"residuals, replaces cls_feat)", flush=True)
            else:
                self.vsa = VideoSubtreeAttn(
                    fpn_dim, K=_vsa_k, heads=_vsa_h, xm=_vsa_xm,
                    xm_only=_vsa_xmo, softphase=_vsa_sp,
                    softphase_mix=_vsa_mix, posemb=_vsa_pe,
                    noself=_vsa_noself, out_init=_vsa_oinit,
                    min_lvl=_vsa_minl,
                    learn_bounds=_osdt.environ.get('TIFAD_VSA_LEARN_BOUNDS', '') == '1')
                print(f"[VSA] active (video-side subtree attention: "
                      f"K={_vsa_k} chunks, heads={_vsa_h}, zero-init "
                      f"residuals, replaces cls_feat; v2: xm={_vsa_xm} "
                      f"xm_only={_vsa_xmo} softphase={_vsa_sp} "
                      f"mix={_vsa_mix} posemb={_vsa_pe} "
                      f"align_w={self.vsa_align_w}; v3: noself={_vsa_noself} "
                      f"out_init={_vsa_oinit} min_lvl={_vsa_minl} "
                      f"align_uw={self.vsa_align_uw} "
                      f"xm_rescore={self.vsa_xm_rescore})", flush=True)

        # ---- VSA_PRE (TIFAD_VSA_PRE=1): PRE-cross-attention placement of
        # the video-side subtree attention, symmetric to the TEXT side
        # (text.py composes a node's representation from its sub-action
        # descriptions BEFORE Trunk's joint text-video cross-attention
        # runs; TIFAD_VSA=1/v1 above composes the video side's equivalent
        # AFTER Trunk already ran, so it can only ever rescore cls_feat).
        # Reuses the SAME VideoSubtreeAttn class as v1 (own params, own
        # module attribute `vsa_pre` so TIFAD_FREEZE_EXCEPT=vsa_pre selects
        # exactly `vsa_pre.*`) but applied to a single "level" -- the
        # embedding-network output (backbone.embed(); FPN dim, full T0
        # resolution, literally the tensor about to become Trunk's `vid`
        # argument before ANY of its TiCA layers run; see backbones.py
        # ConvTransformerBackbone.embed()/.trunk() split). See forward()
        # for the reference-span + single-live-Trunk-pass plumbing (the
        # comment block just above `self.backbone.embed(...)`).
        # Both out_proj's are zero-init -> exact init-equality; byte-inert
        # when TIFAD_VSA_PRE is unset (module never constructed, and
        # forward()'s TIFAD_VSA_PRE branch is skipped entirely).
        self.vsa_pre = None
        if _osdt.environ.get('TIFAD_VSA_PRE', '') == '1':
            from .vsa import VideoSubtreeAttn
            _vsap_k = int(_osdt.environ.get('TIFAD_VSA_PRE_K', '10') or 10)
            _vsap_h = int(_osdt.environ.get('TIFAD_VSA_PRE_HEADS', '4') or 4)
            self.vsa_pre = VideoSubtreeAttn(fpn_dim, K=_vsap_k, heads=_vsap_h)
            print(f"[VSA-PRE] active (PRE-cross-attention video-side "
                  f"subtree attention: K={_vsap_k} chunks, heads={_vsap_h}, "
                  f"zero-init residuals, refines Trunk's video input -> "
                  f"reaches cls/reg/fg/cn)", flush=True)

        # ---- QH (TIFAD_QH=1): CLASS-AGNOSTIC LOCALIZATION-QUALITY HEAD,
        # VSA v4 (libs/modeling/qhead.py; review memo autoresearch/
        # vsa_review_v4.md). A SEPARATE head on the DETACHED pre-VSA cls
        # pyramid: per location, its own feature f_n + 12 span pools of
        # level 0 over the 1.5x-expanded DECODED span (pre / K=10 in-span /
        # post, SSN-TadTR context) [+ the ordered phase-similarity map
        # against the calibrated text phases of the location's class,
        # TIFAD_QH_PHASEMAP=1] -> MLP -> sigmoid q = predicted tIoU of the
        # decoded span with its best GT. Trained SEQUENTIALLY on the frozen
        # champion (TIFAD_INIT_FROM + TIFAD_FREEZE_EXCEPT=qhead): soft BCE
        # on the tIoU target (TIFAD_QH_LOSS=bce|l1) + SSN fragment-pair
        # margin ranking (TIFAD_QH_RANK_W, default 0.5; TIFAD_QH_MARGIN
        # 0.2; TIFAD_QH_BG_RATIO 1.0). Eval: pre-NMS score = score^(1-b) *
        # q^b with b = TIFAD_QH_BETA (default 0.5; 0 == byte-identical
        # champion path), so soft-NMS is ORDERED by the combined score.
        # NEVER retargets an existing head (iouq lesson), never applies a
        # per-video-normalised multiplier (compl lesson); the trunk gets no
        # gradient (inputs detached). Constructed LAST (RNG note as VSA);
        # attribute `qhead` so TIFAD_FREEZE_EXCEPT=qhead selects exactly
        # `qhead.*`. Byte-inert when TIFAD_QH is unset.
        self.qhead = None
        if _osdt.environ.get('TIFAD_QH', '') == '1':
            from .qhead import QualityHead
            _qh_k = int(_osdt.environ.get('TIFAD_QH_K', '10') or 10)
            _qh_pm = _osdt.environ.get('TIFAD_QH_PHASEMAP', '') == '1'
            self.qh_beta = float(
                _osdt.environ.get('TIFAD_QH_BETA', '0.5') or 0)
            self.qh_comb = _osdt.environ.get('TIFAD_QH_COMB', 'geo') or 'geo'
            assert self.qh_comb in ('geo', 'prod'), \
                f"[QH] TIFAD_QH_COMB must be geo|prod, got {self.qh_comb}"
            if _qh_pm:
                assert self.text_features.ensemble_prompts is not None, \
                    "[QH-PHASEMAP] requires TIFAD_PROMPT_ENSEMBLE"
                _qh_kt = len(next(iter(
                    self.text_features.ensemble_prompts.values()))) - 1
                assert _qh_kt == _qh_k, \
                    f"[QH-PHASEMAP] text phases ({_qh_kt}) != TIFAD_QH_K ({_qh_k})"
            self.qhead = QualityHead(
                fpn_dim, K=_qh_k, phasemap=_qh_pm,
                raw=_osdt.environ.get('TIFAD_QH_RAW', '1') != '0',
                struct=_osdt.environ.get('TIFAD_QH_STRUCT', '') == '1',
                dropout=float(_osdt.environ.get('TIFAD_QH_DROPOUT', '0') or 0),
                lvlfeat=_osdt.environ.get('TIFAD_QH_LVLFEAT', '') == '1',
                n_levels=len(self.fpn_strides),
                hidden=int(_osdt.environ.get('TIFAD_QH_HIDDEN', '256') or 256),
                ctx=float(_osdt.environ.get('TIFAD_QH_CTX', '0.5') or 0.5),
                loss=_osdt.environ.get('TIFAD_QH_LOSS', 'bce') or 'bce',
                rank_w=float(_osdt.environ.get('TIFAD_QH_RANK_W', '0.5') or 0),
                margin=float(_osdt.environ.get('TIFAD_QH_MARGIN', '0.2') or 0),
                bg_ratio=float(_osdt.environ.get('TIFAD_QH_BG_RATIO', '1.0') or 0),
                beta=self.qh_beta)
            print(f"[QH] active (class-agnostic quality head: K={_qh_k} "
                  f"in-span chunks + pre/post context {self.qhead.ctx}, "
                  f"raw={self.qhead.raw} struct={self.qhead.struct} "
                  f"phasemap={_qh_pm} lvlfeat={self.qhead.lvlfeat} "
                  f"in_dim={self.qhead.in_dim}, "
                  f"loss={self.qhead.loss_type}, "
                  f"rank_w={self.qhead.rank_w}, margin={self.qhead.margin}, "
                  f"bg_ratio={self.qhead.bg_ratio}, eval beta={self.qh_beta} "
                  f"comb={self.qh_comb})",
                  flush=True)

        # ---- BR (TIFAD_BR=1, R1 2026-09-17): BOUNDARY REFINEMENT HEAD on
        # the frozen trunk (qhead.BoundaryHead): the QH's detached context
        # pools + f_n + level one-hot + log-duration -> (ds, de) =
        # tanh * TIFAD_BR_RANGE (0.5) * L per side; L1 + DIoU on candidates
        # matched at tIoU >= 0.5 (/ loss_normalizer, x TIFAD_BR_W). Eval:
        # TIFAD_BR_APPLY=1 corrects the decoded spans before soft-NMS
        # (TIFAD_BR_ITERS re-pooling rounds); zero-init last layer =>
        # corrections exactly 0 => byte-identical decode untrained.
        # Attribute `brhead` (TIFAD_FREEZE_EXCEPT=brhead). Inert unset.
        self.brhead = None
        if _osdt.environ.get('TIFAD_BR', '') == '1':
            from .qhead import BoundaryHead
            self.br_apply = _osdt.environ.get('TIFAD_BR_APPLY', '') == '1'
            self.brhead = BoundaryHead(
                fpn_dim,
                K=int(_osdt.environ.get('TIFAD_QH_K', '10') or 10),
                hidden=int(_osdt.environ.get('TIFAD_BR_HIDDEN', '256') or 256),
                ctx=float(_osdt.environ.get('TIFAD_QH_CTX', '0.5') or 0.5),
                rng=float(_osdt.environ.get('TIFAD_BR_RANGE', '0.5') or 0.5),
                w=float(_osdt.environ.get('TIFAD_BR_W', '1.0') or 0),
                n_levels=len(self.fpn_strides),
                iters=int(_osdt.environ.get('TIFAD_BR_ITERS', '1') or 1))
            if self.qhead is not None:
                assert (self.qhead.K == self.brhead.K
                        and self.qhead.ctx == self.brhead.ctx), \
                    "[BR] K / ctx must match the quality head"
            print(f"[BR] active (boundary refinement head: range="
                  f"{self.brhead.rng} L, w={self.brhead.w}, iters="
                  f"{self.brhead.iters}, in_dim={self.brhead.in_dim}, "
                  f"apply={self.br_apply})", flush=True)

        # ---- PH (TIFAD_PH_SRC=probe|text|head, VSA v5 2026-09-18; see
        # libs/modeling/phaseparse.py and autoresearch/vsa_v5_spec.md):
        # PHASE-PARSED EXTENTS. Per-location phase log-posteriors (K=10
        # phases + background) at level 0 of the DETACHED pre-VSA cls
        # trunk from one of three sources (probe head buffers / centered
        # text cosine / a new sequentially trained head `phasehead`), a
        # batched transcript-constrained Viterbi over each pre-NMS
        # candidate's 1.5x-expanded window, and three consumers applied
        # BEFORE soft-NMS: TIFAD_PH_EXTENT=<alpha> (span blend with the
        # parsed extent), TIFAD_PH_COMPL=<beta> (score *= completeness^b),
        # TIFAD_PH_BOUND=1 (phase-profile boundary head `ph_brhead`,
        # FREEZE_EXCEPT=phasehead,brhead). Trunk never touched (inputs
        # detached). Byte-inert when TIFAD_PH_SRC is unset.
        self.phasehead = None
        self.ph_brhead = None
        _ph_src = _osdt.environ.get('TIFAD_PH_SRC', '')
        if _ph_src:
            from .phaseparse import PhaseEvidence, PhaseBoundaryHead
            _ph_k = int(_osdt.environ.get('TIFAD_PH_K', '10') or 10)
            self.ph_alpha = float(_osdt.environ.get('TIFAD_PH_EXTENT', '0') or 0)
            self.ph_beta = float(_osdt.environ.get('TIFAD_PH_COMPL', '0') or 0)
            # TIFAD_PH_MINLLR unset -> no gating (the parse's llr vs the
            # all-background path is negative on average under scaled
            # likelihoods; gate only when set explicitly)
            _ph_ml = _osdt.environ.get('TIFAD_PH_MINLLR', '')
            self.ph_minllr = float(_ph_ml) if _ph_ml != '' else float('-inf')
            self.ph_maxw = int(_osdt.environ.get('TIFAD_PH_MAXW', '128') or 128)
            self.ph_ctx = float(_osdt.environ.get('TIFAD_PH_CTX', '0.5') or 0.5)
            self.ph_w = float(_osdt.environ.get('TIFAD_PH_W', '1.0') or 0)
            self.ph_bg_ratio = float(
                _osdt.environ.get('TIFAD_PH_BG_RATIO', '1.0') or 0)
            # TIFAD_PH_TARGET=pseudo (v5h: text-aligned pseudo cuts) |
            # decile (v5p: position decile inside the GT, no text at all)
            self.ph_target = _osdt.environ.get('TIFAD_PH_TARGET', 'pseudo') or 'pseudo'
            assert self.ph_target in ('pseudo', 'decile'), \
                f"[PH] TIFAD_PH_TARGET must be pseudo|decile, got {self.ph_target}"
            if _ph_src == 'text':
                assert self.text_features.ensemble_prompts is not None, \
                    "[PH-text] requires TIFAD_PROMPT_ENSEMBLE"
                _ph_kt = len(next(iter(
                    self.text_features.ensemble_prompts.values()))) - 1
                assert _ph_kt == _ph_k, \
                    f"[PH-text] text phases ({_ph_kt}) != TIFAD_PH_K ({_ph_k})"
            # TIFAD_PH_FEAT=cls (default: the level-0 cls-trunk feature the
            # class matmul reads) | stem (the embedding-network output
            # BEFORE the text-conditioned stem transformer blocks, stashed
            # by the backbone) | raw (the frozen I3D input features, D =
            # input_dim). TIFAD_PH_WHITEN=seg|global|pca (phaseparse
            # docstring); global / pca statistics from the champion's
            # level-0 train-location dump (probe_dump/<tag>_split{n}_train
            # .npz) when the head is constructed for training.
            self.ph_feat = _osdt.environ.get('TIFAD_PH_FEAT', 'cls') or 'cls'
            assert self.ph_feat in ('cls', 'stem', 'raw'), \
                f"[PH] TIFAD_PH_FEAT must be cls|stem|raw, got {self.ph_feat}"
            _ph_dim = input_dim if self.ph_feat == 'raw' else fpn_dim
            _ph_wh = _osdt.environ.get('TIFAD_PH_WHITEN', '') or None
            # TIFAD_PH_CTR=1 (+ TIFAD_PH_CTR_MODE=branch|joint): phase-chunk
            # supervised contrastive loss (phaseparse.PhaseContrast,
            # attribute `phasectr`); 'branch' adds the 2-block transformer
            # `phasebranch` on the detached stem features whose output the
            # phase head reads; 'joint' puts the projection head on the
            # (non-detached) cls trunk (trunk unfrozen via TIFAD_LR_MULT_REST).
            self.phasectr = None
            self.phasebranch = None
            self.ph_ctr_mode = None
            if _osdt.environ.get('TIFAD_PH_CTR', '') == '1':
                from .phaseparse import PhaseContrast, PhaseBranch
                self.ph_ctr_mode = _osdt.environ.get('TIFAD_PH_CTR_MODE', 'branch') or 'branch'
                assert self.ph_ctr_mode in ('branch', 'joint'), \
                    f"[PH-CTR] mode must be branch|joint, got {self.ph_ctr_mode}"
                self.ph_ctr_w = float(_osdt.environ.get('TIFAD_PH_CTR_W', '1.0') or 0)
                self.phasectr = PhaseContrast(
                    _ph_dim, K=_ph_k,
                    temp=float(_osdt.environ.get('TIFAD_PH_CTR_TEMP', '0.1') or 0.1),
                    div_w=float(_osdt.environ.get('TIFAD_PH_CTR_DIV', '0.1') or 0))
                if self.ph_ctr_mode == 'branch':
                    self.phasebranch = PhaseBranch(
                        _ph_dim, n_head=int(_osdt.environ.get('TIFAD_PH_BRANCH_HEADS', '4') or 4),
                        n_blocks=int(_osdt.environ.get('TIFAD_PH_BRANCH_BLOCKS', '2') or 2))
            self.phasehead = PhaseEvidence(
                _ph_dim, _ph_src, K=_ph_k, whiten=_ph_wh,
                bg_bias=float(_osdt.environ.get('TIFAD_PH_BGBIAS', '0') or 0),
                tau=float(_osdt.environ.get('TIFAD_PH_TAU', '10') or 10),
                probe_path=_osdt.environ.get(
                    'TIFAD_PH_PROBE',
                    'data/phase_pseudo/probe_dump/probe_head_seqchain_split{n}.pt'
                ).format(n=split_num),
                hidden=int(_osdt.environ.get('TIFAD_PH_HIDDEN', '0') or 0)
                or None,
                bg_mode=_osdt.environ.get('TIFAD_PH_BG', '') or None,
                lam=float(_osdt.environ.get('TIFAD_PH_LAM', '1.0') or 0))
            if _ph_wh in ('global', 'pca'):
                _ph_dump = _osdt.environ.get(
                    'TIFAD_PH_WHITEN_DUMP',
                    'data/phase_pseudo/probe_dump/seqchain_split{n}_train.npz'
                ).format(n=split_num)
                assert self.ph_feat == 'cls', \
                    "[PH] global/pca whitening statistics exist for the cls feature only"
                if _osdt.path.isfile(_ph_dump):
                    import numpy as _npph
                    _d = _npph.load(_ph_dump)
                    _X = torch.from_numpy(_d['feat'][_d['level'] == 0].astype('float32'))
                    self.phasehead.set_whiten_stats(_X)
                    print(f"[PH] whitening '{_ph_wh}' statistics from {_ph_dump}: "
                          f"{_X.size(0)} level-0 train locations", flush=True)
                else:
                    print(f"[PH] whitening '{_ph_wh}': dump {_ph_dump} missing -- "
                          f"statistics must come from the checkpoint", flush=True)
            if _ph_wh == 'seg':
                assert _osdt.environ.get('TIFAD_PH_BOUND', '') != '1', \
                    "[PH] seg whitening is not combined with the boundary head"

            if _osdt.environ.get('TIFAD_PH_BOUND', '') == '1':
                self.ph_brhead = PhaseBoundaryHead(
                    K=_ph_k,
                    W=int(_osdt.environ.get('TIFAD_PH_BOUND_W', '32') or 32),
                    hidden=int(_osdt.environ.get('TIFAD_BR_HIDDEN', '256') or 256),
                    ctx=self.ph_ctx,
                    rng=float(_osdt.environ.get('TIFAD_BR_RANGE', '0.5') or 0.5),
                    w=float(_osdt.environ.get('TIFAD_BR_W', '1.0') or 0),
                    n_levels=len(self.fpn_strides),
                    iters=int(_osdt.environ.get('TIFAD_PH_BOUND_ITERS', '1') or 1))
            print(f"[PH] active (phase-parsed extents: src={_ph_src} K={_ph_k} "
                  f"bg={self.phasehead.bg_mode} bg_bias={self.phasehead.bg_bias} "
                  f"lam={self.phasehead.lam} "
                  f"tau={self.phasehead.tau} "
                  f"ctx={self.ph_ctx} maxw={self.ph_maxw}; extent alpha="
                  f"{self.ph_alpha} minllr={self.ph_minllr} compl beta="
                  f"{self.ph_beta} bound={self.ph_brhead is not None} "
                  f"target={self.ph_target} feat={self.ph_feat} dim={_ph_dim} "
                  f"whiten={self.phasehead.whiten} ctr={self.ph_ctr_mode}"
                  + (f" [W={self.ph_brhead.W}/side in_dim={self.ph_brhead.in_dim}"
                     f" iters={self.ph_brhead.iters}]"
                     if self.ph_brhead is not None else '')
                  + (f"; probe={self.phasehead.probe_path}"
                     if _ph_src == 'probe' else '') + ")", flush=True)

        # ---- Boundary arms B2-B4 (2026-09-18, phaseparse.py): BM boundary
        # map + snapping (TIFAD_BM=1 constructs `bmhead`, TIFAD_BM_SNAP=<r>
        # applies it at eval, TIFAD_BM_THR peak threshold 0.3), SHRINK
        # sanity sweep (TIFAD_SHRINK=<f>, eval-only), RE-ANCHORING
        # (TIFAD_REANCHOR=1 post-NMS top-200 | pre = pre-NMS top-200 for
        # the CAND-dump diagnostics). All inert when unset.
        self.bmhead = None
        if _osdt.environ.get('TIFAD_BM', '') == '1':
            from .phaseparse import BoundaryMap
            self.bmhead = BoundaryMap(
                fpn_dim,
                hidden=int(_osdt.environ.get('TIFAD_BM_HIDDEN', '256') or 256),
                neg_ratio=float(_osdt.environ.get('TIFAD_BM_NEG', '3') or 3),
                widen=int(_osdt.environ.get('TIFAD_BM_WIDEN', '1') or 1),
                thr=float(_osdt.environ.get('TIFAD_BM_THR', '0.3') or 0.3))
            self.bm_snap = float(_osdt.environ.get('TIFAD_BM_SNAP', '0') or 0)
            self.bm_w = float(_osdt.environ.get('TIFAD_BM_W', '1.0') or 0)
            print(f"[BM] active (start/end boundary map: widen +-{self.bmhead.widen}, "
                  f"neg {self.bmhead.neg_ratio}:1, thr {self.bmhead.thr}; eval snap "
                  f"r={self.bm_snap})", flush=True)
        self.shrink_f = float(_osdt.environ.get('TIFAD_SHRINK', '0') or 0)
        if self.shrink_f:
            print(f"[SHRINK] eval-only: every candidate length x{self.shrink_f} "
                  f"about its center before soft-NMS", flush=True)
        self.reanchor = _osdt.environ.get('TIFAD_REANCHOR', '') or ''
        assert self.reanchor in ('', '1', 'pre'), \
            f"[REANCHOR] TIFAD_REANCHOR must be 1|pre, got {self.reanchor}"
        self.reanchor_k = int(_osdt.environ.get('TIFAD_REANCHOR_K', '200') or 200)
        if self.reanchor:
            print(f"[REANCHOR] eval-only: top-{self.reanchor_k} candidates "
                  f"{'after' if self.reanchor == '1' else 'BEFORE'} NMS re-read the "
                  f"regression head at the location nearest their center (same "
                  f"level), span = mean(original, re-read)", flush=True)


    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def _vattr_code_matrix(self, class_names):
        """VATTR per-class-list binary code matrix by NAME lookup, cached
        per list (CPU float32). Returns (code (C, V), present (C,) bool):
        classes missing from the attrs json get an all-zero code row and
        present=False (they are skipped by both the aux BCE and the score
        blend; all 20 THUMOS classes are present in the default json)."""
        key = tuple(class_names)
        if key not in self._vattr_code_cache:
            code = torch.zeros(len(class_names), self.vattr_code.size(1))
            present = torch.zeros(len(class_names), dtype=torch.bool)
            for ci, c in enumerate(class_names):
                ji = self._vattr_cid.get(c)
                if ji is None:
                    continue
                code[ci] = self.vattr_code[ji].detach().cpu().float()
                present[ci] = True
            self._vattr_code_cache[key] = (code, present)
        return self._vattr_code_cache[key]

    def _poseev_code_matrix(self, class_names):
        """POSEEV per-class-list binary code matrix by NAME lookup, cached
        per list (CPU float32) -- _vattr_code_matrix mirrored onto the
        POSEEV attrs json. Returns (code (C, V), present (C,) bool):
        classes missing from the json get an all-zero row and
        present=False (skipped by both the facet BCE and the score
        blend)."""
        key = tuple(class_names)
        if key not in self._poseev_code_cache:
            code = torch.zeros(len(class_names), self.poseev_code.size(1))
            present = torch.zeros(len(class_names), dtype=torch.bool)
            for ci, c in enumerate(class_names):
                ji = self._poseev_cid.get(c)
                if ji is None:
                    continue
                code[ci] = self.poseev_code[ji].detach().cpu().float()
                present[ci] = True
            self._poseev_code_cache[key] = (code, present)
        return self._poseev_code_cache[key]

    def _poseev_batch(self, video_list, max_len):
        """Batch the per-item pose tensors (234 x T_i) to (B, 234, max_len)
        with zero right-padding -- mirrors preprocessing(): train items are
        already <= max_seq_len (truncate_feats slices pose with the SAME
        window as feats), eval items may exceed the padded max_len only
        never (eval pads UP to a divisible length), but truncate
        defensively. An item without the key (dataset not POSEEV-gated)
        contributes zeros with a one-time warning. NaN/Inf sanitized: the
        zero-init gate multiplies this branch's scores by 0.0 and
        0.0 * NaN would poison the logits."""
        rows = []
        for x in video_list:
            p = x.get('pose_feats', None)
            if p is None:
                if not self._poseev_warned_nokey:
                    print("[POSEEV] WARNING: item without 'pose_feats' "
                          "(dataset not gated by TIFAD_POSEEV?) -> zeros",
                          flush=True)
                    self._poseev_warned_nokey = True
                p = torch.zeros(self._poseev_dim, 0)
            p = torch.nan_to_num(p.float(), nan=0.0, posinf=0.0,
                                 neginf=0.0)
            if p.size(-1) > max_len:
                p = p[:, :max_len]
            rows.append(F.pad(p, (0, max_len - p.size(-1))))
        return torch.stack(rows, dim=0).to(self.device)

    def _phasehead_batch(self, video_list, max_len, with_targets):
        """PHASEHEAD: batch the per-item vmae_grid tensors (768 x T_i) to
        (B, 768, max_len) with zero right-padding and, when with_targets,
        the per-item phase target vectors ((T_i,) long) to (B, max_len)
        with -1 right-padding (-1 = ignored by both losses, so padding
        and unlabeled steps take the same path). Mirrors _poseev_batch:
        items without the keys (dataset not PHASEHEAD-gated) contribute
        zeros / all -1 with a one-time warning; NaN/Inf sanitized."""
        rows, tgts = [], []
        for x in video_list:
            p = x.get('phase_feats', None)
            if p is None:
                if not self._phasehead_warned_nokey:
                    print("[PHASEHEAD] WARNING: item without "
                          "'phase_feats' (dataset not gated by "
                          "TIFAD_PHASEHEAD?) -> zeros / all -1",
                          flush=True)
                    self._phasehead_warned_nokey = True
                p = torch.zeros(self._phasehead_dim, 0)
            p = torch.nan_to_num(p.float(), nan=0.0, posinf=0.0,
                                 neginf=0.0)
            if p.size(-1) > max_len:
                p = p[:, :max_len]
            rows.append(F.pad(p, (0, max_len - p.size(-1))))
            if with_targets:
                t = x.get('phase_targets', None)
                if t is None:
                    t = torch.full((0,), -1, dtype=torch.long)
                t = t.long()
                if t.size(0) > max_len:
                    t = t[:max_len]
                tgts.append(F.pad(t, (0, max_len - t.size(0)), value=-1))
        feats = torch.stack(rows, dim=0).to(self.device)
        if not with_targets:
            return feats, None
        return feats, torch.stack(tgts, dim=0).to(self.device)

    def _aspec_matrix(self, class_names, device):
        """Binary (C, A) class->attribute incidence matrix for the given
        class list, cached on the module keyed by the class tuple."""
        key = tuple(class_names)
        if getattr(self, '_aspec_mat_key', None) != key:
            missing = [c for c in class_names
                       if c not in self.aspec_class2attrs]
            if missing:
                raise KeyError(
                    "[ASPEC] classes missing from TIFAD_ASPEC_ATTRS: "
                    f"{missing}")
            aid = {a: i for i, a in enumerate(self.aspec_attr_names)}
            M = torch.zeros(len(class_names), len(self.aspec_attr_names))
            for ci, c in enumerate(class_names):
                for a in self.aspec_class2attrs[c]:
                    M[ci, aid[a]] = 1.0
            self._aspec_mat = M
            self._aspec_mat_key = key
        return self._aspec_mat.to(device)

    def _sib_family_groups(self, class_names, fam_map_path, device):
        """Sibling groups shared by SIBCE/SIBCON: for every family with >= 2
        members among `class_names`, the member logit-column indices; plus a
        (C, 2) lookup of (group_id, idx_within_group) per class ((-1, -1) for
        singleton-family classes, which both losses skip). Cached on the
        module keyed by (class tuple, map path), mirroring _aspec_matrix."""
        key = (tuple(class_names), fam_map_path)
        if getattr(self, '_sib_key', None) != key:
            import json as _json
            _map = _json.load(open(fam_map_path))
            _missing = [c for c in class_names if c not in _map]
            if _missing:
                raise KeyError(
                    f"[SIB] classes missing from {fam_map_path}: {_missing}")
            _byfam = {}
            for _ci, _c in enumerate(class_names):
                _byfam.setdefault(_map[_c], []).append(_ci)
            _groups = [torch.tensor(v, dtype=torch.long)
                       for _f, v in sorted(_byfam.items()) if len(v) >= 2]
            _lut = torch.full((len(class_names), 2), -1, dtype=torch.long)
            for _g, _mem in enumerate(_groups):
                for _p, _ci in enumerate(_mem.tolist()):
                    _lut[_ci, 0] = _g
                    _lut[_ci, 1] = _p
            self._sib_groups_cache = _groups
            self._sib_lut = _lut
            self._sib_key = key
            _n_multi = int((_lut[:, 0] >= 0).sum())
            print(f"[SIB] {fam_map_path}: {len(_groups)} multi-member "
                  f"families covering {_n_multi}/{len(class_names)} classes",
                  flush=True)
        return ([g.to(device) for g in self._sib_groups_cache],
                self._sib_lut.to(device))

    def forward(self, video_list):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_inputs, batched_masks = self.preprocessing(video_list)
        batch_size, _, _ = batched_inputs.shape

        # text embedding
        mode ='train' if self.training else 'test'
        # ---- PROTO COLLECT (EXEM-style visual-prototype transfer, part 1):
        # scripts/collect_protos.py runs the model in eval() over the TRAIN
        # loader to harvest seen-class visual prototypes; it needs the SEEN
        # (train) class text set even though self.training is False.
        # Env: TIFAD_PROTO_COLLECT=1. Inert when unset or during training.
        import os as _osp
        if _osp.environ.get('TIFAD_PROTO_COLLECT', '') == '1' and not self.training:
            mode = 'train'
        # ---- TIFAD_SELFTRAIN (STOV-style self-training, scripts/self_train.py):
        # batches of pseudo-labeled TEST videos are marked by the driver with
        # item['st_pseudo']=True and must be classified against the TEST
        # (unseen) class vocabulary even though self.training is True. The
        # driver guarantees homogeneous batches (never mixes seen + pseudo
        # items), so checking video_list[0] is sufficient. Inert when the env
        # is unset (normal runs byte-identical).
        if (_osp.environ.get('TIFAD_SELFTRAIN', '') == '1' and self.training
                and len(video_list) > 0 and isinstance(video_list[0], dict)
                and video_list[0].get('st_pseudo', False)):
            mode = 'test'
        text_embed, new_cls_num = self.text_features(batch_size, mode)

        # ---- VSA_PRE (TIFAD_VSA_PRE=1, see __init__ / libs/modeling/vsa.py
        # + libs/modeling/backbones.py embed()/trunk()): PRE-cross-attention
        # placement of the video-side subtree attention, symmetric to the
        # text side's pre-Trunk phase composition. Two backbone.embed()/
        # trunk() calls per batch:
        #   (1) REFERENCE (no_grad): embed() the raw video once, run trunk()
        #       UNMODIFIED and read the frozen champion's OWN reg_head
        #       offsets at level 0 (T0 resolution == embed()'s output
        #       resolution) -- this is the candidate span TIFAD_VSA_PRE
        #       chunks against. Using the REFINED tensor's own offsets here
        #       would be circular (the span would depend on the very
        #       refinement it is meant to seed), so this reference reads
        #       reg_head on the UNREFINED embedding.
        #   (2) LIVE: vsa_pre refines the (detached) embedded video with
        #       that reference span (reusing VideoSubtreeAttn's chain_attn
        #       + clip_attn exactly as v1, K=10 chunks), then trunk() runs
        #       ONCE, with gradient, on the REFINED video -- so (unlike
        #       TIFAD_VSA=1/v1, which only ever rescoress cls_feat after
        #       Trunk already ran) the refinement reaches cls, reg, fg AND
        #       cn, because it is now part of Trunk's own input.
        # Zero-init out_proj's => vsa_pre is the identity at step 0 => the
        # LIVE trunk() call reduces to exactly the REFERENCE trunk() call,
        # so a freshly-constructed TIFAD_VSA_PRE=1 model reproduces the
        # champion's own unfused numbers exactly (init-equality). Requires
        # no other block between backbone and reg_head (GNN3/DT/PDA/HTCA/
        # LEVELAXIS/CTXATTN/VIDSLOT/TDNECK) -- none are part of the THUMOS
        # seqchain champion recipe this stage trains on; asserted below so
        # a future combination doesn't silently go stale.
        if self.vsa_pre is not None:
            assert all(m is None for m in (
                self.taxgnn3, self.dt_module, self.pda, self.htca_attn,
                self.levelaxis, self.ctxattn, self.vidslot, self.tdneck)), (
                "[VSA-PRE] reference pass only re-runs backbone.embed()+"
                "trunk()+reg_head; incompatible with GNN3/DT/PDA/HTCA/"
                "LEVELAXIS/CTXATTN/VIDSLOT/TDNECK active in the same run "
                "(they would also need to run inside the reference pass)")
            with torch.no_grad():
                _vp_embd, _vp_emask = self.backbone.embed(
                    batched_inputs, batched_masks)
                _vp_rfeat, _vp_rmask, _ = self.backbone.trunk(
                    _vp_embd, text_embed, _vp_emask)
                _vp_roff = self.reg_head(list(_vp_rfeat), list(_vp_rmask))
                _vp_roff = _vp_roff[0].permute(0, 2, 1).detach()  # (B,T0,2)
            _vp_out, _, _ = self.vsa_pre(
                [_vp_embd], [_vp_roff], [_vp_rmask[0]])
            updated_feat, fpn_masks, updated_text = self.backbone.trunk(
                _vp_out[0], text_embed, _vp_emask)
        else:
            updated_feat, fpn_masks, updated_text = self.backbone(batched_inputs, text_embed, batched_masks)
        updated_feat = [x.transpose(2, 1) for x in updated_feat]

        updated_text = updated_text[:, :new_cls_num]

        updated_text_lv = []
        for l in range(len(updated_feat)):
            updated_text_lv.append(updated_text)

        updated_feat = [feat.permute(0, 2, 1) for feat in updated_feat]

        # ---- TaxGNN3 forward (TIFAD_GNN3=1): instance-level taxonomy graph
        # over the CURRENT class set. Train: in-batch GT segments join as
        # LIVE instance nodes (level-0 features mean-pooled inside each
        # segment; <=K per class, per-class dropout p simulates text-only
        # insertion). Eval: text-only graph -- test classes insert exactly
        # like dropped-out train classes. Outputs BOTH zero-init gated
        # residuals: (i) per-class on updated_text (feeds the cls matmul),
        # (ii) per-snippet on updated_feat via graph-conditioned cross-
        # attention (feeds fg/reg/cn heads AND the matmul; gradients to the
        # backbone). Runs BEFORE DT/HTCA. Inert when TIFAD_GNN3 unset.
        if self.taxgnn3 is not None:
            _g3_names = (self.text_features.train_classes if mode == 'train'
                         else self.text_features.test_classes)
            if updated_text.size(1) != len(_g3_names):
                raise RuntimeError(
                    f"[TaxGNN3] text rows ({updated_text.size(1)}) != "
                    f"#classes ({len(_g3_names)}) -- incompatible with "
                    "class-expanding eval switches (e.g. TIFAD_SUBCLASS)")
            _g3_inst_f = _g3_inst_l = _g3_inst_se = None
            if self.training and mode == 'train':
                _f0 = updated_feat[0]                          # (B, D, T0)
                _T0 = _f0.size(-1)
                _by_cls = {}
                for _bi, _v in enumerate(video_list):
                    _segs = _v['segments']
                    _labs = _v['labels']
                    if _segs is None:
                        continue
                    for _si in range(_segs.size(0)):
                        _by_cls.setdefault(int(_labs[_si]), []).append(
                            (_bi, float(_segs[_si, 0]),
                             float(_segs[_si, 1])))
                _fs, _ls, _ses = [], [], []
                for _ci, _items in _by_cls.items():
                    # per-CLASS instance dropout: train the text-only
                    # insertion condition used for unseen classes at eval
                    if float(torch.rand(())) < self.taxgnn3.inst_drop:
                        continue
                    if len(_items) > self.taxgnn3.inst_per_class:
                        _perm = torch.randperm(len(_items))
                        _items = [_items[int(_j)] for _j in
                                  _perm[:self.taxgnn3.inst_per_class]]
                    for _bi, _s, _e in _items:
                        _s0 = min(max(int(_s), 0), _T0 - 1)
                        _e0 = min(max(int(math.ceil(_e)), _s0 + 1), _T0)
                        _fs.append(_f0[_bi, :, _s0:_e0].mean(dim=-1))
                        _ls.append(_ci)
                        _ses.append((_s0 / _T0, _e0 / _T0))
                if _fs:
                    _g3_inst_f = torch.stack(_fs, dim=0)       # (M, D) live
                    _g3_inst_l = torch.tensor(
                        _ls, dtype=torch.long, device=_g3_inst_f.device)
                    _g3_inst_se = torch.tensor(
                        _ses, device=_g3_inst_f.device)
            _g3_act, _g3_nodes, _g3_resid = self.taxgnn3(
                _g3_names, self.text_features.extract_text_emb,
                inst_feats=_g3_inst_f, inst_labels=_g3_inst_l,
                inst_se=_g3_inst_se, training=self.training)
            # (i) class-embedding residual, per-class gated (gate==0.1 at
            # init; residual==0 at init => exact init-equality)
            _g3_gf = self.taxgnn3.gate_feats(
                updated_text, _g3_act.unsqueeze(0).expand(
                    updated_text.size(0), -1, -1))             # (B, C, 3)
            _g3_gate = torch.sigmoid(self.taxgnn3.cls_gate(_g3_gf))
            updated_text = updated_text + _g3_gate * _g3_resid.unsqueeze(0)
            updated_text_lv = [updated_text
                               for _ in range(len(updated_feat))]
            # (ii) snippet residual via graph-conditioned cross-attention
            # over the TEXT-node outputs (same key set at train and eval)
            _g3_kv = _g3_nodes.unsqueeze(0)
            _g3_feats = []
            for _feat, _m in zip(updated_feat, fpn_masks):
                _q = _feat.transpose(1, 2)                     # (B, T_l, D)
                _kv = _g3_kv.expand(_q.size(0), -1, -1)
                _ctx, _ = self.taxgnn3.snip_attn(_q, _kv, _kv,
                                                 need_weights=False)
                _gs = torch.sigmoid(self.taxgnn3.snip_gate(
                    self.taxgnn3.gate_feats(_q, _ctx)))        # (B, T_l, 1)
                _r = self.taxgnn3.snip_head(_ctx) * _gs \
                    * _m.transpose(1, 2).to(_q.dtype)
                _g3_feats.append((_q + _r).transpose(1, 2))
            updated_feat = _g3_feats

        # ---- DT forward (TIFAD_DT=1): TIF+APA per pyramid level. The fused
        # phase features \hat F_v (Eq. 9) REPLACE updated_feat as input to the
        # fg / regression / centerness heads below, and the APA logits replace
        # the label-level text matmul. Inert when TIFAD_DT unset.
        dt_cls_logits = None
        if self.dt_module is not None:
            _dt_cls_names = (self.text_features.train_classes if mode == 'train'
                             else self.text_features.test_classes)
            # DT-FIX(A): hand the DT path the SAME calibration Ti-FAD's stock
            # matmul uses (learned temperature + text-derived prior bias) so
            # the sigmoid focal loss sees logits at the scale/prior it was
            # designed for (stock first-iter cls_loss ~0.5; uncalibrated DT
            # logits produced ~522).
            dt_cls_logits, updated_feat = self.dt_module(
                updated_feat, fpn_masks, self.text_features, _dt_cls_names,
                self.logit_scale, self.bias_lang, self.bias0)

        # ---- PDA forward (TIFAD_PDA=1): per FPN level, TIF gates F_v into
        # four phase streams, APA yields the adaptively weighted phase-wise
        # class logits (REPLACING the label-level text matmul below), and
        # the Eq. 9 MLP over the concatenated streams REPLACES updated_feat
        # as input of the fg / regression / centerness heads. Phase text
        # banks come from text.py's stash (with graph). Inert when unset.
        pda_cls_logits = None
        if self.pda is not None:
            _pda_phase = getattr(self.text_features, '_pda_phase', None)
            assert _pda_phase is not None, "[PDA] phase text stash missing"
            self.text_features._pda_phase = None
            pda_cls_logits, updated_feat, _pda_stats = self.pda(
                updated_feat, fpn_masks, _pda_phase,
                self.logit_scale, self.bias_lang, self.bias0)
            if not getattr(self, '_pda_announced', False):
                self._pda_announced = True
                print(f"[PDA] forward: C={pda_cls_logits[0].size(-1)} "
                      f"levels={len(pda_cls_logits)} "
                      f"T0={pda_cls_logits[0].size(1)} "
                      f"tif_keep={_pda_stats['tif_keep']:.3f} "
                      f"max_prompt_tokens="
                      f"{self.text_features.pda_text.max_tokens}", flush=True)

        # ---- HTCA forward (TIFAD_HTCA=1): each snippet attends over the
        # trained hierarchy anchor vocabulary; the residual (zero-init
        # out_proj => byte-identical at step 0) is added to updated_feat
        # BEFORE the fg/reg/cn heads AND the classification matmul below.
        # HTCA-NOTE: keys/values are the CLASS-AGNOSTIC sub-bank
        # concat(E_attr all V rows, E_fam all F rows) -- flattening the
        # per-class banks (C*N keys) is prohibitive on ANet (C=180+). The
        # full per-class bank is still stashed at
        # self.text_features._htca_bank/_htca_mask. (When TIFAD_DT is on,
        # dt_cls_logits were computed above, so HTCA then only reaches the
        # fg/reg/cn heads.)
        if self.htca_attn is not None:
            _anch = torch.cat([self.text_features.htca_attr_emb.weight,
                               self.text_features.htca_fam_emb.weight],
                              dim=0)                         # (V+F, D)
            _htca_feats = []
            for _feat, _m in zip(updated_feat, fpn_masks):
                _q = _feat.transpose(1, 2)                   # (B, T_l, D)
                _kv = _anch.unsqueeze(0).expand(_q.size(0), -1, -1)
                _r, _ = self.htca_attn(_q, _kv, _kv, need_weights=False)
                # add the residual only at valid positions; exact no-op at
                # init since _r == 0 (init-equality invariant)
                _q = _q + _r * _m.transpose(1, 2).to(_r.dtype)
                _htca_feats.append(_q.transpose(1, 2))
            updated_feat = _htca_feats

        # ---- LEVELAXIS forward (TIFAD_LEVELAXIS=1): scale attention over
        # the L=6 pyramid levels at every timestep, per-level residual
        # (see LevelAxisScaleAttn). Runs AFTER all feature-residual blocks
        # (GNN3/DT/HTCA), BEFORE the heads; both cls and reg sides see the
        # refined pyramid. Deltas are masked; exactly 0 at init (zero-init
        # out_proj + level embeddings). Inert when TIFAD_LEVELAXIS unset.
        if self.levelaxis is not None:
            updated_feat = self.levelaxis(updated_feat, fpn_masks)

        # ---- CTXATTN forward (TIFAD_CTXATTN=1): semantic-hierarchy
        # context attention (see SemCtxAttn). Bank = cached no_grad CPU
        # text embs (CONFGATE precedent): activity rows built ONCE from
        # "a video of <family> activity"; action rows built PER CLASS-LIST
        # tuple from the class prompts ("a video of action <c>" via
        # get_prompt -- same source as the base pipeline). The trainable
        # projection / level embedding / attention stay LIVE. Placement:
        # alongside LEVELAXIS, BEFORE the TDNECK split -- this is
        # cls-flavored evidence, and refining updated_feat here means the
        # TDNECK sem (cls) pyramid is built FROM the refined features
        # (simplest placement that keeps the cls routing consistent); the
        # reg side also sees it in v1. Deltas masked; exactly 0 at init
        # (zero-init out_proj). Inert when TIFAD_CTXATTN unset.
        if self.ctxattn is not None:
            _cx_names = (self.text_features.train_classes if mode == 'train'
                         else self.text_features.test_classes)
            if self._ctxattn_fam_cache is None:
                with torch.no_grad():
                    self._ctxattn_fam_cache = (
                        self.text_features.extract_text_emb(
                            [f'a video of {f} activity'
                             for f in self._ctxattn_fams],
                            is_prompt=False).detach().cpu())
            _cx_key = tuple(_cx_names)
            if _cx_key not in self._ctxattn_act_cache:
                with torch.no_grad():
                    self._ctxattn_act_cache[_cx_key] = (
                        self.text_features.extract_text_emb(
                            list(_cx_names),
                            is_prompt=True).detach().cpu())
            _cx_dev = updated_feat[0].device
            _cx_dt = updated_feat[0].dtype
            _cx_bank = torch.cat(
                [self._ctxattn_fam_cache,
                 self._ctxattn_act_cache[_cx_key]],
                dim=0).to(device=_cx_dev, dtype=_cx_dt)
            updated_feat = self.ctxattn(
                updated_feat, fpn_masks, _cx_bank,
                n_activity=len(self._ctxattn_fams))

        # ---- VIDSLOT forward (TIFAD_VIDSLOT=<K>): temporal slot
        # discovery (see TemporalSlotAttn) -- slots gather level-0
        # context, every level scatters it back as a masked residual.
        # Placement: after CTXATTN, BEFORE the TDNECK split (both cls and
        # reg sides see it, v1). Exactly 0 at init (zero-init MHA-2
        # out_proj). Inert when TIFAD_VIDSLOT unset.
        if self.vidslot is not None:
            updated_feat = self.vidslot(updated_feat, fpn_masks)

        # ---- TDNECK forward (TIFAD_TDNECK=1): top-down semantic pyramid
        # (see TopDownSemNeck). cls_feat is what the CLASSIFICATION side
        # consumes: the base text matmul plus every block that feeds /
        # rescales the cls logits from video features (ASPEC head, VPW,
        # PHASELOGIT, ATTRVIDATTN, CONFGATE, VATTR blend, PROTO stash +
        # blend, SIBCON, VATTR aux feats, CCR prior, FINEQ) -- they all
        # read cls_feat below so their evidence stays consistent with the
        # logits they modify. The reg/fg/cn heads, HAF, and the
        # feature-space aux pulls (VIDPHASE/VIDACT) keep the RAW
        # updated_feat (task decoupling). When TDNECK is off, cls_feat is
        # the SAME list object (no copy, byte-inert); sem == updated_feat
        # exactly at init (zero-init gates).
        cls_feat = updated_feat
        if self.tdneck is not None:
            cls_feat = self.tdneck(updated_feat, fpn_masks)

        # ---- HAF logits (TIFAD_HAF=<w>): per FPN level, cosine sims of
        # snippet features against the trained hierarchy anchors, logsumexp
        # over the anchor axis -> foreground-evidence logit (B, T_l).
        # Train: auxiliary focal loss in losses(). Eval: blended into out_fg
        # ONLY when TIFAD_HAF_EVAL is nonzero (default 0 = train-only).
        haf_logits = None
        _haf_eval = float(_osp.environ.get('TIFAD_HAF_EVAL', '0') or 0)
        if ((self.haf_w != 0 and self.training)
                or (_haf_eval != 0 and not self.training
                    and self.text_features.htca_attr_emb is not None)):
            _anch_n = F.normalize(
                torch.cat([self.text_features.htca_attr_emb.weight,
                           self.text_features.htca_fam_emb.weight], dim=0),
                dim=-1)                                      # (V+F, D)
            # centered by log(#anchors) so the init logit ~ 0 (sigmoid 0.5)
            # instead of ~log(V+F) (sigmoid ~0.97) -- HAF-FIX for the focal
            # loss miscalibration that sank the first A/B.
            _haf_c = math.log(_anch_n.size(0))
            haf_logits = [
                torch.logsumexp(
                    F.normalize(_feat.transpose(1, 2), dim=-1) @ _anch_n.t(),
                    dim=-1) - _haf_c                         # (B, T_l)
                for _feat in updated_feat
            ]

        # foreground (out_fg)
        out_fg = self.fg_head(updated_feat, fpn_masks)
        out_fg = [x.permute(0, 2, 1) for x in out_fg]

        # ---- HAF eval blend: out_fg += TIFAD_HAF_EVAL * haf_logit.
        # Inert unless TIFAD_HAF_EVAL is set nonzero at eval.
        if haf_logits is not None and not self.training and _haf_eval != 0:
            out_fg = [f + _haf_eval * h.unsqueeze(-1)
                      for f, h in zip(out_fg, haf_logits)]

        # regressgion (out_offsets)
        out_offsets = self.reg_head(updated_feat, fpn_masks)      
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]

        # centerness (out_centerness)
        out_centerness = self.cn_head(updated_feat, fpn_masks)
        out_centerness = [x.permute(0, 2, 1) for x in out_centerness]

        # ---- VSA forward (TIFAD_VSA=1, see __init__ / libs/modeling/vsa.py):
        # per location, pool the candidate segment [n - d_s, n + d_e]
        # (DETACHED offsets, level grid) into K ordered chunks, banded
        # neighbour self-attention among them, then the location's own
        # feature attends over [f_n; chunks]. The result REPLACES cls_feat
        # for everything downstream on the classification side (the base
        # cosine matmul and every cls-side block that reads cls_feat --
        # TDNECK routing convention), in BOTH train and eval, on every
        # level and location. updated_feat (reg/fg/cn, HAF, feature-space
        # aux pulls) is untouched: a NEW list is returned. Exactly
        # cls_feat at init (zero-init out_proj's); inert when unset.
        # V2: `_vsa_in` keeps the PRE-VSA cls pyramid (the tensor the
        # SOFTPHASE head reads; the leaf loss is fed the same one below),
        # `_vsa_match` the XM matching terms F x (B, T_l, C) (None when
        # XM is off), `_vsa_ph` the text phase bank of the CURRENT
        # vocabulary calibrated EXACTLY like the class rows in the base
        # matmul (L2-normalize, text_proj(x/2)) -- same class order as the
        # logit columns (asserted below when the term is added).
        _vsa_in = cls_feat
        _vsa_match = None
        _vsa_align = None
        # ---- QH phase bank (TIFAD_QH_PHASEMAP=1): the SAME ordered text
        # phase stash the VSA-XM term reads (text.py sets it for either
        # env), calibrated EXACTLY like the class rows of the base matmul
        # (L2-normalize, text_proj(x/2)); rows in cls_name == logit-column
        # order. Read BEFORE the VSA hook (which consumes the stash);
        # DETACHED (the head never trains the text side). None otherwise.
        # ---- PH-text bank (TIFAD_PH_SRC=text): the same stash, calibrated
        # the same way, DETACHED; centered / normalized inside the phase
        # evidence module. Read first; consumed only when no other reader
        # (VSA-XM / QH-PHASEMAP) follows. None for the probe / head sources.
        _ph_bank = None
        if self.phasehead is not None and self.phasehead.src == 'text':
            _ph_lp = getattr(self.text_features, '_vsa_phase_embs', None)
            assert _ph_lp is not None, \
                "[PH-text] phase stash missing (text.py ensemble path)"
            if not ((self.qhead is not None and self.qhead.phasemap)
                    or (self.vsa is not None and self.vsa.xm)):
                self.text_features._vsa_phase_embs = None   # consume
            with torch.no_grad():
                _ph_bank = self.text_proj(
                    _ph_lp / _ph_lp.norm(dim=-1, keepdim=True) / 2.0
                ).detach()                                   # (C, K, D)
        _qh_ph = None
        if self.qhead is not None and self.qhead.phasemap:
            _qh_lp = getattr(self.text_features, '_vsa_phase_embs', None)
            assert _qh_lp is not None, \
                "[QH-PHASEMAP] phase stash missing (text.py ensemble path)"
            if self.vsa is None or not self.vsa.xm:
                self.text_features._vsa_phase_embs = None   # consume
            with torch.no_grad():
                _qh_ph = self.text_proj(
                    _qh_lp / _qh_lp.norm(dim=-1, keepdim=True) / 2.0
                ).detach()                                   # (C, K, D)
        if self.vsa is not None:
            _vsa_ph = None
            if self.vsa.xm:
                _vsa_lp = getattr(self.text_features, '_vsa_phase_embs', None)
                assert _vsa_lp is not None, \
                    "[VSA-XM] phase stash missing (text.py ensemble path)"
                self.text_features._vsa_phase_embs = None   # consume
                _vsa_ph = self.text_proj(
                    _vsa_lp / _vsa_lp.norm(dim=-1, keepdim=True) / 2.0)
            cls_feat, _vsa_match, _vsa_align = self.vsa(
                cls_feat, [o.detach() for o in out_offsets], fpn_masks,
                phase_embs=_vsa_ph,
                phase_head=(self.leaf_head if self.vsa.softphase else None),
                want_align=(self.training and mode == 'train'
                            and getattr(self, 'vsa_align_w', 0) != 0))

        # ---- ASPEC forward (TIFAD_ASPEC_ATTRS set): per-snippet attribute
        # logits from the shared specialist head on every FPN level of
        # updated_feat. Computed only when consumed: train loss (TIFAD_ASPEC
        # != 0), eval composition (TIFAD_ASPEC_ALPHA != 0), or the diag
        # stash (TIFAD_ASPEC_COLLECT=1). Inert when TIFAD_ASPEC_ATTRS unset.
        aspec_logits = None
        _aspec_alpha = float(_osp.environ.get('TIFAD_ASPEC_ALPHA', '0') or 0)
        _aspec_collect = (
            _osp.environ.get('TIFAD_ASPEC_COLLECT', '') == '1')
        if self.aspec_head is not None and (
                (self.training and self.aspec_w != 0)
                or (not self.training
                    and (_aspec_alpha != 0 or _aspec_collect))):
            # TDNECK: attribute evidence composes into cls logits -> sem
            aspec_logits = self.aspec_head(cls_feat, fpn_masks)
            # F x (B, A, T_l) -> F x (B, T_l, A)
            aspec_logits = [x.permute(0, 2, 1) for x in aspec_logits]

        # classification (out_cls_logits)
        updated_text_lv = [text / text.norm(dim=-1, keepdim=True) for text in updated_text_lv]
        updated_text_bias_lv = [torch.matmul(text, self.bias_lang) + self.bias0 for text in updated_text_lv]        
        updated_text_lv = [self.text_proj(text / 2.0) for text in updated_text_lv]
        for i, text in enumerate(updated_text_lv):
            if torch.isnan(text).any():
                print(f"updated_text_lv[{i}] contains NaN!")

        # ---- SIBORTH (TIFAD_SIBORTH=<beta>, EVAL ONLY): nearest-sibling
        # anchor decorrelation on the CALIBRATED class text matrix -- the
        # exact (B, C, D) per-level vectors the base matmul below consumes
        # (post row-norm + text_proj). For each class i, find its most
        # cosine-similar sibling j among the ORIGINAL anchors and subtract
        # beta times the projection of c_i onto unit(c_j); all projections
        # use the ORIGINAL anchors (no cascade). The corrected row is
        # rescaled back to c_i's ORIGINAL norm, so the change is purely
        # directional (matmul logit scale / calibration intact); the bias
        # (computed above from the pre-projection text) is untouched. NOT
        # cached per class-list: the backbone's TiCA cross-attention makes
        # these anchors video-conditioned (batch-dependent), so a cache
        # would freeze the first video's anchors -- computed per forward
        # instead (B=1, C~20 at eval: negligible). Byte-inert when unset
        # or 0, and during training.
        _sib_beta = float(_osp.environ.get('TIFAD_SIBORTH', '0') or 0)
        if _sib_beta != 0 and not self.training:
            if not getattr(self, '_siborth_announced', False):
                print(f"[SIBORTH] active (beta={_sib_beta}, eval-only "
                      f"anchor decorrelation)", flush=True)
                self._siborth_announced = True
            with torch.no_grad():
                _sib_out = []
                for _sib_t in updated_text_lv:               # (B, C, D)
                    _sib_norm = _sib_t.norm(dim=-1, keepdim=True)
                    _sib_hat = _sib_t / _sib_norm.clamp(min=1e-6)
                    _sib_cos = _sib_hat @ _sib_hat.transpose(1, 2)  # (B,C,C)
                    _sib_C = _sib_cos.size(-1)
                    _sib_cos = _sib_cos - 3.0 * torch.eye(
                        _sib_C, device=_sib_cos.device,
                        dtype=_sib_cos.dtype).unsqueeze(0)
                    _sib_j = _sib_cos.argmax(dim=-1)         # (B, C)
                    _sib_jhat = torch.gather(
                        _sib_hat, 1,
                        _sib_j.unsqueeze(-1).expand(
                            -1, -1, _sib_hat.size(-1)))      # (B, C, D)
                    _sib_proj = (_sib_t * _sib_jhat).sum(
                        -1, keepdim=True)                    # (B, C, 1)
                    _sib_new = _sib_t - _sib_beta * _sib_proj * _sib_jhat
                    _sib_new = _sib_new * (
                        _sib_norm
                        / _sib_new.norm(dim=-1, keepdim=True).clamp(
                            min=1e-6))
                    _sib_out.append(_sib_new)
                updated_text_lv = _sib_out

        # ---- PROTO COLLECT (part 2): stash the level-0 features and the
        # projected per-class text matrix / bias exactly as used in the matmul
        # below, so scripts/collect_protos.py can read them off the module.
        # Env: TIFAD_PROTO_COLLECT=1. Inert when unset or during training.
        if _osp.environ.get('TIFAD_PROTO_COLLECT', '') == '1' and not self.training:
            self._proto_stash = {
                'feat0': cls_feat[0].detach(),                # (B, D, T0)
                'text0': updated_text_lv[0].detach(),         # (B, C, D) projected
                'bias0': updated_text_bias_lv[0].detach(),    # (B, C)
            }

        assert _vsa_match is None or (
            dt_cls_logits is None and pda_cls_logits is None), \
            "[VSA-XM] incompatible with TIFAD_DT / TIFAD_PDA logits"
        if dt_cls_logits is not None:
            # ---- DT: APA adaptive phase-wise logits (Eq. 5-7) replace the
            # label-level text matmul. Ti-FAD's focal cls loss / inference
            # consume them unchanged (Eq. 8 realized via existing plumbing).
            out_cls_logits = dt_cls_logits
        elif pda_cls_logits is not None:
            # ---- PDA: adaptive phase-wise logits C_cls (Eq. 5-7) replace
            # the label-level text matmul; the fork's focal cls loss /
            # inference consume them unchanged.
            out_cls_logits = pda_cls_logits
        else:
            # TDNECK: the cls matmul reads the sem pyramid (cls_feat ==
            # updated_feat when TIFAD_TDNECK unset)
            updated_feat_text_matmul = [
                (feat.transpose(2, 1) @ text.transpose(2, 1) / self.logit_scale.exp()) + bias.unsqueeze(1).repeat(1, feat.size(-1), 1)
                for feat, text, bias in zip(cls_feat, updated_text_lv, updated_text_bias_lv)
            ]

            out_cls_logits = updated_feat_text_matmul

            # ---- VSA-XM term (TIFAD_VSA_XM=1, see vsa.py (A)): logits +=
            # lam * tau * m with tau(n, c) = ||v_n|| ||p_c|| / scale, the
            # (detached) factor multiplying the base cosine at (n, c) --
            # the base logit above is <v_n, p_c> / scale + bias -- so the
            # matching cosine m enters in base-cosine units. lam is
            # ZERO-init (vsa.xm_lam) => exact init-equality. Same class
            # order as the text rows (the stash is built from the same
            # cls_name list as text_emb; asserted on the column count).
            if _vsa_match is not None:
                _xm_out = []
                for _xl, _xm, _xf, _xt in zip(
                        out_cls_logits, _vsa_match, cls_feat, updated_text_lv):
                    if _xm is None:                 # MINLVL pass-through
                        _xm_out.append(_xl)
                        continue
                    assert _xm.size(-1) == _xl.size(-1), \
                        (f"[VSA-XM] phase-bank rows ({_xm.size(-1)}) != "
                         f"logit columns ({_xl.size(-1)}) -- incompatible "
                         "with class-expanding switches")
                    with torch.no_grad():
                        _xtau = (_xf.norm(dim=1).unsqueeze(-1)
                                 * _xt.norm(dim=-1).unsqueeze(1)
                                 / self.logit_scale.exp())      # (B, T, C)
                    _xm_out.append(_xl + self.vsa.xm_lam * _xtau * _xm)
                out_cls_logits = _xm_out

        # ---- VPW forward (TIFAD_VPW=1): video-conditioned phase weighting.
        # Placed BEFORE the SUBCLASS / LEVEL-COMPOSE pools so the phase bank
        # rows (one per class in text.py's cls_name, INCLUDING a level-
        # compose expansion) align with the logit columns; incompatible with
        # TIFAD_SUBCLASS (rows would mismatch -- asserted below). Residual
        # scaled by vpw_alpha (init 0) => exact init-equality.
        if self.vpw_gate is not None:
            _vpw_per = getattr(self.text_features, '_vpw_per', None)
            assert _vpw_per is not None, \
                "[VPW] phase stash missing (text.py ensemble path not run)"
            self.text_features._vpw_per = None          # consume the stash
            _vpw_ph = _vpw_per[:, 1:, :]                # (C, K, D) phases
            assert _vpw_ph.size(1) == self.vpw_k, \
                f"[VPW] K mismatch: {_vpw_ph.size(1)} vs {self.vpw_k}"
            assert _vpw_ph.size(0) == out_cls_logits[0].size(-1), \
                (f"[VPW] phase-bank rows ({_vpw_ph.size(0)}) != logit "
                 f"columns ({out_cls_logits[0].size(-1)}) -- incompatible "
                 "with class-expanding switches (e.g. TIFAD_SUBCLASS)")
            # project phases EXACTLY like the class embeddings in the base
            # matmul: L2-normalize, text_proj(x/2), score / logit_scale.exp()
            # (no bias term -- the residual rides on the biased base logit)
            _pn = _vpw_ph / _vpw_ph.norm(dim=-1, keepdim=True)
            _pp = self.text_proj(_pn / 2.0)             # (C, K, D)
            _vscale = self.logit_scale.exp()
            _new_lg = []
            for _feat, _lg in zip(cls_feat, out_cls_logits):
                _q = _feat.transpose(2, 1)              # (B, T_l, D)
                _w = torch.softmax(self.vpw_gate(_q), dim=-1)  # (B, T_l, K)
                # K matmuls of (B,T,D)@(D,C) instead of one (B,T,C,K)
                # einsum: same result, no 4-D intermediate
                _v = None
                for _k in range(self.vpw_k):
                    _sk = (_q @ _pp[:, _k, :].t()) / _vscale   # (B, T_l, C)
                    _ck = _w[..., _k].unsqueeze(-1) * _sk
                    _v = _ck if _v is None else _v + _ck
                _new_lg.append(_lg + self.vpw_alpha * _v)
            out_cls_logits = _new_lg

        # ---- PHASELOGIT forward (TIFAD_PHASELOGIT=1): score-space phase
        # composition, see __init__. Placed like VPW -- BEFORE the SUBCLASS
        # / LEVEL-COMPOSE pools -- so the phase-bank rows align with the
        # logit columns (asserted; incompatible with class-expanding
        # switches). The softmax over phases is computed ONLINE (running
        # max / numerator / denominator), so the cost stays K matmuls of
        # (B, T_l, C) like VPW -- no 4-D (B, T, C, K) tensor is formed.
        if self.phaselogit_lambda is not None:
            _pl_per = getattr(self.text_features, '_phaselogit_per', None)
            assert _pl_per is not None, \
                "[PHASELOGIT] phase stash missing (text.py ensemble path " \
                "not run)"
            self.text_features._phaselogit_per = None   # consume the stash
            _pl_ph = _pl_per[:, 1:, :]                  # (C, K, D) phases
            assert _pl_ph.size(1) == self.phaselogit_k, \
                f"[PHASELOGIT] K mismatch: {_pl_ph.size(1)} vs " \
                f"{self.phaselogit_k}"
            assert _pl_ph.size(0) == out_cls_logits[0].size(-1), \
                (f"[PHASELOGIT] phase-bank rows ({_pl_ph.size(0)}) != logit "
                 f"columns ({out_cls_logits[0].size(-1)}) -- incompatible "
                 "with class-expanding switches (e.g. TIFAD_SUBCLASS)")
            # project phases EXACTLY like the class embeddings in the base
            # matmul: L2-normalize, text_proj(x/2), score / logit_scale.exp()
            _pl_n = _pl_ph / _pl_ph.norm(dim=-1, keepdim=True)
            _pl_p = self.text_proj(_pl_n / 2.0)         # (C, K, D)
            _pl_scale = self.logit_scale.exp()
            _pl_gate = torch.tanh(self.phaselogit_lambda)
            _pl_tau = self.phaselogit_tau
            _new_lg = []
            for _feat, _lg in zip(cls_feat, out_cls_logits):
                _q = _feat.transpose(2, 1)              # (B, T_l, D)
                _pl_m = _pl_num = _pl_den = None
                for _k in range(self.phaselogit_k):
                    _sk = (_q @ _pl_p[:, _k, :].t()) / _pl_scale  # (B,T,C)
                    _lk = _sk / _pl_tau
                    if _pl_m is None:                   # online softmax init
                        _pl_m = _lk
                        _pl_den = torch.ones_like(_sk)
                        _pl_num = _sk
                    else:
                        _pl_mn = torch.maximum(_pl_m, _lk)
                        _pl_sc = torch.exp(_pl_m - _pl_mn)
                        _pl_e = torch.exp(_lk - _pl_mn)
                        _pl_den = _pl_den * _pl_sc + _pl_e
                        _pl_num = _pl_num * _pl_sc + _pl_e * _sk
                        _pl_m = _pl_mn
                _new_lg.append(_lg + _pl_gate * (_pl_num / _pl_den))
            out_cls_logits = _new_lg

        # ---- ATTRVIDATTN forward (TIFAD_ATTRVIDATTN=1): score-space
        # present-vs-absent attribute contrast, see __init__. Placed like
        # VPW/PHASELOGIT -- BEFORE the SUBCLASS / LEVEL-COMPOSE pools -- so
        # the per-class contrast columns align with the logit columns
        # (asserted; incompatible with class-expanding switches).
        if self.attrvid_gamma is not None:
            _av_names = (self.text_features.train_classes if mode == 'train'
                         else self.text_features.test_classes)
            # (a) raw CLIP embeddings of the attribute-value names
            # (underscores -> spaces), cached no_grad on CPU (CONFGATE
            # precedent); calibration below stays LIVE (text_proj /
            # logit_scale train).
            if self._attrvid_val_cache is None:
                with torch.no_grad():
                    self._attrvid_val_cache = (
                        self.text_features.extract_text_emb(
                            [v.replace('_', ' ')
                             for v in self._attrvid_vocab],
                            is_prompt=False).detach().cpu())
            # (b) per-class present/absent contrast matrix (V, C), cached
            # per class list: +1/|present_c| on present rows, -1/|absent_c|
            # on absent rows, all-zero column for classes not in the json.
            _av_key = tuple(_av_names)
            if _av_key not in self._attrvid_w_cache:
                _av_V = len(self._attrvid_vocab)
                _av_W = torch.zeros(_av_V, len(_av_names))
                for _av_i, _av_c in enumerate(_av_names):
                    _av_pres = self._attrvid_present.get(_av_c)
                    if not _av_pres:
                        continue
                    _av_pm = torch.zeros(_av_V, dtype=torch.bool)
                    _av_pm[_av_pres] = True
                    _av_W[_av_pm, _av_i] = 1.0 / int(_av_pm.sum())
                    _av_na = int((~_av_pm).sum())
                    if _av_na > 0:
                        _av_W[~_av_pm, _av_i] = -1.0 / _av_na
                self._attrvid_w_cache[_av_key] = _av_W
            _av_W = self._attrvid_w_cache[_av_key]
            assert _av_W.size(1) == out_cls_logits[0].size(-1), \
                (f"[ATTRVIDATTN] class count ({_av_W.size(1)}) != logit "
                 f"columns ({out_cls_logits[0].size(-1)}) -- incompatible "
                 "with class-expanding switches (e.g. TIFAD_SUBCLASS)")
            _av_dev = out_cls_logits[0].device
            _av_dt = out_cls_logits[0].dtype
            # calibrate the value embeddings EXACTLY like the class
            # embeddings in the base matmul (VPW recipe): L2-normalize,
            # text_proj(x/2), score / logit_scale.exp() (no bias -- the
            # residual rides on the biased base logit)
            _av_e = self._attrvid_val_cache.to(device=_av_dev, dtype=_av_dt)
            _av_n = _av_e / _av_e.norm(dim=-1, keepdim=True)
            _av_p = self.text_proj(_av_n / 2.0)                  # (V, D)
            _av_scale = self.logit_scale.exp()
            _av_g = torch.tanh(self.attrvid_gamma)
            _av_Wd = _av_W.to(device=_av_dev, dtype=_av_dt)
            _new_lg = []
            for _feat, _lg in zip(cls_feat, out_cls_logits):
                _av_s = (_feat.transpose(2, 1) @ _av_p.t()) / _av_scale
                _new_lg.append(_lg + _av_g * (_av_s @ _av_Wd))   # (B,T,C)
            out_cls_logits = _new_lg

        # ---- CONFGATE forward (TIFAD_CONFGATE set): confusability-adaptive
        # family-axis blend, see __init__. Placed like VPW -- BEFORE the
        # SUBCLASS / LEVEL-COMPOSE pools -- so the per-class conf weights
        # align with the logit columns (asserted; incompatible with
        # class-expanding switches). Train AND eval scoring.
        if self.confgate_alpha is not None:
            _cg_names = (self.text_features.train_classes if mode == 'train'
                         else self.text_features.test_classes)
            # (a) family raw CLIP embeddings: same text pathway as class
            # names (text_features.extract_text_emb), cached no_grad on CPU;
            # calibration below stays LIVE (text_proj / logit_scale train).
            if self._confgate_fam_cache is None:
                with torch.no_grad():
                    self._confgate_fam_cache = (
                        self.text_features.extract_text_emb(
                            [f'a video of {f} activity'
                             for f in self._confgate_fams],
                            is_prompt=False).detach().cpu())
            # (b) per-class confusability over the CURRENT class list:
            # cosine to the nearest OTHER class (raw CLIP name embs),
            # min-max normalized to [0,1]; cached per class-list. Classes
            # missing from the fam map get weight 0 (skipped).
            _cg_key = tuple(_cg_names)
            if _cg_key not in self._confgate_conf_cache:
                with torch.no_grad():
                    _ce = self.text_features.extract_text_emb(
                        list(_cg_names), is_prompt=False).detach().cpu()
                    _cn = _ce / _ce.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    _csim = _cn @ _cn.t()
                    _csim.fill_diagonal_(-2.0)
                    _conf = _csim.max(-1).values                    # (C,)
                    _conf = ((_conf - _conf.min())
                             / (_conf.max() - _conf.min()).clamp(min=1e-6))
                _cg_fidx, _cg_msk = [], []
                for _c in _cg_names:
                    _f = self._confgate_map.get(_c)
                    _cg_fidx.append(0 if _f is None else self._confgate_fid[_f])
                    _cg_msk.append(0.0 if _f is None else 1.0)
                self._confgate_conf_cache[_cg_key] = (
                    _conf, torch.as_tensor(_cg_fidx, dtype=torch.long),
                    torch.as_tensor(_cg_msk, dtype=_conf.dtype))
            _conf, _cg_fidx, _cg_msk = self._confgate_conf_cache[_cg_key]
            assert _conf.numel() == out_cls_logits[0].size(-1), \
                (f"[CONFGATE] class count ({_conf.numel()}) != logit columns "
                 f"({out_cls_logits[0].size(-1)}) -- incompatible with "
                 "class-expanding switches (e.g. TIFAD_SUBCLASS)")
            _cg_dev = out_cls_logits[0].device
            _cg_dt = out_cls_logits[0].dtype
            _cg_w = (torch.tanh(self.confgate_alpha)
                     * (_conf * _cg_msk).to(device=_cg_dev, dtype=_cg_dt))
            _cg_fidx = _cg_fidx.to(_cg_dev)
            # calibrate family embs EXACTLY like the class embeddings in the
            # base matmul (VPW recipe): L2-norm, text_proj(x/2), /scale
            _cg_fe = self._confgate_fam_cache.to(device=_cg_dev, dtype=_cg_dt)
            _cg_fn = _cg_fe / _cg_fe.norm(dim=-1, keepdim=True)
            _cg_fp = self.text_proj(_cg_fn / 2.0)                   # (F, D)
            _cg_scale = self.logit_scale.exp()
            _new_lg = []
            for _feat, _lg in zip(cls_feat, out_cls_logits):
                _cg_fl = (_feat.transpose(2, 1) @ _cg_fp.t()) / _cg_scale
                # (B, T_l, F) -> gather each class's family column -> (B,T,C)
                _new_lg.append(
                    _lg + _cg_w.view(1, 1, -1) * _cg_fl[..., _cg_fidx])
            out_cls_logits = _new_lg

        # ---- VATTR forward (TIFAD_VATTR=1): score-space attribute-
        # COMBINATION blend, see __init__. Train AND eval scoring. Per FPN
        # level, a = sigmoid(vattr_head(feat)) (B, T_l, V); class score
        # s_c = <a, code_c_centered> / ||code_c_centered||_1 with codes
        # centered by the COLUMN mean over the CURRENT class list (train =
        # seen classes, eval = test classes; per-list (V, C) weights cached
        # on CPU). Blend: logit_c += tanh(vattr_gate) * s_c -- gate init 0
        # => EXACT init-equality. Placed like VPW/PHASELOGIT -- BEFORE the
        # SUBCLASS / LEVEL-COMPOSE pools -- so the per-class columns align
        # with the logit columns (asserted; incompatible with
        # class-expanding switches). Classes missing from the attrs json
        # get a zero score column (skipped; centering over PRESENT rows).
        if self.vattr_head is not None:
            _va_names = (self.text_features.train_classes if mode == 'train'
                         else self.text_features.test_classes)
            _va_key = tuple(_va_names)
            if _va_key not in self._vattr_score_cache:
                _va_code, _va_pres = self._vattr_code_matrix(list(_va_names))
                if bool(_va_pres.any()):
                    _va_mean = _va_code[_va_pres].mean(dim=0, keepdim=True)
                else:
                    _va_mean = torch.zeros(1, _va_code.size(1))
                _va_cc = _va_code - _va_mean                     # (C, V)
                _va_cc[~_va_pres] = 0.0                          # skip missing
                _va_den = _va_cc.abs().sum(dim=1).clamp(min=1e-6)
                self._vattr_score_cache[_va_key] = (
                    (_va_cc / _va_den.unsqueeze(1)).t().contiguous())  # (V, C)
            _va_W = self._vattr_score_cache[_va_key]
            assert _va_W.size(1) == out_cls_logits[0].size(-1), \
                (f"[VATTR] class count ({_va_W.size(1)}) != logit "
                 f"columns ({out_cls_logits[0].size(-1)}) -- incompatible "
                 "with class-expanding switches (e.g. TIFAD_SUBCLASS)")
            _va_Wd = _va_W.to(device=out_cls_logits[0].device,
                              dtype=out_cls_logits[0].dtype)
            _va_g = torch.tanh(self.vattr_gate)
            _new_lg = []
            for _feat, _lg in zip(cls_feat, out_cls_logits):
                _va_a = torch.sigmoid(
                    self.vattr_head(_feat.transpose(2, 1)))      # (B, T_l, V)
                _new_lg.append(_lg + _va_g * (_va_a @ _va_Wd))   # (B, T_l, C)
            out_cls_logits = _new_lg

        # ---- POSEEV forward (TIFAD_POSEEV=1, see __init__): pose
        # evidence branch, train AND eval scoring. Encode the batched pose
        # tensor once at full grid resolution, strided-max-pool it to every
        # pyramid length (kernel=stride=fpn_strides[l] -- the pose tensor
        # rides the level-0 input grid, so this reproduces the pyramid's
        # temporal geometry), read attribute logits, and blend the
        # facet-restricted VATTR-style class score into the cls logits
        # through the zero-init tanh gate (EXACT init-equality). Placed
        # with the VATTR blend -- BEFORE the SUBCLASS / LEVEL-COMPOSE
        # pools -- so the per-class columns align with the logit columns
        # (asserted). The level-concatenated attribute logits are stashed
        # for the train aux BCE in losses() (same level order as
        # fpn_masks, so pos_mask indexes them directly). Inert when unset.
        _poseev_logits_cat = None
        if self.poseev_enc is not None:
            _pe_dt = out_cls_logits[0].dtype
            _pe_pose = self._poseev_batch(
                video_list, batched_inputs.size(-1)).to(_pe_dt)
            _pe_enc = self.poseev_enc(_pe_pose)              # (B, D, T)
            _pe_lgts = []
            for _pe_l, _pe_m in enumerate(fpn_masks):
                _pe_s = self.fpn_strides[_pe_l]
                _pe_e = (F.max_pool1d(_pe_enc, kernel_size=_pe_s,
                                      stride=_pe_s)
                         if _pe_s > 1 else _pe_enc)
                _pe_Tl = _pe_m.size(-1)
                assert _pe_e.size(-1) == _pe_Tl, \
                    (f"[POSEEV] pooled length {_pe_e.size(-1)} != pyramid "
                     f"level {_pe_l} length {_pe_Tl} (stride {_pe_s})")
                _pe_lgts.append(
                    self.poseev_head(_pe_e.transpose(1, 2)))  # (B, T_l, V)
            _pe_names = (self.text_features.train_classes if mode == 'train'
                         else self.text_features.test_classes)
            _pe_key = tuple(_pe_names)
            if _pe_key not in self._poseev_score_cache:
                _pe_code, _pe_pres = self._poseev_code_matrix(
                    list(_pe_names))
                _pe_cf = _pe_code[:, self._poseev_fidx]      # (C, Vf)
                if bool(_pe_pres.any()):
                    _pe_mean = _pe_cf[_pe_pres].mean(dim=0, keepdim=True)
                else:
                    _pe_mean = torch.zeros(1, _pe_cf.size(1))
                _pe_cc = _pe_cf - _pe_mean                   # (C, Vf)
                _pe_cc[~_pe_pres] = 0.0                      # skip missing
                _pe_den = _pe_cc.abs().sum(dim=1).clamp(min=1e-6)
                self._poseev_score_cache[_pe_key] = (
                    (_pe_cc / _pe_den.unsqueeze(1)).t().contiguous())
            _pe_W = self._poseev_score_cache[_pe_key]        # (Vf, C)
            assert _pe_W.size(1) == out_cls_logits[0].size(-1), \
                (f"[POSEEV] class count ({_pe_W.size(1)}) != logit "
                 f"columns ({out_cls_logits[0].size(-1)}) -- incompatible "
                 "with class-expanding switches (e.g. TIFAD_SUBCLASS)")
            _pe_Wd = _pe_W.to(device=out_cls_logits[0].device, dtype=_pe_dt)
            _pe_fidx_d = self._poseev_fidx.to(out_cls_logits[0].device)
            _pe_g = torch.tanh(self.poseev_gate)
            _new_lg = []
            for _pe_lg_l, _lg in zip(_pe_lgts, out_cls_logits):
                _pe_a = torch.sigmoid(
                    _pe_lg_l[..., _pe_fidx_d])               # (B, T_l, Vf)
                _new_lg.append(_lg + _pe_g * (_pe_a @ _pe_Wd))
            out_cls_logits = _new_lg
            if mode == 'train':
                # (B, sum T_l, V) for the aux BCE (train forwards only;
                # SELFTRAIN pseudo batches run mode=='test' and skip it,
                # matching the VATTR plumbing)
                _poseev_logits_cat = torch.cat(_pe_lgts, dim=1)

        # ---- SUBCLASS max-pool (pairs with text.py TIFAD_SUBCLASS): logits
        # over C*(1+S) interleaved pseudo-classes -> max over each class's
        # group -> C. Env: TIFAD_SUBCLASS_S=<S>. Eval-only, inert unset.
        import os as _oss
        _scs = int(_oss.environ.get('TIFAD_SUBCLASS_S', '0') or 0)
        if _scs > 0 and not self.training:
            _G = 1 + _scs
            out_cls_logits = [
                lg.view(lg.size(0), lg.size(1), lg.size(2) // _G, _G).amax(-1)
                for lg in out_cls_logits
            ]

        # ---- FINEGYM LEVEL COMPOSE max-pool (FINEGYM_LEVEL_COMPOSE=1,
        # EVAL ONLY; pairs with the class-list expansion in text.py): the
        # matmul above scored the EXPANDED [activity, member elements...]
        # pseudo-class list; amax within each variable-size group folds the
        # element-level (bottom) evidence back into the C activity logits.
        # Placed HERE (mirroring TIFAD_SUBCLASS) so downstream eval blocks
        # (ASNORM / PROTO / ASPEC / CCR) and inference see exactly C
        # columns; like SUBCLASS, do not combine with switches that consume
        # updated_text_lv per-class at eval. Inert when unset.
        _lvl_sizes = getattr(self.text_features, '_lvl_group_sizes', None)
        if _lvl_sizes is not None and not self.training:
            out_cls_logits = [_level_compose_pool(lg, _lvl_sizes)
                              for lg in out_cls_logits]

        # ---- ASNORM (speaker-verification AS-Norm migrated, EVAL ONLY):
        # per class c, subtract the mean logit of its top-N most-similar
        # impostor classes (cohort by text-emb cosine) at each snippet.
        # Env: TIFAD_ASNORM=<N> (int cohort size). Inert when unset/train.
        import os as _osn
        _asn = int(_osn.environ.get('TIFAD_ASNORM', '0') or 0)
        if _asn > 0 and not self.training:
            with torch.no_grad():
                _t = updated_text_lv[0][0]                     # (C, D) proj space
                _tn = _t / _t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                _sim = _tn @ _tn.t()                           # (C, C)
                _C = _sim.size(0)
                _k = min(_asn, _C - 1)
                _sim.fill_diagonal_(-2.0)
                _coh = _sim.topk(_k, dim=-1).indices           # (C, k)
                out_cls_logits = [
                    lg - lg[..., _coh].mean(-1)                # (B,T,C)-(B,T,C,k).mean
                    for lg in out_cls_logits
                ]

        # ---- PROTO BLEND (EXEM-style visual-prototype transfer, EVAL ONLY):
        # predict each eval class's VISUAL prototype from its projected text
        # vector via a ridge map W fit on seen classes (text -> mean level-0
        # feature inside GT segments; see scripts/collect_protos.py), then
        # blend the prototype logits into the text-classifier logits:
        #   logits = (1-g)*orig + g*((feat^T @ P^T)/logit_scale + bias)
        # Env: TIFAD_PROTO_FILE=<pt path>, TIFAD_PROTO_GAMMA=<float>.
        # Inert when unset or during training.
        _pf = _osn.environ.get('TIFAD_PROTO_FILE', '')
        if _pf and not self.training:
            _g = float(_osn.environ.get('TIFAD_PROTO_GAMMA', '0.2') or 0.2)
            if getattr(self, '_proto_blob_file', None) != _pf:
                _blob = torch.load(_pf, map_location='cpu')
                self._proto_W = _blob['W'].float()
                self._proto_blob_file = _pf
            with torch.no_grad():
                _W = self._proto_W.to(
                    device=updated_text_lv[0].device,
                    dtype=updated_text_lv[0].dtype,
                )
                _blended = []
                for lg, feat, text, bias in zip(
                        out_cls_logits, cls_feat,
                        updated_text_lv, updated_text_bias_lv):
                    # predicted visual prototypes for the CURRENT class set
                    _P = text @ _W                             # (B, C, D)
                    if _osn.environ.get('TIFAD_PROTO_NORM', '') == '1':
                        # match prototype row norms to the text row norms so
                        # the two logit terms live on the same scale
                        _tn = text.norm(dim=-1, keepdim=True)
                        _pn = _P.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        _P = _P * (_tn / _pn)
                    _plg = (
                        feat.transpose(2, 1) @ _P.transpose(2, 1)
                        / self.logit_scale.exp()
                    ) + bias.unsqueeze(1).repeat(1, feat.size(-1), 1)
                    _blended.append((1.0 - _g) * lg + _g * _plg)
                out_cls_logits = _blended

        # ---- ASPEC diag stash (TIFAD_ASPEC_COLLECT=1, EVAL ONLY): keep the
        # attribute logits and the PRE-composition text logits so
        # scripts/aspec_diag.py can score attribute composition vs text
        # logits independently (mirrors the TIFAD_PROTO_COLLECT pattern).
        if (aspec_logits is not None and not self.training
                and _aspec_collect):
            self._aspec_stash = {
                'aspec': [x.detach() for x in aspec_logits],       # F x (B,T_l,A)
                'cls':   [x.detach() for x in out_cls_logits],     # F x (B,T_l,C)
                'masks': [m.detach() for m in fpn_masks],          # F x (B,1,T_l)
            }

        # ---- ASPEC eval composition (TIFAD_ASPEC_ALPHA != 0, EVAL ONLY):
        # class logit += alpha * mean_{a in attrs(c)} logsigmoid(attr_logit_a)
        # per snippet -- unseen class scores composed from the trained
        # per-attribute VIDEO specialists. Applied after all other eval-only
        # logit blocks, before the clamp. Inert when alpha unset/0.
        if (aspec_logits is not None and not self.training
                and _aspec_alpha != 0):
            _acls_names = (self.text_features.train_classes if mode == 'train'
                           else self.text_features.test_classes)
            _AM = self._aspec_matrix(_acls_names, aspec_logits[0].device)
            _AM = _AM.to(aspec_logits[0].dtype)                    # (C, A)
            _acnt = _AM.sum(-1).clamp(min=1.0)                     # (C,)
            # TIFAD_ASPEC_GATE=1: entropy-gated composition -- weight the
            # attribute term per snippet by the TEXT logits' normalized
            # entropy (text confident -> trust text; text uncertain -> lean
            # on attribute composition). Fixes the fixed-alpha conflict
            # (alpha helping ThrowDiscus hurts PoleVault).
            import os as _osg, math as _mg
            _agate = _osg.environ.get('TIFAD_ASPEC_GATE', '') == '1'
            # TIFAD_ASPEC_SIBSOFT=1: replace the absolute composition with a
            # WITHIN-FAMILY log-softmax -- for each taxonomy family (fam map
            # from TIFAD_HTCA_FAMS, default configs/thumos14_fam_map.json)
            # with >= 2 members among the CURRENT class set, subtract the
            # family logsumexp of the alpha-weighted composition term, so the
            # attribute evidence only discriminates between siblings;
            # singleton families keep the absolute term.
            # TIFAD_ASPEC_PARENT=<beta>: additionally add beta * family_logit
            # (logsumexp over the family members' PRE-composition class
            # logits). Both EVAL-ONLY, inert when unset.
            _sibsoft = _osg.environ.get('TIFAD_ASPEC_SIBSOFT', '') == '1'
            _parent_b = float(
                _osg.environ.get('TIFAD_ASPEC_PARENT', '0') or 0)
            _sib_groups = None
            if _sibsoft or _parent_b != 0:
                _sib_fam_path = _osg.environ.get(
                    'TIFAD_HTCA_FAMS', 'configs/thumos14_fam_map.json')
                _sib_groups, _ = self._sib_family_groups(
                    list(_acls_names), _sib_fam_path,
                    aspec_logits[0].device)
            _new_lgs = []
            for lg, al in zip(out_cls_logits, aspec_logits):
                _comp = (F.logsigmoid(al) @ _AM.t()) / _acnt
                if _sibsoft or _parent_b != 0:
                    _term = _aspec_compose_sibsoft(
                        _aspec_alpha * _comp, lg, _sib_groups,
                        _sibsoft, _parent_b)
                    if _agate:
                        _p = torch.softmax(lg, dim=-1)
                        _H = -(_p * torch.log(_p.clamp(min=1e-9))).sum(-1)
                        _g = (_H / _mg.log(max(_AM.size(0), 2))).unsqueeze(-1)
                        _new_lgs.append(lg + _g * _term)
                    else:
                        _new_lgs.append(lg + _term)
                elif _agate:
                    _p = torch.softmax(lg, dim=-1)
                    _H = -(_p * torch.log(_p.clamp(min=1e-9))).sum(-1)
                    _g = (_H / _mg.log(max(_AM.size(0), 2))).unsqueeze(-1)
                    _new_lgs.append(lg + _aspec_alpha * _g * _comp)
                else:
                    _new_lgs.append(lg + _aspec_alpha * _comp)
            out_cls_logits = _new_lgs

        out_cls_logits = [torch.clamp(logit, min=-50000, max=50000) for logit in out_cls_logits]

        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks] 

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = self.point_generator(updated_feat)

        # ---- PROBE STASH (TIFAD_PROBE_STASH=1; scripts/phase_probe.py):
        # expose the cls-trunk pyramid the class cosine reads (cls_feat),
        # the valid masks and the point grid on the module so the post-hoc
        # phase probe can harvest per-location features in BOTH modes
        # (eval on test videos; train mode under no_grad on train videos,
        # where the LEAFLOSS targets are appended below). Detached, no
        # scoring-path change; byte-inert when unset.
        # TIFAD_PROBE_STASH_POST (representation probe on the vsa3
        # checkpoints, scripts/repvalid_variant.py runs unchanged): the
        # legacy stash reads `cls_feat`, which with TIFAD_VSA=1 is ALREADY
        # the pyramid returned by self.vsa(...) (the video ASA replaces
        # cls_feat in place above) -- i.e. the post-ASA video subtree
        # embedding; without VSA it is the trunk feature. The env makes the
        # choice explicit, same dict / keys / shapes:
        #   '1'   : post-VSA cls_feat (== legacy when VSA is on)
        #   '0'   : PRE-VSA cls trunk `_vsa_in` (== cls_feat when VSA is off)
        #   unset : legacy (cls_feat), byte-identical to before the env
        # Both are computed by this point, so no restructuring is needed;
        # the TIFAD_PROBE_FEAT=ph override below still wins. Banner once.
        if _osp.environ.get('TIFAD_PROBE_STASH', '') == '1':
            _ps_post = _osp.environ.get('TIFAD_PROBE_STASH_POST', '')
            _ps_src = _vsa_in if _ps_post == '0' else cls_feat
            if _ps_post and not getattr(self, '_probe_stash_post_banner', False):
                self._probe_stash_post_banner = True
                print(f"[PROBE-STASH] TIFAD_PROBE_STASH_POST={_ps_post}: stashing "
                      f"the {'PRE-VSA cls trunk (_vsa_in)' if _ps_post == '0' else 'post-VSA cls_feat'}"
                      f" (VSA {'active' if self.vsa is not None else 'off: identical'})",
                      flush=True)
            self._probe_stash = {
                'feats': [f.detach() for f in _ps_src],       # F x (B, D, T_l)
                'masks': [m.detach() for m in fpn_masks],     # F x (B, T_l)
                'points': [p.detach() for p in points],       # F x (T_l, 4)
                'fg': [f.detach() for f in out_fg],           # F x (B, T_l, 1)
            }

        # ---- PH feature (TIFAD_PH_SRC set): the level-0 tensor the phase
        # evidence reads, (B, D_ph, T0): cls (pre-VSA cls trunk) | stem
        # (backbone embedding output) | raw (I3D input); through the
        # contrastive branch when TIFAD_PH_CTR_MODE=branch (its params
        # get gradient from the contrastive loss; the head detaches).
        _ph_x0 = None
        if self.phasehead is not None:
            if self.ph_feat == 'cls':
                _ph_x0 = _vsa_in[0]
            elif self.ph_feat == 'stem':
                _ph_x0 = self.backbone._ph_embd_out
            else:
                _ph_x0 = batched_inputs
            assert _ph_x0.size(-1) == fpn_masks[0].size(-1), \
                f"[PH] feature length {_ph_x0.size(-1)} != level-0 {fpn_masks[0].size(-1)}"
            if self.phasebranch is not None:
                _ph_x0 = self.phasebranch(_ph_x0, fpn_masks[0])
            # TIFAD_PROBE_FEAT=ph: the post-hoc probe (scripts/phase_probe.py)
            # reads THIS feature (level 0 only) instead of the cls pyramid
            if _osp.environ.get('TIFAD_PROBE_STASH', '') == '1' and \
                    _osp.environ.get('TIFAD_PROBE_FEAT', '') == 'ph':
                self._probe_stash['feats'] = [_ph_x0.detach()]
                self._probe_stash['masks'] = [fpn_masks[0].detach()]
                self._probe_stash['points'] = [points[0].detach()]

        # return loss during training
        if self.training:
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels = [x['labels'].to(self.device) for x in video_list]

            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets, gt_fg, gt_cn = self.label_points(
                points, gt_segments, gt_labels, new_cls_num)

            # ---- SIBCON plumbing (TIFAD_SIBCON=<w>, TRAIN ONLY): hand
            # losses() the level-0 snippet features and the matmul-space
            # (projected) text vectors for the sibling-hard-negative InfoNCE.
            # Inert (both None) when TIFAD_SIBCON unset.
            _sibcon_w = float(_osp.environ.get('TIFAD_SIBCON', '0') or 0)
            # TDNECK: SIBCON aligns features to the cls text space -> sem
            _sibcon_feat0 = cls_feat[0] if _sibcon_w != 0 else None
            _sibcon_text0 = updated_text_lv[0] if _sibcon_w != 0 else None

            # ---- VATTR plumbing (TIFAD_VATTR=1, TRAIN ONLY; skipped for
            # SELFTRAIN pseudo batches, which run mode=='test' against the
            # test vocabulary): hand losses() the per-point trunk features
            # concatenated over levels, (B, sum T_l, D) -- SAME level order
            # as the fpn_masks / gt_cls concatenation, so pos_mask indexes
            # them directly. The stop-gradient happens at the point of use
            # in losses(). Inert (None) when TIFAD_VATTR unset.
            _vattr_feats = None
            if self.vattr_head is not None and mode == 'train':
                # TDNECK: the aux BCE must train vattr_head on the SAME
                # pyramid the score blend reads (cls_feat) -> sem
                _vattr_feats = torch.cat(
                    [f.transpose(1, 2) for f in cls_feat], dim=1)

            # ---- LEAFLOSS plumbing (TIFAD_LEAFLOSS=1, TRAIN ONLY; skipped
            # for SELFTRAIN pseudo batches, mode=='test'): per-location
            # (class, phase) targets from the pseudo phase cuts + the SAME
            # level-concatenated trunk features the cls matmul reads
            # (cls_feat; VATTR layout, pos_mask indexes them directly). The
            # trunk is NOT detached: the phase CE is meant to shape it.
            # Inert (both None) when unset.
            _leaf_tg = _leaf_feats = None
            if getattr(self, 'leafloss', False) and mode == 'train':
                _leaf_tg = self._leaf_targets(
                    torch.cat(points, dim=0), gt_labels, video_list)
                # VSA-SOFTPHASE: the phase head inside vsa.py reads the
                # PRE-VSA cls pyramid, so its leaf CE is trained on the
                # same tensor (`_vsa_in`); otherwise unchanged (cls_feat).
                _leaf_src = (_vsa_in if (self.vsa is not None
                                         and self.vsa.softphase)
                             else cls_feat)
                _leaf_feats = torch.cat(
                    [f.transpose(1, 2) for f in _leaf_src], dim=1)
                # PROBE STASH (see above): also expose the (B, FT, 2)
                # [class, phase] targets + point->GT assignment
                if _osp.environ.get('TIFAD_PROBE_STASH', '') == '1':
                    self._probe_stash['leaf_targets'] = _leaf_tg.detach()
                    self._probe_stash['leaf_assign'] = list(self._leaf_assign)

            # compute the loss and return
            # ---- VSA-ALIGN plumbing (TIFAD_VSA_ALIGN_W, TRAIN ONLY): the
            # adapted chunk vectors concatenated over levels, (B, FT, K, D)
            # (VATTR layout: pos_mask indexes them directly), and the
            # adapted phase bank (C, K, D) of the train classes.
            _align_feats = _align_et = _align_valid = None
            if _vsa_align is not None and mode == 'train':
                _zv_l, _va_l = [], []
                for _zv, _fm in zip(_vsa_align['zv'], fpn_masks):
                    _Bm, _Tm = _fm.size(0), _fm.size(-1)
                    if _zv is None:                 # MINLVL pass-through
                        _ref = next(z for z in _vsa_align['zv'] if z is not None)
                        _zv_l.append(_ref.new_zeros(
                            _Bm, _Tm, _ref.size(2), _ref.size(3)))
                        _va_l.append(torch.zeros(_Bm, _Tm, dtype=torch.bool,
                                                 device=_fm.device))
                    else:
                        _zv_l.append(_zv)
                        _va_l.append(torch.ones(_Bm, _Tm, dtype=torch.bool,
                                                device=_fm.device))
                _align_feats = torch.cat(_zv_l, dim=1)      # (B, FT, K, D)
                _align_valid = torch.cat(_va_l, dim=1)      # (B, FT)
                _align_et = _vsa_align['et']

            losses = self.losses(
                fpn_masks,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets, new_cls_num,
                out_fg, gt_fg,
                out_centerness, gt_cn,
                haf_logits=haf_logits,
                aspec_logits=aspec_logits,
                sibcon_feat0=_sibcon_feat0,
                sibcon_text0=_sibcon_text0,
                vattr_feats=_vattr_feats,
                poseev_logits=_poseev_logits_cat,
                leaf_targets=_leaf_tg,
                leaf_feats=_leaf_feats,
                align_feats=_align_feats,
                align_et=_align_et,
                align_valid=_align_valid,
                dfl_logits=self.reg_head.dfl_logits,
                dfl_bins=([getattr(self.reg_head, f'dfl_bins_{_l}')
                           for _l in range(len(self.fpn_strides))]
                          if self.reg_head.dfl_logits is not None else None),
            )

            # ---- COMPL train loss (TIFAD_COMPL=1, see __init__; skipped
            # for SELFTRAIN pseudo batches, which run mode=='test' --
            # VIDPHASE precedent). Mines SSN-style candidates around every
            # GT segment (level-0 grid units, same convention as
            # _pool_gt_segment_feats), pools {pre, 3 thirds, post} from
            # DETACHED cls_feat[0] (the head must not reshape the trunk)
            # and trains compl_head ALONE with BCE * compl_w.
            if self.compl_head is not None and mode == 'train':
                _cp_feat0 = cls_feat[0].detach()          # (B, D, T0)
                _cp_x, _cp_y = [], []
                for _cp_bi in range(_cp_feat0.size(0)):
                    _cp_tv = int(fpn_masks[0][_cp_bi].sum().item())
                    _cp_c, _cp_l = _compl_mine(gt_segments[_cp_bi], _cp_tv)
                    if _cp_c is None:
                        continue
                    _cp_x.append(_compl_pool(
                        _cp_feat0[_cp_bi, :, :_cp_tv], _cp_c))
                    _cp_y.append(_cp_l)
                if _cp_x:
                    _cp_lg = self.compl_head(
                        torch.cat(_cp_x)).squeeze(-1).float()
                    _cp_loss = self.compl_w * \
                        F.binary_cross_entropy_with_logits(
                            _cp_lg, torch.cat(_cp_y))
                else:
                    # keep the head's graph alive (unused-param safety;
                    # PHASEHEAD precedent)
                    _cp_loss = 0.0 * sum(
                        p.sum() for p in self.compl_head.parameters())
                losses['compl_loss'] = _cp_loss
                losses['final_loss'] = losses['final_loss'] + _cp_loss

            # ---- QH train loss (TIFAD_QH=1, see __init__ / qhead.py;
            # skipped for SELFTRAIN pseudo batches, mode=='test'). The
            # head scores the detector's OWN decoded candidates at every
            # valid location of every level (DETACHED offsets / features)
            # against the tIoU with the seen GT of the item; all inputs
            # are detached inside the head (trunk untouched).
            if (self.qhead is not None or self.brhead is not None) \
                    and mode == 'train':
                losses = self._s2_train(
                    losses, _vsa_in, fpn_masks, points, out_offsets,
                    out_cls_logits, gt_cls_labels, gt_segments, _qh_ph,
                    out_fg[0])

            # ---- PH train losses (TIFAD_PH_SRC=head [+ TIFAD_PH_BOUND=1],
            # see __init__ / phaseparse.py; skipped for SELFTRAIN pseudo
            # batches): phase CE on level-0 locations against the pseudo
            # phase cuts of the seen GT (+ sampled background), and the
            # phase-profile boundary head's L1 + DIoU on the detector's
            # OWN decoded candidates. All inputs detached.
            if self.phasehead is not None and mode == 'train' and (
                    self.phasehead.src == 'head' or self.ph_brhead is not None
                    or self.phasectr is not None):
                losses = self._ph_train(
                    losses, _ph_x0, fpn_masks, points, out_offsets,
                    gt_segments, video_list, _ph_bank,
                    __import__('libs.modeling.phaseparse', fromlist=['x'])
                    .actionness_from_fg(out_fg, points, fpn_masks))

            # ---- BM train loss (TIFAD_BM=1, B2): start / end BCE on the
            # level-0 locations against the seen GT boundaries (+-1).
            if self.bmhead is not None and mode == 'train':
                _bm_lg = self.bmhead.logits(_vsa_in[0])            # (B, T0, 2)
                _bm_sum = _bm_lg.new_zeros(())
                _bm_n, _bm_np = 0, 0
                for _b in range(_bm_lg.size(0)):
                    _l, _n, _npos = self.bmhead.train_loss(
                        _bm_lg[_b], points[0][:, 0], fpn_masks[0][_b],
                        gt_segments[_b])
                    _bm_sum = _bm_sum + _l
                    _bm_n += _n
                    _bm_np += _npos
                if _bm_n == 0:
                    _bm_loss = 0.0 * sum(p.sum() for p in self.bmhead.parameters())
                else:
                    _bm_loss = self.bm_w * _bm_sum / _bm_n
                losses['bm_loss'] = _bm_loss
                losses['bm_npos'] = torch.tensor(float(_bm_np), device=_bm_lg.device)
                losses['final_loss'] = losses['final_loss'] + _bm_loss

            # ---- V1 VIDPHASE aux loss (TIFAD_VIDPHASE=<w>, TRAIN ONLY;
            # skipped for SELFTRAIN pseudo batches, which run mode=='test'):
            # InfoNCE between the 3 mean-pooled thirds of every GT segment
            # (level-0 features) and the 3-chunk phase-text bank from the
            # per-sentence ensemble embeddings stashed by text.py. Gradients
            # reach BOTH the backbone (video thirds) and text_projection
            # (phase bank). Mean CE * w added to final_loss.
            if self.vidphase_w != 0 and mode == 'train':
                _vp_per = getattr(self.text_features, '_vidphase_per', None)
                if _vp_per is None:
                    raise RuntimeError(
                        "[VIDPHASE] per-sentence stash missing -- is "
                        "TIFAD_PROMPT_ENSEMBLE set?")
                # ---- VIDPHASE_DECAY (TIFAD_VIDPHASE_DECAY=1): effective
                # weight w * max(0, 1 - step/decay_steps); the step counter
                # increments once per training forward. Past decay_steps the
                # loss is skipped entirely (weight 0). Inert when unset
                # (decay_steps == 0 keeps w unchanged).
                _vp_w_eff = self.vidphase_w
                if self.vidphase_decay_steps > 0:
                    _vp_w_eff = self.vidphase_w * max(
                        0.0, 1.0 - self._vp_step / self.vidphase_decay_steps)
                    self._vp_step += 1
                _vp_feats = _vp_labs = None
                if _vp_w_eff != 0:
                    _vp_feats, _vp_labs = _pool_gt_segment_feats(
                        updated_feat[0], gt_segments, gt_labels, 3)
                if _vp_feats is not None:
                    # ---- VIDPHASE_DETACH (TIFAD_VIDPHASE_DETACH=1): the
                    # InfoNCE queries are stop-gradient trunk features run
                    # through the trainable Linear head -- the trunk gets
                    # ZERO gradient from this aux loss (text side unchanged).
                    if self.vidphase_detach_proj is not None:
                        _vp_feats = self.vidphase_detach_proj(
                            _vp_feats.detach())
                    _vp_bank = _vidphase_phase_bank(_vp_per, 3)
                    if _vp_bank.size(0) != new_cls_num:
                        raise RuntimeError(
                            f"[VIDPHASE] bank rows {_vp_bank.size(0)} != "
                            f"#train classes {new_cls_num}")
                    _vp_loss = _vp_w_eff * _vidphase_infonce(
                        _vp_feats, _vp_labs, _vp_bank, tau=0.1)
                    losses['vidphase_loss'] = _vp_loss
                    losses['final_loss'] = losses['final_loss'] + _vp_loss
                # release the stashed graph either way
                self.text_features._vidphase_per = None

            # ---- V2 VIDACT aux loss + EMA prototype update
            # (TIFAD_VIDACT=<w>, TRAIN ONLY): pull each GT-segment feature
            # toward its ACTIVITY's EMA prototype (cosine, prototype
            # DETACHED -- segments move to prototypes, not vice versa),
            # weighted by the learned Kendall logvar (exp(-s)*w*L + s/2).
            # Prototypes update AFTER the loss (no grad); activities unseen
            # so far initialize from their first batch mean and contribute
            # no loss that step.
            if self.vidact_w != 0 and mode == 'train':
                _va_feats, _va_labs = _pool_gt_segment_feats(
                    updated_feat[0], gt_segments, gt_labels, 1)
                if _va_feats is not None:
                    _va_feats = _va_feats.squeeze(1)               # (M, D)
                    _classes = list(self.text_features.train_classes)
                    if getattr(self, '_vidact_key', None) != tuple(_classes):
                        _missing = [c for c in _classes
                                    if c not in self._vidact_map]
                        if _missing:
                            raise KeyError(
                                "[VIDACT] train classes missing from the "
                                f"activity map: {_missing}")
                        self._vidact_cls2fam = torch.tensor(
                            [self._vidact_fid[self._vidact_map[c]]
                             for c in _classes], dtype=torch.long)
                        self._vidact_key = tuple(_classes)
                    _fam = self._vidact_cls2fam.to(
                        _va_feats.device)[_va_labs]                # (M,)
                    _ready = self.vidact_proto_ready[_fam] > 0
                    if bool(_ready.any()):
                        _p = F.normalize(
                            self.vidact_protos[_fam[_ready]].detach(),
                            dim=-1)
                        _f = F.normalize(_va_feats[_ready], dim=-1)
                        _pull = (1.0 - (_p * _f).sum(dim=-1)).mean()
                        _s = self.vidact_uw_logvar[0]
                        _va_loss = (torch.exp(-_s) * (self.vidact_w * _pull)
                                    + 0.5 * _s)
                        losses['vidact_loss'] = _va_loss
                        losses['vidact_s'] = _s.detach()
                        losses['final_loss'] = (losses['final_loss']
                                                + _va_loss)
                    with torch.no_grad():
                        for _g in _fam.unique():
                            _m = _fam == _g
                            _bm = _va_feats[_m].mean(dim=0)
                            if self.vidact_proto_ready[_g] > 0:
                                self.vidact_protos[_g] = (
                                    self.vidact_momentum
                                    * self.vidact_protos[_g]
                                    + (1.0 - self.vidact_momentum) * _bm)
                            else:
                                self.vidact_protos[_g] = _bm
                                self.vidact_proto_ready[_g] = 1.0

            # ---- TIFAD_SELFTRAIN: expose the concatenated per-point cls
            # logits + valid mask so scripts/self_train.py can compute the
            # Mean-Teacher consistency term (MSE between student and EMA-
            # teacher class probabilities on pseudo-labeled batches). The
            # driver pops these keys before logging/backward. Inert when the
            # env is unset.
            if _osp.environ.get('TIFAD_SELFTRAIN', '') == '1':
                losses['st_cls_logits'] = torch.cat(out_cls_logits, dim=1)
                losses['st_valid_mask'] = torch.cat(fpn_masks, dim=1)

            # ---- PHASEHEAD train losses (TIFAD_PHASEHEAD=1, see
            # __init__; skipped for SELFTRAIN pseudo batches, which run
            # mode=='test' -- VIDPHASE precedent). The branch consumes
            # the raw vmae_grid tensor and shares zero parameters with
            # the trunk, so these losses train ONLY phasehead_enc /
            # phasehead_head (no detach needed) and never move the
            # scoring path. (a) phase_ce: CE over grid steps with target
            # >= 0 (ignore_index=-1 also swallows the batch padding).
            # (b) phase_mono: within labeled runs, hinge on the softmax
            # expectation E[phase|t] being non-decreasing --
            # mean(relu(E_t - E_{t+1} - 0.0)) over adjacent step pairs
            # that are BOTH labeled (pairs never straddle unlabeled gaps;
            # two back-to-back segments with no gap would contribute one
            # spurious pair, which the Stage-A boundaries make measure-
            # zero). Logits in float32 (AMP-safe).
            if self.phasehead_enc is not None and mode == 'train':
                _ph_x, _ph_tg = self._phasehead_batch(
                    video_list, batched_inputs.size(-1), True)
                _ph_lg = self.phasehead_head(
                    self.phasehead_enc(_ph_x).transpose(1, 2)
                ).float()                                  # (B, T, 10)
                _ph_valid = _ph_tg >= 0
                if bool(_ph_valid.any()):
                    _ph_ce = F.cross_entropy(
                        _ph_lg[_ph_valid], _ph_tg[_ph_valid])
                else:
                    # keep the branch graph alive (unused-param safety)
                    _ph_ce = 0.0 * _ph_lg.sum()
                _ph_E = (F.softmax(_ph_lg, dim=-1)
                         * torch.arange(self._phasehead_K,
                                        device=_ph_lg.device,
                                        dtype=_ph_lg.dtype)).sum(-1)
                _ph_pair = _ph_valid[:, :-1] & _ph_valid[:, 1:]
                if bool(_ph_pair.any()):
                    _ph_mono = F.relu(
                        _ph_E[:, :-1] - _ph_E[:, 1:])[_ph_pair].mean()
                else:
                    _ph_mono = 0.0 * _ph_lg.sum()
                losses['phase_ce'] = self.phasehead_w * _ph_ce
                losses['phase_mono'] = self.phasehead_mono * _ph_mono
                losses['final_loss'] = (losses['final_loss']
                                        + losses['phase_ce']
                                        + losses['phase_mono'])

            return losses

        else:
            # ---- TF-CADE CCR (TIFAD_CCR=1, EVAL ONLY): internal foreground
            # video-prior (ACA, Eq. 3-7) -> per-class softmax prior consumed
            # by CCR (Eq. 8) inside inference_single_video, BEFORE NMS.
            # Composes with TIFAD_FUSE_ALPHA, which fuses the EXTERNAL
            # classifier prior later in postprocessing.py -- CCR here supplies
            # the paper's SELF-computed substitute/third opinion.
            # Envs: TIFAD_CCR=1, TIFAD_CCR_SIGMA (default 1.0; paper Table 7d).
            # Byte-inert when TIFAD_CCR unset.
            ccr_prior = None
            if _osp.environ.get('TIFAD_CCR', '') == '1':
                with torch.no_grad():
                    ccr_prior = tfcade_ccr_prior(
                        out_cls_logits, fpn_masks,
                        cls_feat, updated_text_lv,
                        sigma=float(
                            _osp.environ.get('TIFAD_CCR_SIGMA', '1') or 1),
                        subclass_s=_scs,
                    )
                if not getattr(self, '_ccr_announced', False):
                    print(f"[CCR] TF-CADE prior ON  sigma="
                          f"{_osp.environ.get('TIFAD_CCR_SIGMA', '1') or 1}  "
                          f"levels={len(out_cls_logits)}  "
                          f"C={ccr_prior.size(-1)}", flush=True)
                    self._ccr_announced = True

            # ---- FINEQ (TIFAD_FINEQ=<beta in (0,1]>, EVAL ONLY): dynamic
            # fine-resolution re-scoring of coarse-level candidates. Classes
            # whose raw CLIP name embedding has max cosine > tau
            # ACTIVE list are "confusable-flagged" (CONFGATE machinery).
            # For flagged-class candidates decoded from FPN level >= 2,
            # inference_single_video re-scores the segment against the
            # LEVEL-0 features: mean-pool updated_feat[0] over the whole
            # span + 4 equal sub-windows (5 pools), score each pool with
            # the SAME level-0 calibrated text matrix / bias / logit-scale
            # the base matmul used, fine = max over pools of the sigmoid,
            # then blend geometrically: score' = score^(1-beta)*fine^beta.
            # Optional boundary snap to the level-0 fg*cn local argmax
            # (TIFAD_FINEQ_SNAP=1). NO new parameters; plumbed like
            # ccr_prior. v1 note: TIFAD_FINEQ_SHORT (expected-duration
            # flagging) is parsed but NOT implemented -- tau-flagging only.
            # Byte-inert when TIFAD_FINEQ unset.
            fineq_ctx = None
            _fq_env = _osp.environ.get('TIFAD_FINEQ', '')
            if _fq_env != '':
                _fq_beta = float(_fq_env)
                assert 0.0 < _fq_beta <= 1.0, \
                    "[FINEQ] TIFAD_FINEQ (beta) must be in (0, 1]"
                _fq_tau = float(
                    _osp.environ.get('TIFAD_FINEQ_TAU', '0.85') or 0.85)
                _fq_snap = _osp.environ.get('TIFAD_FINEQ_SNAP', '') == '1'
                _fq_names = tuple(
                    self.text_features.train_classes if mode == 'train'
                    else self.text_features.test_classes)
                if not hasattr(self, '_fineq_flag_cache'):
                    self._fineq_flag_cache = {}   # (classes, tau) -> flags
                _fq_key = (_fq_names, _fq_tau)
                if _fq_key not in self._fineq_flag_cache:
                    with torch.no_grad():
                        _fe = self.text_features.extract_text_emb(
                            list(_fq_names), is_prompt=False).detach().cpu()
                        _fn = _fe / _fe.norm(dim=-1, keepdim=True).clamp(
                            min=1e-6)
                        _fsim = _fn @ _fn.t()
                        _fsim.fill_diagonal_(-2.0)
                        self._fineq_flag_cache[_fq_key] = (
                            _fsim.max(-1).values > _fq_tau)
                _fq_flags = self._fineq_flag_cache[_fq_key]
                assert _fq_flags.numel() == out_cls_logits[0].size(-1), \
                    (f"[FINEQ] class count ({_fq_flags.numel()}) != logit "
                     f"columns ({out_cls_logits[0].size(-1)}) -- "
                     "incompatible with class-expanding switches")
                if not getattr(self, '_fineq_announced', False):
                    _fq_short = float(
                        _osp.environ.get('TIFAD_FINEQ_SHORT', '0') or 0)
                    if _fq_short != 0:
                        print("[FINEQ] NOTE: TIFAD_FINEQ_SHORT (duration "
                              "flagging) not implemented in v1 -- ignored "
                              "(tau-flagging only)", flush=True)
                    print(f"[FINEQ] active (beta={_fq_beta}, tau={_fq_tau},"
                          f" snap={int(_fq_snap)}, "
                          f"flagged={int(_fq_flags.sum())} classes)",
                          flush=True)
                    self._fineq_announced = True
                fineq_ctx = {
                    'feat0': cls_feat[0],               # (B, D, T0)
                    'text0': updated_text_lv[0],        # (B, C, D) projected
                    'bias0': updated_text_bias_lv[0],   # (B, C)
                    'flags': _fq_flags,                 # (C,) bool, CPU
                    'beta': _fq_beta,
                    'snap': _fq_snap,
                    'scale': self.logit_scale.exp(),
                }

            # ---- SUPPORT completeness rescoring (TIFAD_SUPPORT=<beta>,
            # EVAL ONLY, env-gated, byte-inert unset): build the per-video
            # level-0 class-evidence curve exactly as the PHASEPROF dump
            # does (11-sentence ensemble: class + 10 phases, mean-pooled;
            # projected like the base cls matmul, bias omitted), then
            # robust-z per class over the whole video (median/MAD -- no GT
            # used, affine-invariant so the omitted bias cancels). The
            # curve is consumed in inference_single_video: each decoded
            # candidate's score is multiplied by C^beta with
            # C = clipped-z mass inside the span / mass in the span
            # dilated by 0.5*len on each side, BEFORE soft-NMS.
            # Convention documented: level-0 grid, mean over the 11
            import os as _ossp
            support_ctx = None
            _sup_beta = float(_ossp.environ.get('TIFAD_SUPPORT', '0') or 0)
            if _sup_beta > 0:
                _sup_per = getattr(
                    self.text_features, '_dumpphase_per', None)
                assert _sup_per is not None, \
                    "[SUPPORT] stash missing (needs TIFAD_PROMPT_ENSEMBLE)"
                assert len(video_list) == 1, "[SUPPORT] eval B=1 only"
                _sup_n = _sup_per / _sup_per.norm(dim=-1, keepdim=True)
                _sup_p = self.text_proj(_sup_n / 2.0)        # (C, K, D)
                _sup_T0 = int(fpn_masks[0][0].sum().item())
                _sup_q = cls_feat[0][0, :, :_sup_T0].transpose(1, 0)
                _sup_s = torch.einsum(
                    'td,ckd->tck', _sup_q, _sup_p) / self.logit_scale.exp()
                _sup_ev = _sup_s.mean(-1).float()            # (T0, C)
                _sup_med = _sup_ev.median(dim=0, keepdim=True).values
                _sup_mad = (_sup_ev - _sup_med).abs().median(
                    dim=0, keepdim=True).values
                _sup_z = (_sup_ev - _sup_med) / (1.4826 * _sup_mad + 1e-6)
                support_ctx = {
                    'czp': torch.cat([
                        _sup_z.new_zeros(1, _sup_z.size(1)),
                        _sup_z.clamp(min=0).cumsum(dim=0)]),  # (T0+1, C)
                    't0': points[0][:_sup_T0, 0].contiguous(),
                    'beta': _sup_beta,
                }
                if not getattr(self, '_support_announced', False):
                    print(f"[SUPPORT] active (pre-NMS completeness, "
                          f"beta={_sup_beta}, dilate=0.5)", flush=True)
                    self._support_announced = True

            # ---- COMPL eval rescoring context (TIFAD_COMPL=1, see
            # __init__): hand inference the DETACHED level-0 cls_feat so
            # inference_single_video can score every decoded candidate with
            # the SAME {pre, thirds, post} pooling the head was trained on
            # (FINEQ feat0 plumbing precedent) and multiply its score by
            # (sigmoid(compl)/per-video-median)^beta pre-soft-NMS. Exactly
            # the identity at init (zero-init final layer -> ratio 1.0).
            # Inert (None -> byte-identical decode) when TIFAD_COMPL unset.
            compl_ctx = None
            if self.compl_head is not None:
                compl_ctx = {
                    'feat0': cls_feat[0].detach(),        # (B, D, T0)
                    'beta': self.compl_beta,
                }
                if not getattr(self, '_compl_announced', False):
                    print(f"[COMPL] active (SSN completeness head, "
                          f"W={self.compl_w}, beta={self.compl_beta})",
                          flush=True)
                    self._compl_announced = True

            # ---- PHASEPROF dump (TIFAD_DUMP_PHASEPROF=<dir>, EVAL ONLY,
            # env-gated, byte-inert unset): per test video, save the
            # per-FPN-level phase-vs-video score profiles (T_l, C, K=11
            # sentences: 0=class, 1..10=ordered phases), projected EXACTLY
            # like the base cls matmul (L2-norm, text_proj(x/2),
            # / logit_scale.exp(); bias omitted -- per-class constant).
            # One .npz per video with the feature-grid coords + timing meta
            # so offline analysis can map scores to seconds.
            import os as _ospp
            _dpp_dir = _ospp.environ.get('TIFAD_DUMP_PHASEPROF', '')
            if _dpp_dir:
                import numpy as _nppp
                _dpp_per = getattr(
                    self.text_features, '_dumpphase_per', None)
                assert _dpp_per is not None, \
                    "[PHASEPROF] stash missing (needs TIFAD_PROMPT_ENSEMBLE)"
                self.text_features._dumpphase_per = None
                assert len(video_list) == 1, "[PHASEPROF] eval B=1 only"
                _dpp_n = _dpp_per / _dpp_per.norm(dim=-1, keepdim=True)
                _dpp_p = self.text_proj(_dpp_n / 2.0)      # (C, K, D)
                _dpp_scale = self.logit_scale.exp()
                _ospp.makedirs(_dpp_dir, exist_ok=True)
                _dpp_out = {}
                for _li, (_f, _m, _pt) in enumerate(
                        zip(cls_feat, fpn_masks, points)):
                    _T = int(_m[0].sum().item())
                    _q = _f[0, :, :_T].transpose(1, 0)     # (T, D)
                    _s = torch.einsum(
                        'td,ckd->tck', _q, _dpp_p) / _dpp_scale
                    _dpp_out[f'scores_l{_li}'] = \
                        _s.half().cpu().numpy()
                    _dpp_out[f'pts_l{_li}'] = \
                        _pt[:_T, 0].cpu().numpy()
                _v0 = video_list[0]
                _nppp.savez_compressed(
                    _ospp.path.join(_dpp_dir, _v0['video_id'] + '.npz'),
                    fps=_v0['fps'], duration=_v0['duration'],
                    feat_stride=_v0['feat_stride'],
                    nframes=_v0['feat_num_frames'],
                    class_names=_nppp.asarray(
                        self.text_features.test_classes, dtype=object),
                    **_dpp_out)

            # ---- PHASEHEAD eval dump (TIFAD_PHASEHEAD=1 +
            # TIFAD_PHASEHEAD_DUMP=<dir>, EVAL ONLY): stash the per-video
            # phase logits (T, 10) as <dir>/<video_id>.npy for Stage-C
            # offline work (e.g. held-out Spearman(predicted phase, time)
            # against the pseudo labels). T = the item's level-0 grid
            # length (phase_feats length), so rows map 1:1 onto I3D grid
            # steps. NO scoring-path change in v1: inference below is
            # untouched. Re-dumped (overwritten) at every eval epoch, so
            # the files always reflect the LAST evaluated checkpoint.
            # Inert when either env is unset.
            if self.phasehead_enc is not None and self.phasehead_dump:
                import numpy as _npph
                _osp.makedirs(self.phasehead_dump, exist_ok=True)
                _ph_x, _ = self._phasehead_batch(
                    video_list, batched_inputs.size(-1), False)
                with torch.no_grad():
                    _ph_lg = self.phasehead_head(
                        self.phasehead_enc(_ph_x).transpose(1, 2)
                    ).float()                              # (B, Tmax, 10)
                for _ph_bi, _ph_v in enumerate(video_list):
                    _ph_p = _ph_v.get('phase_feats', None)
                    _ph_T = (_ph_p.size(-1) if _ph_p is not None
                             else int(fpn_masks[0][_ph_bi].sum().item()))
                    _npph.save(
                        _osp.path.join(self.phasehead_dump,
                                       _ph_v['video_id'] + '.npy'),
                        torch.nan_to_num(
                            _ph_lg[_ph_bi, :_ph_T]).cpu().numpy())

            # torch.sum(out_cls_logits[0].sigmoid(), 1)
            # decode the actions (sigmoid / stride, etc)
            # ---- VSA-XM-RESCORE ctx (TIFAD_VSA_XM_RESCORE=<gamma>, EVAL
            # ONLY): the raw ordered match m F x (B, T_l, C) (None on
            # MINLVL pass-through levels) -- consumed per video below.
            _xm_ctx = None
            if (_vsa_match is not None
                    and getattr(self, 'vsa_xm_rescore', 0) != 0):
                _xm_ctx = {'m': _vsa_match, 'gamma': self.vsa_xm_rescore}
            # ---- QH ctx (TIFAD_QH=1, EVAL): the DETACHED pre-VSA cls
            # pyramid (f_n per level + level-0 span pools), the level-0
            # valid mask and the calibrated phase bank, consumed per
            # candidate in inference_single_video. None (byte-identical
            # decode) when beta == 0 unless the CAND dump wants q.
            _qh_ctx = None
            _s2_need = (self.qhead is not None and (
                self.qh_beta != 0
                or bool(_osp.environ.get('TIFAD_DUMP_CAND', '')))) or (
                self.brhead is not None and self.br_apply)
            if _s2_need:
                _qh_ctx = {
                    'feats': [f.detach() for f in _vsa_in],  # F x (B, D, T_l)
                    'mask0': fpn_masks[0],                   # (B, T0)
                    'phase': _qh_ph,                         # (C, K, D) | None
                    'fg0': out_fg[0].detach(),               # (B, T0, 1) logits
                    'beta': (self.qh_beta if self.qhead is not None else 0.0),
                    'comb': (self.qh_comb if self.qhead is not None else 'geo'),
                    'qh': self.qhead is not None,
                    'br': self.brhead is not None and self.br_apply,
                    'br_iters': (self.brhead.iters if self.brhead is not None
                                 else 1),
                }
                if not getattr(self, '_qh_announced', False):
                    if self.qhead is not None:
                        print(f"[QH] eval rescoring active (beta={self.qh_beta}, "
                              f"comb={self.qh_comb}, phasemap={self.qhead.phasemap}; "
                              f"soft-NMS ordered by the combined score)", flush=True)
                    if _qh_ctx['br']:
                        print(f"[BR] eval boundary refinement active (iters="
                              f"{_qh_ctx['br_iters']})", flush=True)
                    self._qh_announced = True
            # ---- PH ctx (TIFAD_PH_SRC set, EVAL): the level-0 phase
            # log-posteriors of the DETACHED pre-VSA cls trunk ((B, T0,
            # K+1); text source: (B, C, T0, K+1)) + the level-0 mask;
            # consumed per video in inference_single_video. None (byte-
            # identical decode) when no consumer is on unless the CAND
            # dump wants the parse.
            _ph_ctx = None
            if self.phasehead is not None and (
                    self.ph_alpha != 0 or self.ph_beta != 0
                    or self.ph_brhead is not None
                    or bool(_osp.environ.get('TIFAD_DUMP_CAND', ''))):
                with torch.no_grad():
                    from .phaseparse import actionness_from_fg
                    _ph_fg = actionness_from_fg(out_fg, points, fpn_masks)
                    _ph_lp = (self.phasehead.logpost(_ph_x0, bank=_ph_bank,
                                                     fg0=_ph_fg)
                              if self.phasehead.whiten != 'seg' else None)
                _ph_ctx = {
                    'lp': _ph_lp, 'mask0': fpn_masks[0],
                    'feat0': (_ph_x0.detach() if self.phasehead.whiten == 'seg'
                              else None),
                    'fg': _ph_fg,
                    'alpha': self.ph_alpha, 'beta': self.ph_beta,
                    'minllr': self.ph_minllr, 'maxw': self.ph_maxw,
                    'ctx': self.ph_ctx, 'K': self.phasehead.K,
                    'text': self.phasehead.src == 'text',
                    'bound': self.ph_brhead is not None,
                }
                if not getattr(self, '_ph_announced', False):
                    print(f"[PH] eval active (src={self.phasehead.src}, extent "
                          f"alpha={self.ph_alpha}, minllr={self.ph_minllr}, "
                          f"compl beta={self.ph_beta}, bound="
                          f"{self.ph_brhead is not None}; applied before "
                          f"soft-NMS)", flush=True)
                    self._ph_announced = True
            # ---- BND ctx (B2-B4): boundary-map probabilities, shrink
            # factor, re-anchoring inputs (per-level points / offsets);
            # consumed in inference_single_video / postprocessing.
            _bnd_ctx = None
            if (self.bmhead is not None and self.bm_snap > 0) or self.shrink_f \
                    or self.reanchor:
                _bnd_ctx = {
                    'prob': (torch.sigmoid(self.bmhead.logits(_vsa_in[0]).float())
                             if (self.bmhead is not None and self.bm_snap > 0)
                             else None),                            # (B, T0, 2)
                    'mask0': fpn_masks[0],
                    'snap_r': (self.bm_snap if self.bmhead is not None else 0.0),
                    'shrink': self.shrink_f,
                    'reanchor': self.reanchor,
                    'reanchor_k': self.reanchor_k,
                    'points': points,                               # F x (T_l, 4)
                    'offsets': [o.detach() for o in out_offsets],   # F x (B, T_l, 2)
                }
            # ---- ROOT READOUT (TIFAD_ROOT_READOUT=<json class->root map>,
            # EVAL ONLY; scripts/root_readout.py): detection at the ROOT
            # level of the tree (l_d = 1) WITHOUT retraining. The per-
            # location class logits over the current test vocabulary are
            # replaced by the aggregated ancestor scores u_a = logsumexp_{c
            # in children(a)} logit_c (the same aggregation FAMCE
            # supervises), one column per root in sorted root order; the
            # decode / soft-NMS / evaluator then run unchanged on the root
            # columns and the results carry the root names as class_names.
            # Byte-inert when unset.
            self._root_names = None
            _rr_map = _osp.environ.get('TIFAD_ROOT_READOUT', '')
            if _rr_map:
                _rr_classes = list(self.text_features.test_classes)
                if getattr(self, '_rr_key', None) != tuple(_rr_classes):
                    import json as _json
                    _rr_m = _json.load(open(_rr_map))
                    _rr_missing = [c for c in _rr_classes if c not in _rr_m]
                    assert not _rr_missing, \
                        f"[ROOT-READOUT] classes missing from map: {_rr_missing}"
                    _rr_roots = sorted({_rr_m[c] for c in _rr_classes})
                    self._rr_members = [
                        [i for i, c in enumerate(_rr_classes) if _rr_m[c] == r]
                        for r in _rr_roots]
                    self._rr_roots = _rr_roots
                    self._rr_key = tuple(_rr_classes)
                    print(f"[ROOT-READOUT] active: {len(_rr_classes)} classes -> "
                          f"{len(_rr_roots)} roots {_rr_roots}", flush=True)
                out_cls_logits = [
                    torch.stack([torch.logsumexp(lg[..., m], dim=-1)
                                 for m in self._rr_members], dim=-1)
                    for lg in out_cls_logits]
                self._root_names = list(self._rr_roots)
            results = self.inference(
                video_list, points, fpn_masks,
                out_cls_logits, out_offsets, out_fg, out_centerness,
                ccr_prior=ccr_prior,
                fineq_ctx=fineq_ctx,
                support_ctx=support_ctx,
                compl_ctx=compl_ctx,
                xm_ctx=_xm_ctx,
                qh_ctx=_qh_ctx,
                ph_ctx=_ph_ctx,
                bnd_ctx=_bnd_ctx,
            )

            return results


    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])   
        max_len = feats_lens.max(0).values.item()      

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len  
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len] 
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)
        return batched_inputs, batched_masks

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels, new_cls_num):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        concat_points = torch.cat(points, dim=0)
        gt_cls, gt_offset, gt_fg, gt_cn = [], [], [], []
        # [LEAFLOSS] collect the per-video point->GT assignment (stashed by
        # label_points_single_video) for _leaf_targets; inert when unset
        if getattr(self, 'leafloss', False):
            self._leaf_assign = []
        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            cls_targets, reg_targets, fg_targets, cn_targets = self.label_points_single_video(
                concat_points, gt_segment, gt_label, new_cls_num
            )
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)
            gt_fg.append(fg_targets)
            gt_cn.append(cn_targets)

        return gt_cls, gt_offset, gt_fg, gt_cn

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label, new_cls_num):
        # concat_points : F T x 4 (t, regressoin range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]     
        num_gts = gt_segment.shape[0]        

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, new_cls_num), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            fg_target = torch.zeros(num_pts, device=gt_label.device)
            cn_targets = torch.zeros(num_pts, device=gt_label.device)
            if getattr(self, 'leafloss', False):
                self._leaf_assign.append(None)          # [LEAFLOSS] no GT
            return cls_targets, reg_targets, fg_target, cn_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)  
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]      
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]     
        reg_targets = torch.stack((left, right), dim=-1)


        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])  
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask==0, float('inf'))
        lens.masked_fill_(inside_regress_range==0, float('inf'))
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)

        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2; video_target: C
        gt_label_one_hot = F.one_hot(
            gt_label, new_cls_num 
        ).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        
        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]
        
        # foreground target
        fg_target = cls_targets.sum(dim=-1)
        fg_target.clamp_(min=0.0, max=1.0)
        fg_target = fg_target.to(reg_targets.device)

        # centerness target
        mask = reg_targets.min(-1)[0] > 0
        sigma = 1.0
        gaussian_cn_targets = torch.exp(-0.5 * ((reg_targets / sigma) ** 2))
        gaussian_cn_values, _ = gaussian_cn_targets.min(dim=-1)
        gaussian_cn_targets = gaussian_cn_values * mask.float()

        # [LEAFLOSS] stash the point -> GT-segment assignment: the shortest
        # candidate segment (min_len_inds) wherever a point is positive
        # (min_len < inf). The cls targets above are untouched.
        if getattr(self, 'leafloss', False):
            self._leaf_assign.append(
                (min_len_inds, min_len < float('inf')))

        return cls_targets, reg_targets, fg_target, gaussian_cn_targets

    @torch.no_grad()
    def _leaf_targets(self, concat_points, gt_labels, video_list):
        """[LEAFLOSS] per-location phase targets, (B, FT, 2) long:
        column 0 = GT class index, column 1 = pseudo phase index 0..9;
        -1 where the location is not positive, or its assigned segment has
        no pseudo entry (NaN cuts). Phase index = number of the 9 internal
        cut points (end of phases 0..8, feature-grid units) at or before
        the point's time -> monotone non-decreasing in time inside a
        segment by construction; points just outside the pseudo interval
        cover (snippet-center jitter at the edges) clamp to phase 0 / 9."""
        ft = concat_points.size(0)
        out = []
        for b, gt_label in enumerate(gt_labels):
            tg = torch.full((ft, 2), -1, dtype=torch.long,
                            device=concat_points.device)
            asg = self._leaf_assign[b] if b < len(self._leaf_assign) else None
            cuts = video_list[b].get('leaf_cuts', None) \
                if isinstance(video_list[b], dict) else None
            if asg is not None and cuts is not None and cuts.numel() > 0:
                inds, pos = asg
                cuts = cuts.to(device=concat_points.device,
                               dtype=concat_points.dtype)      # (N, 9)
                pidx = torch.nonzero(pos, as_tuple=True)[0]     # (P,)
                j = inds[pidx]                                  # (P,)
                cj = cuts[j]                                    # (P, 9)
                has = ~torch.isnan(cj).any(dim=-1)              # (P,)
                t = concat_points[pidx, 0]                      # (P,)
                k = (t.unsqueeze(-1) >= cj).sum(dim=-1)         # (P,) 0..9
                sel = pidx[has]
                tg[sel, 0] = gt_label[j[has]].to(torch.long)
                tg[sel, 1] = k[has].to(torch.long)
                # optional debug (TIFAD_LEAFLOSS_DEBUG=<n segments>): assert
                # monotonicity of the phase index in time per segment
                if self._leaf_dbg_left > 0:
                    for jj in torch.unique(j[has]).tolist():
                        if self._leaf_dbg_left <= 0:
                            break
                        m = (j == jj) & has
                        tt, kk = t[m], k[m]
                        order = torch.argsort(tt)
                        kk_s = kk[order]
                        mono = bool((kk_s[1:] >= kk_s[:-1]).all()) \
                            if kk_s.numel() > 1 else True
                        assert mono, f"[LEAFLOSS] non-monotone phases {kk_s}"
                        print(f"[LEAFLOSS-DBG] vid={video_list[b].get('video_id')}"
                              f" seg#{jj} cls={int(gt_label[jj])} "
                              f"n_pts={int(m.sum())} t(grid)="
                              f"[{tt.min().item():.1f},{tt.max().item():.1f}]"
                              f" phases(sorted by t)={kk_s.tolist()} "
                              f"monotone={mono}", flush=True)
                        self._leaf_dbg_left -= 1
            out.append(tg)
        return torch.stack(out)

    def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls_labels, gt_offsets, new_cls_num,
        out_fg, gt_fg,
        out_centerness, gt_cn,
        haf_logits=None,
        aspec_logits=None,
        sibcon_feat0=None,
        sibcon_text0=None,
        vattr_feats=None,
        poseev_logits=None,
        leaf_targets=None,
        leaf_feats=None,
        align_feats=None,
        align_et=None,
        align_valid=None,
        dfl_logits=None,
        dfl_bins=None,
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # dfl_logits: F x (B, T_l, 2, nbins) bin logits (TIFAD_DFL) or None;
        # dfl_bins: F x (nbins,) bin centres per level
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)    
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]
        # ---- DFL (TIFAD_DFL): the positives' bin logits and bin centres,
        # row-aligned with pred_offsets (same pos_mask; the optional
        # CLAMP_REG row filter below is applied to them too)
        dfl_pos = dfl_bins_pos = None
        if dfl_logits is not None:
            dfl_pos = torch.cat(dfl_logits, dim=1)[pos_mask]       # (P, 2, n)
            _dfb = torch.cat([b.unsqueeze(0).expand(m.size(1), -1)
                              for b, m in zip(dfl_bins, fpn_masks)], dim=0)
            dfl_bins_pos = _dfb.unsqueeze(0).expand(
                pos_mask.size(0), -1, -1)[pos_mask]                # (P, n)

        # ---- IOUQ (TIFAD_IOUQ=1): per-positive-point tIoU between the
        # DECODED segment and the matched GT segment, used further below as
        # the centerness-branch target. Both offset tensors are in
        # stride-normalized (l, r) units around the SAME point, so the 1-D
        # IoU is computed directly in offset space (scale cancels):
        #   inter = min(l_p, l_g) + min(r_p, r_g)
        #   union = max(l_p, l_g) + max(r_p, r_g)
        # Predicted offsets are DETACHED: the quality head must score the
        # regressor's output, not steer it -- backpropagating through the
        # offsets would add a second, conflicting gradient path into the
        # regressor (the DIoU reg loss already owns that signal) and would
        # let the regressor drift toward extents that are merely easy to
        # score. Computed BEFORE the optional TIFAD_CLAMP_REG row filtering
        # so indices stay aligned with pos_mask.
        iouq_targets = None
        if getattr(self, 'iouq_on', False):
            with torch.no_grad():
                _po = pred_offsets.detach().clamp(min=0.0)   # (#Pos, 2)
                _go = gt_offsets.clamp(min=0.0)              # (#Pos, 2)
                _inter = torch.min(_po, _go).sum(-1)
                _union = torch.max(_po, _go).sum(-1)
                iouq_targets = (
                    _inter / _union.clamp(min=1e-6)).clamp(0.0, 1.0)
        # TIFAD_CLAMP_REG=1: clamp rare negative GT offsets (center-sampling
        # radius windows extending past sub-radius segments, e.g. FineGym
        # elements) to the segment edge. Inert when unset.
        import os as _oscr
        if _oscr.environ.get('TIFAD_CLAMP_REG', '') == '1':
            gt_offsets = gt_offsets.clamp(min=0.0)
            # drop degenerate targets (both offsets ~0 after clamp): their
            # DIoU union is ~0 -> NaN gradients -> NaN predictions.
            _keep = gt_offsets.sum(-1) > 1e-3
            gt_offsets = gt_offsets[_keep]
            pred_offsets = pred_offsets[_keep]
            if dfl_pos is not None:
                dfl_pos = dfl_pos[_keep]
                dfl_bins_pos = dfl_bins_pos[_keep]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)
        self.loss_normalizer2 = max(num_pos, 1)  

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]

        # optinal label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (new_cls_num + 1)

        # focal loss
        import os as _ospl
        _pda_loss = _ospl.environ.get('TIFAD_PDA_LOSS', '')
        if _pda_loss == 'bce':
            # PDA reproduction variant: plain (sigmoid) cross-entropy, no
            # focal modulation, same normaliser as the fork's focal loss
            cls_loss = F.binary_cross_entropy_with_logits(
                torch.cat(out_cls_logits, dim=1)[valid_mask],
                gt_target, reduction='sum')
        elif _pda_loss == 'softmaxce':
            # PDA reproduction variant: softmax cross-entropy over classes on
            # the positive time steps only (one-hot targets, Eq. 8)
            _lg = torch.cat(out_cls_logits, dim=1)[valid_mask]
            _pos = gt_target.sum(dim=-1) > 0
            _tgt = gt_target.argmax(dim=-1)
            cls_loss = F.cross_entropy(_lg[_pos], _tgt[_pos], reduction='sum') \
                if _pos.any() else 0 * _lg.sum()
        else:
            cls_loss = sigmoid_focal_loss(
                torch.cat(out_cls_logits, dim=1)[valid_mask],
                gt_target,
                reduction='sum'
            )
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer
        # ---- DFL loss (TIFAD_DFL, GFL eq. 6): for each side the two bins
        # bracketing the target y get cross-entropy mass (b_{j+1}-y)/d and
        # (y-b_j)/d; targets beyond the top bin clip to it. Sum over
        # positives x 2 sides / loss_normalizer x TIFAD_DFL_W (0.25, GFL).
        dfl_loss = None
        if dfl_pos is not None:
            if num_pos == 0 or dfl_pos.size(0) == 0:
                dfl_loss = 0 * dfl_pos.sum()
            else:
                _n = dfl_pos.size(-1)
                _d = (dfl_bins_pos[:, 1] - dfl_bins_pos[:, 0]).clamp(min=1e-6)
                _y = gt_offsets.clamp(min=0.0).float()
                _y = torch.minimum(_y, dfl_bins_pos[:, -1:].expand(-1, 2))
                _j = torch.div(_y, _d.unsqueeze(1), rounding_mode='floor')
                _j = _j.long().clamp(0, _n - 2)                    # (P, 2)
                _lo = _j * _d.unsqueeze(1)
                _whi = ((_y - _lo) / _d.unsqueeze(1)).clamp(0.0, 1.0)
                _lsm = F.log_softmax(dfl_pos.float(), dim=-1)      # (P, 2, n)
                _lj = _lsm.gather(2, _j.unsqueeze(-1)).squeeze(-1)
                _lj1 = _lsm.gather(2, (_j + 1).unsqueeze(-1)).squeeze(-1)
                dfl_loss = -((1.0 - _whi) * _lj + _whi * _lj1).sum()
                dfl_loss = self.dfl_w * dfl_loss / self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # 3. foreground loss
        pred_fg = torch.cat(out_fg, dim=1).squeeze(-1)
        fg_loss = sigmoid_focal_loss(
            pred_fg[valid_mask],
            torch.stack(gt_fg)[valid_mask],
            reduction='sum'
        )
        fg_loss /= self.loss_normalizer 

        # 4. centerness loss
        gt_cn_all = torch.stack(gt_cn)                       # (B, FT)
        # IOUQ: overwrite POSITIVE-point targets with the decoded-segment
        # tIoU computed above (soft targets in [0,1] -- same sigmoid focal
        # loss form the gaussian centerness targets already use, i.e.
        # quality-focal in spirit). Negatives keep their existing targets.
        if iouq_targets is not None and num_pos > 0:
            gt_cn_all = gt_cn_all.clone()
            _new_cn = iouq_targets.to(gt_cn_all.dtype)
            # IOUQ_MINLVL: only positives on FPN levels >= minlvl take the
            # IoU target; finer-level positives (short actions, where small
            # boundary errors give harshly low IoU) keep the gaussian
            # target. Level id per point mirrors the fpn_masks / gt_cn
            # concat-over-levels order (fpn_masks[l] is (B, T_l)).
            if self.iouq_minlvl > 0:
                _lvl = torch.cat([
                    torch.full((m.shape[1],), l, dtype=torch.long,
                               device=pos_mask.device)
                    for l, m in enumerate(fpn_masks)])           # (FT,)
                _pos_lvl = _lvl.unsqueeze(0).expand(
                    pos_mask.shape)[pos_mask]                    # (#Pos,)
                _new_cn = torch.where(
                    _pos_lvl >= self.iouq_minlvl,
                    _new_cn, gt_cn_all[pos_mask])
            gt_cn_all[pos_mask] = _new_cn
        cn_loss = sigmoid_focal_loss(
            torch.cat(out_centerness, dim=1)[valid_mask].squeeze(-1),
            gt_cn_all[valid_mask],
            reduction='sum'
        )
        cn_loss /= self.loss_normalizer

        # ---- HAF auxiliary foreground loss (TIFAD_HAF=<w>, TRAIN ONLY):
        # focal loss on the hierarchy-anchor logsumexp logits against the
        # SAME fg targets / valid-mask plumbing as fg_loss above. Inert
        # (haf_logits is None) when TIFAD_HAF unset.
        haf_loss = None
        if haf_logits is not None and self.haf_w != 0:
            pred_haf = torch.cat(haf_logits, dim=1)            # (B, FT)
            haf_loss = self.haf_w * sigmoid_focal_loss(
                pred_haf[valid_mask],
                torch.stack(gt_fg)[valid_mask],
                reduction='sum'
            ) / self.loss_normalizer

        # ---- FAMCE (hierarchical multi-task supervision, TRAIN ONLY):
        # auxiliary FAMILY-level (superclass) focal loss migrated from
        # hierarchical text classification: family logit = logsumexp over its
        # member-class logit columns, family target = max over the member
        # columns of the (possibly soft) cls targets. Supervising the family
        # decision surface lets it transfer to unseen leaf classes.
        # Env: TIFAD_FAMCE=<float weight>,
        #      TIFAD_FAMCE_MAP=<json path {class_name: family_name}>.
        # Inert (byte-identical) when unset.
        import os as _osfc
        _fw = float(_osfc.environ.get('TIFAD_FAMCE', '0') or 0)
        fam_ce_loss = None
        if _fw > 0 and self.training:
            _classes = list(self.text_features.train_classes)
            if getattr(self, '_famce_key', None) != tuple(_classes):
                import json as _json
                _map = _json.load(open(_osfc.environ['TIFAD_FAMCE_MAP']))
                _missing = [c for c in _classes if c not in _map]
                if _missing:
                    raise KeyError(
                        "[FAMCE] train classes missing from TIFAD_FAMCE_MAP "
                        f"({_osfc.environ['TIFAD_FAMCE_MAP']}): {_missing}")
                _fams = sorted({_map[c] for c in _classes})
                _fid = {f: i for i, f in enumerate(_fams)}
                _fam_idx = torch.tensor(
                    [_fid[_map[c]] for c in _classes], dtype=torch.long)
                # per-family member column indices, in logit-column order
                self._famce_members = [
                    torch.nonzero(_fam_idx == f, as_tuple=True)[0]
                    for f in range(len(_fams))
                ]
                self._famce_key = tuple(_classes)
                print(f"[FAMCE] weight={_fw}, F={len(_fams)}", flush=True)
            # same concat-over-levels + valid_mask indexing as cls_loss above
            _lg = torch.cat(out_cls_logits, dim=1)[valid_mask]     # (N, C)
            if _lg.size(-1) != len(_classes):
                raise RuntimeError(
                    f"[FAMCE] logit columns ({_lg.size(-1)}) != "
                    f"#train classes ({len(_classes)})")
            _tg = gt_cls[valid_mask]                               # (N, C) pre-smoothing
            fam_logits = torch.stack(
                [torch.logsumexp(_lg[:, m.to(_lg.device)], dim=-1)
                 for m in self._famce_members], dim=-1)            # (N, F)
            fam_targets = torch.stack(
                [_tg[:, m.to(_tg.device)].amax(dim=-1)
                 for m in self._famce_members], dim=-1)            # (N, F)
            # ---- FAMCE_MARGCONS (TIFAD_FAMCE_MARGCONS=1): replace the
            # independent family focal CE by MARGINALIZATION-CONSISTENCY
            # (Test-Time-Amendment style, NeurIPS'23): the family "prob" is
            # the SUM of its member leaf sigmoid probs (clamped to [eps,
            # 1-eps]) -- no separately-normalized coarse head -- scored with
            # the Jensen-Shannon divergence between the Bernoulli(fam_p) and
            # Bernoulli(fam_target) distributions. Same _fw weight /
            # loss_normalizer plumbing as the focal path it replaces.
            if _osfc.environ.get('TIFAD_FAMCE_MARGCONS', '') == '1':
                _eps = 1e-6
                _pl = torch.sigmoid(_lg)                           # (N, C)
                _fam_p = torch.stack(
                    [_pl[:, m.to(_pl.device)].sum(dim=-1)
                     for m in self._famce_members], dim=-1
                ).clamp(_eps, 1.0 - _eps)                          # (N, F)
                _fam_t = fam_targets.clamp(0.0, 1.0)
                _mid = 0.5 * (_fam_p + _fam_t)
                _xly = torch.special.xlogy
                _kl_t = (_xly(_fam_t, _fam_t) - _xly(_fam_t, _mid)
                         + _xly(1 - _fam_t, 1 - _fam_t)
                         - _xly(1 - _fam_t, (1 - _mid).clamp(min=_eps)))
                _kl_p = (_xly(_fam_p, _fam_p) - _xly(_fam_p, _mid)
                         + _xly(1 - _fam_p, 1 - _fam_p)
                         - _xly(1 - _fam_p, (1 - _mid).clamp(min=_eps)))
                fam_ce_loss = _fw * (0.5 * (_kl_t + _kl_p)).sum() \
                    / self.loss_normalizer
            else:
                fam_ce_loss = _fw * sigmoid_focal_loss(
                    fam_logits, fam_targets, reduction='sum'
                ) / self.loss_normalizer

            # ---- gradient-proxy variants (COSGATE / PCGRAD): both use the
            # LOGIT-gradient proxy -- grads of each loss wrt the designated
            # shared tensors out_cls_logits (the only main-loss term that
            # shares parameters with famce through the logits is cls_loss,
            # so g_main := d cls_loss / d logits). autograd.grad only
            # traverses the cheap loss-side subgraph (cat/mask/focal), NOT
            # the backbone, so both cost ~2 tiny extra backwards. Documented
            # DOC-CHOICE: full-parameter-space grads would need two full
            # backbone backwards per step; the logit-level proxy is the
            # sanctioned fallback.
            _cosgate = _osfc.environ.get('TIFAD_FAMCE_COSGATE', '') == '1'
            _pcgrad = _osfc.environ.get('TIFAD_FAMCE_PCGRAD', '') == '1'
            if ((_cosgate or _pcgrad) and fam_ce_loss.requires_grad
                    and out_cls_logits[0].requires_grad):
                _gm = torch.autograd.grad(
                    cls_loss, out_cls_logits,
                    retain_graph=True, allow_unused=True)
                _ga = torch.autograd.grad(
                    fam_ce_loss, out_cls_logits,
                    retain_graph=True, allow_unused=True)
                _dot = _lg.new_zeros(())
                _nm = _lg.new_zeros(())
                _na = _lg.new_zeros(())
                for _m_, _a_ in zip(_gm, _ga):
                    if _m_ is None or _a_ is None:
                        continue
                    _dot = _dot + (_m_ * _a_).sum()
                    _nm = _nm + _m_.pow(2).sum()
                    _na = _na + _a_.pow(2).sum()
                _cos = _dot / (_nm.sqrt() * _na.sqrt() + 1e-12)
                self._famce_cos_diag = _cos.detach()
                if _cosgate and _cos.item() <= 0:
                    # Du et al. 1812.02224: drop the aux loss on conflict
                    # (cos <= 0); never touch the main gradient.
                    fam_ce_loss = fam_ce_loss.detach() * 0.0
                elif _pcgrad and _dot.item() < 0:
                    # asymmetric PCGrad: project the FAMCE logit-gradient
                    # onto the normal plane of the main logit-gradient;
                    # realized as a linear surrogate whose gradient wrt the
                    # logits is exactly the projected gradient (value kept
                    # at the original famce value for logging).
                    _coef = (_dot / _nm.clamp(min=1e-12)).detach()
                    _sur = _lg.new_zeros(())
                    for _o_, _m_, _a_ in zip(out_cls_logits, _gm, _ga):
                        if _a_ is None:
                            continue
                        _gp = (_a_ - _coef * (_m_ if _m_ is not None
                                              else 0.0)).detach()
                        _sur = _sur + (_o_ * _gp).sum()
                    fam_ce_loss = (_sur - _sur.detach()
                                   + fam_ce_loss.detach())

        # ---- LEAFLOSS (TIFAD_LEAFLOSS=1, TRAIN ONLY; see __init__): the
        # mirror image of FAMCE BELOW the detected action. Positive
        # locations carrying a pseudo phase target (leaf_targets[..., 1]
        # >= 0, valid) are scored against the 10 ordered phase embeddings
        # of THEIR GT class with the base cls-matmul calibration
        # (L2-normalize, text_proj(x/2), feat . e / logit_scale.exp(); the
        # per-class bias is constant across phases and drops out of the
        # softmax), softmax CE vs. the pseudo phase index, summed and
        # divided by loss_normalizer, times TIFAD_LEAFLOSS_W. Gradients
        # reach the trunk (cls_feat), text_proj / logit_scale and the
        # phase-sentence path of text.py. Inert (leaf_targets None) when
        # unset. Also logs leaf_frac = labeled positives / positives.
        leaf_loss = leaf_frac = leaf_ce_mean = leaf_acc = None
        if (getattr(self, 'leafloss', False) and leaf_targets is not None
                and leaf_feats is not None and self.training):
            _lp = getattr(self.text_features, '_leaf_phase_embs', None)
            if _lp is None:
                raise RuntimeError(
                    "[LEAFLOSS] phase stash missing (text.py ensemble path "
                    "not run in train mode?)")
            self.text_features._leaf_phase_embs = None   # consume the stash
            if _lp.size(0) != gt_cls.size(-1) or _lp.size(1) != 10:
                raise RuntimeError(
                    f"[LEAFLOSS] phase bank {tuple(_lp.shape)} vs "
                    f"#train classes {gt_cls.size(-1)} x 10")
            _lsel = torch.logical_and(leaf_targets[..., 1] >= 0, valid_mask)
            _ln = int(_lsel.sum().item())
            leaf_frac = torch.tensor(
                _ln / max(num_pos, 1), device=gt_cls.device)
            # same calibration as the base cls matmul (VPW recipe)
            _lpn = _lp / _lp.norm(dim=-1, keepdim=True)
            _lpp = self.text_proj(_lpn / 2.0)                  # (C, 10, D)
            if _ln > 0:
                _lq = leaf_feats[_lsel]                          # (P, D)
                _lc = leaf_targets[..., 0][_lsel]                # (P,)
                _lk = leaf_targets[..., 1][_lsel]                # (P,)
                if self.leaf_head is not None:
                    # TIFAD_LEAFLOSS_DETACH=1: the phase head reads the trunk
                    # features but sends no gradient into the trunk, so the
                    # detection objective is untouched by construction
                    if __import__('os').environ.get('TIFAD_LEAFLOSS_DETACH', '') == '1':
                        _lq = _lq.detach()
                    _lg_ph = self.leaf_head(_lq)                 # (P, 10)
                else:
                    _lg_ph = torch.einsum(
                        'pd,pkd->pk', _lq, _lpp[_lc]) / self.logit_scale.exp()
                _lce = F.cross_entropy(_lg_ph.float(), _lk, reduction='sum')
                leaf_loss = _lce / self.loss_normalizer
                # diagnostics: mean per-location CE (chance = ln 10 = 2.30)
                # and top-1 phase accuracy over the labeled positives
                leaf_ce_mean = (_lce / _ln).detach()
                leaf_acc = (_lg_ph.argmax(-1) == _lk).float().mean().detach()
            else:
                # keep the graph alive (unused-param safety)
                leaf_loss = 0.0 * (leaf_feats.sum() + _lpp.sum())
                if self.leaf_head is not None:
                    leaf_loss = leaf_loss + 0.0 * sum(
                        p.sum() for p in self.leaf_head.parameters())
            leaf_loss = self.leafloss_w * leaf_loss

        # ---- VSA-ALIGN loss (TIFAD_VSA_ALIGN_W=<w>, TRAIN ONLY; vsa.py
        # (E)): at every positive location (pos_mask; class = the one-hot
        # gt_cls argmax) chunk k is scored against the 10 phases j of its
        # GT class through the XM adaptors (both sides L2-normed inside
        # vsa.py), cos / 0.07, softmax CE with target j = k, MEAN over k,
        # SUM over positives / loss_normalizer, times w. align_acc =
        # fraction of (positive, chunk) pairs whose argmax phase is k
        # (chance 0.1). Direct add to final_loss. Inert when unset.
        align_loss = align_acc = None
        if align_feats is not None and align_et is not None:
            _apm = pos_mask if align_valid is None \
                else torch.logical_and(pos_mask, align_valid)   # MINLVL
            _an = int(_apm.sum().item())
            if _an > 0:
                _az = align_feats[_apm]                          # (P, K, D)
                _ac = gt_cls[_apm].argmax(-1)                    # (P,)
                _ae = align_et[_ac]                              # (P, K, D)
                _al = torch.einsum('pkd,pjd->pkj', _az, _ae) / 0.07
                _aK = _al.size(1)
                _atg = torch.arange(_aK, device=_al.device).unsqueeze(0) \
                    .expand(_an, _aK)
                _ace = F.cross_entropy(_al.reshape(-1, _aK).float(),
                                       _atg.reshape(-1), reduction='sum') / _aK
                align_loss = self.vsa_align_w * _ace / self.loss_normalizer
                align_acc = (_al.argmax(-1) == _atg).float().mean().detach()
            else:
                align_loss = 0.0 * (align_feats.sum() + align_et.sum())

        # ---- ASPEC train loss (TIFAD_ASPEC=<w>, TRAIN ONLY): multi-label
        # attribute targets = union of the GT class's attribute set at each
        # point (background points get all-zero targets), scored with the
        # SAME valid_mask / loss_normalizer plumbing as cls_loss above.
        # Targets use the PRE-smoothing one-hot gt_cls. Inert (aspec_logits
        # is None) when TIFAD_ASPEC / TIFAD_ASPEC_ATTRS unset.
        aspec_loss = None
        if aspec_logits is not None and self.aspec_w != 0:
            _AM = self._aspec_matrix(
                list(self.text_features.train_classes), gt_cls.device)
            if gt_cls.size(-1) != _AM.size(0):
                raise RuntimeError(
                    f"[ASPEC] gt_cls columns ({gt_cls.size(-1)}) != "
                    f"#train classes ({_AM.size(0)})")
            _attr_tg = (gt_cls[valid_mask] @ _AM).clamp(max=1.0)   # (N, A)
            aspec_loss = self.aspec_w * sigmoid_focal_loss(
                torch.cat(aspec_logits, dim=1)[valid_mask],
                _attr_tg,
                reduction='sum'
            ) / self.loss_normalizer

        # ---- SIBCE (within-parent conditional softmax, TRAIN ONLY): at
        # positive points, cross-entropy with the softmax RESTRICTED to the
        # GT class's family (same-family train classes; families need >= 2
        # members -- singleton-family positives are skipped). Directly
        # maximizes the sibling margin (Making-Better-Mistakes conditional
        # factorization; failure-(b) fix). Uses the SAME pos_mask /
        # loss_normalizer plumbing as cls_loss above.
        # Env: TIFAD_SIBCE=<w>; family map json path reused from
        # TIFAD_HTCA_FAMS (default configs/thumos14_fam_map.json; setting
        # that var alone is inert in text.py -- banks need TIFAD_HTCA/HAF).
        # Inert (byte-identical) when TIFAD_SIBCE unset.
        import os as _ossb
        _sibce_w = float(_ossb.environ.get('TIFAD_SIBCE', '0') or 0)
        _sib_fams = _ossb.environ.get(
            'TIFAD_HTCA_FAMS', 'configs/thumos14_fam_map.json')
        sibce_loss = None
        if _sibce_w != 0:
            _classes = list(self.text_features.train_classes)
            _lg_all = torch.cat(out_cls_logits, dim=1)          # (B, FT, C)
            if _lg_all.size(-1) != len(_classes):
                raise RuntimeError(
                    f"[SIBCE] logit columns ({_lg_all.size(-1)}) != "
                    f"#train classes ({len(_classes)})")
            _groups, _lut = self._sib_family_groups(
                _classes, _sib_fams, _lg_all.device)
            _lg_pos = _lg_all[pos_mask]                          # (P, C)
            # one GT class per positive point (pre-smoothing one-hot;
            # argmax breaks rare multi-label ties by lowest column)
            _gt_idx = gt_cls[pos_mask].argmax(dim=-1)            # (P,)
            _gid = _lut[_gt_idx, 0]
            _pid = _lut[_gt_idx, 1]
            sibce_loss = 0.0 * _lg_pos.sum()
            for _g, _mem in enumerate(_groups):
                _sel = _gid == _g
                if not bool(_sel.any()):
                    continue
                sibce_loss = sibce_loss + F.cross_entropy(
                    _lg_pos[_sel][:, _mem], _pid[_sel], reduction='sum')
            sibce_loss = _sibce_w * sibce_loss / self.loss_normalizer

        # ---- SIBCON (sibling-hard-negative InfoNCE, TRAIN ONLY): at LEVEL-0
        # positive points, contrast the snippet feature column against the
        # matmul-space class text vectors -- positive = GT class vector,
        # negatives = ONLY its same-family siblings (cosine sims, temperature
        # 0.1). Same family map / valid-mask / normalizer plumbing as SIBCE;
        # singleton-family positives are skipped.
        # Env: TIFAD_SIBCON=<w>. Inert (sibcon_feat0 is None) when unset.
        sibcon_loss = None
        _sibcon_w = float(_ossb.environ.get('TIFAD_SIBCON', '0') or 0)
        if _sibcon_w != 0 and sibcon_feat0 is not None:
            _classes = list(self.text_features.train_classes)
            if sibcon_text0.size(1) != len(_classes):
                raise RuntimeError(
                    f"[SIBCON] text rows ({sibcon_text0.size(1)}) != "
                    f"#train classes ({len(_classes)})")
            _groups, _lut = self._sib_family_groups(
                _classes, _sib_fams, sibcon_feat0.device)
            _T0 = fpn_masks[0].size(1)
            _pos0 = pos_mask[:, :_T0]                            # (B, T0)
            _fp = F.normalize(
                sibcon_feat0.transpose(1, 2)[_pos0], dim=-1)     # (P0, D)
            _b0 = torch.nonzero(_pos0, as_tuple=True)[0]         # (P0,)
            _tx = F.normalize(sibcon_text0, dim=-1)[_b0]         # (P0, C, D)
            _sim = torch.einsum('nd,ncd->nc', _fp, _tx) / 0.1    # (P0, C)
            _gt0 = gt_cls[:, :_T0][_pos0].argmax(dim=-1)         # (P0,)
            _gid = _lut[_gt0, 0]
            _pid = _lut[_gt0, 1]
            sibcon_loss = 0.0 * (_fp.sum() + _tx.sum())
            for _g, _mem in enumerate(_groups):
                _sel = _gid == _g
                if not bool(_sel.any()):
                    continue
                sibcon_loss = sibcon_loss + F.cross_entropy(
                    _sim[_sel][:, _mem], _pid[_sel], reduction='sum')
            sibcon_loss = _sibcon_w * sibcon_loss / self.loss_normalizer

        # ---- VATTR aux (TIFAD_VATTR=1, TRAIN ONLY; see __init__):
        # (a) BCEWithLogits(vattr_head(stopgrad(feat)), code[GT class]) at
        #     the SAME positive points as the focal loss (pos_mask over the
        #     level-concatenated grid) -- trunk features DETACHED, so the
        #     trunk receives ZERO gradient from this aux (VIDPHASE_DETACH
        #     lesson). Weak labels: every positive point inherits its GT
        #     class's binary attribute-code row (multi-label ties -> union,
        #     clamped to 1; ASPEC precedent). Only classes present in the
        #     attrs json contribute (points whose GT classes are all
        #     missing are dropped).
        # (b) orthogonality on the head rows:
        #     ||W_n W_n^T - I||_F^2 / V^2, W_n = row-L2-normalized weight.
        # Both scaled by the linear decay factor max(0, 1 - step/decay)
        # when TIFAD_VATTR_DECAY_STEPS is set (VIDPHASE decay pattern:
        # per-forward counter). Inert (vattr_feats is None) when unset.
        vattr_loss = None
        if (self.vattr_head is not None and vattr_feats is not None
                and self.training):
            _va_w_eff = 1.0
            if self.vattr_decay_steps > 0:
                _va_w_eff = max(
                    0.0, 1.0 - self._vattr_step / self.vattr_decay_steps)
                self._vattr_step += 1
            if _va_w_eff > 0:
                _classes = list(self.text_features.train_classes)
                if gt_cls.size(-1) != len(_classes):
                    raise RuntimeError(
                        f"[VATTR] gt_cls columns ({gt_cls.size(-1)}) != "
                        f"#train classes ({len(_classes)})")
                _va_code, _va_pres = self._vattr_code_matrix(_classes)
                _va_code = _va_code.to(device=gt_cls.device,
                                       dtype=vattr_feats.dtype)   # (C, V)
                _va_pres = _va_pres.to(device=gt_cls.device,
                                       dtype=_va_code.dtype)      # (C,)
                _va_gt = gt_cls[pos_mask]                         # (P, C)
                _va_keep = (_va_gt @ _va_pres) > 0                # (P,)
                if bool(_va_keep.any()):
                    _va_tg = (_va_gt[_va_keep] @ _va_code).clamp(
                        max=1.0)                                  # (P', V)
                    # stop-gradient trunk features -> live linear head
                    _va_lg = self.vattr_head(
                        vattr_feats.detach()[pos_mask][_va_keep])
                    _va_bce = F.binary_cross_entropy_with_logits(
                        _va_lg, _va_tg, reduction='mean')
                else:
                    _va_bce = 0.0 * self.vattr_head.weight.sum()
                _va_Wn = F.normalize(self.vattr_head.weight, dim=1)  # (V, D)
                _va_G = _va_Wn @ _va_Wn.t()
                _va_I = torch.eye(_va_G.size(0), device=_va_G.device,
                                  dtype=_va_G.dtype)
                _va_orth = (_va_G - _va_I).pow(2).sum() \
                    / float(_va_G.size(0) ** 2)
                vattr_loss = _va_w_eff * (self.vattr_w * _va_bce
                                          + self.vattr_orth * _va_orth)

        # ---- POSEEV aux (TIFAD_POSEEV=1, TRAIN ONLY; see __init__):
        # BCEWithLogits of the pose attribute head at the SAME positive
        # points as the focal loss (pos_mask over the level-concatenated
        # grid), restricted to the FACET SUBSET columns; weak labels = the
        # GT class's binary code row (multi-label ties -> union, clamped;
        # VATTR precedent), classes absent from the attrs json dropped.
        # NO detach here (deliberate, unlike VATTR): the pose branch takes
        # the raw pose tensor as input and shares ZERO parameters with the
        # detection trunk, so backprop through poseev_logits reaches only
        # poseev_enc / poseev_head -- interference with the trunk is
        # structurally impossible. Weight TIFAD_POSEEV_W (default 0.1).
        # Inert (poseev_logits is None) when unset.
        poseev_loss = None
        if (self.poseev_head is not None and poseev_logits is not None
                and self.training):
            _classes = list(self.text_features.train_classes)
            if gt_cls.size(-1) != len(_classes):
                raise RuntimeError(
                    f"[POSEEV] gt_cls columns ({gt_cls.size(-1)}) != "
                    f"#train classes ({len(_classes)})")
            _pe_code, _pe_pres = self._poseev_code_matrix(_classes)
            _pe_code = _pe_code.to(device=gt_cls.device,
                                   dtype=poseev_logits.dtype)   # (C, V)
            _pe_presf = _pe_pres.to(device=gt_cls.device,
                                    dtype=_pe_code.dtype)       # (C,)
            _pe_fidx = self._poseev_fidx.to(gt_cls.device)
            _pe_gt = gt_cls[pos_mask]                           # (P, C)
            _pe_keep = (_pe_gt @ _pe_presf) > 0                 # (P,)
            if bool(_pe_keep.any()):
                _pe_tg = (_pe_gt[_pe_keep] @ _pe_code).clamp(
                    max=1.0)[:, _pe_fidx]                       # (P', Vf)
                _pe_lg = poseev_logits[pos_mask][_pe_keep][:, _pe_fidx]
                _pe_bce = F.binary_cross_entropy_with_logits(
                    _pe_lg, _pe_tg, reduction='mean')
            else:
                # keep the branch graph alive (DDP unused-param safety)
                _pe_bce = 0.0 * poseev_logits.sum()
            poseev_loss = self.poseev_w * _pe_bce

        # return a dict of losses
        final_loss = 0.5 * cls_loss + reg_loss +  fg_loss +  cn_loss
        if dfl_loss is not None:
            # DFL joins the main detection term (inside the Kendall 'main'
            # slot under FAMCE-UW, like reg_loss)
            final_loss = final_loss + dfl_loss
        losses_out = {'cls_loss'   : cls_loss,
                'reg_loss'   : reg_loss,
                'fg_loss' : fg_loss,
                'cn_loss' : cn_loss,
                'final_loss' : final_loss}
        if dfl_loss is not None:
            losses_out['dfl_loss'] = dfl_loss
        if fam_ce_loss is not None:
            losses_out['fam_ce_loss'] = fam_ce_loss
            if self.famce_uw_logvar is not None:
                # Kendall uncertainty weighting over {main, famce}
                # (TIFAD_FAMCE_UW=1): exp(-s)*L + s/2 per task. s init 0 =>
                # exactly final_loss + fam_ce_loss at step 0.
                _s = self.famce_uw_logvar
                losses_out['final_loss'] = (
                    torch.exp(-_s[0]) * final_loss + 0.5 * _s[0]
                    + torch.exp(-_s[1]) * fam_ce_loss + 0.5 * _s[1])
                losses_out['uw_s_main'] = _s[0].detach()
                losses_out['uw_s_fam'] = _s[1].detach()
                # [FAMCE-UW3] Kendall over {main, famce, vattr}
                # (TIFAD_FAMCE_UW3=1): third logvar entry weights the
                # vattr aux the same way -- exp(-s2)*vattr + s2/2 --
                # REPLACING the fixed direct add below (vattr_w remains
                # the inner scale inside vattr_loss). Consumes vattr_loss
                # so the plain-add branch further down is skipped.
                if getattr(self, 'famce_uw3', False) \
                        and vattr_loss is not None:
                    losses_out['final_loss'] = (
                        losses_out['final_loss']
                        + torch.exp(-_s[2]) * vattr_loss + 0.5 * _s[2])
                    losses_out['uw_s_vattr'] = _s[2].detach()
                    losses_out['vattr_loss'] = vattr_loss
                    vattr_loss = None
            else:
                losses_out['final_loss'] = final_loss + fam_ce_loss
            if getattr(self, '_famce_cos_diag', None) is not None:
                losses_out['famce_cos'] = self._famce_cos_diag
        # ---- LEAFLOSS combination (TIFAD_LEAFLOSS=1): with FAMCE-UW the
        # third logvar entry (created in __init__ only under LEAFLOSS)
        # weights the already w-scaled leaf loss -- exp(-s2)*leaf + s2/2 --
        # mirroring the famce term above, so the objective is
        #   exp(-s0)*L_det + s0/2 + exp(-s1)*w_act*L_act + s1/2
        #   + exp(-s2)*w_leaf*L_leaf + s2/2;
        # without FAMCE-UW the weighted leaf loss is added directly.
        if leaf_loss is not None:
            losses_out['leaf_loss'] = leaf_loss
            if leaf_frac is not None:
                losses_out['leaf_frac'] = leaf_frac
            if leaf_ce_mean is not None:
                losses_out['leaf_ce'] = leaf_ce_mean
                losses_out['leaf_acc'] = leaf_acc
            if (self.famce_uw_logvar is not None
                    and self.famce_uw_logvar.numel() >= 3
                    and not getattr(self, 'famce_uw3', False)):
                _s = self.famce_uw_logvar
                losses_out['final_loss'] = (
                    losses_out['final_loss']
                    + torch.exp(-_s[2]) * leaf_loss + 0.5 * _s[2])
                losses_out['uw_s_leaf'] = _s[2].detach()
            else:
                losses_out['final_loss'] = losses_out['final_loss'] + leaf_loss
        if align_loss is not None:
            losses_out['align_loss'] = align_loss
            if align_acc is not None:
                losses_out['align_acc'] = align_acc
            # VSA-ALIGN-UW: 4th Kendall slot, exp(-s3)*w*L_align + s3/2
            # (leaf stays on slot 2); else direct add.
            if (getattr(self, 'vsa_align_uw', False)
                    and self.famce_uw_logvar is not None
                    and self.famce_uw_logvar.numel() >= 4):
                _s = self.famce_uw_logvar
                losses_out['final_loss'] = (
                    losses_out['final_loss']
                    + torch.exp(-_s[3]) * align_loss + 0.5 * _s[3])
                losses_out['uw_s_align'] = _s[3].detach()
            else:
                losses_out['final_loss'] = losses_out['final_loss'] + align_loss
        if haf_loss is not None:
            losses_out['haf_loss'] = haf_loss
            losses_out['final_loss'] = losses_out['final_loss'] + haf_loss
        if aspec_loss is not None:
            losses_out['aspec_loss'] = aspec_loss
            losses_out['final_loss'] = losses_out['final_loss'] + aspec_loss
        if sibce_loss is not None:
            losses_out['sibce_loss'] = sibce_loss
            losses_out['final_loss'] = losses_out['final_loss'] + sibce_loss
        if sibcon_loss is not None:
            losses_out['sibcon_loss'] = sibcon_loss
            losses_out['final_loss'] = losses_out['final_loss'] + sibcon_loss
        if vattr_loss is not None:
            losses_out['vattr_loss'] = vattr_loss
            losses_out['final_loss'] = losses_out['final_loss'] + vattr_loss
        if poseev_loss is not None:
            losses_out['poseev_loss'] = poseev_loss
            losses_out['final_loss'] = losses_out['final_loss'] + poseev_loss
        return losses_out

    
    @torch.no_grad()
    def inference(
        self,
        video_list,
        points, fpn_masks,
        out_cls_logits, out_offsets,
        out_fg,
        out_centerness,
        ccr_prior=None,
        fineq_ctx=None,
        support_ctx=None,
        compl_ctx=None,
        xm_ctx=None,
        qh_ctx=None,
        ph_ctx=None,
        bnd_ctx=None,
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # ccr_prior: (B, C) TF-CADE video-level class prior or None (inert)
        # fineq_ctx: FINEQ re-scoring context (batch tensors) or None (inert)
        # qh_ctx: QH quality-head context (batch tensors) or None (inert)
        # ph_ctx: PH phase-parse context (batch tensors) or None (inert)
        results = []

        # 1: gather video meta information
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        vid_ft_stride = [x['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
            zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        ):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            fg_per_vid = [x[idx] for x in out_fg]
            cn_per_vid = [x[idx].sigmoid() for x in out_centerness]
            # ---- FINEQ: slice the batch context down to this video
            # (B=1 at eval; mirrors the ccr_prior[idx] slicing above).
            fineq_per_vid = None
            if fineq_ctx is not None:
                fineq_per_vid = dict(fineq_ctx)
                fineq_per_vid.update(
                    feat0=fineq_ctx['feat0'][idx],     # (D, T0)
                    text0=fineq_ctx['text0'][idx],     # (C, D)
                    bias0=fineq_ctx['bias0'][idx],     # (C,)
                )
            # ---- COMPL: slice the batch context down to this video
            # (B=1 at eval; FINEQ slicing precedent).
            compl_per_vid = None
            if compl_ctx is not None:
                compl_per_vid = {
                    'feat0': compl_ctx['feat0'][idx],  # (D, T0)
                    'beta': compl_ctx['beta'],
                }
            # ---- XM-RESCORE: slice the match to this video
            xm_per_vid = None
            if xm_ctx is not None:
                xm_per_vid = {
                    'm': [(None if _m is None else _m[idx])
                          for _m in xm_ctx['m']],          # F x (T_l, C)
                    'gamma': xm_ctx['gamma'],
                }
            # ---- QH: slice the pyramid / mask to this video (B=1 at eval)
            qh_per_vid = None
            if qh_ctx is not None:
                qh_per_vid = {
                    'feats': [_f[idx] for _f in qh_ctx['feats']],  # F x (D, T_l)
                    'mask0': qh_ctx['mask0'][idx],                # (T0,)
                    'phase': qh_ctx['phase'],
                    'fg0': qh_ctx['fg0'][idx],                    # (T0, 1)
                    'beta': qh_ctx['beta'],
                    'comb': qh_ctx.get('comb', 'geo'),
                    'qh': qh_ctx.get('qh', True),
                    'br': qh_ctx.get('br', False),
                    'br_iters': qh_ctx.get('br_iters', 1),
                }
            # ---- PH: slice the phase log-posteriors / mask to this video
            ph_per_vid = None
            if ph_ctx is not None:
                ph_per_vid = dict(ph_ctx)
                ph_per_vid['lp'] = (None if ph_ctx['lp'] is None
                                    else ph_ctx['lp'][idx])  # (T0, K+1) | (C, T0, K+1)
                ph_per_vid['mask0'] = ph_ctx['mask0'][idx]  # (T0,)
                ph_per_vid['feat0'] = (None if ph_ctx['feat0'] is None
                                       else ph_ctx['feat0'][idx])  # (D, T0)
                ph_per_vid['fg'] = ph_ctx['fg'][idx]        # (T0,)
            bnd_per_vid = None
            if bnd_ctx is not None:
                bnd_per_vid = dict(bnd_ctx)
                bnd_per_vid['prob'] = (None if bnd_ctx['prob'] is None
                                       else bnd_ctx['prob'][idx])  # (T0, 2)
                bnd_per_vid['mask0'] = bnd_ctx['mask0'][idx]
                bnd_per_vid['offsets'] = [o[idx] for o in bnd_ctx['offsets']]
            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid, fg_per_vid, cn_per_vid,
                ccr_prior=(ccr_prior[idx] if ccr_prior is not None else None),
                fineq_ctx=fineq_per_vid,
                support_ctx=support_ctx,
                compl_ctx=compl_per_vid,
                xm_ctx=xm_per_vid,
                qh_ctx=qh_per_vid,
                ph_ctx=ph_per_vid,
                bnd_ctx=bnd_per_vid,
            )
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results_per_vid['class_names'] = (
                getattr(self, '_root_names', None)          # ROOT READOUT
                or self.text_features.test_classes)
            # ---- CAND dump (see inference_single_video): write the
            # pre-NMS candidate factors to <dir>/<video_id>.npz and drop
            # the extra key so postprocessing sees the stock dict.
            _cd = results_per_vid.pop('cand_extra', None)
            if _cd is not None:
                import os as _oscd2
                import numpy as _npcd
                _cdir = _oscd2.environ['TIFAD_DUMP_CAND']
                _oscd2.makedirs(_cdir, exist_ok=True)
                _npcd.savez_compressed(
                    _oscd2.path.join(_cdir, vidx + '.npz'),
                    segs=results_per_vid['segments'].cpu().numpy(),
                    score=results_per_vid['scores'].cpu().numpy(),
                    label=results_per_vid['labels'].cpu().numpy(),
                    fps=fps, duration=vlen, feat_stride=stride,
                    nframes=nframes,
                    **{k: v.cpu().numpy() for k, v in _cd.items()})
            results.append(results_per_vid)

        # step 3: postprocssing
        results = self.postprocessing(results)

        return results

    @torch.no_grad()
    def inference_single_video(
        self,
        points,
        fpn_masks,
        out_cls_logits,
        out_offsets,
        out_fg,
        out_cn,
        ccr_prior=None,
        fineq_ctx=None,
        support_ctx=None,
        compl_ctx=None,
        xm_ctx=None,
        qh_ctx=None,
        ph_ctx=None,
        bnd_ctx=None,
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        # xm_ctx: {'m': F x (T_l, C) raw ordered match or None, 'gamma'}
        # ccr_prior: (C,) TF-CADE video-level class prior or None (inert)
        # fineq_ctx: FINEQ per-video re-scoring context or None (inert)
        # qh_ctx: {'feats': F x (D, T_l), 'mask0': (T0,), 'phase': (C, K, D)
        #          or None, 'beta'} QH quality-head context or None (inert)
        # ph_ctx: {'lp': (T0, K+1) | (C, T0, K+1), 'mask0': (T0,), ...} PH
        #          phase-parse context or None (inert)
        segs_all = []
        scores_all = []
        cls_idxs_all = []
        pts_idxs_all = []
        lvls_all = []       # FINEQ only: source FPN level per candidate
        ph_lvl_all = []     # PH only: source FPN level per candidate
        xm_all = []         # XM-RESCORE only: m(n, c) per candidate
        qh_all = []         # QH only: predicted quality q per candidate
        # ---- CAND dump (TIFAD_DUMP_CAND=<dir>, EVAL ONLY, env-gated,
        # byte-inert unset): per-candidate score FACTORS (sigmoid cls /
        # centerness / fg at the anchor point) + anchor pt + FPN level,
        # pre-NMS, for offline decomposition of the decode score
        # sqrt(cls*cn)*fg. Saved per video by inference() below.
        import os as _oscd
        _cand_dump = bool(_oscd.environ.get('TIFAD_DUMP_CAND', ''))
        _cand_extra = {k: [] for k in
                       ('cls_s', 'cn_s', 'fg_s', 'pt', 'lvl')} \
            if _cand_dump else None
        if _cand_extra is not None and qh_ctx is not None \
                and qh_ctx.get('qh', True):
            _cand_extra['q'] = []
        # ---- CAND dump extension (TIFAD_DUMP_CAND_LOGITS=1, with
        # TIFAD_DUMP_CAND; scripts/tree_consistency.py): additionally dump
        # every candidate's location index WITHIN its level ('pti') and the
        # RAW per-location class logits of every level over the full
        # current vocabulary ('logits_l<i>' (T_i, C), 'mask_l<i>' (T_i,)),
        # so the ancestor-level readout at a detection's own location can
        # be computed offline. Inert unless both envs are set.
        _cand_logits = (_cand_extra is not None and
                        _oscd.environ.get('TIFAD_DUMP_CAND_LOGITS', '') == '1')
        _cand_full = {}
        if _cand_logits:
            _cand_extra['pti'] = []

        # loop over fpn levels
        for lvl_i, (cls_i, offsets_i, pts_i, mask_i, fg_i, cn_i) in enumerate(zip(
                out_cls_logits, out_offsets, points, fpn_masks, out_fg, out_cn)):

            # sigmoid normalization for output logits
            fg_i = (fg_i.sigmoid() * mask_i.unsqueeze(-1))
            cn_i = (cn_i.sigmoid() * mask_i.unsqueeze(-1))

            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1))
            # ---- TF-CADE CCR (Eq. 8): P~ = sqrt(sigmoid(P_cls) . softmax(S_fg)),
            # broadcast the (C,) video-level prior over time. DOC-AMBIG: the
            # paper's score chain stops at P~; Ti-FAD further multiplies by
            # centerness/foreground below -- CCR replaces sigmoid(P_cls)
            # inside the stock chain. Inert when ccr_prior is None.
            if ccr_prior is not None:
                pred_prob = torch.sqrt(pred_prob * ccr_prior.unsqueeze(0))
            pred_prob = torch.sqrt(pred_prob * cn_i)
            pred_prob = pred_prob * fg_i
            pred_prob = pred_prob.flatten()
            
            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1]
            topk_cls = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_cls.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_cls = topk_cls[idxs[:num_topk]].clone()


            # fix a warning in pytorch 1.9
            num_cls = cls_i.size(-1)
            pt_idxs =  torch.div(          
                topk_cls, num_cls, rounding_mode='floor'
            )
            cls_idxs = torch.fmod(topk_cls, num_cls)
            
            
            # 3. gather predicted offsets
            offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]  

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1)
            

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            # ---- SUPPORT (TIFAD_SUPPORT, see forward): completeness
            # multiplier applied to the kept candidates' scores at this
            # exact point (post candidate selection, pre soft-NMS).
            kept_scores = pred_prob[keep_idxs2]
            if support_ctx is not None and kept_scores.numel() > 0:
                kept_scores = kept_scores * self._support_mult(
                    support_ctx, pred_segs[keep_idxs2],
                    cls_idxs[keep_idxs2])
            scores_all.append(kept_scores)
            cls_idxs_all.append(cls_idxs[keep_idxs2])
            pts_idxs_all.append(pts[:, 0][keep_idxs2])
            if fineq_ctx is not None:
                lvls_all.append(torch.full_like(cls_idxs[keep_idxs2], lvl_i))
            if ph_ctx is not None or bnd_ctx is not None:
                ph_lvl_all.append(torch.full_like(cls_idxs[keep_idxs2], lvl_i))
            if xm_ctx is not None:
                _mi = xm_ctx['m'][lvl_i]
                if _mi is None:                     # MINLVL pass-through
                    xm_all.append(torch.zeros_like(kept_scores))
                else:
                    xm_all.append(_mi[pt_idxs, cls_idxs][keep_idxs2]
                                  .to(kept_scores.dtype))
            # ---- QH (TIFAD_QH=1): predicted quality q of every kept
            # candidate of this level from its anchor feature, the
            # level-0 pools of its DECODED span (1.5x context) and the
            # phase map of ITS class (the same inputs the head was trained
            # on). Combined with the score after the level concat below.
            if qh_ctx is not None and qh_ctx.get('qh', True):
                qh_all.append(self._qh_score(
                    qh_ctx, lvl_i, pt_idxs[keep_idxs2],
                    cls_idxs[keep_idxs2], pred_segs[keep_idxs2]))
            # ---- BR (TIFAD_BR=1 TIFAD_BR_APPLY=1): refine the kept
            # candidates' spans (q above was read off the ORIGINAL decoded
            # span, as trained); scores unchanged. The raw span is kept
            # for the CAND dump ('segs_raw').
            if qh_ctx is not None and qh_ctx.get('br', False):
                _raw_segs = pred_segs[keep_idxs2]
                _ref_segs = self._br_refine(
                    qh_ctx, lvl_i, pt_idxs[keep_idxs2], _raw_segs)
                segs_all[-1] = _ref_segs
                if _cand_extra is not None:
                    _cand_extra.setdefault('segs_raw', []).append(_raw_segs)
            if _cand_extra is not None:
                _k2 = keep_idxs2
                _cand_extra['cls_s'].append(
                    cls_i.sigmoid()[pt_idxs, cls_idxs][_k2])
                _cand_extra['cn_s'].append(cn_i[pt_idxs, 0][_k2])
                _cand_extra['fg_s'].append(fg_i[pt_idxs, 0][_k2])
                _cand_extra['pt'].append(pts[:, 0][_k2])
                _cand_extra['lvl'].append(
                    torch.full_like(cls_idxs[_k2], lvl_i))
                if qh_ctx is not None and qh_ctx.get('qh', True):
                    _cand_extra['q'].append(qh_all[-1])
                if _cand_logits:
                    _cand_extra['pti'].append(pt_idxs[_k2])
                    _cand_full[f'logits_l{lvl_i}'] = cls_i.detach()
                    _cand_full[f'mask_l{lvl_i}'] = mask_i.detach()

        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all, pts_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all, pts_idxs_all]
        ]

        # ---- FINEQ re-scoring (see forward): flagged-class candidates from
        # FPN level >= 2 get a fine score from level-0 pooled features and
        # a geometric blend (optionally a boundary snap first). Inert when
        # fineq_ctx is None (env unset).
        if fineq_ctx is not None:
            segs_all, scores_all = self._fineq_rescore(
                segs_all, scores_all, cls_idxs_all, torch.cat(lvls_all),
                fineq_ctx, out_fg[0], out_cn[0], fpn_masks[0])

        # ---- Boundary arms B2-B4 (see __init__ / phaseparse.py), BEFORE
        # the parse and soft-NMS: (B2) snap to the boundary-map peaks,
        # (B3) shrink about the center, (B4 'pre') re-anchor the top-k.
        # The original regressed spans go to the CAND dump as 'segs_raw0'.
        if bnd_ctx is not None and segs_all.numel() > 0:
            from .phaseparse import shrink_spans, reanchor_spans
            if _cand_extra is not None:
                _cand_extra['segs_raw0'] = [segs_all.clone()]
            _Tv = int(bnd_ctx['mask0'].sum().item())
            if bnd_ctx['prob'] is not None and bnd_ctx['snap_r'] > 0:
                segs_all = self.bmhead.snap(bnd_ctx['prob'], segs_all, _Tv,
                                            bnd_ctx['snap_r'])
            if bnd_ctx['shrink']:
                segs_all = shrink_spans(segs_all, bnd_ctx['shrink'])
            if bnd_ctx['reanchor'] == 'pre':
                _k = min(int(bnd_ctx['reanchor_k']), scores_all.numel())
                _top = scores_all.topk(_k).indices
                segs_all = segs_all.clone()
                segs_all[_top] = reanchor_spans(
                    segs_all[_top], torch.cat(ph_lvl_all)[_top],
                    bnd_ctx['points'], bnd_ctx['offsets'])
        elif ph_ctx is not None and _cand_extra is not None and segs_all.numel() > 0:
            _cand_extra['segs_raw0'] = [segs_all.clone()]

        # ---- PH (TIFAD_PH_SRC set, see __init__ / phaseparse.py): parse
        # every kept candidate's 1.5x-expanded window with the phase
        # Viterbi; the extent blend / completeness multiplier / phase-
        # profile boundary head act here, after the level concat and
        # BEFORE soft-NMS (scores and spans of the champion's own
        # candidate pool; the QH multiplier below, if on, follows).
        if ph_ctx is not None and segs_all.numel() > 0:
            segs_all, scores_all, _ph_dump = self._ph_apply(
                ph_ctx, segs_all, scores_all, cls_idxs_all,
                torch.cat(ph_lvl_all))
            if _cand_extra is not None:
                for _k, _v in _ph_dump.items():
                    _cand_extra[_k] = [_v]

        # ---- QH combination (TIFAD_QH=1, see __init__ / qhead.py):
        # pre-NMS score = score^(1 - beta) * q^beta (TadTR sqrt(p * q) at
        # beta 0.5). Applied here, after the level concat and BEFORE
        # soft-NMS, so the NMS (which iterates by the score it is handed,
        # libs/utils/nms.py) is ORDERED by the quality-aware score
        # (IoU-Net). The candidate POOL (pre_nms_thresh / top-k /
        # duration filter above) is the champion's own. Exactly the
        # champion's scores at beta == 0 (qh_ctx is then None unless the
        # CAND dump asked for q, in which case the multiplier is skipped).
        # TIFAD_QH_COMB=geo (default, the spec's sqrt form) | prod: score *
        # q^beta (AFSD / BREM product form). Measured with the UNTRAINED
        # 58.92 although it is rank-neutral -- s^(1-beta) compresses the
        # score range and soft-NMS's decayed-vs-undecayed ordering shifts;
        # prod is a uniform rescale under a constant q (only the NMS
        # min_score floor moves), so it isolates the head's contribution.
        if (qh_ctx is not None and qh_ctx.get('qh', True)
                and segs_all.numel() > 0 and qh_ctx['beta'] != 0):
            _qh_q = torch.cat(qh_all).to(scores_all.dtype)       # (N,)
            _qh_b = float(qh_ctx['beta'])
            if qh_ctx.get('comb', 'geo') == 'prod':
                scores_all = scores_all * _qh_q.clamp(min=1e-12) ** _qh_b
            else:
                scores_all = (scores_all.clamp(min=1e-12) ** (1.0 - _qh_b)
                              * _qh_q.clamp(min=1e-12) ** _qh_b)

        # ---- COMPL (TIFAD_COMPL=1, see __init__): trained completeness
        # multiplier on ALL candidates, pre soft-NMS. Applied AFTER the
        # level concat (unlike SUPPORT's per-level hook) so the per-video
        # median normalizer sees the whole candidate pool. Inert when
        # compl_ctx is None (env unset).
        if compl_ctx is not None and segs_all.numel() > 0:
            scores_all = scores_all * self._compl_mult(
                compl_ctx, segs_all, int(fpn_masks[0].sum().item()))

        # ---- VSA-XM-RESCORE (TIFAD_VSA_XM_RESCORE=<gamma>, EVAL ONLY):
        # completeness rescoring by the raw ordered chunk-to-phase match
        # m(n, c) in [-1, 1] (before lambda / tau): score *= exp(gamma *
        # (m - mbar_c)), mbar_c = mean of m over THIS video's pre-NMS
        # candidates of class c (relative term). Applied here, after the
        # level concat and before soft-NMS (COMPL precedent): soft-NMS
        # drops candidate identity, so m cannot be carried into
        # postprocess_results; the multiplier commutes with the DURPRIOR /
        # fusion multipliers applied there. Inert when xm_ctx is None.
        if xm_ctx is not None and segs_all.numel() > 0:
            _xm_m = torch.cat(xm_all)                            # (N,)
            _xm_C = out_cls_logits[0].size(-1)
            _xm_sum = torch.zeros(_xm_C, dtype=_xm_m.dtype,
                                  device=_xm_m.device).index_add_(
                0, cls_idxs_all, _xm_m)
            _xm_cnt = torch.zeros(_xm_C, dtype=_xm_m.dtype,
                                  device=_xm_m.device).index_add_(
                0, cls_idxs_all, torch.ones_like(_xm_m))
            _xm_bar = _xm_sum / _xm_cnt.clamp(min=1.0)
            scores_all = scores_all * torch.exp(
                xm_ctx['gamma'] * (_xm_m - _xm_bar[cls_idxs_all]))
            if not getattr(self, '_xm_rescore_announced', False):
                self._xm_rescore_announced = True
                print(f"[VSA-XM-RESCORE] active (gamma={xm_ctx['gamma']}, "
                      f"N={_xm_m.numel()} cands, m mean={_xm_m.mean():.4f} "
                      f"std={_xm_m.std():.4f})", flush=True)

        results = {'segments' : segs_all,
                   'scores'   : scores_all,
                   'labels'   : cls_idxs_all,
                   }
        if bnd_ctx is not None and bnd_ctx['reanchor'] == '1' \
                and segs_all.numel() > 0:
            results['reanchor'] = {
                'segs': segs_all.detach(), 'lvl': torch.cat(ph_lvl_all),
                'labels': cls_idxs_all, 'k': int(bnd_ctx['reanchor_k']),
                'points': bnd_ctx['points'], 'offsets': bnd_ctx['offsets']}
        if _cand_extra is not None:
            results['cand_extra'] = {
                k: torch.cat(v) for k, v in _cand_extra.items()}
            results['cand_extra'].update(_cand_full)
        return results

    @staticmethod
    @torch.no_grad()
    def _support_mult(ctx, segs, cls_idxs):
        """SUPPORT completeness multiplier C^beta (see forward).
        C = clipped-z evidence mass inside the candidate span / mass in
        the span dilated by 0.5*len on each side, on the level-0 grid.
        segs are in level-0 feature-grid units (as is ctx['t0'])."""
        t0, czp, beta = ctx['t0'], ctx['czp'], ctx['beta']
        T0 = t0.numel()
        s, e = segs[:, 0], segs[:, 1]
        L = (e - s).clamp(min=1e-3)

        def idx(x):
            return torch.searchsorted(t0, x.contiguous()).clamp(0, T0)

        i0 = idx(s)
        i1 = torch.minimum(torch.maximum(idx(e), i0 + 1),
                           torch.full_like(i0, T0))
        i0 = i0.clamp(max=T0 - 1)
        j0 = idx(s - 0.5 * L)
        k1 = idx(e + 0.5 * L)
        c = cls_idxs.long()
        m_in = czp[i1, c] - czp[i0, c]
        m_lo = (czp[i0, c] - czp[j0, c]).clamp(min=0)
        m_hi = (czp[k1, c] - czp[i1, c]).clamp(min=0)
        comp = (m_in + 1e-3) / (m_in + m_lo + m_hi + 1e-3)
        return comp.clamp(min=1e-4) ** beta

    @torch.no_grad()
    def _compl_mult(self, ctx, segs, t_valid):
        """COMPL multiplier (see __init__): (sigmoid(compl)/median)^beta,
        the SAME {pre, thirds, post} pooling as training. segs are in
        level-0 feature-grid units (FINEQ convention), so they index the
        ctx['feat0'] columns directly; the median is over THIS video's
        candidate pool, making the multiplier exactly 1.0 at init
        (zero-init head -> all logits exactly 0)."""
        pooled = _compl_pool(ctx['feat0'][:, :t_valid], segs)
        p = self.compl_head(pooled).squeeze(-1).float().sigmoid()
        med = p.median().clamp(min=1e-4)
        return (p / med).clamp(min=1e-4, max=1e4) ** ctx['beta']

    # ------------------------------------------------------------------
    # STAGE-2 heads on the frozen trunk: QH (TIFAD_QH=1, quality) and BR
    # (TIFAD_BR=1, boundary refinement); see libs/modeling/qhead.py.
    @torch.no_grad()
    def _s2_cand(self, ctx, lvl, pt_idxs, segs):
        """Shared detached inputs of N candidates of ONE video decoded
        from level `lvl` at anchors pt_idxs with spans segs (N, 2) in
        level-0 units: f_n (N, D), pools (N, K+2, D), fgp (N, K+2) or
        None, ld (N, F+1)."""
        from .qhead import pool_span_chunks, lvl_dur_feat
        head = self.qhead if self.qhead is not None else self.brhead
        f_l = ctx['feats'][lvl]                                    # (D, T_l)
        f_n = f_l[:, pt_idxs].t()                                  # (N, D)
        segs = segs.to(f_l.dtype)
        pools = pool_span_chunks(
            ctx['feats'][0].unsqueeze(0), ctx['mask0'].unsqueeze(0),
            segs.unsqueeze(0), K=head.K, ctx=head.ctx)[0]          # (N, K+2, D)
        fgp = None
        if self.qhead is not None and self.qhead.struct:
            fgp = pool_span_chunks(
                ctx['fg0'].t().unsqueeze(0).to(f_l.dtype),
                ctx['mask0'].unsqueeze(0), segs.unsqueeze(0),
                K=head.K, ctx=head.ctx)[0, :, :, 0]                # (N, K+2)
        ld = lvl_dur_feat(torch.full_like(pt_idxs, lvl), segs,
                          len(self.fpn_strides), self.max_seq_len)
        return f_n, pools, fgp, ld

    @torch.no_grad()
    def _qh_score(self, ctx, lvl, pt_idxs, cls_idxs, segs):
        """Predicted quality q (N,) in (0, 1) for N candidates."""
        if segs.numel() == 0:
            return segs.new_zeros((0,))
        f_n, pools, fgp, ld = self._s2_cand(ctx, lvl, pt_idxs, segs)
        phase = None
        if self.qhead.phasemap:
            phase = ctx['phase'][cls_idxs.long()]                  # (N, K, D)
        return self.qhead(f_n, pools, phase, fgp, ld)

    @torch.no_grad()
    def _br_refine(self, ctx, lvl, pt_idxs, segs):
        """R1: corrected spans (N, 2) for N candidates (TIFAD_BR_ITERS
        rounds of re-pooling on the refined span)."""
        if segs.numel() == 0:
            return segs
        out = segs
        for _ in range(max(int(ctx.get('br_iters', 1)), 1)):
            f_n, pools, _, ld = self._s2_cand(ctx, lvl, pt_idxs, out)
            out = self.brhead.apply(out, self.brhead.rel(f_n, pools, ld))
        return out

    def _s2_train(self, losses, feats, fpn_masks, points, out_offsets,
                  out_cls_logits, gt_cls_labels, gt_segments, phase_bank,
                  fg0=None):
        """Stage-2 train losses (QH and/or BR; see qhead.py). feats: the
        PRE-VSA cls pyramid F x (B, D, T_l); fpn_masks F x (B, T_l);
        points F x (T_l, 4); out_offsets F x (B, T_l, 2); out_cls_logits
        F x (B, T_l, C); gt_cls_labels B x (FT, C) one-hot; gt_segments
        B x (G, 2) level-0 units; phase_bank (C, K, D) or None; fg0
        (B, T0, 1) fg logits. Adds the losses and diagnostics to
        `losses`. Every input is detached: the trunk is never touched."""
        from .qhead import pool_span_chunks, qh_corr, lvl_dur_feat
        head = self.qhead if self.qhead is not None else self.brhead
        B = feats[0].size(0)
        D = feats[0].size(1)
        with torch.no_grad():
            spans, f_n, lvl_l = [], [], []
            for li, (f_l, off, pts) in enumerate(
                    zip(feats, out_offsets, points)):
                off = off.detach()                                 # (B, T, 2)
                t0 = pts[:, 0].unsqueeze(0)
                st = pts[:, 3].unsqueeze(0)
                spans.append(torch.stack(
                    [t0 - off[..., 0] * st, t0 + off[..., 1] * st], dim=-1))
                f_n.append(f_l.detach().transpose(1, 2))           # (B, T, D)
                lvl_l.append(torch.full((pts.size(0),), li,
                                        device=f_l.device, dtype=torch.long))
            spans = torch.cat(spans, dim=1)                        # (B, FT, 2)
            f_n = torch.cat(f_n, dim=1)                            # (B, FT, D)
            valid = torch.cat(fpn_masks, dim=1)                    # (B, FT)
            lvl_all = torch.cat(lvl_l).unsqueeze(0).expand(B, -1)  # (B, FT)
            FT = spans.size(1)
            pools = pool_span_chunks(
                feats[0].detach(), fpn_masks[0], spans.to(f_n.dtype),
                K=head.K, ctx=head.ctx)                            # (B, FT, K+2, D)
            ld = lvl_dur_feat(lvl_all.reshape(-1), spans.reshape(-1, 2),
                              len(self.fpn_strides), self.max_seq_len)
        f_flat = f_n.reshape(B * FT, D)
        p_flat = pools.reshape(B * FT, head.K + 2, D)
        # ---- QH
        if self.qhead is not None:
            qh = self.qhead
            with torch.no_grad():
                phase = None
                if qh.phasemap:
                    gt_cls = torch.stack(gt_cls_labels)            # (B, FT, C)
                    pos = gt_cls.sum(-1) > 0
                    c_pred = torch.cat(out_cls_logits, dim=1).detach().argmax(-1)
                    c_star = torch.where(pos, gt_cls.argmax(-1), c_pred)
                    phase = phase_bank[c_star.reshape(-1)]         # (B*FT, K, D)
                fgp = None
                if qh.struct:
                    fgp = pool_span_chunks(
                        fg0.detach().transpose(1, 2).to(f_n.dtype),
                        fpn_masks[0], spans.to(f_n.dtype), K=qh.K,
                        ctx=qh.ctx)[..., 0].reshape(B * FT, qh.K + 2)
            q_logit = qh.logits(f_flat, p_flat, phase, fgp, ld).reshape(B, FT)
            qual = q_logit.new_zeros(())
            rank = q_logit.new_zeros(())
            n_s, n_pairs, n_corr = 0, 0, 0
            q_s, t_s = [], []
            for b in range(B):
                o = qh.train_losses(q_logit[b], spans[b], gt_segments[b],
                                    valid[b])
                qual = qual + o['qual']
                rank = rank + o['rank']
                n_s += o['n']
                n_pairs += o['npairs']
                n_corr += o['pair_correct']
                if o['q_s'] is not None:
                    q_s.append(o['q_s'])
                    t_s.append(o['t_s'])
            if n_s == 0:
                # keep the head's graph alive (unused-param safety)
                qh_loss = 0.0 * sum(p.sum() for p in qh.parameters())
                qh_rank = qh_loss
            else:
                qh_loss = qual / n_s
                qh_rank = qh.rank_w * rank / max(n_pairs, 1)
            losses['qh_loss'] = qh_loss
            losses['qh_rank'] = qh_rank
            losses['qh_n'] = torch.tensor(float(n_s), device=q_logit.device)
            losses['qh_npairs'] = torch.tensor(float(n_pairs),
                                               device=q_logit.device)
            # diagnostics only when defined (the train-loop AverageMeter
            # would be poisoned by a NaN on an iteration without pairs)
            if n_pairs > 0:
                losses['qh_pair_acc'] = torch.tensor(
                    n_corr / n_pairs, device=q_logit.device)
            if q_s:
                pr, sp = qh_corr(torch.cat(q_s), torch.cat(t_s))
                losses['qh_pearson'] = pr
                losses['qh_spearman'] = sp
            losses['final_loss'] = losses['final_loss'] + qh_loss + qh_rank
        # ---- BR (R1): relative corrections for every location; L1 +
        # DIoU on the candidates matched at tIoU >= 0.5, / loss_normalizer
        if self.brhead is not None:
            br = self.brhead
            rel = br.rel(f_flat, p_flat, ld).reshape(B, FT, 2)
            l1 = rel.new_zeros(())
            dio = rel.new_zeros(())
            n_m, t_b, t_a = 0, 0.0, 0.0
            for b in range(B):
                _l1, _di, _n, _tb, _ta = br.train_losses(
                    rel[b], spans[b], gt_segments[b], valid[b],
                    ctr_diou_loss_1d)
                l1 = l1 + _l1
                dio = dio + _di
                n_m += _n
                t_b += _tb
                t_a += _ta
            if n_m == 0:
                br_loss = 0.0 * sum(p.sum() for p in br.parameters())
                losses['br_l1'] = br_loss
                losses['br_diou'] = br_loss
            else:
                br_l1 = br.w * l1 / self.loss_normalizer
                br_diou = br.w * dio / self.loss_normalizer
                br_loss = br_l1 + br_diou
                losses['br_l1'] = br_l1
                losses['br_diou'] = br_diou
                losses['br_n'] = torch.tensor(float(n_m), device=rel.device)
                # mean tIoU of the matched candidates before / after the
                # correction (train batch)
                losses['br_tiou_before'] = torch.tensor(t_b / n_m,
                                                        device=rel.device)
                losses['br_tiou_after'] = torch.tensor(t_a / n_m,
                                                       device=rel.device)
            losses['final_loss'] = losses['final_loss'] + br_loss
        return losses

    # ------------------------------------------------------------------
    # PH (TIFAD_PH_SRC): phase-parsed extents, VSA v5 (phaseparse.py).
    @torch.no_grad()
    def _ph_apply(self, ctx, segs, scores, cls_idxs, lvls):
        """EVAL: N candidates of ONE video (segs (N, 2) level-0 units,
        scores (N,), cls_idxs (N,), lvls (N,) FPN level). (1) phase-
        profile boundary head (TIFAD_PH_BOUND=1), (2) Viterbi parse of
        the expanded window, (3) extent blend (alpha) / completeness
        multiplier (beta). Returns (segs, scores, dump dict)."""
        from .phaseparse import build_windows, phase_parse, gather_rows
        from .qhead import lvl_dur_feat
        lp = ctx['lp']
        Tv = int(ctx['mask0'].sum().item())
        K = int(ctx['K'])
        cls_l = cls_idxs.long() if ctx['text'] else None
        out = {'ph_segs_raw': segs.clone()}
        if Tv < 1:
            return segs, scores, out
        cur = segs.float()
        if ctx['bound']:
            assert lp is not None, "[PH] boundary head needs per-video log-posteriors"
            for _ in range(max(int(self.ph_brhead.iters), 1)):
                prof = self.ph_brhead.profile(lp, cur, Tv, cls_l)
                ld = lvl_dur_feat(lvls, cur, len(self.fpn_strides),
                                  self.max_seq_len)
                cur = self.ph_brhead.apply(cur, self.ph_brhead.rel(prof, ld))
            out['ph_bound'] = cur.clone()
        pos, valid, _ = build_windows(cur, Tv, ctx=ctx['ctx'], max_w=ctx['maxw'])
        if lp is None:
            # 'seg' whitening: per-candidate features minus the mean over
            # the candidate's own span, through the head, chunked over N
            from .qhead import pool_span_chunks
            x0 = ctx['feat0']                                   # (D, T0)
            mean = pool_span_chunks(x0.unsqueeze(0), ctx['mask0'].unsqueeze(0),
                                    cur.unsqueeze(0), K=1, ctx=0.0)[0, :, 1]  # (N, D)
            rows = []
            for i0 in range(0, pos.size(0), 512):
                xr = x0[:, pos[i0:i0 + 512]].permute(1, 2, 0).float()  # (n, W, D)
                xr = xr - mean[i0:i0 + 512].unsqueeze(1)
                lg = self.phasehead.logits(xr)
                fgr = ctx['fg'][pos[i0:i0 + 512]] if ctx['fg'] is not None else None
                rows.append(self.phasehead.lp_from_head_logits(lg, fgr))
            rows = torch.cat(rows)                              # (N, W, K+1)
        else:
            rows = gather_rows(lp, pos, cls_l)                  # (N, W, K+1)
        parsed, ok, compl, llr, _ = phase_parse(rows, pos, valid, cur, K)
        out.update(ph_parsed=parsed, ph_ok=ok, ph_compl=compl, ph_llr=llr)
        alpha = float(ctx['alpha'])
        if alpha != 0:
            use = ok & (llr >= float(ctx['minllr']))
            blend = alpha * parsed + (1.0 - alpha) * cur
            blend[:, 1] = torch.maximum(blend[:, 1], blend[:, 0] + 1e-3)
            cur = torch.where(use.unsqueeze(1), blend, cur)
        beta = float(ctx['beta'])
        if beta != 0:
            scores = scores * compl.clamp(min=1e-4).to(scores.dtype) ** beta
        return cur.to(segs.dtype), scores, out

    def _ph_train(self, losses, x0_in, fpn_masks, points, out_offsets,
                  gt_segments, video_list, bank=None, fg0=None):
        """TRAIN: phase-head CE on level-0 locations (TIFAD_PH_SRC=head)
        and / or the phase-profile boundary head (TIFAD_PH_BOUND=1) on
        the detector's own decoded candidates of every level, and / or
        the phase-chunk contrastive loss (TIFAD_PH_CTR=1). x0_in: the
        level-0 phase feature (B, D, T0) (branch output / cls trunk /
        stem / raw; the head detaches it, the contrastive loss keeps its
        graph for 'joint' and 'branch')."""
        from .qhead import lvl_dur_feat, pool_span_chunks
        ph = self.phasehead
        x0 = x0_in.detach()                                       # (B, D, T0)
        B = x0.size(0)
        t0 = points[0][:, 0]                                      # (T0,)
        m0 = fpn_masks[0]                                         # (B, T0)
        dev = x0.device
        # ---- contrastive phase chunks (TIFAD_PH_CTR=1)
        if self.phasectr is not None:
            from .phaseparse import gt_chunks
            spans, gv = gt_chunks(x0, m0, gt_segments, K=ph.K)   # (B, G, 2)
            chunks = pool_span_chunks(x0_in.float(), m0, spans, K=ph.K,
                                      ctx=0.0)[:, :, 1:ph.K + 1]  # (B, G, K, D)
            chunks = chunks[gv]                                   # (M, K, D)
            sup, div, n_anc = self.phasectr(chunks)
            ctr = self.ph_ctr_w * (sup + self.phasectr.div_w * div)
            if n_anc == 0:
                ctr = ctr + 0.0 * sum(p.sum() for p in self.phasectr.parameters())
                if self.phasebranch is not None:
                    ctr = ctr + 0.0 * sum(p.sum() for p in self.phasebranch.parameters())
            losses['ctr_loss'] = ctr
            losses['ctr_supcon'] = sup.detach()
            losses['ctr_div'] = div.detach()
            losses['ctr_n'] = torch.tensor(float(n_anc), device=dev)
            losses['final_loss'] = losses['final_loss'] + ctr
        if ph.src == 'head':
            lg = ph.logits(x0.transpose(1, 2)) if ph.whiten != 'seg' else None
            ce = x0.new_zeros(())
            n_s, ph_c, ph_n, bg_c, bg_n = 0, 0, 0, 0, 0
            for b in range(B):
                cuts = video_list[b].get('leaf_cuts', None) \
                    if isinstance(video_list[b], dict) else None
                assert cuts is not None or self.ph_target == 'decile' \
                    or ph.whiten == 'seg', \
                    ("[PH-head] item has no 'leaf_cuts' (the dataset loads the "
                     "pseudo phase cuts only under TIFAD_PH_SRC=head / "
                     "TIFAD_LEAFLOSS=1 in training)")
                if ph.whiten == 'seg':
                    o = ph.train_loss_seg(x0[b], t0, m0[b], gt_segments[b],
                                          bg_ratio=self.ph_bg_ratio,
                                          ctx=self.ph_ctx)
                else:
                    o = ph.train_loss(lg[b], t0, m0[b], gt_segments[b], cuts,
                                      bg_ratio=self.ph_bg_ratio,
                                      target_mode=self.ph_target)
                ce = ce + o['ce']
                n_s += o['n']
                ph_c += o['ph_correct']
                ph_n += o['ph_n']
                bg_c += o['bg_correct']
                bg_n += o['bg_n']
            if n_s == 0:
                ph_loss = 0.0 * sum(p.sum() for p in ph.parameters())
            else:
                ph_loss = self.ph_w * ce / n_s
            losses['ph_loss'] = ph_loss
            losses['ph_n'] = torch.tensor(float(n_s), device=dev)
            if ph_n > 0:
                losses['ph_acc'] = torch.tensor(ph_c / ph_n, device=dev)
            if bg_n > 0:
                losses['ph_bgacc'] = torch.tensor(bg_c / bg_n, device=dev)
            losses['final_loss'] = losses['final_loss'] + ph_loss
        if self.ph_brhead is None:
            return losses
        lp_all = ph.logpost(x0, bank=bank, fg0=fg0)              # (B, T0, K+1)
        br = self.ph_brhead
        with torch.no_grad():
            spans, lvl_l = [], []
            for li, (off, pts) in enumerate(zip(out_offsets, points)):
                off = off.detach()
                tt = pts[:, 0].unsqueeze(0)
                st = pts[:, 3].unsqueeze(0)
                spans.append(torch.stack(
                    [tt - off[..., 0] * st, tt + off[..., 1] * st], dim=-1))
                lvl_l.append(torch.full((pts.size(0),), li, device=dev,
                                        dtype=torch.long))
            spans = torch.cat(spans, dim=1)                        # (B, FT, 2)
            valid = torch.cat(fpn_masks, dim=1)                    # (B, FT)
            lvl_all = torch.cat(lvl_l)
        l1 = x0.new_zeros(())
        dio = x0.new_zeros(())
        n_m, t_b, t_a = 0, 0.0, 0.0
        for b in range(B):
            Tv = int(m0[b].sum().item())
            if Tv < 1:
                continue
            with torch.no_grad():
                lp_b = lp_all[b]                # (T0, K+1) | (C, T0, K+1)
                sp = spans[b].float()
                cls_l = None
                if ph.src == 'text':
                    # train mode: class of the GT the candidate best
                    # overlaps (background candidates: class 0)
                    from .qhead import seg_tiou
                    g = gt_segments[b].float()
                    if g.numel() > 0:
                        _ti, _ = seg_tiou(sp, g)
                        cls_l = video_list[b]['labels'].to(dev)[
                            _ti.argmax(dim=1)].long()
                    else:
                        cls_l = torch.zeros(sp.size(0), dtype=torch.long, device=dev)
                prof = br.profile(lp_b, sp, Tv, cls_l)
                ld = lvl_dur_feat(lvl_all, sp, len(self.fpn_strides),
                                  self.max_seq_len)
            rel = br.rel(prof, ld)
            _l1, _di, _n, _tb, _ta = br.train_losses(
                rel, sp, gt_segments[b], valid[b], ctr_diou_loss_1d)
            l1 = l1 + _l1
            dio = dio + _di
            n_m += _n
            t_b += _tb
            t_a += _ta
        if n_m == 0:
            phb = 0.0 * sum(p.sum() for p in br.parameters())
            losses['phb_l1'] = phb
            losses['phb_diou'] = phb
        else:
            phb_l1 = br.w * l1 / self.loss_normalizer
            phb_diou = br.w * dio / self.loss_normalizer
            phb = phb_l1 + phb_diou
            losses['phb_l1'] = phb_l1
            losses['phb_diou'] = phb_diou
            losses['phb_n'] = torch.tensor(float(n_m), device=dev)
            losses['phb_tiou_before'] = torch.tensor(t_b / n_m, device=dev)
            losses['phb_tiou_after'] = torch.tensor(t_a / n_m, device=dev)
        losses['final_loss'] = losses['final_loss'] + phb
        return losses

    @torch.no_grad()
    def _fineq_rescore(self, segs, scores, cls_idxs, lvls, ctx,
                       fg0, cn0, mask0):
        """FINEQ (TIFAD_FINEQ, EVAL ONLY -- see forward): re-score
        flagged-class candidates decoded from FPN level >= 2 against the
        LEVEL-0 features. segs are in level-0 feature-grid units (points /
        offsets are denormalized by the per-level stride during decoding),
        so they index updated_feat[0] columns directly.
          fg0:   (T0, 1) level-0 foreground LOGITS (per-vid)
          cn0:   (T0, 1) level-0 centerness, already sigmoided (per-vid)
          mask0: (T0,)   level-0 validity mask (per-vid)
        Returns (segs, scores) -- segs only change under TIFAD_FINEQ_SNAP=1.
        """
        if segs.numel() == 0:
            return segs, scores
        flags = ctx['flags'].to(segs.device)
        sel = flags[cls_idxs] & (lvls >= 2)
        if not bool(sel.any()):
            return segs, scores
        feat0 = ctx['feat0']                                 # (D, T0)
        Tv = int(mask0.sum().item())                         # valid length
        if Tv < 1:
            return segs, scores
        s = segs[sel, 0].clamp(min=0.0, max=float(Tv))
        e = segs[sel, 1].clamp(min=0.0, max=float(Tv))
        e = torch.maximum(e, s + 1.0)                        # >=1 grid unit
        # ---- optional boundary snap (TIFAD_FINEQ_SNAP=1): within
        # +/- 0.5*len of each boundary, snap to the local argmax of the
        # level-0 foreground*centerness signal.
        if ctx['snap']:
            sig = (fg0.sigmoid().squeeze(-1) * cn0.squeeze(-1)
                   * mask0.to(fg0.dtype))                    # (T0,)
            half = 0.5 * (e - s)
            s_new = s.clone()
            e_new = e.clone()
            for i in range(s.numel()):
                w = float(half[i])
                for b, dst in ((float(s[i]), s_new), (float(e[i]), e_new)):
                    lo = max(0, int(math.floor(b - w)))
                    hi = min(Tv, int(math.ceil(b + w)) + 1)
                    if hi > lo:
                        dst[i] = float(lo + int(sig[lo:hi].argmax()))
            s = s_new
            e = torch.maximum(e_new, s_new + 1.0)
            segs = segs.clone()
            segs[sel, 0] = s
            segs[sel, 1] = e
        # ---- 5 mean-pools via prefix sums: whole span + 4 equal windows
        P = torch.cat(
            [feat0.new_zeros(feat0.size(0), 1),
             feat0[:, :Tv].cumsum(dim=-1)], dim=-1)          # (D, Tv+1)
        tc = ctx['text0'][cls_idxs[sel]]                     # (N, D)
        bc = ctx['bias0'][cls_idxs[sel]]                     # (N,)
        scale = ctx['scale']
        windows = [(s, e)] + [
            (s + k * (e - s) / 4.0, s + (k + 1) * (e - s) / 4.0)
            for k in range(4)
        ]
        fine = None
        for a, b in windows:
            ai = a.floor().long().clamp(min=0, max=Tv - 1)
            bi = torch.maximum(b.ceil().long().clamp(min=1, max=Tv), ai + 1)
            pool = ((P[:, bi] - P[:, ai])
                    / (bi - ai).to(P.dtype).unsqueeze(0)).t()  # (N, D)
            # score EXACTLY like the base matmul: feat . text / scale + bias
            f = ((pool * tc).sum(dim=-1) / scale + bc).sigmoid()
            fine = f if fine is None else torch.maximum(fine, f)
        beta = ctx['beta']
        scores = scores.clone()
        scores[sel] = (scores[sel].clamp(min=1e-12) ** (1.0 - beta)
                       * fine.clamp(min=1e-12) ** beta)
        return segs, scores

    @staticmethod
    def _stitch_segments(segs, scores, labels, gap_ratio, score_mode,
                         maxlen, minscore_rel=0.0, add_mode=False,
                         discount=1.0):
        """TIFAD_STITCH helper (see postprocessing). CPU tensors:
        segs (N, 2) seconds, scores (N,), labels (N,). Per class, greedily
        merges start-sorted consecutive detections that overlap or whose
        gap <= gap_ratio * min(len_a, len_b), to fixpoint. maxlen > 0 skips
        any merge whose union span would exceed maxlen seconds.
        minscore_rel > 0: only dets with score >= minscore_rel * per-class
        max participate; the weak tail passes through untouched. add_mode:
        keep ALL originals and append merged unions (count >= 2) as extra
        detections. Returns (segs, scores, labels) re-sorted by score
        descending."""
        out_s, out_e, out_sc, out_lb = [], [], [], []
        for c in labels.unique().tolist():
            idxs = (labels == c).nonzero(as_tuple=True)[0].tolist()
            if minscore_rel > 0:
                c_smax = max(float(scores[i]) for i in idxs)
                floor = minscore_rel * c_smax
                weak = [i for i in idxs if float(scores[i]) < floor]
                idxs = [i for i in idxs if float(scores[i]) >= floor]
                # weak tail: pass through untouched (never in add_mode --
                # there ALL originals are emitted below anyway)
                if not add_mode:
                    for i in weak:
                        out_s.append(float(segs[i, 0]))
                        out_e.append(float(segs[i, 1]))
                        out_sc.append(float(scores[i]))
                        out_lb.append(c)
            if add_mode:
                # keep every original detection
                for i in ((labels == c).nonzero(as_tuple=True)[0].tolist()):
                    out_s.append(float(segs[i, 0]))
                    out_e.append(float(segs[i, 1]))
                    out_sc.append(float(scores[i]))
                    out_lb.append(c)
            if len(idxs) == 0:
                continue
            # item = [start, end, score_max, score_sum, count]
            items = sorted(
                ([float(segs[i, 0]), float(segs[i, 1]),
                  float(scores[i]), float(scores[i]), 1] for i in idxs),
                key=lambda x: (x[0], x[1]))
            changed = True
            while changed:
                changed = False
                merged = [items[0]]
                for it in items[1:]:
                    cur = merged[-1]
                    gap = it[0] - cur[1]
                    ok = (gap <= 0.0) or (
                        gap <= gap_ratio * min(cur[1] - cur[0],
                                               it[1] - it[0]))
                    if ok and maxlen > 0:
                        ok = (max(cur[1], it[1]) - cur[0]) <= maxlen
                    if ok:
                        cur[1] = max(cur[1], it[1])
                        cur[2] = max(cur[2], it[2])
                        cur[3] += it[3]
                        cur[4] += it[4]
                        changed = True
                    else:
                        merged.append(it)
                items = merged
            for s0, e0, smax, ssum, cnt in items:
                if add_mode and cnt < 2:
                    continue     # singleton: original already emitted
                sc0 = ssum / cnt if score_mode == 'mean' else smax
                if cnt >= 2:
                    sc0 = sc0 * discount
                out_s.append(s0)
                out_e.append(e0)
                out_sc.append(sc0)
                out_lb.append(c)
        if len(out_s) == 0:
            return segs, scores, labels
        segs = torch.tensor(
            [[a, b] for a, b in zip(out_s, out_e)], dtype=segs.dtype)
        scores = torch.tensor(out_sc, dtype=scores.dtype)
        labels = torch.tensor(out_lb, dtype=labels.dtype)
        order = scores.argsort(descending=True)
        return segs[order], scores[order], labels[order]

    @staticmethod
    def _rerank_detections(segs, scores, labels, lam=0.0, theta=0.8,
                           norm='none', mu=0.0, nu=0.5):
        """TIFAD_RERANK helper (see postprocessing). Rescores detections
        WITHOUT creating or deleting any segment. CPU tensors: segs (N, 2)
        seconds, scores (N,), labels (N,). Per video per class:
          CONTAINMENT VOTES (lam > 0): each det d collects the scores of
          same-class dets f whose span lies >= theta inside d;
          score'(d) = score(d) + lam * V(d), with V optionally normalized
          by vote count ('count') or by d's length in seconds ('len').
          FRAGMENT DEMOTION (mu > 0): each det f with a same-class LONGER
          det d s.t. score(d) >= nu * score(f) is demoted by
          score'(f) *= (1 - mu * maxcontain(f)), maxcontain = max overlap
          fraction of f inside such d.
        Both terms are computed from the ORIGINAL scores, then applied.
        Returns (segs, scores, labels) re-sorted by new score desc."""
        new_scores = scores.clone()
        for c in labels.unique().tolist():
            m = (labels == c).nonzero(as_tuple=True)[0]
            if m.numel() < 2:
                continue
            s = segs[m, 0].double()
            e = segs[m, 1].double()
            p = scores[m].double()
            ln = (e - s).clamp(min=1e-6)
            # pairwise intersection: inter[i, j] between det i and det j
            inter = (torch.minimum(e[:, None], e[None, :])
                     - torch.maximum(s[:, None], s[None, :])).clamp(min=0)
            # contain[i, j]: fraction of i's span inside j
            contain = inter / ln[:, None]
            eye = torch.eye(m.numel(), dtype=torch.bool)
            q = p.clone()
            if lam > 0:
                votes = (contain >= theta) & ~eye        # i votes for j
                v = (votes.double() * p[:, None]).sum(dim=0)  # (N_c,)
                if norm == 'count':
                    v = v / votes.double().sum(dim=0).clamp(min=1)
                elif norm == 'len':
                    v = v / ln
                q = q + lam * v
            if mu > 0:
                longer = (ln[None, :] > ln[:, None]) & ~eye
                strong = p[None, :] >= (nu * p[:, None])
                mc = (contain * (longer & strong).double()).max(dim=1).values
                q = q * (1.0 - mu * mc)
            new_scores[m] = q.to(new_scores.dtype)
        order = new_scores.argsort(descending=True)
        return segs[order], new_scores[order], labels[order]

    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps 
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            class_names = results_per_vid['class_names']

            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu() 
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                # segs, scores, labels, temp_idxs = batched_nms(
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms = (self.test_nms_method == 'soft'),
                    multiclass = self.test_multiclass_nms,
                    sigma = self.test_nms_sigma,
                    voting_thresh = self.test_voting_thresh
                )

            # ---- B4 re-anchoring (TIFAD_REANCHOR=1): the top-k detections
            # after NMS (score-voted spans) find their nearest pre-NMS
            # candidate of the same class -> its pyramid level; the location
            # nearest the detection's center on that level re-reads the
            # regression head and the re-read span is averaged in.
            _ra = results_per_vid.pop('reanchor', None)
            if _ra is not None and segs.shape[0] > 0:
                from .phaseparse import reanchor_spans
                _k = min(_ra['k'], segs.shape[0])
                _top = scores.topk(_k).indices
                _pre = _ra['segs'].cpu().float()
                _plab = _ra['labels'].cpu()
                _lv = torch.zeros(_k, dtype=torch.long)
                for _i, _j in enumerate(_top.tolist()):
                    _same = (_plab == labels[_j]).nonzero(as_tuple=True)[0]
                    if _same.numel() == 0:
                        _same = torch.arange(_pre.size(0))
                    _d = (_pre[_same] - segs[_j].float()).abs().sum(dim=1)
                    _lv[_i] = _ra['lvl'][_same[_d.argmin()]].item()
                _dev = _ra['points'][0].device
                _new = reanchor_spans(
                    segs[_top].to(_dev), _lv.to(_dev), _ra['points'],
                    _ra['offsets']).cpu().to(segs.dtype)
                segs = segs.clone()
                segs[_top] = _new
            # 3: convert from feature grids to seconds  (real time)
            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                # truncate all boundaries within [0, duration]
                segs[segs<=0.0] *= 0.0
                segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen

            # ---- TIFAD_STITCH (EVAL ONLY, env-gated, byte-inert unset):
            # same-class fragment stitching, AFTER NMS + grid->seconds.
            # Placed post-NMS on purpose: NMS already removed near-duplicate
            # candidates, so the survivors are the actual fragmented outputs
            # (the DETAD double-detection / 0.1<=IoU<0.5 near-miss sink);
            # stitching pre-NMS would instead fuse thousands of raw
            # near-duplicate candidates and distort spans. Seconds units so
            # TIFAD_STITCH_MAXLEN is in seconds (gap_ratio is scale-free).
            # Per class: sort by start; greedily merge consecutive A,B when
            # they overlap or gap(B.start-A.end) <= gap_ratio*min(len A,B);
            # merged = [min start, max end], score = max (mean via
            # TIFAD_STITCH_SCORE=mean); iterate to fixpoint.
            # TIFAD_STITCH_MINSCORE=<rel float>: only detections with score
            # >= rel * (per-video-per-class max score) participate in
            # stitching; the rest pass through untouched. NEEDED under this
            # repo's soft-NMS (up to 2000 decayed-score survivors/video):
            # without it, chain-merging the dense weak tail produces
            # gap_ratio=0). TIFAD_STITCH_MODE=add keeps the original
            # fragments and APPENDS the merged unions instead of replacing.
            import os as _osst
            _st_env = _osst.environ.get('TIFAD_STITCH', '')
            if _st_env != '' and segs.shape[0] > 0:
                _st_gap = float(_st_env)
                _st_mode = _osst.environ.get('TIFAD_STITCH_SCORE', 'max')
                _st_maxlen = float(
                    _osst.environ.get('TIFAD_STITCH_MAXLEN', '0') or 0)
                _st_rel = float(
                    _osst.environ.get('TIFAD_STITCH_MINSCORE', '0') or 0)
                _st_add = (_osst.environ.get(
                    'TIFAD_STITCH_MODE', 'replace') == 'add')
                # TIFAD_STITCH_DISCOUNT: multiply merged-union scores by
                # this (default 1.0) so unions rank below the fragments
                # they came from -- a wrong union then costs little, a
                # right one still adds recall at strict tIoU.
                _st_disc = float(
                    _osst.environ.get('TIFAD_STITCH_DISCOUNT', '1') or 1)
                if not getattr(self, '_stitch_announced', False):
                    print(f"[STITCH] active (gap_ratio={_st_gap}, "
                          f"score={_st_mode}, minscore={_st_rel}, "
                          f"mode={'add' if _st_add else 'replace'}, "
                          f"discount={_st_disc})",
                          flush=True)
                    self._stitch_announced = True
                segs, scores, labels = self._stitch_segments(
                    segs, scores, labels, _st_gap, _st_mode, _st_maxlen,
                    _st_rel, _st_add, _st_disc)

            # ---- TIFAD_RERANK=1 (EVAL ONLY, env-gated, byte-inert
            # unset): containment-vote / fragment-demotion rescoring of
            # the post-NMS detections -- no segment is created or
            # removed, only scores (hence ranks) change. Targets the
            # DETAD boundary sink: covering long dets exist for ~96% of
            # GTs but are outranked by their near-miss fragments.
            import os as _osrr
            if _osrr.environ.get('TIFAD_RERANK', '') == '1' \
                    and segs.shape[0] > 0:
                _rr_lam = float(
                    _osrr.environ.get('TIFAD_RERANK_LAMBDA', '0') or 0)
                _rr_theta = float(
                    _osrr.environ.get('TIFAD_RERANK_THETA', '0.8') or 0.8)
                _rr_norm = _osrr.environ.get('TIFAD_RERANK_NORM', 'none')
                _rr_mu = float(
                    _osrr.environ.get('TIFAD_RERANK_MU', '0') or 0)
                _rr_nu = float(
                    _osrr.environ.get('TIFAD_RERANK_NU', '0.5') or 0.5)
                if not getattr(self, '_rerank_announced', False):
                    print(f"[RERANK] active (lambda={_rr_lam}, "
                          f"theta={_rr_theta}, norm={_rr_norm}, "
                          f"mu={_rr_mu}, nu={_rr_nu})", flush=True)
                    self._rerank_announced = True
                segs, scores, labels = self._rerank_detections(
                    segs, scores, labels, _rr_lam, _rr_theta, _rr_norm,
                    _rr_mu, _rr_nu)

            # 4: repack the results
            processed_results.append(
                {'video_id' : vidx,
                 'segments' : segs,
                 'scores'   : scores,
                 'labels'   : labels,
                 'class_names': class_names,
                 }
            )
        return processed_results
