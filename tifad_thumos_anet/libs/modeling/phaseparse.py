"""PH (TIFAD_PH_SRC=probe|text|head): PHASE-PARSED EXTENTS, VSA v5.

Spec: /data3/xiaodan8/actionformer6/autoresearch/vsa_v5_spec.md. Every
earlier video-side arm (v1-v4, R1) used the ordered-phase signal to
RESCORE a location; none used it to DECIDE THE EXTENT of a candidate.
v5 parses each pre-NMS candidate's 1.5x-expanded window with a
transcript-constrained Viterbi (Kuehne/Richard HMM, NN-Viterbi) over the
state chain BG -> phase 1 -> ... -> phase K -> BG on the FROZEN champion.

P1  PhaseEvidence (attribute `phasehead`): per-location log-posteriors
    lp[t, k] over K = 10 phases + 1 background at level 0 of the pyramid
    (the pre-VSA cls trunk, `_vsa_in[0]`), three sources:
      probe : the scripts/phase_probe.py head Linear(D,D)-ReLU-Linear(D,K)
              trained on the champion's own train-video features (this
              module's CLI, `python -m libs.modeling.phaseparse
              --train-probe`), loaded as NON-PERSISTENT buffers (the
              champion checkpoint still loads strictly) + a fixed
              background logit TIFAD_PH_BGBIAS (default 0; the probe never
              saw background). Class-agnostic.
      text  : centered text cosine. The K calibrated phase embeddings of
              the candidate's class (text.py `_vsa_phase_embs` through
              text_proj(normalize(x)/2), the base-matmul space) MINUS the
              mean over all phases of all classes in the vocabulary (kills
              the collinear component that put the raw phase cosine at
              chance), L2-normalized; logit = tau * cos(f_t, z~_k(c)),
              tau = TIFAD_PH_TAU (10); background = TIFAD_PH_BGBIAS. No
              parameters; works for unseen classes from their sentences.
      head  : a NEW in-model head Linear(D,D)-ReLU-Linear(D,K+1) trained
              SEQUENTIALLY on the frozen champion (TIFAD_INIT_FROM +
              TIFAD_FREEZE_EXCEPT=phasehead), inputs detached, on level-0
              locations: target = text-aligned pseudo phase of the seen GT
              segment the location lies in (dataset `leaf_cuts`, loaded
              under TIFAD_PH_SRC=head), background outside every GT
              (random 1:1 sample, TIFAD_PH_BG_RATIO); cross-entropy
              (TIFAD_PH_W, default 1). Diagnostics ph_acc / ph_bgacc.
P2  phase_parse: batched Viterbi per video over the window
    [s - 0.5L, e + 0.5L] of every pre-NMS candidate (TIFAD_PH_CTX 0.5;
    <= TIFAD_PH_MAXW = 128 locations, longer windows subsampled uniformly)
    with states {BG_pre, 1..K, BG_post} (self-loops, no skips, start in
    BG_pre or 1, end in K or BG_post). Outputs the parsed extent [first
    location in phase 1 .. last location in phase K] (boundary = midpoint
    to the neighbouring background location, +-0.5 at the window edge),
    completeness = distinct phases visited INSIDE the original span / K,
    and the log-likelihood ratio of the parse vs the all-background path.
    Windows with < K locations cannot host the chain: no parse (span
    kept, completeness 1, llr 0).
P3  consumers (inference_single_video, after the level concat, BEFORE the
    QH combination and soft-NMS, so NMS sees the corrected spans):
      TIFAD_PH_EXTENT=<alpha> : span <- alpha * parsed + (1 - alpha) *
                                regressed per side where the parse exists
                                and llr >= TIFAD_PH_MINLLR (default 0).
      TIFAD_PH_COMPL=<beta>   : score <- score * completeness^beta (product
                                form; rank-neutral under a constant).
      TIFAD_PH_BOUND=1        : PhaseBoundaryHead (attribute `ph_brhead`,
                                selected by FREEZE_EXCEPT=phasehead,brhead):
                                the (K+1) x 2W log-posterior profile of the
                                expanded window (W = TIFAD_PH_BOUND_W = 32
                                per side) + level one-hot + log-duration ->
                                (ds, de) = tanh * TIFAD_BR_RANGE (0.5) * L
                                per side, zero-init (exactly 0 at init); L1
                                + DIoU on candidates with tIoU >= 0.5 to a
                                seen GT (qhead.BoundaryHead losses), applied
                                at eval before the parse / soft-NMS
                                (TIFAD_PH_BOUND_ITERS re-profiling rounds).
Byte-inert when TIFAD_PH_SRC is unset (nothing constructed, no code path
changes). The CAND dump (TIFAD_DUMP_CAND) carries ph_segs_raw / ph_parsed
/ ph_ok / ph_compl / ph_llr (+ ph_bound) for scripts/qh_diag.py.
"""
import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from .qhead import BoundaryHead

NEG = -1e9      # finite "minus infinity" for the Viterbi (no inf - inf)


class PhaseEvidence(nn.Module):
    """Per-location phase log-posteriors (K phases + background)."""

    def __init__(self, dim, src, K=10, bg_bias=0.0, tau=10.0,
                 probe_path='', hidden=None, bg_mode=None, lam=1.0,
                 whiten=None, n_pc=5):
        super().__init__()
        self.lam = float(lam)
        # de-anisotropy of the head input (TIFAD_PH_WHITEN): 'global' =
        # (x - mean) / std over the training locations; 'pca' = remove the
        # top-n_pc principal components (all-but-the-top, Mu & Viswanath
        # 2018) then / std; 'seg' = subtract, per candidate, the mean
        # feature over the candidate's own span (per-window path,
        # meta_archs._ph_apply / train_loss_seg). Stats live in PERSISTENT
        # buffers (the trained checkpoint carries them; set_whiten_stats).
        self.whiten = whiten or None
        assert self.whiten in (None, 'global', 'pca', 'seg'), \
            f"[PH] TIFAD_PH_WHITEN must be seg|global|pca, got '{whiten}'"
        self.n_pc = int(n_pc)
        assert src in ('probe', 'text', 'head'), \
            f"[PH] TIFAD_PH_SRC must be probe|text|head, got '{src}'"
        self.src = src
        self.bg_mode = bg_mode or ('const' if src == 'head' else 'fg')
        assert self.bg_mode in ('const', 'fg'), \
            f"[PH] TIFAD_PH_BG must be const|fg, got '{self.bg_mode}'"
        self.K = int(K)
        self.dim = int(dim)
        self.bg_bias = float(bg_bias)
        self.tau = float(tau)
        self.hidden = int(hidden or dim)
        self.probe_path = probe_path
        self.net = None
        if src == 'probe':
            assert os.path.isfile(probe_path), \
                (f"[PH] probe head weights missing: {probe_path} (run "
                 f"`python -m libs.modeling.phaseparse --train-probe`)")
            sd = torch.load(probe_path, map_location='cpu')
            w1, b1 = sd['net.0.weight'].float(), sd['net.0.bias'].float()
            w2, b2 = sd['net.2.weight'].float(), sd['net.2.bias'].float()
            assert w1.shape == (self.hidden, self.dim) and \
                w2.shape == (self.K, self.hidden), \
                f"[PH] probe shapes {tuple(w1.shape)} / {tuple(w2.shape)}"
            # non-persistent: the champion checkpoint keeps loading strict
            self.register_buffer('w1', w1, persistent=False)
            self.register_buffer('b1', b1, persistent=False)
            self.register_buffer('w2', w2, persistent=False)
            self.register_buffer('b2', b2, persistent=False)
        elif src == 'head':
            self.net = nn.Sequential(
                nn.Linear(self.dim, self.hidden), nn.ReLU(inplace=True),
                nn.Linear(self.hidden, self.K + 1))
        if self.whiten in ('global', 'pca'):
            self.register_buffer('w_mean', torch.zeros(self.dim))
            self.register_buffer('w_std', torch.ones(self.dim))
            self.register_buffer('w_P', torch.zeros(self.n_pc, self.dim))
            self.register_buffer('w_ready', torch.zeros(()))

    @torch.no_grad()
    def set_whiten_stats(self, X):
        """X (N, D) training-location features (float). global: mean /
        std. pca: mean, top-n_pc PCs of the centered features, std of the
        residual after their removal."""
        X = X.float()
        mean = X.mean(dim=0)
        Xc = X - mean
        if self.whiten == 'pca':
            # top principal directions via SVD of the (subsampled) data
            if Xc.size(0) > 20000:
                idx = torch.randperm(Xc.size(0), generator=torch.Generator()
                                     .manual_seed(0))[:20000]
                Xs = Xc[idx]
            else:
                Xs = Xc
            _, _, V = torch.linalg.svd(Xs, full_matrices=False)
            P = V[:self.n_pc]                                  # (n_pc, D)
            Xc = Xc - (Xc @ P.t()) @ P
            self.w_P.copy_(P.to(self.w_P.device))
        self.w_mean.copy_(mean.to(self.w_mean.device))
        self.w_std.copy_(Xc.std(dim=0).clamp(min=1e-4).to(self.w_std.device))
        self.w_ready.fill_(1.0)

    def whiten_x(self, x):
        """x (..., D) -> whitened (global / pca); identity otherwise."""
        if self.whiten not in ('global', 'pca'):
            return x
        x = x - self.w_mean
        if self.whiten == 'pca':
            x = x - (x @ self.w_P.t()) @ self.w_P
        return x / self.w_std

    # ------------------------------------------------------------------
    def logits(self, x):
        """x (..., D) -> (..., K+1) logits incl. background (probe/head).
        The input is DETACHED: the trunk never receives gradient."""
        x = self.whiten_x(x.detach().float())
        if self.src == 'probe':
            h = F.relu(F.linear(x, self.w1, self.b1))
            lg = F.linear(h, self.w2, self.b2)
            bg = lg.new_full(lg.shape[:-1] + (1,), self.bg_bias)
            return torch.cat([lg, bg], dim=-1)
        if self.src == 'head':
            return self.net(x)
        raise RuntimeError("[PH] text source has no logits(); use logpost")

    @torch.no_grad()
    def logpost(self, x0, bank=None, fg0=None):
        """x0 (B, D, T0) level-0 trunk features. probe / head ->
        (B, T0, K+1); text -> (B, C, T0, K+1) for the C classes of the
        current vocabulary (bank (C, K, D) calibrated phase embeddings).
        bg_mode 'const': background logit = the fixed bias (head: its own
        learned logit). bg_mode 'fg' (default for probe / text, whose
        phase logits never saw background and are a softmax WITHIN the
        action): factorize with the detector's class-agnostic foreground
        head, fg0 (B, T0) ACTIONNESS logits (actionness_from_fg: logsumexp
        over the levels' fg heads at the nearest location) -- p(k|t) = sigmoid(fg_t) *
        softmax_K(phase)_k, p(bg|t) = (1 - sigmoid(fg_t)) * exp(bias),
        renormalized; head: product of experts (its own K+1 posterior
        times the fg split), renormalized."""
        x = x0.detach().float().transpose(1, 2)              # (B, T0, D)
        K = self.K
        if self.src == 'text':
            assert bank is not None, "[PH-text] phase bank missing"
            z = bank.detach().float()                          # (C, K, D)
            z = z - z.reshape(-1, z.size(-1)).mean(dim=0)      # de-collinearize
            z = F.normalize(z, dim=-1)
            cos = torch.einsum('btd,ckd->bctk', F.normalize(x, dim=-1), z)
            lg = cos * self.tau                                # (B, C, T0, K)
            lg = torch.cat([lg, lg.new_full(lg.shape[:-1] + (1,),
                                            self.bg_bias)], dim=-1)
        else:
            lg = self.logits(x)                                # (B, T0, K+1)
        logK = math.log(K)
        if self.src == 'head':
            return self.lp_from_head_logits(lg, fg0)
        if self.bg_mode == 'const':
            # the spec's fixed-bias background against the raw phase logits
            return F.log_softmax(lg, dim=-1)
        assert fg0 is not None, "[PH] bg_mode=fg needs the level-0 fg logits"
        fg = fg0.detach().float()                              # (B, T0)
        la = F.logsigmoid(fg)                                  # log p(action)
        lb = F.logsigmoid(-fg)                                 # log p(bg)
        if lg.dim() == 4:                                      # text: (B, C, T0, .)
            la = la.unsqueeze(1)
            lb = lb.unsqueeze(1)
        # scaled likelihood: p(k | action, t) / (1 / K), weight lam
        lph = la.unsqueeze(-1) + self.lam * (
            F.log_softmax(lg[..., :K], dim=-1) + logK)
        lbg = (lb + self.bg_bias).unsqueeze(-1).expand(*lph.shape[:-1], 1)
        return F.log_softmax(torch.cat([lph, lbg], dim=-1), dim=-1)

    def lp_from_head_logits(self, lg, fg0=None):
        """head source: calibrated (K+1)-way posteriors -> hybrid-HMM
        scaled likelihoods: the phase posteriors are divided by their
        prior 1/K relative to background (+ bias); 'fg' mode multiplies in
        the detector's actionness split as a second expert. lg (..., K+1),
        fg0 (...) broadcastable to lg[..., 0]."""
        K = self.K
        lp = F.log_softmax(lg.float(), dim=-1)
        lph = lp[..., :K] + math.log(K)
        lbg = lp[..., K:] + self.bg_bias
        if self.bg_mode == 'fg':
            assert fg0 is not None, "[PH] bg_mode=fg needs the fg logits"
            fg = fg0.detach().float()
            lph = lph + F.logsigmoid(fg).unsqueeze(-1)
            lbg = lbg + F.logsigmoid(-fg).unsqueeze(-1)
        return F.log_softmax(torch.cat([lph, lbg], dim=-1), dim=-1)

    # ------------------------------------------------------------------
    def train_loss(self, lg, t0, valid, gts, cuts, bg_ratio=1.0,
                   generator=None, target_mode='pseudo'):
        """Head-source CE for ONE item. lg (T0, K+1) logits; t0 (T0,)
        level-0 times (grid units); valid (T0,) bool; gts (G, 2) seen GT
        spans (grid units); cuts (G, 9) pseudo phase cut points (NaN row
        = no pseudo entry -> the segment's locations are ignored).
        target_mode 'pseudo' = phase index from the text-aligned cuts (v5h);
        'decile' (v5p, TIFAD_PH_TARGET=decile) = position decile inside
        the GT, floor(K (t - s) / (e - s)) clipped to 0..K-1 -- nothing
        text-derived (cuts unused). Returns dict: ce (sum), n,
        ph_correct, ph_n, bg_correct, bg_n."""
        K = self.K
        T0 = lg.size(0)
        out = {'ce': lg.new_zeros(()), 'n': 0, 'ph_correct': 0, 'ph_n': 0,
               'bg_correct': 0, 'bg_n': 0}
        with torch.no_grad():
            target = torch.full((T0,), -1, dtype=torch.long, device=lg.device)
            has_seg = torch.zeros(T0, dtype=torch.bool, device=lg.device)
            use_cuts = (target_mode == 'pseudo' and cuts is not None
                        and cuts.numel() > 0)
            if gts is not None and gts.numel() > 0 and (
                    use_cuts or target_mode == 'decile'):
                gts = gts.float()
                inside = (t0[:, None] >= gts[None, :, 0]) & \
                    (t0[:, None] <= gts[None, :, 1])              # (T0, G)
                lens = torch.where(inside, (gts[:, 1] - gts[:, 0])[None, :],
                                   torch.full_like(inside, float('inf'),
                                                   dtype=torch.float32))
                j = lens.argmin(dim=1)
                has_seg = torch.isfinite(lens.gather(1, j[:, None]).squeeze(1))
                if use_cuts:
                    cuts = cuts.to(device=lg.device, dtype=torch.float32)
                    cj = cuts[j]                                   # (T0, 9)
                    has_cuts = ~torch.isnan(cj).any(dim=-1)
                    k = (t0[:, None] >= cj).sum(dim=-1).clamp(0, K - 1)
                else:
                    gj = gts[j]                                    # (T0, 2)
                    has_cuts = torch.ones_like(has_seg)
                    k = torch.floor(K * (t0 - gj[:, 0])
                                    / (gj[:, 1] - gj[:, 0]).clamp(min=1e-3)
                                    ).long().clamp(0, K - 1)
                sel_ph = has_seg & has_cuts & valid
                target[sel_ph] = k[sel_ph]
            elif gts is not None and gts.numel() > 0:
                gts = gts.float()
                has_seg = ((t0[:, None] >= gts[None, :, 0]) &
                           (t0[:, None] <= gts[None, :, 1])).any(dim=1)
            ph_mask = valid & (target >= 0)
            bg_mask = valid & ~has_seg
            n_ph = int(ph_mask.sum())
            n_bg = int(bg_mask.sum())
            n_keep = int(min(n_bg, max(int(round(bg_ratio * n_ph)), 16)))
            bg_sel = torch.zeros_like(bg_mask)
            if n_keep > 0:
                bi = bg_mask.nonzero(as_tuple=True)[0]
                pick = torch.randperm(bi.numel(), device=lg.device,
                                      generator=generator)[:n_keep]
                bg_sel[bi[pick]] = True
            target[bg_sel] = K
            sel = ph_mask | bg_sel
        if not bool(sel.any()):
            return out
        ce = F.cross_entropy(lg[sel].float(), target[sel], reduction='sum')
        with torch.no_grad():
            pred = lg.argmax(dim=-1)
            out['ph_correct'] = int((pred[ph_mask] == target[ph_mask]).sum())
            out['ph_n'] = n_ph
            out['bg_correct'] = int((pred[bg_sel] == K).sum())
            out['bg_n'] = int(bg_sel.sum())
        out['ce'] = ce
        out['n'] = int(sel.sum())
        return out


    def train_loss_seg(self, x0, t0, valid, gts, bg_ratio=1.0, ctx=0.5,
                       generator=None):
        """'seg' whitening, head source, ONE item: for every GT window
        [s - ctx L, e + ctx L] the location features MINUS the mean
        feature over [s, e] go through the head; targets = position
        decile inside the GT, background for window locations outside
        every GT (sampled bg_ratio : 1), locations inside another GT
        ignored. x0 (D, T0) features; t0 (T0,); valid (T0,) bool; gts
        (G, 2). Returns the train_loss dict."""
        K = self.K
        out = {'ce': x0.new_zeros(()), 'n': 0, 'ph_correct': 0, 'ph_n': 0,
               'bg_correct': 0, 'bg_n': 0}
        if gts is None or gts.numel() == 0:
            return out
        gts = gts.float()
        with torch.no_grad():
            inside_any = ((t0[:, None] >= gts[None, :, 0]) &
                          (t0[:, None] <= gts[None, :, 1])).any(dim=1)
        xs, ys, is_ph = [], [], []
        for g in range(gts.size(0)):
            s, e = gts[g, 0], gts[g, 1]
            L = (e - s).clamp(min=1e-3)
            with torch.no_grad():
                win = valid & (t0 >= s - ctx * L) & (t0 <= e + ctx * L)
                inn = win & (t0 >= s) & (t0 <= e)
                if not bool(inn.any()):
                    continue
                mean = x0[:, inn].float().mean(dim=1)             # (D,)
                bgm = win & ~inside_any
                n_ph = int(inn.sum())
                n_keep = int(min(int(bgm.sum()), max(int(round(bg_ratio * n_ph)), 4)))
                bg_sel = torch.zeros_like(bgm)
                if n_keep > 0:
                    bi = bgm.nonzero(as_tuple=True)[0]
                    pick = torch.randperm(bi.numel(), device=x0.device,
                                          generator=generator)[:n_keep]
                    bg_sel[bi[pick]] = True
                sel = inn | bg_sel
                k = torch.floor(K * (t0 - s) / L).long().clamp(0, K - 1)
                y = torch.where(inn, k, torch.full_like(k, K))[sel]
            xs.append(x0[:, sel].t().float() - mean)
            ys.append(y)
            is_ph.append(inn[sel])
        if not xs:
            return out
        X = torch.cat(xs)
        Y = torch.cat(ys)
        P = torch.cat(is_ph)
        lg = self.logits(X)
        out['ce'] = F.cross_entropy(lg.float(), Y, reduction='sum')
        out['n'] = int(Y.numel())
        with torch.no_grad():
            pred = lg.argmax(dim=-1)
            out['ph_correct'] = int((pred[P] == Y[P]).sum())
            out['ph_n'] = int(P.sum())
            out['bg_correct'] = int((pred[~P] == K).sum())
            out['bg_n'] = int((~P).sum())
        return out


# ----------------------------------------------------------------------
class PhaseBranch(nn.Module):
    """TIFAD_PH_CTR_MODE=branch: a 2-block pre-LN transformer (masked
    full self-attention, blocks.MaskedMHA + conv-1 FFN) on the DETACHED
    stem features, trained by the phase-chunk contrastive loss only; the
    phase head reads its output. Attribute `phasebranch`."""

    def __init__(self, dim, n_head=4, n_blocks=2, ffn_mult=4, pdrop=0.0):
        super().__init__()
        from .blocks import LayerNorm
        self.dim = int(dim)
        self.n_head = int(n_head)
        self.ln1 = nn.ModuleList([LayerNorm(dim) for _ in range(n_blocks)])
        self.ln2 = nn.ModuleList([LayerNorm(dim) for _ in range(n_blocks)])
        # fused qkv + out projections; attention via the memory-efficient
        # scaled_dot_product_attention kernel (no T x T materialization)
        self.qkv = nn.ModuleList([nn.Conv1d(dim, 3 * dim, 1) for _ in range(n_blocks)])
        self.proj = nn.ModuleList([nn.Conv1d(dim, dim, 1) for _ in range(n_blocks)])
        self.ffn = nn.ModuleList([nn.Sequential(
            nn.Conv1d(dim, dim * ffn_mult, 1), nn.GELU(),
            nn.Conv1d(dim * ffn_mult, dim, 1)) for _ in range(n_blocks)])
        self.ln_out = LayerNorm(dim)

    def _attn(self, x, qkv, proj, mask):
        B, D, T = x.shape
        h = self.n_head
        q, k, v = qkv(x).view(B, 3, h, D // h, T).unbind(1)        # (B, h, d, T)
        q, k, v = (z.transpose(2, 3) for z in (q, k, v))            # (B, h, T, d)
        am = mask[:, None, None, :]                                 # (B, 1, 1, T)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=am)  # (B, h, T, d)
        out = out.transpose(2, 3).reshape(B, D, T)
        return proj(out)

    def forward(self, x, mask):
        """x (B, D, T); mask (B, T) bool -> (B, D, T)."""
        mask = mask.bool()
        mf = mask.unsqueeze(1).float()                          # (B, 1, T)
        x = x.detach().float() * mf
        for ln1, qkv, proj, ln2, ff in zip(self.ln1, self.qkv, self.proj,
                                            self.ln2, self.ffn):
            x = x + self._attn(ln1(x), qkv, proj, mask) * mf
            x = x + ff(ln2(x)) * mf
        return self.ln_out(x) * mf


class PhaseContrast(nn.Module):
    """TIFAD_PH_CTR=1: supervised contrastive loss (Khosla et al. 2020)
    on the K=10 position-decile chunks of every seen GT segment in the
    batch (chunk = mean of the level-0 location features inside it,
    qhead.pool_span_chunks), through a projection head dim -> dim -> 128
    (L2-normalized), temperature 0.1: positives = the same decile of any
    OTHER instance (any class), negatives = every other chunk INCLUDING
    the other deciles of the same instance. Plus the instance-diversity
    term: mean pairwise cosine among the 10 projected chunks of one
    instance, weight div_w (0.1). Attribute `phasectr`."""

    def __init__(self, dim, K=10, out_dim=128, temp=0.1, div_w=0.1):
        super().__init__()
        self.K = int(K)
        self.temp = float(temp)
        self.div_w = float(div_w)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True),
                                  nn.Linear(dim, out_dim))

    def forward(self, chunks):
        """chunks (M, K, D) (M instances of the batch, graph attached for
        'joint') -> (supcon loss, diversity loss, n_anchors)."""
        M, K, D = chunks.shape
        z = F.normalize(self.proj(chunks.float().reshape(M * K, D)), dim=-1)
        zero = z.new_zeros(())
        if M < 2:
            return zero + 0.0 * z.sum(), zero + 0.0 * z.sum(), 0
        lab = torch.arange(K, device=z.device).repeat(M)             # (MK,)
        inst = torch.arange(M, device=z.device).repeat_interleave(K)
        sim = (z @ z.t()) / self.temp                                 # (MK, MK)
        eye = torch.eye(M * K, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(eye, -1e9)
        pos = (lab[:, None] == lab[None, :]) & (inst[:, None] != inst[None, :]) & ~eye
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        npos = pos.float().sum(dim=1)
        anc = npos > 0
        supcon = -(logp * pos.float()).sum(dim=1)[anc] / npos[anc]
        zi = z.reshape(M, K, -1)
        cos = torch.einsum('mkd,mjd->mkj', zi, zi)
        offd = ~torch.eye(K, dtype=torch.bool, device=z.device)
        div = cos[:, offd].mean()
        return supcon.mean(), div, int(anc.sum())


@torch.no_grad()
def gt_chunks(x0, mask0, gts_list, K=10):
    """x0 (B, D, T0) (grad allowed by the caller's choice), mask0 (B, T0),
    gts_list B x (G_b, 2) -> chunk INDEX helper: returns (spans (B, Gmax,
    2), valid (B, Gmax)) padded, for qhead.pool_span_chunks."""
    B = x0.size(0)
    G = max([g.size(0) for g in gts_list] + [1])
    spans = x0.new_zeros((B, G, 2))
    valid = torch.zeros((B, G), dtype=torch.bool, device=x0.device)
    for b, g in enumerate(gts_list):
        if g.numel() > 0:
            spans[b, :g.size(0)] = g.float()
            valid[b, :g.size(0)] = True
    return spans, valid


# ----------------------------------------------------------------------
@torch.no_grad()
def build_windows(segs, Tv, ctx=0.5, max_w=128):
    """Sampled level-0 positions of the 1.5x-expanded window of every
    span. segs (N, 2) float grid units; Tv = number of valid level-0
    locations. Returns pos (N, W) long in [0, Tv), valid (N, W) bool,
    n_win (N,) = window size before subsampling."""
    s = segs[:, 0].float()
    e = torch.maximum(segs[:, 1].float(), s + 1e-3)
    L = e - s
    lo = torch.floor(s - ctx * L).clamp(0, max(Tv - 1, 0)).long()
    hi = torch.ceil(e + ctx * L).clamp(1, max(Tv, 1)).long()     # exclusive
    hi = torch.maximum(hi, lo + 1)
    n = hi - lo                                                   # (N,)
    Wm = int(min(max(int(n.max().item()), 1), max_w))
    j = torch.arange(Wm, device=segs.device).unsqueeze(0)         # (1, Wm)
    sub = n > Wm
    seq_pos = lo[:, None] + j
    step_pos = lo[:, None] + torch.round(
        j.float() * (n[:, None] - 1).float() / max(Wm - 1, 1)).long()
    pos = torch.where(sub[:, None], step_pos, seq_pos)
    valid = sub[:, None] | (j < n[:, None])
    pos = pos.clamp(0, max(Tv - 1, 0))
    return pos, valid, n


@torch.no_grad()
def phase_parse(lp_rows, pos, valid, segs, K):
    """Batched transcript-constrained Viterbi. lp_rows (N, W, K+1)
    log-posteriors at the sampled positions pos (N, W) (grid units),
    valid (N, W) bool; segs (N, 2) the candidate spans. States: 0 =
    BG_pre, 1..K = phases, K+1 = BG_post; transitions s->s, s->s+1;
    start in {0, 1}; end in {K, K+1}. Returns parsed (N, 2), ok (N,)
    bool, compl (N,), llr (N,), states (N, W) long."""
    N, W, _ = lp_rows.shape
    S = K + 2
    dev = lp_rows.device
    lp = lp_rows.float()
    bg = lp[..., K]                                               # (N, W)
    emis = torch.cat([bg.unsqueeze(-1), lp[..., :K], bg.unsqueeze(-1)],
                     dim=-1)                                      # (N, W, S)
    emis = emis.masked_fill(~valid.unsqueeze(-1), 0.0)
    score = emis.new_full((N, S), NEG)
    score[:, 0] = emis[:, 0, 0]
    score[:, 1] = emis[:, 0, 1]
    back = torch.zeros((N, W, S), dtype=torch.bool, device=dev)
    negcol = emis.new_full((N, 1), NEG)
    for t in range(1, W):
        move = torch.cat([negcol, score[:, :-1]], dim=1)
        take = (move > score) & valid[:, t].unsqueeze(1)
        score = torch.where(take, move, score) + emis[:, t]
        back[:, t] = take
    end_sc, end_arg = score[:, K:K + 2].max(dim=1)
    ok = end_sc > NEG / 2
    st = end_arg + K
    states = torch.zeros((N, W), dtype=torch.long, device=dev)
    states[:, W - 1] = st
    ar = torch.arange(N, device=dev)
    for t in range(W - 1, 0, -1):
        st = st - back[ar, t, st].long()
        states[:, t - 1] = st
    posf = pos.float()
    big = 1e9
    okv = ok.unsqueeze(1) & valid
    is0 = okv & (states == 0)
    is1 = okv & (states == 1)
    isK = okv & (states == K)
    isB = okv & (states == K + 1)
    first1 = torch.where(is1, posf, torch.full_like(posf, big)).min(dim=1).values
    lastK = torch.where(isK, posf, torch.full_like(posf, -big)).max(dim=1).values
    last0 = torch.where(is0, posf, torch.full_like(posf, -big)).max(dim=1).values
    firstB = torch.where(isB, posf, torch.full_like(posf, big)).min(dim=1).values
    s_p = torch.where(is0.any(dim=1), 0.5 * (last0 + first1), first1 - 0.5)
    e_p = torch.where(isB.any(dim=1), 0.5 * (lastK + firstB), lastK + 0.5)
    parsed = torch.stack([s_p, e_p], dim=1)
    parsed = torch.where(ok.unsqueeze(1), parsed, segs.float())
    # completeness: distinct phases visited inside the ORIGINAL span
    inside = valid & (posf >= segs[:, 0:1].float()) & \
        (posf <= segs[:, 1:2].float())
    ph_in = okv & inside & (states >= 1) & (states <= K)
    oh = torch.zeros((N, S), device=dev).scatter_add_(
        1, states.masked_fill(~ph_in, 0), ph_in.float())
    compl = oh[:, 1:K + 1].gt(0).sum(dim=1).float() / K
    compl = torch.where(ok, compl, torch.ones_like(compl))
    llr = end_sc - (bg * valid.float()).sum(dim=1)
    llr = torch.where(ok, llr, torch.zeros_like(llr))
    return parsed, ok, compl, llr, states


@torch.no_grad()
def actionness_from_fg(out_fg, points, fpn_masks):
    """Class-agnostic ACTIONNESS on the level-0 grid from the detector's
    per-level foreground head: for every level-0 time, the fg logit of the
    NEAREST location of every level (invalid -> -30), combined by
    logsumexp over levels. The per-level fg head is positive only for GT
    whose length falls in that level's regression range, so no single
    logits; points F x (T_l, 4); fpn_masks F x (B, T_l). -> (B, T0)."""
    t0 = points[0][:, 0]
    acc = []
    for fl, pts, ml in zip(out_fg, points, fpn_masks):
        tl = pts[:, 0].contiguous()
        f = fl.detach().float()[..., 0]                            # (B, T_l)
        f = torch.where(ml.bool(), f, torch.full_like(f, -30.0))
        idx = torch.searchsorted(tl, t0.contiguous()).clamp(0, tl.numel() - 1)
        idx2 = (idx - 1).clamp(min=0)
        near = torch.where((tl[idx] - t0).abs() <= (tl[idx2] - t0).abs(),
                           idx, idx2)                              # (T0,)
        acc.append(f[:, near])
    return torch.logsumexp(torch.stack(acc, dim=-1), dim=-1)      # (B, T0)


def gather_rows(lp, pos, cls_idxs=None):
    """lp (T0, K+1) or (C, T0, K+1) [text source, cls_idxs (N,)]; pos
    (N, W) -> (N, W, K+1)."""
    if lp.dim() == 3:
        assert cls_idxs is not None, "[PH-text] class indices missing"
        return lp[cls_idxs.long().unsqueeze(1).expand_as(pos), pos]
    return lp[pos]


# ----------------------------------------------------------------------
class PhaseBoundaryHead(BoundaryHead):
    """v5e boundary head on the PHASE PROFILE of the expanded window
    (see module docstring). Inherits apply / train_losses from R1."""

    def __init__(self, K=10, W=32, hidden=256, ctx=0.5, rng=0.5, w=1.0,
                 n_levels=6, iters=1):
        nn.Module.__init__(self)
        self.K = int(K)
        self.W = int(W)
        self.ctx = float(ctx)
        self.rng = float(rng)
        self.w = float(w)
        self.n_levels = int(n_levels)
        self.iters = int(iters)
        self.in_dim = 2 * self.W * (self.K + 1) + self.n_levels + 1
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @torch.no_grad()
    def profile(self, lp, segs, Tv, cls_idxs=None):
        """lp (T0, K+1) | (C, T0, K+1); segs (N, 2) -> (N, 2W, K+1):
        log-posteriors (clamped at -10, / 10) at 2W positions uniformly
        over [s - ctx L, e + ctx L]; positions outside [0, Tv) are 0."""
        s = segs[:, 0].float()
        e = torch.maximum(segs[:, 1].float(), s + 1e-3)
        L = e - s
        n2 = 2 * self.W
        j = (torch.arange(n2, device=segs.device).float() + 0.5) / n2
        p = (s - self.ctx * L)[:, None] + j[None, :] * ((1 + 2 * self.ctx) * L)[:, None]
        idx = p.round().long()
        inb = (idx >= 0) & (idx < Tv)
        idx = idx.clamp(0, max(Tv - 1, 0))
        rows = gather_rows(lp, idx, cls_idxs).float()
        rows = rows.clamp(min=-10.0) / 10.0
        return rows * inb.unsqueeze(-1).float()

    def rel(self, prof, ld):
        """prof (N, 2W, K+1) detached profile; ld (N, F+1) level one-hot
        + log-duration -> relative corrections (N, 2) in (-rng, rng)."""
        x = torch.cat([prof.detach().flatten(1).float(),
                       ld.detach().float()], dim=1)
        return torch.tanh(self.mlp(x).float()) * self.rng


# ----------------------------------------------------------------------
# CLI: train + save the probe head on the CHAMPION's own train-video
# features (scripts/phase_probe.py machinery; the existing probe dumps
# were harvested from the leafref stage-1 checkpoints, whose cls trunk is
# conditioned on a different text design, so the champion is re-dumped).
def _cli():
    import argparse
    import sys
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-probe', action='store_true')
    ap.add_argument('--splits', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--ckpt',
                    default='ckpt/thumos14_i3d_seqchain_{n}/mAP_best.pth.tar')
    ap.add_argument('--config', default='configs/thumos14_i3d.yaml')
    ap.add_argument('--dump-dir', default='data/phase_pseudo/probe_dump')
    ap.add_argument('--tag', default='seqchain')
    ap.add_argument('--rule', default='in', choices=['in', 'leaf'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--attndesign', default='phasechain',
                    help="champion text design (TIFAD_ATTNDESIGN); '' = none")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    os.chdir(root)
    if args.attndesign:
        os.environ['TIFAD_ATTNDESIGN'] = args.attndesign
    os.environ.pop('TIFAD_PH_SRC', None)
    sys.path.insert(0, os.path.join(root, 'scripts'))
    import numpy as np
    import phase_probe as pp                     # sets the reference envs
    from libs.core import load_config
    from libs.utils import fix_random_seed
    assert args.train_probe, "nothing to do (use --train-probe)"
    for n in args.splits:
        t0 = time.time()
        tr_path = os.path.join(args.dump_dir, f'{args.tag}_split{n}_train.npz')
        out_path = os.path.join(args.dump_dir,
                                f'probe_head_{args.tag}_split{n}.pt')
        cfg = load_config(args.config)
        if not os.path.exists(tr_path):
            fix_random_seed(cfg['init_rand_seed'], include_cuda=True)
            model, notes = pp.build_model(cfg, n, args.ckpt.format(n=n))
            print(f"[probe-train] split {n}: {args.ckpt.format(n=n)} loaded; "
                  f"notes={notes}", flush=True)
            pp.dump_train(model, cfg, n, tr_path)
            del model
            torch.cuda.empty_cache()
        tr = dict(np.load(tr_path))
        m = tr[f'{args.rule}_phase'] >= 0
        X = tr['feat'][m].astype(np.float32)
        y = tr[f'{args.rule}_phase'][m].astype(np.int64)
        # held-out check: 10% of the VIDEOS
        vids = tr['video'][m]
        uv = np.unique(vids)
        rs = np.random.RandomState(args.seed)
        ho = set(rs.choice(uv, size=max(1, len(uv) // 10), replace=False).tolist())
        hm = np.array([v in ho for v in vids])
        head = pp.train_head(X[~hm], y[~hm], args.seed, X.shape[1],
                             epochs=args.epochs)
        pa, _ = pp.predict(head, X[hm])
        acc_ho = float((pa == y[hm]).mean())
        acc3_ho = float((pp.MERGE3[pa] == pp.MERGE3[y[hm]]).mean())
        # final head on ALL rows (the probe of record)
        head = pp.train_head(X, y, args.seed, X.shape[1], epochs=args.epochs)
        pa, _ = pp.predict(head, X)
        sd = {k: v.detach().cpu() for k, v in head.state_dict().items()}
        sd['meta'] = {'ckpt': args.ckpt.format(n=n), 'rule': args.rule,
                      'seed': args.seed, 'epochs': args.epochs,
                      'n_rows': int(len(y)), 'train_acc10': float((pa == y).mean()),
                      'heldout_video_acc10': acc_ho,
                      'heldout_video_acc3': acc3_ho,
                      'phase_hist': np.bincount(y, minlength=pp.N_PHASE).tolist()}
        torch.save(sd, out_path)
        print(f"[probe-train] split {n}: rows={len(y)} train acc10="
              f"{sd['meta']['train_acc10']:.3f} held-out-video acc10="
              f"{acc_ho:.3f} acc3={acc3_ho:.3f} -> {out_path} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == '__main__':
    _cli()


# ----------------------------------------------------------------------
# Boundary arms B2-B4 (2026-09-18): start/end boundary map with
# snapping (TIFAD_BM=1 / TIFAD_BM_SNAP=<r>), shrink sanity sweep
# (TIFAD_SHRINK=<f>), re-anchoring (TIFAD_REANCHOR=1|pre). All eval-time
# span edits happen BEFORE soft-NMS (re-anchoring: after NMS by default).
class BoundaryMap(nn.Module):
    """B2: class-agnostic START / END probability per level-0 location
    (BSN/BMN-style boundary map) on the DETACHED level-0 trunk features:
    LayerNorm -> MLP (D -> hidden -> 2) -> sigmoid. Train (sequential on
    the frozen champion, FREEZE_EXCEPT=bmhead): BCE against the seen GT
    boundaries widened to +-1 location, negatives sampled neg_ratio (3) x
    positives per side. Eval (TIFAD_BM_SNAP=r): for every pre-NMS
    candidate [s, e] of length L, the start snaps to the arg-max start
    probability within +-r L of s if that peak exceeds thr (0.3); the end
    likewise. Attribute `bmhead`."""

    def __init__(self, dim, hidden=256, neg_ratio=3.0, widen=1, thr=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, 2))
        self.neg_ratio = float(neg_ratio)
        self.widen = int(widen)
        self.thr = float(thr)

    def logits(self, x0):
        """x0 (B, D, T0) -> (B, T0, 2) start / end logits (input detached)."""
        return self.mlp(self.norm(x0.detach().float().transpose(1, 2)))

    def train_loss(self, lg, t0, valid, gts, generator=None):
        """ONE item: lg (T0, 2); t0 (T0,); valid (T0,) bool; gts (G, 2).
        Returns (bce_sum, n, n_pos)."""
        T0 = lg.size(0)
        z = lg.new_zeros(())
        if gts is None or gts.numel() == 0 or not bool(valid.any()):
            return z, 0, 0
        gts = gts.float()
        with torch.no_grad():
            tgt = torch.zeros((T0, 2), device=lg.device)
            for side in (0, 1):
                d = (t0[:, None] - gts[None, :, side]).abs().min(dim=1).values
                tgt[:, side] = (d <= self.widen + 0.5).float()
            sel = torch.zeros((T0, 2), dtype=torch.bool, device=lg.device)
            for side in (0, 1):
                pos = valid & (tgt[:, side] > 0)
                neg = valid & (tgt[:, side] == 0)
                n_pos = int(pos.sum())
                n_keep = int(min(int(neg.sum()), max(int(round(self.neg_ratio * n_pos)), 8)))
                sel[:, side] = pos
                if n_keep > 0:
                    ni = neg.nonzero(as_tuple=True)[0]
                    pick = torch.randperm(ni.numel(), device=lg.device,
                                          generator=generator)[:n_keep]
                    sel[ni[pick], side] = True
        if not bool(sel.any()):
            return z, 0, 0
        bce = F.binary_cross_entropy_with_logits(lg.float()[sel], tgt[sel],
                                                 reduction='sum')
        return bce, int(sel.sum()), int((tgt[sel] > 0).sum())

    @torch.no_grad()
    def snap(self, prob, segs, Tv, r):
        """prob (T0, 2) start / end probabilities; segs (N, 2) -> snapped
        (N, 2). Search window +-r L around each side (clamped to the valid
        range), arg-max peak must exceed thr."""
        if segs.numel() == 0 or Tv < 1:
            return segs
        s = segs[:, 0].float()
        e = torch.maximum(segs[:, 1].float(), s + 1e-3)
        L = e - s
        out = segs.float().clone()
        t = torch.arange(Tv, device=segs.device).float()
        for side, b in ((0, s), (1, e)):
            lo = (b - r * L).unsqueeze(1)
            hi = (b + r * L).unsqueeze(1)
            inw = (t[None, :] >= lo) & (t[None, :] <= hi)         # (N, Tv)
            p = prob[:Tv, side][None, :].expand_as(inw)
            p = torch.where(inw, p, torch.full_like(p, -1.0))
            pk, pi = p.max(dim=1)
            use = pk > self.thr
            out[:, side] = torch.where(use, pi.float(), b)
        out[:, 1] = torch.maximum(out[:, 1], out[:, 0] + 1e-3)
        return out.to(segs.dtype)


@torch.no_grad()
def shrink_spans(segs, f):
    """B3: scale every span's length by f about its center."""
    if segs.numel() == 0:
        return segs
    c = 0.5 * (segs[:, 0] + segs[:, 1])
    h = 0.5 * (segs[:, 1] - segs[:, 0]) * f
    return torch.stack([c - h, c + h], dim=1)


@torch.no_grad()
def reanchor_spans(segs, lvls, points, offsets):
    """B4: for every span, the location NEAREST its center on the SAME
    pyramid level re-reads the regression head; the re-read span is
    averaged with the original. segs (N, 2) level-0 units; lvls (N,)
    long; points F x (T_l, 4); offsets F x (T_l, 2) (this video)."""
    if segs.numel() == 0:
        return segs
    out = segs.float().clone()
    c = 0.5 * (segs[:, 0] + segs[:, 1]).float()
    for l in lvls.unique().tolist():
        m = lvls == l
        pts = points[l]
        tl = pts[:, 0].contiguous()
        idx = torch.searchsorted(tl, c[m].contiguous()).clamp(0, tl.numel() - 1)
        idx2 = (idx - 1).clamp(min=0)
        near = torch.where((tl[idx] - c[m]).abs() <= (tl[idx2] - c[m]).abs(),
                           idx, idx2)
        off = offsets[l][near].float()
        st = pts[near, 3]
        rs = tl[near] - off[:, 0] * st
        re_ = tl[near] + off[:, 1] * st
        out[m, 0] = 0.5 * (out[m, 0] + rs)
        out[m, 1] = 0.5 * (out[m, 1] + re_)
    out[:, 1] = torch.maximum(out[:, 1], out[:, 0] + 1e-3)
    return out.to(segs.dtype)
