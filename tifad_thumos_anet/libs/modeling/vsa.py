"""VSA (TIFAD_VSA=1): VIDEO-SIDE SUBTREE ATTENTION.

Mirror of the TEXT-side subtree composition (text.py: TIFAD_ATTNDESIGN=
phasechain banded neighbour self-attention among the ordered phase
embeddings, then the H4b action query attending over [action; phases]),
applied PER CANDIDATE SEGMENT on the video side:

  1. chunk pooling : for every location n of pyramid level l, the
     regression head's (DETACHED) offsets give the candidate segment
     [n - d_s, n + d_e] on the level's own grid; it is clamped to the
     level, split into K ordered temporal chunks (video-side phases) and
     the trunk features are mean-pooled inside each chunk via integral
     (cumulative-sum) features -- O(T_l * K) gathers, no loops. Padded
     timesteps are excluded from the pools (masked cumsum + count).
  2. neighbour self-attention among the K chunk embeddings with the SAME
     banded mask as the text side (|i-j| <= 1 attend, else masked),
     residual, out_proj ZERO-INIT.
  3. action-clip cross-attention: the location's own feature f_n queries
     [f_n; refined chunks] (K+1 keys), residual, out_proj ZERO-INIT.
  4. the result V_l (B, D, T_l) REPLACES the cls-trunk feature the cosine
     scoring reads (train AND inference, every level, every location --
     negatives included); the reg/fg/cn heads keep the raw features.

Init-equality invariant: BOTH out_proj's are zero-init (weight AND bias),
so V_l == X_l bitwise at step 0 (the residual adds an exact 0.0). Keep
this if you modify the block. Parameters are shared across levels.
Envs: TIFAD_VSA=1, TIFAD_VSA_K (default 10), TIFAD_VSA_HEADS (default 4).
Byte-inert when TIFAD_VSA is unset (module never constructed).

VERSION 2 (2026-09-17) -- three env-gated additions, each init-equal:

  (A) TIFAD_VSA_XM=1  CROSS-MODAL ORDERED PHASE MATCHING. The refined
      video chunks Z'_n (after step 2) are matched, chunk k to TEXT phase
      k, against every class c of the current vocabulary:
          m(n, c) = sum_k softmax(w)_k * cos(A_v z'_{n,k}, A_t e_k(c))
      A_v, A_t: Linear(D, D) IDENTITY-init adaptors (video / text side),
      w in R^K position weights ZERO-init (uniform softmax), e_k(c) the
      class's k-th refined phase-sentence embedding (text.py stash
      `_vsa_phase_embs`, taken after the phasechain neighbour attention
      when that design is active) passed through the model's OWN text
      calibration (L2-normalize, text_proj(x/2)) by the caller so both
      sides live in the base matmul's space. meta_archs adds
          logits(n, c) += lam * tau(n, c) * m(n, c)
      with lam (`xm_lam`) a learned scalar ZERO-init (init-equality) and
      tau(n, c) = ||v_n|| * ||p_c|| / logit_scale.exp() the (detached)
      scale that multiplies the base cosine at that (n, c) -- the base
      logit is <v_n, p_c> / logit_scale.exp() + bias -- so lam * m is
      expressed in base-cosine units. One einsum per level:
      (B, T, K, D) x (C, K, D) -> (B, T, C).
      TIFAD_VSA_XM_ONLY=1: keep the matching term but SKIP step 3 (the
      v1 clip attention; v_n = f_n; `clip_attn` is not constructed) to
      isolate the term.
  (B) TIFAD_VSA_SOFTPHASE=1  PREDICTED PHASE TIMESPANS. The trunk phase
      head of the LEAFLOSS code (meta_archs `leaf_head`, MLP D->D->10,
      requires TIFAD_LEAFLOSS=1 TIFAD_LEAFLOSS_HEAD=1; train it DETACHED
      with TIFAD_LEAFLOSS_DETACH=1 TIFAD_LEAFLOSS_W=1.0 in the same
      stage) gives a phase posterior p(k|t) = softmax(head(f_t.detach()))
      at every location of the level. Chunk k of candidate n is ALSO
      pooled as the posterior-weighted mean of f_t over the WHOLE span
      (integral features of p(k|t) f_t and of p(k|t): K x D channels),
      and the two poolings are blended
          z_k = (1 - mix) * z_equal_k + mix * z_post_k
      (TIFAD_VSA_SOFTPHASE_MIX, default 0.5). Spans whose posterior mass
      for phase k is < 1e-3 keep z_equal_k (guard). Same in train / eval.
      NOTE: with SOFTPHASE the caller feeds the leaf loss the PRE-VSA
      cls features (the same tensor the head reads here).
  (C) unfrozen regression / global low-lr: no code here -- see
      train_eval.py TIFAD_FREEZE_EXCEPT (comma list, e.g.
      vsa,leaf_head,reg_head) and train_utils.make_optimizer
      TIFAD_LR_MULT_REST.
  (D) TIFAD_VSA_POSEMB=1  learned phase-index (positional) embeddings
      `posemb` (K, D), init normal std 0.02, ADDED ONLY TO THE TOKENS the
      two attentions consume (q/k/v of the chain attention, the K chunk
      keys/values of the clip attention) -- never to the residual path
      (Z' for the XM term / align loss, v_n) -- so attention can become
      position-selective (the v1 diagnosis: both attentions stayed
      uniform, in_proj never moved). out_proj's stay zero-init => still
      exactly init-equal.
  (E) TIFAD_VSA_ALIGN_W=<w>  phase-ALIGNMENT loss (train only; requires
      XM): at positive locations n with GT class c, chunk k is scored
      against the 10 phases j of c with the SAME adaptors as the XM term,
      cos(A_v z'_{n,k}, A_t e_j(c)) / 0.07, softmax CE with target j = k,
      mean over k, sum over positives / loss_normalizer, times w
      (meta_archs losses(): align_loss, diagnostic align_acc = fraction
      of chunks whose argmax phase is k). Gradients reach A_v, A_t,
      chain_attn (in_proj too -- NOT gated by a zero out_proj here),
      posemb; clip_attn is untouched by it.
VERSION 3 (2026-09-17):
  TIFAD_VSA_NOSELF=1     clip-attention keys/values are the K chunk tokens
                         only (no self token f_n).
  TIFAD_VSA_OUTINIT=<s>  out_proj WEIGHTS of chain_attn / clip_attn are
                         normal(std=s) instead of zero (bias stays 0) so
                         in_proj receives gradient from step 0; init-
                         equality then holds only approximately.
  TIFAD_VSA_MINLVL=<l>   VSA runs only on pyramid levels >= l; lower levels
                         pass through unchanged (v_n = f_n, no XM term, no
                         alignment supervision -- their zv are None).
  TIFAD_VSA_XM_RESCORE=<gamma>  (meta_archs, EVAL ONLY) candidate scores
                         *= exp(gamma * (m(n, c) - mean_c m)) with m the raw
                         ordered match in [-1, 1] and the mean over the
                         video's pre-NMS candidates of class c.
  TIFAD_VSA_ALIGN_UW=1   (meta_archs) Kendall logvar slot 3 for the
                         alignment loss.
All are byte-inert when unset (v1 code path untouched).

NESTED (2026-09-25): TIFAD_VSA_NESTED=1 (with TIFAD_VSA=1) selects
`VideoSubtreeAttnNested` below -- a SECOND level of sub-actions (Kp leaf
chunks per action chunk, TIFAD_VSA_KP, default 3) under the K action
chunks; the v2/v3 flags must be off. Same v1 call / return convention.
"""
import os as _os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class VideoSubtreeAttn(nn.Module):

    def __init__(self, dim, K=10, heads=4, xm=False, xm_only=False,
                 softphase=False, softphase_mix=0.5, posemb=False,
                 noself=False, out_init=0.0, min_lvl=0,
                 learn_bounds=False, tau_init=0.1):
        super().__init__()
        assert K >= 1, "[VSA] K must be >= 1"
        self.K = K
        self.xm = bool(xm)
        self.xm_only = bool(xm_only) and self.xm
        self.softphase = bool(softphase)
        self.softphase_mix = float(softphase_mix)
        self.noself = bool(noself)
        self.min_lvl = int(min_lvl)
        self.out_init = float(out_init)

        def _init_out(m):
            if self.out_init > 0:
                nn.init.normal_(m.out_proj.weight, std=self.out_init)
            else:
                nn.init.zeros_(m.out_proj.weight)
            nn.init.zeros_(m.out_proj.bias)
        # step 2: banded neighbour self-attention among the K chunks
        self.chain_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        _init_out(self.chain_attn)
        # step 3: action-clip cross-attention (query = own feature);
        # skipped entirely under XM_ONLY (v_n = f_n)
        self.clip_attn = None
        if not self.xm_only:
            self.clip_attn = nn.MultiheadAttention(dim, heads,
                                                   batch_first=True)
            _init_out(self.clip_attn)
        # (A) cross-modal ordered phase matching
        if self.xm:
            self.xm_av = nn.Linear(dim, dim)      # video-side adaptor
            self.xm_at = nn.Linear(dim, dim)      # text-side adaptor
            for _a in (self.xm_av, self.xm_at):
                nn.init.eye_(_a.weight)
                nn.init.zeros_(_a.bias)
            self.xm_w = nn.Parameter(torch.zeros(K))     # position weights
            self.xm_lam = nn.Parameter(torch.zeros([]))  # term weight, 0
        # (D) phase-index embeddings for the attention tokens only
        self.posemb = None
        if posemb:
            self.posemb = nn.Parameter(torch.empty(K, dim))
            nn.init.normal_(self.posemb, std=0.02)
        # banded mask, True = MASKED (|i-j| > 1); PyTorch turns a bool mask
        # into the additive 0 / -inf mask internally. Non-persistent so the
        # state_dict holds only the two MHA's (INIT_FROM: vsa.* missing only).
        _i = torch.arange(K)
        self.register_buffer(
            'band_mask', (_i[:, None] - _i[None, :]).abs() > 1,
            persistent=False)
        # learnable NON-EQUAL chunk boundaries
        # (TIFAD_VSA_LEARN_BOUNDS=1; see _pool_learned). Built after the MHAs
        # so their init is unchanged; all heads zero-init => uniform widths
        # (= the equal split) and the global temperature at step 0.
        self.learn_bounds = bool(learn_bounds)
        if self.learn_bounds:
            self.width_loc = nn.Linear(dim, K)     # widths from the location feature f_n
            self.width_cnt = nn.Linear(dim, 1)     # + per-chunk content term (equal-split means)
            self.tau_loc = nn.Linear(dim, 1)       # per-location log-temperature offset
            for _m in (self.width_loc, self.width_cnt, self.tau_loc):
                nn.init.zeros_(_m.weight)
                nn.init.zeros_(_m.bias)
            self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())
            # TIFAD_VSA_LEARN_GAPS=1: sub-actions need not be
            # adjacent -- K+1 learnable BACKGROUND GAPS (before, between, after the K chunks)
            # share the span with the K widths (softmax over 2K+1 logits). Gap logits start
            # at -6 (bias) => each gap ~e^-6/K of the span, <0.3 % in total: the init is the
            # equal split up to that shift; the model can open gaps where frames are background.
            self.learn_gaps = _os.environ.get('TIFAD_VSA_LEARN_GAPS', '') == '1'
            if self.learn_gaps:
                self.gap_loc = nn.Linear(dim, K + 1)
                nn.init.zeros_(self.gap_loc.weight)
                nn.init.constant_(self.gap_loc.bias, -6.0)
            # query-chunk size for the checkpointed membership pooling (memory only)
            self.LB_CHUNK = int(_os.environ.get('TIFAD_LB_CHUNK', '256'))

    # ------------------------------------------------------------------
    def _span(self, offsets, T):
        """offsets: (B, T, 2) DETACHED (d_s, d_e) in grid units.
        Returns integer span [s_i, e_i] (inclusive) per location, clamped
        to the level, and its length L = e_i - s_i + 1 >= 1."""
        n = torch.arange(T, device=offsets.device, dtype=offsets.dtype)
        s = (n.unsqueeze(0) - offsets[..., 0]).clamp(0, T - 1)  # (B, T)
        e = (n.unsqueeze(0) + offsets[..., 1]).clamp(0, T - 1)  # (B, T)
        s_i = torch.floor(s).long()
        e_i = torch.ceil(e).long()
        e_i = torch.maximum(e_i, s_i)          # guard: >= 1 grid unit
        return s_i, e_i, (e_i - s_i + 1)

    def _pool_chunks(self, x, offsets, mask):
        """x: (B, D, T) level features; offsets: (B, T, 2) DETACHED
        (d_s, d_e) in grid units; mask: (B, 1, T) bool valid.
        Returns Z (B, T, K, D): mean of x inside each of the K ordered
        chunks of the candidate segment at every location."""
        B, D, T = x.shape
        K = self.K
        m = mask.to(x.dtype)                                    # (B, 1, T)
        # integral features / counts: cs[..., t] = sum_{u < t}
        cs = F.pad((x * m).cumsum(dim=-1), (1, 0))              # (B, D, T+1)
        cm = F.pad(m.cumsum(dim=-1), (1, 0))                    # (B, 1, T+1)
        s_i, e_i, L = self._span(offsets, T)                    # (B, T)
        k = torch.arange(K + 1, device=x.device)                # (K+1,)
        # chunk boundaries (integer units): s_i + floor(L * k / K)
        bnd = s_i.unsqueeze(-1) + (L.unsqueeze(-1) * k) // K    # (B, T, K+1)
        # TIFAD_VSA_JITTER=<delta>: every
        # interior cut moves by U(-delta, delta) chunk lengths (seeded, so a run is
        # deterministic), cuts re-sorted so chunks stay ordered and contiguous, kept
        # inside the span. Eval-only probe of the trained equal-split model's tolerance
        # to boundary error. Byte-inert when unset.
        _jit = float(_os.environ.get('TIFAD_VSA_JITTER', '0') or 0)
        if _jit > 0 and K > 1:
            _g = torch.Generator(device=x.device); _g.manual_seed(int(_os.environ.get('TIFAD_VSA_JITTER_SEED', '0') or 0))
            _eps = (torch.rand(B, T, K - 1, generator=_g, device=x.device) * 2 - 1) * _jit * (L.unsqueeze(-1).to(torch.float32) / K)
            _b = bnd.to(torch.float32)
            # interior cuts only: clamp to [s_i, e_i] (so lo + 1 <= e_i + 1 <= T, the gather range), then sort
            # among themselves; the end cuts s_i and e_i + 1 never move (sorting the whole row would let a
            # pushed interior cut swap with an end cut and leave the span)
            _in = _b[..., 1:-1] + _eps
            _in = torch.minimum(torch.maximum(_in, s_i.unsqueeze(-1).to(_b.dtype)), e_i.unsqueeze(-1).to(_b.dtype))
            _b[..., 1:-1] = torch.sort(_in, dim=-1).values
            bnd = torch.round(_b).long()
        lo = bnd[..., :-1]
        hi = torch.maximum(bnd[..., 1:], lo + 1)   # >= 1 unit per chunk
        # (segments shorter than K units repeat units across chunks);
        # hi <= s_i + L = e_i + 1 <= T, so the gather stays in range
        lo = lo.reshape(B, T * K)
        hi = hi.reshape(B, T * K)
        cs_t = cs.transpose(1, 2)                               # (B, T+1, D)
        bidx = torch.arange(B, device=x.device).unsqueeze(-1)   # (B, 1)
        num = cs_t[bidx, hi] - cs_t[bidx, lo]                   # (B, T*K, D)
        cnt = (cm[:, 0, :][bidx, hi] - cm[:, 0, :][bidx, lo])   # (B, T*K)
        Z = num / cnt.clamp(min=1.0).unsqueeze(-1)
        return Z.reshape(B, T, K, D)

    def _pool_learned(self, x, offsets, mask):
        """Learnable non-equal chunk boundaries (port of actionformer6 vsa.py).
        x (B, D, T); offsets (B, T, 2) DETACHED grid units; mask (B, 1, T).
        1. continuous span [s, e] = [n - d_s, n + d_e + 1], e - s >= 1, in [0, T];
        2. coarse pass: equal-split chunk means Zeq (exact, as _pool_chunks);
        3. K width logits = width_loc(f_n) + width_cnt(Zeq_k) -> softmax w;
           boundaries b_k = s + L * cumsum(w)_k (monotone); zero-init => w = 1/K;
        4. soft membership of unit u in chunk k: sig((u+.5-b_{k-1})/tau) -
           sig((u+.5-b_k)/tau), tau = exp(log_tau + tau_loc(f_n)) * L / K;
           padded units excluded; Z_k = weighted mean. Returns (B, T, K, D)."""
        B, D, T = x.shape
        K = self.K
        Zeq = self._pool_chunks(x, offsets, mask)                           # (B, T, K, D)
        off = offsets.float()
        n = torch.arange(T, device=x.device, dtype=torch.float32)
        s = (n.unsqueeze(0) - off[..., 0]).clamp(0.0, float(T))
        e = (n.unsqueeze(0) + off[..., 1] + 1.0).clamp(0.0, float(T))
        e = torch.maximum(e, (s + 1.0).clamp(max=float(T)))
        s = torch.minimum(s, e - 1.0).clamp(min=0.0)
        L = e - s                                                           # (B, T) >= 1
        f = x.transpose(1, 2)                                               # (B, T, D)
        logits = self.width_loc(f).float() + self.width_cnt(Zeq).squeeze(-1).float()   # (B, T, K)
        if getattr(self, 'learn_gaps', False):
            # interleave: gap_0, w_1, gap_1, w_2, ..., w_K, gap_K  -> 2K+1 shares of the span
            g = self.gap_loc(f).float()                                     # (B, T, K+1)
            z = torch.stack([g[..., :K], logits], dim=-1).reshape(B, T, 2 * K)   # g0 w1 g1 w2 ... g_{K-1} wK
            z = torch.cat([z, g[..., K:]], dim=-1)                          # + gK  -> (B, T, 2K+1)
            w = z.softmax(dim=-1)
            cum = torch.cat([w.new_zeros(B, T, 1), w.cumsum(dim=-1)], dim=-1)   # (B, T, 2K+2)
            edges = s.unsqueeze(-1) + L.unsqueeze(-1) * cum
            lo, hi = edges[..., 1:-1:2], edges[..., 2::2]                   # chunk k: [after gap_{k-1}, before gap_k]
        else:
            w = logits.softmax(dim=-1)
            cum = torch.cat([w.new_zeros(B, T, 1), w.cumsum(dim=-1)], dim=-1)  # (B, T, K+1)
            bnd = s.unsqueeze(-1) + L.unsqueeze(-1) * cum                       # (B, T, K+1)
            lo, hi = bnd[..., :-1], bnd[..., 1:]
        tau = (self.log_tau.float() + self.tau_loc(f).float().squeeze(-1)).exp() * L / K
        tau = tau.clamp(min=1e-2)                                           # (B, T)
        # Memory (2026-11-07, after vsalb_s0/s1 OOMed on a shared card): the
        # (B, T, K, T) membership is ~0.5 GB per tensor at T=2304, K=10 and the
        # unchunked form kept ~1.3 GB of it for backward. Form it in query
        # chunks of LB_CHUNK locations under activation checkpointing instead:
        # same arithmetic per location, nothing T x T saved, peak ~T/LB_CHUNK
        # smaller. Verified equal to the unchunked form (values and grads).
        u = torch.arange(T, device=x.device, dtype=torch.float32) + 0.5
        m = mask.to(torch.float32).view(B, 1, 1, T)

        def _chunk(lo_c, hi_c, tau_c, x_, m_):
            uu = u.view(1, 1, 1, T); tt = tau_c.view(B, -1, 1, 1)
            memb = (torch.sigmoid((uu - lo_c.unsqueeze(-1)) / tt) - torch.sigmoid((uu - hi_c.unsqueeze(-1)) / tt)).clamp(min=0.0) * m_   # (B, t, K, T)
            num = torch.einsum('btku,bdu->btkd', memb.to(x_.dtype), x_)      # (B, t, K, D)
            den = memb.sum(-1).clamp(min=1e-4).unsqueeze(-1)
            return (num.float() / den).to(x_.dtype)

        step = max(1, min(T, int(self.LB_CHUNK)))
        use_ckpt = self.training and torch.is_grad_enabled()
        outs = []
        for t0 in range(0, T, step):
            args = (lo[:, t0:t0 + step], hi[:, t0:t0 + step], tau[:, t0:t0 + step], x, m)
            outs.append(checkpoint(_chunk, *args, use_reentrant=False) if use_ckpt else _chunk(*args))
        return torch.cat(outs, dim=1)                                       # (B, T, K, D)

    def _pool_softphase(self, x, offsets, mask, phase_head):
        """(B) posterior-weighted chunk pooling. x: (B, D, T); phase_head:
        MLP D -> K applied to x.detach() at every location.
        Returns Z_post (B, T, K, D) and the posterior mass per chunk
        cnt (B, T, K) inside the candidate span [s_i, e_i]."""
        B, D, T = x.shape
        K = self.K
        m = mask.to(x.dtype)                                    # (B, 1, T)
        xt = x.transpose(1, 2)                                  # (B, T, D)
        p = torch.softmax(phase_head(xt.detach()).float(), dim=-1)
        assert p.size(-1) == K, \
            f"[VSA-SOFTPHASE] phase head outputs {p.size(-1)} != K={K}"
        p = (p * m.transpose(1, 2)).to(x.dtype)                 # (B, T, K)
        # integral features of p(k|t) f_t (K*D channels) and of p(k|t)
        pf = (p.unsqueeze(-1) * xt.unsqueeze(2)).reshape(B, T, K * D)
        cpf = F.pad(pf.cumsum(dim=1), (0, 0, 1, 0))             # (B, T+1, K*D)
        cp = F.pad(p.cumsum(dim=1), (0, 0, 1, 0))               # (B, T+1, K)
        s_i, e_i, _ = self._span(offsets, T)                    # (B, T)
        hi = e_i + 1                                            # <= T
        bidx = torch.arange(B, device=x.device).unsqueeze(-1)   # (B, 1)
        num = cpf[bidx, hi] - cpf[bidx, s_i]                    # (B, T, K*D)
        cnt = cp[bidx, hi] - cp[bidx, s_i]                      # (B, T, K)
        Z = num.reshape(B, T, K, D) / cnt.clamp(min=1e-6).unsqueeze(-1)
        return Z, cnt

    # ------------------------------------------------------------------
    def forward(self, feats, offsets, fpn_masks, phase_embs=None,
                phase_head=None, want_align=False):
        """feats: F x (B, D, T_l); offsets: F x (B, T_l, 2) (detached by
        the caller); fpn_masks: F x (B, 1, T_l) bool.
        phase_embs: (C, K, D) CALIBRATED text phase embeddings of the
        current vocabulary (XM only); phase_head: the trunk phase head
        (SOFTPHASE only); want_align: also return the adapted, L2-normed
        chunk vectors zv F x (B, T_l, K, D) and the adapted, L2-normed
        phase bank et (C, K, D) for the (E) alignment loss (XM only).
        Returns (F x (B, D, T_l) video-side subtree embeddings,
                 F x (B, T_l, C) matching terms m(n, c) or None,
                 {'zv': [...], 'et': et} or None)."""
        if self.xm:
            assert phase_embs is not None, "[VSA-XM] phase_embs missing"
            assert phase_embs.dim() == 3 and phase_embs.size(1) == self.K, \
                f"[VSA-XM] phase bank {tuple(phase_embs.shape)} vs K={self.K}"
            et = F.normalize(self.xm_at(phase_embs), dim=-1)    # (C, K, D)
            sw = torch.softmax(self.xm_w, dim=0)                # (K,)
        if self.softphase:
            assert phase_head is not None, "[VSA-SOFTPHASE] head missing"
        assert not want_align or self.xm, "[VSA-ALIGN] requires TIFAD_VSA_XM=1"
        out, match, zvs = [], [], []
        for lvl, (x, off, mask) in enumerate(zip(feats, offsets, fpn_masks)):
            B, D, T = x.shape
            K = self.K
            if lvl < self.min_lvl:
                # MINLVL: pass-through level (no subtree, no XM term)
                out.append(x)
                if self.xm:
                    match.append(None)
                    if want_align:
                        zvs.append(None)
                continue
            if self.learn_bounds:
                Z = self._pool_learned(x, off, mask)            # (B, T, K, D), non-equal chunks
            else:
                Z = self._pool_chunks(x, off, mask)             # (B, T, K, D)
            if self.softphase:
                Zp, cnt = self._pool_softphase(x, off, mask, phase_head)
                w = self.softphase_mix * (cnt >= 1e-3).to(Z.dtype)
                w = w.unsqueeze(-1)                             # (B, T, K, 1)
                Z = (1.0 - w) * Z + w * Zp
            Z = Z.reshape(B * T, K, D)
            # step 2: banded neighbour self-attention, zero-init residual;
            # (D) posemb enters the attention TOKENS only, not the residual
            Zt = Z if self.posemb is None else Z + self.posemb.unsqueeze(0)
            _u, _ = self.chain_attn(Zt, Zt, Zt, attn_mask=self.band_mask,
                                    need_weights=False)
            Z = Z + _u
            f = x.transpose(1, 2).reshape(B * T, 1, D)          # (B*T, 1, D)
            if self.xm:
                # (A) ordered chunk-to-phase cosine, position-weighted
                zv = F.normalize(self.xm_av(Z.reshape(B, T, K, D)), dim=-1)
                mt = torch.einsum('btkd,ckd->btc', zv * sw.view(1, 1, K, 1),
                                  et)                           # (B, T, C)
                match.append(mt * mask.transpose(1, 2).to(mt.dtype))
                if want_align:
                    zvs.append(zv)
            if self.clip_attn is None:
                out.append(x)                                   # XM_ONLY
                continue
            # step 3: own feature attends over [f_n; refined chunks]
            Zk = Z if self.posemb is None else Z + self.posemb.unsqueeze(0)
            # NOSELF: keys/values = the K chunks only
            kv = Zk if self.noself else torch.cat([f, Zk], dim=1)
            r, _ = self.clip_attn(f, kv, kv, need_weights=False)
            # masked residual (padded locations keep f_n); exactly 0 at init
            r = r.reshape(B, T, D) * mask.transpose(1, 2).to(r.dtype)
            v = x.transpose(1, 2) + r                           # (B, T, D)
            out.append(v.transpose(1, 2))
        align = {'zv': zvs, 'et': et} if (want_align and self.xm) else None
        return out, (match if self.xm else None), align


class VideoSubtreeAttnNested(nn.Module):
    """TWO-LEVEL ("nested") video-side subtree attention (TIFAD_VSA=1
    TIFAD_VSA_NESTED=1; envs TIFAD_VSA_K = K action chunks, TIFAD_VSA_KP =
    Kp leaf chunks per action chunk, TIFAD_VSA_HEADS). Port of
    actionformer6/libs/modeling/vsa.py VideoSubtreeAttnNested, adapted to
    this repo's v1 call / return convention: forward(feats, offsets,
    fpn_masks, phase_embs=None, phase_head=None, want_align=False) ->
    (F x (B, D, T_l), None, None); the v2/v3 attributes the meta-arch
    reads (xm, xm_only, softphase, min_lvl) are fixed False / 0.

    The same ASA block as VideoSubtreeAttn, with a second level of
    sub-actions BELOW the action chunks (mirror of the text side's
    TIFAD_SUB_ENSEMBLE). For every location n of every pyramid level, with
    the clamped integer candidate span [s, e] from `_span` (detached
    regression offsets, level grid units):

      1. leaf pooling   : the span is split into K*Kp ordered equal chunks
         and mean-pooled (integral features, padded steps excluded)
         -> Zleaf (B, T, K*Kp, D). Leaf a*Kp + p is the p-th sub-action of
         action chunk a (leaves grouped contiguously by action).
      2. action pooling : the SAME span split into K ordered equal chunks
         -> Zact (B, T, K, D), computed directly from the integral
         features (exact mean over the whole action chunk, not a mean of
         its Kp leaf means, which differ when leaf widths are unequal).
      3. phase -> action (chain_attn_phase, clip_attn_phase): for every
         (location, action chunk a) the Kp leaves are refined by banded
         neighbour self-attention (|i-j| <= 1, Kp x Kp, residual); then
         Zact[a] (query) attends over [Zact[a]; refined leaves] (1+Kp
         keys, residual). Batched as (B*T*K, Kp, D): leaves of different
         actions never attend to each other.
      4. action -> node (chain_attn_act, clip_attn_act): banded neighbour
         self-attention among the K refined action embeddings (residual);
         then f_n attends over [f_n; refined actions] (1+K keys), masked
         residual exactly as in VideoSubtreeAttn (padded locations keep
         f_n).

    Init-equality invariant: all FOUR out_proj's are zero-init (weight AND
    bias), so every residual adds an exact 0.0 and the output == input
    bitwise at step 0. The band masks are non-persistent buffers, so the
    state_dict holds exactly the four MHA's (INIT_FROM: vsa.* missing
    only). Parameters are shared across pyramid levels; the four MHA's
    are distinct (no sharing between the phase and the action level).

    Short spans: a span with fewer than K*Kp (resp. K) units repeats units
    across neighbouring chunks (see `_pool_from_integral`), so some leaves
    (resp. actions) hold identical means; nothing is left empty and no
    index leaves the level. `_span` runs in float32 regardless of the
    model dtype (bf16 cannot represent integers > 256 exactly, which
    would push s_i / e_i to T and out of the integral table).
    """

    def __init__(self, dim, K=10, Kp=3, heads=4):
        super().__init__()
        assert K >= 1 and Kp >= 1, "[VSA-nested] K and Kp must be >= 1"
        self.K = int(K)
        self.Kp = int(Kp)
        # v1 call-site convention (meta_archs reads these)
        self.xm = False
        self.xm_only = False
        self.softphase = False
        self.min_lvl = 0
        # step 3: phase level (sub-actions of one action chunk)
        self.chain_attn_phase = nn.MultiheadAttention(dim, heads,
                                                      batch_first=True)
        self.clip_attn_phase = nn.MultiheadAttention(dim, heads,
                                                     batch_first=True)
        # step 4: action level (action chunks of one location)
        self.chain_attn_act = nn.MultiheadAttention(dim, heads,
                                                    batch_first=True)
        self.clip_attn_act = nn.MultiheadAttention(dim, heads,
                                                   batch_first=True)
        for m in (self.chain_attn_phase, self.clip_attn_phase,
                  self.chain_attn_act, self.clip_attn_act):
            nn.init.zeros_(m.out_proj.weight)
            nn.init.zeros_(m.out_proj.bias)
        # banded masks, True = MASKED (|i-j| > 1); non-persistent
        _p = torch.arange(self.Kp)
        self.register_buffer(
            'band_mask_phase', (_p[:, None] - _p[None, :]).abs() > 1,
            persistent=False)
        _a = torch.arange(self.K)
        self.register_buffer(
            'band_mask_act', (_a[:, None] - _a[None, :]).abs() > 1,
            persistent=False)

    # ------------------------------------------------------------------
    @staticmethod
    def _span(offsets, T):
        """offsets: (B, T, 2) DETACHED (d_s, d_e) in grid units. Returns
        the integer span [s_i, e_i] (inclusive) per location, clamped to
        the level, and its length L = e_i - s_i + 1 >= 1. Computed in
        float32 unconditionally (see class docstring)."""
        offsets = offsets.float()
        n = torch.arange(T, device=offsets.device, dtype=offsets.dtype)
        s = (n.unsqueeze(0) - offsets[..., 0]).clamp(0, T - 1)  # (B, T)
        e = (n.unsqueeze(0) + offsets[..., 1]).clamp(0, T - 1)  # (B, T)
        s_i = torch.floor(s).long()
        e_i = torch.ceil(e).long()
        e_i = torch.maximum(e_i, s_i)          # guard: >= 1 grid unit
        return s_i, e_i, (e_i - s_i + 1)

    @staticmethod
    def _integral(x, mask):
        """x: (B, D, T); mask: (B, 1, T) bool valid. Returns the masked
        integral features cs_t (B, T+1, D) and counts cm (B, T+1) with
        cs_t[:, t] = sum_{u < t} x[..., u] * m[u] (padded steps add 0)."""
        m = mask.to(x.dtype)                                    # (B, 1, T)
        cs = F.pad((x * m).cumsum(dim=-1), (1, 0))              # (B, D, T+1)
        cm = F.pad(m.cumsum(dim=-1), (1, 0))                    # (B, 1, T+1)
        return cs.transpose(1, 2), cm[:, 0, :]

    @staticmethod
    def _pool_from_integral(cs_t, cm, s_i, L, K):
        """Z (B, T, K, D): exact mean of x inside each of the K ordered
        equal chunks of the span at every location. Chunk c covers the
        integer units [s_i + floor(L*c/K), s_i + floor(L*(c+1)/K)),
        widened to >= 1 unit (short spans repeat units across chunks).
        hi <= s_i + L = e_i + 1 <= T, so the gather never leaves the
        (T+1)-long integral table, for ANY K."""
        B, T = s_i.shape
        k = torch.arange(K + 1, device=cs_t.device)             # (K+1,)
        bnd = s_i.unsqueeze(-1) + (L.unsqueeze(-1) * k) // K    # (B, T, K+1)
        # TIFAD_VSA_JITTER=<delta>: every
        # interior cut moves by U(-delta, delta) chunk lengths (seeded, so a run is
        # deterministic), cuts re-sorted so chunks stay ordered and contiguous, kept
        # inside the span. Eval-only probe of the trained equal-split model's tolerance
        # to boundary error. Byte-inert when unset.
        _jit = float(_os.environ.get('TIFAD_VSA_JITTER', '0') or 0)
        if _jit > 0 and K > 1:
            _g = torch.Generator(device=cs_t.device); _g.manual_seed(int(_os.environ.get('TIFAD_VSA_JITTER_SEED', '0') or 0))
            _eps = (torch.rand(B, T, K - 1, generator=_g, device=cs_t.device) * 2 - 1) * _jit * (L.unsqueeze(-1).to(torch.float32) / K)
            _b = bnd.to(torch.float32)
            # interior cuts only: clamp to [s_i, e_i] (so lo + 1 <= e_i + 1 <= T, the gather range), then sort
            # among themselves; the end cuts s_i and e_i + 1 never move (sorting the whole row would let a
            # pushed interior cut swap with an end cut and leave the span)
            _in = _b[..., 1:-1] + _eps
            _in = torch.minimum(torch.maximum(_in, s_i.unsqueeze(-1).to(_b.dtype)), (s_i + L - 1).unsqueeze(-1).to(_b.dtype))
            _b[..., 1:-1] = torch.sort(_in, dim=-1).values
            bnd = torch.round(_b).long()
        lo = bnd[..., :-1]
        hi = torch.maximum(bnd[..., 1:], lo + 1)   # >= 1 unit per chunk
        lo = lo.reshape(B, T * K)
        hi = hi.reshape(B, T * K)
        bidx = torch.arange(B, device=cs_t.device).unsqueeze(-1)  # (B, 1)
        num = cs_t[bidx, hi] - cs_t[bidx, lo]                   # (B, T*K, D)
        cnt = (cm[bidx, hi] - cm[bidx, lo])                     # (B, T*K)
        Z = num / cnt.clamp(min=1.0).unsqueeze(-1)
        return Z.reshape(B, T, K, cs_t.shape[-1])

    def _pool_levels(self, x, offsets, mask):
        """One integral pass, two poolings. Returns Zleaf (B, T, K*Kp, D)
        and Zact (B, T, K, D) (steps 1-2 of the class docstring)."""
        T = x.shape[-1]
        cs_t, cm = self._integral(x, mask)
        s_i, e_i, L = self._span(offsets, T)
        Zleaf = self._pool_from_integral(cs_t, cm, s_i, L, self.K * self.Kp)
        Zact = self._pool_from_integral(cs_t, cm, s_i, L, self.K)
        return Zleaf, Zact

    # ------------------------------------------------------------------
    @staticmethod
    def _mha_chunked(mha, q, k, v, attn_mask=None, chunk=16384):
        """nn.MultiheadAttention over (N, L, D) inputs in chunks of at most
        `chunk` sequences (CUDA grid-size limit on very large N)."""
        if q.shape[0] <= chunk:
            return mha(q, k, v, attn_mask=attn_mask, need_weights=False)[0]
        return torch.cat([
            mha(q[i:i + chunk], k[i:i + chunk], v[i:i + chunk],
                attn_mask=attn_mask, need_weights=False)[0]
            for i in range(0, q.shape[0], chunk)], dim=0)

    def forward(self, feats, offsets, fpn_masks, phase_embs=None,
                phase_head=None, want_align=False):
        """feats: F x (B, D, T_l); offsets: F x (B, T_l, 2) (detached by
        the caller, grid units of the level); fpn_masks: F x (B, 1, T_l)
        bool. The v2/v3 keyword arguments are accepted for the v1 call
        site and must be unset. Returns (F x (B, D, T_l) video-side
        subtree embeddings (== feats bitwise while all out_proj's are
        zero), None, None)."""
        assert phase_embs is None and phase_head is None and not want_align, \
            "[VSA-nested] XM / SOFTPHASE / ALIGN are not supported"
        out = []
        K, Kp = self.K, self.Kp
        for x, off, mask in zip(feats, offsets, fpn_masks):
            B, D, T = x.shape
            Zleaf, Zact = self._pool_levels(x, off, mask)
            # ---- step 3: phase -> action, one sequence per (n, a) ----
            P = Zleaf.reshape(B * T * K, Kp, D)                 # leaves of a
            # The phase level has B*T*K sequences (>1e5 on long THUMOS test
            # videos, T up to 13824 at inference): one MHA call over all of
            # them hits the CUDA grid limit ("invalid configuration
            # argument"), so the calls are chunked over the sequence dim
            # (per-sequence computation, results unchanged).
            P = P + self._mha_chunked(self.chain_attn_phase, P, P, P,
                                      attn_mask=self.band_mask_phase)
            q = Zact.reshape(B * T * K, 1, D)                   # action query
            kv = torch.cat([q, P], dim=1)                       # (B*T*K, 1+Kp, D)
            r = self._mha_chunked(self.clip_attn_phase, q, kv, kv)
            A = (q + r).reshape(B * T, K, D)                    # Zact_tilde
            # ---- step 4: action -> node, one sequence per location ----
            A = A + self._mha_chunked(self.chain_attn_act, A, A, A,
                                      attn_mask=self.band_mask_act)
            f = x.transpose(1, 2).reshape(B * T, 1, D)          # (B*T, 1, D)
            kv = torch.cat([f, A], dim=1)                       # (B*T, K+1, D)
            r = self._mha_chunked(self.clip_attn_act, f, kv, kv)
            # masked residual (padded locations keep f_n); exactly 0 at init
            r = r.reshape(B, T, D) * mask.transpose(1, 2).to(r.dtype)
            v = x.transpose(1, 2) + r                           # (B, T, D)
            out.append(v.transpose(1, 2))
        return out, None, None
