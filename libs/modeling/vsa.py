"""Video-side action--sub-action (subtree) attention, port of the Ti-FAD fork's
VideoSubtreeAttn VERSION 1 ONLY (tifad/libs/modeling/vsa.py; no v2/v3 flags).

Mirror of the TEXT-side subtree composition (text_pathway.py: phase-level
banded neighbour attention among the ordered phase sentences, then the
action query attending over [action; phases]), applied PER CANDIDATE
SEGMENT on the video side. For every location n of every pyramid level:

  1. chunk pooling : the regression head's DETACHED offsets (d_s, d_e), in
     the level's own grid units, give the candidate span [n - d_s, n + d_e];
     it is clamped to the level, split into K ordered equal temporal chunks
     (video-side phases) and the classification-trunk features are
     mean-pooled inside each chunk via integral (cumulative-sum) features.
     Padded timesteps are excluded (masked cumsum + count).
  2. banded neighbour self-attention among the K chunk embeddings
     (|i-j| <= 1 attend, else masked), residual, out_proj ZERO-INIT.
  3. action-clip cross-attention: the location's own feature f_n queries
     [f_n; refined chunks] (K+1 keys), residual, out_proj ZERO-INIT.
  4. the result (B, D, T_l) REPLACES the cls-trunk feature the scaled-cosine
     text classifier reads (train AND inference, every level, every
     location); the regression head keeps the raw features.

Init-equality invariant: BOTH out_proj's are zero-init (weight AND bias), so
the output == input bitwise at step 0 (the residual adds an exact 0.0).
Parameters are shared across pyramid levels. Config-gated in meta_archs
(model.use_video_subtree_attn); the module is never constructed when off.

VideoSubtreeAttnNested (below) is the two-level variant (action chunks
with Kp sub-action leaves each); selected by model.video_subtree_nested and
assigned to the same `video_subtree_attn` attribute.
"""
import os as _os
import torch
import torch.nn.functional as F
from torch import nn
try:                                        # torch >= 2.3
    from torch.nn.attention import sdpa_kernel, SDPBackend
except ImportError:                         # local adaptation: torch 2.0 here
    from contextlib import contextmanager

    class SDPBackend:                       # only MATH is ever requested below
        MATH = 'math'

    @contextmanager
    def sdpa_kernel(_backend):
        """Same effect as the 2.3+ API: force the math SDPA kernel."""
        with torch.backends.cuda.sdp_kernel(
            enable_math=True, enable_flash=False, enable_mem_efficient=False
        ):
            yield

# 2026-09-22 (local): the fused scaled-dot-product-attention kernels crash with "CUDA error: an
# illegal memory access" in the BACKWARD of these tiny-sequence (K+1 = 4 / Kp = 10 keys) bf16
# attentions with an additive band mask when the pyramid has several levels (reproduced
# standalone at B=40 with T in {288,144,72,36}; fp32 and single-level runs pass). The math
# backend is exact for these shapes and costs nothing at <= 11 keys, so force it inside the
# video-side attention only.


def _order_shuffle(Z, dim, env='A6_VSA_SHUFFLE'):
    """eval-time permutation of the ordered chunk tokens along `dim`
    (A6_VSA_SHUFFLE=reverse | perm; perm = one fixed seeded permutation per length). Byte-inert when unset."""
    mode = _os.environ.get(env, '')
    if not mode:
        return Z
    n = Z.shape[dim]
    if n < 2:
        return Z
    if mode == 'reverse':
        idx = torch.arange(n - 1, -1, -1, device=Z.device)
    else:
        g = torch.Generator().manual_seed(int(_os.environ.get(env + '_SEED', '0') or 0) + n)
        idx = torch.randperm(n, generator=g).to(Z.device)
    return Z.index_select(dim, idx)


class VideoSubtreeAttn(nn.Module):

    def __init__(self, dim, K=10, heads=4, learn_bounds=False, tau_init=0.1, no_chain=False, mean_chunks=False):
        super().__init__()
        assert K >= 1, "[VSA] K must be >= 1"
        self.K = int(K)
        # step 2: banded neighbour self-attention among the K chunks
        self.chain_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        # step 3: action-clip cross-attention (query = own feature)
        self.clip_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        for m in (self.chain_attn, self.clip_attn):
            nn.init.zeros_(m.out_proj.weight)
            nn.init.zeros_(m.out_proj.bias)
        # banded mask, True = MASKED (|i-j| > 1); PyTorch turns a bool mask
        # into the additive 0 / -inf mask internally. Non-persistent so the
        # state_dict holds only the two MHA's (init_from: only vsa.* missing).
        _i = torch.arange(self.K)
        self.register_buffer(
            'band_mask', (_i[:, None] - _i[None, :]).abs() > 1,
            persistent=False)
        # [local] learnable, non-equal chunk boundaries (see _pool_learned). Built AFTER the two MHA's so
        # their init RNG draws are unchanged; all heads zero-init => uniform widths (= the equal split) and
        # the global temperature at step 0.
        self.no_chain = bool(no_chain); self.mean_chunks = bool(mean_chunks)   # mechanism test (see nested class)
        self.learn_bounds = bool(learn_bounds)
        if self.learn_bounds:
            self.width_loc = nn.Linear(dim, self.K)    # widths from the location feature f_n
            self.width_cnt = nn.Linear(dim, 1)         # + per-chunk content term from the equal-split chunk means
            self.tau_loc = nn.Linear(dim, 1)           # per-location log-temperature offset (boundary uncertainty)
            for m in (self.width_loc, self.width_cnt, self.tau_loc):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
            # global temperature, relative to the mean chunk width L/K
            self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())

    # ------------------------------------------------------------------
    def _span(self, offsets, T):
        """offsets: (B, T, 2) DETACHED (d_s, d_e) in grid units.
        Returns integer span [s_i, e_i] (inclusive) per location, clamped
        to the level, and its length L = e_i - s_i + 1 >= 1.

        NB: this arithmetic must be done in a precision that can represent
        every integer in [0, T] exactly, independent of whichever dtype the
        surrounding model happens to run in (fp32/fp16/bf16). bf16 has only
        an 8-bit mantissa, so integers above 256 already round to the
        nearest even value (e.g. T-1=287 rounds to 288); doing the clamp in
        offsets.dtype under bf16 training silently produced s_i/e_i == T
        (one past the valid [0, T-1] range), which is exactly what fed the
        out-of-bounds gather in _pool_chunks. float32 is exact for T well
        beyond any FPN length used here, so upcast unconditionally.
        """
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
        """x: (B, D, T) level features; mask: (B, 1, T) bool valid.
        Returns the masked integral features cs_t (B, T+1, D) and counts
        cm (B, T+1) with cs_t[:, t] = sum_{u < t} x[..., u] * m[u], so the
        mean over any integer window [lo, hi) is one subtraction per end
        (padded timesteps contribute 0 to both sum and count)."""
        m = mask.to(x.dtype)                                    # (B, 1, T)
        # integral features / counts: cs[..., t] = sum_{u < t}
        cs = F.pad((x * m).cumsum(dim=-1), (1, 0))              # (B, D, T+1)
        cm = F.pad(m.cumsum(dim=-1), (1, 0))                    # (B, 1, T+1)
        return cs.transpose(1, 2), cm[:, 0, :]                  # (B,T+1,D),(B,T+1)

    @staticmethod
    def _pool_from_integral(cs_t, cm, s_i, L, K):
        """cs_t (B, T+1, D) / cm (B, T+1) from `_integral`; s_i, L (B, T)
        from `_span`; K = number of ordered equal chunks. Returns
        Z (B, T, K, D): the exact mean of x inside each of the K chunks of
        the candidate span at every location. Chunk c covers the integer
        units [s_i + floor(L*c/K), s_i + floor(L*(c+1)/K)), widened to >= 1
        unit, so spans shorter than K units REPEAT units across neighbouring
        chunks (those chunks then hold identical means) rather than leaving
        a chunk empty. Upper bound hi <= s_i + L = e_i + 1 <= T, so the
        gather never leaves the (T+1)-long integral table, for ANY K."""
        B, T = s_i.shape
        D = cs_t.shape[-1]
        k = torch.arange(K + 1, device=cs_t.device)             # (K+1,)
        # chunk boundaries (integer units): s_i + floor(L * k / K)
        bnd = s_i.unsqueeze(-1) + (L.unsqueeze(-1) * k) // K    # (B, T, K+1)
        # A6_VSA_JITTER=<delta>: every interior
        # cut moves by U(-delta, delta) chunk lengths (seeded), cuts re-sorted and kept in
        # [s_i, s_i + L - 1] (last cut s_i + L), so chunks stay ordered, contiguous and the
        # gather in range. Applies to every equal-split pooling (action and leaf level of
        # the nested module). Eval-only probe; byte-inert when unset.
        _jit = float(_os.environ.get('A6_VSA_JITTER', '0') or 0)
        if _jit > 0 and K > 1:
            _g = torch.Generator(device=cs_t.device); _g.manual_seed(int(_os.environ.get('A6_VSA_JITTER_SEED', '0') or 0))
            _eps = (torch.rand(B, T, K - 1, generator=_g, device=cs_t.device) * 2 - 1) * _jit * (L.unsqueeze(-1).to(torch.float32) / K)
            _b = bnd.to(torch.float32)
            # interior cuts only: clamp to [s_i, s_i + L - 1], then sort among themselves; the end cuts never move
            _in = _b[..., 1:-1] + _eps
            _in = torch.minimum(torch.maximum(_in, s_i.unsqueeze(-1).to(_b.dtype)), (s_i + L - 1).unsqueeze(-1).to(_b.dtype))
            _b[..., 1:-1] = torch.sort(_in, dim=-1).values
            bnd = torch.round(_b).long()
        lo = bnd[..., :-1]
        hi = torch.maximum(bnd[..., 1:], lo + 1)   # >= 1 unit per chunk
        # (segments shorter than K units repeat units across chunks);
        # hi <= s_i + L = e_i + 1 <= T, so the gather stays in range
        lo = lo.reshape(B, T * K)
        hi = hi.reshape(B, T * K)
        bidx = torch.arange(B, device=cs_t.device).unsqueeze(-1)  # (B, 1)
        num = cs_t[bidx, hi] - cs_t[bidx, lo]                   # (B, T*K, D)
        cnt = (cm[bidx, hi] - cm[bidx, lo])                     # (B, T*K)
        Z = num / cnt.clamp(min=1.0).unsqueeze(-1)
        return Z.reshape(B, T, K, D)

    def _pool_chunks(self, x, offsets, mask, K=None):
        """x: (B, D, T) level features; offsets: (B, T, 2) DETACHED
        (d_s, d_e) in grid units; mask: (B, 1, T) bool valid; K: number of
        chunks (default self.K). Returns Z (B, T, K, D): mean of x inside
        each of the K ordered chunks of the candidate segment at every
        location. (Same ops in the same order as the original single-K
        implementation: `_integral` + `_span` + `_pool_from_integral`.)"""
        K = self.K if K is None else int(K)
        T = x.shape[-1]
        cs_t, cm = self._integral(x, mask)                      # (B,T+1,D),(B,T+1)
        s_i, e_i, L = self._span(offsets, T)                    # (B, T)
        return self._pool_from_integral(cs_t, cm, s_i, L, K)

    def _pool_learned(self, x, offsets, mask):
        """[local] Learnable non-equal chunk boundaries with refinement.

        x (B, D, T) level features; offsets (B, T, 2) DETACHED (d_s, d_e) grid units; mask (B, 1, T).
        1. continuous candidate span [s, e] = [n - d_s, n + d_e + 1] (grid unit u covers [u, u+1)),
           e - s >= 1, clamped to [0, T];
        2. coarse pass: the equal-split chunk means Zeq (exact integral pooling, as in `_pool_chunks`);
        3. refinement: K width logits = width_loc(f_n) + width_cnt(Zeq_k) -> softmax widths w (sum 1),
           boundaries b_0 = s < b_1 < ... < b_K = e with b_k = s + L * cumsum(w)_k (monotone by
           construction); zero-init heads => w = 1/K, i.e. the equal split;
        4. differentiable pooling: membership of unit u in chunk k is the sigmoid window
           sig((u+.5 - b_{k-1}) / tau) - sig((u+.5 - b_k) / tau), with tau = exp(log_tau + tau_loc(f_n)) * L / K
           (a per-location, span-relative temperature: small = sharp/confident boundaries, large =
           soft/uncertain ones); padded units excluded; Z_k = weighted mean. Gradients reach the
           boundaries (hence the width heads) through the window slopes.
        Returns Z (B, T, K, D)."""
        B, D, T = x.shape
        K = self.K
        Zeq = self._pool_chunks(x, offsets, mask)                           # (B, T, K, D)
        off = offsets.float()
        n = torch.arange(T, device=x.device, dtype=torch.float32)
        # same convention as `_span` (units floor(n - d_s) .. ceil(n + d_e) inclusive), made continuous
        s = (n.unsqueeze(0) - off[..., 0]).clamp(0.0, float(T))
        e = (n.unsqueeze(0) + off[..., 1] + 1.0).clamp(0.0, float(T))
        e = torch.maximum(e, (s + 1.0).clamp(max=float(T)))
        s = torch.minimum(s, e - 1.0).clamp(min=0.0)
        L = e - s                                                           # (B, T) >= 1
        f = x.transpose(1, 2)                                               # (B, T, D)
        logits = self.width_loc(f).float() + self.width_cnt(Zeq).squeeze(-1).float()   # (B, T, K)
        w = logits.softmax(dim=-1)
        cum = torch.cat([w.new_zeros(B, T, 1), w.cumsum(dim=-1)], dim=-1)  # (B, T, K+1), 0 .. 1
        bnd = s.unsqueeze(-1) + L.unsqueeze(-1) * cum                       # (B, T, K+1)
        tau = (self.log_tau.float() + self.tau_loc(f).float().squeeze(-1)).exp() * L / K
        tau = tau.clamp(min=1e-2)                                           # (B, T)
        u = torch.arange(T, device=x.device, dtype=torch.float32) + 0.5
        sig = torch.sigmoid((u.view(1, 1, 1, T) - bnd.unsqueeze(-1)) / tau.view(B, T, 1, 1))   # (B, T, K+1, T)
        memb = (sig[:, :, :-1] - sig[:, :, 1:]).clamp(min=0.0)              # (B, T, K, T)
        memb = memb * mask.to(memb.dtype).view(B, 1, 1, T)
        num = torch.einsum('btku,bdu->btkd', memb.to(x.dtype), x)           # (B, T, K, D)
        den = memb.sum(-1).clamp(min=1e-4).unsqueeze(-1)
        return (num.float() / den).to(x.dtype)

    # ------------------------------------------------------------------
    def forward(self, feats, offsets, fpn_masks):
        """feats: F x (B, D, T_l) cls-trunk features; offsets: F x (B, T_l, 2)
        (detached by the caller, grid units of the level); fpn_masks:
        F x (B, 1, T_l) bool. Returns F x (B, D, T_l) video-side subtree
        embeddings (== feats bitwise while both out_proj's are zero)."""
        out = []
        with sdpa_kernel(SDPBackend.MATH):   # fresh context per call (contextmanager objects are single-use)
          return self._forward_impl(feats, offsets, fpn_masks, out)

    def _forward_impl(self, feats, offsets, fpn_masks, out):
        for x, off, mask in zip(feats, offsets, fpn_masks):
            B, D, T = x.shape
            K = self.K
            if self.learn_bounds:
                Z = self._pool_learned(x, off, mask)            # (B, T, K, D), non-equal chunks
            else:
                Z = self._pool_chunks(x, off, mask)             # (B, T, K, D)
            Z = Z.reshape(B * T, K, D)
            Z = _order_shuffle(Z, 1)                            # order control (eval-only env), inert otherwise
            if self.mean_chunks:
                Z = Z.mean(dim=1, keepdim=True)                 # one token per location (no slots)
            # step 2: banded neighbour self-attention, zero-init residual
            if not self.no_chain and not self.mean_chunks:
                _u, _ = self.chain_attn(Z, Z, Z, attn_mask=self.band_mask,
                                        need_weights=False)
                Z = Z + _u
            # step 3: own feature attends over [f_n; refined chunks]
            f = x.transpose(1, 2).reshape(B * T, 1, D)          # (B*T, 1, D)
            kv = torch.cat([f, Z], dim=1)                       # (B*T, K+1, D)
            r, _ = self.clip_attn(f, kv, kv, need_weights=False)
            # masked residual (padded locations keep f_n); exactly 0 at init
            r = r.reshape(B, T, D) * mask.transpose(1, 2).to(r.dtype)
            v = x.transpose(1, 2) + r                           # (B, T, D)
            out.append(v.transpose(1, 2))
        return out


class VideoSubtreeAttnNested(nn.Module):
    """Two-level ("nested") video-side subtree attention: the same ASA block
    as VideoSubtreeAttn, but with a second level of sub-actions BELOW the
    action chunks, mirroring the full text-side tree
    (activity -> action -> phase). Same forward signature / return as
    VideoSubtreeAttn (drop-in behind model.video_subtree_nested).

    For every location n of every pyramid level, with the clamped integer
    candidate span [s, e] from `_span` (detached regression offsets, level
    grid units):

      1. leaf pooling   : the span is split into K*Kp ordered equal chunks
         and mean-pooled (integral features, padded steps excluded)
         -> Zleaf (B, T, K*Kp, D). Leaf a*Kp + p is the p-th sub-action of
         action chunk a, i.e. the leaves are grouped contiguously by action.
      2. action pooling : the SAME span split into K ordered equal chunks
         -> Zact (B, T, K, D). Computed directly from the integral features,
         so Zact[a] is the exact mean over the whole action chunk (not a
         mean of its Kp leaf means, which would differ when leaf widths
         are unequal / repeated for short spans).
      3. phase -> action (chain_attn_phase, clip_attn_phase): for every
         (location, action chunk a), the Kp leaves of a are refined by
         banded neighbour self-attention (|i-j| <= 1 attend, Kp x Kp,
         residual); then the action embedding Zact[a] (query) attends over
         [Zact[a]; refined leaves] (1+Kp keys, residual) -> Zact_tilde
         (B, T, K, D). Batched as (B*T*K, Kp, D): every action chunk is an
         independent sequence, so leaves of different actions never attend
         to each other.
      4. action -> node (chain_attn_act, clip_attn_act): banded neighbour
         self-attention among the K refined action embeddings (residual);
         then the location's own feature f_n attends over
         [f_n; refined actions] (1+K keys), masked residual exactly as in
         VideoSubtreeAttn (padded locations keep f_n).

    Init-equality invariant: all FOUR out_proj's are zero-init (weight AND
    bias), so every residual adds an exact 0.0 and the output == input
    bitwise at step 0. The band masks are non-persistent buffers, so the
    state_dict holds exactly the four MHA's. Parameters are shared across
    pyramid levels (as in VideoSubtreeAttn); all four MHA's are distinct
    from each other (no sharing between the phase and the action level).

    Short spans: a span with fewer than K*Kp (resp. K) units repeats units
    across neighbouring chunks (see `_pool_from_integral`), so some leaves
    (resp. actions) hold identical means; nothing is left empty and no
    index leaves the level.
    """

    def __init__(self, dim, K=3, Kp=10, heads=4, learn_bounds=False, tau_init=0.1, no_chain=False, mean_chunks=False):
        super().__init__()
        assert K >= 1 and Kp >= 1, "[VSA-nested] K and Kp must be >= 1"
        self.K = int(K)
        self.Kp = int(Kp)
        # step 3: phase level (sub-actions of one action chunk)
        self.chain_attn_phase = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.clip_attn_phase = nn.MultiheadAttention(dim, heads, batch_first=True)
        # step 4: action level (action chunks of one location)
        self.chain_attn_act = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.clip_attn_act = nn.MultiheadAttention(dim, heads, batch_first=True)
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
        # learnable NON-EQUAL action-chunk boundaries
        # (model.video_subtree_learn_bounds). The K action widths are predicted
        # from the location feature plus the equal-split chunk content; the Kp
        # leaves of each action split its (soft) span equally. Built after the
        # MHAs so their init is unchanged; zero-init => equal split at step 0.
        # no_chain: skip both banded neighbour attentions
        # (chunks go straight to their parent query); mean_chunks: collapse the chunk tokens of
        # each level to their mean (no slots at all). Used with the boundary-jitter probe.
        self.no_chain = bool(no_chain); self.mean_chunks = bool(mean_chunks)
        self.learn_bounds = bool(learn_bounds)
        if self.learn_bounds:
            self.width_loc = nn.Linear(dim, self.K)
            self.width_cnt = nn.Linear(dim, 1)
            self.tau_loc = nn.Linear(dim, 1)
            for _m in (self.width_loc, self.width_cnt, self.tau_loc):
                nn.init.zeros_(_m.weight)
                nn.init.zeros_(_m.bias)
            self.log_tau = nn.Parameter(torch.tensor(float(tau_init)).log())

    # span / pooling numerics are shared with the flat module
    _span = VideoSubtreeAttn._span
    _integral = staticmethod(VideoSubtreeAttn._integral)
    _pool_from_integral = staticmethod(VideoSubtreeAttn._pool_from_integral)

    def _pool_levels(self, x, offsets, mask):
        """One integral pass, two poolings. Returns Zleaf (B, T, K*Kp, D)
        and Zact (B, T, K, D) (see steps 1-2 of the class docstring)."""
        T = x.shape[-1]
        cs_t, cm = self._integral(x, mask)
        s_i, e_i, L = self._span(offsets, T)
        Zleaf = self._pool_from_integral(cs_t, cm, s_i, L, self.K * self.Kp)
        Zact = self._pool_from_integral(cs_t, cm, s_i, L, self.K)
        return Zleaf, Zact

    def _pool_levels_learned(self, x, offsets, mask):
        """Learned action boundaries, equal leaf split inside each action.
        x (B, D, T); offsets (B, T, 2) DETACHED grid units; mask (B, 1, T).
        Continuous span [s, e] (as the flat `_pool_learned`); K width logits =
        width_loc(f_n) + width_cnt(Zact_eq) -> softmax w; action boundaries
        b_a = s + L cumsum(w)_a; leaf boundaries b_{a,p} = b_a + w_a L p/Kp;
        soft sigmoid membership with tau = exp(log_tau + tau_loc(f_n)) * L/K for
        actions and * L/(K Kp) for leaves; padded units excluded.
        Returns Zleaf (B, T, K*Kp, D), Zact (B, T, K, D)."""
        B, D, T = x.shape
        K, Kp = self.K, self.Kp
        _, Zact_eq = self._pool_levels(x, offsets, mask)                   # coarse pass (exact)
        off = offsets.float()
        n = torch.arange(T, device=x.device, dtype=torch.float32)
        s = (n.unsqueeze(0) - off[..., 0]).clamp(0.0, float(T))
        e = (n.unsqueeze(0) + off[..., 1] + 1.0).clamp(0.0, float(T))
        e = torch.maximum(e, (s + 1.0).clamp(max=float(T)))
        s = torch.minimum(s, e - 1.0).clamp(min=0.0)
        L = e - s                                                           # (B, T)
        f = x.transpose(1, 2)                                               # (B, T, D)
        logits = self.width_loc(f).float() + self.width_cnt(Zact_eq).squeeze(-1).float()   # (B, T, K)
        w = logits.softmax(dim=-1)
        cumA = torch.cat([w.new_zeros(B, T, 1), w.cumsum(dim=-1)], dim=-1)  # (B, T, K+1)
        frac = torch.arange(Kp, device=x.device, dtype=torch.float32) / Kp  # (Kp,)
        cumL = (cumA[:, :, :-1].unsqueeze(-1) + w.unsqueeze(-1) * frac).reshape(B, T, K * Kp)
        cumL = torch.cat([cumL, cumA[:, :, -1:]], dim=-1)                   # (B, T, K*Kp+1)
        tau0 = (self.log_tau.float() + self.tau_loc(f).float().squeeze(-1)).exp()   # (B, T)
        u = torch.arange(T, device=x.device, dtype=torch.float32) + 0.5

        def pool(cum, tau):
            bnd = s.unsqueeze(-1) + L.unsqueeze(-1) * cum                   # (B, T, M+1)
            sig = torch.sigmoid((u.view(1, 1, 1, T) - bnd.unsqueeze(-1)) / tau.clamp(min=1e-2).view(B, T, 1, 1))
            memb = (sig[:, :, :-1] - sig[:, :, 1:]).clamp(min=0.0)          # (B, T, M, T)
            memb = memb * mask.to(memb.dtype).view(B, 1, 1, T)
            num = torch.einsum('btmu,bdu->btmd', memb.to(x.dtype), x)
            den = memb.sum(-1).clamp(min=1e-4).unsqueeze(-1)
            return (num.float() / den).to(x.dtype)

        Zact = pool(cumA, tau0 * L / K)
        Zleaf = pool(cumL, tau0 * L / (K * Kp))
        return Zleaf, Zact

    def forward(self, feats, offsets, fpn_masks):
        """feats: F x (B, D, T_l) cls-trunk features; offsets: F x (B, T_l, 2)
        (detached by the caller, grid units of the level); fpn_masks:
        F x (B, 1, T_l) bool. Returns F x (B, D, T_l) video-side subtree
        embeddings (== feats bitwise while all out_proj's are zero)."""
        out = []
        with sdpa_kernel(SDPBackend.MATH):   # fresh context per call (contextmanager objects are single-use)
          return self._forward_impl(feats, offsets, fpn_masks, out)

    def _forward_impl(self, feats, offsets, fpn_masks, out):
        K, Kp = self.K, self.Kp
        for x, off, mask in zip(feats, offsets, fpn_masks):
            B, D, T = x.shape
            if self.learn_bounds:
                Zleaf, Zact = self._pool_levels_learned(x, off, mask)
            else:
                Zleaf, Zact = self._pool_levels(x, off, mask)
            # ---- step 3: phase -> action, one sequence per (n, a) ----
            P = Zleaf.reshape(B * T * K, Kp, D)                 # leaves of a
            P = _order_shuffle(P, 1)                            # order control: leaves within an action
            if self.mean_chunks:
                P = P.mean(dim=1, keepdim=True)                 # one token per action
            if not self.no_chain and not self.mean_chunks:
                _u, _ = self.chain_attn_phase(P, P, P, attn_mask=self.band_mask_phase,
                                              need_weights=False)
                P = P + _u
            q = Zact.reshape(B * T * K, 1, D)                   # action query
            kv = torch.cat([q, P], dim=1)                       # (B*T*K, 1+Kp, D)
            r, _ = self.clip_attn_phase(q, kv, kv, need_weights=False)
            A = (q + r).reshape(B * T, K, D)                    # Zact_tilde
            A = _order_shuffle(A, 1)                            # order control: action chunks within the span
            # ---- step 4: action -> node, one sequence per location ----
            if self.mean_chunks:
                A = A.mean(dim=1, keepdim=True)                 # one token per location
            if not self.no_chain and not self.mean_chunks:
                _u, _ = self.chain_attn_act(A, A, A, attn_mask=self.band_mask_act,
                                            need_weights=False)
                A = A + _u
            f = x.transpose(1, 2).reshape(B * T, 1, D)          # (B*T, 1, D)
            kv = torch.cat([f, A], dim=1)                       # (B*T, K+1, D) or (B*T, 2, D)
            r, _ = self.clip_attn_act(f, kv, kv, need_weights=False)
            # masked residual (padded locations keep f_n); exactly 0 at init
            r = r.reshape(B, T, D) * mask.transpose(1, 2).to(r.dtype)
            v = x.transpose(1, 2) + r                           # (B, T, D)
            out.append(v.transpose(1, 2))
        return out
