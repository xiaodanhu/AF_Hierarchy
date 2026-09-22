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
import torch
import torch.nn.functional as F
from torch import nn


class VideoSubtreeAttn(nn.Module):

    def __init__(self, dim, K=10, heads=4):
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

    # ------------------------------------------------------------------
    def forward(self, feats, offsets, fpn_masks):
        """feats: F x (B, D, T_l) cls-trunk features; offsets: F x (B, T_l, 2)
        (detached by the caller, grid units of the level); fpn_masks:
        F x (B, 1, T_l) bool. Returns F x (B, D, T_l) video-side subtree
        embeddings (== feats bitwise while both out_proj's are zero)."""
        out = []
        for x, off, mask in zip(feats, offsets, fpn_masks):
            B, D, T = x.shape
            K = self.K
            Z = self._pool_chunks(x, off, mask)                 # (B, T, K, D)
            Z = Z.reshape(B * T, K, D)
            # step 2: banded neighbour self-attention, zero-init residual
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

    def __init__(self, dim, K=3, Kp=10, heads=4):
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

    def forward(self, feats, offsets, fpn_masks):
        """feats: F x (B, D, T_l) cls-trunk features; offsets: F x (B, T_l, 2)
        (detached by the caller, grid units of the level); fpn_masks:
        F x (B, 1, T_l) bool. Returns F x (B, D, T_l) video-side subtree
        embeddings (== feats bitwise while all out_proj's are zero)."""
        out = []
        K, Kp = self.K, self.Kp
        for x, off, mask in zip(feats, offsets, fpn_masks):
            B, D, T = x.shape
            Zleaf, Zact = self._pool_levels(x, off, mask)
            # ---- step 3: phase -> action, one sequence per (n, a) ----
            P = Zleaf.reshape(B * T * K, Kp, D)                 # leaves of a
            _u, _ = self.chain_attn_phase(P, P, P, attn_mask=self.band_mask_phase,
                                          need_weights=False)
            P = P + _u
            q = Zact.reshape(B * T * K, 1, D)                   # action query
            kv = torch.cat([q, P], dim=1)                       # (B*T*K, 1+Kp, D)
            r, _ = self.clip_attn_phase(q, kv, kv, need_weights=False)
            A = (q + r).reshape(B * T, K, D)                    # Zact_tilde
            # ---- step 4: action -> node, one sequence per location ----
            _u, _ = self.chain_attn_act(A, A, A, attn_mask=self.band_mask_act,
                                        need_weights=False)
            A = A + _u
            f = x.transpose(1, 2).reshape(B * T, 1, D)          # (B*T, 1, D)
            kv = torch.cat([f, A], dim=1)                       # (B*T, K+1, D)
            r, _ = self.clip_attn_act(f, kv, kv, need_weights=False)
            # masked residual (padded locations keep f_n); exactly 0 at init
            r = r.reshape(B, T, D) * mask.transpose(1, 2).to(r.dtype)
            v = x.transpose(1, 2) + r                           # (B, T, D)
            out.append(v.transpose(1, 2))
        return out
