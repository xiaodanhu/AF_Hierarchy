# ---------------------------------------------------------------------------
# DT: "Decompose and Transfer: CoT-Prompting Enhanced Alignment for
# Open-Vocabulary Temporal Action Detection" (Zhu et al., CVPR 2026),
# re-implemented on top of Ti-FAD.
#
# The paper's PDA framework has three modules:
#   * CSD (CoT-Prompting Semantic Decomposition): an LLM (GPT-4o) decomposes
#     each action label into phase descriptions {start, middle, end, global}.
#     Descriptions are authored OFFLINE and supplied here as a json file
#     (TIFAD_DT_PHASES). Each description d_c^p is wrapped into the paper's
#     standardized prompt "a video of people's motion that [Description]",
#     encoded by the frozen CLIP text encoder f_text, then mapped by a
#     TRAINED per-phase projection head Phi_txt^p (Eq. 1):
#         t_c^p = Phi_txt^p(f_text(d_c^p)) in R^D
#   * TIF (Text-infused Foreground Filtering, Eq. 2-3): per phase p,
#         S_fg^p = Softmax_T( max_C( F_v F_t^{pT} ) ) in R^T
#     binarized at its temporal mean, and used as a hard gate on F_v to
#     produce phase-specific visual streams F_v^p.
#   * APA (Adaptive Phase-wise Alignment, Eq. 4-7): per phase, cross-attention
#     infuses phase text into F_v^p (Eq. 4), per-phase classification logits
#     C_cls^p = Fbar_v^p F_t^{pT} (Eq. 5), adaptively aggregated with sigmoid
#     weights w_p = Sigmoid(W_p(F_v^p)) (Eq. 6-7). Localization uses the
#     concatenation of the four phase streams projected back to D by a
#     multi-layer MLP (Eq. 9), which is then fed to Ti-FAD's existing
#     foreground-aware / regression (/ centerness) heads and losses
#     (L = L_cls + L_fg + L_loc, Eq. 10 -- Ti-FAD's loss plumbing is reused).
#
# DT-FIX (init-equality repair, see per-site comments): because this fork
# reuses Ti-FAD's SIGMOID focal loss (the paper's Eq. 8 is CrossEntropy),
# the APA logits are calibrated exactly like the stock label-level matmul
# (normalized text, / logit_scale.exp(), + text-derived prior bias), the TIF
# gate is straight-through (hard forward / soft backward), Eq. 9 enters as a
# zero-init RESIDUAL on the stock features, and V / W_p are zero-init -- so
# at step 0 the DT model matches stock Ti-FAD (logits == prior bias, fused
# features == backbone features) and the DT terms fade in through training.
#
# Everything is gated by env TIFAD_DT=1 + TIFAD_DT_PHASES=<json>; when unset
# none of this file is even imported by meta_archs, so stock Ti-FAD behavior
# is byte-identical.
#
# Training recipe from the paper (operator creates configs; noted here only):
#   12 epochs on THUMOS14 / 7 epochs on ActivityNet v1.3, Adam optimizer,
#   two-stream I3D features, single A100.
# ---------------------------------------------------------------------------

import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# canonical phase order; 'global' is the holistic action description
DT_PHASES = ('start', 'middle', 'end', 'global')

# the paper's standardized prompt: 'a video of people's motion that [Description]'
DT_PROMPT_TEMPLATE = "a video of people's motion that {}"


class DTPhaseTextEncoder(nn.Module):
    """CSD text side (Sec. 3.2, Eq. 1).

    Loads the phase-description json {class_name: {phase: sentence}} and holds
    the four TRAINED per-phase projection heads Phi_txt^p (Linear D->D). The
    frozen CLIP text encoder itself lives in Ti-FAD's TextFeatures module and
    is passed in at call time (NOT stored here) so its parameters are not
    re-registered under this module.
    """

    def __init__(self, phases_file, emb_dim):
        super().__init__()
        with open(phases_file, 'r') as f:
            self.phase_desc = json.load(f)
        # tolerate common key aliases in the authored json
        self._aliases = {
            'start': ('start',),
            'middle': ('middle', 'mid'),
            'end': ('end',),
            'global': ('global', 'glob'),
        }
        # per-phase trained projection heads Phi_txt^p (Eq. 1)
        self.proj = nn.ModuleDict({p: nn.Linear(emb_dim, emb_dim) for p in DT_PHASES})

    def _desc(self, cls_name, phase):
        entry = self.phase_desc.get(cls_name)
        if entry is None:
            raise KeyError(
                f"[DT] class '{cls_name}' missing from TIFAD_DT_PHASES json")
        for k in self._aliases[phase]:
            if k in entry:
                return entry[k]
        raise KeyError(
            f"[DT] class '{cls_name}' has no '{phase}' description "
            f"(keys: {sorted(entry.keys())})")

    def _clip_encode(self, text_features, prompts):
        """Encode prompts with the frozen CLIP text encoder of TextFeatures.

        Tokenizes with truncation (CoT sentences can exceed CLIP's 77-token
        context, which TextFeatures.extract_text_emb does not guard against).
        Falls back to extract_text_emb for stub encoders (e.g. smoke tests).
        """
        tok = getattr(text_features, 'tokenizer', None)
        model = getattr(text_features, 'txt_model', None)
        if tok is not None and model is not None and hasattr(model, 'get_text_features'):
            device = next(model.parameters()).device
            toks = tok(prompts, padding=True, truncation=True, max_length=77,
                       return_tensors='pt').to(device)
            return model.get_text_features(**toks)
        return text_features.extract_text_emb(prompts)

    def forward(self, text_features, cls_names):
        """Returns ({phase: (C, D)} projected phase texts t_c^p,
                    {phase: (C, D)} L2-normalized raw CLIP embeddings).

        Re-encoded every forward (like stock Ti-FAD) because CLIP's final
        text_projection is trainable even when the encoder body is frozen.
        """
        prompts = [
            DT_PROMPT_TEMPLATE.format(self._desc(c, p))
            for p in DT_PHASES for c in cls_names
        ]
        emb = self._clip_encode(text_features, prompts)          # (4*C, D)
        emb = emb.view(len(DT_PHASES), len(cls_names), -1)       # (4, C, D)
        # DT-FIX(A): L2-normalize the CLIP embeddings and project emb/2,
        # mirroring the stock label-level path exactly
        # (text/||text|| -> text_proj(text/2)). Raw CLIP text norms are O(10);
        # previously the unnormalized embeddings flowed into Eq. 4/5 and blew
        # the classification logits up to O(100+) (first-iter cls_loss 522 vs
        # stock 0.46), so training spent itself crushing magnitudes instead of
        # learning alignment. The paper's Eq. 1/5 are silent on normalization,
        # but its base detector [24] (Ti-FAD) normalizes -- "Following [24]".
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        proj = {p: self.proj[p](emb[i] / 2.0) for i, p in enumerate(DT_PHASES)}
        norm_emb = {p: emb[i] for i, p in enumerate(DT_PHASES)}
        return proj, norm_emb


def dt_foreground_filter(feat, phase_text, valid_mask):
    """TIF (Sec. 3.3, Eq. 2-3) for one phase.

    feat       : (B, T, D) visual features F_v (one pyramid level)
    phase_text : (C, D) projected phase-p text embeddings F_t^p
    valid_mask : (B, T) bool, True on valid (non-padded) snippets

    Returns (F_v^p (B, T, D), gate (B, T)).
    """
    # per-timestep max-over-classes similarity, softmax-normalized over time
    sim = feat @ phase_text.t()                                   # (B, T, C)
    score = sim.max(dim=-1).values                                # (B, T)
    score = score.masked_fill(~valid_mask, float('-inf'))
    score = score.softmax(dim=-1)                                 # Eq. 2, in [0,1]
    # binarize at the average over all (valid) temporal positions.
    # DT-AMBIG: Eq. 2's Softmax is over the temporal axis (the only axis of the
    # R^T output), so the temporal-mean threshold equals 1/T_valid; we compute
    # the mean over VALID positions only since Ti-FAD pads sequences.
    n_valid = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    thr = score.sum(dim=-1, keepdim=True) / n_valid               # (B, 1)
    hard = torch.logical_and(score >= thr, valid_mask).to(feat.dtype)  # \hat S_fg^p
    # DT-FIX(B): straight-through binarization. The forward pass keeps the
    # paper's exact hard binary gate (Eq. 3 explicitly binarizes at the
    # temporal-average threshold), but the backward pass routes gradients
    # through the smoothed certainty map soft = T_valid * Softmax(score)
    # (mean 1 over valid positions), so the phase text projections Phi_txt^p
    # receive gradient THROUGH the gate and TIF can learn which snippets to
    # keep. Previously the gate was a dead (non-differentiable) branch: with
    # random-init projections it zeroed ~half of all snippets -- including
    # foreground -- from step 0 and could only be corrected indirectly via
    # the APA path.
    soft = (score * n_valid.to(feat.dtype)) * valid_mask.to(feat.dtype)
    gate = hard + (soft - soft.detach())                          # fwd == hard
    return feat * gate.unsqueeze(-1), gate                        # Eq. 3


class DTAdaptivePhaseAlignment(nn.Module):
    """APA classification branch (Sec. 3.4, Eq. 4-7)."""

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        # DT-AMBIG: Eq. 4 writes a single cross-attention with projections
        # Q/K/V applied per phase; we read it as ONE shared single-head
        # attention (shared Q/K/V weights) applied to each phase stream --
        # the simplest reading consistent with "we first leverage
        # cross-attention mechanism" (singular).
        self.q = nn.Linear(emb_dim, emb_dim)
        self.k = nn.Linear(emb_dim, emb_dim)
        self.v = nn.Linear(emb_dim, emb_dim)
        # DT-FIX(init-equality): zero-init the V projection so Fbar_v^p = 0 at
        # step 0 and hence C_cls^p = 0 (Eq. 5): the aggregated DT logits start
        # EXACTLY at the calibrated text bias (sigmoid == text_prior_prob),
        # matching the PtTransformerClsHead prior_prob convention that Ti-FAD's
        # sigmoid focal loss expects. V gets nonzero gradient from step 0
        # (fbar = attn @ V(ft), dL/dV != 0), then Q/K/weight_net wake up --
        # the standard zero-init-residual schedule (a la GPT-2/ControlNet).
        nn.init.zeros_(self.v.weight)
        nn.init.zeros_(self.v.bias)
        # DT-AMBIG: Eq. 6's W_p is subscripted by phase -> one weighting
        # network PER phase; we use Linear(D, 1), i.e. w_p is a per-timestep
        # scalar in (0,1) broadcast over classes when weighting C_cls^p.
        self.weight_net = nn.ModuleDict(
            {p: nn.Linear(emb_dim, 1) for p in DT_PHASES})
        # DT-FIX(D): zero-init every W_p so w_p = sigmoid(0) = 0.5 uniformly
        # at step 0 -- Eq. 7 starts as unbiased (scaled) mean aggregation and
        # the adaptive weighting fades in, instead of a random-init phase
        # dominating the early aggregated logits.
        for p in DT_PHASES:
            nn.init.zeros_(self.weight_net[p].weight)
            nn.init.zeros_(self.weight_net[p].bias)

    def forward(self, phase_feats, phase_texts, logit_scale):
        """
        phase_feats : {phase: (B, T, D)} TIF-filtered streams F_v^p
        phase_texts : {phase: (C, D)} projected phase text F_t^p
        logit_scale : scalar Parameter, Ti-FAD's learned log temperature
        Returns C_cls (B, T, C) -- Eq. 7 adaptive aggregation (bias is added
        by the caller so the prior enters ONCE, after the phase sum).
        """
        logits = None
        scale = 1.0 / math.sqrt(self.emb_dim)
        for p in DT_PHASES:
            fv = phase_feats[p]                                   # (B, T, D)
            ft = phase_texts[p]                                   # (C, D)
            # Eq. 4: text-aware visual representation (no residual is stated)
            attn = torch.softmax(
                (self.q(fv) @ self.k(ft).t()) * scale, dim=-1)    # (B, T, C)
            fbar = attn @ self.v(ft)                              # (B, T, D)
            # Eq. 5: similarity with the phase text of ALL categories.
            # DT-FIX(A): divide by Ti-FAD's learned temperature exactly like
            # the stock label-level matmul. The paper writes Eq. 5 as a plain
            # matmul but trains with CrossEntropy (Eq. 8, shift-invariant);
            # this fork reuses Ti-FAD's SIGMOID focal loss, which is scale-
            # and prior-sensitive, so the DT logits must carry the same
            # calibration the stock path gives it (/(logit_scale.exp()),
            # + text-derived bias added by the caller).
            c_p = (fbar @ ft.t()) / logit_scale.exp()             # (B, T, C)
            # Eq. 6-7: adaptive phase weighting from the (pre-attention) F_v^p
            w_p = torch.sigmoid(self.weight_net[p](fv))           # (B, T, 1)
            logits = w_p * c_p if logits is None else logits + w_p * c_p
        return logits


class DTLocalizationFusion(nn.Module):
    """APA localization branch (Sec. 3.4, Eq. 9): L([F^start;F^mid;F^end;F^glob]).

    DT-AMBIG: 'L is implemented by multi-layer MLP' without depth/width; we
    use the smallest multi-layer reading: Linear(4D->D) -> ReLU -> Linear(D->D).
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(len(DT_PHASES) * emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )
        # DT-FIX(C): zero-init the output layer; the caller adds the result as
        # a RESIDUAL on top of the stock features (fused = F_v + MLP([...])),
        # so at step 0 the fg / regression / centerness heads see features
        # byte-identical to stock Ti-FAD and the phase-fusion stream fades in.
        # Previously a random-init MLP of hard-gated streams fully REPLACED
        # the backbone features, wrecking localization from the start. Eq. 9's
        # "projection L to match the original feature space" is compatible
        # with a residual parameterization (which reproduces the original
        # space exactly at init).
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, phase_feats):
        x = torch.cat([phase_feats[p] for p in DT_PHASES], dim=-1)  # (B, T, 4D)
        return self.mlp(x)                                          # (B, T, D) residual delta


class DTPhaseDecomposeAlign(nn.Module):
    """Top-level DT module wired into TiFAD.forward.

    Takes the Ti-FAD backbone's feature pyramid and produces
      * per-level classification logits C_cls (replacing the label-level
        text matmul, calibrated with the stock logit_scale / text bias), and
      * per-level fused features \\hat F_v = F_v + MLP([...]) (a zero-init
        residual refinement of the inputs of the foreground-aware /
        regression / centerness heads, Eq. 9),
    so Ti-FAD's existing focal cls loss, foreground-aware loss and DIoU
    localization loss are reused unchanged (Eq. 10).

    DT-AMBIG: the paper describes a single temporal-transformer output
    F_v in R^{TxD} and never mentions Ti-FAD's feature pyramid; since Ti-FAD
    classifies and regresses at EVERY pyramid level with level-shared heads,
    we apply TIF+APA independently at each level with the SAME (shared)
    DT parameters -- the minimal-change faithful reading.

    DT-AMBIG: the Ti-FAD backbone's mutual text<->video cross-attention is
    left untouched and still consumes the stock label-level embeddings
    ("Following [24], we first extract ... processed through the temporal
    transformer"); CSD phase texts enter only through TIF and APA.
    """

    def __init__(self, emb_dim, phases_file):
        super().__init__()
        self.text_enc = DTPhaseTextEncoder(phases_file, emb_dim)
        self.apa = DTAdaptivePhaseAlignment(emb_dim)
        self.fusion = DTLocalizationFusion(emb_dim)

    def forward(self, fpn_feats, fpn_masks, text_features, cls_names,
                logit_scale, bias_lang, bias0):
        """
        fpn_feats : list of (B, D, T_l) backbone pyramid features
        fpn_masks : list of (B, 1, T_l) bool validity masks
        text_features : Ti-FAD TextFeatures module (frozen CLIP text encoder)
        cls_names : list of C class names (train or test split)
        logit_scale / bias_lang / bias0 : Ti-FAD's stock calibration params
            (owned by the parent TiFAD module -- passed in, not registered
            here, so they exist exactly once in the state dict)

        Returns (out_cls_logits: list of (B, T_l, C),
                 fused_feats:    list of (B, D, T_l)).
        """
        phase_texts, norm_embs = self.text_enc(text_features, cls_names)

        # DT-FIX(A): per-class prior bias computed from the phase texts the
        # same way stock computes it from the label text
        # (normalized_text @ bias_lang + bias0); averaged over phases so the
        # prior enters ONCE after Eq. 7's phase sum. bias_lang is zero-init,
        # so at step 0 every class starts at bias0 = -log((1-p)/p) with
        # p = text_prior_prob -- the calibrated starting point Ti-FAD's
        # sigmoid focal loss expects.
        mean_emb = torch.stack(
            [norm_embs[p] for p in DT_PHASES], dim=0).mean(dim=0)  # (C, D)
        cls_bias = mean_emb @ bias_lang + bias0                    # (C,)

        out_cls_logits, fused_feats = [], []
        for feat, mask in zip(fpn_feats, fpn_masks):
            x = feat.transpose(1, 2)                              # (B, T, D)
            m = mask.squeeze(1).bool()                            # (B, T)
            # TIF: phase-specific foreground-filtered streams (Eq. 2-3)
            phase_feats = {
                p: dt_foreground_filter(x, phase_texts[p], m)[0]
                for p in DT_PHASES
            }
            # APA: adaptive phase-wise classification (Eq. 4-7) + calibration
            logits = self.apa(phase_feats, phase_texts, logit_scale)
            logits = logits + cls_bias.view(1, 1, -1)             # (B, T, C)
            # DT-FIX: no mask-multiply on the logits -- stock leaves logits
            # unmasked too (losses/inference gather by fpn_masks); zeroing
            # them would erase the calibrated bias at padded positions.
            # localization stream (Eq. 9) as a zero-init RESIDUAL on the
            # stock features (DT-FIX(C)); only the delta is masked so padded
            # positions stay byte-identical to stock.
            fused = x + self.fusion(phase_feats) * m.unsqueeze(-1).to(x.dtype)
            out_cls_logits.append(logits)
            fused_feats.append(fused.transpose(1, 2))             # (B, D, T)
        return out_cls_logits, fused_feats
