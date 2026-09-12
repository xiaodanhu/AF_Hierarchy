"""Cascaded phrase + activity heads for FineGym hierarchy supervision.

The action head stays unchanged from the flat baseline (PtTransformerClsHead +
PtTransformerRegHead at all FPN levels). Two new heads are added on top:

  PhraseHead: reads the trunk feature X at FPN levels 1, 2, 3 (mid + coarse).
              Outputs 14-way phrase logits + 2-d phrase boundary regression
              at each of those 3 levels. Also exposes its pre-classifier
              feature for the activity cascade.

  ActivityHead: reads the phrase head's pre-classifier feature at FPN level 3
                only. Outputs 4-way activity logits + 2-d activity boundary
                regression.

Both heads reuse the existing PtTransformerClsHead / PtTransformerRegHead
implementations from libs.modeling.meta_archs to keep the gradient + init
behaviour identical to the action head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhraseHead(nn.Module):
    """Cascaded phrase head: classification + boundary regression on FPN levels 1-3."""

    def __init__(
        self,
        in_dim,
        feat_dim,
        num_phrases=14,
        num_fpn_levels=3,           # mid + coarse - levels 1, 2, 3 of the 4-level FPN
        head_kernel_size=3,
        prior_prob=0.01,
        with_ln=True,
        head_num_layers=3,
    ):
        super().__init__()
        # Lazy import to avoid circular import (meta_archs imports this module).
        from .meta_archs import PtTransformerClsHead, PtTransformerRegHead
        self.cls_head = PtTransformerClsHead(
            in_dim, feat_dim, num_phrases,
            kernel_size=head_kernel_size,
            prior_prob=prior_prob,
            with_ln=with_ln,
            num_layers=head_num_layers,
            empty_cls=[],
        )
        self.reg_head = PtTransformerRegHead(
            in_dim, feat_dim, num_fpn_levels,
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=with_ln,
        )

    def forward(self, fpn_feats_subset, fpn_masks_subset, return_feats=True):
        """fpn_feats_subset / fpn_masks_subset: only the levels this head runs on.

        Returns:
            cls_logits: tuple of len(fpn_feats_subset) of (B, num_phrases, T_l) tensors
            reg_offsets: tuple of len(fpn_feats_subset) of (B, 2, T_l) tensors
            cls_pre_feats: tuple of pre-classifier feats (for activity cascade)
        """
        if return_feats:
            cls_logits, cls_pre_feats = self.cls_head(
                fpn_feats_subset, fpn_masks_subset, return_feats=True
            )
        else:
            cls_logits = self.cls_head(fpn_feats_subset, fpn_masks_subset)
            cls_pre_feats = None
        reg_offsets = self.reg_head(fpn_feats_subset, fpn_masks_subset)
        return cls_logits, reg_offsets, cls_pre_feats


class ActivityHead(nn.Module):
    """Cascaded activity head: cls + reg on FPN level 3 only.

    Reads the phrase head's pre-classifier feature at the coarsest FPN level.
    """

    def __init__(
        self,
        in_dim,
        feat_dim,
        num_activities=4,
        head_kernel_size=3,
        prior_prob=0.01,
        with_ln=True,
        head_num_layers=3,
    ):
        super().__init__()
        from .meta_archs import PtTransformerClsHead, PtTransformerRegHead
        self.cls_head = PtTransformerClsHead(
            in_dim, feat_dim, num_activities,
            kernel_size=head_kernel_size,
            prior_prob=prior_prob,
            with_ln=with_ln,
            num_layers=head_num_layers,
            empty_cls=[],
        )
        # Activity reg only runs on 1 FPN level.
        self.reg_head = PtTransformerRegHead(
            in_dim, feat_dim, 1,  # single FPN level
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=with_ln,
        )

    def forward(self, feat_level3, mask_level3):
        """feat_level3: (B, in_dim, T_3) - single-level input from phrase head's pre-cls feat
        mask_level3:    (B, 1, T_3) bool"""
        # PtTransformerClsHead/RegHead expect tuples of FPN levels.
        cls_logits = self.cls_head((feat_level3,), (mask_level3,))
        reg_offsets = self.reg_head((feat_level3,), (mask_level3,))
        return cls_logits[0], reg_offsets[0]


class StructuralConsistencyLoss(nn.Module):
    """Soft containment penalty: action_pred should lie inside phrase_pred."""

    def __init__(self):
        super().__init__()

    def forward(self, action_offsets, phrase_offsets, positive_mask):
        """Compute the soft containment penalty.

        Args:
            action_offsets: (N, 2) - (left_offset, right_offset) of action prediction
                            at positive tokens. Convention: token at point t predicts
                            [t - left_offset, t + right_offset] as its action segment.
            phrase_offsets: (N, 2) - same convention for phrase prediction at the
                            same tokens.
            positive_mask:  (N,) bool - only count positions that are positive for
                            BOTH action and phrase (otherwise either is meaningless).

        Returns:
            scalar loss (mean over positive_mask=True positions, 0 if none).
        """
        if action_offsets.numel() == 0 or phrase_offsets.numel() == 0:
            return action_offsets.sum() * 0.0  # zero with grad

        if positive_mask is None or positive_mask.sum() == 0:
            return action_offsets.sum() * 0.0

        a = action_offsets[positive_mask]
        p = phrase_offsets[positive_mask]
        # action: [t - a[..,0], t + a[..,1]],  phrase: [t - p[..,0], t + p[..,1]]
        # Want: phrase_start <= action_start AND action_end <= phrase_end
        # phrase_start - action_start = (t - p[..,0]) - (t - a[..,0]) = a[..,0] - p[..,0]
        # action_end   - phrase_end   = (t + a[..,1]) - (t + p[..,1]) = a[..,1] - p[..,1]
        left_pen  = F.relu(a[:, 0] - p[:, 0])
        right_pen = F.relu(a[:, 1] - p[:, 1])
        return (left_pen + right_pen).mean()
