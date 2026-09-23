import json
import math

import torch
from torch import nn
import os
from torch.nn import functional as F

from .models import register_meta_arch, make_backbone, make_neck, make_generator
from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss, weighted_multilabel_loss

from ..utils import batched_nms
from collections import defaultdict

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
        empty_cls = [],
        use_hierarchical_embedding=False,
        action_to_phrase=None,
        action_to_activity=None,
        num_phrases=None,
        num_activities=None,
        with_final_cls=True,     # t3 text pathway: False -> trunk only, the
                                 # scaled-cosine text classifier replaces the
                                 # final conv (no dead params under deepspeed)
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
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        # Final classifier: standard MaskedConv1D OR factored HierarchicalMaskedConv1D.
        if not with_final_cls:
            self.cls_head = None
        elif use_hierarchical_embedding:
            assert action_to_phrase is not None and action_to_activity is not None, \
                "use_hierarchical_embedding=True requires action_to_phrase and action_to_activity"
            assert num_phrases is not None and num_activities is not None, \
                "use_hierarchical_embedding=True requires num_phrases and num_activities"
            from .hierarchical_embedding import HierarchicalMaskedConv1D
            self.cls_head = HierarchicalMaskedConv1D(
                feat_dim=feat_dim,
                num_classes=num_classes,
                action_to_phrase=action_to_phrase,
                action_to_activity=action_to_activity,
                num_phrases=num_phrases,
                num_activities=num_activities,
                kernel_size=kernel_size,
                prior_prob=prior_prob,
                empty_cls=empty_cls,
            )
        else:
            self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )
            # use prior in model initialization to improve stability
            if prior_prob > 0:
                bias_value = -(math.log((1 - prior_prob) / prior_prob))
                torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)
            # empty categories
            if len(empty_cls) > 0:
                bias_value = -(math.log((1 - 1e-6) / 1e-6))
                for idx in empty_cls:
                    torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks, return_feats=False):
        assert len(fpn_feats) == len(fpn_masks)
        assert (self.cls_head is not None) or return_feats, \
            "with_final_cls=False requires return_feats=True"

        # apply the classifier for each pyramid level
        out_logits = tuple()
        out_feats = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            if return_feats:
                out_feats += (cur_out, )  # trunk features before final classifier
            if self.cls_head is not None:
                cur_logits, _ = self.cls_head(cur_out, cur_mask)
                out_logits += (cur_logits, )

        # fpn_masks remains the same
        if return_feats:
            return out_logits, out_feats
        return out_logits


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
        with_ln=False
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
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)), )

        # fpn_masks remains the same
        return out_offsets



class PtTransformerRotationHead(nn.Module):
    """Phase 2D v9: video-grounded rotation counting head (Variant B).

    A 1-layer local-attention transformer over FPN level-0 tokens to extract
    temporal context, then a Linear projection to a per-token scalar = predicted
    rotation count ∈ R. Trained via smooth-L1 against the GT class's
    rotation_count value (continuous, in {0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}).

    Rationale: counting rotations in video requires temporal context. A single
    token (1/16 of a sliding window) cannot determine "how many rotations" by
    itself. Local attention with window=15 covers the typical action span.

    At inference, the scalar is converted to 10 binary rotation-derivative
    attribute probabilities (rotation_X, has_any_rotation, has_extra_revolution,
    rotation_two_or_more) via Gaussian RBF + logistic - see compute_rotation_binary_probs.
    """
    def __init__(
        self,
        input_dim,
        n_head=8,
        mha_win_size=15,
        num_layers=1,
        attn_pdrop=0.0,
        proj_pdrop=0.1,
        path_pdrop=0.1,
        with_ln=True,
    ):
        super().__init__()
        from .blocks import TransformerBlock
        # Stack `num_layers` local-attention transformer blocks (v11: default 2 was found
        # to add capacity for rotation extraction).
        self.tx_blocks = nn.ModuleList([
            TransformerBlock(
                n_embd=input_dim,
                n_head=n_head,
                mha_win_size=mha_win_size,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                use_rel_pe=False,
            ) for _ in range(num_layers)
        ])
        self.norm = LayerNorm(input_dim) if with_ln else nn.Identity()
        # Linear over channel dim → scalar per token
        self.proj = nn.Conv1d(input_dim, 1, kernel_size=1, bias=True)
        # Bias to a sensible prior (mean rotation_count ~ 0.6 across 99 classes)
        torch.nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, fpn_feat_lvl0, fpn_mask_lvl0, return_feats=False):
        """
        Args:
            fpn_feat_lvl0: (B, C, T) features at the finest FPN level
            fpn_mask_lvl0: (B, 1, T) bool mask
        Returns:
            (B, T) per-token rotation count predictions
            (optionally also the (B, C, T) trunk features before the final proj)
        """
        x, mask = fpn_feat_lvl0, fpn_mask_lvl0
        for blk in self.tx_blocks:
            x, mask = blk(x, mask)
        x = self.norm(x)
        out = self.proj(x)                              # (B, 1, T)
        if return_feats:
            return out.squeeze(1), x
        return out.squeeze(1)                           # (B, T)


class PtTransformerCategoricalSpecialistHead(nn.Module):
    """Phase 2D v10: generic categorical specialist head (e.g., body_shape 5-way,
    direction 3-way, multiplicity 2-way, named_element 6-way).

    Architecture mirrors the v9 rotation head: a 1-layer local-attention transformer
    over FPN level-0 tokens extracts temporal context, then a Conv1d projection
    produces per-token logits over `num_classes` categories.

    Training: per-token softmax CE on positive tokens where the class has a valid
    target (target_per_class[c] ∈ [0, num_classes), valid_per_class[c] == True).
    Inference: gather log-softmax at the held-out class's target index, then center
    within siblings (same matrix as v6/v9) and add to phrase logit.
    """
    def __init__(
        self,
        input_dim,
        num_classes,
        n_head=8,
        mha_win_size=17,
        num_layers=1,
        attn_pdrop=0.0,
        proj_pdrop=0.1,
        path_pdrop=0.1,
        with_ln=True,
    ):
        super().__init__()
        from .blocks import TransformerBlock
        self.num_classes = num_classes
        # Stack `num_layers` local-attention transformer blocks.
        self.tx_blocks = nn.ModuleList([
            TransformerBlock(
                n_embd=input_dim,
                n_head=n_head,
                mha_win_size=mha_win_size,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                use_rel_pe=False,
            ) for _ in range(num_layers)
        ])
        self.norm = LayerNorm(input_dim) if with_ln else nn.Identity()
        self.proj = nn.Conv1d(input_dim, num_classes, kernel_size=1, bias=True)
        torch.nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, fpn_feat_lvl0, fpn_mask_lvl0, return_feats=False):
        """Returns (B, num_classes, T) per-token category logits.
        With return_feats=True, also returns the (B, C, T) trunk features."""
        x, mask = fpn_feat_lvl0, fpn_mask_lvl0
        for blk in self.tx_blocks:
            x, mask = blk(x, mask)
        x = self.norm(x)
        if return_feats:
            return self.proj(x), x
        return self.proj(x)


@register_meta_arch("LocPointTransformer")
class PtTransformer(nn.Module):
    """
        Transformer based model for single stage action localization
    """
    def __init__(
        self,
        backbone_type,         # a string defines which backbone we use
        fpn_type,              # a string defines which fpn we use
        backbone_arch,         # a tuple defines #layers in embed / stem / branch
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
        num_classes,           # number of action classes
        train_cfg,             # other cfg for training
        test_cfg,              # other cfg for testing
        active_learning_method='uniform',  # adaptive or uniform
        use_hierarchy=False,
        num_phrases=14,
        num_activities=4,
        verbalizer_path="",
        use_hierarchical_embedding=False,                # NEW: factored 3-tier classifier
        verbalizer_path_for_embedding="",                # NEW: required when use_hierarchical_embedding=True
        # Cascaded hierarchy
        use_cascaded_hierarchy=False,
        cascade_phrase_levels=(1, 2, 3),
        cascade_activity_levels=(3,),
        cascade_gradient_stop=False,
        use_aux_attribute_head=False,            # Phase 2D ZSL: auxiliary multi-label attribute head
        aux_attribute_table_path="",             # path to finegym_attribute_table_v2.json
        use_rotation_regression_head=False,      # Phase 2D v9: video-grounded rotation counting
        rotation_head_n_head=8,
        rotation_head_mha_win_size=15,
        # ---- Phase 2D v10: multi-specialist heads ----
        use_body_shape_head=False,               # 5-way: stretched/tucked/piked/ring/straddle
        use_direction_head=False,                # 3-way: forward/backward/sideward
        use_multiplicity_head=False,             # 2-way: single/double salto
        use_named_element_head=False,            # 6-way: switch_leap/johnson/giant/clear/stalder/pike_sole_circle
        specialist_head_n_head=8,
        specialist_head_mha_win_size=17,
        specialist_head_num_layers=1,            # Phase 2D v11: 2 layers found to help capacity
        # ---- Phase 2D v11: additional specialists + split rotation ----
        use_motion_category_head=False,          # 8-way: salto/turn/leap/jump/walkover/handstand/dismount/transition
        use_leg_configuration_head=False,        # 3-way: split_leg / both_leg_landing / one_leg_landing
        use_inverted_phase_head=False,           # 2-way: head-below-hips at some point (LMA shape orientation)
        use_hands_on_apparatus_head=False,       # 2-way: hands gripping apparatus
        split_rotation_head=False,               # if True: use aerial_twist + floor_turn instead of single rotation_head
        rotation_head_num_layers=1,              # match v9 default; v11 typically uses 2
        use_sibling_softmax=False,               # if True: replace sibling-centering with sibling-softmax
        phrase_head_num_layers_override=0,       # if > 0: override head_num_layers specifically for cascaded phrase head
        # ———— Phase 2D v16: alternative scoring formulations (inference-only) ————
        use_raw_log_prob=False,                  # if True: skip sibling-softmax, use raw per-token log-prob
        # ———— Phase 2D v17: attribute-axis contrastive learning ————
        # Adds a per-specialist margin loss. For each positive token of class c, the
        # specialist's prediction must rank c's target HIGHER than any within-phrase
        # sibling's target by `attr_axis_contrast_margin`. Sharpens specialist
        # confidence on within-phrase discriminative axes — directly attacks the
        # specialist-accuracy gap between v11 (24% with raw log-prob) and oracle (37%).
        use_attr_axis_contrast=False,
        attr_axis_contrast_margin=1.0,
        lambda_attr_axis_contrast=0.5,
        # ———— FineGym Table-3 (t3): shared frozen-CLIP-TEXT pathway ————
        # Replaces the closed-set flat classification head by scaled cosine
        # of the cls-head trunk features vs per-class CLIP text embeddings.
        # See libs/modeling/text_pathway.py for the three-system layout.
        use_text_cls_head=False,
        text_prompt_template='a video of action {}',   # system 1 prompt
        text_ensemble_path='',        # champion HP: 11-sentence ensemble json
        text_phase_attention=False,   # champion PA: zero-init class->phase attn
        text_phase_neighbor_attention=False,  # APA phase-level (stage-2) attention
        text_sentence_dropout=0.0,    # champion HP: h14 dropout (train only)
        text_stem_degrade=0.0,        # champion HP: h18 parent-stem degrade
        text_cache_dir='./cache_text_emb',
        # nested two-level text composition for PHRASE-level classes (the
        # verbalizer is then configs/finegym_phrase_verbalizer.json and the
        # ensemble configs/finegym_phrase_ensemble.json; members come from
        # the 99-action verbalizer/ensemble). Byte-inert when False.
        text_nested=False,
        text_member_verbalizer_path='',   # configs/finegym_zsl_verbalizer.json
        text_member_ensemble_path='',     # configs/finegym_qwen_ensemble.json
        # system 2 (Ti-FAD re-implementation)
        use_text_video_cross_attn=False,   # TiCA per FPN level
        use_fg_head=False,                 # foreground weighting of cls scores
        # champion AS: 14-way phrase-level focal CE + 2-way Kendall UW
        use_activity_align_loss=False,
        # parent levels the AS loss supervises; one Kendall weight per level.
        # ('phrase',) = 3-level model; ('phrase', 'apparatus') = 4-level model
        activity_align_levels=('phrase',),
        # champion DP: eval-time duration prior
        use_duration_prior=False,
        duration_prior_path='',
        duration_prior_gamma=0.3,
        # champion stage 3: VIDEO BRANCH of the action--sub-action attention
        # (libs/modeling/vsa.py, Ti-FAD fork VideoSubtreeAttn v1). Per
        # location, the detached regression span is pooled into K ordered
        # chunks, banded neighbour attention among them, then the location
        # feature queries [f_n; chunks]; the result REPLACES the cls-trunk
        # feature the cosine classifier reads. Zero-init out_proj's => the
        # stage starts exactly at the stage-2 model. Byte-inert when False
        # (module never built; forward path untouched).
        use_video_subtree_attn=False,
        video_subtree_K=10,
        video_subtree_heads=4,
        # nested (two-level) variant, vsa.py VideoSubtreeAttnNested: each of
        # the K action chunks is further split into Kp ordered sub-action
        # leaves; leaves -> action (banded + clip attn), then actions ->
        # location as above. Same attribute / forward call; byte-inert
        # when False (the flat module is built as before).
        video_subtree_nested=False,
        video_subtree_Kp=10,
        # ———— Strict-ZSL ablation: if True, exclude held-out class segments from
        # specialist losses (attribute targets + rotation + contrastive). Demonstrates
        # methodology rigor by ensuring held-out class videos never contribute to
        # any training signal beyond phrase/activity/regression (which are coarser
        # than action class identity). Important for top-conference reviewer scrutiny.
        strict_zsl=False,
        use_grad_checkpoint=False,  # enable gradient checkpointing on CLIP backbone
        clip_forward_chunk=0,       # if >0, chunk CLIP per-frame forward (memory cap)
        **kwargs,  # absorb unused keys (e.g., HGDD-only flags present in shared config defaults)
    ):
        super().__init__()
         # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor**i for i in range(
            fpn_start_level, backbone_arch[-1]+1
        )]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        # #classes = num_classes + 1 (background) with last category as background
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
        self.num_classes = num_classes
        self.active_learning_method = active_learning_method

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
        self.train_cfg = train_cfg
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']
        self.reduction = train_cfg['reduction']

        # test time config
        self.test_cfg = test_cfg
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

        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        self.backbone_type = backbone_type
        # assert self.backbone_type in ['convTransformer', 'conv', 'ActionFormerWithViViT', 'ActionFormerWithViT']
        use_gradient_checkpoint = train_cfg.get('use_gradient_checkpoint', False)
        if backbone_type == 'convTransformer':
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
                    'use_gradient_checkpoint' : use_gradient_checkpoint,
                }
            )
        elif backbone_type == 'ActionFormerWithCLIP':
            self.backbone = make_backbone(
                'ActionFormerWithCLIP',
                **{
                    'input_dim' : input_dim,
                    'embd_dim' : embd_dim,
                    'n_head': n_head,
                    'embd_kernel_size': embd_kernel_size,
                    'max_seq_len': max_seq_len,
                    'backbone_arch' : backbone_arch,
                    'mha_win_size': self.mha_win_size,
                    'scale_factor' : scale_factor,
                    'embd_with_ln' : embd_with_ln,
                    'attn_pdrop' : 0.0,
                    'proj_pdrop' : self.train_dropout,
                    'path_pdrop' : self.train_droppath,
                    'use_abs_pe' : use_abs_pe,
                    'use_rel_pe' : use_rel_pe,
                    'pretrained' : True,
                    'use_grad_checkpoint' : use_grad_checkpoint,
                    'clip_forward_chunk' : clip_forward_chunk,
                }
            )
        else:
            self.backbone = make_backbone(
                'conv',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'n_embd_ks': embd_kernel_size,
                    'arch': backbone_arch,
                    'scale_factor': scale_factor,
                    'with_ln' : embd_with_ln
                }
            )
        if isinstance(embd_dim, (list, tuple)):
            embd_dim = sum(embd_dim)

        # fpn network: convs
        assert fpn_type in ['fpn', 'identity']
        self.neck = make_neck(
            fpn_type,
            **{
                'in_channels' : [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel' : fpn_dim,
                'scale_factor' : scale_factor,
                'start_level' : fpn_start_level,
                'with_ln' : fpn_with_ln
            }
        )

        # location generator: points
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len' : max_seq_len * max_buffer_len_factor,
                'fpn_strides' : self.fpn_strides,
                'regression_range' : self.reg_range
            }
        )

        # ---- Hierarchical embedding lookups (used by use_hierarchical_embedding AND
        # Phase 2D v4 parent-conditional ZSL inference) ----
        action_to_phrase_list = None
        action_to_activity_list = None
        num_phrases_for_embed = None
        num_activities_for_embed = None
        if use_hierarchical_embedding or (use_cascaded_hierarchy and verbalizer_path_for_embedding):
            from .hierarchy_utils import (
                load_verbalizer, get_action_to_phrase_map, get_action_to_activity_map
            )
            assert verbalizer_path_for_embedding, \
                "use_hierarchical_embedding=True (or cascaded ZSL) requires verbalizer_path_for_embedding"
            verb = load_verbalizer(verbalizer_path_for_embedding)
            a2p_dict = get_action_to_phrase_map(verb)
            a2a_dict = get_action_to_activity_map(verb)
            action_to_phrase_list = [a2p_dict[i] for i in range(num_classes)]
            action_to_activity_list = [a2a_dict[i] for i in range(num_classes)]
            num_phrases_for_embed = max(action_to_phrase_list) + 1
            num_activities_for_embed = max(action_to_activity_list) + 1

        # Phase 2D v4 ZSL: register action_to_phrase as a buffer for inference.
        # Used to look up the parent phrase of each held-out class.
        if action_to_phrase_list is not None:
            self.register_buffer(
                'action_to_phrase_buf',
                torch.tensor(action_to_phrase_list, dtype=torch.long),
            )
        else:
            self.action_to_phrase_buf = None

        # held_out_sibling_avg_matrix is registered later (in the held-out load block)
        # when both held-out ids and action_to_phrase_list are available.

        # classfication and regerssion heads
        self.use_text_cls_head = bool(use_text_cls_head)
        self.cls_head = PtTransformerClsHead(
            fpn_dim, head_dim, self.num_classes,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls'],
            use_hierarchical_embedding=use_hierarchical_embedding,
            action_to_phrase=action_to_phrase_list,
            action_to_activity=action_to_activity_list,
            num_phrases=num_phrases_for_embed,
            num_activities=num_activities_for_embed,
            # t3 text pathway replaces the final closed-set classifier
            with_final_cls=not self.use_text_cls_head,
        )
        self.reg_head = PtTransformerRegHead(
            fpn_dim, head_dim, len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln
        )

        # ———— FineGym Table-3: shared frozen-CLIP-TEXT pathway ————
        # Construction ORDER is part of the init-equality contract between
        # the three systems: shared modules (backbone/neck/heads and the
        # pathway's proj Linear) are built BEFORE any system-specific module
        # (PA attention inside the pathway, TiCA levels, fg head), so the
        # shared modules consume identical RNG draws in all three systems.
        self.text_pathway = None
        self.text_tv_attn = None
        self.fg_head = None
        self.as_uw_logvar = None
        self.duration_prior = None
        self.use_activity_align_loss = bool(use_activity_align_loss)
        if self.use_text_cls_head:
            from .text_pathway import (
                TextClsPathway, TextVideoCrossAttnLevel, DurationPrior)
            assert verbalizer_path_for_embedding, \
                "use_text_cls_head requires verbalizer_path_for_embedding"
            self.text_pathway = TextClsPathway(
                verbalizer_path=verbalizer_path_for_embedding,
                head_dim=head_dim,
                prior_prob=self.train_cls_prior_prob,
                ensemble_path=text_ensemble_path,
                phase_attention=text_phase_attention,
                phase_neighbor_attention=text_phase_neighbor_attention,
                sentence_dropout=text_sentence_dropout,
                stem_degrade=text_stem_degrade,
                prompt_template=text_prompt_template,
                cache_dir=text_cache_dir,
                nested=text_nested,
                member_verbalizer_path=text_member_verbalizer_path,
                member_ensemble_path=text_member_ensemble_path,
            )
            assert self.text_pathway.num_classes == num_classes
            if use_text_video_cross_attn:
                # one TiCA module per FPN level (the fork has one per branch
                # transformer block, i.e. per level)
                self.text_tv_attn = nn.ModuleList([
                    TextVideoCrossAttnLevel(
                        dim=head_dim, n_head=n_head,
                        max_len=max_seq_len // (self.scale_factor ** l),
                    ) for l in range(len(self.fpn_strides))
                ])
                print(f"[t3-tifad] TiCA: {len(self.fpn_strides)} per-level "
                      "text<-video cross-attention modules")
            if use_fg_head:
                # Ti-FAD foreground branch: same head template as the fork
                # (tifad meta_archs builds fg_head as a 1-way ClsHead)
                self.fg_head = PtTransformerClsHead(
                    fpn_dim, head_dim, 1,
                    kernel_size=head_kernel_size,
                    prior_prob=self.train_cls_prior_prob,
                    with_ln=head_with_ln,
                    num_layers=head_num_layers,
                )
                print("[t3-tifad] fg head active "
                      "(eval score = sigmoid(cls) * sigmoid(fg))")
            if self.use_activity_align_loss:
                # AS (FAMCE-UW): per-phrase member columns + 2-way Kendall
                # log-variances (init 0 => weights (1,1): plain sum at step 0)
                # Grouping choice (DOC): 14-way PHRASE level, not the 4-way
                # apparatus level. The 14 phrase families match the
                # granularity of THUMOS's 9-family FAMCE grouping where this
                # component was validated; every held-out class shares its
                # phrase with seen siblings (informative parent supervision),
                # whereas the 4-way apparatus split is nearly determined by
                # background context and carries little compositional signal.
                # 4-level extension: each entry of activity_align_levels is a
                # parent level; its logit = logsumexp over member action
                # columns, its target = amax over the same columns.
                _level_maps = {
                    'phrase': self.text_pathway.action_to_phrase,
                    'apparatus': self.text_pathway.action_to_activity,
                }
                self._as_levels = []
                for lvl in activity_align_levels:
                    a2x = _level_maps[lvl]
                    n_x = int(a2x.max().item()) + 1
                    self._as_levels.append((lvl, [
                        torch.nonzero(a2x == p, as_tuple=True)[0]
                        for p in range(n_x)
                    ]))
                self.as_uw_logvar = nn.Parameter(
                    torch.zeros(1 + len(self._as_levels)))
                print("[t3-AS] focal CE over parent levels "
                      + ", ".join(f"{l} ({len(m)} groups)" for l, m in self._as_levels)
                      + f" + {1 + len(self._as_levels)}-way Kendall UW (init s=0)")
            if use_duration_prior:
                assert duration_prior_path, \
                    "use_duration_prior requires duration_prior_path"
                self.duration_prior = DurationPrior(
                    duration_prior_path, gamma=duration_prior_gamma)

        # ———— Cascaded phrase + activity heads ————
        self.use_cascaded_hierarchy = use_cascaded_hierarchy
        self.cascade_phrase_levels = list(cascade_phrase_levels)
        self.cascade_activity_levels = list(cascade_activity_levels)
        self.cascade_gradient_stop = cascade_gradient_stop
        if self.use_cascaded_hierarchy:
            from .cascaded_hierarchy_heads import (
                PhraseHead, ActivityHead, StructuralConsistencyLoss,
            )
            assert len(self.cascade_activity_levels) <= 1, \
                "cascade_activity_levels must be 0 or 1 FPN level (use [] to disable activity head)"
            assert all(l < len(self.fpn_strides) for l in self.cascade_phrase_levels), \
                f"cascade_phrase_levels {self.cascade_phrase_levels} out of range for {len(self.fpn_strides)} FPN levels"
            assert all(l < len(self.fpn_strides) for l in self.cascade_activity_levels)
            # Make sure num_phrases / num_activities are tracked on self for the cascade path,
            # even when use_hierarchy=False.
            self.num_phrases = num_phrases
            self.num_activities = num_activities
            self.phrase_cascade_head = PhraseHead(
                in_dim=fpn_dim,
                feat_dim=head_dim,
                num_phrases=num_phrases,
                num_fpn_levels=len(self.cascade_phrase_levels),
                head_kernel_size=head_kernel_size,
                prior_prob=train_cfg.get('cls_prior_prob', 0.01),
                with_ln=head_with_ln,
                head_num_layers=head_num_layers,
            )
            if len(self.cascade_activity_levels) == 1:
                self.activity_cascade_head = ActivityHead(
                    in_dim=head_dim,        # consumes phrase head's pre-cls feat (head_dim, not fpn_dim)
                    feat_dim=head_dim,
                    num_activities=num_activities,
                    head_kernel_size=head_kernel_size,
                    prior_prob=train_cfg.get('cls_prior_prob', 0.01),
                    with_ln=head_with_ln,
                    head_num_layers=head_num_layers,
                )
            else:
                self.activity_cascade_head = None
            self.structural_consistency_loss = StructuralConsistencyLoss()

        # Store verbalizer path for contrastive loss
        self._verbalizer_path = verbalizer_path





        # ---- Phase 2D ZSL: load held-out class ids from attribute table ----
        # The held-out set is shared by v1/v2/v3/v4 ZSL variants (attribute, CLIP,
        # parent-conditional). It's read from the attribute table's `held_out_classes`
        # field whenever the path is provided, regardless of which ZSL head is active.
        self.aux_attr_held_out_ids = []
        if aux_attribute_table_path:
            with open(aux_attribute_table_path) as _f:
                _t = json.load(_f)
            self.aux_attr_held_out_ids = [int(aid[1:]) for aid in _t.get('held_out_classes', [])]
            if self.aux_attr_held_out_ids:
                print(f"[ZSL] held-out classes: {len(self.aux_attr_held_out_ids)} "
                      f"({self.aux_attr_held_out_ids[:5]}...)")
                # v6 hybrid: build the (H, H) sibling-averaging matrix.
                # Row h is uniform over held-out classes sharing h's parent phrase.
                # Used to center attribute LL across within-phrase siblings so
                # unique-phrase held-out classes get zero attribute contribution.
                if action_to_phrase_list is not None:
                    _held = self.aux_attr_held_out_ids
                    _H = len(_held)
                    _parents = [action_to_phrase_list[h] for h in _held]
                    _S = torch.zeros(_H, _H, dtype=torch.float32)
                    for _i in range(_H):
                        _sib_idx = [_j for _j in range(_H) if _parents[_j] == _parents[_i]]
                        _w = 1.0 / len(_sib_idx)
                        for _j in _sib_idx:
                            _S[_i, _j] = _w
                    self.register_buffer('held_out_sibling_avg_matrix', _S)
                    _ndup = sum(1 for _i in range(_H) if (_S[_i] > 0).sum().item() > 1)
                    print(f"[ZSL] sibling-avg matrix built: {_ndup}/{_H} held-out classes "
                          f"share their phrase with another held-out class")
                else:
                    self.held_out_sibling_avg_matrix = None



        # ---- Phase 2D v9: rotation counting head (Variant B local-attention transformer) ----
        self.use_rotation_regression_head = use_rotation_regression_head
        self.rotation_head = None
        # Default all rotation heads to None so forward()'s `is not None` checks are
        # safe when use_rotation_regression_head=False (e.g., FineDiving configs).
        self.rotation_head = None
        self.aerial_twist_head = None
        self.floor_turn_head = None
        self.split_rotation_head = False
        # rotation_count_per_class is registered as a buffer below when the head is enabled.
        if self.use_rotation_regression_head:
            assert aux_attribute_table_path, \
                "use_rotation_regression_head=True requires aux_attribute_table_path"
            with open(aux_attribute_table_path) as _f:
                _t = json.load(_f)
            # Build per-class rotation count buffer.
            _rcs = torch.zeros(num_classes, dtype=torch.float32)
            for aid, entry in _t['actions'].items():
                _rcs[int(aid[1:])] = float(entry.get('rotation_count', 0.0))
            self.register_buffer('rotation_count_per_class', _rcs)

            # Collect attribute indices to skip from BCE training (rotation derivatives).
            _aux_names = _t['attribute_names']

            # v11: optionally split into aerial_twist + floor_turn instead of single head.
            self.split_rotation_head = bool(split_rotation_head)
            if self.split_rotation_head:
                self.rotation_head = None
                self.aerial_twist_head = PtTransformerRotationHead(
                    input_dim=fpn_dim,
                    n_head=int(rotation_head_n_head),
                    mha_win_size=int(rotation_head_mha_win_size),
                    num_layers=int(rotation_head_num_layers),
                    with_ln=head_with_ln,
                )
                self.floor_turn_head = PtTransformerRotationHead(
                    input_dim=fpn_dim,
                    n_head=int(rotation_head_n_head),
                    mha_win_size=int(rotation_head_mha_win_size),
                    num_layers=int(rotation_head_num_layers),
                    with_ln=head_with_ln,
                )
                # Derive per-class rotation type from motion_* attributes.
                # 1 = aerial (salto/walkover/dismount): rotation_count = twist count
                # 2 = floor  (turn/leap/jump):          rotation_count = turn count
                # 0 = none   (handstand/transition/none of the above)
                _rot_type = torch.zeros(num_classes, dtype=torch.long)
                _aerial_set = {'motion_salto', 'motion_walkover', 'motion_dismount'}
                _floor_set  = {'motion_turn', 'motion_leap', 'motion_jump'}
                _aerial_idx = [_aux_names.index(n) for n in _aerial_set if n in _aux_names]
                _floor_idx  = [_aux_names.index(n) for n in _floor_set if n in _aux_names]
                for aid, entry in _t['actions'].items():
                    cid = int(aid[1:])
                    attrs = entry['attrs']
                    is_aerial = any(int(attrs[i]) == 1 for i in _aerial_idx)
                    is_floor  = any(int(attrs[i]) == 1 for i in _floor_idx)
                    if is_aerial and not is_floor:
                        _rot_type[cid] = 1
                    elif is_floor and not is_aerial:
                        _rot_type[cid] = 2
                    elif is_aerial and is_floor:
                        # Rare: takes both → treat as aerial (twist dominates)
                        _rot_type[cid] = 1
                    else:
                        _rot_type[cid] = 0
                self.register_buffer('rotation_type_per_class', _rot_type)
                n_aer = int((_rot_type == 1).sum())
                n_flr = int((_rot_type == 2).sum())
                n_none = int((_rot_type == 0).sum())
                print(f"[ZSL v11] split rotation: {n_aer} aerial-twist classes, "
                      f"{n_flr} floor-turn classes, {n_none} no-rotation classes "
                      f"(n_layers={rotation_head_num_layers}, win={rotation_head_mha_win_size})")
            else:
                raise NotImplementedError(
                    "single (non-split) rotation head not included in the ZSL "
                    "migration — set split_rotation_head: True in the config")

        # ———— Phase 2D v10: multi-specialist heads (body_shape, direction, multiplicity, named_element) ————
        # Each is a small categorical (or binary) head with its own local-attention
        # block over fpn_lvl0. Per-class targets derived from the K=48 attribute table.
        # Inference: gather log-softmax at the held-out class's target, center within
        # siblings (same matrix as v9), add to phrase logit.
        self.use_body_shape_head = use_body_shape_head
        self.use_direction_head = use_direction_head
        self.use_multiplicity_head = use_multiplicity_head
        self.use_named_element_head = use_named_element_head
        # v11 additions
        self.use_motion_category_head = use_motion_category_head
        self.use_leg_configuration_head = use_leg_configuration_head
        self.use_inverted_phase_head = use_inverted_phase_head
        self.use_hands_on_apparatus_head = use_hands_on_apparatus_head
        self.use_sibling_softmax = bool(use_sibling_softmax)
        self.use_raw_log_prob = bool(use_raw_log_prob)
        self.use_attr_axis_contrast = bool(use_attr_axis_contrast)
        self.attr_axis_contrast_margin = float(attr_axis_contrast_margin)
        self.strict_zsl = bool(strict_zsl)
        self.body_shape_head = None
        self.direction_head = None
        self.multiplicity_head = None
        self.named_element_head = None
        self.motion_category_head = None
        self.leg_configuration_head = None
        self.inverted_phase_head = None
        self.hands_on_apparatus_head = None
        # Spec: candidate attribute names per specialist (must match aux_attr_names)
        self._specialist_specs = {
            'body_shape':       ['body_stretched', 'body_tucked', 'body_piked', 'body_ring', 'body_straddle'],
            'direction':        ['direction_forward', 'direction_backward', 'direction_sideward'],
            'multiplicity':     ['multiplicity_double'],   # binary; we treat as 2-way {single=0, double=1}
            'named_element':    ['submotion_switch_leap', 'submotion_johnson', 'submotion_giant_circle',
                                 'submotion_clear_circle', 'submotion_stalder', 'submotion_pike_sole_circle'],
            # v11: motion category (mutually-exclusive 8-way).
            'motion_category':  ['motion_salto', 'motion_turn', 'motion_leap', 'motion_jump',
                                 'motion_walkover', 'motion_handstand', 'motion_dismount', 'motion_transition'],
            # v11: leg configuration (mutually-exclusive 3-way, LMA limb relationship).
            'leg_configuration':['split_leg', 'both_leg_landing', 'one_leg_landing'],
            # v11: inverted phase (binary — head below hips at some point).
            'inverted_phase':   ['inverted_phase'],
            # v11: hands on apparatus (binary).
            'hands_on_apparatus':['hands_on_apparatus'],
        }
        _any_specialist = (use_body_shape_head or use_direction_head or
                           use_multiplicity_head or use_named_element_head or
                           use_motion_category_head or use_leg_configuration_head or
                           use_inverted_phase_head or use_hands_on_apparatus_head)
        if _any_specialist:
            assert aux_attribute_table_path, "v10 specialists require aux_attribute_table_path"
            with open(aux_attribute_table_path) as _f:
                _t = json.load(_f)
            _aux_names = _t['attribute_names']
            # Dataset-generic specialists: the attribute table may carry its own
            # slot definitions ({slot: {candidates: [...], binary: bool}}), e.g.
            # for THUMOS14. The 8 slot NAMES stay fixed (all head/loss/inference
            # plumbing is keyed on them); only their semantics change per dataset.
            _spec_binary_map = {'multiplicity': True, 'inverted_phase': True,
                                'hands_on_apparatus': True}
            if 'specialist_specs' in _t:
                self._specialist_specs = {
                    k: v['candidates'] for k, v in _t['specialist_specs'].items()}
                _spec_binary_map = {
                    k: bool(v.get('binary', False))
                    for k, v in _t['specialist_specs'].items()}
                print(f"[generic-spec] specialist slots overridden from attribute "
                      f"table: { {k: len(v) for k, v in self._specialist_specs.items()} }")

            def _build_targets(candidate_attrs, binary=False):
                """Return (target[num_classes], valid[num_classes])."""
                tgt = torch.full((num_classes,), -1, dtype=torch.long)
                val = torch.zeros(num_classes, dtype=torch.bool)
                idxs = [_aux_names.index(n) for n in candidate_attrs]
                for aid, entry in _t['actions'].items():
                    cid = int(aid[1:])
                    flags = [int(entry['attrs'][i]) for i in idxs]
                    if binary:
                        # 2-way: target = flag value (0 or 1); always valid
                        tgt[cid] = flags[0]
                        val[cid] = True
                    else:
                        if sum(flags) > 0:
                            # Choose the first attribute that is 1 (handles rare multi-set cases)
                            tgt[cid] = flags.index(1)
                            val[cid] = True
                return tgt, val

            def _add_specialist(name, candidates, binary=False):
                tgt, val = _build_targets(candidates, binary=binary)
                self.register_buffer(f'{name}_target_per_class', tgt)
                self.register_buffer(f'{name}_valid_per_class', val)
                n_classes_head = 2 if binary else len(candidates)
                # background-aware: extra last class = 'none'; trained on sampled
                # background tokens so log p(real attr) is calibrated off-GT.
                if bool(train_cfg.get('background_aware_specialists', False)):
                    n_classes_head += 1
                head = PtTransformerCategoricalSpecialistHead(
                    input_dim=fpn_dim,
                    num_classes=n_classes_head,
                    n_head=int(specialist_head_n_head),
                    mha_win_size=int(specialist_head_mha_win_size),
                    num_layers=int(specialist_head_num_layers),
                    with_ln=head_with_ln,
                )
                print(f"[ZSL v10] {name} head ({n_classes_head}-way, n_layers={specialist_head_num_layers}, "
                      f"win={specialist_head_mha_win_size}): "
                      f"{int(val.sum())}/{num_classes} classes have valid target")
                return head

            if use_body_shape_head:
                self.body_shape_head = _add_specialist('body_shape', self._specialist_specs['body_shape'])
            if use_direction_head:
                self.direction_head = _add_specialist('direction', self._specialist_specs['direction'])
            if use_multiplicity_head:
                self.multiplicity_head = _add_specialist(
                    'multiplicity', self._specialist_specs['multiplicity'],
                    binary=_spec_binary_map.get('multiplicity', True)
                )
            if use_named_element_head:
                self.named_element_head = _add_specialist(
                    'named_element', self._specialist_specs['named_element']
                )
            # v11
            if use_motion_category_head:
                self.motion_category_head = _add_specialist(
                    'motion_category', self._specialist_specs['motion_category']
                )
            if use_leg_configuration_head:
                self.leg_configuration_head = _add_specialist(
                    'leg_configuration', self._specialist_specs['leg_configuration'],
                    binary=_spec_binary_map.get('leg_configuration', False)
                )
            if use_inverted_phase_head:
                self.inverted_phase_head = _add_specialist(
                    'inverted_phase', self._specialist_specs['inverted_phase'],
                    binary=_spec_binary_map.get('inverted_phase', True)
                )
            if use_hands_on_apparatus_head:
                self.hands_on_apparatus_head = _add_specialist(
                    'hands_on_apparatus', self._specialist_specs['hands_on_apparatus'],
                    binary=_spec_binary_map.get('hands_on_apparatus', True)
                )


            # ———— Phase 2D v17: precompute sibling_target_mask per categorical specialist ————
            # For each class c and specialist A, mark which OTHER target values are
            # held by within-phrase siblings of c (seen + held-out). Used by attr-axis
            # contrastive loss: at a positive token of class c, specialist A is pushed
            # to rank c's target ABOVE these sibling targets by a margin.
            if action_to_phrase_list is not None:
                _cat_specs = [
                    ('body_shape',          5),
                    ('direction',           3),
                    ('multiplicity',        2),
                    ('named_element',       6),
                    ('motion_category',     8),
                    ('leg_configuration',   3),
                    ('inverted_phase',      2),
                    ('hands_on_apparatus',  2),
                ]
                from collections import defaultdict
                phrase_to_classes = defaultdict(list)
                for _ci, _pi in enumerate(action_to_phrase_list):
                    phrase_to_classes[_pi].append(_ci)
                for _sn, _num_cat in _cat_specs:
                    _tgt_attr = f'{_sn}_target_per_class'
                    _val_attr = f'{_sn}_valid_per_class'
                    if not (hasattr(self, _tgt_attr) and hasattr(self, _val_attr)):
                        continue
                    # Use the head's ACTUAL category count (specialist_specs may
                    # override slot arities per dataset, e.g. THUMOS env = 6-way
                    # in the body_shape slot vs FineGym's 5-way).
                    _head_obj = getattr(self, f'{_sn}_head', None)
                    if _head_obj is not None and hasattr(_head_obj, 'num_classes'):
                        _num_cat = int(_head_obj.num_classes)
                    _tgt_per_class = getattr(self, _tgt_attr)
                    _val_per_class = getattr(self, _val_attr)
                    # mask[c, k] = 1 if k is target of any within-phrase sibling of c, AND k != target[c].
                    _mask = torch.zeros(num_classes, _num_cat, dtype=torch.float32)
                    for _c in range(num_classes):
                        if not bool(_val_per_class[_c].item()):
                            continue
                        _c_target = int(_tgt_per_class[_c].item())
                        _parent = action_to_phrase_list[_c]
                        for _c_other in phrase_to_classes[_parent]:
                            if _c_other == _c:
                                continue
                            if not bool(_val_per_class[_c_other].item()):
                                continue
                            _o_target = int(_tgt_per_class[_c_other].item())
                            if _o_target != _c_target and 0 <= _o_target < _num_cat:
                                _mask[_c, _o_target] = 1.0
                    self.register_buffer(f'{_sn}_sibling_target_mask', _mask, persistent=False)
                    print(f"[ZSL v17] {_sn} sibling_target_mask: "
                          f"avg {_mask.sum(dim=-1).mean().item():.2f} sibling targets per class")


        # ———— champion stage 3: video-side subtree attention ————
        # Built LAST so every shared module above consumes the same RNG
        # draws as in stages 1-2; requires the text pathway (it rewrites the
        # cls-trunk features the cosine classifier reads). Attribute name
        # `video_subtree_attn` => --freeze_except video_subtree_attn selects
        # exactly its 4 MHA tensors x 2.
        self.video_subtree_attn = None
        if use_video_subtree_attn:
            assert self.text_pathway is not None, \
                "use_video_subtree_attn requires use_text_cls_head"
            if video_subtree_nested:
                # two-level variant: K action chunks x Kp sub-action leaves;
                # same attribute so freeze_except / the forward call are
                # unchanged (4 MHA's x 2 tensors each)
                from .vsa import VideoSubtreeAttnNested
                self.video_subtree_attn = VideoSubtreeAttnNested(
                    head_dim, K=int(video_subtree_K), Kp=int(video_subtree_Kp),
                    heads=int(video_subtree_heads))
                print(f"[t3-VSA] NESTED video-side subtree attention active: "
                      f"K={int(video_subtree_K)} action chunks x "
                      f"Kp={int(video_subtree_Kp)} sub-action leaves, "
                      f"heads={int(video_subtree_heads)}, zero-init residuals "
                      f"(4 MHA's), replaces the cls-trunk feature")
            else:
                from .vsa import VideoSubtreeAttn
                self.video_subtree_attn = VideoSubtreeAttn(
                    head_dim, K=int(video_subtree_K), heads=int(video_subtree_heads))
                print(f"[t3-VSA] video-side subtree attention active: "
                      f"K={int(video_subtree_K)} chunks, heads={int(video_subtree_heads)}, "
                      f"zero-init residuals, replaces the cls-trunk feature")

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9
        self.half_enable = True # False True

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, video_list):

        bs = len(video_list)
        # if self.backbone_type == 'convTransformer':
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        batched_inputs, batched_masks = self.preprocessing(video_list)
        
        if self.half_enable:
            # Cast inputs to the model's actual parameter dtype (fp16 under the
            # fp16 deepspeed engine, bf16 under bf16) instead of hardcoding .half().
            _pdt = next(self.backbone.parameters()).dtype
            batched_inputs = batched_inputs.to(_pdt)
            batched_masks = batched_masks.to(_pdt)

        # forward the network (backbone -> neck -> heads)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = self.point_generator(fpn_feats)

        # out_offset: List[B, 2, T_i]  (computed before the classifier: the
        # stage-3 video branch reads the DETACHED offsets; the reg head has
        # no RNG, so the reordering is byte-inert)
        out_offsets = self.reg_head(fpn_feats, fpn_masks)

        # out_cls: List[B, #cls + 1, T_i]
        self._fg_logits = None
        if self.text_pathway is not None:
            # t3 text pathway: trunk features -> scaled cosine vs class
            # text embeddings (train-time HP dropout / stem degrade + PA
            # live inside class_embeddings()).
            _, cls_feats = self.cls_head(fpn_feats, fpn_masks, return_feats=True)
            if self.video_subtree_attn is not None:
                # stage 3 (video branch, vsa.py): per location, pool the
                # candidate span [n - d_s, n + d_e] (detached offsets, level
                # grid) into K ordered chunks, neighbour attention among
                # them, then f_n attends over [f_n; chunks]. REPLACES the
                # cls-trunk features for the cosine classifier below (train
                # and eval, every level); reg head keeps the raw features.
                cls_feats = self.video_subtree_attn(
                    cls_feats,
                    [o.detach().permute(0, 2, 1) for o in out_offsets],
                    fpn_masks)
            cls_emb = self.text_pathway.class_embeddings()          # (C, D)
            out_cls_logits = tuple()
            for lvl, (f, m) in enumerate(zip(cls_feats, fpn_masks)):
                if self.text_tv_attn is not None:
                    # system 2 (Ti-FAD): per-level text<-video TiCA update,
                    # then re-classify with the level-adapted embeddings
                    txt = cls_emb.unsqueeze(0).expand(f.shape[0], -1, -1)
                    prelim = self.text_pathway.cosine_logits(f, cls_emb)
                    txt = self.text_tv_attn[lvl](f, m, txt, prelim)
                    logits = self.text_pathway.cosine_logits(f, txt)
                else:
                    logits = self.text_pathway.cosine_logits(f, cls_emb)
                out_cls_logits += (logits, )
            if self.fg_head is not None:
                out_fg = self.fg_head(fpn_feats, fpn_masks)         # (B,1,T_i)
                self._fg_logits = [x.permute(0, 2, 1) for x in out_fg]
        else:
            out_cls_logits = self.cls_head(fpn_feats, fpn_masks)


        # ———— Cascaded phrase + activity heads (forward) ————
        self._cascade_phrase_logits = None
        self._cascade_phrase_offsets = None
        self._cascade_activity_logits = None
        self._cascade_activity_offsets = None
        if self.use_cascaded_hierarchy:
            phrase_feats_in = tuple(fpn_feats[l] for l in self.cascade_phrase_levels)
            phrase_masks_in = tuple(fpn_masks[l] for l in self.cascade_phrase_levels)
            if self.cascade_gradient_stop:
                phrase_feats_in = tuple(f.detach() for f in phrase_feats_in)
            phrase_cls, phrase_reg, phrase_pre_feats = self.phrase_cascade_head(
                phrase_feats_in, phrase_masks_in, return_feats=True,
            )
            # Permute phrase outputs to (B, T_l, C) convention used by the existing pipeline.
            self._cascade_phrase_logits = [x.permute(0, 2, 1) for x in phrase_cls]
            self._cascade_phrase_offsets = [x.permute(0, 2, 1) for x in phrase_reg]
            # coarse->fine conditioning: expose the level-0 phrase-trunk features
            # (detached in _spec_in below so specialist CE cannot corrupt the trunk)
            self._phrase_trunk_feats_lvl0 = None
            if 0 in self.cascade_phrase_levels:
                self._phrase_trunk_feats_lvl0 = phrase_pre_feats[
                    self.cascade_phrase_levels.index(0)]


            # Activity head: cascade on phrase pre-cls feat at the activity level.
            # Skipped entirely when cascade_activity_levels is empty (2-level setup).
            if self.activity_cascade_head is not None:
                activity_level = self.cascade_activity_levels[0]
                phrase_idx_in_subset = self.cascade_phrase_levels.index(activity_level)
                activity_in_feat = phrase_pre_feats[phrase_idx_in_subset]
                activity_in_mask = phrase_masks_in[phrase_idx_in_subset]
                act_cls, act_reg = self.activity_cascade_head(activity_in_feat, activity_in_mask)
                self._cascade_activity_logits = act_cls.permute(0, 2, 1)   # (B, T_3, num_activities)
                self._cascade_activity_offsets = act_reg.permute(0, 2, 1)  # (B, T_3, 2)


        # ---- Phase 2D v11: split rotation heads (forward) ----
        # Per-token scalar prediction of rotation count (aerial twist / floor
        # turn). Trained with smooth-L1; at inference contributes a log-RBF
        # tiebreaker to held-out class scores.
        self._aerial_twist_logits = None    # (B, T_0)
        self._floor_turn_logits = None      # (B, T_0)
        # All specialist heads read the raw level-0 FPN features, OR (coarse->fine
        # conditioning, train_cfg['specialists_on_phrase_feats']) the DETACHED
        # level-0 phrase-trunk features - worth +2.7..+9.7 on THUMOS when the
        # phrase level is the weak link (see STRICT_ZSL notes SS2.2).
        _spec_in = fpn_feats[0]
        if self.train_cfg.get('specialists_on_phrase_feats', False):
            _pt = getattr(self, '_phrase_trunk_feats_lvl0', None)
            assert _pt is not None, \
                "specialists_on_phrase_feats requires cascade level 0"
            assert _pt.shape[1] == fpn_feats[0].shape[1], \
                f"phrase trunk dim {_pt.shape[1]} != specialist input dim {fpn_feats[0].shape[1]}"
            _spec_in = _pt.detach()

        def _run_spec(head, name):
            return head(_spec_in, fpn_masks[0])

        if self.aerial_twist_head is not None:
            self._aerial_twist_logits = _run_spec(self.aerial_twist_head, 'aerial_twist')
        if self.floor_turn_head is not None:
            self._floor_turn_logits = _run_spec(self.floor_turn_head, 'floor_turn')

        # ———— Phase 2D v10: multi-specialist heads (forward) ————
        # Each runs over fpn_lvl0 features and produces per-token category logits.
        self._body_shape_logits = None
        self._direction_logits = None
        self._multiplicity_logits = None
        self._named_element_logits = None
        if self.body_shape_head is not None:
            self._body_shape_logits = _run_spec(self.body_shape_head, 'body_shape')
        if self.direction_head is not None:
            self._direction_logits = _run_spec(self.direction_head, 'direction')
        if self.multiplicity_head is not None:
            self._multiplicity_logits = _run_spec(self.multiplicity_head, 'multiplicity')
        if self.named_element_head is not None:
            self._named_element_logits = _run_spec(self.named_element_head, 'named_element')
        # v11
        self._motion_category_logits = None
        self._leg_configuration_logits = None
        self._inverted_phase_logits = None
        self._hands_on_apparatus_logits = None
        if self.motion_category_head is not None:
            self._motion_category_logits = _run_spec(self.motion_category_head, 'motion_category')
        if self.leg_configuration_head is not None:
            self._leg_configuration_logits = _run_spec(self.leg_configuration_head, 'leg_configuration')
        if self.inverted_phase_head is not None:
            self._inverted_phase_logits = _run_spec(self.inverted_phase_head, 'inverted_phase')
        if self.hands_on_apparatus_head is not None:
            self._hands_on_apparatus_logits = _run_spec(self.hands_on_apparatus_head, 'hands_on_apparatus')


        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]

        # ———— t3 system 2 (Ti-FAD): eval-time foreground weighting ————
        # decode score = sigmoid(cls) * sigmoid(fg) (the fork uses
        # sqrt(sig(cls)*sig(cn))*sig(fg); we have no cn head). Folded back
        # into logit space so the downstream sigmoid decode is unchanged.
        if (not self.training) and (self._fg_logits is not None):
            new_logits = []
            for cls_l, fg_l in zip(out_cls_logits, self._fg_logits):
                p = (cls_l.float().sigmoid() * fg_l.float().sigmoid()).clamp(
                    1e-6, 1.0 - 1e-6)
                new_logits.append(
                    (p.log() - (1.0 - p).log()).to(cls_l.dtype))
            out_cls_logits = new_logits

        # ———— Phase 2D ZSL: overwrite held-out class slots with derived logits ————
        # Held-out classes were never seen by the flat head, so their flat-head
        # logits are uninformative. Replace those slots with the composed score:
        # phrase logit at the parent phrase + per-specialist raw log-prob at the
        # class's target attribute + rotation log-RBF (inference only).
        if (
            self.use_cascaded_hierarchy
            and self.aux_attr_held_out_ids
            and not self.training
        ):
            held_ids = torch.tensor(self.aux_attr_held_out_ids,
                                    dtype=torch.long, device=out_cls_logits[0].device)
            # ---- oracle diagnostics (eval-only, off unless train_cfg['oracle_specialists']) ----
            # For tokens covered by a GT segment, replace the named component of the
            # composed score with its ground-truth value: 'phrase' -> +/-8 phrase logit,
            # '<specialist>' -> 0 / -12 log-prob at the covering class's target attribute.
            # Uncovered (background) tokens keep the real predictions, so localization
            # is NOT oracled — only attribute/phrase recognition on action tokens.
            oracle_set = set(self.train_cfg.get('oracle_specialists', []) or [])
            if 'all' in oracle_set:
                oracle_set |= {'phrase', 'body_shape', 'direction', 'multiplicity',
                               'named_element', 'motion_category', 'leg_configuration',
                               'inverted_phase', 'hands_on_apparatus'}
            _oracle_cover_cache = {}

            def _oracle_cover(T_l, lvl, device):
                key = (T_l, lvl)
                if key not in _oracle_cover_cache:
                    B_oc = len(video_list)
                    cover = torch.full((B_oc, T_l), -1, dtype=torch.long, device=device)
                    centers = (torch.arange(T_l, device=device, dtype=torch.float32) + 0.5) \
                        * float(2 ** lvl)
                    for b_oc, item in enumerate(video_list):
                        segs, labs = item.get('segments'), item.get('labels')
                        if segs is None or labs is None or len(segs) == 0:
                            continue
                        segs = segs.to(device)
                        for i_oc in range(len(labs)):
                            inside = (centers >= segs[i_oc, 0]) & (centers <= segs[i_oc, 1])
                            cover[b_oc, inside] = int(labs[i_oc])
                    _oracle_cover_cache[key] = cover
                return _oracle_cover_cache[key]

            if (
                self.use_cascaded_hierarchy
                and self._cascade_phrase_logits is not None
                and self.action_to_phrase_buf is not None
            ):
                # v4 parent-conditional path: score(h, t) = phrase_logit[parent(h), t]
                # v6 hybrid (BCE attr head):
                #   + α_attr · centered_attr_LL (within-phrase sibling tiebreaker)
                # v9' rotation hybrid (rotation regression head):
                #   + α_rot · centered_log_RBF (rotation-count sibling tiebreaker)
                # Centering ensures unique-phrase held-out classes (no held-out siblings
                # sharing their parent phrase) get zero contribution → identical to v4.
                parent_phrases = self.action_to_phrase_buf.to(held_ids.device)[held_ids]  # (H,)

                # ---- v10/v11 multi-specialist: prep ----
                # For each enabled specialist, collect (logits, held_target, held_valid, alpha).
                v10_specialist_args = []
                for name, logits, alpha_key, default_alpha in [
                    ('body_shape',          self._body_shape_logits,          'held_out_body_shape_alpha',          1.0),
                    ('direction',           self._direction_logits,           'held_out_direction_alpha',           1.0),
                    ('multiplicity',        self._multiplicity_logits,        'held_out_multiplicity_alpha',        1.0),
                    ('named_element',       self._named_element_logits,       'held_out_named_element_alpha',       1.0),
                    # v11
                    ('motion_category',     self._motion_category_logits,     'held_out_motion_category_alpha',     1.0),
                    ('leg_configuration',   self._leg_configuration_logits,   'held_out_leg_configuration_alpha',   1.0),
                    ('inverted_phase',      self._inverted_phase_logits,      'held_out_inverted_phase_alpha',      1.0),
                    ('hands_on_apparatus',  self._hands_on_apparatus_logits,  'held_out_hands_on_apparatus_alpha',  1.0),
                ]:
                    if logits is None:
                        continue
                    tgt_per_class = getattr(self, f'{name}_target_per_class')
                    val_per_class = getattr(self, f'{name}_valid_per_class')
                    held_tgt = tgt_per_class.to(out_cls_logits[0].device)[held_ids].long()   # (H,)
                    held_val = val_per_class.to(out_cls_logits[0].device)[held_ids].float()  # (H,)
                    alpha = float(self.train_cfg.get(alpha_key, default_alpha))
                    v10_specialist_args.append((name, logits, held_tgt, held_val, alpha))

                # ---- v11 split rotation: prep ----
                # Each rotation head (aerial_twist / floor_turn) only contributes to
                # held-out classes whose rotation_type matches. Build per-head held-out
                # masks and target tensors.
                use_split_rotation_inf = (
                    self._aerial_twist_logits is not None
                    and self._floor_turn_logits is not None
                    and hasattr(self, 'rotation_type_per_class')
                )
                if use_split_rotation_inf:
                    rot_type_h = self.rotation_type_per_class.to(out_cls_logits[0].device)[held_ids]  # (H,)
                    held_rot_v11 = self.rotation_count_per_class.to(out_cls_logits[0].device)[held_ids]  # (H,)
                    aer_mask_h = (rot_type_h == 1).float()
                    flr_mask_h = (rot_type_h == 2).float()
                    alpha_aer = float(self.train_cfg.get('held_out_aerial_twist_alpha', 0.1))
                    alpha_flr = float(self.train_cfg.get('held_out_floor_turn_alpha',  0.1))
                    sigma_split = float(self.train_cfg.get('held_out_rotation_sigma', 0.15))
                    r_aer = self._aerial_twist_logits          # (B, T_0)
                    r_flr = self._floor_turn_logits            # (B, T_0)
                    T0_split = r_aer.shape[1]
                for idx_in_cascade, lvl in enumerate(self.cascade_phrase_levels):
                    phrase_logits_lvl = self._cascade_phrase_logits[idx_in_cascade]
                    phrase_logits_held = phrase_logits_lvl[..., parent_phrases]  # (B, T_l, H)
                    if 'phrase' in oracle_set:
                        T_ph = phrase_logits_lvl.shape[1]
                        cover_l = _oracle_cover(T_ph, lvl, phrase_logits_held.device)
                        a2p = self.action_to_phrase_buf.to(cover_l.device)
                        ph_map = torch.where(
                            cover_l >= 0, a2p[cover_l.clamp(min=0)],
                            torch.full_like(cover_l, -1))
                        ph_match = ph_map.unsqueeze(-1) == parent_phrases.view(1, 1, -1)
                        # Suppress-only oracle: on GT-covered tokens, push down
                        # held classes whose parent phrase does NOT match the
                        # covering class's phrase; matching candidates keep their
                        # PREDICTED logit (no saturation — score calibration and
                        # ranking vs seen classes stay realistic).
                        phrase_logits_held = torch.where(
                            (cover_l >= 0).unsqueeze(-1) & ~ph_match,
                            torch.tensor(-8.0, device=ph_match.device,
                                         dtype=phrase_logits_held.dtype),
                            phrase_logits_held)
                    held_scores = phrase_logits_held

                    # ---- v10 multi-specialist: per-level scoring contribution ----
                    # For each specialist:
                    #   log_p[b, t_l, c] = log softmax over categories at downsampled t_l
                    #   gather at held_target → (B, T_l, H), zero out invalid held classes,
                    #   center within phrase-siblings (same S matrix), add to held_scores.
                    if v10_specialist_args:
                        T_l = phrase_logits_lvl.shape[1]
                        for name, sp_logits, sp_held_tgt, sp_held_val, sp_alpha in v10_specialist_args:
                            T0_sp = sp_logits.shape[2]
                            if T_l == T0_sp:
                                sp_lvl = sp_logits
                            else:
                                sp_lvl = F.adaptive_avg_pool1d(sp_logits.float(), T_l)
                            log_p_sp = F.log_softmax(sp_lvl.float(), dim=1).permute(0, 2, 1)  # (B, T_l, C)
                            B_sp, T_sp, C_sp = log_p_sp.shape
                            H_sp = sp_held_tgt.shape[0]
                            tgt_safe = sp_held_tgt.clone()
                            tgt_safe[tgt_safe < 0] = 0
                            log_p_h = log_p_sp.gather(
                                -1, tgt_safe.view(1, 1, H_sp).expand(B_sp, T_sp, H_sp)
                            )                                                              # (B, T_l, H)
                            if name in oracle_set:
                                cover_l = _oracle_cover(T_sp, lvl, log_p_h.device)
                                tgt_pc = getattr(self, f'{name}_target_per_class').to(
                                    cover_l.device).long()
                                val_pc = getattr(self, f'{name}_valid_per_class').to(
                                    cover_l.device)
                                cov_tgt = torch.where(
                                    cover_l >= 0, tgt_pc[cover_l.clamp(min=0)],
                                    torch.full_like(cover_l, -1))
                                cov_ok = (cover_l >= 0) \
                                    & (val_pc[cover_l.clamp(min=0)] > 0) & (cov_tgt >= 0)
                                sp_match = cov_tgt.unsqueeze(-1) == sp_held_tgt.view(1, 1, -1)
                                oracle_v = torch.where(
                                    sp_match,
                                    torch.tensor(0.0, device=sp_match.device),
                                    torch.tensor(-12.0, device=sp_match.device),
                                ).to(log_p_h.dtype)
                                log_p_h = torch.where(
                                    cov_ok.unsqueeze(-1), oracle_v, log_p_h)
                            # v16 raw log-prob: skip sibling normalization, use the
                            # per-token log-prob at the target attribute directly. This
                            # is unbounded (sibling-softmax was bounded by ±log(k)),
                            # giving more discriminative magnitude when the specialist
                            # is confident but uncalibrated against siblings. The bound
                            # ±log(k) of sibling-softmax was identified as too weak to
                            # discriminate categorical attributes (BB_dismounts crashed
                            # under oracle phrase because body_shape's ±0.07 contribution
                            # was overwhelmed by phrase magnitude).
                            if self.use_raw_log_prob:
                                centered_sp = log_p_h
                            else:
                                centered_sp = self._apply_sibling_normalization(
                                    log_p_h, valid_mask=sp_held_val.bool(),
                                )
                            held_scores = held_scores + sp_alpha * centered_sp.to(held_scores.dtype)

                    # ---- v11 split rotation contribution (per-level) ----
                    # Each rotation head adds a centered log-RBF only to held-out classes
                    # whose rotation_type matches (aerial vs floor). The valid mask zeroes
                    # out contributions for classes where the head doesn't apply.
                    if use_split_rotation_inf:
                        T_l = phrase_logits_lvl.shape[1]
                        for r_pred, mask_h, alpha_split, head_name in [
                            (r_aer, aer_mask_h, alpha_aer, 'aerial_twist'),
                            (r_flr, flr_mask_h, alpha_flr, 'floor_turn'),
                        ]:
                            if T_l == T0_split:
                                r_lvl = r_pred
                            else:
                                r_lvl = F.adaptive_avg_pool1d(
                                    r_pred.unsqueeze(1).float(), T_l
                                ).squeeze(1)
                            log_rbf_split = -((r_lvl.float().unsqueeze(-1) - held_rot_v11.view(1, 1, -1).float()) ** 2) \
                                            / (2.0 * sigma_split * sigma_split)
                            # v16: raw log-prob option for rotation too.
                            if self.use_raw_log_prob:
                                centered_split = log_rbf_split
                            else:
                                centered_split = self._apply_sibling_normalization(
                                    log_rbf_split, valid_mask=mask_h.bool(),
                                )
                            # Only this head's matching-type classes contribute.
                            centered_split = centered_split * mask_h.view(1, 1, -1)
                            held_scores = held_scores + alpha_split * centered_split.to(held_scores.dtype)


                    out_cls_logits[lvl] = out_cls_logits[lvl].clone()
                    out_cls_logits[lvl][..., held_ids] = held_scores
                # FPN levels not in cascade_phrase_levels: by default leave the
                # (uninformative) flat-head logit unchanged. With
                # train_cfg['extend_held_out_to_all_levels'], propagate the
                # composed score from the coarsest cascaded level by pooling —
                # long actions anchored at coarse levels otherwise can never be
                # detected as held-out classes (their flat-head logits were
                # never trained).
                if self.train_cfg.get('extend_held_out_to_all_levels', False):
                    src_scores = held_scores  # (B, T_src, H) — coarsest cascaded level
                    for lvl2 in range(len(out_cls_logits)):
                        if lvl2 in self.cascade_phrase_levels:
                            continue
                        T_t = out_cls_logits[lvl2].shape[1]
                        pooled = F.adaptive_avg_pool1d(
                            src_scores.permute(0, 2, 1).float(), T_t
                        ).permute(0, 2, 1)
                        out_cls_logits[lvl2] = out_cls_logits[lvl2].clone()
                        out_cls_logits[lvl2][..., held_ids] = pooled.to(
                            out_cls_logits[lvl2].dtype)


        out_phrase_logits = None
        out_activity_logits = None



        # return loss during training
        if self.training:
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels = [x['labels'].to(self.device) for x in video_list]

            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets, gt_cls_raw_labels = self.label_points(
                points, gt_segments, gt_labels)

            # held_out_mode=exclude (train_cfg, propagated from the dataset
            # block by train_shard.py): every token whose centre lies inside
            # a held-out (unseen-class) segment is EXCLUDED from all losses
            # (classification, boundary regression, ancestor) -- neither
            # positive nor background. Only the extent of those segments is
            # used, to know what to skip; their labels enter no target.
            heldout_span_mask = None
            if str(self.train_cfg.get('held_out_mode', 'keep')) == 'exclude' \
                    and self.aux_attr_held_out_ids:
                _t = torch.cat(points, dim=0)[:, 0]                  # (FT,)
                _held = set(int(h) for h in self.aux_attr_held_out_ids)
                _rows = []
                for _seg, _lab in zip(gt_segments, gt_labels):
                    _m = torch.zeros_like(_t, dtype=torch.bool)
                    _sel = torch.tensor([int(l) in _held for l in _lab.tolist()],
                                        dtype=torch.bool, device=_seg.device)
                    if _sel.any():
                        _hs = _seg[_sel]                              # (H, 2)
                        _m = ((_t[:, None] >= _hs[None, :, 0]) &
                              (_t[:, None] <= _hs[None, :, 1])).any(dim=1)
                    _rows.append(_m)
                heldout_span_mask = torch.stack(_rows)               # (B, FT)

            # compute the loss and return
            losses = self.losses(
                fpn_masks,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets, gt_cls_raw_labels, gt_labels,
                gt_segments=gt_segments,
                heldout_span_mask=heldout_span_mask,
            )


            if self.use_cascaded_hierarchy:
                cas_losses = self.cascaded_hierarchy_losses(
                    fpn_masks=[m.unsqueeze(1) if m.dim() == 2 else m for m in fpn_masks],
                    points=points,
                    gt_segments=gt_segments,
                    gt_labels=gt_labels,
                    video_list=video_list,
                    out_offsets=out_offsets,
                    current_epoch=getattr(self, '_current_epoch', 0),
                )
                for k, v in cas_losses.items():
                    if k != 'cascaded_total':
                        losses[k] = v
                losses['final_loss'] = losses['final_loss'] + cas_losses['cascaded_total']




            return losses

        else:
            # ---- ancestor-consistent scoring (eval only, FG_ANC_GATE=1) ----
            # Diagnostic of 2026-09-23 (FineGym l_d=2, exclude protocol): an
            # unseen element set fires on the SEEN set with the same movement
            # on another apparatus (FX turns -> BB turns, BB dismount -> UB
            # dismount) because their descriptions differ only by the
            # apparatus word. The hierarchy says an action can only occur
            # where its ancestor occurs, so every class logit is gated by its
            # immediate ancestor's score read from the same logits (Eq. 4:
            # logsumexp over the ancestor's SEEN members). Log-space:
            # logit_c += logsigmoid(z_{a(c)}), i.e. p_c *= p_{a(c)} for small
            # probabilities. Uses no unseen annotation; applies to all classes.
            if os.environ.get('FG_ANC_GATE', '') == '1' and self.text_pathway is not None:
                if getattr(self, '_as_levels', None):
                    lvl_name, _members = self._as_levels[0]  # immediate ancestor level
                else:
                    # no ancestor loss in this model (e.g. the Ti-FAD baseline):
                    # gate by the apparatus groups of the text pathway's tree
                    lvl_name = 'apparatus'
                    _a2a = self.text_pathway.action_to_activity
                    _members = [torch.nonzero(_a2a == a, as_tuple=True)[0]
                                for a in range(int(_a2a.max().item()) + 1)]
                _held = set(int(h) for h in self.aux_attr_held_out_ids)
                _a2x = {'phrase': self.text_pathway.action_to_phrase,
                        'apparatus': self.text_pathway.action_to_activity}[lvl_name]
                gated = []
                for lg in out_cls_logits:                    # (B, T, C)
                    z = lg.new_full((lg.shape[0], lg.shape[1], len(_members)), -1e4)
                    for a, m in enumerate(_members):
                        m_seen = torch.tensor([int(c) for c in m.tolist() if int(c) not in _held],
                                              device=lg.device, dtype=torch.long)
                        if m_seen.numel():
                            z[..., a] = torch.logsumexp(lg[..., m_seen], dim=-1)
                    gate = F.logsigmoid(z)[..., _a2x.to(lg.device)]   # (B, T, C)
                    gated.append(lg + gate)
                out_cls_logits = gated
                if not getattr(self, '_anc_gate_logged', False):
                    self._anc_gate_logged = True
                    print(f"[anc-gate] eval-time ancestor-consistent scoring on: level={lvl_name}, "
                          f"{len(_members)} ancestors, seen members only")
            # decode the actions (sigmoid / stride, etc)
            results = self.inference(
                video_list, points, fpn_masks,
                out_cls_logits, out_offsets,
                out_phrase_logits=out_phrase_logits,
                out_activity_logits=out_activity_logits
            )

            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels = [x['labels'].to(self.device) for x in video_list]

            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets, _ = self.label_points(
                points, gt_segments, gt_labels)

            # compute all_logits:
            valid_mask = torch.cat(fpn_masks, dim=1)
            gt_cls = torch.stack(gt_cls_labels)
            pos_mask = ((gt_cls.sum(-1) > 0) & valid_mask).cpu()
            all_logits = torch.cat(out_cls_logits, dim=1).cpu()
            # Save logits for each video and for each label
            self.last_video_label_logits = {}
            for i, video in enumerate(video_list):
                label_logits = defaultdict(list)
                pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
                if pos_indices.numel() > 0:
                    gt_cl_ = torch.argmax(gt_cls[i][pos_indices], dim=1)

                    logit_ = all_logits[i][pos_indices].float().sigmoid().cpu()
                    for g, l in zip(gt_cl_, logit_):
                        label_logits[int(g)].append(l.float())
                self.last_video_label_logits[video['video_id']] = label_logits

            return results

    def _apply_sibling_normalization(self, log_p_h, valid_mask=None):
        """Normalize per-held-out log-scores within their parent-phrase sibling group.

        Two modes, gated by `self.use_sibling_softmax`:
          - False (default v6/v9/v10): SIBLING CENTERING.
              centered[h] = log_p[h] - mean(log_p[h'] for h' in siblings(h))
          - True (v11): SIBLING SOFTMAX.
              norm[h]    = log_p[h] - logsumexp(log_p[h'] for h' in siblings(h))

        v12+ adds `valid_mask` (H,) bool: invalid siblings are excluded from the
        normalization denominator (set to -inf in logsumexp). Previously invalid
        siblings had their log_p zeroed to 0, but 0 » typical log-softmax values
        (~-1.6 to 0) → polluted the denominator and made valid siblings'
        log-softmax artificially negative. With valid_mask, the denominator
        correctly sums only over valid siblings.

        For unique-phrase classes (sibling group of size 1), both modes return 0.

        log_p_h: (B, T_l, H) — per-token log-scores for the H held-out classes.
        valid_mask: (H,) bool tensor, optional.
        Returns same shape as log_p_h.
        """
        if getattr(self, 'use_sibling_softmax', False):
            H = log_p_h.shape[-1]
            sib_mask = (self.held_out_sibling_avg_matrix > 0).to(log_p_h.device)  # (H, H)
            if valid_mask is not None:
                # Restrict the denominator: only sum over valid sibling columns.
                full_mask = sib_mask & valid_mask.to(log_p_h.device).bool().view(1, H)
            else:
                full_mask = sib_mask
            ndim = log_p_h.dim()
            mask_view = full_mask.view(*([1] * (ndim - 1)), H, H)
            expanded = log_p_h.unsqueeze(-2).expand(*([-1] * (ndim - 1)), H, H)
            neg_inf = torch.tensor(
                float('-inf'), dtype=expanded.dtype, device=expanded.device,
            )
            expanded = torch.where(mask_view, expanded, neg_inf)
            lse = torch.logsumexp(expanded, dim=-1)   # (..., H)
            # When an anchor has no valid siblings, lse = -inf. Without correction,
            # log_p_h - (-inf) = +inf, and a subsequent multiplication by 0 (e.g.,
            # held_val mask or discrim mask) produces NaN. Replace lse=-inf with
            # log_p_h itself so the difference is 0 (no contribution).
            lse = torch.where(torch.isinf(lse) & (lse < 0), log_p_h, lse)
            return log_p_h - lse
        else:
            # Sibling centering. If valid_mask is provided, mean only over valid sibs.
            if valid_mask is not None:
                sib_mask = (self.held_out_sibling_avg_matrix > 0).to(log_p_h.device).float()
                valid_f = valid_mask.to(log_p_h.device).float().view(1, -1)
                S_eff = sib_mask * valid_f                                              # (H, H_cand)
                row_sum = S_eff.sum(dim=-1, keepdim=True)                               # (H, 1)
                # Anchors with no valid sibling: result should be 0, not raw log_p_h.
                has_valid = (row_sum.squeeze(-1) > 0).to(log_p_h.dtype)                 # (H,)
                S_eff = S_eff / row_sum.clamp_min(1e-8)
                mean = torch.einsum('bth,ih->bti', log_p_h, S_eff.to(log_p_h.dtype))
                centered = log_p_h - mean
                # Zero out anchors with no valid sibling.
                ndim = log_p_h.dim()
                centered = centered * has_valid.view(*([1] * (ndim - 1)), H)
                return centered
            else:
                S = self.held_out_sibling_avg_matrix.to(log_p_h)
                mean = torch.einsum('bth,ih->bti', log_p_h, S)
                return log_p_h - mean


    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        input_shape = feats[0].shape
        if len(input_shape) == 2:  # batched features: (feature_dim x sequence_length)
            input_type = "features"
            feature_dim = input_shape[0]
        elif len(input_shape) == 4:  # batched frame sequences: (3 x sequence_length x width x height)
            input_type = "frames"
            feats = [feat.permute(0, 2, 3, 1) for feat in feats] # -> (3 x width x height x sequence_length)
    
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len

            if input_type == "features":
                # batch input shape B, C, T
                batch_shape = [len(feats), feats[0].shape[0], max_len]
            elif input_type == "frames":
                # batch input shape B, 3, W, H, T
                batch_shape = [len(feats), 3, input_shape[2], input_shape[3], max_len]
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
            batched_inputs = F.pad(feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)
        
        if input_type == "frames":
            # (B x 3 x width x height x sequence_length) -> (B x sequence_length x 3 x width x height)
            batched_inputs = batched_inputs.permute(0, 4, 1, 2, 3)

        return batched_inputs, batched_masks

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)
        gt_cls, gt_offset, gt_cls_label = [], [], []

        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            cls_targets, reg_targets, cls_targets_label = self.label_points_single_video(
                concat_points, gt_segment, gt_label
            )
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)
            gt_cls_label.append(cls_targets_label)

        return gt_cls, gt_offset, gt_cls_label

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label, num_classes=None):
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        # num_classes : override self.num_classes (used for phrase labels)
        _num_classes = num_classes if num_classes is not None else self.num_classes
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, _num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            cls_targets_label = gt_segment.new_full((num_pts,), -1, dtype=torch.long)
            return cls_targets, reg_targets, cls_targets_label

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

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(
            gt_label, _num_classes
        ).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        cls_targets_label = min_len_mask @ gt_label.to(reg_targets.dtype)
        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets, cls_targets_label
    
    def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls_labels, gt_offsets, gt_cls_raw_labels, gt_labels,
        gt_segments=None,
        heldout_span_mask=None,
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)
        # held_out_mode=exclude: tokens inside held-out segments leave EVERY
        # loss (see forward); as_valid_mask is the ancestor-loss token set.
        as_valid_mask = valid_mask
        if heldout_span_mask is not None:
            if not getattr(self, '_exclude_logged', False):
                self._exclude_logged = True
                print(f"[held_out exclude] first batch: {int((heldout_span_mask & valid_mask).sum())}"
                      f"/{int(valid_mask.sum())} valid tokens lie inside held-out segments "
                      "and are excluded from the cls / reg / ancestor losses")
            valid_mask = valid_mask & (~heldout_span_mask)
            as_valid_mask = valid_mask

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)

        # Phase 2D v4/v5 ZSL: mask out positive tokens whose class label is in the
        # held-out set for action cls loss. reg loss + normalizer behavior depends
        # on the v7 ablation flag `revert_num_pos_v4_style`:
        #   - False (default): reg and normalizer use seen-only positives (v6).
        #   - True (v7b/v4-revert): reg and normalizer use ALL positives (v4).
        # Phrase / activity / consistency losses always use all positives.
        cls_valid_mask = valid_mask
        seen_pos_mask = pos_mask
        if self.aux_attr_held_out_ids:
            held_t = torch.tensor(self.aux_attr_held_out_ids,
                                  dtype=torch.long, device=gt_cls.device)
            heldout_pos_mask = (gt_cls[..., held_t].sum(dim=-1) > 0)   # (B, FT) bool
            cls_valid_mask = valid_mask & (~heldout_pos_mask)
            seen_pos_mask = pos_mask & (~heldout_pos_mask)
        revert_num_pos_v4 = bool(self.train_cfg.get('revert_num_pos_v4_style', False))
        reg_pos_mask = pos_mask if revert_num_pos_v4 else seen_pos_mask
        if heldout_span_mask is not None:
            # exclude mode: held-out positives never reach the boundary loss
            # or the normalizer, whatever the v4-revert flag says
            reg_pos_mask = seen_pos_mask

        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[reg_pos_mask]
        gt_offsets = torch.stack(gt_offsets)[reg_pos_mask]

        # update the loss normalizer (seen-only by default, or all-pos with v4 revert)
        num_pos = reg_pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[cls_valid_mask]

        # optinal label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)

        def one_hot_to_digits(one_hot_vector):
            """Converts one-hot encoded vector to digits. Sets -1 for all-zero vectors."""

            result = torch.argmax(one_hot_vector, axis=1)
            background = torch.sum(one_hot_vector, axis=1) == 0

            # Replace all-zero rows with -1
            result[background] = -1
            # result[[all(value == 0 for value in values) for values in one_hot_vector]] = -1

            return result
        
        # focal loss
        cls_logits_valid = torch.cat(out_cls_logits, dim=1)[cls_valid_mask]
        if self.use_text_cls_head and self.aux_attr_held_out_ids:
            # t3 leakage-aware protocol: with the text pathway, held-out
            # class COLUMNS get no supervision at all (train = seen classes
            # only). Row masking above already removed held-out POSITIVE
            # tokens; this additionally removes the negative pressure on
            # held-out embeddings at seen/background tokens, so held-out
            # classes are scored purely zero-shot at eval.
            col_mask = torch.ones(
                self.num_classes, device=cls_logits_valid.device,
                dtype=torch.float32)
            col_mask[torch.tensor(self.aux_attr_held_out_ids,
                                  device=cls_logits_valid.device)] = 0.0
            cls_loss = (sigmoid_focal_loss(
                cls_logits_valid, gt_target, reduction='none'
            ) * col_mask).sum()
        else:
            cls_loss = sigmoid_focal_loss(
                cls_logits_valid, gt_target, reduction='sum')

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

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight

        # ---- Phase 2D ZSL: auxiliary attribute BCE loss ----
        losses_out = {'cls_loss': cls_loss, 'reg_loss': reg_loss}




        # ---- Phase 2D v11: split rotation loss (aerial_twist + floor_turn) ----
        # Same per-segment pooling as v9, but each head trains only on segments
        # whose class is in its rotation type (aerial vs floor). Classes outside
        # the head's type are skipped — their tokens contribute zero gradient.
        if (
            self.aerial_twist_head is not None
            and self.floor_turn_head is not None
            and self._aerial_twist_logits is not None
            and self._floor_turn_logits is not None
            and gt_segments is not None
        ):
            r_aer = self._aerial_twist_logits
            r_flr = self._floor_turn_logits
            B, T0 = r_aer.shape
            aer_preds, aer_gts = [], []
            flr_preds, flr_gts = [], []
            for b in range(B):
                segs_b = gt_segments[b]
                lbls_b = gt_labels[b]
                if segs_b.numel() == 0:
                    continue
                for i in range(segs_b.shape[0]):
                    s_f = segs_b[i, 0].item()
                    e_f = segs_b[i, 1].item()
                    tok_lo = max(0, int(s_f))
                    tok_hi = min(T0 - 1, int(e_f) + 1)
                    if tok_hi <= tok_lo:
                        continue
                    cls_idx = int(lbls_b[i].item())
                    rot_type = int(self.rotation_type_per_class[cls_idx].item())
                    if rot_type == 0:
                        continue                # no-rotation classes: neither head trains
                    # Strict-ZSL ablation: exclude held-out segments from rotation supervision.
                    if self.strict_zsl and self.aux_attr_held_out_ids and cls_idx in self.aux_attr_held_out_ids:
                        continue
                    gt = self.rotation_count_per_class[cls_idx]
                    if rot_type == 1:           # aerial twist
                        aer_preds.append(r_aer[b, tok_lo:tok_hi].float().mean())
                        aer_gts.append(gt)
                    else:                        # floor turn
                        flr_preds.append(r_flr[b, tok_lo:tok_hi].float().mean())
                        flr_gts.append(gt)
            if aer_preds:
                aerial_loss = F.smooth_l1_loss(
                    torch.stack(aer_preds), torch.stack(aer_gts), beta=0.5, reduction='mean'
                )
            else:
                aerial_loss = r_aer.sum() * 0.0
            if flr_preds:
                floor_loss = F.smooth_l1_loss(
                    torch.stack(flr_preds), torch.stack(flr_gts), beta=0.5, reduction='mean'
                )
            else:
                floor_loss = r_flr.sum() * 0.0
            lambda_aer = self.train_cfg.get('lambda_aerial_twist', 1.0)
            lambda_flr = self.train_cfg.get('lambda_floor_turn', 1.0)
            final_loss = final_loss + lambda_aer * aerial_loss + lambda_flr * floor_loss
            losses_out['aerial_twist_loss'] = aerial_loss.detach()
            losses_out['floor_turn_loss'] = floor_loss.detach()

        # ---- Phase 2D v10: multi-specialist categorical losses (per-token CE) ----
        # For each specialist head, build a per-token target tensor over fpn_lvl0
        # by walking through GT segments and broadcasting the GT class's specialist
        # target across the segment's tokens. Classes without a valid target are
        # skipped (target stays -100 = ignore_index). Held-out segments ARE included
        # — specialist targets are class-side metadata (many-to-one projections), so
        # they don't leak class identity, same reasoning as rotation.
        if gt_segments is not None:
            specialist_configs = [
                ('body_shape',          self._body_shape_logits,          'lambda_body_shape',          0.5),
                ('direction',           self._direction_logits,           'lambda_direction',           0.5),
                ('multiplicity',        self._multiplicity_logits,        'lambda_multiplicity',        0.5),
                ('named_element',       self._named_element_logits,       'lambda_named_element',       0.5),
                # v11
                ('motion_category',     self._motion_category_logits,     'lambda_motion_category',     0.5),
                ('leg_configuration',   self._leg_configuration_logits,   'lambda_leg_configuration',   0.5),
                ('inverted_phase',      self._inverted_phase_logits,      'lambda_inverted_phase',      0.5),
                ('hands_on_apparatus',  self._hands_on_apparatus_logits,  'lambda_hands_on_apparatus',  0.5),
            ]
            for name, logits, lam_key, lam_default in specialist_configs:
                if logits is None:
                    continue
                tgt_per_class = getattr(self, f'{name}_target_per_class')
                val_per_class = getattr(self, f'{name}_valid_per_class')
                B, C, T0 = logits.shape
                # Build per-token target tensor; -100 = ignore. Also track per-token
                # class for v17 contrastive loss.
                tgt_token = torch.full(
                    (B, T0), -100, dtype=torch.long, device=logits.device,
                )
                cls_token = torch.full(
                    (B, T0), -100, dtype=torch.long, device=logits.device,
                )
                covered = torch.zeros((B, T0), dtype=torch.bool, device=logits.device)
                for b in range(B):
                    segs_b = gt_segments[b]
                    lbls_b = gt_labels[b]
                    if segs_b.numel() == 0:
                        continue
                    for i in range(segs_b.shape[0]):
                        cls_idx = int(lbls_b[i].item())
                        # mark ALL GT spans (incl. held-out / invalid-target) so
                        # background sampling below never treats an unlabeled
                        # action as background.
                        _lo = max(0, int(segs_b[i, 0].item()))
                        _hi = min(T0 - 1, int(segs_b[i, 1].item()) + 1)
                        if _hi > _lo:
                            covered[b, _lo:_hi] = True
                        if not bool(val_per_class[cls_idx].item()):
                            continue
                        # v17 strict-ZSL ablation: exclude held-out class segments
                        # from specialist supervision entirely (held-out class clips
                        # do not contribute attribute targets either).
                        if self.strict_zsl and self.aux_attr_held_out_ids and cls_idx in self.aux_attr_held_out_ids:
                            continue
                        target = int(tgt_per_class[cls_idx].item())
                        tok_lo = max(0, int(segs_b[i, 0].item()))
                        tok_hi = min(T0 - 1, int(segs_b[i, 1].item()) + 1)
                        if tok_hi <= tok_lo:
                            continue
                        tgt_token[b, tok_lo:tok_hi] = target
                        cls_token[b, tok_lo:tok_hi] = cls_idx
                # background-aware: sample ~n_pos background tokens (valid,
                # outside every GT span) with target = the extra 'none' class.
                if bool(self.train_cfg.get('background_aware_specialists', False)):
                    lvl0_valid = fpn_masks[0]
                    if lvl0_valid.dim() > 2:
                        lvl0_valid = lvl0_valid.squeeze(1)
                    bg_pool = lvl0_valid.bool() & (~covered) & (tgt_token < 0)
                    n_pos_bg = int((tgt_token >= 0).sum().item())
                    bg_idx = bg_pool.view(-1).nonzero(as_tuple=False).squeeze(-1)
                    if n_pos_bg > 0 and bg_idx.numel() > 0:
                        take = min(n_pos_bg, bg_idx.numel())
                        sel = bg_idx[torch.randperm(bg_idx.numel(), device=bg_idx.device)[:take]]
                        tgt_flat_view = tgt_token.view(-1)
                        tgt_flat_view[sel] = C - 1   # 'none' = last channel
                # Compute CE on valid tokens only.
                mask = (tgt_token >= 0)
                n_valid = int(mask.sum().item())
                if n_valid > 0:
                    preds_flat = logits.permute(0, 2, 1)[mask].float()       # (M, C)
                    tgts_flat = tgt_token[mask]                                # (M,)
                    cls_flat = cls_token[mask]                                 # (M,)
                    spec_loss = F.cross_entropy(
                        preds_flat, tgts_flat, reduction='mean',
                        label_smoothing=float(self.train_cfg.get(
                            'specialist_label_smoothing', 0.0)))
                else:
                    spec_loss = logits.sum() * 0.0
                lam = self.train_cfg.get(lam_key, lam_default)
                final_loss = final_loss + lam * spec_loss
                losses_out[f'{name}_loss'] = spec_loss.detach()

                # ---- v17 attribute-axis contrastive (margin) loss ----
                # For each positive token at class c, log_softmax must rank c's target
                # higher than any within-phrase sibling target by margin m. Hard pairs
                # only (within-phrase, different target), not all pairs.
                if self.use_attr_axis_contrast and n_valid > 0:
                    sib_attr = f'{name}_sibling_target_mask'
                    if hasattr(self, sib_attr):
                        sib_mask_per_class = getattr(self, sib_attr).to(preds_flat.device)
                        # background tokens (cls_flat == -100) have no siblings
                        cls_safe = cls_flat.clamp(min=0)
                        sib_mask_per_token = sib_mask_per_class[cls_safe]       # (M, C_attr)
                        sib_mask_per_token = sib_mask_per_token * (cls_flat >= 0).view(-1, 1)
                        if sib_mask_per_token.shape[1] < preds_flat.shape[1]:
                            sib_mask_per_token = F.pad(
                                sib_mask_per_token,
                                (0, preds_flat.shape[1] - sib_mask_per_token.shape[1]))
                        log_p = F.log_softmax(preds_flat, dim=-1)               # (M, C)
                        log_p_correct = log_p.gather(
                            -1, tgts_flat.view(-1, 1)
                        ).squeeze(-1)                                            # (M,)
                        diff = log_p - log_p_correct.view(-1, 1) + self.attr_axis_contrast_margin
                        per_token_loss = (sib_mask_per_token * F.relu(diff)).sum(dim=-1)
                        norm = sib_mask_per_token.sum(dim=-1).clamp_min(1.0)
                        contrast_loss = (per_token_loss / norm).mean()
                    else:
                        contrast_loss = logits.sum() * 0.0
                    lam_contrast = float(
                        self.train_cfg.get('lambda_attr_axis_contrast', 0.5)
                    )
                    final_loss = final_loss + lam_contrast * contrast_loss
                    losses_out[f'{name}_contrast_loss'] = contrast_loss.detach()


        # ———— t3 system 2 (Ti-FAD): foreground head loss ————
        # Class-agnostic foreground focal loss (fork pattern: fg_loss added
        # directly to the total). Foreground is coarser than action identity,
        # so ALL positives contribute — including held-out segments —
        # consistent with the regression loss under revert_num_pos_v4_style
        # (leakage-aware protocol: only ACTION-level supervision is masked).
        if self._fg_logits is not None:
            fg_logits_all = torch.cat(self._fg_logits, dim=1).squeeze(-1)  # (B, FT)
            fg_loss = sigmoid_focal_loss(
                fg_logits_all[valid_mask],
                pos_mask[valid_mask].float(),
                reduction='sum',
            ) / self.loss_normalizer
            lambda_fg = float(self.train_cfg.get('lambda_fg', 1.0))
            final_loss = final_loss + lambda_fg * fg_loss
            losses_out['fg_loss'] = fg_loss.detach()

        # ———— t3 champion (AS): 14-way phrase-level focal CE + Kendall UW ————
        # Exact tifad FAMCE-UW pattern: phrase logits = logsumexp over member
        # class logit columns; phrase targets = amax over member columns of
        # the PRE-masking one-hot (so held-out segments DO carry their parent
        # phrase target — parent-level supervision is allowed by the
        # leakage-aware protocol); focal over ALL valid tokens, normalized by
        # loss_normalizer; combined with the main loss via 2-way Kendall
        # uncertainty weighting exp(-s0)*main + s0/2 + exp(-s1)*as + s1/2
        # (s init 0 => plain sum at step 0).
        if self.use_activity_align_loss and self.as_uw_logvar is not None:
            _lg = torch.cat(out_cls_logits, dim=1)[as_valid_mask]  # (N, C)
            _tg = gt_cls[as_valid_mask]                            # (N, C) pre-smoothing
            lambda_as = float(self.train_cfg.get('lambda_activity_align', 0.3))
            _s = self.as_uw_logvar
            total = torch.exp(-_s[0]) * final_loss + 0.5 * _s[0]
            losses_out['as_uw_s_main'] = _s[0].detach()
            # exclude / background protocols: the ancestor logit and target are
            # formed over the SEEN member columns only. With held-out columns
            # inside the log-sum-exp, an unseen class receives positive
            # gradient at every span of its seen siblings (its softmax share
            # of the ancestor loss) and is pushed down elsewhere, i.e. it is
            # trained into a sibling detector (FineGym l_d=2 diagnostic,
            # 2026-09-23: FX turns fired on every FX element). 'keep' is left
            # as it was (the paper's l_d=3 runs).
            _hom = str(self.train_cfg.get('held_out_mode', 'keep'))
            _held_cols = set(int(h) for h in self.aux_attr_held_out_ids) \
                if _hom in ('exclude', 'background') else set()
            for li, (lvl, members) in enumerate(self._as_levels, start=1):
                _members = [m.to(_lg.device) for m in members]
                if _held_cols:
                    _members = [m[torch.tensor([int(c) not in _held_cols for c in m.tolist()],
                                               device=m.device, dtype=torch.bool)]
                                for m in _members]
                    _members = [m for m in _members if m.numel() > 0]
                as_logits = torch.stack(
                    [torch.logsumexp(_lg[:, m], dim=-1) for m in _members], dim=-1)
                as_targets = torch.stack(
                    [_tg[:, m].amax(dim=-1) for m in _members], dim=-1)
                as_loss = lambda_as * sigmoid_focal_loss(
                    as_logits, as_targets.float(), reduction='sum'
                ) / self.loss_normalizer
                total = total + torch.exp(-_s[li]) * as_loss + 0.5 * _s[li]
                losses_out[f'as_loss_{lvl}'] = as_loss.detach()
                losses_out[f'as_uw_s_{lvl}'] = _s[li].detach()
            final_loss = total

        losses_out['final_loss'] = final_loss
        return losses_out


    def cascaded_hierarchy_losses(
        self,
        fpn_masks,         # tuple/list of (B, 1, T_l) bool — original FPN masks before squeeze
        points,            # tuple of (T_l, 4) per-FPN-level point grid
        gt_segments,       # list of len B; each (Na_i, 2) action segments per video
        gt_labels,         # list of len B; each (Na_i,) action class labels per video
        video_list,        # raw input list — to extract phrase + activity GT from data_dict
        out_offsets,       # tuple of (B, T_l, 2) action regression outputs (post-permute)
        current_epoch=0,
    ):
        """Compute cascaded phrase + activity + structural-consistency losses.

        Returns a dict with keys phrase_cls_loss, phrase_reg_loss, activity_cls_loss,
        activity_reg_loss, consistency_loss, cascaded_total — to be added to the
        outer losses dict in forward().
        """
        from .losses import sigmoid_focal_loss, ctr_diou_loss_1d
        device = self.device

        # GT from video_list (set by FineGymSlideDataset.__getitem__).
        gt_phrase_segments = [item['phrase_segments'].to(device) for item in video_list]
        gt_phrase_labels = [item['phrase_labels'].to(device) for item in video_list]
        # Activity GT only consumed when activity head is enabled (3-level setup).
        if self.activity_cascade_head is not None:
            gt_activity_segments = [item['activity_segments'].to(device) for item in video_list]
            gt_activity_labels = [item['activity_labels'].to(device) for item in video_list]

        # ============== Phrase target assignment ==============
        phrase_logits = self._cascade_phrase_logits      # list of (B, T_l, num_phrases)
        phrase_offsets = self._cascade_phrase_offsets    # list of (B, T_l, 2)
        phrase_points = [points[l] for l in self.cascade_phrase_levels]
        concat_phrase_points = torch.cat(phrase_points, dim=0)

        gt_phrase_cls_per_video = []
        gt_phrase_offs_per_video = []
        for vidx in range(len(gt_phrase_segments)):
            cls_t, offs_t, _ = self.label_points_single_video(
                concat_phrase_points,
                gt_phrase_segments[vidx],
                gt_phrase_labels[vidx],
                num_classes=self.num_phrases,
            )
            gt_phrase_cls_per_video.append(cls_t)
            gt_phrase_offs_per_video.append(offs_t)
        # Stack across videos: each is (sum_T_l, num_phrases) or (sum_T_l, 2).
        gt_phrase_cls = torch.stack(gt_phrase_cls_per_video, dim=0)  # (B, sum_T_l, num_phrases)
        gt_phrase_offs = torch.stack(gt_phrase_offs_per_video, dim=0) # (B, sum_T_l, 2)

        # Concatenate model outputs over levels: each (B, T_l, C) → (B, sum_T_l, C)
        phrase_logits_cat = torch.cat(phrase_logits, dim=1)
        phrase_offsets_cat = torch.cat(phrase_offsets, dim=1)

        # Build valid mask for phrase levels: (B, sum_T_l)
        phrase_masks_subset = [fpn_masks[l].squeeze(1) for l in self.cascade_phrase_levels]
        phrase_valid_mask = torch.cat(phrase_masks_subset, dim=1)  # (B, sum_T_l)

        phrase_pos_mask = (gt_phrase_cls.sum(dim=-1) > 0) & phrase_valid_mask
        # Phrase classification loss (focal, only over valid tokens).
        phrase_cls_loss = sigmoid_focal_loss(
            phrase_logits_cat[phrase_valid_mask],
            gt_phrase_cls[phrase_valid_mask].float(),
            reduction='sum',
        ) / max(phrase_pos_mask.sum().item(), 1.0)
        # Phrase regression DIoU loss only at positive tokens.
        if phrase_pos_mask.sum() > 0:
            phrase_reg_loss = ctr_diou_loss_1d(
                phrase_offsets_cat[phrase_pos_mask],
                gt_phrase_offs[phrase_pos_mask],
                reduction='sum',
        ) / max(phrase_pos_mask.sum().item(), 1.0)
        else:
            phrase_reg_loss = phrase_offsets_cat.sum() * 0.0

        # ============== Activity target assignment ==============
        # Skipped entirely when activity head is disabled (2-level setup).
        if self.activity_cascade_head is not None:
            activity_level = self.cascade_activity_levels[0]
            activity_logits = self._cascade_activity_logits       # (B, T_3, num_activities)
            activity_offsets = self._cascade_activity_offsets     # (B, T_3, 2)
            activity_points = points[activity_level]              # (T_3, 4)

            gt_activity_cls_per_video = []
            gt_activity_offs_per_video = []
            for vidx in range(len(gt_activity_segments)):
                cls_t, offs_t, _ = self.label_points_single_video(
                    activity_points,
                    gt_activity_segments[vidx],
                    gt_activity_labels[vidx],
                    num_classes=self.num_activities,
                )
                gt_activity_cls_per_video.append(cls_t)
                gt_activity_offs_per_video.append(offs_t)
            gt_activity_cls = torch.stack(gt_activity_cls_per_video, dim=0)
            gt_activity_offs = torch.stack(gt_activity_offs_per_video, dim=0)

            activity_valid_mask = fpn_masks[activity_level].squeeze(1)
            activity_pos_mask = (gt_activity_cls.sum(dim=-1) > 0) & activity_valid_mask

            activity_cls_loss = sigmoid_focal_loss(
                activity_logits[activity_valid_mask],
                gt_activity_cls[activity_valid_mask].float(),
                reduction='sum',
            ) / max(activity_pos_mask.sum().item(), 1.0)

            if activity_pos_mask.sum() > 0:
                activity_reg_loss = ctr_diou_loss_1d(
                    activity_offsets[activity_pos_mask],
                    gt_activity_offs[activity_pos_mask],
                    reduction='sum',
                ) / max(activity_pos_mask.sum().item(), 1.0)
            else:
                activity_reg_loss = activity_offsets.sum() * 0.0
        else:
            activity_cls_loss = phrase_offsets_cat.sum() * 0.0
            activity_reg_loss = phrase_offsets_cat.sum() * 0.0

        # ============== Structural consistency loss ==============
        # Action positives at the phrase levels (need to recompute action target
        # assignment on the phrase-level points to align with phrase_offsets).
        warm = self.train_cfg.get('consistency_warmup_epochs', 2)
        ramp = max(1, self.train_cfg.get('consistency_ramp_epochs', 1))
        if current_epoch < warm:
            consist_warmup_factor = 0.0
        elif current_epoch < warm + ramp:
            consist_warmup_factor = (current_epoch - warm + 1) / ramp
        else:
            consist_warmup_factor = 1.0

        # Always compute the consistency value (so it's visible in the log even
        # during the warmup epochs); only the *contribution* to final_loss is
        # gated by consist_warmup_factor below.
        consist_loss = phrase_offsets_cat.sum() * 0.0
        if phrase_pos_mask.sum() > 0:
            # Action offsets at the phrase levels (concatenate from out_offsets list).
            action_offsets_at_phrase_levels = torch.cat(
                [out_offsets[l] for l in self.cascade_phrase_levels], dim=1
            )  # (B, sum_T_l, 2)

            # "Action contains this token" — DON'T use label_points_single_video here:
            # that filters by FPN regression-range, and short FineGym actions (1-3 tokens)
            # never qualify as positives at the phrase levels' reg_range [4, +inf]. We only
            # need pure interval containment: is the phrase-level token inside any action?
            # The action reg head still produces valid predictions at these tokens because
            # it runs on all FPN levels.
            phrase_centers = concat_phrase_points[:, 0]   # (sum_T_l,)
            action_contains_per_video = []
            for vidx in range(len(gt_segments)):
                segs = gt_segments[vidx]   # (Na, 2)
                if segs.shape[0] == 0:
                    action_contains_per_video.append(
                        torch.zeros(phrase_centers.shape[0], dtype=torch.bool, device=phrase_centers.device)
                    )
                    continue
                inside = (phrase_centers[:, None] >= segs[None, :, 0]) & \
                         (phrase_centers[:, None] <= segs[None, :, 1])
                action_contains_per_video.append(inside.any(dim=-1))
            action_pos_mask = torch.stack(action_contains_per_video, dim=0) & phrase_valid_mask

            both_pos = action_pos_mask & phrase_pos_mask  # (B, sum_T_l)
            if both_pos.sum() > 0:
                a = action_offsets_at_phrase_levels[both_pos]   # (P, 2)
                p = phrase_offsets_cat[both_pos]                # (P, 2)
                consist_loss = self.structural_consistency_loss(
                    a, p, positive_mask=torch.ones(a.shape[0], dtype=torch.bool, device=a.device),
                )

        # ============== Weighted sum ==============
        lambda_phrase_cls = self.train_cfg.get('lambda_phrase_cls', 0.5)
        lambda_phrase_reg = self.train_cfg.get('lambda_phrase_reg', 0.5)
        lambda_activity_cls = self.train_cfg.get('lambda_activity_cls', 0.25)
        lambda_activity_reg = self.train_cfg.get('lambda_activity_reg', 0.25)
        lambda_consist = self.train_cfg.get('lambda_structural_consistency', 0.25)

        cascaded_total = (
            lambda_phrase_cls * phrase_cls_loss
            + lambda_phrase_reg * phrase_reg_loss
            + lambda_activity_cls * activity_cls_loss
            + lambda_activity_reg * activity_reg_loss
            + lambda_consist * consist_warmup_factor * consist_loss
        )

        return {
            'phrase_cls_loss': phrase_cls_loss.detach(),
            'phrase_reg_loss': phrase_reg_loss.detach(),
            'activity_cls_loss': activity_cls_loss.detach(),
            'activity_reg_loss': activity_reg_loss.detach(),
            'consistency_loss': consist_loss.detach(),
            'cascaded_total': cascaded_total,
        }


    @torch.no_grad()
    def inference(
        self,
        video_list,
        points, fpn_masks,
        out_cls_logits, out_offsets,
        out_phrase_logits=None, out_activity_logits=None
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
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
            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid
            )

            # ———— t3 champion (DP): eval-time duration prior ————
            # score *= exp(-gamma/2 * z^2), z = (log dur_sec - mu_c)/sigma_c,
            # applied PRE-NMS on the candidate set (segments still on the
            # feature grid here; stride/fps convert durations to seconds).
            if self.duration_prior is not None and \
                    results_per_vid['segments'].numel() > 0:
                _segs = results_per_vid['segments']
                _dur_sec = (_segs[:, 1] - _segs[:, 0]).float() * \
                    float(stride) / float(fps)
                results_per_vid['scores'] = self.duration_prior.rescore(
                    results_per_vid['scores'],
                    results_per_vid['labels'],
                    _dur_sec,
                )

            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
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
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # loop over fpn levels
        for cls_i, offsets_i, pts_i, mask_i in zip(
                out_cls_logits, out_offsets, points, fpn_masks
            ):
            # sigmoid normalization for output logits
            pred_prob = (cls_i.float().sigmoid() * mask_i.unsqueeze(-1).float()).flatten()

            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            pt_idxs =  torch.div(
                topk_idxs, self.num_classes, rounding_mode='floor'
            )
            cls_idxs = torch.fmod(topk_idxs, self.num_classes)

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
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {'segments' : segs_all,
                   'scores'   : scores_all,
                   'labels'   : cls_idxs_all}

        return results

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
            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
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
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                # truncate all boundaries within [0, duration]
                segs[segs<=0.0] *= 0.0
                segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen
            
            # 4: repack the results
            result_dict = {
                'video_id' : vidx,
                'segments' : segs,
                'scores'   : scores,
                'labels'   : labels,
            }
            processed_results.append(result_dict)

        return processed_results