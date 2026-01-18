# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
# from transformers import SwinModel
from ..my_model.vision_transformer import vit_b_32, ViT_B_32_Weights, vit_l_32, ViT_L_32_Weights
import numpy as np
from transformers import VivitConfig, VivitModel
from transformers.models.vivit.modeling_vivit import VivitTubeletEmbeddings

def get_encoder(name):
    if name == 'vit_b_32':
        model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
    if name == 'vit_l_32':
        model = vit_l_32(weights=ViT_L_32_Weights.IMAGENET1K_V1)
    return model


import torch
import torch.nn as nn
from timm.models.vision_transformer import vit_large_patch16_224

class ViViTEncoder(nn.Module):
    def __init__(self, pretrained=True, num_frames=32, num_classes=157):
        super(ViViTEncoder, self).__init__()
        # Load pre-trained ViViT model
        self.vivit = vit_large_patch16_224(pretrained=pretrained)
        self.vivit.head = nn.Identity()  # Remove classification head

        # Extend positional embeddings to handle temporal dimensions
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, self.vivit.embed_dim))
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)

    def forward(self, x):
        # Input shape: (batch_size, num_frames, channels, height, width)
        b, t, c, h, w = x.size()
        x = x.view(-1, c, h, w)  # Merge batch and temporal dimensions
        x = self.vivit.patch_embed(x)  # Extract patch embeddings
        x = x.view(b, t, -1) + self.temporal_embedding  # Add temporal embeddings
        return self.vivit.forward_features(x)  # Return ViViT's encoded features


class VivitPrompt(nn.Module):
    def __init__(self, num_frames, id2class, tubelet_size):
        super(VivitPrompt, self).__init__()
        self.num_frames = num_frames
        # Initializing a ViViT google/vivit-b-16x2-kinetics400 style configuration
        self.config = VivitConfig(num_frames=self.num_frames, tubelet_size=tubelet_size, id2label = id2class)
        # config.id2label = id2class

        # Initializing a model (with random weights) from the google/vivit-b-16x2-kinetics400 style configuration
        # self.vivit = VivitModel(config).cuda()
        # self.vivit.embeddings.patch_embeddings.projection = nn.Conv3d(
        #     config.num_channels, config.hidden_size, kernel_size=config.tubelet_size, stride=config.tubelet_size
        # )
        self.vivit = VivitModel.from_pretrained("jegormeister/vivit-b-16x2-kinetics400")
        self.vivit.embeddings.patch_embeddings = VivitTubeletEmbeddings(self.config)
        self.vivit.embeddings.position_embeddings = nn.Parameter(torch.zeros(1, self.vivit.embeddings.patch_embeddings.num_patches + 1, self.config.hidden_size))

        # self.vivit = VivitModel.from_pretrained("jegormeister/vivit-b-16x2-kinetics400")#.cuda()
        # self.vivit.embeddings.position_embeddings = nn.Parameter(torch.zeros(1, 6273, self.config.hidden_size))
        # self.vivit = self.vivit.cuda()

        for name, param in self.vivit.named_parameters():
            if 'embeddings' not in name:
                param.requires_grad = False

        # for name, param in self.vivit.named_parameters():
        #     if param.requires_grad:
        #         print(name)


    def forward(self, x, mask=None):
        # ViViT model forward pass
        #print(x.shape) 
        # x: torch.Size([1, 2, 3, 224, 224])
        B = len(x)
        outputs = self.vivit(pixel_values=x)

        # Extract the hidden states
        # last_hidden_state Size([1, 3137, 768]); pooler_output: Size([1, 768])
        # hidden_states = outputs.pooler_output

        # # Reshape the hidden states to separate frames
        hidden_states = outputs.last_hidden_state[:,:-1].view(B, self.num_frames, -1, self.config.hidden_size)[:,:,0].permute(0,2,1) # [4, 1568, 768] --> [4, 32, 49, 768] --> [4, 32, 768]

        # compute the mask
        # masks = []
        # for i in hidden_states:
        #     # downsample the mask using nearest neighbor
        #     out_mask = F.interpolate(
        #         mask.to(x.dtype), size=i.size(-1), mode='nearest'
        #     )
        #     masks.append(out_mask.bool())

        return hidden_states, mask  # Shape: (batch_size, hidden_dim, num_frames)


