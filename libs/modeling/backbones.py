import torch
from torch import nn
from torch.nn import functional as F

from .models import register_backbone
from .blocks import (get_sinusoid_encoding, TransformerBlock, MaskedConv1D,
                     ConvBlock, LayerNorm)


from timm.models.vision_transformer import vit_large_patch16_224
from .models import make_backbone
from transformers import VivitConfig, VivitModel, TimesformerConfig, TimesformerModel, VideoMAEConfig, VideoMAEModel, CLIPVisionConfig, CLIPVisionModel
from ..my_model.vision_transformer import vit_b_16, ViT_B_16_Weights, vit_b_32, ViT_B_32_Weights
import numpy as np

# class ViTEncoder(nn.Module):
#     def __init__(self, pretrained=True, num_frames=32, embd_dim=768):
#         super(ViTEncoder, self).__init__()
#         # Load pre-trained ViT model
#         # self.vit = vit_large_patch16_224(pretrained=pretrained)
#         self.vit = vit_l_32(weights=ViT_L_32_Weights.IMAGENET1K_V1)
#         self.vit.head = nn.Identity()  # Remove classification head

#         # Temporal positional embedding for vit
#         self.temporal_embedding = nn.Parameter(torch.zeros(1, self.vit.patch_embed.num_patches, self.vit.embed_dim, device="cuda"))
#         nn.init.trunc_normal_(self.temporal_embedding, std=0.02)

#         # Linear projection to match decoder input dimension (if needed)
#         # self.vit.head = nn.Linear(self.vit.embed_dim, embd_dim)
#         if self.vit.embed_dim != embd_dim:
#             self.projection = nn.Linear(self.vit.embed_dim, embd_dim)
#         else:
#             self.projection = nn.Identity()

#     def forward(self, x):
#         # Input shape: (batch_size, num_frames, channels, height, width)
#         b, t, c, h, w = x.size()
#         x = x.contiguous().view(-1, c, h, w)  # Merge batch and temporal dimensions

#         if False:
#             x = self.vit.patch_embed(x)  # Extract patch embeddings
#             x = x + self.temporal_embedding  # Add temporal embeddings
#         else:
#             # Forward through vit encoder
#             x = self.vit.forward_features(x)
#         x = x[:, self.vit.num_prefix_tokens:].mean(dim=1) # Average over all patches
#         x = self.projection(x)
#         return x.view(b, t, -1).permute(0,2,1)  # Project features to match decoder input dimension
    

class ViTPrompt(nn.Module):
    def __init__(self, prompt_len=10, embd_dim=2048):
        super(ViTPrompt, self).__init__()

        # self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
        # Use only the first 4 layers of the pre-trained ViT model
        # self.vit.encoder.layers = self.vit.encoder.layers[:4]
        
        # Reinitialize the head with a new layer
        self.vit.heads[0] = nn.Identity()

        for param in self.vit.parameters():
            param.requires_grad = False
        
        hidden_dim = self.vit.hidden_dim
        num_layers = len(self.vit.encoder.layers) 
        prompt_dim = 768 # 1024
        val = np.sqrt(6. / float(hidden_dim + prompt_dim))
        self.prompts = nn.Parameter(torch.zeros(1, num_layers, prompt_len, hidden_dim), 
                                    requires_grad=True)

        nn.init.uniform_(self.prompts, -val, val)
        
        # Linear projection (if needed)
        if hidden_dim != embd_dim:
            self.projection = nn.Linear(hidden_dim, embd_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x):
        # Input shape: (batch_size, num_frames, channels, height, width)
        b, t, c, h, w = x.size()
        x = x.contiguous().view(-1, c, h, w)  # Merge batch and temporal dimensions
        y = self.vit(x, prompts=self.prompts)

        y = self.projection(y.view(b, t, -1))

        return y.permute(0,2,1)  # Project features to match decoder input dimension
    
@register_backbone("ActionFormerWithViT")
class ActionFormerWithViT(nn.Module):
    def __init__(
        self,
        input_dim,                 # input feature dimension
        embd_dim,                  # embedding dimension (after convolution)
        n_head,                    # number of head for self-attention in transformers
        embd_kernel_size,          # conv kernel size of the embedding network
        max_seq_len,               # max sequence length
        backbone_arch = (2, 2, 5), # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*6,     # size of local window for mha
        scale_factor = 2,          # dowsampling rate for the branch
        embd_with_ln = False,      # if to attach layernorm after conv
        attn_pdrop = 0.0,          # dropout rate for the attention map
        proj_pdrop = 0.0,          # dropout rate for the projection / MLP
        path_pdrop = 0.0,          # droput rate for drop path
        use_abs_pe = False,        # use absolute position embedding
        use_rel_pe = False,        # use relative position embedding
        pretrained=True
    ):
        super(ActionFormerWithViT, self).__init__()

        # ViT Encoder
        self.encoder = ViTPrompt(prompt_len=10, embd_dim=input_dim)
        # self.encoder = ViTEncoder(pretrained=pretrained, num_frames=max_seq_len, embd_dim=input_dim)

        # Decoder Initialization using make_backbone
        self.backbone = make_backbone(
            'convTransformer',
            **{
                'n_in' : input_dim,
                'n_embd' : embd_dim,
                'n_head': n_head,
                'n_embd_ks': embd_kernel_size,
                'max_len': max_seq_len,
                'arch' : backbone_arch,
                'mha_win_size': mha_win_size,
                'scale_factor' : scale_factor,
                'with_ln' : embd_with_ln,
                'attn_pdrop' : attn_pdrop,
                'proj_pdrop' : proj_pdrop,
                'path_pdrop' : path_pdrop,
                'use_abs_pe' : use_abs_pe,
                'use_rel_pe' : use_rel_pe
            }
        )

    def forward(self, x, mask):
        # Extract features using ViViT encoder
        features = self.encoder(x)  # Shape: (batch_size, num_frames, embd_dim)

        # Pass features to the decoder (backbone)
        output = self.backbone(features, mask)
        return output


class ViViTEncoder(nn.Module):
    def __init__(self, pretrained=True, num_frames=16, out_dim=768, image_size=224, tubelet_size=[1, 16, 16]):
        super(ViViTEncoder, self).__init__()

        # Configure ViViT to reduce memory usage
        self.config = VivitConfig(
            num_frames=num_frames,
            image_size=image_size,
            tubelet_size=tubelet_size,
            # hidden_size=embd_dim,
            num_hidden_layers=3,
            num_attention_heads=3,  # Reduce number of heads if necessary
        )
        self.vivit = VivitModel.from_pretrained(
            "google/vivit-b-16x2-kinetics400",
            config=self.config,
            ignore_mismatched_sizes=True
        )

        # Freeze all layers except the last transformer block
        for name, param in self.vivit.named_parameters():
            if 'embeddings' in name or "layer.2" in name:  # Fine-tune only the last transformer block
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Linear projection (if needed)
        if self.config.hidden_size != out_dim:
            self.projection = nn.Linear(self.config.hidden_size, out_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        # Input shape: (batch_size, num_frames, channels, height, width)
        b, t, c, h, w = x.size()

        # Pass through ViViT
        x = self.vivit(x)
        last_hidden_state = x['last_hidden_state'][:, :-1]

        # Extract features for each frame
        num_patches_per_frame = last_hidden_state.size(1) // t
        last_hidden_state = last_hidden_state.view(b, t, num_patches_per_frame, -1)  # Shape: (batch_size, num_frames, num_patches_per_frame, embd_dim)

        # Aggregate patch features to get a single feature vector per frame
        frame_features = last_hidden_state.mean(dim=2)  # Shape: (batch_size, num_frames, embd_dim)

        frame_features = self.projection(frame_features) # Project features to match decoder input dimension
        return frame_features.permute(0,2,1)


'''
from transformers import VivitConfig, VivitModel
from torchsummary import summary
vivit = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
summary(vivit, (2, 2, 3, 224, 224))
'''

@register_backbone("ActionFormerWithViViT")
class ActionFormerWithViViT(nn.Module):
    def __init__(
        self,
        input_dim,                 # input feature dimension
        embd_dim,                  # embedding dimension (after convolution)
        n_head,                    # number of head for self-attention in transformers
        embd_kernel_size,          # conv kernel size of the embedding network
        max_seq_len,               # max sequence length
        backbone_arch = (2, 2, 5), # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*6,     # size of local window for mha
        scale_factor = 2,          # dowsampling rate for the branch
        embd_with_ln = False,      # if to attach layernorm after conv
        attn_pdrop = 0.0,          # dropout rate for the attention map
        proj_pdrop = 0.0,          # dropout rate for the projection / MLP
        path_pdrop = 0.0,          # droput rate for drop path
        use_abs_pe = False,        # use absolute position embedding
        use_rel_pe = False,        # use relative position embedding
        pretrained=True
    ):
        super(ActionFormerWithViViT, self).__init__()

        # ViViT Encoder
        self.encoder = ViViTEncoder(pretrained=pretrained, num_frames=max_seq_len, out_dim=input_dim, image_size=128)

        # Decoder Initialization using make_backbone
        self.backbone = make_backbone(
            'convTransformer',
            **{
                'n_in' : input_dim,
                'n_embd' : embd_dim,
                'n_head': n_head,
                'n_embd_ks': embd_kernel_size,
                'max_len': max_seq_len,
                'arch' : backbone_arch,
                'mha_win_size': mha_win_size,
                'scale_factor' : scale_factor,
                'with_ln' : embd_with_ln,
                'attn_pdrop' : attn_pdrop,
                'proj_pdrop' : proj_pdrop,
                'path_pdrop' : path_pdrop,
                'use_abs_pe' : use_abs_pe,
                'use_rel_pe' : use_rel_pe
            }
        )

    def forward(self, x, mask):
        # Extract features using ViViT encoder
        features = self.encoder(x)  # Shape: (batch_size, num_frames, embd_dim)

        # Pass features to the decoder (backbone)
        output = self.backbone(features, mask)
        return output



#########################
# class ViViTPrompt(nn.Module): # get mAP = 10%
#     def __init__(self, pretrained=True, num_frames=16, out_dim=768, image_size=224, tubelet_size=[1, 16, 16], prompt_len=10):
#         super(ViViTPrompt, self).__init__()

#         # Initialize ViViT configuration and model.
#         self.config = VivitConfig(
#             num_frames=num_frames,
#             image_size=image_size,
#             tubelet_size=tubelet_size,
#             num_hidden_layers=2,
#             num_attention_heads=2,
#         )
#         self.vivit = VivitModel.from_pretrained(
#             "google/vivit-b-16x2-kinetics400",
#             config=self.config,
#             ignore_mismatched_sizes=True
#         )

#         # Freeze all original ViViT parameters.
#         # for param in self.vivit.parameters():
#         #     param.requires_grad = False

#         for name, param in self.vivit.named_parameters():
#             if 'embeddings' in name:  # Fine-tune only the embeddings
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False

#         hidden_dim = self.config.hidden_size
#         num_layers = self.config.num_hidden_layers

#         # Create learnable prompt tokens for each transformer layer.
#         # Shape: (1, num_layers, prompt_len, hidden_dim)
#         self.prompts = nn.Parameter(
#             torch.zeros(1, num_layers, prompt_len, hidden_dim),
#             requires_grad=True)
#         # Initialize prompts uniformly.
#         val = np.sqrt(6.0 / (hidden_dim + hidden_dim))
#         nn.init.uniform_(self.prompts, -val, val)

#         # Linear projection layer if output embedding dimension differs.
#         if hidden_dim != out_dim:
#             self.projection = nn.Linear(hidden_dim, out_dim)
#         else:
#             self.projection = nn.Identity()

#         # dropout for regularization.
#         self.prompt_dropouts = [nn.Dropout(0.1) for i in range(num_layers)]
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, x):
#         """
#         Args:
#             x: Input video tensor of shape (batch_size, num_frames, channels, height, width)
#         Returns:
#             Frame-wise features of shape (batch_size, embd_dim, num_frames)
#         """
#         b, t, c, h, w = x.size()

#         # 1. Obtain initial token embeddings.
#         #    This uses the ViViT embeddings module (see modeling_vivit.py).
#         x_embeds = self.vivit.embeddings(x)  # shape: (b, seq_len, hidden_dim)
#         orig_seq_len = x_embeds.size(1)

#         # 2. Iterate over each transformer layer in the encoder,
#         #    injecting prompt tokens at each layer.
#         #    (The ViViT encoder stores its layers in the attribute "layer".)
#         for i, layer in enumerate(self.vivit.encoder.layer):
#             # Get prompt tokens for the current layer (expand to batch size).
#             prompt = self.prompts[:, i, :, :].expand(b, -1, -1)  # (b, prompt_len, hidden_dim)
#             prompt = self.prompt_dropouts[i](prompt)
#             # Concatenate prompt tokens to the current sequence.
#             x_embeds = torch.cat([x_embeds, prompt], dim=1)
#             # Pass through the transformer layer.
#             # Note: The layer returns a tuple; we take the first element.
#             layer_out = layer(x_embeds)[0] if isinstance(layer(x_embeds), tuple) else layer(x_embeds)
#             # Remove prompt tokens after the layer.
#             x_embeds = layer_out[:, :orig_seq_len, :]

#         # 3. Apply final layer normalization.
#         x_embeds = self.vivit.layernorm(x_embeds)

#         # 4. Extract frame-level features.
#         #    Assume the first token is a global classification token.
#         #    Discard it and use the remaining patch tokens.
#         patch_tokens = x_embeds[:, 1:, :]  # shape: (b, num_tokens-1, hidden_dim)
#         num_patch_tokens = patch_tokens.size(1)
#         # Assume tokens are arranged so that (num_tokens - 1) divides evenly by num_frames.
#         patches_per_frame = num_patch_tokens // t
#         # Reshape tokens to (batch_size, num_frames, patches_per_frame, hidden_dim)
#         patch_tokens = patch_tokens.view(b, t, patches_per_frame, -1)
#         # Aggregate patch tokens (e.g. average pooling) to get one feature per frame.
#         frame_features = patch_tokens.mean(dim=2)  # (b, t, hidden_dim)

#         # 5. Optionally project and apply dropout.
#         frame_features = self.projection(frame_features)
#         frame_features = self.dropout(frame_features)

#         # Permute to (batch_size, embd_dim, num_frames) for downstream compatibility.
#         return frame_features.permute(0, 2, 1)


class ViViTPrompt(nn.Module):
    def __init__(self, pretrained=True, num_frames=16, out_dim=768, image_size=224, tubelet_size=[1, 16, 16], prompt_len=1000):
        super(ViViTPrompt, self).__init__()

        # Initialize ViViT configuration and model.
        self.config = VivitConfig(
            num_frames=num_frames,
            image_size=image_size,
            tubelet_size=tubelet_size,
            num_hidden_layers=3,
            num_attention_heads=3,
        )
        self.vivit = VivitModel.from_pretrained(
            "google/vivit-b-16x2-kinetics400",
            config=self.config,
            ignore_mismatched_sizes=True
        )

        # Freeze all original ViViT parameters.
        # for param in self.vivit.parameters():
        #     param.requires_grad = False

        for name, param in self.vivit.named_parameters():
            if 'embeddings' in name:  # Fine-tune only the embeddings
                param.requires_grad = True
            else:
                param.requires_grad = False

        hidden_dim = self.config.hidden_size
        num_layers = self.config.num_hidden_layers

        # Create learnable prompt tokens for each transformer layer.
        # Shape: (1, num_layers, prompt_len, hidden_dim)
        self.prompts = nn.Parameter(
            torch.zeros(1, num_layers, prompt_len, hidden_dim),
            requires_grad=True)
        # Initialize prompts uniformly.
        val = np.sqrt(6.0 / (hidden_dim + hidden_dim))
        nn.init.uniform_(self.prompts, -val, val)

        # Linear projection layer if output embedding dimension differs.
        if hidden_dim != out_dim:
            self.projection = nn.Linear(hidden_dim, out_dim)
        else:
            self.projection = nn.Identity()

        # dropout for regularization.
        self.prompt_dropouts = nn.ModuleList([nn.Dropout(0.1) for _ in range(num_layers)])
        self.dropout = nn.Dropout(0.1)
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Args:
            x: Input video tensor of shape (batch_size, num_frames, channels, height, width)
        Returns:
            Frame-wise features of shape (batch_size, embd_dim, num_frames)
        """
        b, t, c, h, w = x.size()

        # 1. Obtain initial token embeddings.
        #    This uses the ViViT embeddings module (see modeling_vivit.py).
        x_embeds = self.vivit.embeddings(x)  # shape: (b, seq_len, hidden_dim)
        orig_seq_len = x_embeds.size(1)

        # 2. Iterate over each transformer layer in the encoder,
        #    injecting prompt tokens at each layer.
        #    (The ViViT encoder stores its layers in the attribute "layer".)
        for i, layer in enumerate(self.vivit.encoder.layer):
            # Get prompt tokens for the current layer (expand to batch size).
            prompt = self.prompts[:, i, :, :].expand(b, -1, -1)  # (b, prompt_len, hidden_dim)
            prompt = self.prompt_dropouts[i](prompt)
            # Concatenate prompt tokens to the current sequence.
            x_embeds = torch.cat([x_embeds, prompt], dim=1)
            # Pass through the transformer layer.
            # Note: The layer returns a tuple; we take the first element.
            layer_out = layer(x_embeds)[0] if isinstance(layer(x_embeds), tuple) else layer(x_embeds)
            # Remove prompt tokens after the layer.
            x_embeds = layer_out[:, :orig_seq_len, :]

        # 3. Apply final layer normalization.
        x_embeds = self.layernorm(x_embeds)

        # 4. Extract frame-level features.
        #    Assume the first token is a global classification token.
        #    Discard it and use the remaining patch tokens.
        patch_tokens = x_embeds[:, 1:, :]  # shape: (b, num_tokens-1, hidden_dim)
        num_patch_tokens = patch_tokens.size(1)
        # Assume tokens are arranged so that (num_tokens - 1) divides evenly by num_frames.
        patches_per_frame = num_patch_tokens // t
        # Reshape tokens to (batch_size, num_frames, patches_per_frame, hidden_dim)
        patch_tokens = patch_tokens.view(b, t, patches_per_frame, -1)
        # Aggregate patch tokens (e.g. average pooling) to get one feature per frame.
        frame_features = patch_tokens.mean(dim=2)  # (b, t, hidden_dim)

        # 5. Optionally project and apply dropout.
        frame_features = self.projection(frame_features)
        frame_features = self.dropout(frame_features)

        # Permute to (batch_size, embd_dim, num_frames) for downstream compatibility.
        return frame_features.permute(0, 2, 1)

@register_backbone("ActionFormerViViTPrompt")
class ActionFormerViViTPrompt(nn.Module):
    def __init__(
        self,
        input_dim,                 # input feature dimension
        embd_dim,                  # embedding dimension (after convolution)
        n_head,                    # number of head for self-attention in transformers
        embd_kernel_size,          # conv kernel size of the embedding network
        max_seq_len,               # max sequence length
        backbone_arch = (2, 2, 5), # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*6,     # size of local window for mha
        scale_factor = 2,          # dowsampling rate for the branch
        embd_with_ln = False,      # if to attach layernorm after conv
        attn_pdrop = 0.0,          # dropout rate for the attention map
        proj_pdrop = 0.0,          # dropout rate for the projection / MLP
        path_pdrop = 0.0,          # droput rate for drop path
        use_abs_pe = False,        # use absolute position embedding
        use_rel_pe = False,        # use relative position embedding
        pretrained=True
    ):
        super(ActionFormerViViTPrompt, self).__init__()

        # ViViT Prompt Encoder
        self.encoder = ViViTPrompt(pretrained=pretrained, num_frames=max_seq_len, out_dim=input_dim, image_size=128)

        # Decoder Initialization using make_backbone
        self.backbone = make_backbone(
            'convTransformer',
            **{
                'n_in' : input_dim,
                'n_embd' : embd_dim,
                'n_head': n_head,
                'n_embd_ks': embd_kernel_size,
                'max_len': max_seq_len,
                'arch' : backbone_arch,
                'mha_win_size': mha_win_size,
                'scale_factor' : scale_factor,
                'with_ln' : embd_with_ln,
                'attn_pdrop' : attn_pdrop,
                'proj_pdrop' : proj_pdrop,
                'path_pdrop' : path_pdrop,
                'use_abs_pe' : use_abs_pe,
                'use_rel_pe' : use_rel_pe
            }
        )

    def forward(self, x, mask):
        # Extract features using ViViT encoder
        features = self.encoder(x)  # Shape: (batch_size, num_frames, embd_dim)

        # Pass features to the decoder (backbone)
        output = self.backbone(features, mask)
        return output
#########################


class TimesformerEncoder(nn.Module):
    def __init__(self, pretrained=True, num_frames=16, out_dim=768, image_size=224):
        super(TimesformerEncoder, self).__init__()

        self.config = TimesformerConfig(
            num_frames=num_frames,
            image_size=image_size,
            num_hidden_layers=6,
            num_attention_heads=6,  # Reduce number of heads if necessary
        )
        self.Timesformer = TimesformerModel.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            config=self.config,
            ignore_mismatched_sizes=True
        )

        # Freeze all layers except the last transformer block
        for name, param in self.Timesformer.named_parameters():
            if 'embeddings' in name or "layer.5" in name:  # Fine-tune only the last transformer block
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Linear projection (if needed)
        if self.config.hidden_size != out_dim:
            self.projection = nn.Linear(self.config.hidden_size, out_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        # Input shape: (batch_size, num_frames, channels, height, width)
        b, t, c, h, w = x.size()

        # Pass through Timesformer
        x = self.Timesformer(pixel_values=x)
        last_hidden_state = x['last_hidden_state'][:, :-1]

        # Extract features for each frame
        num_patches_per_frame = last_hidden_state.size(1) // t
        last_hidden_state = last_hidden_state.view(b, t, num_patches_per_frame, -1)  # Shape: (batch_size, num_frames, num_patches_per_frame, embd_dim)

        # Aggregate patch features to get a single feature vector per frame
        frame_features = last_hidden_state.mean(dim=2)  # Shape: (batch_size, num_frames, embd_dim)

        frame_features = self.projection(frame_features) # Project features to match decoder input dimension
        return frame_features.permute(0,2,1)

@register_backbone("ActionFormerWithTimesformer")
class ActionFormerWithTimesformer(nn.Module):
    def __init__(
        self,
        input_dim,                 # input feature dimension
        embd_dim,                  # embedding dimension (after convolution)
        n_head,                    # number of head for self-attention in transformers
        embd_kernel_size,          # conv kernel size of the embedding network
        max_seq_len,               # max sequence length
        backbone_arch = (2, 2, 5), # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*6,     # size of local window for mha
        scale_factor = 2,          # dowsampling rate for the branch
        embd_with_ln = False,      # if to attach layernorm after conv
        attn_pdrop = 0.0,          # dropout rate for the attention map
        proj_pdrop = 0.0,          # dropout rate for the projection / MLP
        path_pdrop = 0.0,          # droput rate for drop path
        use_abs_pe = False,        # use absolute position embedding
        use_rel_pe = False,        # use relative position embedding
        pretrained=True
    ):
        super(ActionFormerWithTimesformer, self).__init__()

        # Timesformer Encoder
        self.encoder = TimesformerEncoder(pretrained=pretrained, num_frames=max_seq_len, out_dim=input_dim, image_size=128)

        # Decoder Initialization using make_backbone
        self.backbone = make_backbone(
            'convTransformer',
            **{
                'n_in' : input_dim,
                'n_embd' : embd_dim,
                'n_head': n_head,
                'n_embd_ks': embd_kernel_size,
                'max_len': max_seq_len,
                'arch' : backbone_arch,
                'mha_win_size': mha_win_size,
                'scale_factor' : scale_factor,
                'with_ln' : embd_with_ln,
                'attn_pdrop' : attn_pdrop,
                'proj_pdrop' : proj_pdrop,
                'path_pdrop' : path_pdrop,
                'use_abs_pe' : use_abs_pe,
                'use_rel_pe' : use_rel_pe
            }
        )

    def forward(self, x, mask):
        # Extract features using Timesformer encoder
        features = self.encoder(x)  # Shape: (batch_size, num_frames, embd_dim)

        # Pass features to the decoder (backbone)
        output = self.backbone(features, mask)
        return output



class VideoMAEEncoder(nn.Module):
    def __init__(self, pretrained=True, num_frames=16, out_dim=768, image_size=224, tubelet_size=1):
        super(VideoMAEEncoder, self).__init__()

        self.config = VideoMAEConfig(
            num_frames=num_frames,
            image_size=image_size,
            tubelet_size=tubelet_size,
            num_hidden_layers=3,
            num_attention_heads=3,  # Reduce number of heads if necessary
        )
        self.VideoMAE = VideoMAEModel.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            config=self.config,
            ignore_mismatched_sizes=True
        )

        # Freeze all layers except the last transformer block
        for name, param in self.VideoMAE.named_parameters():
            if 'embeddings' in name or "layer.2" in name:  # Fine-tune only the last transformer block
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Linear projection (if needed)
        if self.config.hidden_size != out_dim:
            self.projection = nn.Linear(self.config.hidden_size, out_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        # Input shape: (batch_size, num_frames, channels, height, width)
        b, t, c, h, w = x.size()

        # Pass through VideoMAE
        x = self.VideoMAE(x)
        # last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.
        last_hidden_state = x['last_hidden_state']

        # Extract features for each frame
        num_patches_per_frame = last_hidden_state.size(1) // t
        last_hidden_state = last_hidden_state.view(b, t, num_patches_per_frame, -1)  # Shape: (batch_size, num_frames, num_patches_per_frame, embd_dim)

        # Aggregate patch features to get a single feature vector per frame
        frame_features = last_hidden_state.mean(dim=2)  # Shape: (batch_size, num_frames, embd_dim)

        frame_features = self.projection(frame_features) # Project features to match decoder input dimension
        return frame_features.permute(0,2,1)

@register_backbone("ActionFormerWithVideoMAE")
class ActionFormerWithVideoMAE(nn.Module):
    def __init__(
        self,
        input_dim,                 # input feature dimension
        embd_dim,                  # embedding dimension (after convolution)
        n_head,                    # number of head for self-attention in transformers
        embd_kernel_size,          # conv kernel size of the embedding network
        max_seq_len,               # max sequence length
        backbone_arch = (2, 2, 5), # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*6,     # size of local window for mha
        scale_factor = 2,          # dowsampling rate for the branch
        embd_with_ln = False,      # if to attach layernorm after conv
        attn_pdrop = 0.0,          # dropout rate for the attention map
        proj_pdrop = 0.0,          # dropout rate for the projection / MLP
        path_pdrop = 0.0,          # droput rate for drop path
        use_abs_pe = False,        # use absolute position embedding
        use_rel_pe = False,        # use relative position embedding
        pretrained=True
    ):
        super(ActionFormerWithVideoMAE, self).__init__()

        # VideoMAE Encoder
        self.encoder = VideoMAEEncoder(pretrained=pretrained, num_frames=max_seq_len, out_dim=input_dim, image_size=128)

        # Decoder Initialization using make_backbone
        self.backbone = make_backbone(
            'convTransformer',
            **{
                'n_in' : input_dim,
                'n_embd' : embd_dim,
                'n_head': n_head,
                'n_embd_ks': embd_kernel_size,
                'max_len': max_seq_len,
                'arch' : backbone_arch,
                'mha_win_size': mha_win_size,
                'scale_factor' : scale_factor,
                'with_ln' : embd_with_ln,
                'attn_pdrop' : attn_pdrop,
                'proj_pdrop' : proj_pdrop,
                'path_pdrop' : path_pdrop,
                'use_abs_pe' : use_abs_pe,
                'use_rel_pe' : use_rel_pe
            }
        )

    def forward(self, x, mask):
        # Extract features using VideoMAE encoder
        features = self.encoder(x)  # Shape: (batch_size, num_frames, embd_dim)

        # Pass features to the decoder (backbone)
        output = self.backbone(features, mask)
        return output
    


class CLIPEncoder(nn.Module):
    def __init__(self, pretrained=True, num_frames=16, out_dim=768, image_size=224):
        super(CLIPEncoder, self).__init__()

        self.config = CLIPVisionConfig(image_size = image_size)
        # self.clip_image_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
        #     config=self.config,
        #     ignore_mismatched_sizes=True).vision_model
        self.clip_image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32",
            config=self.config,
            ignore_mismatched_sizes=True)


        # Freeze all layers except the last few
        for name, param in self.clip_image_encoder.named_parameters():
            # if 'embeddings' in name or "encoder.layer.11" in name or "encoder.layer.10" in name or "layernorm" in name:  # Adjust layers to fine-tune
            param.requires_grad = True
            # else:
                # param.requires_grad = False

        # Linear projection (if needed)
        if self.clip_image_encoder.config.hidden_size != out_dim:
            self.projection = nn.Linear(self.clip_image_encoder.config.hidden_size, out_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        # Input shape: (batch_size, num_frames, channels, height, width)
        b, t, c, h, w = x.size()

        # Pass through CLIP
        x = x.contiguous().view(-1, c, h, w)  # Merge batch and temporal dimensions
        last_hidden_state = self.clip_image_encoder(pixel_values=x).last_hidden_state  # Extract frame embeddings
        last_hidden_state = last_hidden_state.mean(dim=1) # x = x[:, 0]

        # Extract features for each frame
        last_hidden_state = last_hidden_state.view(b, t, -1)  # Shape: (batch_size, num_frames, embd_dim)

        frame_features = self.projection(last_hidden_state) # Project features to match decoder input dimension
        return frame_features.permute(0,2,1)

@register_backbone("ActionFormerWithCLIP")
class ActionFormerWithCLIP(nn.Module):
    def __init__(
        self,
        input_dim,                 # input feature dimension
        embd_dim,                  # embedding dimension (after convolution)
        n_head,                    # number of head for self-attention in transformers
        embd_kernel_size,          # conv kernel size of the embedding network
        max_seq_len,               # max sequence length
        backbone_arch = (2, 2, 5), # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*6,     # size of local window for mha
        scale_factor = 2,          # dowsampling rate for the branch
        embd_with_ln = False,      # if to attach layernorm after conv
        attn_pdrop = 0.0,          # dropout rate for the attention map
        proj_pdrop = 0.0,          # dropout rate for the projection / MLP
        path_pdrop = 0.0,          # droput rate for drop path
        use_abs_pe = False,        # use absolute position embedding
        use_rel_pe = False,        # use relative position embedding
        pretrained=True
    ):
        super(ActionFormerWithCLIP, self).__init__()

        # CLIP Encoder
        self.encoder = CLIPEncoder(pretrained=pretrained, num_frames=max_seq_len, out_dim=input_dim, image_size=224)

        # Decoder Initialization using make_backbone
        self.backbone = make_backbone(
            'convTransformer',
            **{
                'n_in' : input_dim,
                'n_embd' : embd_dim,
                'n_head': n_head,
                'n_embd_ks': embd_kernel_size,
                'max_len': max_seq_len,
                'arch' : backbone_arch,
                'mha_win_size': mha_win_size,
                'scale_factor' : scale_factor,
                'with_ln' : embd_with_ln,
                'attn_pdrop' : attn_pdrop,
                'proj_pdrop' : proj_pdrop,
                'path_pdrop' : path_pdrop,
                'use_abs_pe' : use_abs_pe,
                'use_rel_pe' : use_rel_pe
            }
        )

    def forward(self, x, mask):
        # Extract features using CLIP encoder
        features = self.encoder(x)  # Shape: (batch_size, num_frames, embd_dim)

        # Pass features to the decoder (backbone)
        output = self.backbone(features, mask)
        return output


@register_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_in,                  # input feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch = (2, 2, 5),      # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*6, # size of local window for mha
        scale_factor = 2,      # dowsampling rate for the branch
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
        use_rel_pe = False,    # use relative position embedding
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[2])
        self.n_in = n_in
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe

        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe
                )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe
                )
            )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # stem transformer
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x, )
        out_masks = (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks


@register_backbone("conv")
class ConvBackbone(nn.Module):
    """
        A backbone that with only conv
    """
    def __init__(
        self,
        n_in,               # input feature dimension
        n_embd,             # embedding dimension (after convolution)
        n_embd_ks,          # conv kernel size of the embedding network
        arch = (2, 2, 5),   # (#convs, #stem convs, #branch convs)
        scale_factor = 2,   # dowsampling rate for the branch
        with_ln=False,      # if to use layernorm
    ):
        super().__init__()
        assert len(arch) == 3
        self.n_in = n_in
        self.arch = arch
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # stem network using convs
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(ConvBlock(n_embd, 3, 1))

        # main branch using convs with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(ConvBlock(n_embd, 3, self.scale_factor))

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # stem conv
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x, )
        out_masks = (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks