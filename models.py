import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import math
import torchvision.models as models
from einops.layers.torch import Rearrange
from MSFormer_utils import OverlapPatchEmbeddings, EfficientTransformerBlock, EfficientTransformerBlock_v2
import yaml



# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

### Create models for the different baselines using segmentation models pytorch
def create_model(architecture='Unet', encoder_name='resnet50', in_channels=3, num_classes=1, encoder_weights='imagenet'):
    model = getattr(smp, architecture)(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes
    )
    return model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# New Convolutional Block for Ablation Models
class ConvBlock_v2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Baseline Model (Simple UNet)
class MSFormer_baseline(nn.Module):
    def __init__(self, input_dim, num_classes=1):
        super().__init__()
        self.model = smp.Unet(
            encoder_name='resnet18',
            encoder_weights='imagenet',
            in_channels=input_dim,
            classes=num_classes
        )
    
    def forward(self, x):
        out = self.model(x)
        return out, None, None, None, None, None, None, None

# SAE + AGF Model
class MSFormer_SAE_AGF(nn.Module):
    def __init__(self, input_dim, embed_size=100, img_size=448):
        super().__init__()
        self.dim = embed_size
        self.patch_size = 16
        
        # Three-scale patch embeddings (SAE)
        self.patch_embed_f = OverlapPatchEmbeddings(patch_size=1, stride=1, padding=0, in_ch=input_dim, dim=self.dim)
        self.patch_embed_s = OverlapPatchEmbeddings(patch_size=3, stride=1, padding=1, in_ch=input_dim, dim=self.dim)
        self.patch_embed_l = OverlapPatchEmbeddings(patch_size=3, stride=2, padding=1, in_ch=input_dim, dim=self.dim)
        
        # Convolutional blocks instead of transformers
        self.conv_f = ConvBlock_v2(self.dim, self.dim)
        self.conv_s = ConvBlock_v2(self.dim, self.dim)
        self.conv_l = ConvBlock_v2(self.dim, self.dim)
        
        # Up-projection for large branch
        self.linear_up = nn.Linear(self.dim, 4 * self.dim, bias=False)
        self.proj_l = nn.Conv2d(4 * self.dim, self.dim, kernel_size=1)
        
        # Attention-based fusion module (AGF)
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(6 * self.dim, 3, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final prediction
        self.linear_down = nn.Linear(self.dim, config['model']['nChannel'], bias=False)
    
    def forward(self, x):
        # Three-scale embeddings
        x_f, h_f, w_f = self.patch_embed_f(x)
        x_f = Rearrange('b (h w) d -> b d h w', h=h_f, w=w_f)(x_f)
        x_s, h_s, w_s = self.patch_embed_s(x)
        x_s = Rearrange('b (h w) d -> b d h w', h=h_s, w=w_s)(x_s)
        x_l, h_l, w_l = self.patch_embed_l(x)
        x_l = Rearrange('b (h w) d -> b d h w', h=h_l, w=w_l)(x_l)
        
        # Apply conv blocks
        x_f = self.conv_f(x_f)
        x_s = self.conv_s(x_s)
        x_l = self.conv_l(x_l)
        
        # Upsample large branch
        B, C_l, _, _ = x_l.shape
        x_l = x_l.permute(0, 2, 3, 1).reshape(B, h_l * w_l, C_l)
        x_l = self.linear_up(x_l)
        x_l = x_l.view(B, h_l, w_l, 4 * self.dim).permute(0, 3, 1, 2)
        x_l = F.interpolate(x_l, size=(h_s, w_s), mode='bilinear', align_corners=False)
        
        # Concatenate features
        x_cat = torch.cat([x_l, x_s, x_f], dim=1)
        
        # Compute attention weights
        attention_weights = self.attention_fusion(x_cat)
        
        # Split concatenated features
        x_l_split = x_cat[:, :4 * self.dim, :, :]
        x_s_split = x_cat[:, 4 * self.dim:5 * self.dim, :, :]
        x_f_split = x_cat[:, 5 * self.dim:, :, :]
        
        # Project large scale
        x_l_proj = self.proj_l(x_l_split)
        
        # Apply attention weights
        weighted_x_l = x_l_proj * attention_weights[:, 0:1, :, :]
        weighted_x_s = x_s_split * attention_weights[:, 1:2, :, :]
        weighted_x_f = x_f_split * attention_weights[:, 2:3, :, :]
        
        # Sum weighted features
        fused_features = weighted_x_l + weighted_x_s + weighted_x_f
        
        # Project to output channels
        out = self.linear_down(fused_features.permute(0, 2, 3, 1))
        out = out.permute(0, 3, 1, 2)
        return out, None, None, None, None, None, None, None

# DAT + AGF Model
class MSFormer_DAT_AGF(nn.Module):
    def __init__(self, input_dim, embed_size=100, img_size=257):
        super().__init__()
        self.dim = embed_size
        
        # Single-scale embedding
        self.patch_embed = OverlapPatchEmbeddings(patch_size=3, stride=1, padding=1, in_ch=input_dim, dim=self.dim)
        
        # Directed attention transformer block (DAT)
        self.block = EfficientTransformerBlock_v2(self.dim, self.dim, self.dim, head_count=1, token_mlp='mix_skip')
        self.bn = nn.BatchNorm2d(self.dim)
        
        # Attention-based fusion (simplified for single scale)
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(self.dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.linear_down = nn.Linear(self.dim, config['model']['nChannel'], bias=False)
    
    def forward(self, x):
        x, h, w = self.patch_embed(x)
        x = Rearrange('b (h w) d -> b d h w', h=h, w=w)(x)
        x, context, attention_map = self.block(x, h, w)
        x = self.bn(x)
        
        # Apply attention weighting (simplified AGF for single scale)
        attention_weights = self.attention_fusion(x)
        fused_features = x * attention_weights
        
        out = self.linear_down(fused_features.permute(0, 2, 3, 1))
        out = out.permute(0, 3, 1, 2)
        return out, None, attention_map, None, None, context, None, None


## Grok version 1
## Introduced another scale for the patch embedding( Fine scale)
class MSFormer_v1(nn.Module):
    def __init__(self, input_dim, embed_size=100, img_size=257,
                 large_nlayers=[1], small_nlayers=[1], fine_nlayers=[1]):
        super(MSFormer_v1, self).__init__()
        
        # ---- core dimensions ----
        self.dim         = embed_size
        self.dim_scale   = 2
        self.patch_size  = 16
        self.large_nlayers = large_nlayers[0]
        self.small_nlayers = small_nlayers[0]
        self.fine_nlayers  = fine_nlayers[0]
        
        # ---- three-scale patch embeddings ----
        self.patch_embed_f = OverlapPatchEmbeddings(
            patch_size=1, stride=1, padding=0,
            in_ch=input_dim, dim=self.dim
        )
        self.patch_embed_s = OverlapPatchEmbeddings(
            patch_size=3, stride=1, padding=1,
            in_ch=input_dim, dim=self.dim
        )
        self.patch_embed_l = OverlapPatchEmbeddings(
            patch_size=3, stride=2, padding=1,
            in_ch=input_dim, dim=self.dim
        )
        
        # ---- dynamic patch→token modules for spatial correlation ----
        self.unfold     = nn.Unfold(kernel_size=self.patch_size,
                                    stride=self.patch_size,
                                    padding=0)
        self.patch_proj = nn.Linear(self.dim * self.patch_size**2,
                                    self.dim)
        
        # ---- Efficient Transformer blocks ----
        self.block_f = nn.ModuleList([
            EfficientTransformerBlock(self.dim, self.dim, self.dim, head_count=1, token_mlp='mix_skip')
            for _ in range(self.fine_nlayers)
        ])
        self.block_s = nn.ModuleList([
            EfficientTransformerBlock(self.dim, self.dim, self.dim, head_count=1, token_mlp='mix_skip')
            for _ in range(self.small_nlayers)
        ])
        self.block_l = nn.ModuleList([
            EfficientTransformerBlock(self.dim, self.dim, self.dim, head_count=1, token_mlp='mix_skip')
            for _ in range(self.large_nlayers)
        ])
        
        # ---- BatchNorm after each block ----
        self.bn2_f = nn.ModuleList([nn.BatchNorm2d(self.dim)
                                    for _ in range(self.fine_nlayers)])
        self.bn2_s = nn.ModuleList([nn.BatchNorm2d(self.dim)
                                    for _ in range(self.small_nlayers)])
        self.bn2_l = nn.ModuleList([nn.BatchNorm2d(self.dim)
                                    for _ in range(self.large_nlayers)])
        
        # ---- up-projection of large branch tokens ----
        self.linear_up   = nn.Linear(self.dim, 4 * self.dim, bias=False)
        
        # ---- final prediction: concat 4*dim (large) + dim (small) + dim (fine) = 6*dim features ----
        self.linear_down = nn.Linear(6 * self.dim, config['model']['nChannel'], bias=False)
        
        # ---- spatial-correlation heads ----
        self.softmax = nn.Softmax(dim=-1)
        self.bn3     = nn.BatchNorm2d(self.dim)
    
    def patches_from(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Turn B×C×H×W → (possibly padded) patches → B×L×dim
        """
        B, C, H, W = feat.shape
        
        # pad to multiple of patch_size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        f = F.pad(feat, (0, pad_w, 0, pad_h))   # (l, r, t, b)
        
        # unfold into (B, C*p*p, L)
        p = self.unfold(f)
        p = p.transpose(1, 2)                   # (B, L, C*p*p)
        return self.patch_proj(p)               # (B, L, dim)
    
    def forward(self, x: torch.Tensor):
        # ---- three-scale embeddings ----
        x_f, h_f, w_f = self.patch_embed_f(x)
        x_f = Rearrange('b (h w) d -> b d h w', h=h_f, w=w_f)(x_f)
        
        x_s, h_s, w_s = self.patch_embed_s(x)
        x_s = Rearrange('b (h w) d -> b d h w', h=h_s, w=w_s)(x_s)
        
        x_l, h_l, w_l = self.patch_embed_l(x)
        x_l = Rearrange('b (h w) d -> b d h w', h=h_l, w=w_l)(x_l)
        
        # ---- fine-branch transformer + BN ----
        for i in range(self.fine_nlayers):
            x_f, context_f, attention_map_f = self.block_f[i](x_f, h_f, w_f)
            x_f = self.bn2_f[i](x_f)
        
        # ---- small-branch transformer + BN ----
        for i in range(self.small_nlayers):
            x_s, context_s, attention_map_s = self.block_s[i](x_s, h_s, w_s)
            x_s = self.bn2_s[i](x_s)
        
        # ---- large-branch transformer + BN ----
        for i in range(self.large_nlayers):
            x_l, context_l, attention_map_l = self.block_l[i](x_l, h_l, w_l)
            x_l = self.bn2_l[i](x_l)
        
        # ---- upsample large branch into small grid ----
        B, C_l, _, _ = x_l.shape
        x_l = x_l.permute(0, 2, 3, 1).reshape(B, h_l*w_l, C_l)
        x_l = self.linear_up(x_l)  # (B, h_l*w_l, 4*dim)
        x_l = x_l.view(B, h_l, w_l, 4*self.dim).permute(0, 3, 1, 2)
        x_l = F.interpolate(x_l, size=(h_s, w_s), mode='bilinear', align_corners=False)
        
        # ---- concat and predict ----
        x_cat = torch.cat([x_l, x_s, x_f], dim=1)  # B×(6*dim)×h_s×w_s
        out   = self.linear_down(x_cat.permute(0, 2, 3, 1))  # B×(h_s*w_s)×nChannel
        out   = out.permute(0, 3, 1, 2)  # B×nChannel×h_s×w_s
        
        # ---- spatial-correlation self-attn ----
        out2 = self.bn3(out)
        Q_tensor = out2[:, :, 1:, 1:]
        K_tensor = out2[:, :, :-1, :-1]
        
        Q = self.patches_from(Q_tensor)  # B×L×dim
        K = self.patches_from(K_tensor)  # B×L×dim
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) * (self.patch_size ** -0.5)
        attn   = self.softmax(scores)
        
        return out, attn, attention_map_f, attention_map_s, attention_map_l, context_f, context_s, context_l


## Grok version 2 
## Introduced another scale + modified Efficient Transformer Block with directed convolutions
class MSFormer_v2(nn.Module):
    def __init__(self, input_dim, embed_size=100, img_size=257,
                 large_nlayers=[1], small_nlayers=[1], fine_nlayers=[1]):
        super(MSFormer_v2, self).__init__()
        
        # ---- core dimensions ----
        self.dim         = embed_size
        self.dim_scale   = 2
        self.patch_size  = 16
        self.large_nlayers = large_nlayers[0]
        self.small_nlayers = small_nlayers[0]
        self.fine_nlayers  = fine_nlayers[0]
        
        # ---- three-scale patch embeddings ----
        self.patch_embed_f = OverlapPatchEmbeddings(
            patch_size=1, stride=1, padding=0,
            in_ch=input_dim, dim=self.dim
        )
        self.patch_embed_s = OverlapPatchEmbeddings(
            patch_size=3, stride=1, padding=1,
            in_ch=input_dim, dim=self.dim
        )
        self.patch_embed_l = OverlapPatchEmbeddings(
            patch_size=3, stride=2, padding=1,
            in_ch=input_dim, dim=self.dim
        )
        
        # ---- dynamic patch→token modules for spatial correlation ----
        self.unfold     = nn.Unfold(kernel_size=self.patch_size,
                                    stride=self.patch_size,
                                    padding=0)
        self.patch_proj = nn.Linear(self.dim * self.patch_size**2,
                                    self.dim)
        
        # ---- Efficient Transformer blocks ----
        self.block_f = nn.ModuleList([
            EfficientTransformerBlock_v2(self.dim, self.dim, self.dim, head_count=1, token_mlp='mix_skip')
            for _ in range(self.fine_nlayers)
        ])
        self.block_s = nn.ModuleList([
            EfficientTransformerBlock_v2(self.dim, self.dim, self.dim, head_count=1, token_mlp='mix_skip')
            for _ in range(self.small_nlayers)
        ])
        self.block_l = nn.ModuleList([
            EfficientTransformerBlock_v2(self.dim, self.dim, self.dim, head_count=1, token_mlp='mix_skip')
            for _ in range(self.large_nlayers)
        ])
        
        # ---- BatchNorm after each block ----
        self.bn2_f = nn.ModuleList([nn.BatchNorm2d(self.dim)
                                    for _ in range(self.fine_nlayers)])
        self.bn2_s = nn.ModuleList([nn.BatchNorm2d(self.dim)
                                    for _ in range(self.small_nlayers)])
        self.bn2_l = nn.ModuleList([nn.BatchNorm2d(self.dim)
                                    for _ in range(self.large_nlayers)])
        
        # ---- up-projection of large branch tokens ----
        self.linear_up   = nn.Linear(self.dim, 4 * self.dim, bias=False)
        
        # ---- final prediction: concat 4*dim (large) + dim (small) + dim (fine) = 6*dim features ----
        self.linear_down = nn.Linear(6 * self.dim, config['model']['nChannel'], bias=False)
        
        # ---- spatial-correlation heads ----
        self.softmax = nn.Softmax(dim=-1)
        self.bn3     = nn.BatchNorm2d(self.dim)
    
    def patches_from(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Turn B×C×H×W → (possibly padded) patches → B×L×dim
        """
        B, C, H, W = feat.shape
        
        # pad to multiple of patch_size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        f = F.pad(feat, (0, pad_w, 0, pad_h))   # (l, r, t, b)
        
        # unfold into (B, C*p*p, L)
        p = self.unfold(f)
        p = p.transpose(1, 2)                   # (B, L, C*p*p)
        return self.patch_proj(p)               # (B, L, dim)
    
    def forward(self, x: torch.Tensor):
        # ---- three-scale embeddings ----
        x_f, h_f, w_f = self.patch_embed_f(x)
        x_f = Rearrange('b (h w) d -> b d h w', h=h_f, w=w_f)(x_f)
        
        x_s, h_s, w_s = self.patch_embed_s(x)
        x_s = Rearrange('b (h w) d -> b d h w', h=h_s, w=w_s)(x_s)
        
        x_l, h_l, w_l = self.patch_embed_l(x)
        x_l = Rearrange('b (h w) d -> b d h w', h=h_l, w=w_l)(x_l)
        
        # ---- fine-branch transformer + BN ----
        for i in range(self.fine_nlayers):
            x_f, context_f, attention_map_f = self.block_f[i](x_f, h_f, w_f)
            x_f = self.bn2_f[i](x_f)
        
        # ---- small-branch transformer + BN ----
        for i in range(self.small_nlayers):
            x_s, context_s, attention_map_s = self.block_s[i](x_s, h_s, w_s)
            x_s = self.bn2_s[i](x_s)
        
        # ---- large-branch transformer + BN ----
        for i in range(self.large_nlayers):
            x_l, context_l, attention_map_l = self.block_l[i](x_l, h_l, w_l)
            x_l = self.bn2_l[i](x_l)
        
        # ---- upsample large branch into small grid ----
        B, C_l, _, _ = x_l.shape
        x_l = x_l.permute(0, 2, 3, 1).reshape(B, h_l*w_l, C_l)
        x_l = self.linear_up(x_l)  # (B, h_l*w_l, 4*dim)
        x_l = x_l.view(B, h_l, w_l, 4*self.dim).permute(0, 3, 1, 2)
        x_l = F.interpolate(x_l, size=(h_s, w_s), mode='bilinear', align_corners=False)
        
        # ---- concat and predict ----
        x_cat = torch.cat([x_l, x_s, x_f], dim=1)  # B×(6*dim)×h_s×w_s
        out   = self.linear_down(x_cat.permute(0, 2, 3, 1))  # B×(h_s*w_s)×nChannel
        out   = out.permute(0, 3, 1, 2)  # B×nChannel×h_s×w_s
        
        # ---- spatial-correlation self-attn ----
        out2 = self.bn3(out)
        Q_tensor = out2[:, :, 1:, 1:]
        K_tensor = out2[:, :, :-1, :-1]
        
        Q = self.patches_from(Q_tensor)  # B×L×dim
        K = self.patches_from(K_tensor)  # B×L×dim
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) * (self.patch_size ** -0.5)
        attn   = self.softmax(scores)
        
        return out, attn, attention_map_f, attention_map_s, attention_map_l, context_f, context_s, context_l


## Grok version 3
# - Attention based weighting strategy for the different feature scales 
class MSFormer_v3(nn.Module):
    def __init__(self, input_dim, embed_size=100, img_size=257,
                 large_nlayers=[1], small_nlayers=[1], fine_nlayers=[1]):
        super(MSFormer_v3, self).__init__()
        
        # ---- core dimensions ----
        self.dim = embed_size
        self.dim_scale = 2
        self.patch_size = 16
        self.large_nlayers = large_nlayers[0]
        self.small_nlayers = small_nlayers[0]
        self.fine_nlayers = fine_nlayers[0]
        
        # ---- three-scale patch embeddings ----
        self.patch_embed_f = OverlapPatchEmbeddings(
            patch_size=1, stride=1, padding=0,
            in_ch=input_dim, dim=self.dim
        )
        self.patch_embed_s = OverlapPatchEmbeddings(
            patch_size=3, stride=1, padding=1,
            in_ch=input_dim, dim=self.dim
        )
        self.patch_embed_l = OverlapPatchEmbeddings(
            patch_size=3, stride=2, padding=1,
            in_ch=input_dim, dim=self.dim
        )
        
        # ---- dynamic patch→token modules for spatial correlation ----
        self.unfold = nn.Unfold(kernel_size=self.patch_size,
                                stride=self.patch_size,
                                padding=0)
        self.patch_proj = nn.Linear(self.dim * self.patch_size**2,
                                    self.dim)
        
        # ---- Efficient Transformer blocks ----
        self.block_f = nn.ModuleList([
            EfficientTransformerBlock_v2(self.dim, self.dim, self.dim, head_count=1, token_mlp='mix_skip')
            for _ in range(self.fine_nlayers)
        ])
        self.block_s = nn.ModuleList([
            EfficientTransformerBlock_v2(self.dim, self.dim, self.dim, head_count=1, token_mlp='mix_skip')
            for _ in range(self.small_nlayers)
        ])
        self.block_l = nn.ModuleList([
            EfficientTransformerBlock_v2(self.dim, self.dim, self.dim, head_count=1, token_mlp='mix_skip')
            for _ in range(self.large_nlayers)
        ])
        
        # ---- BatchNorm after each block ----
        self.bn2_f = nn.ModuleList([nn.BatchNorm2d(self.dim)
                                    for _ in range(self.fine_nlayers)])
        self.bn2_s = nn.ModuleList([nn.BatchNorm2d(self.dim)
                                    for _ in range(self.small_nlayers)])
        self.bn2_l = nn.ModuleList([nn.BatchNorm2d(self.dim)
                                    for _ in range(self.large_nlayers)])
        
        # ---- up-projection of large branch tokens ----
        self.linear_up = nn.Linear(self.dim, 4 * self.dim, bias=False)
        
        # ---- Attention-based fusion module ----
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(6 * self.dim, 3, kernel_size=1),  # Reduce to 3 channels (one for each scale)
            nn.Sigmoid()  # Attention weights between 0 and 1
        )
        
        # ---- final prediction: weighted sum of features ----
        self.linear_down = nn.Linear(self.dim, config['model']['nChannel'], bias=False)
        
        # ---- spatial-correlation heads ----
        self.softmax = nn.Softmax(dim=-1)
        self.bn3 = nn.BatchNorm2d(self.dim)
        
        # Projection - This 1×1 convolution reduces the channel dimension from 4 * dim to dim.
        self.proj_l = nn.Conv2d(4 * self.dim, self.dim, kernel_size=1)
    
    def patches_from(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Turn B×C×H×W → (possibly padded) patches → B×L×dim
        """
        B, C, H, W = feat.shape
        
        # pad to multiple of patch_size
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        f = F.pad(feat, (0, pad_w, 0, pad_h))  # (l, r, t, b)
        
        # unfold into (B, C*p*p, L)
        p = self.unfold(f)
        p = p.transpose(1, 2)  # (B, L, C*p*p)
        return self.patch_proj(p)  # (B, L, dim)
    
    # forward for MSFormer which doesn't return attention maps
    # def forward(self, x: torch.Tensor):
    #     # ---- three-scale embeddings ----
    #     x_f, h_f, w_f = self.patch_embed_f(x)
    #     x_f = Rearrange('b (h w) d -> b d h w', h=h_f, w=w_f)(x_f)
        
    #     x_s, h_s, w_s = self.patch_embed_s(x)
    #     x_s = Rearrange('b (h w) d -> b d h w', h=h_s, w=w_s)(x_s)
        
    #     x_l, h_l, w_l = self.patch_embed_l(x)
    #     x_l = Rearrange('b (h w) d -> b d h w', h=h_l, w=w_l)(x_l)
        
    #     # ---- fine-branch transformer + BN ----
    #     for i in range(self.fine_nlayers):
    #         x_f, context_f = self.block_f[i](x_f, h_f, w_f)
    #         x_f = self.bn2_f[i](x_f)
        
    #     # ---- small-branch transformer + BN ----
    #     for i in range(self.small_nlayers):
    #         x_s, context_s = self.block_s[i](x_s, h_s, w_s)
    #         x_s = self.bn2_s[i](x_s)
        
    #     # ---- large-branch transformer + BN ----
    #     for i in range(self.large_nlayers):
    #         x_l, context_l = self.block_l[i](x_l, h_l, w_l)
    #         x_l = self.bn2_l[i](x_l)
        
    #     # ---- upsample large branch into small grid ----
    #     B, C_l, _, _ = x_l.shape
    #     x_l = x_l.permute(0, 2, 3, 1).reshape(B, h_l * w_l, C_l)
    #     x_l = self.linear_up(x_l)  # (B, h_l*w_l, 4*dim)
    #     x_l = x_l.view(B, h_l, w_l, 4 * self.dim).permute(0, 3, 1, 2)
    #     x_l = F.interpolate(x_l, size=(h_s, w_s), mode='bilinear', align_corners=False)
        
    #     # ---- concat features from all scales ----
    #     x_cat = torch.cat([x_l, x_s, x_f], dim=1)  # B×(6*dim)×h_s×w_s
        
    #     # ---- compute attention weights for each scale ----
    #     # Compute attention weights for each scale
    #     attention_weights = self.attention_fusion(x_cat)  # B × 3 × h_s × w_s

    #     # Split concatenated features into scales
    #     x_l_split = x_cat[:, :4 * self.dim, :, :]  # B × (4 * dim) × h_s × w_s
    #     x_s_split = x_cat[:, 4 * self.dim:5 * self.dim, :, :]  # B × dim × h_s × w_s
    #     x_f_split = x_cat[:, 5 * self.dim:, :, :]  # B × dim × h_s × w_s

    #     # Project large scale features
    #     x_l_proj = self.proj_l(x_l_split)  # B × dim × h_s × w_s

    #     # Apply attention weights to each scale's features
    #     weighted_x_l = x_l_proj * attention_weights[:, 0:1, :, :]  # B × dim × h_s × w_s
    #     weighted_x_s = x_s_split * attention_weights[:, 1:2, :, :]  # B × dim × h_s × w_s
    #     weighted_x_f = x_f_split * attention_weights[:, 2:3, :, :]  # B × dim × h_s × w_s

    #     # Sum the weighted features
    #     fused_features = weighted_x_l + weighted_x_s + weighted_x_f  # B × dim × h_s × w_s

    #     # Project to output channels
    #     out = self.linear_down(fused_features.permute(0, 2, 3, 1))  # B × h_s × w_s × nChannel
    #     out = out.permute(0, 3, 1, 2)  # B × nChannel × h_s × w_s
        
    #     # ---- spatial-correlation self-attn ----
    #     out2 = self.bn3(out)
    #     Q_tensor = out2[:, :, 1:, 1:]
    #     K_tensor = out2[:, :, :-1, :-1]
        
    #     Q = self.patches_from(Q_tensor)  # B×L×dim
    #     K = self.patches_from(K_tensor)  # B×L×dim
        
    #     scores = torch.matmul(Q, K.transpose(-1, -2)) * (self.patch_size ** -0.5)
    #     attn = self.softmax(scores)
        
    #     return out, attn, context_f, context_s, context_l


    def forward(self, x: torch.Tensor):
        # ---- three-scale embeddings ----
        x_f, h_f, w_f = self.patch_embed_f(x)
        x_f = Rearrange('b (h w) d -> b d h w', h=h_f, w=w_f)(x_f)
        
        x_s, h_s, w_s = self.patch_embed_s(x)
        x_s = Rearrange('b (h w) d -> b d h w', h=h_s, w=w_s)(x_s)
        
        x_l, h_l, w_l = self.patch_embed_l(x)
        x_l = Rearrange('b (h w) d -> b d h w', h=h_l, w=w_l)(x_l)
        
        # ---- fine-branch transformer + BN ----
        for i in range(self.fine_nlayers):
            x_f, context_f, attention_map_f = self.block_f[i](x_f, h_f, w_f)
            x_f = self.bn2_f[i](x_f)
        
        # ---- small-branch transformer + BN ----
        for i in range(self.small_nlayers):
            x_s, context_s, attention_map_s = self.block_s[i](x_s, h_s, w_s)
            x_s = self.bn2_s[i](x_s)
        
        # ---- large-branch transformer + BN ----
        for i in range(self.large_nlayers):
            x_l, context_l, attention_map_l = self.block_l[i](x_l, h_l, w_l)
            x_l = self.bn2_l[i](x_l)
        
        # ---- upsample large branch into small grid ----
        B, C_l, _, _ = x_l.shape
        x_l = x_l.permute(0, 2, 3, 1).reshape(B, h_l * w_l, C_l)
        x_l = self.linear_up(x_l)  # (B, h_l*w_l, 4*dim)
        x_l = x_l.view(B, h_l, w_l, 4 * self.dim).permute(0, 3, 1, 2)
        x_l = F.interpolate(x_l, size=(h_s, w_s), mode='bilinear', align_corners=False)
        
        # ---- concat features from all scales ----
        x_cat = torch.cat([x_l, x_s, x_f], dim=1)  # B×(6*dim)×h_s×w_s
        
        # ---- compute attention weights for each scale ----
        # Compute attention weights for each scale
        attention_weights = self.attention_fusion(x_cat)  # B × 3 × h_s × w_s

        # Split concatenated features into scales
        x_l_split = x_cat[:, :4 * self.dim, :, :]  # B × (4 * dim) × h_s × w_s
        x_s_split = x_cat[:, 4 * self.dim:5 * self.dim, :, :]  # B × dim × h_s × w_s
        x_f_split = x_cat[:, 5 * self.dim:, :, :]  # B × dim × h_s × w_s

        # Project large scale features
        x_l_proj = self.proj_l(x_l_split)  # B × dim × h_s × w_s

        # Apply attention weights to each scale's features
        weighted_x_l = x_l_proj * attention_weights[:, 0:1, :, :]  # B × dim × h_s × w_s
        weighted_x_s = x_s_split * attention_weights[:, 1:2, :, :]  # B × dim × h_s × w_s
        weighted_x_f = x_f_split * attention_weights[:, 2:3, :, :]  # B × dim × h_s × w_s

        # Sum the weighted features
        fused_features = weighted_x_l + weighted_x_s + weighted_x_f  # B × dim × h_s × w_s

        # Project to output channels
        out = self.linear_down(fused_features.permute(0, 2, 3, 1))  # B × h_s × w_s × nChannel
        out = out.permute(0, 3, 1, 2)  # B × nChannel × h_s × w_s
        
        # ---- spatial-correlation self-attn ----
        out2 = self.bn3(out)
        Q_tensor = out2[:, :, 1:, 1:]
        K_tensor = out2[:, :, :-1, :-1]
        
        Q = self.patches_from(Q_tensor)  # B×L×dim
        K = self.patches_from(K_tensor)  # B×L×dim
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) * (self.patch_size ** -0.5)
        attn = self.softmax(scores)
        
        return out, attn, attention_map_f, attention_map_s, attention_map_l, context_f, context_s, context_l
        #return out, attn, context_f, context_s, context_l
    
class SegNet(nn.Module):
    def __init__(self, in_chn=3, out_chn=32, BN_momentum=0.5):
        super(SegNet, self).__init__()

        #SegNet Architecture
        #Takes input of size in_chn = 3 (RGB images have 3 channels)
        #Outputs size label_chn (N # of classes)

        #ENCODING consists of 5 stages
        #Stage 1, 2 has 2 layers of Convolution + Batch Normalization + Max Pool respectively
        #Stage 3, 4, 5 has 3 layers of Convolution + Batch Normalization + Max Pool respectively

        #General Max Pool 2D for ENCODING layers
        #Pooling indices are stored for Upsampling in DECODING layers

        self.in_chn = in_chn
        self.out_chn = out_chn

        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True) 

        self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn53 = nn.BatchNorm2d(512, momentum=BN_momentum)


        #DECODING consists of 5 stages
        #Each stage corresponds to their respective counterparts in ENCODING

        #General Max Pool 2D/Upsampling for DECODING layers
        self.MaxDe = nn.MaxUnpool2d(2, stride=2) 

        self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe53 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe51 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe43 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.BNDe41 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe33 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.BNDe31 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe22 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.BNDe21 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDe12 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvDe11 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)
        self.BNDe11 = nn.BatchNorm2d(self.out_chn, momentum=BN_momentum)

    def forward(self, x):

        #ENCODE LAYERS
        #Stage 1
        x = F.relu(self.BNEn11(self.ConvEn11(x))) 
        x = F.relu(self.BNEn12(self.ConvEn12(x))) 
        x, ind1 = self.MaxEn(x)
        size1 = x.size()

        #Stage 2
        x = F.relu(self.BNEn21(self.ConvEn21(x))) 
        x = F.relu(self.BNEn22(self.ConvEn22(x))) 
        x, ind2 = self.MaxEn(x)
        size2 = x.size()

        #Stage 3
        x = F.relu(self.BNEn31(self.ConvEn31(x))) 
        x = F.relu(self.BNEn32(self.ConvEn32(x))) 
        x = F.relu(self.BNEn33(self.ConvEn33(x)))   
        x, ind3 = self.MaxEn(x)
        size3 = x.size()

        #Stage 4
        x = F.relu(self.BNEn41(self.ConvEn41(x))) 
        x = F.relu(self.BNEn42(self.ConvEn42(x))) 
        x = F.relu(self.BNEn43(self.ConvEn43(x)))   
        x, ind4 = self.MaxEn(x)
        size4 = x.size()

        #Stage 5
        x = F.relu(self.BNEn51(self.ConvEn51(x))) 
        x = F.relu(self.BNEn52(self.ConvEn52(x))) 
        x = F.relu(self.BNEn53(self.ConvEn53(x)))   
        x, ind5 = self.MaxEn(x)
        size5 = x.size()

        #DECODE LAYERS
        #Stage 5
        x = self.MaxDe(x, ind5, output_size=size4)
        x = F.relu(self.BNDe53(self.ConvDe53(x)))
        x = F.relu(self.BNDe52(self.ConvDe52(x)))
        x = F.relu(self.BNDe51(self.ConvDe51(x)))

        #Stage 4
        x = self.MaxDe(x, ind4, output_size=size3)
        x = F.relu(self.BNDe43(self.ConvDe43(x)))
        x = F.relu(self.BNDe42(self.ConvDe42(x)))
        x = F.relu(self.BNDe41(self.ConvDe41(x)))

        #Stage 3
        x = self.MaxDe(x, ind3, output_size=size2)
        x = F.relu(self.BNDe33(self.ConvDe33(x)))
        x = F.relu(self.BNDe32(self.ConvDe32(x)))
        x = F.relu(self.BNDe31(self.ConvDe31(x)))

        #Stage 2
        x = self.MaxDe(x, ind2, output_size=size1)
        x = F.relu(self.BNDe22(self.ConvDe22(x)))
        x = F.relu(self.BNDe21(self.ConvDe21(x)))

        #Stage 1
        x = self.MaxDe(x, ind1)
        x = F.relu(self.BNDe12(self.ConvDe12(x)))
        x = self.ConvDe11(x)

        x = F.softmax(x, dim=1)

        return x


## Models for ablation studies

### Baseline model
class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, patch_size, stride, padding, in_ch, dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=stride, padding=padding)
    
    def forward(self, x):
        x = self.proj(x)
        _, _, h, w = x.shape
        x = Rearrange('b c h w -> b (h w) c')(x)
        return x, h, w

class MSFormer_v3_Baseline(nn.Module):
    def __init__(self, input_dim, embed_size=100, img_size=257, config={'model': {'nChannel': 10}}):
        super().__init__()
        self.dim = embed_size
        
        # Single-scale embedding
        self.patch_embed = OverlapPatchEmbeddings(patch_size=3, stride=1, padding=1, in_ch=input_dim, dim=self.dim)
        
        # Convolutional layer instead of transformer
        self.conv = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
        
        # Output projection
        self.linear_down = nn.Linear(self.dim, config['model']['nChannel'], bias=False)
    
    def forward(self, x):
        # Single-scale embedding
        x, h, w = self.patch_embed(x)
        x = Rearrange('b (h w) d -> b d h w', h=h, w=w)(x)
        
        # Convolutional processing
        x = self.conv(x)
        
        # Output projection
        out = self.linear_down(x.permute(0, 2, 3, 1))
        out = out.permute(0, 3, 1, 2)
        return out

## SAE only
class MSFormer_v3_SAE_Only(nn.Module):
    def __init__(self, input_dim, embed_size=100, img_size=257, config={'model': {'nChannel': 10}}):
        super().__init__()
        self.dim = embed_size
        
        # Multi-scale embeddings
        self.patch_embed_f = OverlapPatchEmbeddings(patch_size=1, stride=1, padding=0, in_ch=input_dim, dim=self.dim)
        self.patch_embed_s = OverlapPatchEmbeddings(patch_size=3, stride=1, padding=1, in_ch=input_dim, dim=self.dim)
        self.patch_embed_l = OverlapPatchEmbeddings(patch_size=3, stride=2, padding=1, in_ch=input_dim, dim=self.dim)
        
        # Convolutional layers
        self.conv_f = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
        self.conv_s = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
        self.conv_l = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
        
        # Output projection
        self.linear_down = nn.Linear(self.dim, config['model']['nChannel'], bias=False)
    
    def forward(self, x):
        # Fine scale
        x_f, h_f, w_f = self.patch_embed_f(x)
        x_f = Rearrange('b (h w) d -> b d h w', h=h_f, w=w_f)(x_f)
        x_f = self.conv_f(x_f)
        
        # Small scale
        x_s, h_s, w_s = self.patch_embed_s(x)
        x_s = Rearrange('b (h w) d -> b d h w', h=h_s, w=w_s)(x_s)
        x_s = self.conv_s(x_s)
        
        # Large scale
        x_l, h_l, w_l = self.patch_embed_l(x)
        x_l = Rearrange('b (h w) d -> b d h w', h=h_l, w=w_l)(x_l)
        x_l = self.conv_l(x_l)
        x_l = F.interpolate(x_l, size=(h_s, w_s), mode='bilinear', align_corners=False)
        
        # Simple fusion: averaging
        fused_features = (x_f + x_s + x_l) / 3
        
        # Ensure batch dimension is preserved in the final projection
        B, C, H, W = fused_features.shape
        out = fused_features.permute(0, 2, 3, 1)  # [B, H, W, C]
        out = self.linear_down(out)  # [B, H, W, nChannel]
        out = out.permute(0, 3, 1, 2)  # [B, nChannel, H, W]
        # print('Output shape', out.shape)
        return out

## SAE_DAT
class MSFormer_v3_SAE_DAT(nn.Module):
    def __init__(self, input_dim, embed_size=100, img_size=257, config={'model': {'nChannel': 10}}):
        super().__init__()
        self.dim = embed_size
        
        # Multi-scale embeddings
        self.patch_embed_f = OverlapPatchEmbeddings(patch_size=1, stride=1, padding=0, in_ch=input_dim, dim=self.dim)
        self.patch_embed_s = OverlapPatchEmbeddings(patch_size=3, stride=1, padding=1, in_ch=input_dim, dim=self.dim)
        self.patch_embed_l = OverlapPatchEmbeddings(patch_size=3, stride=2, padding=1, in_ch=input_dim, dim=self.dim)
        
        # Transformer blocks
        self.block_f = nn.ModuleList([EfficientTransformerBlock_v2(self.dim, self.dim, self.dim, head_count=1, token_mlp='mix_skip')])
        self.block_s = nn.ModuleList([EfficientTransformerBlock_v2(self.dim, self.dim, self.dim, head_count=1, token_mlp='mix_skip')])
        self.block_l = nn.ModuleList([EfficientTransformerBlock_v2(self.dim, self.dim, self.dim, head_count=1, token_mlp='mix_skip')])
        
        # Output projection
        self.linear_down = nn.Linear(self.dim, config['model']['nChannel'], bias=False)
    
    def forward(self, x):
        # Fine scale
        x_f, h_f, w_f = self.patch_embed_f(x)
        x_f = Rearrange('b (h w) d -> b (h w) d', h=h_f, w=w_f)(x_f)
        for block in self.block_f:
            x_f, _, _ = block(x_f, h_f, w_f)
        x_f = Rearrange('b (h w) d -> b d h w', h=h_f, w=w_f)(x_f)
        
        # Small scale
        x_s, h_s, w_s = self.patch_embed_s(x)
        x_s = Rearrange('b (h w) d -> b (h w) d', h=h_s, w=w_s)(x_s)
        for block in self.block_s:
            x_s, _, _ = block(x_s, h_s, w_s)
        x_s = Rearrange('b (h w) d -> b d h w', h=h_s, w=w_s)(x_s)
        
        # Large scale
        x_l, h_l, w_l = self.patch_embed_l(x)
        x_l = Rearrange('b (h w) d -> b (h w) d', h=h_l, w=w_l)(x_l)
        for block in self.block_l:
            x_l, _, _ = block(x_l, h_l, w_l)
        x_l = Rearrange('b (h w) d -> b d h w', h=h_l, w=w_l)(x_l)
        x_l = F.interpolate(x_l, size=(h_s, w_s), mode='bilinear', align_corners=False)
        
        # Simple fusion: averaging
        fused_features = (x_f + x_s + x_l) / 3
        
        # Output projection
        out = self.linear_down(fused_features.permute(0, 2, 3, 1))
        out = out.permute(0, 3, 1, 2)
        return out


## SAE_AGF
class ConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, h, w):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x, None, None  # Return None for context and attention maps, as they are not used

class MSFormer_v3_SAE_AGF(nn.Module):
    def __init__(self, input_dim, embed_size=100, img_size=257, config={'model': {'nChannel': 10}}):
        super().__init__()
        self.dim = embed_size
        
        # Multi-scale embeddings (SAE)
        self.patch_embed_f = OverlapPatchEmbeddings(patch_size=1, stride=1, padding=0, in_ch=input_dim, dim=self.dim)
        self.patch_embed_s = OverlapPatchEmbeddings(patch_size=3, stride=1, padding=1, in_ch=input_dim, dim=self.dim)
        self.patch_embed_l = OverlapPatchEmbeddings(patch_size=3, stride=2, padding=1, in_ch=input_dim, dim=self.dim)
        
        # Convolutional blocks instead of transformers
        self.conv_block_f = ConvBlock(self.dim)
        self.conv_block_s = ConvBlock(self.dim)
        self.conv_block_l = ConvBlock(self.dim)
        
        # Upsampling and projection for large scale
        self.linear_up = nn.Linear(self.dim, 4 * self.dim, bias=False)
        self.proj_l = nn.Conv2d(4 * self.dim, self.dim, kernel_size=1)
        
        # Attention-Guided Fusion (AGF)
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(6 * self.dim, 3, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.linear_down = nn.Linear(self.dim, config['model']['nChannel'], bias=False)
    
    def forward(self, x):
        # Fine scale
        x_f, h_f, w_f = self.patch_embed_f(x)
        x_f = Rearrange('b (h w) d -> b d h w', h=h_f, w=w_f)(x_f)
        x_f, _, _ = self.conv_block_f(x_f, h_f, w_f)
        
        # Small scale
        x_s, h_s, w_s = self.patch_embed_s(x)
        x_s = Rearrange('b (h w) d -> b d h w', h=h_s, w=w_s)(x_s)
        x_s, _, _ = self.conv_block_s(x_s, h_s, w_s)
        
        # Large scale
        x_l, h_l, w_l = self.patch_embed_l(x)
        x_l = Rearrange('b (h w) d -> b d h w', h=h_l, w=w_l)(x_l)
        x_l, _, _ = self.conv_block_l(x_l, h_l, w_l)
        
        # Upsample large scale
        B, C_l, _, _ = x_l.shape
        x_l = x_l.permute(0, 2, 3, 1).reshape(B, h_l * w_l, C_l)
        x_l = self.linear_up(x_l)
        x_l = x_l.view(B, h_l, w_l, 4 * self.dim).permute(0, 3, 1, 2)
        x_l = F.interpolate(x_l, size=(h_s, w_s), mode='bilinear', align_corners=False)
        
        # Concatenate features for fusion
        x_cat = torch.cat([x_l, x_s, x_f], dim=1)  # B × (6*dim) × h_s × w_s
        
        # Compute attention weights
        attention_weights = self.attention_fusion(x_cat)  # B × 3 × h_s × w_s
        
        # Split concatenated features
        x_l_split = x_cat[:, :4 * self.dim, :, :]
        x_s_split = x_cat[:, 4 * self.dim:5 * self.dim, :, :]
        x_f_split = x_cat[:, 5 * self.dim:, :, :]
        
        # Project large scale
        x_l_proj = self.proj_l(x_l_split)
        
        # Apply attention weights
        weighted_x_l = x_l_proj * attention_weights[:, 0:1, :, :]
        weighted_x_s = x_s_split * attention_weights[:, 1:2, :, :]
        weighted_x_f = x_f_split * attention_weights[:, 2:3, :, :]
        
        # Sum weighted features
        fused_features = weighted_x_l + weighted_x_s + weighted_x_f
        
        # Output projection
        out = self.linear_down(fused_features.permute(0, 2, 3, 1))
        out = out.permute(0, 3, 1, 2)
        return out
    

