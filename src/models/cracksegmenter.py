"""
CrackSegmenter (Multi-Scale Transformer) models for crack segmentation.

This module contains the proposed CrackSegmenter architecture variants:
- MSFormer_SAE_AGF: Scale-Aware Embedding + Attention-Guided Fusion
- MSFormer_SAE: Scale-Aware Embedding only
- MSFormer_AGF: Attention-Guided Fusion only
- MSFormer_v1, v2, v3: Different architectural variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from .utils import OverlapPatchEmbeddings, EfficientTransformerBlock, EfficientTransformerBlock_v2


class ConvBlock_v2(nn.Module):
    """Enhanced convolutional block for CrackSegmenter models."""
    
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


class MSFormer_SAE_AGF(nn.Module):
    """
    CrackSegmenter with Scale-Aware Embedding (SAE) and Attention-Guided Fusion (AGF).
    
    This is the main proposed architecture that combines multi-scale feature
    extraction with intelligent feature fusion.
    """
    
    def __init__(self, input_dim, embed_size=100, img_size=448):
        super().__init__()
        self.dim = embed_size
        self.patch_size = 16
        
        # Three-scale patch embeddings (SAE)
        self.patch_embed_f = OverlapPatchEmbeddings(
            patch_size=1, stride=1, padding=0, in_ch=input_dim, dim=self.dim
        )
        self.patch_embed_s = OverlapPatchEmbeddings(
            patch_size=3, stride=1, padding=1, in_ch=input_dim, dim=self.dim
        )
        self.patch_embed_l = OverlapPatchEmbeddings(
            patch_size=3, stride=2, padding=1, in_ch=input_dim, dim=self.dim
        )
        
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
        self.linear_down = nn.Linear(self.dim, 100, bias=False)  # nChannel from config
    
    def forward(self, x):
        # Three-scale embeddings
        x_f, h_f, w_f = self.patch_embed_f(x)
        x_f = Rearrange('b (h w) d -> b d h w', h=h_f, w=w_f)(x_f)
        x_s, h_s, w_s = self.patch_embed_s(x)
        x_s = Rearrange('b (h w) d -> b d h s', h=h_s, w=w_s)(x_s)
        x_l, h_l, w_l = self.patch_embed_l(x)
        x_l = Rearrange('b (h w) d -> b d h w', h=h_l, w=w_l)(x_l)
        
        # Apply conv blocks
        x_f = self.conv_f(x_f)
        x_s = self.conv_s(x_s)
        x_l = self.conv_l(x_l)
        
        # Up-projection for large branch
        x_l_flat = x_l.flatten(2).transpose(1, 2)
        x_l_up = self.linear_up(x_l_flat)
        x_l_up = x_l_up.transpose(1, 2).view(x_l.size(0), -1, h_l, w_l)
        x_l_up = self.proj_l(x_l_up)
        
        # Resize features to match finest scale
        x_s_resized = F.interpolate(x_s, size=(h_f, w_f), mode='bilinear', align_corners=False)
        x_l_resized = F.interpolate(x_l_up, size=(h_f, w_f), mode='bilinear', align_corners=False)
        
        # Concatenate all features
        x_concat = torch.cat([x_f, x_s_resized, x_l_resized], dim=1)
        
        # Attention-based fusion
        attention_weights = self.attention_fusion(x_concat)
        
        # Apply attention weights
        x_f_att = x_f * attention_weights[:, 0:1, :, :]
        x_s_att = x_s_resized * attention_weights[:, 1:2, :, :]
        x_l_att = x_l_resized * attention_weights[:, 2:3, :, :]
        
        # Final fusion
        x_final = x_f_att + x_s_att + x_l_att
        
        # Final prediction
        x_final_flat = x_final.flatten(2).transpose(1, 2)
        output = self.linear_down(x_final_flat)
        output = output.transpose(1, 2).view(x_final.size(0), -1, h_f, w_f)
        
        return output, None, None, None, None, None, None, None


class MSFormer_SAE(nn.Module):
    """CrackSegmenter with Scale-Aware Embedding only."""
    
    def __init__(self, input_dim, embed_size=100, img_size=448):
        super().__init__()
        self.dim = embed_size
        
        # Multi-scale patch embeddings
        self.patch_embed_f = OverlapPatchEmbeddings(
            patch_size=1, stride=1, padding=0, in_ch=input_dim, dim=self.dim
        )
        self.patch_embed_s = OverlapPatchEmbeddings(
            patch_size=3, stride=1, padding=1, in_ch=input_dim, dim=self.dim
        )
        self.patch_embed_l = OverlapPatchEmbeddings(
            patch_size=3, stride=2, padding=1, in_ch=input_dim, dim=self.dim
        )
        
        # Simple fusion without attention
        self.fusion_conv = nn.Conv2d(3 * self.dim, self.dim, kernel_size=1)
        self.linear_down = nn.Linear(self.dim, 100, bias=False)
    
    def forward(self, x):
        # Extract multi-scale features
        x_f, h_f, w_f = self.patch_embed_f(x)
        x_f = Rearrange('b (h w) d -> b d h w', h=h_f, w=w_f)(x_f)
        x_s, h_s, w_s = self.patch_embed_s(x)
        x_s = Rearrange('b (h w) d -> b d h w', h=h_s, w=w_s)(x_s)
        x_l, h_l, w_l = self.patch_embed_l(x)
        x_l = Rearrange('b (h w) d -> b d h w', h=h_l, w=w_l)(x_l)
        
        # Resize to finest scale
        x_s_resized = F.interpolate(x_s, size=(h_f, w_f), mode='bilinear', align_corners=False)
        x_l_resized = F.interpolate(x_l, size=(h_f, w_f), mode='bilinear', align_corners=False)
        
        # Concatenate and fuse
        x_concat = torch.cat([x_f, x_s_resized, x_l_resized], dim=1)
        x_fused = self.fusion_conv(x_concat)
        
        # Final prediction
        x_final_flat = x_fused.flatten(2).transpose(1, 2)
        output = self.linear_down(x_final_flat)
        output = output.transpose(1, 2).view(x_fused.size(0), -1, h_f, w_f)
        
        return output, None, None, None, None, None, None, None


class MSFormer_AGF(nn.Module):
    """CrackSegmenter with Attention-Guided Fusion only."""
    
    def __init__(self, input_dim, embed_size=100, img_size=448):
        super().__init__()
        self.dim = embed_size
        
        # Single-scale embedding
        self.patch_embed = OverlapPatchEmbeddings(
            patch_size=3, stride=1, padding=1, in_ch=input_dim, dim=self.dim
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(self.dim, num_heads=8, batch_first=True)
        
        # Final prediction
        self.linear_down = nn.Linear(self.dim, 100, bias=False)
    
    def forward(self, x):
        # Single-scale embedding
        x_embed, h, w = self.patch_embed(x)
        
        # Apply attention
        x_attended, _ = self.attention(x_embed, x_embed, x_embed)
        
        # Reshape and predict
        x_reshaped = x_attended.transpose(1, 2).view(x.size(0), self.dim, h, w)
        x_final_flat = x_reshaped.flatten(2).transpose(1, 2)
        output = self.linear_down(x_final_flat)
        output = output.transpose(1, 2).view(x_reshaped.size(0), -1, h, w)
        
        return output, None, None, None, None, None, None, None


class MSFormer_v1(nn.Module):
    """CrackSegmenter version 1 - Basic multi-scale architecture."""
    
    def __init__(self, input_dim, embed_size=100, img_size=448):
        super().__init__()
        self.dim = embed_size
        
        # Multi-scale embeddings
        self.patch_embed_f = OverlapPatchEmbeddings(
            patch_size=1, stride=1, padding=0, in_ch=input_dim, dim=self.dim
        )
        self.patch_embed_s = OverlapPatchEmbeddings(
            patch_size=3, stride=1, padding=1, in_ch=input_dim, dim=self.dim
        )
        
        # Simple fusion
        self.fusion = nn.Conv2d(2 * self.dim, self.dim, kernel_size=1)
        self.linear_down = nn.Linear(self.dim, 100, bias=False)
    
    def forward(self, x):
        # Extract features at two scales
        x_f, h_f, w_f = self.patch_embed_f(x)
        x_f = Rearrange('b (h w) d -> b d h w', h=h_f, w=w_f)(x_f)
        x_s, h_s, w_s = self.patch_embed_s(x)
        x_s = Rearrange('b (h w) d -> b d h w', h=h_s, w=w_s)(x_s)
        
        # Resize and fuse
        x_s_resized = F.interpolate(x_s, size=(h_f, w_f), mode='bilinear', align_corners=False)
        x_concat = torch.cat([x_f, x_s_resized], dim=1)
        x_fused = self.fusion(x_concat)
        
        # Predict
        x_final_flat = x_fused.flatten(2).transpose(1, 2)
        output = self.linear_down(x_final_flat)
        output = output.transpose(1, 2).view(x_fused.size(0), -1, h_f, w_f)
        
        return output, None, None, None, None, None, None, None


class MSFormer_v2(nn.Module):
    """CrackSegmenter version 2 - Enhanced multi-scale with transformers."""
    
    def __init__(self, input_dim, embed_size=100, img_size=448):
        super().__init__()
        self.dim = embed_size
        
        # Multi-scale embeddings
        self.patch_embed_f = OverlapPatchEmbeddings(
            patch_size=1, stride=1, padding=0, in_ch=input_dim, dim=self.dim
        )
        self.patch_embed_s = OverlapPatchEmbeddings(
            patch_size=3, stride=1, padding=1, in_ch=input_dim, dim=self.dim
        )
        self.patch_embed_l = OverlapPatchEmbeddings(
            patch_size=3, stride=2, padding=1, in_ch=input_dim, dim=self.dim
        )
        
        # Transformer blocks
        self.transformer_f = EfficientTransformerBlock(self.dim)
        self.transformer_s = EfficientTransformerBlock(self.dim)
        self.transformer_l = EfficientTransformerBlock(self.dim)
        
        # Fusion
        self.fusion = nn.Conv2d(3 * self.dim, self.dim, kernel_size=1)
        self.linear_down = nn.Linear(self.dim, 100, bias=False)
    
    def forward(self, x):
        # Extract and process features
        x_f, h_f, w_f = self.patch_embed_f(x)
        x_f = self.transformer_f(x_f)
        x_f = Rearrange('b (h w) d -> b d h w', h=h_f, w=w_f)(x_f)
        
        x_s, h_s, w_s = self.patch_embed_s(x)
        x_s = self.transformer_s(x_s)
        x_s = Rearrange('b (h w) d -> b d h w', h=h_s, w=w_s)(x_s)
        
        x_l, h_l, w_l = self.patch_embed_l(x)
        x_l = self.transformer_l(x_l)
        x_l = Rearrange('b (h w) d -> b d h w', h=h_l, w=w_l)(x_l)
        
        # Resize and fuse
        x_s_resized = F.interpolate(x_s, size=(h_f, w_f), mode='bilinear', align_corners=False)
        x_l_resized = F.interpolate(x_l, size=(h_f, w_f), mode='bilinear', align_corners=False)
        
        x_concat = torch.cat([x_f, x_s_resized, x_l_resized], dim=1)
        x_fused = self.fusion(x_concat)
        
        # Predict
        x_final_flat = x_fused.flatten(2).transpose(1, 2)
        output = self.linear_down(x_final_flat)
        output = output.transpose(1, 2).view(x_fused.size(0), -1, h_f, w_f)
        
        return output, None, None, None, None, None, None, None


class MSFormer_v3(nn.Module):
    """CrackSegmenter version 3 - Advanced architecture with enhanced fusion."""
    
    def __init__(self, input_dim, embed_size=100, img_size=448):
        super().__init__()
        self.dim = embed_size
        
        # Multi-scale embeddings
        self.patch_embed_f = OverlapPatchEmbeddings(
            patch_size=1, stride=1, padding=0, in_ch=input_dim, dim=self.dim
        )
        self.patch_embed_s = OverlapPatchEmbeddings(
            patch_size=3, stride=1, padding=1, in_ch=input_dim, dim=self.dim
        )
        self.patch_embed_l = OverlapPatchEmbeddings(
            patch_size=3, stride=2, padding=1, in_ch=input_dim, dim=self.dim
        )
        
        # Enhanced transformer blocks
        self.transformer_f = EfficientTransformerBlock_v2(self.dim)
        self.transformer_s = EfficientTransformerBlock_v2(self.dim)
        self.transformer_l = EfficientTransformerBlock_v2(self.dim)
        
        # Advanced fusion with attention
        self.fusion_attention = nn.MultiheadAttention(self.dim, num_heads=8, batch_first=True)
        self.fusion_conv = nn.Conv2d(3 * self.dim, self.dim, kernel_size=1)
        
        # Final prediction
        self.linear_down = nn.Linear(self.dim, 100, bias=False)
    
    def forward(self, x):
        # Extract and process features
        x_f, h_f, w_f = self.patch_embed_f(x)
        x_f = self.transformer_f(x_f)
        x_f = Rearrange('b (h w) d -> b d h w', h=h_f, w=w_f)(x_f)
        
        x_s, h_s, w_s = self.patch_embed_s(x)
        x_s = self.transformer_s(x_s)
        x_s = Rearrange('b (h w) d -> b d h w', h=h_s, w=w_s)(x_s)
        
        x_l, h_l, w_l = self.patch_embed_l(x)
        x_l = self.transformer_l(x_l)
        x_l = Rearrange('b (h w) d -> b d h w', h=h_l, w=w_l)(x_l)
        
        # Resize to finest scale
        x_s_resized = F.interpolate(x_s, size=(h_f, w_f), mode='bilinear', align_corners=False)
        x_l_resized = F.interpolate(x_l, size=(h_f, w_f), mode='bilinear', align_corners=False)
        
        # Apply attention-based fusion
        x_f_flat = x_f.flatten(2).transpose(1, 2)
        x_s_flat = x_s_resized.flatten(2).transpose(1, 2)
        x_l_flat = x_l_resized.flatten(2).transpose(1, 2)
        
        # Concatenate for attention
        x_all = torch.cat([x_f_flat, x_s_flat, x_l_flat], dim=1)
        x_attended, _ = self.fusion_attention(x_all, x_all, x_all)
        
        # Split back and reshape
        split_size = x_f_flat.size(1)
        x_f_att = x_attended[:, :split_size].transpose(1, 2).view(x_f.size(0), self.dim, h_f, w_f)
        x_s_att = x_attended[:, split_size:2*split_size].transpose(1, 2).view(x_s_resized.size(0), self.dim, h_f, w_f)
        x_l_att = x_attended[:, 2*split_size:].transpose(1, 2).view(x_l_resized.size(0), self.dim, h_f, w_f)
        
        # Final fusion
        x_concat = torch.cat([x_f_att, x_s_att, x_l_att], dim=1)
        x_fused = self.fusion_conv(x_concat)
        
        # Predict
        x_final_flat = x_fused.flatten(2).transpose(1, 2)
        output = self.linear_down(x_final_flat)
        output = output.transpose(1, 2).view(x_fused.size(0), -1, h_f, w_f)
        
        return output, None, None, None, None, None, None, None
