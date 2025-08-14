"""
Model utilities for CrackSegmenter architectures.

This module contains:
- OverlapPatchEmbeddings: Multi-scale patch embedding layers
- EfficientTransformerBlock: Efficient transformer blocks
- Model factory functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OverlapPatchEmbeddings(nn.Module):
    """Overlapping patch embeddings for multi-scale feature extraction."""
    
    def __init__(self, patch_size=16, stride=16, padding=0, in_ch=3, dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.in_ch = in_ch
        self.dim = dim
        
        # Convolutional patch embedding
        self.proj = nn.Conv2d(
            in_ch, dim, 
            kernel_size=patch_size, 
            stride=stride, 
            padding=padding
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        # Apply convolution
        x = self.proj(x)
        
        # Get spatial dimensions
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)
        
        # Apply normalization
        x = self.norm(x)
        
        return x, H, W


class EfficientTransformerBlock(nn.Module):
    """Efficient transformer block for CrackSegmenter."""
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, batch_first=True
        )
        
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path(attn_out)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class EfficientTransformerBlock_v2(nn.Module):
    """Enhanced efficient transformer block for CrackSegmenter v1/v2."""
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, batch_first=True
        )
        
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # Additional residual connection
        self.norm3 = norm_layer(dim)
    
    def forward(self, x):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop_path(attn_out)
        
        # MLP with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        # Additional normalization
        x = self.norm3(x)
        
        return x


def create_model(architecture='Crack-Segmenter-v2', **kwargs):
    """
    Create a model by architecture name.
    
    Args:
        architecture (str): Name of the architecture
        **kwargs: Additional arguments for the model
    
    Returns:
        nn.Module: Configured model
    """
    from .cracksegmenter import (
        MSFormer_SAE_AGF, MSFormer_SAE, MSFormer_AGF,
        MSFormer_v1, MSFormer_v2, MSFormer_v3
    )
    from .baselines import get_baseline_model
    
    # CrackSegmenter architectures
    cracksegmenter_models = {
        'Crack-Segmenter': MSFormer_v3,
        'Crack-Segmenter-v0': MSFormer_v1,
        'Crack-Segmenter-v1': MSFormer_v2,
        'Crack-Segmenter-v2': MSFormer_SAE_AGF,
        'Crack-Segmenter-v3': MSFormer_AGF,
    }
    
    if architecture in cracksegmenter_models:
        return cracksegmenter_models[architecture](**kwargs)
    
    # Baseline models
    try:
        return get_baseline_model(architecture.lower(), **kwargs)
    except ValueError:
        raise ValueError(f"Unsupported architecture: {architecture}")


def get_model(architecture, **kwargs):
    """
    Alias for create_model for backward compatibility.
    
    Args:
        architecture (str): Name of the architecture
        **kwargs: Additional arguments for the model
    
    Returns:
        nn.Module: Configured model
    """
    return create_model(architecture, **kwargs)
