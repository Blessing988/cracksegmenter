import torch
import torch.nn as nn
from typing import Tuple
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F


class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, padding=1, in_ch=3, dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px = self.proj(x)
        px = self.norm(px)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        return fx, H, W
        
        
class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)
    
    
class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        
    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out

    
class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W)+self.fc1(x)))
        out = self.fc2(ax)
        return out
    
    
class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# Old EfficientAttention
class EfficientAttention(nn.Module):
    """
        input  -> x:[B, D, H, W]
        output ->   [B, D, H, W]
    
        in_channels:    int -> Embedding Dimension 
        key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
        value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2) 
        head_count:     int -> It divides the embedding dimension by the head_count and process each part individually
    """
    
    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1) 
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

        
    def forward(self, input_):
        n, _, h, w = input_.size()
        
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        attention_maps = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            
            query = F.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
                        
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]            
            
            context = key @ value.transpose(1, 2) # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w) # n*dv            
            attended_values.append(attended_value)
            
            # Compute attention map for this head and direction
            attention_map = key.mean(dim=1).view(n, h, w)  # Average over channels, reshape to spatial dimensions
            attention_maps.append(attention_map)
                
        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)
        
        # Aggregate attention maps by averaging across all heads and directions
        attention_map = torch.stack(attention_maps, dim=0).mean(dim=0)  # Shape: (n, h, w)
        return attention, context, attention_map
        
        
 # New Efficient Attention from Grok
class EfficientAttention_v2(nn.Module):
    """
    Enhanced attention mechanism with directional convolutions for better detection of linear structures.
    
    Args:
        in_channels (int): Embedding dimension of the input.
        key_channels (int): Key embedding dimension (best: in_channels).
        value_channels (int): Value embedding dimension (best: in_channels or in_channels//2).
        head_count (int): Number of attention heads.
        directions (list): List of kernel sizes for directional convolutions, e.g., [(1,3), (3,1)].
    """
    def __init__(self, in_channels, key_channels, value_channels, head_count=1, directions=[(1,3), (3,1)]):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.directions = directions  # List of kernel sizes for directional convolutions

        # Directional convolutions for keys and queries
        self.keys = nn.ModuleList([
            nn.Conv2d(in_channels, key_channels, kernel_size=k, padding=(k[0]//2, k[1]//2))
            for k in directions
        ])
        self.queries = nn.ModuleList([
            nn.Conv2d(in_channels, key_channels, kernel_size=k, padding=(k[0]//2, k[1]//2))
            for k in directions
        ])
        
        # Value projection remains 1x1 convolution
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        
        # Reprojection layer adjusted for concatenated channels from all directions
        self.reprojection = nn.Conv2d(value_channels * len(directions), in_channels, 1)

    # original forward without attention maps
    # def forward(self, input_):
    #     n, _, h, w = input_.size()
        
    #     # Compute keys, queries, and values for each direction
    #     keys_list = [k(input_).reshape(n, self.key_channels, h * w) for k in self.keys]
    #     queries_list = [q(input_).reshape(n, self.key_channels, h * w) for q in self.queries]
    #     values = self.values(input_).reshape(n, self.value_channels, h * w)
        
    #     head_key_channels = self.key_channels // self.head_count
    #     head_value_channels = self.value_channels // self.head_count
        
    #     attended_values = []
    #     for i in range(self.head_count):
    #         for dir_idx, (keys, queries) in enumerate(zip(keys_list, queries_list)):
    #             key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
    #             query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
    #             value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
                
    #             context = key @ value.transpose(1, 2)  # dk*dv
    #             attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
    #             attended_values.append(attended_value)
        
    #     # Aggregate attended values from all directions and heads
    #     aggregated_values = torch.cat(attended_values, dim=1)
        
    #     # Reproject to original dimension
    #     attention = self.reprojection(aggregated_values)
        
    #     return attention, context      

    def forward(self, input_):
        n, _, h, w = input_.size()
        
        # Compute keys, queries, and values for each direction
        keys_list = [k(input_).reshape(n, self.key_channels, h * w) for k in self.keys]
        queries_list = [q(input_).reshape(n, self.key_channels, h * w) for q in self.queries]
        values = self.values(input_).reshape(n, self.value_channels, h * w)
        
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        attention_maps = []  # List to store attention maps for each head and direction
        
        # Process each head and direction
        for i in range(self.head_count):
            for dir_idx, (keys, queries) in enumerate(zip(keys_list, queries_list)):
                # Compute softened keys and queries for the current head
                key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
                query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
                value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
                
                # Compute context and attended value
                context = key @ value.transpose(1, 2)  # Shape: (n, dk, dv)
                attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
                attended_values.append(attended_value)
                
                # Compute attention map for this head and direction
                attention_map = key.mean(dim=1).view(n, h, w)  # Average over channels, reshape to spatial dimensions
                attention_maps.append(attention_map)
        
        # Aggregate attended values from all directions and heads
        aggregated_values = torch.cat(attended_values, dim=1)
        
        # Reproject to original dimension
        attention = self.reprojection(aggregated_values)
        
        # Aggregate attention maps by averaging across all heads and directions
        attention_map = torch.stack(attention_maps, dim=0).mean(dim=0)  # Shape: (n, h, w)
        
        return attention, context, attention_map

        
class EfficientTransformerBlock(nn.Module):
    """
        Input  -> x (Size: (b, (H*W), d)), H, W
        Output -> (b, (H*W), d)
    """
    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp='mix'):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientAttention(in_channels=in_dim, key_channels=key_dim,
                                       value_channels=value_dim, head_count=1)
        self.norm2 = nn.LayerNorm(in_dim)
        if token_mlp=='mix':
            self.mlp = MixFFN(in_dim, int(in_dim*4))  
        elif token_mlp=='mix_skip':
            self.mlp = MixFFN_skip(in_dim, int(in_dim*4)) 
        else:
            self.mlp = MLP_FFN(in_dim, int(in_dim*4))
        
    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        x = Rearrange('b d h w -> b (h w) d')(x)
        norm_1 = self.norm1(x)
        norm_1 = Rearrange('b (h w) d -> b d h w', h=H, w=W)(norm_1)
        
        attn, context, attention_map = self.attn(norm_1)
        attn = Rearrange('b d h w -> b (h w) d')(attn)
        
        tx = x + attn
        mx = tx + self.mlp(self.norm2(tx), H, W)
        
        mx = Rearrange('b (h w) d -> b d h w', h=H, w=W)(mx)
        
        return mx, context, attention_map
    

## New Grok v2 EfficientTransformerBlock
class EfficientTransformerBlock_v2(nn.Module):
    """
        Input  -> x (Size: (b, (H*W), d)), H, W
        Output -> (b, (H*W), d)
    """
    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp='mix'):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientAttention_v2(in_channels=in_dim, key_channels=key_dim,
                                       value_channels=value_dim, head_count=1)
        self.norm2 = nn.LayerNorm(in_dim)
        if token_mlp=='mix':
            self.mlp = MixFFN(in_dim, int(in_dim*4))  
        elif token_mlp=='mix_skip':
            self.mlp = MixFFN_skip(in_dim, int(in_dim*4)) 
        else:
            self.mlp = MLP_FFN(in_dim, int(in_dim*4))
     
    # actual forward without attention maps    
    # def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
    #     x = Rearrange('b d h w -> b (h w) d')(x)
    #     norm_1 = self.norm1(x)
    #     norm_1 = Rearrange('b (h w) d -> b d h w', h=H, w=W)(norm_1)
        
    #     attn, context = self.attn(norm_1)
    #     attn = Rearrange('b d h w -> b (h w) d')(attn)
        
    #     tx = x + attn
    #     mx = tx + self.mlp(self.norm2(tx), H, W)
        
    #     mx = Rearrange('b (h w) d -> b d h w', h=H, w=W)(mx)
        
    #     return mx, context


    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        x = Rearrange('b d h w -> b (h w) d')(x)
        norm_1 = self.norm1(x)
        norm_1 = Rearrange('b (h w) d -> b d h w', h=H, w=W)(norm_1)
        
        attn, context, attention_map = self.attn(norm_1)
        attn = Rearrange('b d h w -> b (h w) d')(attn)
        
        tx = x + attn
        mx = tx + self.mlp(self.norm2(tx), H, W)
        
        mx = Rearrange('b (h w) d -> b d h w', h=H, w=W)(mx)
        
        return mx, context, attention_map