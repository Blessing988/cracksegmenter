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
        
        
## Introduced another scale for the patch embedding( Fine scale)
class MSFormer_v1(nn.Module):
    def __init__(self, input_dim, embed_size=100, img_size=257,
                 large_nlayers=[1], small_nlayers=[1], fine_nlayers=[1]):
        super(MSFormer_v1, self).__init__()
        
        # ---- core dimensions ----
        self.dim         = embed_size
        self.dim_scale   = 2
        self.patch_size  = 32
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


## Introduced another scale + modified Efficient Transformer Block with directed convolutions
class MSFormer_v2(nn.Module):
    def __init__(self, input_dim, embed_size=100, img_size=257,
                 large_nlayers=[1], small_nlayers=[1], fine_nlayers=[1]):
        super(MSFormer_v2, self).__init__()
        
        # ---- core dimensions ----
        self.dim         = embed_size
        self.dim_scale   = 2
        self.patch_size  = 4
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