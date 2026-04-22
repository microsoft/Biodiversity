# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Union

from . import dla as dla_modules

# Swin Transformer Utilities
def window_partition(x, window_size: int):
    """
    x: (B, H, W, C)
    return: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    windows: (num_windows*B, window_size, window_size, C)
    return: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def _build_shifted_window_mask(Hp, Wp, window_size, shift_size, device, dtype):
    """
    Build attention mask for SW-MSA.
    Returns: (nW, ws*ws, ws*ws) with 0 or -inf values.
    """
    if shift_size == 0:
        return None

    img_mask = torch.zeros((1, Hp, Wp, 1), device=device, dtype=dtype)  # (1, Hp, Wp, 1)
    cnt = 0
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)  # (nW, ws, ws, 1)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class DropPath(nn.Module):
    """Stochastic depth: drop paths per sample on residual branches."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class WindowAttention(nn.Module):
    """Window-based Multi-head Self-Attention with relative position bias."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Compute relative position indices
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        x: (nW*B, N, C)
        mask: (nW, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, N, N
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block (W-MSA / SW-MSA + MLP + DropPath) """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path and drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, mask_matrix: Optional[torch.Tensor] = None):
        """
        x: (B, H, W, C)
        mask_matrix: optional; if None and shift > 0, generated internally
        """
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # Pad to multiples of window_size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = nn.functional.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift and mask
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
            if attn_mask is None:
                attn_mask = _build_shifted_window_mask(
                    Hp, Wp, self.window_size, self.shift_size, x.device, x.dtype
                )
        else:
            shifted_x = x
            attn_mask = None

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, ws, ws, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (nW*B, N, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # (nW*B, N, C)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # (B, H', W', C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        # Residual connection + MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MultiscaleSwinTransformer(nn.Module):
    """
    Multi-scale Swin Transformer with per-scale residual connections.

    Args:
        channels_list: Channels per level (e.g., [16, 32, 64, 128, 256, 512])
        window_sizes: Window size per level (default: [8,8,8,4,4,4])
        num_heads_list: Attention heads per level (default: [1,1,2,4,8,16])
        num_layers_per_scale: Blocks per level, int or list (default: [0,1,2,2,2,3])
    """
    def __init__(
        self,
        channels_list: List[int],
        window_sizes: Optional[List[int]] = None,
        num_heads_list: Optional[List[int]] = None,
        num_layers_per_scale: Optional[Union[int, List[int]]] = None,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        self.num_scales = len(channels_list)

        # Default hyperparameters
        if window_sizes is None:
            window_sizes = [8, 8, 8, 4, 4, 4]
        if num_heads_list is None:
            num_heads_list = [1, 1, 2, 4, 8, 16]
        if num_layers_per_scale is None:
            num_layers_per_scale = [0, 1, 2, 2, 2, 3]

        # Allow single int to be replicated across scales
        if isinstance(num_layers_per_scale, int):
            num_layers_per_scale = [num_layers_per_scale] * self.num_scales

        assert len(channels_list) == len(window_sizes) == len(num_heads_list) == len(num_layers_per_scale), \
            "channels_list, window_sizes, num_heads_list, and num_layers_per_scale must have the same length"

        def _fit_heads(channels, prefer_heads):
            # Adjust heads so that channels % heads == 0
            h = min(prefer_heads, channels)
            while channels % h != 0 and h > 1:
                h -= 1
            return max(h, 1)

        self.swin_blocks = nn.ModuleList()
        for C, ws, h, nl in zip(channels_list, window_sizes, num_heads_list, num_layers_per_scale):
            num_heads = _fit_heads(C, h)
            # Build blocks alternating shift=0 and shift=ws//2
            scale_blocks = nn.ModuleList([
                SwinTransformerBlock(
                    dim=C,
                    num_heads=num_heads,
                    window_size=ws,
                    shift_size=0 if j % 2 == 0 else ws // 2,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=drop_path_rate
                ) for j in range(nl)
            ])
            self.swin_blocks.append(scale_blocks)

    def forward(self, feature_list: List[torch.Tensor]):
        """
        Args:
            feature_list: List of tensors (B, C, H, W)
        Returns:
            List of enhanced tensors (B, C, H, W) with per-scale residual
        """
        enhanced_features = []
        for features, scale_blocks in zip(feature_list, self.swin_blocks):
            if len(scale_blocks) == 0:
                # No Swin at this scale, passthrough
                enhanced_features.append(features)
                continue

            x = features.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
            for block in scale_blocks:
                x = block(x, mask_matrix=None)
            x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

            # Per-scale residual connection
            enhanced_x = features + x
            enhanced_features.append(enhanced_x)

        return enhanced_features


class OWLT_Architecture(nn.Module):
    """
    OWL-T hybrid architecture:
      - DLA backbone with multi-level features
      - Multi-scale Swin Transformer for feature refinement
      - DLAUp decoder from first_level
      - Localization head (heatmap output)
    """

    def __init__(
        self,
        num_layers: int = 34,
        pretrained_cnn: bool = True,
        down_ratio: Optional[int] = 2,
        head_conv: int = 64,
        swin_num_layers_per_scale: Optional[Union[int, List[int]]] = None,
        swin_window_sizes: Optional[List[int]] = None,
        swin_num_heads: Optional[List[int]] = None,
        drop_path_rate: float = 0.1
    ):
        super().__init__()

        assert down_ratio in [1, 2, 4, 8, 16], f"down_ratio must be 1/2/4/8/16, got {down_ratio}"
        base_name = f'dla{num_layers}'

        self.down_ratio = down_ratio
        self.head_conv = head_conv
        self.first_level = int(np.log2(down_ratio))

        # DLA backbone
        base = dla_modules.__dict__[base_name](pretrained=pretrained_cnn, return_levels=True)
        setattr(self, 'base_0', base)
        setattr(self, 'channels_0', base.channels)
        channels = self.channels_0  # e.g., [16, 32, 64, 128, 256, 512]

        # Multi-scale Swin Transformer
        if swin_window_sizes is None:
            swin_window_sizes = [8, 8, 8, 4, 4, 4]
        if swin_num_heads is None:
            swin_num_heads = [1, 1, 2, 4, 8, 16]
        if swin_num_layers_per_scale is None:
            swin_num_layers_per_scale = [0, 1, 2, 2, 2, 3]  # L0→L5; L0=0 dado down_ratio=2

        self.multiscale_swin = MultiscaleSwinTransformer(
            channels_list=channels,
            window_sizes=swin_window_sizes,
            num_heads_list=swin_num_heads,
            num_layers_per_scale=swin_num_layers_per_scale,
            drop_path_rate=drop_path_rate
        )

        # DLAUp decoder from first_level
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = dla_modules.DLAUp(channels[self.first_level:], scales=scales)

        # Bottleneck convolution at last scale
        self.bottleneck_conv = nn.Conv2d(
            channels[-1], channels[-1],
            kernel_size=1, stride=1,
            padding=0, bias=True
        )

        # Localization head
        self.loc_head = nn.Sequential(
            nn.Conv2d(channels[self.first_level], head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        #self.loc_head[-2].bias.data.fill_(0.00)

    def forward(self, input: torch.Tensor):
        # Multi-scale features from DLA backbone
        encode = self.base_0(input)

        # Bottleneck at last level
        encode[-1] = self.bottleneck_conv(encode[-1])

        # Refine with Swin (includes per-scale residual)
        enhanced_encode = self.multiscale_swin(encode)

        # Decode from first_level
        decode_hm = self.dla_up(enhanced_encode[self.first_level:])
        heatmap = self.loc_head(decode_hm)
        return heatmap

    def freeze(self, layers: list) -> None:
        """Freeze layers by attribute name."""
        for layer in layers:
            self._freeze_layer(layer)

    def _freeze_layer(self, layer_name: str) -> None:
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False
