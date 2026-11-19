import math
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven import surrogate

sys.path.insert(0, str(Path(__file__).parent.parent))
from SeqSNN.module.comrt import build_encoding_table, tune_comrt, load_design_point

__all__ = ['spikformer_clipmrt_2d']


def _build_clipmrt_2d_table(Nx: int, Ny: int, strides: List[int], bits: List[int]) -> torch.Tensor:
    """Build (Nx*Ny, 2*total_bits) ClipMRT encoding table for a 2D patch grid.

    Encodes x and y axes independently with the same strides/bits config,
    then concatenates per-patch: [clipmrt_x(i), clipmrt_y(j)].
    """
    table_x = build_encoding_table(Nx, strides, bits)  # (Nx, total_bits)
    table_y = build_encoding_table(Ny, strides, bits)  # (Ny, total_bits)

    i_idx = torch.arange(Nx).repeat_interleave(Ny)  # (N,)
    j_idx = torch.arange(Ny).repeat(Nx)             # (N,)

    enc = torch.cat([table_x[i_idx], table_y[j_idx]], dim=-1)  # (N, 2*total_bits)
    return enc


def generate_clipmrt_matrix_2d(M: torch.Tensor, strides: List[int], bits: List[int]) -> torch.Tensor:
    """Append 2D ClipMRT positional codes to Q or K tensor.

    Args:
        M: shape (T, B, H, N, D) — query or key after head split
        strides, bits: ClipMRT config for one axis (applied identically to x and y)

    Returns:
        (T, B, H, N, D + 2*total_bits)
    """
    T, B, H, N, D = M.shape
    Nx = int(math.isqrt(N))
    Ny = Nx
    assert Nx * Ny == N, f"Expected square patch grid, got N={N}"

    enc = _build_clipmrt_2d_table(Nx, Ny, strides, bits)  # (N, 2*total_bits)
    enc = enc.to(M.device)
    enc = enc.view(1, 1, 1, N, -1).expand(T, B, H, N, -1)
    return torch.cat([M, enc], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, N, C = x.shape
        x_ = x.flatten(0, 1)
        x = self.fc1_linear(x_)
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_linear(x.flatten(0, 1))
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        x = self.fc2_lif(x)
        return x


class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1,
                 strides: Optional[List[int]] = None,
                 bits: Optional[List[int]] = None):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.scale = nn.Parameter(data=torch.tensor(-4.0), requires_grad=True)

        self.strides = strides
        self.bits = bits

        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='torch')

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x):
        T, B, N, C = x.shape

        x_for_qkv = x.flatten(0, 1)  # TB, N, C

        q_linear_out = self.q_linear(x_for_qkv)
        q_linear_out = self.q_bn(q_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        q = generate_clipmrt_matrix_2d(q, self.strides, self.bits)  # T, B, H, N, C//H + 2*total_bits
        q = q.unsqueeze(3)                                           # T, B, H, 1, N, ...
        k = generate_clipmrt_matrix_2d(k, self.strides, self.bits)  # T, B, H, N, C//H + 2*total_bits
        k = k.unsqueeze(4)                                           # T, B, H, N, 1, ...
        attn = torch.sum(1 - (q - k) ** 2, dim=-1)                  # T, B, H, N, N

        attn = attn * torch.sigmoid(self.scale)

        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x)

        x = x.flatten(0, 1)
        x = self.proj_lif(
            self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C)
        )
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1,
                 strides: Optional[List[int]] = None, bits: Optional[List[int]] = None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,
                        strides=strides, bits=bits)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class ConvPE(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.rpe_conv = nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(d_model)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch',
                                        surrogate_function=surrogate.ATan())

    def forward(self, x):
        T, B, L, D = x.shape
        h = w = int(math.isqrt(L))
        x = x.transpose(-1, -2).reshape(T, B, D, h, w)
        x_feat = x.contiguous()
        x = self.rpe_conv(x.flatten(0, 1))
        x = self.rpe_bn(x).reshape(T, B, -1, h, w).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat
        x = x.flatten(-2).transpose(-2, -1)  # T, B, L, D
        return x


class SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()
        x = self.maxpool3(x)

        x = x.reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = x.flatten(-2).transpose(-1, -2)  # T, B, N, C
        return x


class Spikformer_clipmrt_2d(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=256, num_heads=8, mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=2, sr_ratios=1, T=4,
                 pe_type: str = "conv",
                 strides: Optional[List[int]] = None,
                 bits: Optional[List[int]] = None,
                 design_point_path: Optional[str] = None,
                 design_point_key: Optional[str] = None,
                 tune_axis_len: Optional[int] = None,
                 tune_seed: int = 0):
        """
        Args:
            strides, bits: Fixed ClipMRT config per axis. If None, falls through to design_point_path or tune_comrt.
            design_point_path: Path to a design point JSON. Accepts either a flat file
                                ({"strides": [...], "bits": [...], ...}, e.g. design_point_target0p9.json)
                                or a multi-entry file keyed by score
                                ({"0.85": {"strides": [...], "bits": [...], ...}, ...}, e.g. design_points_max_L12.json).
            design_point_key: For a multi-entry design_point_path, the key to select (e.g. "0.85").
                               Required when the file has more than one entry.
            tune_axis_len: Axis length for tuning when no strides/bits/path supplied.
                           Defaults to img_size_h // 4 (accounting for SPS 4× downsampling).
            pe_type: Spatial PE applied to the token sequence before blocks ("conv" or "none").
        """
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.depths = depths

        if strides is not None and bits is not None:
            self.strides = list(strides)
            self.bits = list(bits)
        elif design_point_path is not None:
            self.strides, self.bits, resolved_key = load_design_point(design_point_path, design_point_key)
            key_suffix = f" [{resolved_key}]" if resolved_key else ""
            print(f"[ClipMRT] Loaded from {design_point_path}{key_suffix}: strides={self.strides}  bits={self.bits}")
        else:
            axis_len = tune_axis_len if tune_axis_len is not None else img_size_h // 4
            self.strides, self.bits, _, _ = tune_comrt(L=axis_len, seed=tune_seed, verbose=True)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        patch_embed = SPS(img_size_h=img_size_h, img_size_w=img_size_w,
                          patch_size=patch_size, in_channels=in_channels, embed_dims=embed_dims)

        if pe_type == "conv":
            self.pe = ConvPE(d_model=embed_dims)
        else:
            self.pe = nn.Identity()

        block = nn.ModuleList([
            Block(dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[j],
                  norm_layer=norm_layer, sr_ratio=sr_ratios,
                  strides=self.strides, bits=self.bits)
            for j in range(depths)
        ])

        setattr(self, "patch_embed", patch_embed)
        setattr(self, "block", block)

        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        patch_embed = getattr(self, "patch_embed")
        block = getattr(self, "block")
        x = patch_embed(x)   # T, B, N, C
        x = self.pe(x)        # T, B, N, C
        for blk in block:
            x = blk(x)
        return x.mean(2)      # T, B, C

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x, []


@register_model
def spikformer_clipmrt_2d(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = Spikformer_clipmrt_2d(**kwargs)
    model.default_cfg = _cfg()
    return model
