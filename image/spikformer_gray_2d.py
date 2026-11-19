import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven import surrogate
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial
import copy
import math
from dataclasses import dataclass
import numpy as np

__all__ = ['spikformer_not_xor_gray_2d']


def generate_gray_code_matrix_2d(M, num_bits):
    T, B, H, N, D = M.shape
    
    Nx = int(math.sqrt(N))
    Ny = Nx
    assert Nx * Ny == N, " Nx * Ny "
    
    i_indices = torch.arange(Nx)  # [Nx]
    j_indices = torch.arange(Ny)  # [Ny]

    i_gray = i_indices ^ (i_indices >> 1)  # [Nx]
    j_gray = j_indices ^ (j_indices >> 1)  # [Ny]
    
    i_mat, j_mat = torch.meshgrid(i_indices, j_indices, indexing='ij')

    i_mat = i_mat.flatten()  # [N]
    j_mat = j_mat.flatten()  # [N]
    

    i_gray_mat = i_gray[i_mat]  # [N]
    j_gray_mat = j_gray[j_mat]  # [N]

    i_gray_codes = ((i_gray_mat.unsqueeze(-1) >> torch.arange(num_bits - 1, -1, -1)) & 1).float()
    # j_gray_codes: [N, num_bits]
    j_gray_codes = ((j_gray_mat.unsqueeze(-1) >> torch.arange(num_bits - 1, -1, -1)) & 1).float()
    
    ij_gray_codes = torch.cat([i_gray_codes, j_gray_codes], dim=-1)  # [N, 2 * num_bits]

    ij_gray_codes = ij_gray_codes.view(1, 1, 1, N, 2 * num_bits).expand(T, B, H, N, 2 * num_bits)

    ij_gray_codes = ij_gray_codes.to(M.device)
    result = torch.cat((M, ij_gray_codes), dim=-1)  # [T, B, H, N, D + 2 * num_bits]

    return result

@dataclass
class CPG(nn.Module):
    num_neurons: int = 10
    w_max: float = 10000.
    l_max: int = 5000
    
    def __post_init__(self):
        self._cpg = torch.zeros(self.l_max, self.num_neurons)
        position = torch.arange(0, self.l_max, dtype=torch.float).unsqueeze(1) # MaxL, 1
        div_term = torch.exp(torch.arange(0, self.num_neurons, 2).float() * (-math.log(self.w_max) / self.num_neurons))
        div_term_single = torch.exp(torch.arange(0, self.num_neurons - 1, 2).float() * (-math.log(self.w_max) / self.num_neurons))
        # self._cpg[:, 0::2] = torch.heaviside(torch.sin(position * div_term)-0.8, torch.tensor([1.0]))
        # self._cpg[:, 1::2] = torch.heaviside(torch.cos(position * div_term_single)-0.8, torch.tensor([1.0]))
        self._cpg[:, 1::2] = torch.heaviside(torch.sin(2 * math.pi * position * div_term)-0.8, torch.tensor([1.0]))
        self._cpg[:, 0::2] = torch.heaviside(torch.cos(2 * math.pi * position * div_term_single)-0.8, torch.tensor([1.0]))
    @property
    def cpg(self):
        return self._cpg
    
class CPGLinear(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        output_size: int, 
        cpg: CPG = CPG(),
        dropout: float = 0.1
    ):
        super(CPGLinear, self).__init__()
        self.cpg = nn.Parameter(cpg.cpg, requires_grad=False)
        self.inp_linear = nn.Linear(input_size, output_size)
        self.cpg_linear = nn.Linear(cpg.num_neurons, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor
    ):
        T, B, L, _ = x.shape
        x = x.permute(1, 0, 2, 3) # B, T, L, D
        x = x.flatten(1, 2) # B, TL, D
        cpg = self.cpg[:x.size(-2)]
        x = self.dropout(x)
        x = self.inp_linear(x) + self.cpg_linear(cpg) # B, TL, D
        x = x.transpose(0, 1) # TL, B, D
        x = x.reshape(T, L, B, -1) # T, L, B, D
        x = x.permute(0, 2, 1, 3) # T, B, L, D
        return x
        
def generate_ones_and_zero_matrix(rows, cols):
    random_matrix = torch.randint(0, 2, (rows, cols))
    binary_matrix = torch.where(random_matrix == 0, 0 * torch.ones_like(random_matrix), torch.ones_like(random_matrix))
    return binary_matrix.float()

class RandomPE(nn.Module):
    def __init__(self, d_model, pe_mode="concat", num_pe_neuron=10, neuron_pe_scale=1000.0, dropout=0.1, num_steps=4):
        super(RandomPE, self).__init__()
        self.max_len = 5000 # different from windows
        self.pe_mode=pe_mode
        self.neuron_pe_scale = neuron_pe_scale
        self.dropout = nn.Dropout(p=dropout)
        if self.pe_mode == "concat":
            self.num_pe_neuron = copy.deepcopy(num_pe_neuron)
        elif self.pe_mode == "add":
            self.num_pe_neuron = copy.deepcopy(d_model)
        pe = generate_ones_and_zero_matrix(self.max_len, self.num_pe_neuron) # MaxL, Neur
        pe = pe.unsqueeze(0).transpose(0, 1) # MaxL, 1, Neur
        print("pe.shape: ", pe.shape)
        self.register_buffer("pe", pe)
        if self.pe_mode == "concat":
            self.hid_mlp = nn.Linear(d_model + self.num_pe_neuron, d_model)
            self.hid_bn = nn.BatchNorm1d(d_model)
            self.hid_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x):
        # T, B, L, D
        T, B, L, _ = x.shape
        x = x.permute(1, 0, 2, 3) # B, T, L, D
        x = x.flatten(1, 2) # B, TL, D
        if self.pe_mode == "concat":
            # tmp: TL, 1, Neur -> TL, B, Neur -> B, TL, Neur
            tmp = self.pe[:x.size(-2), :].repeat(1, B, 1).transpose(0, 1)
            x = torch.concat([x, tmp], dim=-1)
            # print(x.shape) # B, TL, D'
        elif self.pe_mode == "add":
            # [B, TL, D] + [1, TL, Neur]
            x = x + self.pe[:x.size(-2), :].transpose(0, 1) 
            # print(x.shape) # B, TL, D
        x = x.transpose(0, 1) # TL, B D
        x = x.reshape(T, L, B, -1) # T, L, B, D
        x = x.permute(0, 2, 1, 3) # T, B, L, D
        if self.pe_mode == "concat":
            x = self.hid_mlp(x.flatten(0,1)) # TB, L, D' to TB, L, D
            x = self.hid_bn(x.transpose(-1,-2)).transpose(-1,-2) # TB, L, D
            x = x.reshape(T, B, L, -1) # T, B, L, D
            # x = x + res # add
            # x = 1-(1-res) * (1-x) # replace "add" with "or"
        return x

class NonePE(nn.Module):
    def __init__(self):
        super(NonePE, self).__init__()
        return

    def forward(self, x):
        return x

class ConvPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, num_steps=4):
        super(ConvPE, self).__init__()
        self.T = num_steps
        self.rpe_conv = nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(d_model)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch', surrogate_function=surrogate.ATan())

    def forward(self, x):
        T, B, L, D = x.shape
        # print(L)
        h = int(math.sqrt(L))
        w = int(math.sqrt(L))
        x = x.transpose(-1,-2) # T, B, D, L
        x = x.reshape(T, B, D, h, w) # T, B, D, h, w
        x_feat = x.contiguous()
        x = x.flatten(0, 1).contiguous() # TB, D, h, w
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, h, w).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat # T, B, D, h, w
        x = x.flatten(-2, -1).transpose(-2,-1) # T, B, L, D
        return x

class StaticPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50000):
        super(StaticPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) # MaxL, D
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # MaxL, 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        div_term_single = torch.exp(torch.arange(0, d_model - 1, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term_single)
        pe = pe.unsqueeze(0).transpose(0, 1) # MaxL, 1, D
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: T, B, L, D
        T, B, L, _ = x.shape
        x = x.flatten(0, 1) # TB, L, D
        x = x.permute(1, 0, 2) # L, TB, D
        x = x + self.pe[:x.size(0), :]
        # x = self.dropout(x)
        x = x.reshape(L, T, B, -1) # L, T, B, D
        x = x.permute(1, 2, 0, 3) # T, B, L, D
        return x

class NeuronPE(nn.Module):
    def __init__(self, d_model, pe_mode="concat", num_pe_neuron=10, neuron_pe_scale=10000.0, dropout=0.1, num_steps=4):
        super(NeuronPE, self).__init__()
        self.max_len = 50000 # different from windows
        self.pe_mode=pe_mode
        self.neuron_pe_scale = neuron_pe_scale
        # self.dropout = nn.Dropout(p=dropout)
        if self.pe_mode == "concat":
            self.num_pe_neuron = copy.deepcopy(num_pe_neuron)
        elif self.pe_mode == "add":
            self.num_pe_neuron = copy.deepcopy(d_model)
        pe = torch.zeros(self.max_len, self.num_pe_neuron) # MaxL, Neur
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1) # MaxL, 1
        div_term = torch.exp(torch.arange(0, self.num_pe_neuron, 2).float() * (-math.log(neuron_pe_scale) / self.num_pe_neuron))
        div_term_single = torch.exp(torch.arange(0, self.num_pe_neuron - 1, 2).float() * (-math.log(neuron_pe_scale) / self.num_pe_neuron))
        pe[:, 1::2] = torch.heaviside(torch.sin(2 * math.pi * position * div_term)-0.8, torch.tensor([1.0]))
        pe[:, 0::2] = torch.heaviside(torch.cos(2 * math.pi * position * div_term_single)-0.8, torch.tensor([1.0]))
        pe = pe.unsqueeze(0).transpose(0, 1) # MaxL, 1, Neur
        print("pe.shape: ", pe.shape)
        self.register_buffer("pe", pe)
        if self.pe_mode == "concat":
            self.hid_mlp = nn.Linear(d_model + self.num_pe_neuron, d_model)
            self.hid_bn = nn.BatchNorm1d(d_model)
            self.hid_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
    
    def forward(self, x):
        # T, B, L, D
        T, B, L, _ = x.shape
        # res = x.contiguous()
        x = x.permute(1, 0, 2, 3) # B, T, L, D
        x = x.flatten(1, 2) # B, TL, D
        if self.pe_mode == "concat":
            # tmp: TL, 1, Neur -> TL, B, Neur -> B, TL, Neur
            tmp = self.pe[:x.size(-2), :].repeat(1, B, 1).transpose(0, 1)
            x = torch.concat([x, tmp], dim=-1)
            # print(x.shape) # B, TL, D'
        elif self.pe_mode == "add":
            # [B, TL, D] + [1, TL, Neur]
            # print(self.pe[:x.size(-2), :].shape)
            x = x + self.pe[:x.size(-2), :].transpose(0, 1) 
            # print(x.shape) # B, TL, D
        x = x.transpose(0, 1) # TL, B, D'
        x = x.reshape(T, L, B, -1) # T, L, B, D'
        x = x.permute(0, 2, 1, 3) # T, B, L, D'
        if self.pe_mode == "concat":
            x = self.hid_mlp(x.flatten(0,1)) # TB, L, D' to TB, L, D
            x = self.hid_bn(x.transpose(-1,-2)).transpose(-1,-2) # TB, L, D
            x = x.reshape(T, B, L, -1) # T, B, L, D
            # x = x + res # add
            # x = 1-(1-res) * (1-x) # replace "add" with "or"
        return x

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
        T,B,N,C = x.shape
        x_ = x.flatten(0, 1)
        x = self.fc1_linear(x_)
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_linear(x.flatten(0,1))
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        x = self.fc2_lif(x)

        return x

class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        # self.scale = 0.125
        self.scale = nn.Parameter(data=torch.tensor(-4.0), requires_grad=True)
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
        T,B,N,C = x.shape

        x_for_qkv = x.flatten(0, 1)  # TB, N, C
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
        q_linear_out = self.q_bn(q_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous() # T, B, H, N, C//H

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        # attn = (q @ k.transpose(-2, -1)) * self.scale  # T, B, H, N, N
        q = generate_gray_code_matrix_2d(q, num_bits=5)
        q = q.unsqueeze(3) # T, B, H, 1, N, C//H+10
        k = generate_gray_code_matrix_2d(k, num_bits=5)
        k = k.unsqueeze(4) # T, B, H, N, 1, C//H+10
        attn = torch.sum(1 - (q-k) ** 2, dim=-1) # T, B, H, N, N

        # attn_min, _ = torch.min(attn.detach(), dim=1, keepdim=True)
        # attn = (attn - attn_min) * torch.sigmoid(self.scale)

        attn = attn * torch.sigmoid(self.scale)

        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x)

        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))

        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
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
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        # self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        # self.rpe_bn = nn.BatchNorm2d(embed_dims)
        # self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_conv(x.flatten(0, 1)) # have some fire value
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
        x = self.proj_bn3(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()

        x = self.maxpool3(x) # torch.Size([512, 384, 8, 8])
        # print(x.shape) # torch.Size([512, 384, 8, 8]) # TB, D, l1, l2 
        
        # # conv pe
        # x_feat = x.reshape(T, B, -1, H//4, W//4).contiguous()
        # x = self.rpe_conv(x)
        # x = self.rpe_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()
        # x = self.rpe_lif(x)
        # x = x + x_feat
        # x = x.flatten(-2).transpose(-1, -2)  # T,B,N,C

        # no pe
        x = x.reshape(T, B, -1, H//4, W//4).contiguous()
        # print(x.shape) # torch.Size([4, 128, 384, 8, 8]) # T, B, C, l1, l2 
        x = x.flatten(-2).transpose(-1, -2)  # T,B,N,C
        return x

class Spikformer_not_xor_gray_2d(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T = 4, 
                 pe_type: str="conv", pe_mode:str="concat",
                 num_pe_neuron: int =12, neuron_pe_scale: float=1000.0
                 ):
        super().__init__()
        self.T = T  # time step
        self.num_classes = num_classes
        self.depths = depths
        self.pe_type = pe_type
        self.pe_mode = pe_mode
        self.num_pe_neuron = num_pe_neuron
        self.neuron_pe_scale = neuron_pe_scale

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SPS(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims)
        if pe_type == "neuron":
            self.pe = NeuronPE(d_model=embed_dims, pe_mode=pe_mode, num_pe_neuron=num_pe_neuron, neuron_pe_scale=neuron_pe_scale, dropout=0.1)
        elif pe_type == "static":
            self.pe = StaticPE(d_model=embed_dims, dropout=0.1)
        elif pe_type == "none": # will not run
            self.pe = NonePE
        elif pe_type == "random":
            self.pe = RandomPE(d_model=embed_dims, pe_mode=pe_mode, num_pe_neuron=num_pe_neuron, neuron_pe_scale=neuron_pe_scale, dropout=0.1)
        elif pe_type == "conv":
            self.pe = ConvPE(d_model=embed_dims)
        
        block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        x = patch_embed(x) # T, B, N, C
        x = self.pe(x) # T, B, N, C'
        for blk in block:
            x = blk(x)
        return x.mean(2)

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x, []


@register_model
def spikformer_not_xor_gray_2d(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = Spikformer_not_xor_gray_2d(
        # img_size_h=224, img_size_w=224,
        # patch_size=16, embed_dims=768, num_heads=12, mlp_ratios=4,
        # in_channels=3, num_classes=1000, qkv_bias=False,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=12, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model