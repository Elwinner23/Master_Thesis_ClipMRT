"""Spikformer with ClipMRT positional encoding + Log-PE, XNOR-style attention.

Combines two mechanisms in the same attention layer:
  - ClipMRT: position codes concatenated onto Q and K before the
    XNOR-style similarity product (see spikformer_comrt.py).
  - Log-PE: an additive symmetric bias over pairwise distance, added
    to the similarity scores before scaling (see spikformer_xnor_log.py).
"""

from typing import Optional, List
from pathlib import Path
import math

import torch
from torch import nn
from spikingjelly.activation_based import surrogate, neuron, functional

from ..base import NETWORKS
from ...module.comrt import build_encoding_table, tune_comrt, diagnostic_score

tau = 2.0
detach_reset = True


def create_symmetric_matrix(L):
    diag_value = math.ceil(math.log2(L))
    matrix = torch.zeros((L, L), dtype=torch.int)
    for i in range(L):
        matrix[i, i] = diag_value
        if i < L - 1:
            values = torch.linspace(diag_value - 1, 0, steps=L - i - 1).round().int()
            matrix[i, i+1:] = values
            matrix[i+1:, i] = values
    return matrix


def generate_clipmrt_matrix(M: torch.Tensor, encoding_table: torch.Tensor) -> torch.Tensor:
    """Append ClipMRT position codes to attention tensor M.

    M: (T, B, H, L, D//H)
    encoding_table: (max_len, total_bits)
    Returns: (T, B, H, L, D//H + total_bits)
    """
    T, B, H, L, _ = M.shape
    pe = encoding_table[:L].to(dtype=M.dtype, device=M.device)  # (L, total_bits)
    pe = pe.view(1, 1, 1, L, -1).expand(T, B, H, L, -1)
    return torch.cat((M, pe), dim=-1)


class ConvPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, num_steps=4):
        super().__init__()
        self.T = num_steps
        self.rpe_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm1d(d_model)
        self.rpe_lif = neuron.LIFNode(
            tau=tau, detach_reset=detach_reset, step_mode="m",
            surrogate_function=surrogate.ATan(),
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        T, B, L, D = x.shape
        x = x.flatten(0, 1).transpose(0, 1)        # L, TB, D
        L, TB, D = x.shape
        x_feat = x.permute(1, 2, 0)                # TB, D, L
        x_feat = self.rpe_conv(x_feat)
        x_feat = (
            self.rpe_bn(x_feat)
            .reshape(self.T, int(TB / self.T), D, L)
            .contiguous()
        )                                           # T, B, D, L
        x_feat = self.rpe_lif(x_feat)
        x_feat = x_feat.flatten(0, 1)              # TB, D, L
        x_feat = self.dropout(x_feat).permute(2, 0, 1)  # L, TB, D
        x = (x + x_feat).transpose(0, 1).reshape(T, B, L, -1)
        return x


class ConvEncoder(nn.Module):
    def __init__(self, output_size: int, kernel_size: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=output_size,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2),
            ),
            nn.BatchNorm2d(output_size),
        )
        self.lif = neuron.LIFNode(
            tau=tau, detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: B, L, C
        inputs = inputs.permute(0, 2, 1).unsqueeze(1)  # B, 1, C, L
        enc = self.encoder(inputs).permute(1, 0, 2, 3)  # T, B, C, L
        return self.lif(enc)


class SSA(nn.Module):
    def __init__(self, length, tau, common_thr, dim, heads=8,
                 qkv_bias=False, qk_scale=0.25, encoding_table=None):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        self.dim = dim
        self.heads = heads
        self.log_scale = nn.Parameter(data=torch.tensor(-4.0), requires_grad=True)

        self.q_m = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = neuron.LIFNode(tau=tau, detach_reset=detach_reset, surrogate_function=surrogate.ATan())

        self.k_m = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = neuron.LIFNode(tau=tau, detach_reset=detach_reset, surrogate_function=surrogate.ATan())

        self.v_m = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = neuron.LIFNode(tau=tau, detach_reset=detach_reset, surrogate_function=surrogate.ATan())

        self.attn_lif = neuron.LIFNode(
            tau=tau, detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
            v_threshold=common_thr / 2,
        )

        self.last_m = nn.Linear(dim, dim)
        self.last_bn = nn.BatchNorm1d(dim)
        self.last_lif = neuron.LIFNode(tau=tau, detach_reset=detach_reset, surrogate_function=surrogate.ATan())

        if encoding_table is not None:
            self.register_buffer("encoding_table", encoding_table)
        else:
            self.encoding_table = None

    def forward(self, x):
        T, B, L, D = x.shape
        x_for_qkv = x.flatten(0, 1)  # TB, L, D

        q_m_out = self.q_m(x_for_qkv)
        q_m_out = self.q_bn(q_m_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D).contiguous()
        q_m_out = self.q_lif(q_m_out)
        q = q_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()
        # q: T, B, H, L, D//H

        k_m_out = self.k_m(x_for_qkv)
        k_m_out = self.k_bn(k_m_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D).contiguous()
        k_m_out = self.k_lif(k_m_out)
        k = k_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        v_m_out = self.v_m(x_for_qkv)
        v_m_out = self.v_bn(v_m_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D).contiguous()
        v_m_out = self.v_lif(v_m_out)
        v = v_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        # Log-PE: additive symmetric bias over pairwise distance
        tmp = create_symmetric_matrix(L).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, L, L]

        # ClipMRT: position codes concatenated onto Q and K
        q = generate_clipmrt_matrix(q, self.encoding_table)  # T, B, H, L, (D//H)+total_bits
        k = generate_clipmrt_matrix(k, self.encoding_table)  # T, B, H, L, (D//H)+total_bits

        # XNOR attention: sim(q_i, k_j) = D - sum(q_i) - sum(k_j) + 2*dot(q_i, k_j)
        D_new = q.size(-1)
        sum_q = q.sum(dim=-1)                            # T, B, H, L
        sum_k = k.sum(dim=-1)                            # T, B, H, L
        qk_matmul = torch.matmul(q, k.transpose(-1, -2))  # T, B, H, L, L
        attn = D_new - sum_q.unsqueeze(-1) - sum_k.unsqueeze(-2) + 2 * qk_matmul  # T, B, H, L, L

        attn = (attn + tmp.to(attn.device)) * torch.sigmoid(self.log_scale)

        x = attn @ v  # T, B, H, L, D//H
        x = x.transpose(2, 3).reshape(T, B, L, D).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.last_m(x)
        x = self.last_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.last_lif(x.reshape(T, B, L, D).contiguous())
        return x


class MLP(nn.Module):
    def __init__(self, length, tau, common_thr, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.lif1 = neuron.LIFNode(tau=tau, detach_reset=detach_reset, surrogate_function=surrogate.ATan())

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.lif2 = neuron.LIFNode(tau=tau, detach_reset=detach_reset, surrogate_function=surrogate.ATan())

    def forward(self, x):
        T, B, L, D = x.shape
        x = x.flatten(0, 1)
        x = self.fc1(x)
        x = self.bn1(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, self.hidden_features).contiguous()
        x = self.lif1(x)
        x = x.flatten(0, 1)
        x = self.fc2(x)
        x = self.bn2(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D).contiguous()
        x = self.lif2(x)
        return x


class Block(nn.Module):
    def __init__(self, length, tau, common_thr, dim, d_ff, heads=8,
                 qkv_bias=False, qk_scale=0.125, encoding_table=None):
        super().__init__()
        self.attn = SSA(
            length=length, tau=tau, common_thr=common_thr,
            dim=dim, heads=heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            encoding_table=encoding_table,
        )
        self.mlp = MLP(
            length=length, tau=tau, common_thr=common_thr,
            in_features=dim, hidden_features=d_ff,
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


@NETWORKS.register_module("SpikformerCoMRT_Log")
class SpikformerCoMRT_Log(nn.Module):
    """Spikformer with ClipMRT + Log-PE positional encoding, XNOR-style attention.

    ClipMRT codes are concatenated onto Q and K, and the Log-PE symmetric
    bias is added to the resulting similarity scores before scaling —
    both injected only inside the attention layer, not concatenated to
    the input features.

    Args:
        dim:            Model dimension.
        d_ff:           Feed-forward hidden size (default 4*dim).
        depths:         Number of Spikformer blocks.
        num_steps:      SNN time steps T.
        heads:          Attention heads.
        max_length:     Maximum sequence length; used to build the ClipMRT table
                         and the Log-PE bias matrix.
        input_size:     Number of input features per timestep.
        strides:        Fixed stride list (skips auto-tuning if provided).
        bits:           Fixed bit-width list (skips auto-tuning if provided).
        max_total_bits: Bit budget cap for tuning.
        common_thr:     LIF threshold.
        weight_file:    Optional checkpoint path.
    """

    def __init__(
        self,
        dim: int,
        d_ff: Optional[int] = None,
        depths: int = 2,
        num_steps: int = 4,
        heads: int = 8,
        max_length: int = 5000,
        input_size: Optional[int] = None,
        pe_mode: Optional[str] = None,  # accepted but ignored — ClipMRT lives in Q/K only
        strides: Optional[List[int]] = None,
        bits: Optional[List[int]] = None,
        max_total_bits: Optional[int] = None,
        common_thr: float = 1.0,
        qkv_bias: bool = False,
        qk_scale: float = 0.125,
        weight_file: Optional[Path] = None,
    ):
        super().__init__()
        self.dim = dim
        self.d_ff = d_ff or dim * 4
        self.T = num_steps
        self.depths = depths

        # Build ClipMRT encoding table
        if strides is not None and bits is not None:
            self.strides = list(strides)
            self.bits = list(bits)
            self.comrt_config = diagnostic_score(strides, bits, max_length)
        else:
            self.strides, self.bits, tuning_score, details = tune_comrt(
                L=max_length,
                max_total_bits=max_total_bits,
                verbose=True,
            )
            self.comrt_config = details

        self.total_bits: int = sum(self.bits)
        table = build_encoding_table(max_length, self.strides, self.bits)
        self.register_buffer("encoding_table", table)  # (max_length, total_bits)

        self.temporal_encoder = ConvEncoder(num_steps)
        self.pe = ConvPE(d_model=input_size, num_steps=num_steps)

        self.encoder = nn.Linear(input_size, dim)
        self.init_bn = nn.BatchNorm1d(dim)
        self.init_lif = neuron.LIFNode(
            tau=tau, detach_reset=detach_reset,
            surrogate_function=surrogate.ATan(),
        )

        self.blocks = nn.ModuleList([
            Block(
                length=max_length, tau=tau, common_thr=common_thr,
                dim=dim, d_ff=self.d_ff, heads=heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                encoding_table=table,
            )
            for _ in range(depths)
        ])

        self.apply(self._init_weights)
        functional.set_step_mode(self, "m")

        if weight_file is not None:
            self.load_state_dict(torch.load(weight_file, map_location="cpu"))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor):
        functional.reset_net(self)

        x = self.temporal_encoder(x)    # B,L,C → T,B,C,L
        x = x.transpose(-2, -1)         # T,B,L,C

        x = self.pe(x)                  # T,B,L,C

        T, B, L, _ = x.shape
        x = self.encoder(x.flatten(0, 1))           # TB,L,dim
        x = self.init_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.reshape(T, B, L, -1)                  # T,B,L,dim
        x = self.init_lif(x)

        for blk in self.blocks:
            x = blk(x)

        out = x.mean(0)                  # B,L,dim
        return out, out.mean(dim=1)      # B,L,dim  /  B,dim

    @property
    def output_size(self) -> int:
        return self.dim

    @property
    def hidden_size(self) -> int:
        return self.dim
