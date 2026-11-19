"""
Clip-mode Multi-Resolution Thermometer (ClipMRT) encoding for Spiking Neural Networks.

Architecture:
  P(i) = [τ_{B1}(f1(i)) ‖ … ‖ τ_{BS}(fS(i))]
  f_s(i) = min(⌊i/m_s⌋, B_s)   (clip response)
  τ_B(t)_k = 1 if k≤t else 0   (thermometer code)

Key constraints:
  • m_1 = 1
  • m_{s+1} ≤ m_s · B_s   (cascade — no gaps)
  • m_S · B_S ≥ L-1       (full coverage)

Stride construction uses Family-C random draws (no coprime requirement).
"""

from __future__ import annotations

import json
import math
import random
from math import gcd
from pathlib import Path
from itertools import product
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def _pairwise_coprime(strides: List[int]) -> bool:
    for i in range(len(strides)):
        for j in range(i + 1, len(strides)):
            if gcd(strides[i], strides[j]) != 1:
                return False
    return True


def _largest_coprime_le(bound: int, existing: List[int]) -> Optional[int]:
    """Return largest integer in [1, bound] coprime with all existing strides."""
    for candidate in range(bound, 0, -1):
        if all(gcd(candidate, m) == 1 for m in existing):
            return candidate
    return None


def _smallest_coprime_ge(start: int, existing: List[int]) -> int:
    """Return smallest integer >= start coprime with all existing strides."""
    candidate = max(start, 1)
    while not all(gcd(candidate, m) == 1 for m in existing):
        candidate += 1
    return candidate


# ---------------------------------------------------------------------------
# Encoding table
# ---------------------------------------------------------------------------

def build_encoding_table(L: int, strides: List[int], bits: List[int]) -> torch.Tensor:
    """Build the (L, Σ B_s) binary encoding table.

    Each row P(i) = concatenation of thermometer codes across scales.
    """
    total_bits = sum(bits)
    table = torch.zeros(L, total_bits, dtype=torch.float32)
    offset = 0
    for m, B in zip(strides, bits):
        for i in range(L):
            t = min(i // m, B)
            table[i, offset : offset + t] = 1.0
        offset += B
    return table


# ---------------------------------------------------------------------------
# Diagnostic score  (§3.3 / Eqs 3.29–3.42)
# ---------------------------------------------------------------------------

def compute_mean_similarity(table: torch.Tensor) -> torch.Tensor:
    """μ(δ) for δ = 0 … L-1.  Returns tensor of length L."""
    L, B = table.shape
    mu = torch.zeros(L)
    mu[0] = float(B)
    for delta in range(1, L):
        n = L - delta
        i_idx = torch.arange(n)
        j_idx = i_idx + delta
        diff = (table[i_idx] != table[j_idx]).float().sum(dim=-1)
        sim = B - diff
        mu[delta] = sim.mean()
    return mu


def build_gray_code_table(L: int, num_bits: Optional[int] = None) -> torch.Tensor:
    """Build a (L, num_bits) binary encoding table using Gray codes.

    num_bits defaults to ceil(log2(max(L, 2))), i.e. the minimum needed.
    """
    if num_bits is None:
        num_bits = math.ceil(math.log2(max(L, 2)))
    indices = torch.arange(L)
    gray = indices ^ (indices >> 1)
    bits = ((gray.unsqueeze(-1) >> torch.arange(num_bits - 1, -1, -1)) & 1).float()
    return bits  # (L, num_bits)


def _compute_scores_from_table(table: torch.Tensor, L: int) -> Dict:
    """Compute f1–f4 and overall score from a pre-built (L, total_bits) table.

    Shared by diagnostic_score (ClipMRT) and diagnostic_score_from_table (any encoding).
    """
    total_bits = table.shape[1]
    mu = compute_mean_similarity(table)

    eps  = 1e-8
    mu1  = mu[1].item()
    muL1 = mu[L - 1].item()

    # f1: pairwise monotonicity
    numerator = 0.0
    denominator = 0.0
    for d2 in range(2, L):
        n2 = L - d2
        if n2 <= 0:
            break
        i_idx  = torch.arange(n2)
        sim_d2 = total_bits - (table[i_idx] != table[i_idx + d2]).float().sum(-1)
        for d1 in range(1, d2):
            n = min(L - d1, n2)
            sim_d1 = total_bits - (table[:n] != table[torch.arange(n) + d1]).float().sum(-1)
            numerator   += (sim_d1 >= sim_d2[:n]).float().sum().item()
            denominator += n
    f1 = numerator / denominator if denominator > 0 else 0.0

    # f2: tail resolution
    half = L // 2
    tail_auc         = sum(mu[d].item() - muL1 for d in range(half, L))
    uniform_tail_auc = sum(
        (mu1 - muL1) * (L - 1 - d) / max(L - 2, 1) for d in range(half, L)
    )
    f2 = float(np.clip(tail_auc / max(uniform_tail_auc, eps), 0.0, 1.0))

    # f3: local resolution
    drops = [mu[d].item() - mu[d + 1].item() for d in range(1, L - 1)]
    if drops:
        drops_sorted = sorted(drops)
        p10_idx = max(0, int(0.1 * len(drops_sorted)) - 1)
        p10 = drops_sorted[p10_idx]
        med = drops_sorted[len(drops_sorted) // 2]
        f3  = float(np.clip(p10 / max(med, eps), 0.0, 1.0))
    else:
        f3 = 0.0

    # f4: consistency — pair-count-weighted NSD
    range_val        = max(mu1 - muL1, eps)
    nsd_weighted_sum = 0.0
    weight_total     = 0.0
    for delta in range(1, L - 1):
        n = L - delta
        if n < 2:
            continue
        i_idx = torch.arange(n)
        vals  = total_bits - (table[i_idx] != table[i_idx + delta]).float().sum(-1)
        sigma = vals.std().item()
        nsd   = sigma / range_val
        nsd_weighted_sum += float(n) * nsd
        weight_total     += float(n)
    nsd_avg = nsd_weighted_sum / weight_total if weight_total > 0 else 0.0
    f4 = float(np.clip(1.0 - 2.0 * nsd_avg, 0.0, 1.0))

    # Overall score
    quality_score = (f1 ** 1 * f2 ** 1 * f3 ** 1.5 * f4 ** 1) ** (1.0 / 4.5)
    b_ref         = math.ceil(math.e * math.log(max(L - 1, 2))) + 1
    bit_factor    = min(1.0, b_ref / max(total_bits, 1))
    final_score   = quality_score * bit_factor

    return {
        "total_bits":         total_bits,
        "b_ref":              b_ref,
        "final_score":        round(final_score,     6),
        "quality_score":      round(quality_score,   6),
        "bit_factor":         round(bit_factor,      6),
        "f1_monotonicity":    round(f1,              6),
        "f2_tail_resolution": round(f2,              6),
        "f3_local_resolution":round(f3,              6),
        "f4_consistency":     round(f4,              6),
        "nsd_avg":            round(nsd_avg,         6),
    }


def diagnostic_score(
    strides: List[int],
    bits: List[int],
    L: int,
    output_path: Optional[Path] = None,
) -> Dict:
    """Compute all diagnostic components and the final score for a ClipMRT config."""
    table = build_encoding_table(L, strides, bits)
    scores = _compute_scores_from_table(table, L)

    result: Dict = {
        "L":       L,
        "strides": list(strides),
        "bits":    list(bits),
        **scores,
        "coverage_ok": bool(list(strides)[-1] * list(bits)[-1] >= L - 1),
        "coprime":     _pairwise_coprime(list(strides)),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as _f:
            json.dump(result, _f, indent=2)

    return result


def diagnostic_score_from_table(
    table: torch.Tensor,
    L: int,
    encoding_name: str = "custom",
    output_path: Optional[Path] = None,
) -> Dict:
    """Compute diagnostic score for any pre-built (L, total_bits) binary encoding table.

    Use this to score gray code, log PE, or any other fixed encoding:
        table = build_gray_code_table(L, num_bits=10)
        result = diagnostic_score_from_table(table, L, encoding_name="gray_code")
    """
    assert table.shape[0] >= L, f"table has {table.shape[0]} rows but L={L}"
    table = table[:L]
    scores = _compute_scores_from_table(table, L)

    result: Dict = {
        "L":             L,
        "encoding_name": encoding_name,
        **scores,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as _f:
            json.dump(result, _f, indent=2)

    return result


def build_log_pe_matrix(L: int) -> torch.Tensor:
    """Build the L×L pairwise similarity matrix for Log PE.

    Diagonal = ceil(log2(L)); off-diagonals decay linearly to 0
    following the same construction used in Spikingformer_XNOR_Log.
    """
    diag_value = math.ceil(math.log2(max(L, 2)))
    matrix = torch.zeros(L, L, dtype=torch.float32)
    for i in range(L):
        matrix[i, i] = float(diag_value)
        if i < L - 1:
            values = torch.linspace(diag_value - 1, 0, steps=L - i - 1).round()
            matrix[i, i + 1:] = values
            matrix[i + 1:, i] = values
    return matrix


def _compute_scores_from_sim_matrix(sim_matrix: torch.Tensor, L: int, total_bits: int) -> Dict:
    """Compute f1-f4 from a precomputed L×L pairwise similarity matrix.

    Used for encodings like Log PE that define similarity directly rather
    than via a binary per-position code.
    total_bits is used only for the bit_factor penalty term.
    """
    eps = 1e-8

    # mu(delta) = mean similarity at lag delta
    mu = torch.zeros(L)
    mu[0] = sim_matrix.diagonal().float().mean()
    for delta in range(1, L):
        n = L - delta
        i_idx = torch.arange(n)
        mu[delta] = sim_matrix[i_idx, i_idx + delta].float().mean()

    mu1  = mu[1].item()
    muL1 = mu[L - 1].item()

    # f1: pairwise monotonicity
    numerator   = 0.0
    denominator = 0.0
    for d2 in range(2, L):
        n2 = L - d2
        if n2 <= 0:
            break
        i_idx  = torch.arange(n2)
        sim_d2 = sim_matrix[i_idx, i_idx + d2].float()
        for d1 in range(1, d2):
            n      = min(L - d1, n2)
            i_n    = torch.arange(n)
            sim_d1 = sim_matrix[i_n, i_n + d1].float()
            numerator   += (sim_d1 >= sim_d2[:n]).float().sum().item()
            denominator += n
    f1 = numerator / denominator if denominator > 0 else 0.0

    # f2: tail resolution
    half = L // 2
    tail_auc         = sum(mu[d].item() - muL1 for d in range(half, L))
    uniform_tail_auc = sum(
        (mu1 - muL1) * (L - 1 - d) / max(L - 2, 1) for d in range(half, L)
    )
    f2 = float(np.clip(tail_auc / max(uniform_tail_auc, eps), 0.0, 1.0))

    # f3: local resolution
    drops = [mu[d].item() - mu[d + 1].item() for d in range(1, L - 1)]
    if drops:
        drops_sorted = sorted(drops)
        p10_idx = max(0, int(0.1 * len(drops_sorted)) - 1)
        p10 = drops_sorted[p10_idx]
        med = drops_sorted[len(drops_sorted) // 2]
        f3  = float(np.clip(p10 / max(med, eps), 0.0, 1.0))
    else:
        f3 = 0.0

    # f4: consistency — pair-count-weighted NSD
    range_val        = max(mu1 - muL1, eps)
    nsd_weighted_sum = 0.0
    weight_total     = 0.0
    for delta in range(1, L - 1):
        n = L - delta
        if n < 2:
            continue
        i_idx = torch.arange(n)
        vals  = sim_matrix[i_idx, i_idx + delta].float()
        sigma = vals.std().item()
        nsd   = sigma / range_val
        nsd_weighted_sum += float(n) * nsd
        weight_total     += float(n)
    nsd_avg = nsd_weighted_sum / weight_total if weight_total > 0 else 0.0
    f4 = float(np.clip(1.0 - 2.0 * nsd_avg, 0.0, 1.0))

    # Overall score
    quality_score = (f1 ** 1 * f2 ** 2 * f3 ** 1.5 * f4 ** 1) ** (1.0 / 5.5)
    b_ref         = math.ceil(math.e * math.log(max(L - 1, 2))) + 1
    bit_factor    = min(1.0, b_ref / max(total_bits, 1))
    final_score   = quality_score * bit_factor

    return {
        "total_bits":         total_bits,
        "b_ref":              b_ref,
        "final_score":        round(final_score,     6),
        "quality_score":      round(quality_score,   6),
        "bit_factor":         round(bit_factor,      6),
        "f1_monotonicity":    round(f1,              6),
        "f2_tail_resolution": round(f2,              6),
        "f3_local_resolution":round(f3,              6),
        "f4_consistency":     round(f4,              6),
        "nsd_avg":            round(nsd_avg,         6),
    }


def diagnostic_score_log_pe(
    L: int,
    output_path: Optional[Path] = None,
) -> Dict:
    """Compute diagnostic score for Log PE.

    Uses the L×L pairwise similarity matrix defined by Spikingformer_XNOR_Log.
    total_bits is set to ceil(log2(L)) — the diagonal value — for the bit_factor term.
    """
    diag_value = math.ceil(math.log2(max(L, 2)))
    sim_matrix = build_log_pe_matrix(L)
    scores = _compute_scores_from_sim_matrix(sim_matrix, L, total_bits=diag_value)

    result: Dict = {"L": L, "encoding_name": "log_pe", **scores}

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as _f:
            json.dump(result, _f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Greedy stride construction
# ---------------------------------------------------------------------------

def _build_strides_greedy(bits: List[int]) -> List[int]:
    """Greedy construction: m_1=1, then largest valid coprime stride per scale."""
    S = len(bits)
    strides = [1]
    for s in range(1, S):
        bound = strides[s - 1] * bits[s - 1]
        m = _largest_coprime_le(bound, strides)
        if m is None:
            m = _smallest_coprime_ge(2, strides)
        strides.append(m)
    return strides


def _build_strides_clipmrt(bits: List[int], L: int, rng: random.Random) -> List[int]:
    """Family-C random stride construction — no coprime constraint.

    Cascade constraint m_{s+1} ≤ m_s · B_s is always respected.
    When coverage (m_S · B_S ≥ L-1) cannot be met within the cascade bound,
    m_S is set to the cascade maximum; callers must verify and adjust bits[-1].
    """
    S = len(bits)
    if S == 1:
        return [1]
    strides = [1]
    for s in range(1, S - 1):
        lo = strides[s - 1]
        hi = strides[s - 1] * bits[s - 1]
        strides.append(rng.choice(list(range(lo + 1, hi + 1))))
    # Coarsest stride: prefer coverage; fall back to cascade maximum
    lo = strides[-1]
    hi = strides[-1] * bits[len(strides) - 1]   # bits[S-2]
    needed = math.ceil((L - 1) / max(bits[-1], 1))
    cands_ok = [c for c in range(lo + 1, hi + 1) if c >= needed]
    m_S = rng.choice(cands_ok) if cands_ok else hi
    strides.append(m_S)
    return strides


# ---------------------------------------------------------------------------
# Bit-allocation enumeration
# ---------------------------------------------------------------------------

def _balanced_bits(S: int, L: int, budget: int) -> List[int]:
    """Equal bit-width allocation: each B_s ≈ ceil((L-1)^{1/S})."""
    B = max(2, math.ceil((L - 1) ** (1.0 / S)))
    return [B] * S


def _enumerate_bit_allocs(S: int, L: int, max_bits: int) -> List[List[int]]:
    """
    Enumerate candidate bit allocations for S scales with total ≤ max_bits
    and product ≥ L-1 (coverage bound).
    """
    # Start from balanced allocation and perturb
    candidates = set()
    base_B = max(2, math.ceil((L - 1) ** (1.0 / S)))

    # Perturb each scale by ±range
    perturb = 2
    vals = list(range(max(2, base_B - perturb), base_B + perturb + 1))
    for combo in product(vals, repeat=S):
        total = sum(combo)
        prod = 1
        for b in combo:
            prod *= b
        if total <= max_bits and prod >= L - 1:
            candidates.add(tuple(sorted(combo, reverse=True)))

    # Also try geometric sequences
    for start in range(2, base_B + 4):
        alloc = [max(2, start - i) for i in range(S)]
        total = sum(alloc)
        prod = 1
        for b in alloc:
            prod *= b
        if total <= max_bits and prod >= L - 1:
            candidates.add(tuple(sorted(alloc, reverse=True)))

    # Fallback: single scale
    candidates.add((max(L - 1, 2),))

    return [list(c) for c in candidates if len(c) == S] or [_balanced_bits(S, L, max_bits)]


# ---------------------------------------------------------------------------
# Tuning procedure  (§3.3 "Design and Construction of CoMRT")
# ---------------------------------------------------------------------------

def tune_comrt(
    L: int,
    max_total_bits: Optional[int] = None,
    K: int = 100,
    seed: int = 0,
    verbose: bool = True,
    output_dir: Optional[Path] = None,
) -> Tuple[List[int], List[int], float, Dict]:
    """Find optimal (strides, bits) for sequence length L via random ClipMRT draws.

    Steps:
      1. S ≈ ln(L-1), bit budget b_ref* = ⌈e·ln(L-1)⌉+1
      2. Enumerate bit allocations with ∏B_s ≥ L-1 and Σ ≤ budget
      3. For each allocation run K Family-C random draws; keep best diagnostic score

    If output_dir is given, writes clipmrt_tune_L{L}.json with the best result.
    """
    if L <= 2:
        return [1], [1], 1.0, {}

    S_opt = max(1, int(math.log(L - 1)))
    b_ref = math.ceil(math.e * math.log(L - 1)) + 1
    if max_total_bits is None:
        max_total_bits = max(b_ref + S_opt, int(b_ref * 1.5))

    S_range = list(range(max(1, S_opt - 1), min(S_opt + 2, 8)))
    rng = random.Random(seed)

    best_strides: List[int] = [1]
    best_bits: List[int] = [L - 1]
    best_score = -1.0
    best_details: Dict = {}

    for S in S_range:
        allocs = _enumerate_bit_allocs(S, L, max_total_bits)
        for bits in allocs:
            if len(bits) != S:
                continue
            for _ in range(K):
                strides = _build_strides_clipmrt(bits, L, rng)
                bits_cur = list(bits)
                if strides[-1] * bits_cur[-1] < L - 1:
                    bits_cur[-1] = max(bits_cur[-1], math.ceil((L - 1) / max(strides[-1], 1)))
                details = diagnostic_score(strides, bits_cur, L)
                score = details["final_score"]
                if score > best_score:
                    best_score = score
                    best_strides = list(strides)
                    best_bits = list(bits_cur)
                    best_details = details

    if verbose:
        print(
            f"[ClipMRT] L={L}  S={len(best_strides)}  "
            f"strides={best_strides}  bits={best_bits}  "
            f"total_bits={sum(best_bits)}  score={best_score:.4f}"
        )
        print(
            f"          f1={best_details.get('f1_monotonicity',0):.3f}  "
            f"f2={best_details.get('f2_tail_resolution',0):.3f}  "
            f"f3={best_details.get('f3_local_resolution',0):.3f}  "
            f"f4={best_details.get('f4_consistency',0):.3f}"
        )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"clipmrt_tune_L{L}.json", "w") as _f:
            json.dump(best_details, _f, indent=2)

    return best_strides, best_bits, best_score, best_details


# ---------------------------------------------------------------------------
# Design point loading
# ---------------------------------------------------------------------------

def load_design_point(
    design_point_path: str,
    design_point_key: Optional[str] = None,
) -> Tuple[List[int], List[int], Optional[str]]:
    """Load (strides, bits) from a precomputed design point JSON.

    Accepts either a flat file ({"strides": [...], "bits": [...], ...}) or a
    multi-entry file keyed by score ({"0.85": {"strides": [...], "bits": [...]}, ...}).

    Returns (strides, bits, resolved_key) where resolved_key is the key that was
    selected from a multi-entry file (or None for a flat file / explicit key).
    """
    with open(design_point_path) as f:
        dp = json.load(f)

    if "strides" not in dp:
        if design_point_key is not None:
            if design_point_key not in dp:
                raise KeyError(
                    f"design_point_key={design_point_key!r} not found in {design_point_path}; "
                    f"available keys: {list(dp.keys())}"
                )
            dp = dp[design_point_key]
        elif len(dp) == 1:
            design_point_key = next(iter(dp))
            dp = dp[design_point_key]
        else:
            raise ValueError(
                f"{design_point_path} contains multiple design points ({list(dp.keys())}); "
                f"pass design_point_key to select one."
            )
    else:
        design_point_key = None

    return list(dp["strides"]), list(dp["bits"]), design_point_key


# ---------------------------------------------------------------------------
# nn.Module: CoMRTPE
# ---------------------------------------------------------------------------

class ClipMRTPE(nn.Module):
    """ClipMRT positional encoding as a PyTorch module.

    Produces binary encoding vectors of shape (L, total_bits) via a
    pre-computed lookup table (registered buffer — no gradients needed).

    The tuning procedure runs once at construction time and the chosen
    (strides, bits) configuration is logged so the caller can inspect it.

    Args:
        max_len:        Maximum sequence length L.
        pe_mode:        "concat" — append to input features;
                        "add"    — project to d_model and add.
        d_model:        Required when pe_mode="add".
        strides:        Fixed stride list (skips tuning if provided).
        bits:           Fixed bit-width list (skips tuning if provided).
        max_total_bits: Bit budget cap for the tuning search.
        dropout:        Dropout applied after PE injection.
        verbose:        Print tuning results.
    """

    def __init__(
        self,
        max_len: int,
        pe_mode: str = "concat",
        d_model: Optional[int] = None,
        strides: Optional[List[int]] = None,
        bits: Optional[List[int]] = None,
        max_total_bits: Optional[int] = None,
        dropout: float = 0.1,
        verbose: bool = True,
        output_dir: Optional[Path] = None,
    ):
        super().__init__()
        self.max_len = max_len
        self.pe_mode = pe_mode
        self.dropout = nn.Dropout(p=dropout)

        # Tuning / construction
        if strides is not None and bits is not None:
            self.strides        = list(strides)
            self.bits           = list(bits)
            self.tuning_details = diagnostic_score(strides, bits, max_len)
            self.tuning_score   = float(self.tuning_details["final_score"])
            if output_dir is not None:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                with open(Path(output_dir) / f"comrt_tune_L{max_len}.json", "w") as _f:
                    json.dump(self.tuning_details, _f, indent=2)
        else:
            self.strides, self.bits, self.tuning_score, self.tuning_details = tune_comrt(
                L=max_len,
                max_total_bits=max_total_bits,
                verbose=verbose,
                output_dir=output_dir,
            )

        self.total_bits: int = sum(self.bits)
        self.num_pe_neuron: int = self.total_bits  # alias used by callers

        # Pre-compute lookup table — shape (max_len, total_bits)
        table = build_encoding_table(max_len, self.strides, self.bits)
        self.register_buffer("pe_table", table)  # (max_len, total_bits)

        # Projection needed for "add" mode
        if pe_mode == "add":
            assert d_model is not None, "d_model required for pe_mode='add'"
            self.proj = nn.Linear(self.total_bits, d_model, bias=False)
        else:
            self.proj = None

    # ------------------------------------------------------------------
    # forward: x shape is (T, B, L, D)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, L, D = x.shape
        assert L <= self.max_len, (
            f"Input length {L} exceeds ClipMRT max_len {self.max_len}"
        )
        # Encoding for positions 0..L-1, shape (L, total_bits)
        pe = self.pe_table[:L]  # (L, total_bits)

        if self.pe_mode == "add":
            # Project to d_model and broadcast over T, B
            pe_proj = self.proj(pe)  # (L, D)
            x = x + pe_proj.unsqueeze(0).unsqueeze(0)  # (T, B, L, D)
        else:
            # concat: repeat over T and B
            pe_exp = pe.unsqueeze(0).unsqueeze(0).expand(T, B, -1, -1)  # (T,B,L,total_bits)
            x = torch.cat([x, pe_exp], dim=-1)  # (T, B, L, D+total_bits)

        return self.dropout(x)

    def extra_repr(self) -> str:
        return (
            f"max_len={self.max_len}, mode={self.pe_mode}, "
            f"strides={self.strides}, bits={self.bits}, "
            f"total_bits={self.total_bits}, score={self.tuning_score:.4f}"
        )


# Backward-compatible alias
CoMRTPE = ClipMRTPE
