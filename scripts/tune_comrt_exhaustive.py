#!/usr/bin/env python3
"""
Exhaustive ClipMRT tuner.

Fixes two narrowness issues in tune_comrt:
  1. S_range is too tight — e.g. for L=12, tune_comrt only checks S∈[1,2,3]
     but S=4,5,6 are valid and reachable within the bit budget.
  2. _enumerate_bit_allocs only perturbs ±2 around the balanced allocation,
     missing many valid integer partitions.

This script replaces both with:
  - Full S sweep from 1 to S_max (default: min(L-1, 8))
  - Complete enumeration of all integer partitions of the bit budget into S parts,
    each ≥ 2, with product ≥ L-1
  - Configurable K (default: 500 random Family-C draws per allocation)

Usage:
    python scripts/tune_comrt_exhaustive.py --L 12
    python scripts/tune_comrt_exhaustive.py --L 12 --S_max 6 --K 1000 --seed 42
    python scripts/tune_comrt_exhaustive.py --L 32 --output_dir design_points/
"""

import argparse
import json
import math
import random
import sys
from itertools import permutations as iperms
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from SeqSNN.module.comrt import build_encoding_table, _compute_scores_from_table, _build_strides_clipmrt

DEFAULT_OUTPUT_DIR = "./design_points"


# ---------------------------------------------------------------------------
# score_v2: same as standard but f2 exponent = 1 instead of 2
# ---------------------------------------------------------------------------

def score_v2(strides: List[int], bits: List[int], L: int) -> Dict:
    table = build_encoding_table(L, strides, bits)
    base = _compute_scores_from_table(table, L)
    f1 = base["f1_monotonicity"]
    f2 = base["f2_tail_resolution"]
    f3 = base["f3_local_resolution"]
    f4 = base["f4_consistency"]
    quality_v2 = (f1**1 * f2**1 * f3**1.5 * f4**1) ** (1.0 / 4.5)
    final_v2 = quality_v2 * base["bit_factor"]
    base["quality_score_v2"] = round(quality_v2, 6)
    base["final_score_v2"] = round(final_v2, 6)
    base["strides"] = list(strides)
    base["bits"] = list(bits)
    base["coverage_ok"] = bool(strides[-1] * bits[-1] >= L - 1)
    return base


# ---------------------------------------------------------------------------
# Full integer partition enumeration
# ---------------------------------------------------------------------------

def _all_partitions(total: int, n_parts: int, min_val: int, acc: List[int], out: List[Tuple]):
    """Enumerate all sorted tuples of length n_parts summing to total, each >= min_val."""
    if n_parts == 0:
        if total == 0:
            out.append(tuple(acc))
        return
    max_val = total - (n_parts - 1) * min_val
    for v in range(min_val, max_val + 1):
        acc.append(v)
        _all_partitions(total - v, n_parts - 1, v, acc, out)
        acc.pop()


def enumerate_all_allocs(S: int, total_bits: int, L: int) -> List[List[int]]:
    """All distinct permutations of integer partitions with sum=total_bits,
    each part >= 2, product >= L-1."""
    sorted_parts: List[Tuple] = []
    _all_partitions(total_bits, S, 2, [], sorted_parts)

    perms: set = set()
    for parts in sorted_parts:
        prod = 1
        for b in parts:
            prod *= b
        if prod < L - 1:
            continue
        for perm in iperms(parts):
            perms.add(perm)

    return [list(p) for p in perms]


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def exhaustive_tune(
    L: int,
    S_max: Optional[int] = None,
    K: int = 500,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[List[int], List[int], float, Dict]:
    if L <= 2:
        return [1], [1], 1.0, {}

    b_ref = math.ceil(math.e * math.log(L - 1)) + 1
    max_total_bits = max(b_ref + 2, int(b_ref * 1.5))

    if S_max is None:
        S_max = min(L - 1, 4)

    rng = random.Random(seed)

    best_strides: List[int] = [1]
    best_bits: List[int] = [L - 1]
    best_score = -1.0
    best_details: Dict = {}

    total_allocs = 0
    total_configs = 0

    for S in range(1, S_max + 1):
        for budget in range(S * 2, max_total_bits + 1):
            allocs = enumerate_all_allocs(S, budget, L)
            total_allocs += len(allocs)
            for bits in allocs:
                for _ in range(K):
                    strides = _build_strides_clipmrt(bits, L, rng)
                    bits_cur = list(bits)
                    if strides[-1] * bits_cur[-1] < L - 1:
                        bits_cur[-1] = max(bits_cur[-1], math.ceil((L - 1) / max(strides[-1], 1)))
                    details = score_v2(strides, bits_cur, L)
                    total_configs += 1
                    score = details["final_score_v2"]

                    if score > best_score:
                        best_score = score
                        best_strides = list(strides)
                        best_bits = list(bits_cur)
                        best_details = details

    if verbose:
        print(
            f"[Exhaustive] L={L}  S_max={S_max}  K={K}  "
            f"alloc_sets={total_allocs}  configs_scored={total_configs}"
        )
        print(
            f"  Best: S={len(best_strides)}  strides={best_strides}  bits={best_bits}  "
            f"total_bits={sum(best_bits)}  score_v2={best_score:.4f}"
        )
        print(
            f"  f1={best_details.get('f1_monotonicity', 0):.3f}  "
            f"f2={best_details.get('f2_tail_resolution', 0):.3f}  "
            f"f3={best_details.get('f3_local_resolution', 0):.3f}  "
            f"f4={best_details.get('f4_consistency', 0):.3f}"
        )

    return best_strides, best_bits, best_score, best_details


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exhaustive ClipMRT tuner.")
    parser.add_argument("--L", type=int, required=True, help="Sequence / axis length")
    parser.add_argument("--S_max", type=int, default=None,
                        help="Maximum number of scales to try (default: min(L-1, 8))")
    parser.add_argument("--K", type=int, default=500,
                        help="Random Family-C draws per allocation (default: 500)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save JSON result (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    print(f"\n{'='*64}")
    print(f"  Exhaustive ClipMRT tuning  L={args.L}  S_max={args.S_max or 'auto'}  K={args.K}  seed={args.seed}")
    print(f"{'='*64}\n")

    strides, bits, score, details = exhaustive_tune(
        L=args.L, S_max=args.S_max, K=args.K, seed=args.seed, verbose=True
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"clipmrt_exhaustive_L{args.L}_seed{args.seed}.json"

    result = {
        "L": args.L,
        "strides": strides,
        "bits": bits,
        **details,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
