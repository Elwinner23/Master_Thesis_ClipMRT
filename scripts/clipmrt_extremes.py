#!/usr/bin/env python3
"""
ClipMRT Extremes  (L=168, S=5, total_bits=15)
==============================================
Builds a large random pool of (strides, bits) configurations via two
Family-C samplers — standard (m_1=1) and free (m_1 random up to L//4) —
then reports the minimum-, median-, and maximum-scoring configs and saves
the full pool to JSON.

Usage:
    python scripts/clipmrt_extremes.py
    python scripts/clipmrt_extremes.py --K 100 --seed 42
    python scripts/clipmrt_extremes.py --output_dir ./my_pool
"""

import argparse
import json
import math
import random
import sys
from itertools import permutations as iperms
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from SeqSNN.module.comrt import diagnostic_score

# ── Fixed experiment parameters ──────────────────────────────────────────────
L          = 64
S_FIXED    = 4
TOTAL_BITS = 13

DEFAULT_OUTPUT_DIR = "./design_points/"


# ---------------------------------------------------------------------------
# Bit-width allocation enumeration
# ---------------------------------------------------------------------------

def _enum_sorted_multisets(rem_sum, rem_count, min_val, acc, result):
    if rem_count == 0:
        if rem_sum == 0:
            result.append(tuple(acc))
        return
    max_val = rem_sum - (rem_count - 1) * min_val
    for b in range(min_val, max_val + 1):
        acc.append(b)
        _enum_sorted_multisets(rem_sum - b, rem_count - 1, b, acc, result)
        acc.pop()


def all_alloc_perms(S: int, total_bits: int, L: int) -> List[Tuple[int, ...]]:
    """All distinct permutations with sum=total_bits, each ≥ 2, product ≥ L-1."""
    sorted_ms: List[Tuple[int, ...]] = []
    _enum_sorted_multisets(total_bits, S, 2, [], sorted_ms)
    perms: set = set()
    for ms in sorted_ms:
        prod = 1
        for b in ms:
            prod *= b
        if prod < L - 1:
            continue
        for perm in iperms(ms):
            perms.add(perm)
    return list(perms)


# ---------------------------------------------------------------------------
# Family-C samplers
# ---------------------------------------------------------------------------

def _build_standard(bits: List[int], L: int, rng: random.Random) -> List[int]:
    """m_1=1, Family-C cascade (high scores)."""
    S = len(bits)
    if S == 1:
        return [1]
    strides = [1]
    for s in range(1, S - 1):
        lo = strides[s - 1]
        hi = strides[s - 1] * bits[s - 1]
        strides.append(rng.choice(list(range(lo + 1, hi + 1))))
    lo = strides[-1]
    hi = strides[-1] * bits[len(strides) - 1]
    needed = math.ceil((L - 1) / max(bits[-1], 1))
    ok = [c for c in range(lo + 1, hi + 1) if c >= needed]
    strides.append(rng.choice(ok) if ok else hi)
    return strides


# ---------------------------------------------------------------------------
# Pool generation
# ---------------------------------------------------------------------------

def generate_pool(L: int, S: int, total_bits: int, K: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    allocs = all_alloc_perms(S, total_bits, L)
    print(f"  {len(allocs)} distinct bit-width permutations  (S={S}, sum={total_bits}, each ≥ 2, product ≥ {L-1})")

    seen: set = set()
    pool: List[Dict] = []

    def _add(strides, bits_in):
        key = (tuple(strides), tuple(bits_in))
        if key in seen:
            return
        seen.add(key)
        det = diagnostic_score(strides, bits_in, L)
        if det["coverage_ok"]:
            pool.append(det)

    for bits_tuple in allocs:
        bits = list(bits_tuple)
        for _ in range(K):
            _add(_build_standard(bits, L, rng), bits)
        # Deterministic tight-cascade (Family-B) as anchor
        strides_B = [1]
        for s in range(1, S):
            strides_B.append(strides_B[-1] * bits[s - 1])
        _add(strides_B, bits)

    scores = [d["final_score"] for d in pool]
    print(f"  Pool: {len(pool)} unique configs (coverage_ok=True only)  "
          f"score range [{min(scores):.4f}, {max(scores):.4f}]")
    return pool


# ---------------------------------------------------------------------------
# Select min / median / max
# ---------------------------------------------------------------------------

def select_extremes(pool: List[Dict]) -> Dict[str, Dict]:
    sorted_pool = sorted(pool, key=lambda d: d["final_score"])
    n = len(sorted_pool)
    return {
        "min":    sorted_pool[0],
        "median": sorted_pool[n // 2],
        "max":    sorted_pool[-1],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build a random ClipMRT pool and report min/median/max diagnostic scores."
    )
    parser.add_argument("--K", type=int, default=50,
                        help="Random draws per allocation per sampler (default: 50)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory for JSON output (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    print(f"\n{'='*64}")
    print(f"  ClipMRT extremes  L={L}  S={S_FIXED}  total_bits={TOTAL_BITS}")
    print(f"  K={args.K} draws/alloc × 2 samplers   seed={args.seed}")
    print(f"{'='*64}")

    pool = generate_pool(L=L, S=S_FIXED, total_bits=TOTAL_BITS, K=args.K, seed=args.seed)
    if not pool:
        print("ERROR: pool is empty.")
        sys.exit(1)

    extremes = select_extremes(pool)

    print(f"\n{'='*64}")
    print(f"  Min / Median / Max")
    print(f"{'='*64}")
    hdr = f"  {'regime':>8}  {'score':>7}  {'strides':>26}  {'bits':>14}  {'f1':>6}  {'f2':>6}  {'f3':>6}  {'f4':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for regime, d in extremes.items():
        print(
            f"  {regime:>8}  {d['final_score']:>7.4f}"
            f"  {str(d['strides']):>26}  {str(d['bits']):>14}"
            f"  {d['f1_monotonicity']:>6.3f}"
            f"  {d['f2_tail_resolution']:>6.3f}"
            f"  {d['f3_local_resolution']:>6.3f}"
            f"  {d['f4_consistency']:>6.3f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full pool
    pool_path = output_dir / f"pool_L{L}_S{S_FIXED}_bits{TOTAL_BITS}_seed{args.seed}.json"
    with open(pool_path, "w") as f:
        json.dump(pool, f, indent=2)

    # Save extremes summary
    summary_path = output_dir / f"extremes_L{L}_seed{args.seed}.json"
    with open(summary_path, "w") as f:
        json.dump({regime: dict(d, regime=regime) for regime, d in extremes.items()}, f, indent=2)

    print(f"\n  Full pool ({len(pool)} configs) → {pool_path}")
    print(f"  Extremes summary          → {summary_path}")


if __name__ == "__main__":
    main()
