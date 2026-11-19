#!/usr/bin/env python3
"""
ClipMRT Design Points  (L=168, S=5, total_bits=15)
====================================================
Generates five ClipMRT configurations for L=168 that land as close as
possible to target diagnostic scores [0.1, 0.3, 0.5, 0.8, 0.9], while
keeping S=5 and total_bits=15 identical across all five configurations.

This allows a controlled downstream experiment: the encoding budget is
constant, so any performance difference is attributable purely to the
diagnostic score.

Key insight for score diversity
---------------------------------
• Forcing m_1=1 (standard ClipMRT) gives resolution at adjacent positions
  and tends to produce high scores (≥ 0.5).
• Allowing m_1 > 1 makes positions 0…m_1-1 indistinguishable → f1 collapses
  → scores can reach near 0.
• Unequal bit allocations (e.g. [2,2,2,2,7]) with no product-coverage
  constraint further expand the low-score regime.

Pool construction
-----------------
1. Enumerate ALL bit-width permutations with sum=15 and each B_s ≥ 2
   (product constraint is relaxed — coverage comes from strides).
2. For each allocation run K "standard" draws (m_1=1, Family-C cascade)
   and K "free" draws (m_1 random in [1, L//4]).
3. Deduplicate; score every unique config.
4. Select the entry closest to each target score.

Usage:
    python scripts/clipmrt_design_points.py
    python scripts/clipmrt_design_points.py --K 60 --seed 42
    python scripts/clipmrt_design_points.py --output_dir ./design_points
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

from SeqSNN.module.comrt import diagnostic_score

# ── Fixed experiment parameters ──────────────────────────────────────────────
L           = 168
S_FIXED     = 5
TOTAL_BITS  = 15
TARGET_SCORES: List[float] = [0.1, 0.3, 0.5, 0.8, 0.9]


# ---------------------------------------------------------------------------
# Bit-width allocation enumeration  (no product constraint)
# ---------------------------------------------------------------------------

def _enum_sorted_multisets(
    rem_sum: int,
    rem_count: int,
    min_val: int,
    acc: List[int],
    result: List[Tuple[int, ...]],
) -> None:
    if rem_count == 0:
        if rem_sum == 0:
            result.append(tuple(acc))
        return
    max_val = rem_sum - (rem_count - 1) * min_val
    for b in range(min_val, max_val + 1):
        acc.append(b)
        _enum_sorted_multisets(rem_sum - b, rem_count - 1, b, acc, result)
        acc.pop()


def all_alloc_perms(S: int, total_bits: int) -> List[Tuple[int, ...]]:
    """All distinct permutations of (S-element, sum=total_bits, each ≥ 2)."""
    sorted_ms: List[Tuple[int, ...]] = []
    _enum_sorted_multisets(total_bits, S, 2, [], sorted_ms)
    perms: set = set()
    for ms in sorted_ms:
        for perm in iperms(ms):
            perms.add(perm)
    return list(perms)


# ---------------------------------------------------------------------------
# Stride samplers
# ---------------------------------------------------------------------------

def _build_standard(bits: List[int], L: int, rng: random.Random) -> List[int]:
    """Family-C random strides with m_1=1 (standard ClipMRT).

    Cascade is always respected; coverage may not be met (caller scores as-is).
    """
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
    m_S = rng.choice(ok) if ok else hi
    strides.append(m_S)
    return strides


def _build_free(
    bits: List[int],
    L: int,
    rng: random.Random,
    m1_max: Optional[int] = None,
) -> List[int]:
    """Random strides with m_1 drawn uniformly from [1, m1_max].

    Large m_1 makes positions 0…m_1-1 indistinguishable, driving scores low.
    Cascade is respected; coverage may not be met.
    """
    if m1_max is None:
        m1_max = max(1, L // 4)   # up to 42 for L=168
    S = len(bits)
    m1 = rng.randint(1, m1_max)
    strides = [m1]
    for s in range(1, S - 1):
        lo = strides[s - 1]
        hi = strides[s - 1] * bits[s - 1]
        cands = list(range(lo + 1, hi + 1))
        strides.append(rng.choice(cands) if cands else lo + 1)
    lo = strides[-1]
    hi = strides[-1] * bits[len(strides) - 1]
    needed = math.ceil((L - 1) / max(bits[-1], 1))
    ok = [c for c in range(lo + 1, hi + 1) if c >= needed]
    # If coverage not achievable within cascade, take a random cascade-valid stride
    m_S = rng.choice(ok) if ok else (rng.choice(list(range(lo + 1, hi + 1))) if hi > lo else lo + 1)
    strides.append(m_S)
    return strides


# ---------------------------------------------------------------------------
# Pool generation
# ---------------------------------------------------------------------------

def generate_pool(
    L: int,
    S: int,
    total_bits: int,
    K: int,
    seed: int,
) -> List[Dict]:
    """Build and score a diverse pool of (strides, bits) configurations."""
    rng = random.Random(seed)
    allocs = all_alloc_perms(S, total_bits)
    print(f"  {len(allocs)} distinct bit-width permutations  (S={S}, sum={total_bits}, each ≥ 2)")

    seen: set = set()
    pool: List[Dict] = []

    def _add(strides: List[int], bits_in: List[int]) -> None:
        key = (tuple(strides), tuple(bits_in))
        if key in seen:
            return
        seen.add(key)
        det = diagnostic_score(strides, bits_in, L)
        pool.append(det)

    for bits_tuple in allocs:
        bits = list(bits_tuple)

        # Standard draws: m_1=1
        for _ in range(K):
            _add(_build_standard(bits, L, rng), bits)

        # Free draws: m_1 random in [1, L//4]
        for _ in range(K):
            _add(_build_free(bits, L, rng), bits)

        # Deterministic Family-B tight cascade (one per allocation)
        strides_B = [1]
        for s in range(1, S):
            strides_B.append(strides_B[-1] * bits[s - 1])
        _add(strides_B, bits)

    scores = [d["final_score"] for d in pool]
    print(f"  Pool: {len(pool)} unique configs  "
          f"score range [{min(scores):.4f}, {max(scores):.4f}]")
    return pool


# ---------------------------------------------------------------------------
# Select design points
# ---------------------------------------------------------------------------

def select_design_points(pool: List[Dict], targets: List[float]) -> List[Dict]:
    """Return the pool entry closest to each target score."""
    selected = []
    for target in targets:
        best = min(pool, key=lambda d: abs(d["final_score"] - target))
        gap = abs(best["final_score"] - target)
        if gap > 0.05:
            print(f"  WARNING  target={target:.2f}  closest={best['final_score']:.4f}"
                  f"  gap={gap:.4f}  (outside achievable range)")
        selected.append(best)
    return selected


# ---------------------------------------------------------------------------
# Pretty summary
# ---------------------------------------------------------------------------

def _print_table(selected: List[Dict], targets: List[float]) -> None:
    hdr = (f"  {'target':>7}  {'actual':>7}  {'strides':>26}  {'bits':>14}"
           f"  {'f1':>6}  {'f2':>6}  {'f3':>6}  {'f4':>6}  {'cov':>5}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for target, d in zip(targets, selected):
        print(
            f"  {target:>7.2f}  {d['final_score']:>7.4f}"
            f"  {str(d['strides']):>26}  {str(d['bits']):>14}"
            f"  {d['f1_monotonicity']:>6.3f}"
            f"  {d['f2_tail_resolution']:>6.3f}"
            f"  {d['f3_local_resolution']:>6.3f}"
            f"  {d['f4_consistency']:>6.3f}"
            f"  {'Y' if d.get('coverage_ok') else 'N':>5}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 5 ClipMRT design points for L=168 at target scores."
    )
    parser.add_argument("--K", type=int, default=50,
                        help="Random draws per allocation per sampler (default: 50)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--output_dir", type=str, default="./design_points",
                        help="Output directory for JSON files (default: ./design_points)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}")
    print(f"  ClipMRT design points  L={L}  S={S_FIXED}  total_bits={TOTAL_BITS}")
    print(f"  K={args.K} draws/alloc × 2 samplers   seed={args.seed}")
    print(f"{'='*68}")

    pool = generate_pool(
        L=L, S=S_FIXED, total_bits=TOTAL_BITS,
        K=args.K, seed=args.seed,
    )

    if not pool:
        print("ERROR: pool is empty — no valid configs found.")
        sys.exit(1)

    selected = select_design_points(pool, TARGET_SCORES)

    print(f"\n{'='*68}")
    print(f"  Selected design points")
    print(f"{'='*68}")
    _print_table(selected, TARGET_SCORES)

    all_results: Dict = {}
    for target, d in zip(TARGET_SCORES, selected):
        label = f"{target:.1f}".replace(".", "p")
        entry = dict(d)
        entry["target_score"] = target
        json_path = output_dir / f"design_point_target{label}.json"
        with open(json_path, "w") as f:
            json.dump(entry, f, indent=2)
        all_results[str(target)] = entry

    summary_path = output_dir / "design_points_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  JSON files written to {output_dir}/")
    print(f"  Summary → {summary_path}")


if __name__ == "__main__":
    main()
