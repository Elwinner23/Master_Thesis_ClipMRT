#!/usr/bin/env python3
"""
Stride-Family Ablation  (CLAUDE commands §"Stride-family ablation")
====================================================================
Compares three stride-construction families — A (Coprime), B (Tight Cascade),
C (Random) — under a fixed bit budget, for each target L.

All three families share:
  • S = floor(ln(L-1))
  • Balanced bit allocation  B_s ≈ (L-1)^{1/S}
  • m_1 = 1

Family A — Coprime
  Middle strides: random-coprime from (m_{s-1}, m_{s-1}B_{s-1}]
  Coarsest stride: largest coprime in range satisfying coverage
  Repeat K draws, keep best diagnostic score

Family B — Tight Cascade (deterministic)
  m_s = Π_{k<s} B_k   →  m_s = m_{s-1} * B_{s-1}
  No search; maximises saturation reach at this budget

Family C — Random (no coprime constraint)
  Strides drawn uniformly from (m_{s-1}, m_{s-1}B_{s-1}], cascade only
  m_S resampled until m_S*B_S ≥ L-1; best of K draws

Usage:
    python scripts/comrt_ablation_families.py --L 12 168 256 1000
    python scripts/comrt_ablation_families.py --L 168 --K 200 --output_dir ./ablation
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from SeqSNN.module.comrt import (
    _largest_coprime_le,
    _smallest_coprime_ge,
    diagnostic_score,
    build_encoding_table,
    compute_mean_similarity,
)


def _coverage_ok(strides: List[int], bits: List[int], L: int) -> bool:
    return strides[-1] * bits[-1] >= L - 1


# ---------------------------------------------------------------------------
# Shared setup helpers
# ---------------------------------------------------------------------------

def _compute_S(L: int) -> int:
    return max(1, int(math.log(max(L - 1, 2))))


def _allocate_bits(S: int, L: int) -> List[int]:
    B = max(2, math.ceil((L - 1) ** (1.0 / S)))
    bits = [B] * S
    prod = 1
    for b in bits:
        prod *= b
    if prod < L - 1:
        bits[-1] = max(bits[-1], math.ceil((L - 1) / max(prod // bits[-1], 1)))
    return bits


def _coprime_candidates(lo: int, hi: int, existing: List[int]) -> List[int]:
    from math import gcd
    return [c for c in range(lo + 1, hi + 1)
            if all(gcd(c, m) == 1 for m in existing)]


# ---------------------------------------------------------------------------
# Family B — Tight Cascade (deterministic)
# ---------------------------------------------------------------------------

def build_B(bits: List[int]) -> List[int]:
    strides = [1]
    prod = 1
    for B in bits[:-1]:
        prod *= B
        strides.append(prod)
    return strides


# ---------------------------------------------------------------------------
# Family A — Coprime  (K random draws)
# ---------------------------------------------------------------------------

def build_A(
    bits: List[int],
    L: int,
    K: int = 100,
    rng: Optional[random.Random] = None,
) -> Tuple[List[int], float, Dict]:
    if rng is None:
        rng = random.Random(0)
    S = len(bits)
    best_strides: List[int] = []
    best_score = -1.0
    best_details: Dict = {}

    for _ in range(K):
        strides: List[int] = [1]
        valid = True

        # Middle scales: s = 2 … S-1
        for s in range(1, S - 1):
            lo   = strides[s - 1]
            hi   = strides[s - 1] * bits[s - 1]
            cands = _coprime_candidates(lo, hi, strides)
            if not cands:
                valid = False
                break
            strides.append(rng.choice(cands))

        if not valid:
            continue

        # Coarsest scale: largest coprime satisfying coverage
        lo  = strides[-1]
        hi  = strides[-1] * bits[-2] if S > 1 else 1
        min_ms = max(lo + 1, math.ceil((L - 1) / max(bits[-1], 1)))
        cands  = _coprime_candidates(lo, hi, strides)
        cov_ok = [c for c in cands if c >= min_ms]
        if cov_ok:
            m_S = max(cov_ok)
        elif cands:
            m_S = max(cands)
        else:
            m_S = _smallest_coprime_ge(min_ms, strides)
        strides.append(m_S)

        det   = diagnostic_score(strides, bits, L)
        score = det["final_score"]
        if score > best_score and _coverage_ok(strides, bits, L):
            best_score   = score
            best_strides = list(strides)
            best_details = det

    if not best_strides:
        best_strides = build_B(bits)   # fallback to B
        best_details = diagnostic_score(best_strides, bits, L)
        best_score   = best_details["final_score"]

    return best_strides, best_score, best_details


# ---------------------------------------------------------------------------
# Family C — Random, no coprime constraint  (K draws)
# ---------------------------------------------------------------------------

def build_C(
    bits: List[int],
    L: int,
    K: int = 100,
    rng: Optional[random.Random] = None,
) -> Tuple[List[int], float, Dict]:
    if rng is None:
        rng = random.Random(0)
    S = len(bits)
    best_strides: List[int] = []
    best_score = -1.0
    best_details: Dict = {}

    for _ in range(K):
        strides: List[int] = [1]
        valid = True

        for s in range(1, S - 1):
            lo    = strides[s - 1]
            hi    = strides[s - 1] * bits[s - 1]
            cands = list(range(lo + 1, hi + 1))
            if not cands:
                valid = False
                break
            strides.append(rng.choice(cands))

        if not valid:
            continue

        lo     = strides[-1]
        hi     = strides[-1] * bits[-2] if S > 1 else 1
        needed = math.ceil((L - 1) / max(bits[-1], 1))
        cands  = [c for c in range(lo + 1, hi + 1) if c >= needed]
        if cands:
            m_S = rng.choice(cands)
        else:
            m_S = max(lo + 1, needed)
        strides.append(m_S)

        det   = diagnostic_score(strides, bits, L)
        score = det["final_score"]
        if score > best_score and _coverage_ok(strides, bits, L):
            best_score   = score
            best_strides = list(strides)
            best_details = det

    if not best_strides:
        best_strides = build_B(bits)
        best_details = diagnostic_score(best_strides, bits, L)
        best_score   = best_details["final_score"]

    return best_strides, best_score, best_details


# ---------------------------------------------------------------------------
# Run ablation for one L
# ---------------------------------------------------------------------------

def ablate_one(L: int, K: int, seed: int, output_dir: Path) -> Dict:
    S    = _compute_S(L)
    bits = _allocate_bits(S, L)
    b    = sum(bits)

    print(f"\n{'='*64}")
    print(f"  Ablation  L={L}   S={S}   bits={bits}   total_bits={b}")
    print(f"{'='*64}")

    # --- Family B (deterministic) ---
    strides_B = build_B(bits)
    # Enforce coverage
    if strides_B[-1] * bits[-1] < L - 1:
        bits_B = list(bits)
        bits_B[-1] = math.ceil((L - 1) / max(strides_B[-1], 1))
    else:
        bits_B = bits
    det_B = diagnostic_score(strides_B, bits_B, L)
    det_B["family"] = "B"

    # --- Family A (K draws) ---
    _, _, det_A = build_A(bits, L, K=K, rng=random.Random(seed))
    det_A["family"] = "A"

    # --- Family C (K draws) ---
    _, _, det_C = build_C(bits, L, K=K, rng=random.Random(seed + 1))
    det_C["family"] = "C"

    families = {"A": det_A, "B": det_B, "C": det_C}

    # Print comparison table
    header = f"  {'Family':>8}  {'strides':>22}  {'bits':>12}  {'score':>8}  {'f1':>6}  {'f2':>6}  {'f3':>6}  {'f4':>6}  {'coprime':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, d in families.items():
        print(
            f"  {name:>8}  {str(d['strides']):>22}  {str(d['bits']):>12}  "
            f"{d['final_score']:>8.4f}  "
            f"{d['f1_monotonicity']:>6.3f}  "
            f"{d['f2_tail_resolution']:>6.3f}  "
            f"{d['f3_local_resolution']:>6.3f}  "
            f"{d['f4_consistency']:>6.3f}  "
            f"{'yes' if d.get('coprime', False) else 'no':>7}"
        )

    # Determine winner
    best_fam = max(families, key=lambda k: families[k]["final_score"])
    print(f"\n  Best family for L={L}: {best_fam}  "
          f"(score={families[best_fam]['final_score']:.4f})")

    result = {
        "L":        L,
        "S":        S,
        "bits":     bits,
        "total_bits": b,
        "K":        K,
        "families": families,
        "best_family": best_fam,
    }

    json_path = output_dir / f"ablation_families_L{L}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → JSON saved to {json_path}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stride-family ablation: compare Families A, B, C for given L values."
    )
    parser.add_argument("--L", type=int, nargs="+", required=True,
                        help="Sequence length(s), e.g. --L 12 168 256 1000")
    parser.add_argument("--K", type=int, default=100,
                        help="Random draws per family (default: 100)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--output_dir", type=str, default="./ablation",
                        help="Directory for JSON output (default: ./ablation)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for L in args.L:
        result = ablate_one(L=L, K=args.K, seed=args.seed, output_dir=output_dir)
        all_results[str(L)] = result

    # Combined summary
    summary_path = output_dir / "ablation_families_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*64}")
    print("  Cross-L summary  (best family per L)")
    print(f"{'='*64}")
    print(f"  {'L':>6}  {'best':>6}  {'score_A':>8}  {'score_B':>8}  {'score_C':>8}")
    print("  " + "-" * 50)
    for L_str, res in all_results.items():
        fams = res["families"]
        print(
            f"  {L_str:>6}  {res['best_family']:>6}  "
            f"{fams['A']['final_score']:>8.4f}  "
            f"{fams['B']['final_score']:>8.4f}  "
            f"{fams['C']['final_score']:>8.4f}"
        )
    print(f"\n  Combined summary → {summary_path}")


if __name__ == "__main__":
    main()
