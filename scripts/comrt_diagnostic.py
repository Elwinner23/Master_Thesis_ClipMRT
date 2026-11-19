#!/usr/bin/env python3
"""
CoMRT Diagnostic Script
=======================
Runs the CoMRT tuning procedure for one or more sequence lengths and writes
a JSON file for each (plus a combined summary), then prints a human-readable
table to the terminal.

Usage examples:
    # Single L, auto-tune, save JSON to ./diagnostics/
    python scripts/comrt_diagnostic.py --L 168

    # Multiple L values at once
    python scripts/comrt_diagnostic.py --L 12 168 256 1000

    # Score a fixed configuration (no tuning)
    python scripts/comrt_diagnostic.py --L 168 --strides 1 4 7 23 --bits 5 5 5 5

    # Custom output directory
    python scripts/comrt_diagnostic.py --L 12 168 256 1000 --output_dir ./results/comrt
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from SeqSNN.module.comrt import (
    build_encoding_table,
    compute_mean_similarity,
    diagnostic_score,
    tune_comrt,
)


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _box(title: str, rows: List[Tuple[str, str]]) -> None:
    width = max(len(k) + len(v) for k, v in rows) + 6
    width = max(width, 50)
    bar   = "─" * width
    print(f"┌{bar}┐")
    print(f"│  {title:<{width - 2}}│")
    print(f"├{bar}┤")
    for k, v in rows:
        row = f"  {k} : {v}"
        print(f"│{row:<{width}}│")
    print(f"└{bar}┘")


def _profile_summary(L: int, strides: List[int], bits: List[int]) -> List[Tuple[str, str]]:
    table = build_encoding_table(L, strides, bits)
    mu    = compute_mean_similarity(table)
    rows  = [
        ("μ(0)  identical",   f"{mu[0].item():.2f}"),
        ("μ(1)  adjacent",    f"{mu[1].item():.2f}"),
    ]
    if L > 5:
        mid = L // 2
        rows.append((f"μ({mid:4d}) mid-range", f"{mu[mid].item():.2f}"))
    rows.append((f"μ({L-1:4d}) max offset", f"{mu[L-1].item():.2f}"))
    mono_ok = all(mu[d] > mu[d + 1] for d in range(1, L - 1))
    rows.append(("Strictly monotone", "YES ✓" if mono_ok else "NO ✗  (mean-collision)"))
    return rows


def _result_rows(d: Dict) -> List[Tuple[str, str]]:
    return [
        ("final_score      ", f"{d['final_score']:.6f}"),
        ("quality_score    ", f"{d['quality_score']:.6f}"),
        ("bit_factor       ", f"{d['bit_factor']:.6f}"),
        ("f1  monotonicity ", f"{d['f1_monotonicity']:.6f}"),
        ("f2  tail_res.    ", f"{d['f2_tail_resolution']:.6f}"),
        ("f3  local_res.   ", f"{d['f3_local_resolution']:.6f}"),
        ("f4  consistency  ", f"{d['f4_consistency']:.6f}"),
        ("nsd_avg          ", f"{d['nsd_avg']:.6f}"),
        ("total_bits       ", str(d["total_bits"])),
        ("b_ref*           ", str(d["b_ref"])),
        ("strides          ", str(d["strides"])),
        ("bits             ", str(d["bits"])),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one(
    L: int,
    strides: Optional[List[int]],
    bits: Optional[List[int]],
    output_dir: Path,
) -> Dict:
    """Tune or score for a single L.  Always writes a JSON file."""
    b_ref = math.ceil(math.e * math.log(max(L - 1, 2))) + 1
    S_opt = max(1, int(math.log(max(L - 1, 2))))
    print(f"\n{'='*60}")
    print(f"  L = {L}    S* ≈ {S_opt}    b_ref* = {b_ref}")
    print(f"{'='*60}")

    json_path = output_dir / f"comrt_diagnostic_L{L}.json"

    if strides is not None and bits is not None:
        # Evaluate a fixed configuration
        if len(strides) != len(bits):
            raise ValueError("--strides and --bits must have the same length")
        print(f"  Evaluating fixed config: strides={strides}  bits={bits}")
        result = diagnostic_score(strides, bits, L, output_path=json_path)
    else:
        # Auto-tune (writes JSON via tune_comrt)
        print(f"  Auto-tuning (output → {json_path}) ...")
        chosen_s, chosen_b, _, result = tune_comrt(
            L=L, verbose=True, output_dir=output_dir
        )
        result.setdefault("strides", chosen_s)
        result.setdefault("bits",    chosen_b)
        # tune_comrt already wrote the JSON; re-write with output_path arg
        # so the file name matches the diagnostic naming convention
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

    # Terminal output
    _box(f"Diagnostic score  L={L}", _result_rows(result))
    prof_rows = _profile_summary(L, result["strides"], result["bits"])
    _box(f"Similarity profile  L={L}", prof_rows)
    print(f"  → JSON saved to {json_path}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="CoMRT diagnostic: auto-tune or score a fixed config for given L values."
    )
    parser.add_argument("--L", type=int, nargs="+", required=True,
                        help="Sequence length(s), e.g. --L 12 168 256 1000")
    parser.add_argument("--strides", type=int, nargs="+", default=None,
                        help="Fixed stride list (applied to every L; skips tuning)")
    parser.add_argument("--bits", type=int, nargs="+", default=None,
                        help="Fixed bit-width list (must pair with --strides)")
    parser.add_argument("--output_dir", type=str, default="./diagnostics",
                        help="Directory for JSON output files (default: ./diagnostics)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for L in args.L:
        result = run_one(
            L=L,
            strides=args.strides,
            bits=args.bits,
            output_dir=output_dir,
        )
        all_results[str(L)] = result

    # Combined summary across all L values
    if len(args.L) > 1:
        summary_path = output_dir / "comrt_diagnostic_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Combined summary → {summary_path}")

        print(f"\n{'='*60}")
        print(f"  Summary across L values")
        print(f"{'='*60}")
        header = f"  {'L':>6}  {'S':>3}  {'bits':>10}  {'total_b':>7}  {'score':>8}  {'f1':>6}  {'f2':>6}  {'f3':>6}  {'f4':>6}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for L_str, d in all_results.items():
            print(
                f"  {L_str:>6}  {len(d['strides']):>3}  "
                f"{str(d['bits']):>10}  {d['total_bits']:>7}  "
                f"{d['final_score']:>8.4f}  "
                f"{d['f1_monotonicity']:>6.3f}  "
                f"{d['f2_tail_resolution']:>6.3f}  "
                f"{d['f3_local_resolution']:>6.3f}  "
                f"{d['f4_consistency']:>6.3f}"
            )


if __name__ == "__main__":
    main()
