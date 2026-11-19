#!/usr/bin/env python3
"""
Compute 25th / 50th / 75th percentile ClipMRT configs from a pool JSON.

Usage:
    python scripts/clipmrt_percentiles.py
    python scripts/clipmrt_percentiles.py --pool design_points/minmaxmedian_pool/pool_L168_S5_bits15_seed42.json
    python scripts/clipmrt_percentiles.py --output_dir design_points/percentiles
"""

import argparse
import json
from pathlib import Path


DEFAULT_POOL      = "./design_points/minmaxmedian_pool/pool_L168_S5_bits15_seed42.json"
DEFAULT_OUTPUT_DIR = "./design_points/percentiles"


def pick_percentile(pool: list, p: float) -> dict:
    """Return the entry closest to the p-th percentile of final_score."""
    sorted_pool = sorted(pool, key=lambda x: x["final_score"])
    idx = int(round(p / 100 * (len(sorted_pool) - 1)))
    idx = max(0, min(idx, len(sorted_pool) - 1))
    return sorted_pool[idx]


def main():
    parser = argparse.ArgumentParser(
        description="Select 25/50/75 percentile configs from a ClipMRT pool."
    )
    parser.add_argument("--pool", default=DEFAULT_POOL,
                        help=f"Path to pool JSON (default: {DEFAULT_POOL})")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory for output JSON (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    pool_path = Path(args.pool)
    if not pool_path.exists():
        print(f"ERROR: pool file not found: {pool_path}")
        raise SystemExit(1)

    with open(pool_path) as f:
        pool = json.load(f)

    print(f"\n  Pool: {pool_path.name}  ({len(pool)} configs)")

    percentiles = {"p25": 25.0, "p50": 50.0, "p75": 75.0}
    result = {}
    for label, p in percentiles.items():
        entry = pick_percentile(pool, p)
        entry = dict(entry)
        entry["regime"] = label
        result[label] = entry

    # Summary table
    print(f"\n  {'regime':>6}  {'score':>8}  {'strides':>26}  {'bits':>16}")
    print("  " + "-" * 62)
    for label, entry in result.items():
        print(f"  {label:>6}  {entry['final_score']:>8.6f}"
              f"  {str(entry['strides']):>26}  {str(entry['bits']):>16}")

    # Score range for context
    scores = sorted(e["final_score"] for e in pool)
    print(f"\n  Pool score range: [{scores[0]:.6f}, {scores[-1]:.6f}]")

    # Write output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pool_path.stem.replace("pool_", "percentiles_")
    output_path = output_dir / f"{stem}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Saved → {output_path}\n")


if __name__ == "__main__":
    main()
