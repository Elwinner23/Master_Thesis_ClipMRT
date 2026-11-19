#!/usr/bin/env python3
"""
Run Gray PE or Log PE experiments across multiple seeds.
=========================================================
Reads an existing YAML config, overrides runtime.seed and runtime.output_dir
for each seed, and runs training sequentially. Generated YAMLs are placed
next to the source config so _base_ relative paths resolve correctly.

Usage:
    python scripts/run_pe_seeds.py --config exp/forecast/spikformer/spikformer_xnor_gray_metr-la.yml --seeds 42 101 1

    python scripts/run_pe_seeds.py \\
        --config exp/forecast/spikformer/spikformer_xnor_log_electricity.yml \\
        --seeds 42 101 123 --dry_run

    python scripts/run_pe_seeds.py \\
        --config exp/forecast/spikformer/spikformer_xnor_gray_solar.yml \\
        --seeds 42 101 123 \\
        --output_root ./outputs/gray_runs
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

DEFAULT_OUTPUT_ROOT = "./outputs/FullMRT6PemsBay/"


def main():
    parser = argparse.ArgumentParser(
        description="Run a Gray/Log PE config across multiple seeds."
    )
    parser.add_argument("--config", required=True,
                        help="Path to the base YAML config file")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 1],
                        help="Seeds to run (default: 42 101 123)")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT,
                        help=f"Root directory for outputs (default: {DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--device", type=int, default=None,
                        help="CUDA device index to use, e.g. --device 1 sets CUDA_VISIBLE_DEVICES=1")
    parser.add_argument("--dry_run", action="store_true",
                        help="Write YAMLs without launching experiments")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    exp_name = config_path.stem          # e.g. "spikformer_xnor_gray_solar"
    config_dir = config_path.parent      # keep generated YAMLs beside the source

    print(f"\n{'='*64}")
    print(f"  PE experiment : {exp_name}")
    print(f"  seeds         : {args.seeds}  ({len(args.seeds)} runs)")
    print(f"  output_root   : {args.output_root}")
    print(f"  device        : {'cuda:' + str(args.device) if args.device is not None else 'default'}")
    print(f"{'='*64}")

    all_configs = []
    for seed in args.seeds:
        run_name   = f"{exp_name}_seed{seed}"
        output_dir = str(Path(args.output_root) / run_name)

        cfg = dict(base_config)
        cfg.setdefault("runtime", {})
        cfg["runtime"]["seed"]       = seed
        cfg["runtime"]["output_dir"] = output_dir

        yaml_path = config_dir / f"{run_name}.yml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        all_configs.append((seed, yaml_path, output_dir, run_name))
        print(f"  seed={seed:>6}  → {yaml_path}")

    print()

    if args.dry_run:
        print("  --dry_run: YAMLs written, no experiments launched.")
        print("\n  To run manually:")
        for seed, yaml_path, _, run_name in all_configs:
            print(f"    python -m SeqSNN.entry.tsforecast {yaml_path}  # {run_name}")
        return

    for i, (seed, yaml_path, output_dir, run_name) in enumerate(all_configs, 1):
        print(f"\n{'='*64}")
        print(f"  [{i}/{len(all_configs)}]  {run_name}")
        print(f"{'='*64}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        cmd = [sys.executable, "-m", "SeqSNN.entry.tsforecast", str(yaml_path)]
        print(f"  CMD: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"\n  WARNING: {run_name} exited with code {result.returncode}. Continuing.")

    print(f"\n{'='*64}")
    print("  All runs complete.")
    print(f"  Results in: {args.output_root}/")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
