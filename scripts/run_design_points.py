#!/usr/bin/env python3
"""
Run downstream experiments for each ClipMRT design point.
==========================================================
Reads design_points_summary.json, generates one YAML config per design
point with strides/bits fixed (no auto-tuning), then launches training.

Generated YAMLs are placed in exp/forecast/<model_dir>/ (same level as the
original config) so the _base_: ../dataset/<name>.yml path resolves correctly.

Usage:
    # SpikformerCoMRT (default), dry run:
    python scripts/run_design_points.py --dry_run

    # SpikingformerCoMRT on metr-la with multiple seeds:
    python scripts/run_design_points.py --network SpikformerCoMRT --dataset metr-la --seeds 42 101 1

    # Custom batch size override:
    python scripts/run_design_points.py --network SpikformerCoMRT --batch_size 32 --seeds 42

    # Custom paths:
    python scripts/run_design_points.py \\
        --summary ./design_points/design_points_summary.json \\
        --dataset electricity \\
        --seeds 42 101 \\
        --output_root ./outputs/design_points/CoMRT_0.125scale
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ── defaults ─────────────────────────────────────────────────────────────────
#"/media/homes/guseinov/thesis/CoMRT/design_points/design_points_max_V2_L12.json"
DEFAULT_SUMMARY     = "/media/homes/guseinov/thesis/CoMRT/design_points/design_point_max_L12_relaxed.json"
DEFAULT_OUTPUT_ROOT = "./outputs/Metr_la24ClipMRT/relaxed"
#/media/homes/guseinov/thesis/CoMRT/design_points/design_points_max_L12.json
# normalize value per dataset (adjust if your setup differs)
DATASET_NORMALIZE = {
    "electricity": 3,
    "solar":       1,
    "metr-la":     2,
    "pems-bay":    2,
    "traffic":     3,
    "exchange":    0,
    "etth1":       3,
    "etth2":       3,
}

# Per-network defaults: config subdir and batch size
NETWORK_DEFAULTS = {
    "SpikformerCoMRT": {
        "config_dir": "exp/forecast/spikformer",
        "batch_size": 16,
    },
    "SpikingformerCoMRT": {
        "config_dir": "exp/forecast/spikingformer",
        "batch_size": 32,
    },
    "SpikformerCoMRT_Log": {
        "config_dir": "exp/forecast/spikformer",
        "batch_size": 16,
    }
}


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------

_YAML_TEMPLATE = """\
_base_:
- ../dataset/{dataset}.yml
_custom_imports_:
- SeqSNN.network
- SeqSNN.dataset
- SeqSNN.runner

data:
  raw_label: False
  window: 12
  horizon: 24
  normalize: {normalize}

runner:
  type: ts
  task: regression
  optimizer: Adam
  lr: 0.001
  weight_decay: 0.0
  loss_fn: mse
  metrics: [r2, rse]
  observe: loss
  lower_is_better: True
  max_epoches: 1000
  early_stop: 30
  batch_size: {batch_size}
  aggregate: True

network:
  type: {network}
  dim: 256
  d_ff: 1024
  depths: 2
  num_steps: 4
  heads: 8
  max_length: 12
  pe_mode: concat
  strides: {strides}
  bits: {bits}
  # diagnostic_score: {score:.4f}  (target: {target})

runtime:
  seed: {seed}
  output_dir: {output_dir}
"""


def _fmt_list(lst):
    return "[" + ", ".join(str(x) for x in lst) + "]"


def generate_yaml(dp: dict, dataset: str, normalize: int, output_dir: str,
                  seed: int, network: str, batch_size: int) -> str:
    return _YAML_TEMPLATE.format(
        dataset=dataset,
        normalize=normalize,
        network=network,
        batch_size=batch_size,
        strides=_fmt_list(dp["strides"]),
        bits=_fmt_list(dp["bits"]),
        score=dp["final_score"],
        target=dp.get("target_score", "?"),
        output_dir=output_dir,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate configs and run downstream experiments for ClipMRT design points."
    )
    parser.add_argument("--summary", default=DEFAULT_SUMMARY,
                        help=f"Path to design_points_summary.json (default: {DEFAULT_SUMMARY})")
    parser.add_argument("--network", default="SpikformerCoMRT",
                        choices=list(NETWORK_DEFAULTS.keys()),
                        help="Network type to use (default: SpikformerCoMRT)")
    parser.add_argument("--dataset", default="solar",
                        help="Dataset name matching exp/forecast/dataset/<name>.yml (default: solar)")
    parser.add_argument("--normalize", type=int, default=None,
                        help="Normalize value (auto-detected from dataset if omitted)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size override (default: per-network — SpikformerCoMRT=64, SpikingformerCoMRT=32)")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT,
                        help=f"Root dir for experiment outputs (default: {DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--dry_run", action="store_true",
                        help="Generate YAML files without running experiments")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="List of random seeds to run (default: 42)")
    parser.add_argument("--device", type=int, default=None,
                        help="CUDA device index to use, e.g. --device 1 sets CUDA_VISIBLE_DEVICES=1")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)

    net_cfg    = NETWORK_DEFAULTS[args.network]
    batch_size = args.batch_size if args.batch_size is not None else net_cfg["batch_size"]
    config_dir = Path(net_cfg["config_dir"])

    normalize = args.normalize
    if normalize is None:
        normalize = DATASET_NORMALIZE.get(args.dataset, 3)
        print(f"  normalize={normalize} (auto from dataset '{args.dataset}')")

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"ERROR: summary file not found: {summary_path}")
        print("  Run scripts/clipmrt_design_points.py first to generate it.")
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    config_dir.mkdir(parents=True, exist_ok=True)

    # short prefix for generated YAML filenames
    net_prefix = "spikformer" if args.network == "SpikformerCoMRT" else "spikingformer"

    total_runs = len(summary) * len(args.seeds)
    print(f"\n{'='*64}")
    print(f"  ClipMRT design-point experiments")
    print(f"  network={args.network}  batch_size={batch_size}")
    print(f"  dataset={args.dataset}  normalize={normalize}")
    print(f"  device={'cuda:' + str(args.device) if args.device is not None else 'default'}")
    print(f"  seeds={args.seeds}  ({len(summary)} configs × {len(args.seeds)} seeds = {total_runs} runs)")
    print(f"{'='*64}")

    # Build all (seed, design_point) combos and generate YAMLs
    all_configs = []
    for seed in args.seeds:
        print(f"\n  seed={seed}")
        print(f"  {'target':>7}  {'actual':>7}  {'strides':>26}  {'bits':>14}")
        print("  " + "-" * 60)
        for target_str, dp in summary.items():
            target = float(target_str)
            score  = dp["final_score"]
            label  = f"score{target_str.replace('.', 'p')}_seed{seed}"
            output_dir = str(Path(args.output_root) / args.network / args.dataset / label)

            yaml_text = generate_yaml(dp, args.dataset, normalize, output_dir,
                                      seed, args.network, batch_size)
            yaml_name = f"{net_prefix}_clipmrt_{args.dataset}_{label}.yml"
            yaml_path = config_dir / yaml_name
            with open(yaml_path, "w") as f:
                f.write(yaml_text)

            all_configs.append((seed, target, score, yaml_path, dp, output_dir))
            print(f"  {target:>7.2f}  {score:>7.4f}"
                  f"  {str(dp['strides']):>26}  {str(dp['bits']):>14}"
                  f"  → {yaml_name}")

    print()

    if args.dry_run:
        print("  --dry_run: YAML files written, no experiments launched.")
        print("\n  To run manually:")
        for seed, _, _, yaml_path, _, _ in all_configs:
            print(f"    python -m SeqSNN.entry.tsforecast {yaml_path}  # seed={seed}")
        return

    # Run all experiments sequentially
    for i, (seed, target, score, yaml_path, dp, output_dir) in enumerate(all_configs, 1):
        print(f"\n{'='*64}")
        print(f"  [{i}/{total_runs}]  seed={seed}  target={target:.2f}  actual={score:.4f}"
              f"  strides={dp['strides']}  bits={dp['bits']}")
        print(f"{'='*64}")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        cmd = [sys.executable, "-m", "SeqSNN.entry.tsforecast", str(yaml_path)]
        print(f"  CMD: {' '.join(str(c) for c in cmd)}\n")

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"\n  WARNING: seed={seed} target={target:.2f} exited with "
                  f"code {result.returncode}. Continuing to next.")

    print(f"\n{'='*64}")
    print("  All design-point experiments complete.")
    print(f"  Results in: {args.output_root}/{args.network}/{args.dataset}/")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
