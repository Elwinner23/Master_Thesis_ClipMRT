#!/usr/bin/env python3
"""
Driver: theoretical energy of all forecasting runs listed in a manifest.
=========================================================================

For every row of the manifest (default: scripts/energy_manifest.csv, one row per
dataset / horizon / configuration / seed, pointing to the checkpoint directory that
produced the value in the results table), the trained model is evaluated once on
the test set with scripts/energy_snn.py. One JSON per run is written to
<out_dir>/runs/, so an interrupted job resumes where it stopped.

Afterwards the results are aggregated into
    <out_dir>/energy_runs.csv       one row per run
    <out_dir>/energy_summary.csv    mean and std over seeds per dataset/horizon/config
    <out_dir>/energy_table.csv      Gray, LogPE and ClipMRT_best per dataset/horizon,
                                    where ClipMRT_best is the configuration (hard or
                                    relaxed) with the higher mean test R^2, as in the
                                    results table.
The standard deviation over seeds is the population std (ddof=0), as in the paper CSV.

Usage:
    # all runs on GPUs 3 and 4 (one worker per GPU)
    python scripts/run_energy.py --devices 3 4

    # on the CPU, or only re-aggregate existing results with another convention
    python scripts/run_energy.py --devices cpu
    python scripts/run_energy.py --aggregate_only --pe_mode precomputed --fold_scale

    # quick test: first 4 runs, 5 batches each
    python scripts/run_energy.py --devices 3 --limit 4 --max_batches 5 --out_dir outputs/energy_test
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def read_manifest(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["horizon"] = int(r["horizon"])
        r["seed"] = int(r["seed"])
        r["r2"] = float(r["r2"])
    return rows


def run_name(row):
    return f"{row['dataset']}_H{row['horizon']}_{row['config']}_seed{row['seed']}"


def json_path(out_dir, row):
    return Path(out_dir) / "runs" / (re.sub(r"[^A-Za-z0-9_.-]", "_", run_name(row)) + ".json")


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def work(rows, args, device):
    import torch
    from energy_snn import measure

    os.chdir(REPO_ROOT)
    todo = [r for r in rows if not json_path(args.out_dir, r).exists()]
    print(f"[{device}] {len(todo)} of {len(rows)} runs to do", flush=True)
    t_start = time.time()
    for i, row in enumerate(todo):
        out = json_path(args.out_dir, row)
        try:
            res = measure(row["run_dir"], row["encoding"], device=device,
                          batch_size=args.batch_size, num_workers=args.num_workers,
                          max_batches=args.max_batches)
        except Exception:
            print(f"[{device}] FAILED {run_name(row)}\n{traceback.format_exc()}", flush=True)
            continue
        res["manifest"] = row
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(res))
        tmp.replace(out)
        if args.max_batches is None and abs(res["r2_check"] - res["r2_reported"]) > 1e-3:
            print(f"[{device}] WARNING R2 mismatch for {run_name(row)}: "
                  f"{res['r2_check']:.5f} vs reported {res['r2_reported']:.5f}", flush=True)
        el = time.time() - t_start
        eta = el / (i + 1) * (len(todo) - i - 1)
        print(f"[{device}] {i + 1}/{len(todo)} {run_name(row)}  {res['seconds']:.1f}s  "
              f"(elapsed {el / 60:.1f} min, eta {eta / 60:.1f} min)", flush=True)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(rows, args):
    from energy_snn import energy_from_counts

    per_run = []
    for row in rows:
        p = json_path(args.out_dir, row)
        if not p.exists():
            continue
        res = json.loads(p.read_text())
        e = energy_from_counts(res["counts"], pe_mode=args.pe_mode, fold_scale=args.fold_scale,
                               count_mode=args.count_mode, e_mac=args.e_mac, e_ac=args.e_ac)
        n = res["n_samples"]
        per_run.append({
            "dataset": row["dataset"], "horizon": row["horizon"], "config": row["config"],
            "encoding": row["encoding"], "seed": row["seed"], "run_dir": row["run_dir"],
            "depths": res.get("depths"), "n_samples": n,
            "energy_uJ": e["energy_pj"] / n / 1e6,
            "mac_per_sample": e["mac"] / n, "ac_per_sample": e["ac"] / n,
            "ac_attention_per_sample": e["ac_attention"] / n,
            "lif_firing_rate": res["lif_firing_rate"],
            "r2": row["r2"], "r2_check": res["r2_check"],
            "full_test_set": res.get("full_test_set", True), "note": row.get("note", ""),
        })
    if not per_run:
        print("No results found to aggregate.")
        return
    out_dir = Path(args.out_dir)
    with open(out_dir / "energy_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run[0])); w.writeheader(); w.writerows(per_run)

    groups = defaultdict(list)
    for r in per_run:
        groups[(r["dataset"], r["horizon"], r["config"])].append(r)
    summary = {}
    for key, rs in groups.items():
        def ms(field):
            v = np.array([r[field] for r in rs], dtype=float)
            return float(v.mean()), float(v.std())
        summary[key] = {
            "dataset": key[0], "horizon": key[1], "config": key[2], "n_seeds": len(rs),
            "energy_uJ_mean": ms("energy_uJ")[0], "energy_uJ_std": ms("energy_uJ")[1],
            "firing_rate_mean": ms("lif_firing_rate")[0], "firing_rate_std": ms("lif_firing_rate")[1],
            "ac_per_sample_mean": ms("ac_per_sample")[0], "mac_per_sample_mean": ms("mac_per_sample")[0],
            "r2_mean": ms("r2")[0],
            "note": "; ".join(sorted({r["note"] for r in rs if r["note"]})),
        }
    with open(out_dir / "energy_summary.csv", "w", newline="") as f:
        fields = list(next(iter(summary.values())))
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        w.writerows(sorted(summary.values(), key=lambda s: (s["dataset"], s["horizon"], s["config"])))

    table = []
    for ds, H in sorted({(k[0], k[1]) for k in summary}):
        get = lambda c: summary.get((ds, H, c))
        clip = [c for c in ("hard", "relaxed") if get(c)]
        best = max(clip, key=lambda c: get(c)["r2_mean"]) if clip else None
        row = {"dataset": ds, "horizon": H}
        for label, c in (("Gray", "Gray"), ("LogPE", "LogPE"), ("ClipMRT_best", best)):
            s = get(c) if c else None
            row[f"{label}_uJ_mean"] = s["energy_uJ_mean"] if s else ""
            row[f"{label}_uJ_std"] = s["energy_uJ_std"] if s else ""
            row[f"{label}_firing_rate"] = s["firing_rate_mean"] if s else ""
        row["ClipMRT_best_config"] = best or ""
        table.append(row)
    with open(out_dir / "energy_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)

    print(f"\nEnergy per test sample (uJ), pe_mode={args.pe_mode}, fold_scale={args.fold_scale}, "
          f"count_mode={args.count_mode}, E_MAC={args.e_mac} pJ, E_AC={args.e_ac} pJ")
    print(f"{'dataset':14s}{'H':>4s}{'Gray':>18s}{'LogPE':>18s}{'ClipMRT_best':>18s}  config")
    for r in table:
        cell = lambda k: (f"{r[k + '_uJ_mean']:.1f}+-{r[k + '_uJ_std']:.1f}" if r[k + "_uJ_mean"] != "" else "-")
        print(f"{r['dataset']:14s}{r['horizon']:>4d}{cell('Gray'):>18s}{cell('LogPE'):>18s}"
              f"{cell('ClipMRT_best'):>18s}  {r['ClipMRT_best_config']}")
    missing = len(rows) - len(per_run)
    if missing:
        print(f"\n{missing} runs of the manifest have no result yet.")
    if any(not r["full_test_set"] for r in per_run):
        print("NOTE: some runs used --max_batches; their numbers are not full test-set estimates.")
    print(f"Wrote {out_dir}/energy_runs.csv, energy_summary.csv, energy_table.csv")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(REPO_ROOT / "scripts" / "energy_manifest.csv"))
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "outputs" / "energy"))
    ap.add_argument("--devices", nargs="+", default=["0"],
                    help="CUDA device indices (one worker each) or 'cpu'")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_batches", type=int, default=None, help="for quick tests only")
    ap.add_argument("--limit", type=int, default=None, help="only the first N manifest rows")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="restrict to configs, e.g. --configs Gray LogPE hard relaxed")
    ap.add_argument("--aggregate_only", action="store_true")
    # energy convention (only used for aggregation; the raw counts are convention-free)
    ap.add_argument("--pe_mode", choices=["dynamic", "precomputed"], default="dynamic")
    ap.add_argument("--fold_scale", action="store_true",
                    help="do not count the sigmoid(theta) multiplication of the scores")
    ap.add_argument("--count_mode", choices=["nnz", "sum"], default="nnz",
                    help="integer inputs after residuals: one AC per non-zero (nnz) or per unit (sum)")
    ap.add_argument("--e_mac", type=float, default=4.6, help="pJ per MAC")
    ap.add_argument("--e_ac", type=float, default=0.9, help="pJ per AC")
    # internal
    ap.add_argument("--worker_device", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--shard", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--num_shards", type=int, default=1, help=argparse.SUPPRESS)
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    if args.configs:
        rows = [r for r in rows if r["config"] in args.configs]
    if args.limit:
        rows = rows[:args.limit]
    Path(args.out_dir, "runs").mkdir(parents=True, exist_ok=True)

    if args.worker_device is not None:  # child process
        work(rows[args.shard::args.num_shards], args, args.worker_device)
        return

    if not args.aggregate_only:
        if args.devices == ["cpu"]:
            work(rows, args, "cpu")
        elif len(args.devices) == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.devices[0]
            work(rows, args, "cuda:0")
        else:
            procs = []
            passthrough = [a for a in sys.argv[1:]]
            for i, dev in enumerate(args.devices):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=dev)
                cmd = [sys.executable, __file__] + passthrough + [
                    "--worker_device", "cuda:0", "--shard", str(i), "--num_shards", str(len(args.devices))]
                log = open(Path(args.out_dir) / f"worker_gpu{dev}.log", "a")
                procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log, dev))
                print(f"started worker on GPU {dev} (log: {log.name})")
            for p, log, dev in procs:
                p.wait()
                log.close()
                print(f"worker on GPU {dev} finished with code {p.returncode}")
    aggregate(rows, args)


if __name__ == "__main__":
    main()
