#!/usr/bin/env bash
# Theoretical energy of all forecasting models in scripts/energy_manifest.csv
# (Gray, LogPE, Hard- and Relaxed-ClipMRT; 16 settings x 3 seeds).
#
# Usage:
#   bash scripts/run_energy.sh                 # GPUs 3 and 4
#   bash scripts/run_energy.sh 3 4 5           # any list of GPU indices
#   bash scripts/run_energy.sh cpu             # CPU only
#
# The measurement is resumable: finished runs are stored in outputs/energy/runs/
# and skipped when the script is started again. At the end, the summary tables
# are written for two counting conventions:
#   outputs/energy/dynamic/                PE bits counted inside q.k^T,
#                                          scale sigma(theta) counted as MAC
#   outputs/energy/precomputed_folded/     PE-PE similarity as a fixed matrix,
#                                          scale folded into the neuron threshold
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DEVICES=("$@")
[ ${#DEVICES[@]} -eq 0 ] && DEVICES=(3 4)

PY="${PYTHON:-$HOME/anaconda3/envs/SeqSNN/bin/python}"
OUT="outputs/energy"
mkdir -p "$OUT"

echo "== measuring on: ${DEVICES[*]}  (python: $PY)"
"$PY" -W ignore scripts/run_energy.py --devices "${DEVICES[@]}" --out_dir "$OUT"

for conv in dynamic precomputed_folded; do
  if [ "$conv" = dynamic ]; then
    FLAGS=(--pe_mode dynamic)
  else
    FLAGS=(--pe_mode precomputed --fold_scale)
  fi
  echo
  echo "== aggregating: $conv"
  "$PY" -W ignore scripts/run_energy.py --aggregate_only --out_dir "$OUT" "${FLAGS[@]}"
  mkdir -p "$OUT/$conv"
  mv "$OUT"/energy_runs.csv "$OUT"/energy_summary.csv "$OUT"/energy_table.csv "$OUT/$conv/"
done

echo
echo "== R2 mismatches (checkpoint does not reproduce its res.json):"
grep -h "WARNING R2 mismatch" "$OUT"/worker_gpu*.log 2>/dev/null || echo "   none reported in worker logs"
echo "Results: $OUT/dynamic/ and $OUT/precomputed_folded/"
