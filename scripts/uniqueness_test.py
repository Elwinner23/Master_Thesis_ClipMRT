import sys, math, random, itertools, json
from pathlib import Path
sys.path.insert(0, '/media/homes/guseinov/thesis/CoMRT')
from SeqSNN.module.comrt import (
    build_encoding_table, _compute_scores_from_table,
    _enumerate_bit_allocs, _build_strides_clipmrt, _pairwise_coprime,
)

def unique_ratio(table):
    codes = [tuple(r) for r in table.int().tolist()]
    return len(set(codes)) / table.shape[0]

def score_v2(strides, bits, L):
    table = build_encoding_table(L, strides, bits)
    base = _compute_scores_from_table(table, L)
    f1, f2, f3, f4 = base['f1_monotonicity'], base['f2_tail_resolution'], base['f3_local_resolution'], base['f4_consistency']
    quality_v2 = (f1**1 * f2**1 * f3**1.5 * f4**1) ** (1.0 / 4.5)
    final_v2 = quality_v2 * base['bit_factor']
    base['quality_score_v2'] = round(quality_v2, 6)
    base['final_score_v2'] = round(final_v2, 6)
    return base

# ---------------- L = 12, exhaustive, budget = 8 ----------------
L = 12
print("=== L=12, budget=8, exhaustive, ranked by NEW score (v2) ===")
results = []
for S in (2, 3, 4):
    for strides in itertools.combinations(range(1, 12), S):
        for bits in itertools.product(range(1, 5), repeat=S):
            if sum(bits) != 8:
                continue
            if strides[-1] * bits[-1] < L - 1:
                continue
            d = score_v2(list(strides), list(bits), L)
            results.append((d['final_score_v2'], d['final_score'], strides, bits, _pairwise_coprime(list(strides))))
results.sort(key=lambda x: x[0])
print(f"n candidates = {len(results)}")

n = len(results)
extremes = {
    "min":    results[0],
    "median": results[n // 2],
    "max":    results[-1],
}

print(f"\n{'='*64}")
print(f"  Min / Median / Max  (ranked by v2 score)")
print(f"{'='*64}")
hdr = f"  {'regime':>8}  {'v2':>7}  {'v1':>7}  {'strides':>20}  {'bits':>12}  {'coprime':>7}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for regime, r in extremes.items():
    print(f"  {regime:>8}  {r[0]:>7.4f}  {r[1]:>7.4f}  {str(r[2]):>20}  {str(r[3]):>12}  {str(r[4]):>7}")

out_dir = Path('/media/homes/guseinov/thesis/CoMRT/design_points')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"extremes_v2_L{L}.json"
output = {}
for regime, r in extremes.items():
    output[regime] = {
        "strides": list(r[2]),
        "bits": list(r[3]),
        "final_score_v2": r[0],
        "final_score": r[1],
        "coprime": r[4],
    }
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved → {out_path}")

# where does the deployed config rank now?
deployed = score_v2([1,2,5],[2,3,3], L)
print("\nDeployed [1,2,5]/[2,3,3]:", deployed['final_score_v2'], "(old v1 score was", deployed['final_score'], ")")
rank = 1 + sum(1 for r in results if r[0] > deployed['final_score_v2'])
print(f"Deployed config rank under v2 score: {rank} / {len(results)}")