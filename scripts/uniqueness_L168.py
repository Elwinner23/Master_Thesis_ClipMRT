import sys, math, random, itertools, json
from pathlib import Path
sys.path.insert(0, '/media/homes/guseinov/thesis/CoMRT')
from SeqSNN.module.comrt import (
    build_encoding_table, _compute_scores_from_table,
    _build_strides_clipmrt, _pairwise_coprime,
)

def score_v2(strides, bits, L):
    table = build_encoding_table(L, strides, bits)
    base = _compute_scores_from_table(table, L)
    f1, f2, f3, f4 = base['f1_monotonicity'], base['f2_tail_resolution'], base['f3_local_resolution'], base['f4_consistency']
    quality_v2 = (f1**1 * f2**1 * f3**1.5 * f4**1) ** (1.0 / 4.5)
    final_v2 = quality_v2 * base['bit_factor']
    base['quality_score_v2'] = round(quality_v2, 6)
    base['final_score_v2'] = round(final_v2, 6)
    return base


# ---------------- L = 168, random Family-C strides, budget = 15 ----------------
L      = 168
BUDGET = 15
K      = 2000   # random stride draws per bit allocation
SEED   = 42

print(f"=== L={L}, budget={BUDGET}, S∈{{4,5}}, K={K} random draws, ranked by v2 score ===")

rng = random.Random(SEED)
seen = set()
pool = []

for S in (4, 5):
    # Enumerate all bit allocations with sum=BUDGET, each ≥ 2, product ≥ L-1
    allocs = []
    for bits in itertools.product(range(2, BUDGET), repeat=S):
        if sum(bits) != BUDGET:
            continue
        prod = 1
        for b in bits: prod *= b
        if prod >= L - 1:
            allocs.append(bits)

    print(f"  S={S}: {len(allocs)} valid bit allocations")

    for bits in allocs:
        bits_list = list(bits)
        for _ in range(K):
            strides = _build_strides_clipmrt(bits_list, L, rng)
            bits_cur = list(bits_list)
            if strides[-1] * bits_cur[-1] < L - 1:
                bits_cur[-1] = max(bits_cur[-1], math.ceil((L - 1) / max(strides[-1], 1)))
            key = (tuple(strides), tuple(bits_cur))
            if key in seen:
                continue
            seen.add(key)
            d = score_v2(strides, bits_cur, L)
            pool.append((d['final_score_v2'], d['final_score'], tuple(strides), tuple(bits_cur), _pairwise_coprime(strides)))

pool.sort(key=lambda x: x[0])
print(f"\n  Pool: {len(pool)} unique configs  "
      f"score_v2 range [{pool[0][0]:.4f}, {pool[-1][0]:.4f}]")

n = len(pool)
extremes = {
    "min":    pool[0],
    "median": pool[n // 2],
    "max":    pool[-1],
}

print(f"\n{'='*64}")
print(f"  Min / Median / Max  (ranked by v2 score)")
print(f"{'='*64}")
hdr = f"  {'regime':>8}  {'v2':>7}  {'v1':>7}  {'strides':>30}  {'bits':>18}  {'coprime':>7}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for regime, r in extremes.items():
    print(f"  {regime:>8}  {r[0]:>7.4f}  {r[1]:>7.4f}  {str(r[2]):>30}  {str(r[3]):>18}  {str(r[4]):>7}")

out_dir = Path('/media/homes/guseinov/thesis/CoMRT/design_points')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"extremes_v2_L{L}.json"
output = {}
for regime, r in extremes.items():
    output[regime] = {
        "strides": list(r[2]),
        "bits":    list(r[3]),
        "final_score_v2": r[0],
        "final_score":    r[1],
        "coprime":        r[4],
    }
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved → {out_path}")

# where does the deployed config rank now?
deployed = score_v2([      1,
      4,
      12,
      36,
      93], [      4,
      3,
      3,
      3,
      2], L)
print(f"\nDeployed [1,4,12,36,93]/[4,3,3,3,2]:  v2={deployed['final_score_v2']}  v1={deployed['final_score']}")
rank = 1 + sum(1 for r in pool if r[0] > deployed['final_score_v2'])
print(f"Deployed config rank under v2 score: {rank} / {len(pool)}")
