# ClipMRT: Multi-Resolution Thermometer Positional Encoding for Spiking Transformers

This repository holds the code for my Master's thesis. It adds **ClipMRT** (Clip-mode Multi-Resolution Thermometer encoding) to spiking Transformers. ClipMRT is a binary positional encoding built for XNOR-style spiking self-attention. It is tested on multivariate time-series forecasting and on image classification (CIFAR-10/100).

The code extends [SeqSNN](https://github.com/microsoft/SeqSNN) (Microsoft) for the forecasting experiments and [Spikformer](https://github.com/ZK-Zhou/spikformer) for the image experiments.

---

## Method in short

### Why a binary positional encoding?

Spiking Transformers only exchange binary spikes, so the usual sinusoidal or learned positional embeddings don't fit naturally. Previous work on relative positional encoding for spiking Transformers (Gray-code PE and Log-PE, [NeurIPS 2025](https://arxiv.org/abs/2501.16745)) adds binary position codes to the queries and keys and scores attention with an **XNOR similarity**:

$$
\mathrm{sim}(q_i, k_j) \;=\; D - \textstyle\sum q_i - \sum k_j + 2\, q_i^\top k_j ,
$$

which counts the bits on which $q_i$ and $k_j$ agree. The positional part of the score is therefore the number of matching bits between the codes of positions $i$ and $j$. A good encoding should make that number fall smoothly and consistently as $|i-j|$ grows.

### ClipMRT encoding

ClipMRT describes a position $i \in \{0,\dots,L-1\}$ at $S$ scales. Each scale $s$ has a stride $m_s$ and a bit width $B_s$:

$$
P(i) = \big[\,\tau_{B_1}(f_1(i)) \,\|\, \dots \,\|\, \tau_{B_S}(f_S(i))\,\big],
\qquad f_s(i) = \min\!\big(\lfloor i/m_s \rfloor,\; B_s\big),
\qquad \tau_B(t)_k = \mathbb{1}[k \le t].
$$

- $\tau_B$ is a **thermometer code**: the first $t$ of the $B$ bits are 1. For two thermometer codes, the number of disagreeing bits is exactly $|t_i - t_j|$. Under XNOR attention, each scale therefore contributes a similarity that decreases linearly with distance.
- $f_s$ is a **clip response**. Scale $s$ counts in steps of $m_s$ and saturates after $B_s$ steps. Fine scales (small $m_s$) separate nearby positions, and coarse scales (large $m_s$) separate distant ones.
- Constraints for a valid ("hard") configuration:
  - $m_1 = 1$, so neighbouring positions are distinguishable;
  - cascade: $m_{s+1} \le m_s B_s$, so each scale picks up before the previous one saturates and there are no gaps;
  - coverage: $m_S B_S \ge L-1$, so the coarsest scale reaches the end of the sequence.

  The **relaxed** variant drops the cascade constraint.

The codes are a fixed lookup table of shape `(L, ΣB_s)`. In every attention block they are appended to the spiking Q and K before the XNOR similarity is computed. There are no extra learnable parameters.

### Diagnostic score: choosing strides and bits

Different `(strides, bits)` choices give very different similarity-vs-distance curves. To pick one without training, the thesis defines a **diagnostic score** on the mean similarity $\mu(\delta)$ at lag $\delta$:

| term | measures |
|---|---|
| $f_1$ monotonicity | how often a closer pair is at least as similar as a farther pair |
| $f_2$ tail resolution | whether $\mu(\delta)$ still separates distances in the far half of the sequence |
| $f_3$ local resolution | whether there are flat regions (10th-percentile drop vs. median drop of $\mu$) |
| $f_4$ consistency | spread of similarity among pairs with the same lag (normalised std) |

$$
Q = \big(f_1\, f_2\, f_3^{1.5}\, f_4\big)^{1/4.5}, \qquad
\text{score} = Q \cdot \min\!\Big(1,\; \tfrac{b_\text{ref}}{\sum_s B_s}\Big), \qquad
b_\text{ref} = \lceil e \ln(L-1) \rceil + 1 .
$$

The bit factor penalises configurations that use more bits than the reference budget $b_\text{ref}$.

**Tuning** (`tune_comrt`): set $S \approx \ln(L-1)$, enumerate bit allocations with $\prod B_s \ge L-1$ inside the budget, draw random cascade-respecting strides ("Family C", no coprimality required), and keep the configuration with the highest score. The scripts in [scripts/](scripts/) also build pools of configurations at fixed budget. These are used to check that a higher diagnostic score goes with better downstream accuracy (min / median / max / percentile / target-score design points).

The same score can be computed for Gray code and Log-PE, so all encodings can be compared on one scale.

### Where it is in the code

| file | content |
|---|---|
| [SeqSNN/module/comrt.py](SeqSNN/module/comrt.py) | encoding table, diagnostic score (ClipMRT, Gray, Log-PE), tuning, design-point loading |
| [SeqSNN/network/snn/spikformer_comrt.py](SeqSNN/network/snn/spikformer_comrt.py) | `SpikformerCoMRT`: Spikformer + ClipMRT in XNOR attention |
| [SeqSNN/network/snn/spikingformer_comrt.py](SeqSNN/network/snn/spikingformer_comrt.py) | `SpikingformerCoMRT` |
| `SeqSNN/network/snn/*_xnor_gray.py`, `*_xnor_log.py` | Gray-code and Log-PE baselines (Spikformer, Spikingformer, QKFormer) |
| [image/spikformer_clipmrt_2d.py](image/spikformer_clipmrt_2d.py) | 2D ClipMRT for images: x and y axes encoded separately, codes concatenated |
| [image/spikformer_clipmrt_1d.py](image/spikformer_clipmrt_1d.py) | 1D ClipMRT over the flattened patch sequence |
| [scripts/](scripts/) | diagnostics, design-point generation, multi-seed runners, energy estimation |
| [design_points/](design_points/) | precomputed `(strides, bits)` configurations (JSON) |

> The network classes are still named `CoMRT` for historical reasons (`CoMRTPE` is an alias of `ClipMRTPE`). All of them implement ClipMRT.

---

## Installation

```bash
conda create -n SeqSNN python=3.10
conda activate SeqSNN
git clone <this-repo-url>
cd Master_Thesis_ClipMRT
pip install -e .
```

The image experiments also need `timm` (a version that still provides `timm.models.layers` and `timm.models.registry`) and `pyyaml`:

```bash
pip install timm pyyaml
```

---

## Time-series forecasting

### Datasets

These are the same datasets as in SeqSNN:

- **METR-LA**, **PEMS-BAY**: [Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) / [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g)
- **Solar**, **Electricity**: [laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data)

```
data/
├── metr-la.h5
├── pems-bay.h5
├── solar-energy/solar_AL.txt
└── electricity/electricity.txt
```

Data paths are set in `exp/forecast/dataset/<name>.yml`.

In the thesis, Electricity and Solar use a look-back window of $L=168$, and METR-LA and PEMS-BAY use $L=12$. Horizons are 6/24/48/96 and seeds are 1, 42 and 101.

### Train a single model

```bash
# ClipMRT with a fixed design point
python -m SeqSNN.entry.tsforecast exp/forecast/spikformer/spikformer_clipmrt_solar_score0p9.yml

# Baselines
python -m SeqSNN.entry.tsforecast exp/forecast/spikformer/spikformer_xnor_gray_solar.yml   # Gray-code PE
python -m SeqSNN.entry.tsforecast exp/forecast/spikformer/spikformer_xnor_log_solar.yml    # Log-PE
```

The relevant part of a ClipMRT config:

```yaml
network:
  type: SpikformerCoMRT        # or SpikingformerCoMRT
  dim: 256
  d_ff: 1024
  depths: 2
  num_steps: 4
  heads: 8
  max_length: 168              # = data.window
  strides: [1, 3, 9, 26, 59]   # omit strides/bits to auto-tune with tune_comrt
  bits:    [3, 3, 3, 3, 3]
```

If `strides` and `bits` are left out, the model runs the tuning procedure when it is built and prints the chosen configuration and its score.

### Run design points / several seeds

```bash
# One run per design point in a JSON file, for several seeds (use --dry_run to only print the commands)
python scripts/run_design_points.py \
    --summary design_points/design_points_max_L168.json \
    --network SpikformerCoMRT --dataset electricity \
    --seeds 1 42 101 --output_root outputs/design_points/electricity

# Gray / Log-PE baselines across seeds
python scripts/run_pe_seeds.py \
    --config exp/forecast/spikformer/spikformer_xnor_gray_electricity.yml \
    --seeds 1 42 101
```

---

## Image classification (CIFAR-10 / CIFAR-100)

The image code is in [image/](image/) and is based on the Spikformer CIFAR training script ([image/train.py](image/train.py), timm-based). Run it from inside `image/`, because the model files are imported as local modules.

With 32×32 images and patch size 4, the token grid is 8×8:

- the **2D** model encodes each axis with $L=8$;
- the **1D** model encodes the flattened sequence with $L=64$.

CIFAR is downloaded automatically to `--data-dir`. The `data_dir` in the YAML files points to the original server path, so override it on the command line.

```bash
cd image

# ClipMRT 2D, CIFAR-10, design point loaded from JSON
python train.py -c cifar10.yml --model spikformer_clipmrt_2d \
    --clipmrt-design-point ../design_points/clipmrt_exhaustive_L8_seed0.json \
    --data-dir ./data/cifar10 --device 0 --output output/train

# ClipMRT 1D over the flattened patch sequence
python train.py -c cifar10.yml --model spikformer_clipmrt_1d \
    --clipmrt-design-point ../design_points/clipmrt_exhaustive_L64_seed0.json \
    --data-dir ./data/cifar10 --device 0

# Explicit strides/bits (take priority over --clipmrt-design-point)
python train.py -c cifar10.yml --model spikformer_clipmrt_2d \
    --clipmrt-strides 1 3 --clipmrt-bits 3 3 --data-dir ./data/cifar10

# CIFAR-100
python train.py -c cifar100/cifar100.yml --model spikformer_clipmrt_2d \
    --clipmrt-design-point ../design_points/clipmrt_exhaustive_L8_seed0.json \
    --data-dir ./data/cifar100
```

ClipMRT options (read only when the model name contains `clipmrt`):

| flag | meaning |
|---|---|
| `--clipmrt-design-point PATH` | design-point JSON: flat (`{"strides":…, "bits":…}`) or keyed by score (`{"0.9": {...}}`) |
| `--clipmrt-design-point-key KEY` | which entry to use from a keyed file |
| `--clipmrt-strides`, `--clipmrt-bits` | explicit configuration (per axis for 2D) |
| `--clipmrt-tune-axis-len N`, `--clipmrt-tune-seed S` | tune on the fly when nothing else is given (default axis length `img_size // 4`) |

Baseline models available through `--model`:

| model | encoding |
|---|---|
| `spikformer_not_xor_gray_2d` | 2D Gray-code PE, XNOR attention |
| `spikformer_not_xor_gray` | 1D Gray-code PE |
| `spikformer_log_2d`, `spikformer_log_1d` | Log-PE |
| `spikformer_not_xor` | XNOR attention without PE |
| `QKFormer_Gray_2D` | QKFormer with 2D Gray-code PE |

---

## Analysis scripts

```bash
# Tune ClipMRT for one or more sequence lengths, or score a fixed configuration
python scripts/comrt_diagnostic.py --L 12 168 256 1000
python scripts/comrt_diagnostic.py --L 168 --strides 1 4 12 36 93 --bits 4 3 3 3 2

# Diagnostic scores of Gray code and Log-PE for comparison
python scripts/gray_code_diagnostic.py --L 168

# Exhaustive search over S and all bit partitions
python scripts/tune_comrt_exhaustive.py --L 12 --K 1000 --seed 42

# Stride-family ablation: A (coprime) vs. B (tight cascade) vs. C (random)
python scripts/comrt_ablation_families.py --L 12 168 256 1000

# Pools of configurations at fixed S and bit budget -> min / median / max, percentiles, target scores
python scripts/clipmrt_extremes.py --K 100 --seed 42
python scripts/clipmrt_percentiles.py --pool design_points/minmaxmedian_pool/pool_L168_S5_bits15_seed42.json
python scripts/clipmrt_design_points.py --K 60 --seed 42

# Theoretical energy (E_MAC = 4.6 pJ, E_AC = 0.9 pJ) of trained models listed in scripts/energy_manifest.csv
bash scripts/run_energy.sh 0 1      # GPU indices, or "cpu"
```

The outputs of these scripts are in [design_points/](design_points/), [diagnostics/](diagnostics/) and [ablation_families/](ablation_families/).

---

## Acknowledgements

- [SeqSNN](https://github.com/microsoft/SeqSNN): time-series SNN framework this code builds on. Related papers:
  - *Efficient and Effective Time-Series Forecasting with Spiking Neural Networks*, ICML 2024 ([arXiv](https://arxiv.org/abs/2402.01533))
  - *Advancing Spiking Neural Networks for Sequential Modeling with Central Pattern Generators*, NeurIPS 2024 ([arXiv](https://arxiv.org/abs/2405.14362))
  - *Toward Relative Positional Encoding in Spiking Transformers*, NeurIPS 2025 ([arXiv](https://arxiv.org/abs/2501.16745)): Gray-code PE, Log-PE and XNOR attention, the baselines in this thesis
- [forecaster](https://github.com/Arthur-Null/SRD): forecasting library that SeqSNN is built on
- [Spikformer](https://github.com/ZK-Zhou/spikformer) (ICLR 2023): image training pipeline
- [SpikingJelly](https://github.com/fangwei123456/spikingjelly), [timm](https://github.com/huggingface/pytorch-image-models)

