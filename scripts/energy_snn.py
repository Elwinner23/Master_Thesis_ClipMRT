#!/usr/bin/env python3
"""
Theoretical energy estimate of a trained SeqSNN Spikformer forecasting model.
==============================================================================

Follows the estimate used by Spikformer/QKFormer (Horowitz 2014; Yao et al. 2023b):

    E = E_MAC * #MAC + E_AC * #AC,       E_MAC = 4.6 pJ, E_AC = 0.9 pJ (45 nm)

A layer whose input is real-valued is counted with MAC operations (dense FLOPs).
A layer whose input is a spike (or an integer spike count, which happens after the
residual additions) is counted with AC operations, one per non-zero input element
and per output it feeds, i.e. SOPs = firing_rate * T * FLOPs.

What is counted (per forward pass, summed over the test set):

  Linear / Conv layers        hooks on every nn.Linear, nn.Conv1d, nn.Conv2d,
                              including the runner head fc_out. Real input -> MAC,
                              integer input -> AC. BatchNorm is assumed folded into
                              the preceding weights and is not counted.
  XNOR attention              counted from the spike counts of q, k and v
                              (hooks on q_lif, k_lif, v_lif of every SSA block):
      qk_content   L * nnz(q)                     q.k^T over the content bits
      qk_pe        L * T*B*H * P                  q.k^T over the appended PE bits,
                                                  P = number of ones in the PE table
      pe_matrix    T*B*H * L^2                    alternative: the PE-PE part of the
                                                  similarity is a fixed L x L matrix
                                                  that is precomputed and added once
      sums         nnz(q) + nnz(k) (+ PE ones)    sum_q and sum_k of the XNOR formula
      combine      3 * T*B*H * L^2                D - sum_q - sum_k + 2 q.k
      logpe_bias   T*B*H * L^2                    Log-PE: bias added to every score
      scale        T*B*H * L^2   (MAC)            multiplication by sigmoid(theta);
                                                  can be folded into the threshold
      attn_v       L * nnz(v)                     scores @ v with spike v
  LIF neurons                 not counted (as in the reference papers).

The two PE conventions ("dynamic": the PE bits are part of q and k at run time,
"precomputed": the PE-PE similarity is a fixed matrix) and folding of the scale are
selected when the energy is computed from the stored counts (see `energy_from_counts`),
so one measurement serves all conventions.

Usage (single run):
    python scripts/energy_snn.py \\
        --run_dir outputs/Electricity6Gray/our_setup/spikformer_xnor_gray_electricity_seed1 \\
        --encoding Gray --device cuda:0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

E_MAC_PJ = 4.6
E_AC_PJ = 0.9

NETWORK_CLASSES = {
    "Gray": ("SeqSNN.network.snn.spikformer_xnor_gray", "Spikformer_XNOR_Gray"),
    "LogPE": ("SeqSNN.network.snn.spikformer_xnor_log", "Spikformer_XNOR_Log"),
    "ClipMRT": ("SeqSNN.network.snn.spikformer_comrt", "SpikformerCoMRT"),
}
GRAY_BITS = 10  # hard-coded in spikformer_xnor_gray.SSA.forward


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _import_class(encoding):
    import importlib
    mod_name, cls_name = NETWORK_CLASSES[encoding]
    return getattr(importlib.import_module(mod_name), cls_name)


def load_run(run_dir, encoding, device):
    """Rebuild network + output head of a finished run and load its best weights."""
    from SeqSNN.dataset.tsforecast import TSMSDataset

    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    data_cfg = dict(cfg["data"])
    data_cfg["dataset_name"] = "test"
    testset = TSMSDataset(**data_cfg)

    net_cfg = dict(cfg["network"])
    net_cfg["input_size"] = testset.num_variables
    net_cfg["max_length"] = testset.max_seq_len
    if encoding == "ClipMRT" and not net_cfg.get("strides"):
        comrt = json.loads((run_dir / "comrt_config.json").read_text())
        net_cfg["strides"], net_cfg["bits"] = comrt["strides"], comrt["bits"]
    network = _import_class(encoding)(**net_cfg)

    ckpt = run_dir / "checkpoints"
    network.load_state_dict(torch.load(ckpt / "network_best.pkl", map_location="cpu", weights_only=True), strict=True)
    full = torch.load(ckpt / "model_best.pkl", map_location="cpu", weights_only=True)
    w, b = full["fc_out.weight"], full["fc_out.bias"]
    fc_out = nn.Linear(w.shape[1], w.shape[0])
    fc_out.load_state_dict({"weight": w, "bias": b})

    # Some run directories have a config.json whose horizon does not match the trained
    # head (the checkpoint is authoritative). Rebuild the test set with the horizon the
    # head was trained for, so that predictions and labels have the same shape.
    if not data_cfg.get("last_label") and w.shape[0] != testset.num_classes:
        n_var = testset.raw_data.shape[1]
        if w.shape[0] % n_var == 0:
            fixed = w.shape[0] // n_var
            print(f"WARNING {run_dir}: config horizon {data_cfg['horizon']} does not match the "
                  f"checkpoint head; using horizon {fixed} from the checkpoint", flush=True)
            data_cfg["horizon"] = fixed
            cfg["data"]["horizon"] = fixed
            testset = TSMSDataset(**data_cfg)

    network.to(device).eval()
    fc_out.to(device).eval()
    aggregate = cfg["runner"].get("aggregate", True)
    reported = json.loads((ckpt / "res.json").read_text())["test"]
    return network, fc_out, aggregate, testset, cfg, reported


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #
def _pe_ones(network, encoding, L):
    """Number of ones in the PE table over L positions, and number of PE bits."""
    if encoding == "Gray":
        idx = torch.arange(L)
        gray = (idx ^ (idx >> 1)) & ((1 << GRAY_BITS) - 1)
        ones = sum(bin(int(g)).count("1") for g in gray)
        return ones, GRAY_BITS
    if encoding == "ClipMRT":
        table = network.encoding_table[:L]
        return int(table.sum().item()), int(table.shape[1])
    return 0, 0


class Counter:
    """Registers hooks and accumulates operation counts on the device."""

    def __init__(self, network, fc_out, encoding):
        self.encoding = encoding
        self.network = network
        self.handles = []
        self.t = {}            # tensor accumulators (avoid GPU syncs per layer)
        self.py = {}           # python-int accumulators (analytic counts)
        self.layers = {}       # per-layer breakdown
        self._pe_cache = {}

        for name, m in list(network.named_modules()) + [("fc_out", fc_out)]:
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                self.handles.append(m.register_forward_hook(self._layer_hook(name)))
        for name, m in network.named_modules():
            if all(hasattr(m, a) for a in ("q_lif", "k_lif", "v_lif", "heads")):
                self.handles.append(m.q_lif.register_forward_hook(self._q_hook(m)))
                self.handles.append(m.k_lif.register_forward_hook(self._kv_hook("k_nnz")))
                self.handles.append(m.v_lif.register_forward_hook(self._kv_hook("v_nnz_L")))
            if type(m).__name__ == "LIFNode":
                self.handles.append(m.register_forward_hook(self._lif_hook))

    # -- helpers --
    def _add_t(self, key, val):
        val = val.double()
        self.t[key] = self.t[key] + val if key in self.t else val

    def _add_py(self, key, val):
        self.py[key] = self.py.get(key, 0) + int(val)

    def _layer_hook(self, name):
        def hook(m, inp, out):
            x = inp[0]
            if isinstance(m, nn.Linear):
                fan_out = m.out_features
                dense = x.numel() * m.out_features
            else:
                k = int(np.prod(m.kernel_size))
                fan_out = m.out_channels * k // m.groups
                dense = out.numel() * (m.in_channels // m.groups) * k
            is_spike = bool(torch.equal(x, torch.round(x))) and bool((x >= 0).all())
            rec = self.layers.setdefault(name, {"type": type(m).__name__, "spike_input": is_spike,
                                                "dense_flops": 0})
            rec["dense_flops"] += dense
            if is_spike:
                self._add_t(f"L:{name}:nnz", (x != 0).sum() * fan_out)
                self._add_t(f"L:{name}:sum", x.sum() * fan_out)
                self._add_t(f"L:{name}:in_nnz", (x != 0).sum())
                self._add_py(f"L:{name}:in_numel", x.numel())
            else:
                self._add_py(f"L:{name}:mac", dense)
        return hook

    def _q_hook(self, ssa):
        def hook(m, inp, out):
            T, B, L, D = out.shape
            H = ssa.heads
            if L not in self._pe_cache:
                self._pe_cache[L] = _pe_ones(self.network, self.encoding, L)
            P, nbits = self._pe_cache[L]
            nnz_q = (out != 0).sum()
            tbh = T * B * H
            self._add_t("A:qk_content", nnz_q * L)
            self._add_t("A:sum_content", nnz_q)
            self._add_py("A:qk_pe", L * tbh * P)          # dynamic convention
            self._add_py("A:sum_pe", 2 * tbh * P)          # PE ones in sum_q and sum_k
            self._add_py("A:pe_matrix", tbh * L * L if nbits > 0 else 0)  # precomputed
            self._add_py("A:combine", 3 * tbh * L * L)
            self._add_py("A:logpe_bias", tbh * L * L if self.encoding == "LogPE" else 0)
            self._add_py("A:scale_mac", tbh * L * L)
            self._L = L
        return hook

    def _kv_hook(self, key):
        def hook(m, inp, out):
            nnz = (out != 0).sum()
            if key == "k_nnz":
                self._add_t("A:sum_content", nnz)
            else:  # scores @ v: every spike of v accumulates one score per query
                self._add_t("A:attn_v", nnz * out.shape[2])
        return hook

    def _lif_hook(self, m, inp, out):
        self._add_t("S:spikes", out.sum())
        self._add_py("S:neurons", out.numel())

    def remove(self):
        for h in self.handles:
            h.remove()

    def results(self):
        res = dict(self.py)
        res.update({k: float(v.item()) for k, v in self.t.items()})
        return res


# --------------------------------------------------------------------------- #
# Energy
# --------------------------------------------------------------------------- #
def energy_from_counts(c, pe_mode="dynamic", fold_scale=False, count_mode="nnz",
                       e_mac=E_MAC_PJ, e_ac=E_AC_PJ):
    """Total MAC/AC counts and energy (pJ) from the raw counts of one run."""
    mac = sum(v for k, v in c.items() if k.startswith("L:") and k.endswith(":mac"))
    ac_layers = sum(v for k, v in c.items() if k.startswith("L:") and k.endswith(f":{count_mode}"))
    ac_attn = c.get("A:qk_content", 0) + c.get("A:sum_content", 0) + c.get("A:combine", 0) \
        + c.get("A:logpe_bias", 0) + c.get("A:attn_v", 0)
    if pe_mode == "dynamic":
        ac_attn += c.get("A:qk_pe", 0) + c.get("A:sum_pe", 0)
    elif pe_mode == "precomputed":
        ac_attn += c.get("A:pe_matrix", 0)
    else:
        raise ValueError(pe_mode)
    if not fold_scale:
        mac += c.get("A:scale_mac", 0)
    ac = ac_layers + ac_attn
    return {"mac": mac, "ac": ac, "ac_layers": ac_layers, "ac_attention": ac_attn,
            "energy_pj": e_mac * mac + e_ac * ac}


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure(run_dir, encoding, device="cuda:0", batch_size=64, num_workers=4, max_batches=None):
    device = torch.device(device)
    if device.type == "cpu":
        # spikformer_xnor_log calls .cuda() on its bias matrix; make that a no-op on CPU.
        torch.Tensor.cuda = lambda self, *a, **k: self
    from SeqSNN.common.function import r2 as r2_fn

    t0 = time.time()
    network, fc_out, aggregate, testset, cfg, reported = load_run(run_dir, encoding, device)
    counter = Counter(network, fc_out, encoding)
    loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    preds, labels, n = [], [], 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        seq_out, emb = network(x)
        pred = fc_out(emb if aggregate else seq_out).squeeze(-1)
        preds.append(pred.float().cpu())
        labels.append(y)
        n += x.shape[0]
    counts = counter.results()
    counter.remove()

    pred_all, label_all = torch.cat(preds), torch.cat(labels)
    try:
        r2_check = float(r2_fn(label_all, pred_all))
    except Exception:  # pragma: no cover - fall back to a plain R^2
        y = label_all.numpy().ravel(); p = pred_all.numpy().ravel()
        r2_check = float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    lif_rate = counts.get("S:spikes", 0) / max(counts.get("S:neurons", 1), 1)
    layer_rates = {}
    for name, rec in counter.layers.items():
        if rec["spike_input"]:
            numel = counts.get(f"L:{name}:in_numel", 0)
            layer_rates[name] = counts.get(f"L:{name}:in_nnz", 0) / numel if numel else 0.0

    return {
        "run_dir": str(run_dir),
        "encoding": encoding,
        "n_samples": n,
        "window": cfg["data"]["window"],
        "horizon": cfg["data"]["horizon"],
        "depths": cfg["network"].get("depths"),
        "num_steps": cfg["network"].get("num_steps"),
        "counts": counts,
        "layers": counter.layers,
        "input_firing_rates": layer_rates,
        "lif_firing_rate": lif_rate,
        "r2_check": r2_check,
        "r2_reported": reported.get("r2"),
        "full_test_set": max_batches is None,
        "seconds": time.time() - t0,
        "device": str(device),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--encoding", required=True, choices=list(NETWORK_CLASSES))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_batches", type=int, default=None, help="for quick tests only")
    ap.add_argument("--out", default=None, help="write the result JSON here")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)  # data paths in config.json are relative to the repo root
    res = measure(args.run_dir, args.encoding, args.device, args.batch_size,
                  args.num_workers, args.max_batches)
    for mode in ("dynamic", "precomputed"):
        e = energy_from_counts(res["counts"], pe_mode=mode)
        print(f"{mode:12s} energy/sample = {e['energy_pj'] / res['n_samples'] / 1e6:.4f} uJ  "
              f"(MAC {e['mac'] / res['n_samples']:.3e}, AC {e['ac'] / res['n_samples']:.3e})")
    print(f"LIF firing rate {res['lif_firing_rate']:.4f}   R2 check {res['r2_check']:.5f} "
          f"(reported {res['r2_reported']:.5f})   {res['seconds']:.1f}s")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
