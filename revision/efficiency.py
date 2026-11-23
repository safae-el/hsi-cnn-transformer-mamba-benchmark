"""E5: efficiency of the benchmark models under a written protocol (answers reviewer 2, comment 6).

Protocol (recorded in the CSV and stated in the paper):
  - input: 13x13 patches, 30 PCA components (published setting); models in eval mode, no_grad
  - inference latency: batch sizes 1 and 256; 20 warm-up passes, then 200 timed passes with
    CUDA events; reported as ms per batch and microseconds per pixel
  - peak memory: torch.cuda.max_memory_allocated during one forward pass, after a reset
  - training step: batch 64, AdamW, forward + backward + step; 10 warm-up, 100 timed steps
  - FLOPs: torch.utils.flop_counter (matrix products and convolutions, including attention);
           the custom selective-scan kernel of Mamba is not counted by this counter, so state-space
           FLOPs are a lower bound (stated in the paper); the published hook-based count is also kept
  - nothing else runs on the GPU during the measurement
  python revision/efficiency.py            -> results/revision/efficiency.csv
"""
import csv
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))
os.chdir(ROOT)

import torch
from torch.utils.flop_counter import FlopCounterMode

from run_revision import make_model
from utils.metrics import estimate_flops

N_CLASSES = {"indian_pines": 16, "pavia_u": 9, "salinas": 16}
MODELS = ["HybridSN", "SpectralFormer", "S2Mamba"]
N_BANDS, PATCH = 30, 13
dev = torch.device("cuda")


def time_forward(model, x, warm=20, runs=200):
    with torch.no_grad():
        for _ in range(warm):
            model(x)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(runs):
            model(x)
        e.record()
        torch.cuda.synchronize()
    return s.elapsed_time(e) / runs                     # ms per batch


def peak_mem(model, x):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 20, torch.cuda.max_memory_allocated() / 2 ** 20


def train_step_ms(model, n_classes, warm=10, runs=100):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    x = torch.randn(64, N_BANDS, PATCH, PATCH, device=dev)
    y = torch.randint(0, n_classes, (64,), device=dev)
    loss_fn = torch.nn.CrossEntropyLoss()
    for i in range(warm + runs):
        if i == warm:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    torch.cuda.synchronize()
    model.eval()
    return (time.perf_counter() - t0) / runs * 1000


def flops_counter(model, x):
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x)
        return fc.get_total_flops() / 1e6
    except Exception as err:                            # custom CUDA ops can break the counter
        print("flop counter failed:", err)
        return float("nan")


def env():
    drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    try:
        import mamba_ssm
        mv = mamba_ssm.__version__
    except Exception:
        mv = "n/a"
    return {"gpu": torch.cuda.get_device_name(0), "driver": drv, "torch": torch.__version__,
            "cuda": torch.version.cuda, "mamba_ssm": mv, "python": platform.python_version()}


def main():
    torch.backends.cudnn.benchmark = False
    E = env()
    out = Path("results/revision/efficiency.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, n_cls in N_CLASSES.items():
        for name in MODELS:
            torch.manual_seed(0)
            model = make_model(name, N_BANDS, n_cls).to(dev).eval()
            x1 = torch.randn(1, N_BANDS, PATCH, PATCH, device=dev)
            x256 = torch.randn(256, N_BANDS, PATCH, PATCH, device=dev)
            lat1, lat256 = time_forward(model, x1), time_forward(model, x256)
            m1_act, m1_tot = peak_mem(model, x1)
            m256_act, m256_tot = peak_mem(model, x256)
            row = {"dataset": ds, "model": name,
                   "params_K": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e3,
                   "flops_M_counter": flops_counter(model, x1),
                   "flops_M_hooks_published": estimate_flops(model, x1),
                   "latency_ms_b1": lat1, "latency_ms_b256": lat256,
                   "us_per_pixel_b256": lat256 / 256 * 1000,
                   "peak_MB_b1_total": m1_tot, "peak_MB_b1_activation": m1_act,
                   "peak_MB_b256_total": m256_tot, "peak_MB_b256_activation": m256_act,
                   "train_step_ms_b64": train_step_ms(model, n_cls), **E}
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
            del model
            torch.cuda.empty_cache()
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("written", out)


if __name__ == "__main__":
    main()
