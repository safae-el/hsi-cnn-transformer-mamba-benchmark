"""Ordered list of every survey-revision run, and what is still missing.

  python revision/plan.py status            progress per block and estimated time left
  python revision/plan.py next              arguments of the next missing run (nothing when all done)
  python revision/plan.py next --fit 25     first missing run whose estimated time fits in 25 min

Blocks, in priority order (inside a block: seed by seed, so every session ends on whole seeds):
  A  random + disjoint split, published seeds 42 123 456      54 runs
  B  label scarcity, 5 per class, Indian Pines, 10 seeds       30 runs
  C  random + disjoint split, seeds 1-7                       126 runs
  D  all bands (no PCA), random split, 5 seeds                 45 runs
  E  MambaHSI, random + disjoint split, 10 seeds (optional)    60 runs
A run counts as done when its row is in results/revision/runs.csv (same key as run_revision.py).
Times are estimated from train_s_per_epoch already logged: runs.csv first, then runs_smoke.csv.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

RUNS = Path("results/revision/runs.csv")
SMOKE = Path("results/revision/runs_smoke.csv")
DATASETS = ["indian_pines", "pavia_u", "salinas"]
MODELS = ["HybridSN", "SpectralFormer", "S2Mamba"]
PUBLISHED = [42, 123, 456]
EXTRA = [1, 2, 3, 4, 5, 6, 7]
EPOCHS = 300
DEFAULT_S_PER_EPOCH = {30: 1.0, 0: 7.0}       # used only before any timing is logged


def job(block, ds, model, split, seed, n_train=30, n_pca=30):
    return {"block": block, "dataset": ds, "model": model, "split": split, "seed": seed,
            "n_train": n_train, "n_pca": n_pca}


def all_jobs():
    jobs = []
    for seed in PUBLISHED:
        for split in ["random", "disjoint"]:
            jobs += [job("A", d, m, split, seed) for d in DATASETS for m in MODELS]
    for seed in PUBLISHED + EXTRA:
        jobs += [job("B", "indian_pines", m, "random", seed, n_train=k) for k in (5, 10, 20) for m in MODELS]
    for seed in EXTRA:
        for split in ["random", "disjoint"]:
            jobs += [job("C", d, m, split, seed) for d in DATASETS for m in MODELS]
    for seed in PUBLISHED + [1, 2]:
        jobs += [job("D", d, m, "random", seed, n_pca=0) for d in DATASETS for m in MODELS]
    for split in ["random", "disjoint"]:
        for seed in PUBLISHED + EXTRA:
            jobs += [job("E", d, "MambaHSI", split, seed) for d in DATASETS]
    return jobs


BLOCK_NAMES = {
    "A": "random + disjoint, seeds 42 123 456",
    "B": "5/10/20 labels/class, IP, 10 seeds",
    "C": "random + disjoint, seeds 1-7",
    "D": "all bands, random, 5 seeds",
    "E": "MambaHSI (optional)",
}


def key(j):
    return (j["dataset"], j["model"], j["split"], str(j["seed"]), str(j["n_train"]), str(j["n_pca"]))


def done_keys():
    if not RUNS.exists():
        return set()
    with open(RUNS) as f:
        return {(r["dataset"], r["model"], r["split"], r["seed"], r["n_train_per_class"], r["n_pca"])
                for r in csv.DictReader(f)}


def logged_times():
    """median train_s_per_epoch per (dataset, model, n_pca) at 30 labels/class; runs.csv wins over smoke."""
    out = {}
    for path in [RUNS, SMOKE]:
        if not path.exists():
            continue
        vals = {}
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["n_train_per_class"] != "30":
                    continue
                vals.setdefault((r["dataset"], r["model"], r["n_pca"]), []).append(float(r["train_s_per_epoch"]))
        for k, v in vals.items():
            if k not in out:
                out[k] = sorted(v)[len(v) // 2]
    return out


N_BANDS = {"indian_pines": 200, "pavia_u": 103, "salinas": 204}


def s_per_epoch(j, times):
    ds, m, p = j["dataset"], j["model"], str(j["n_pca"])
    if (ds, m, p) in times:
        return times[(ds, m, p)]
    # all bands not yet timed on this scene: PCA-30 time of this scene x all-band/PCA-30 ratio
    # measured on Indian Pines for the same model, scaled by the number of bands
    ip_all, ip_pca = times.get(("indian_pines", m, "0")), times.get(("indian_pines", m, "30"))
    here_pca = times.get((ds, m, "30"))
    if p == "0" and ip_all and ip_pca and here_pca:
        return here_pca * ip_all / ip_pca * N_BANDS[ds] / N_BANDS["indian_pines"]
    return DEFAULT_S_PER_EPOCH[j["n_pca"]]


def minutes(j, times):
    s = s_per_epoch(j, times)
    if j["n_train"] < 30:
        s *= max(0.25, j["n_train"] / 30)
    return (s * EPOCHS * 1.2 + 45) / 60        # +20% validation/overhead, +45 s loading and testing


def args_of(j):
    a = (f"--datasets {j['dataset']} --models {j['model']} --split {j['split']} "
         f"--seeds {j['seed']} --n_train {j['n_train']} --n_pca {j['n_pca']}")
    return a + (" --micro_batch 16" if j["n_pca"] == 0 else "")


def label(j):
    bands = "all bands" if j["n_pca"] == 0 else "PCA-30"
    return (f"block {j['block']}: {j['dataset']} {j['model']} {j['split']} seed {j['seed']} "
            f"{j['n_train']}/class {bands}")


def status(jobs, done, times):
    print(f"{'block':5s} {'what':40s} {'done':>9s} {'left':>8s}")
    tot_d = tot_n = 0
    tot_left = 0.0
    for b, name in BLOCK_NAMES.items():
        js = [j for j in jobs if j["block"] == b]
        d = sum(key(j) in done for j in js)
        left = sum(minutes(j, times) for j in js if key(j) not in done)
        tot_d, tot_n, tot_left = tot_d + d, tot_n + len(js), tot_left + left
        print(f"{b:5s} {name:40s} {d:4d}/{len(js):<4d} {left / 60:6.1f} h")
    print(f"{'all':5s} {'':40s} {tot_d:4d}/{tot_n:<4d} {tot_left / 60:6.1f} h")
    pending = [j for j in jobs if key(j) not in done]
    if pending:
        print(f"next: {label(pending[0])} (~{minutes(pending[0], times):.0f} min)")
    else:
        print("everything is done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "next"])
    ap.add_argument("--fit", type=float, default=None, help="minutes available")
    a = ap.parse_args()
    jobs, done, times = all_jobs(), done_keys(), logged_times()
    if a.cmd == "status":
        return status(jobs, done, times)
    for j in jobs:
        if key(j) in done:
            continue
        m = minutes(j, times)
        if a.fit is None or m <= a.fit:
            print(f">>> {label(j)} (~{m:.0f} min)", file=sys.stderr)
            print(args_of(j))
            return


if __name__ == "__main__":
    main()
