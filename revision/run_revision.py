"""Survey revision runs (IEEE Access resubmission, Access-2026-50490).

Same pipeline as run_benchmark.py (PCA fitted on training pixels, z-score, 13x13 patches, AdamW 5e-4,
300 epochs, cosine schedule, best-validation checkpoint), with four switches added:
  --split random|disjoint    published random few-shot split, or its spatially disjoint version
  --seeds ...                any number of seeds (published: 42 123 456)
  --n_pca 30|0               PCA-30 (published) or all bands (0)
  --n_train 30|5             labelled pixels per class (30 main table, 5 scarcity table)
  --micro_batch N             full-band runs only: split each batch of 64 into chunks of N and
                              accumulate gradients (same update as batch 64 for the LayerNorm models
                              SpectralFormer, S2Mamba and MambaHSI; HybridSN, which has BatchNorm, is
                              never chunked)
Every run appends one row to results/revision/runs.csv and saves the test predictions to
results/revision/preds/. Runs already in runs.csv are skipped, so a stopped queue can be restarted.

  python revision/run_revision.py --check_splits                    # split statistics, no training
  python revision/run_revision.py --split random   --seeds 42 123 456 1 2 3 4 5 6 7
  python revision/run_revision.py --split disjoint --seeds 42 123 456 1 2 3 4 5 6 7
  python revision/run_revision.py --split random --n_pca 0 --micro_batch 16 --seeds 42 123 456 1 2
  python revision/run_revision.py --split random --n_train 5 --datasets indian_pines
  python revision/run_revision.py --models MambaHSI --split disjoint --seeds 42 123 456 1 2 3 4 5 6 7
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))
os.chdir(ROOT)                      # data paths in utils/data_loader.py are relative to the benchmark root

import numpy as np
from revision_splits import (DISJOINT_CFG, disjoint_fewshot_split, min_distance,
                             random_fewshot_split)

DATASETS = ["indian_pines", "pavia_u", "salinas"]
MODELS = ["HybridSN", "SpectralFormer", "S2Mamba"]
TRAIN = {"epochs": 300, "lr": 5e-4, "weight_decay": 1e-4, "patience": 300, "batch_size": 64}
PATCH = 13

FIELDS = ["finished", "dataset", "model", "split", "seed", "n_train_per_class", "n_pca", "patch",
          "block", "guard", "split_tries", "n_train", "n_val", "n_test", "n_classes", "n_test_classes",
          "OA", "AA", "Kappa", "MacroF1", "mIoU", "PerClassIoU", "PerClassF1",
          "best_val_OA", "epochs_trained", "train_s_per_epoch", "params_K", "micro_batch", "code_md5", "torch"]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random", "disjoint"], default="random")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--models", nargs="+", default=MODELS,
                    choices=MODELS + ["MambaHSI"])
    ap.add_argument("--n_pca", type=int, default=30, help="0 = all bands")
    ap.add_argument("--n_train", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    ap.add_argument("--out", default="results/revision/runs.csv")
    ap.add_argument("--micro_batch", type=int, default=0,
                    help="gradient accumulation chunk size for LayerNorm models (0 = off)")
    ap.add_argument("--check_splits", action="store_true")
    return ap.parse_args()


def code_md5():
    h = hashlib.md5()
    for f in [ROOT / "revision" / "run_revision.py", ROOT / "revision" / "revision_splits.py",
              ROOT / "utils" / "data_loader.py", ROOT / "utils" / "trainer.py"]:
        h.update(f.read_bytes())
    return h.hexdigest()[:10]


def split_for(dataset, gt, split, n_train, seed):
    if split == "random":
        tr, te = random_fewshot_split(gt, n_train, seed)
        tr, te = np.asarray(tr), np.asarray(te)
        info = {"block": "", "guard": "", "tries": "",
                "n_test_classes": int(len(np.unique(gt.ravel()[te])))}
        return tr, te, info
    cfg = DISJOINT_CFG[dataset]
    tr, te, info = disjoint_fewshot_split(gt, n_train, seed, block=cfg["block"],
                                          train_frac=cfg["train_frac"], guard=PATCH // 2)
    return tr, te, info


def build_loaders(dataset, split, seed, n_train, n_pca):
    """run_benchmark.get_loaders with the split injected; everything else is copied unchanged."""
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    from utils.data_loader import (DATASETS as DCFG, HSIDataset, apply_pca, extract_patches,
                                   load_dataset, normalize)
    data, gt = load_dataset(dataset)
    flat_gt = gt.flatten()
    train_idx, test_idx, info = split_for(dataset, gt, split, n_train, seed)

    # validation carved from the training pixels, exactly as in get_loaders
    lab = flat_gt[train_idx]
    n_cls = len(np.unique(lab))
    val_size = max(n_cls, int(len(train_idx) * 0.1))
    if val_size < len(train_idx) and len(train_idx) - val_size >= n_cls:
        try:
            train_idx, val_idx = train_test_split(train_idx, test_size=val_size, stratify=lab,
                                                  random_state=seed)
        except ValueError:
            np.random.seed(seed)
            np.random.shuffle(train_idx)
            val_idx, train_idx = train_idx[:val_size], train_idx[val_size:]
    else:
        raise RuntimeError("training set too small to carve a validation set")

    data_norm, _ = normalize(data, train_idx)
    if n_pca and n_pca < data_norm.shape[2]:
        data_proc, _ = apply_pca(data_norm, train_idx, n_pca)
    else:
        data_proc = data_norm
    bs = TRAIN["batch_size"]
    mk = lambda idx, shuf: DataLoader(HSIDataset(extract_patches(data_proc, idx, PATCH),
                                                 flat_gt[idx] - 1), batch_size=bs, shuffle=shuf,
                                      num_workers=0)
    loaders = (mk(train_idx, True), mk(val_idx, False), mk(test_idx, False))
    meta = {"n_classes": DCFG[dataset]["n_classes"], "n_bands": data_proc.shape[2],
            "H": gt.shape[0], "W": gt.shape[1]}
    return loaders, meta, (np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)), info


def make_model(name, n_bands, n_classes):
    if name == "MambaHSI":
        from models.mamba_hsi import MambaHSI
        return MambaHSI(n_bands=n_bands, n_classes=n_classes, patch_size=PATCH, d_model=64,
                        d_state=16, d_conv=4, expand=2, n_spectral_groups=5, n_layers=2)
    from run_benchmark import get_model
    return get_model(name, n_bands=n_bands, n_classes=n_classes, patch_size=PATCH)


def accumulated_epoch(micro):
    """utils.trainer.train_one_epoch with each batch split into chunks of `micro` and the gradients
    accumulated before one optimizer step; the loss is weighted so the update equals the full batch."""
    def train_one_epoch(model, loader, optimizer, criterion, device):
        model.train()
        total = 0.0
        for patches, labels in loader:
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            n = labels.size(0)
            for p, l in zip(patches.split(micro), labels.split(micro)):
                loss = criterion(model(p), l) * (l.size(0) / n)
                loss.backward()
                total += loss.item() * n
            optimizer.step()
        return total / len(loader.dataset)
    return train_one_epoch


def done_keys(path):
    if not Path(path).exists():
        return set()
    with open(path) as f:
        return {(r["dataset"], r["model"], r["split"], r["seed"], r["n_train_per_class"], r["n_pca"])
                for r in csv.DictReader(f)}


def check_splits(args):
    from utils.data_loader import load_dataset
    print(f"{'dataset':13s} {'split':8s} {'seed':>4s} {'tries':>5s} {'train':>6s} {'test':>6s} "
          f"{'cls':>4s} {'testcls':>7s} {'test/lab':>8s} {'min_dist':>8s}")
    for ds in args.datasets:
        _, gt = load_dataset(ds)
        for split in ["random", "disjoint"]:
            for seed in args.seeds:
                tr, te, info = split_for(ds, gt, split, args.n_train, seed)
                d = min_distance(gt.shape, te, tr)
                n_lab = int((gt > 0).sum())
                print(f"{ds:13s} {split:8s} {seed:4d} {str(info.get('tries', '')):>5s} {len(tr):6d} "
                      f"{len(te):6d} {len(np.unique(gt[gt > 0])):4d} {info['n_test_classes']:7d} "
                      f"{len(te) / n_lab:8.2f} {d:8d}")
                if split == "disjoint":
                    assert d > PATCH // 2, "guard violated"
                assert len(np.intersect1d(tr, te)) == 0, "train/test overlap"
    print("splits OK")


def main():
    args = parse()
    if args.check_splits:
        return check_splits(args)

    import torch
    import utils.trainer as T
    from utils.trainer import evaluate, set_seed, train_model
    plain_epoch = T.train_one_epoch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "preds").mkdir(exist_ok=True)
    done = done_keys(out)
    md5 = code_md5()

    for ds in args.datasets:
        for model_name in args.models:
            for seed in args.seeds:
                key = (ds, model_name, args.split, str(seed), str(args.n_train), str(args.n_pca))
                if key in done:
                    print("skip", key)
                    continue
                print(f"\n=== {ds} {model_name} split={args.split} seed={seed} "
                      f"n_train={args.n_train} n_pca={args.n_pca or 'all'}")
                set_seed(seed)
                (trl, vl, tel), meta, (tri, vai, tei), info = build_loaders(
                    ds, args.split, seed, args.n_train, args.n_pca)
                model = make_model(model_name, meta["n_bands"], meta["n_classes"]).to(device)
                micro = args.micro_batch if (args.micro_batch and model_name != "HybridSN") else 0
                T.train_one_epoch = accumulated_epoch(micro) if micro else plain_epoch
                cfg = {**TRAIN, "epochs": args.epochs, "patience": args.epochs,
                       "patch_size": PATCH, "seed": seed}
                res = train_model(model, trl, vl, tel, meta["n_classes"], device, cfg)
                y_true, y_pred = evaluate(model, tel, device)          # best-validation weights
                tag = "_".join(map(str, key))
                np.savez_compressed(out.parent / "preds" / f"{tag}.npz", train_idx=tri, val_idx=vai,
                                    test_idx=tei, y_true=y_true, y_pred=y_pred,
                                    H=meta["H"], W=meta["W"])
                row = {
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": ds,
                    "model": model_name, "split": args.split, "seed": seed,
                    "n_train_per_class": args.n_train, "n_pca": args.n_pca, "patch": PATCH,
                    "block": info.get("block", ""), "guard": info.get("guard", ""),
                    "split_tries": info.get("tries", ""), "n_train": len(tri), "n_val": len(vai),
                    "n_test": len(tei), "n_classes": meta["n_classes"],
                    "n_test_classes": info["n_test_classes"],
                    "OA": res["OA"], "AA": res["AA"], "Kappa": res["Kappa"],
                    "MacroF1": res["MacroF1"], "mIoU": res["mIoU"],
                    "PerClassIoU": json.dumps({int(k): float(v) for k, v in res["PerClassIoU"].items()}),
                    "PerClassF1": json.dumps([float(v) for v in res["PerClassF1"]]),
                    "best_val_OA": res["best_val_OA"], "epochs_trained": res["epochs_trained"],
                    "train_s_per_epoch": res["TrainTime_per_epoch_s"], "params_K": res["Params_K"], "micro_batch": micro,
                    "code_md5": md5, "torch": torch.__version__,
                }
                new = not out.exists()
                with open(out, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS)
                    if new:
                        w.writeheader()
                    w.writerow(row)
                print(f"    OA={res['OA'] * 100:.2f} AA={res['AA'] * 100:.2f} "
                      f"mIoU={res['mIoU'] * 100:.2f}  -> {out}")
                del model
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
