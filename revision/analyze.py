"""Tables and statistics for the survey revision (IEEE Access resubmission, Access-2026-50490).

Reads results/revision/runs.csv and results/revision/preds/*.npz and writes to results/revision/analysis/:
  T_main_random.{csv,tex}  random split, PCA-30, 30 labels/class, same columns as the published table  (R2.7)
  T_disjoint.{csv,tex}     random and disjoint split side by side, with the drop                       (R1.3, R2.4)
  T_scarcity.{csv,tex}     Indian Pines, 5/10/20/30 labels per class, OA                               (R2.7)
  T_fullband.{csv,tex}     all bands vs PCA-30 on the random split, paired by seed (same split)        (R2.5)
  T_boundary.{csv,tex}     OA and macro F1 on boundary vs interior test pixels                         (R2.5, R1.5)
  T_splits.csv             split statistics per scene, split and seed (sizes, classes, leakage)        (R1.3)
  T_ip_sameclasses.csv     Indian Pines random split scored on the classes present in the disjoint test set
  tests_pairwise.csv       paired Wilcoxon signed-rank tests between models, Holm-adjusted
  summary.md               plain-language list of what the tests support, plus warnings
  check_consistency.csv    metrics recomputed from the saved predictions vs runs.csv

Metric definitions (all recomputed from the saved predictions, one definition for every table):
  OA; AA and mIoU averaged over the classes present in the test set (as utils/metrics.py);
  Kappa; macro F1 over the classes present in the test set (on the random split this equals the
  published sklearn value; on the disjoint split it does not count classes absent from the test set).
Statistics: mean +- sample SD (ddof=1); 95% CI of a mean from the t distribution; two-sided exact Wilcoxon
signed-rank test paired by seed; Holm adjustment over the model pairs of one scene x setting x metric;
a difference is called significant only when the adjusted p < 0.05. With n paired seeds the smallest
possible two-sided p is 2/2^n: 0.25 for 3 seeds, 0.0625 for 5, 0.002 for 10.
Boundary pixels: test pixels with at least one 8-neighbour carrying a different label, unlabelled
background included (in these scenes labelled fields are mostly separated by unlabelled margins, so
neighbours of another labelled class alone cover under 1% of test pixels; that variant is kept in the CSV).

  python revision/analyze.py                       # from the repository root
  python revision/analyze.py --snapshot DIR        # DIR holds runs.csv, preds/ and gt/ (a copy)
"""
import argparse
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat
from scipy.ndimage import distance_transform_cdt

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["HybridSN", "SpectralFormer", "S2Mamba"]
DATASETS = ["indian_pines", "pavia_u", "salinas"]
NAMES = {"indian_pines": "Indian Pines", "pavia_u": "Pavia University", "salinas": "Salinas"}
GT = {"indian_pines": ("data/indian_pines", "Indian_pines_gt.mat", "indian_pines_gt"),
      "pavia_u": ("data/pavia_u", "PaviaU_gt.mat", "paviaU_gt"),
      "salinas": ("data/salinas", "Salinas_gt.mat", "salinas_gt")}
METRICS = ["OA", "AA", "Kappa", "MacroF1", "mIoU"]
PATCH_RADIUS = 6
ALPHA = 0.05


# ----------------------------------------------------------------------------- inputs
def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None, help="folder with runs.csv, preds/ and gt/")
    a = ap.parse_args()
    if a.snapshot:
        s = Path(a.snapshot)
        return s / "runs.csv", s / "preds", s / "gt", s / "analysis"
    r = ROOT / "results" / "revision"
    return r / "runs.csv", r / "preds", None, r / "analysis"


def load_gt(ds, gt_dir):
    folder, fname, key = GT[ds]
    path = (Path(gt_dir) / fname) if gt_dir else (ROOT / folder / fname)
    return loadmat(path)[key].astype(np.int64)


def pred_file(preds, r):
    tag = f"{r.dataset}_{r.model}_{r.split}_{r.seed}_{r.n_train_per_class}_{r.n_pca}"
    return Path(preds) / f"{tag}.npz"


# ----------------------------------------------------------------------------- metrics
def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cls = np.unique(y_true)
    k = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    n = cm.sum()
    tp = np.diag(cm).astype(float)
    row, col = cm.sum(1).astype(float), cm.sum(0).astype(float)
    po = tp.sum() / n
    pe = (row * col).sum() / n ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = tp / row
        prec = np.where(col > 0, tp / col, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        iou = np.where(row + col - tp > 0, tp / (row + col - tp), 0.0)
    return {"OA": po, "AA": float(np.mean(rec[cls])), "Kappa": (po - pe) / (1 - pe) if pe < 1 else 1.0,
            "MacroF1": float(np.mean(f1[cls])), "mIoU": float(np.mean(iou[cls])),
            "_iou": iou, "_rec": rec, "_f1": f1, "_cls": cls}


def subset_scores(y_true, y_pred, classes):
    """AA, mIoU and OA restricted to `classes` (per-class values still use the full test set)."""
    m = metrics(y_true, y_pred)
    classes = np.asarray(classes)
    keep = np.isin(y_true, classes)
    return {"OA": float(np.mean(y_true[keep] == y_pred[keep])), "AA": float(np.mean(m["_rec"][classes])),
            "mIoU": float(np.mean(m["_iou"][classes]))}


def boundary_masks(gt):
    """any_label: an 8-neighbour has a different label (background included); class_only: a different labelled class."""
    H, W = gt.shape
    pad = np.pad(gt, 1, constant_values=-1)
    cls_b = np.zeros((H, W), bool)
    edge = np.zeros((H, W), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = pad[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            cls_b |= (nb > 0) & (nb != gt)
            edge |= (nb == 0)
    return (cls_b | edge).ravel(), cls_b.ravel()


# ----------------------------------------------------------------------------- statistics
def summ(x):
    x = np.asarray(x, float)
    n = len(x)
    sd = x.std(ddof=1) if n > 1 else np.nan
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
    return {"mean": x.mean() if n else np.nan, "sd": sd, "n": n, "ci95": half}


def wilcoxon_p(d):
    d = np.asarray(d, float)
    if len(d) < 2 or np.allclose(d, 0):
        return 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(stats.wilcoxon(d, alternative="two-sided", method="exact").pvalue)
        except TypeError:                                   # older scipy
            return float(stats.wilcoxon(d, alternative="two-sided", mode="exact").pvalue)
        except ValueError:
            return 1.0


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j]))
        adj[j] = run
    return adj


def pairwise(per_run, setting_col="setting"):
    """per_run: one row per run with dataset, setting, model, seed and the metric columns."""
    rows = []
    for (ds, st), g in per_run.groupby(["dataset", setting_col]):
        models = [m for m in MODELS if m in set(g.model)]
        for met in METRICS:
            block = []
            for a, b in combinations(models, 2):
                A = g[g.model == a].set_index("seed")[met]
                B = g[g.model == b].set_index("seed")[met]
                seeds = sorted(set(A.index) & set(B.index))
                if len(seeds) < 2:
                    continue
                d = A.loc[seeds].values - B.loc[seeds].values
                s = summ(d)
                block.append({"dataset": ds, "setting": st, "metric": met, "model_a": a, "model_b": b,
                              "n_seeds": len(seeds), "mean_diff": s["mean"], "ci95_low": s["mean"] - s["ci95"],
                              "ci95_high": s["mean"] + s["ci95"], "a_wins": int((d > 0).sum()),
                              "p": wilcoxon_p(d)})
            if block:
                adj = holm([r["p"] for r in block])
                for r, q in zip(block, adj):
                    r["p_holm"] = q
                    r["significant"] = bool(q < ALPHA)
                rows += block
    return pd.DataFrame(rows)


def sig_lower_than_best(tests, ds, st, met, best, other):
    if tests.empty or best == other:
        return False
    t = tests[(tests.dataset == ds) & (tests.setting == st) & (tests.metric == met) &
              (((tests.model_a == best) & (tests.model_b == other)) |
               ((tests.model_a == other) & (tests.model_b == best)))]
    return bool(len(t)) and bool(t.significant.iloc[0])


# ----------------------------------------------------------------------------- formatting
def cell(s, met, bold=False, star=False):
    if np.isnan(s["mean"]):
        return "--"
    if met == "Kappa":
        txt = f"{s['mean']:.4f} \\pm {s['sd']:.3f}" if s["n"] > 1 else f"{s['mean']:.4f}"
    else:
        txt = f"{100 * s['mean']:.2f} \\pm {100 * s['sd']:.2f}" if s["n"] > 1 else f"{100 * s['mean']:.2f}"
    if bold:
        txt = f"\\mathbf{{{txt}}}"
    return f"${txt}{'^{*}' if star else ''}$"


def table_cells(per_run, tests, st, ds, met):
    g = per_run[(per_run.setting == st) & (per_run.dataset == ds)]
    models = [m for m in MODELS if m in set(g.model)]
    S = {m: summ(g[g.model == m][met]) for m in models}
    if not S:
        return {}
    best = max(S, key=lambda m: S[m]["mean"])
    return {m: cell(S[m], met, bold=(m == best), star=sig_lower_than_best(tests, ds, st, met, best, m))
            for m in models}


def n_seed_text(per_run, st):
    n = per_run[per_run.setting == st].groupby(["dataset", "model"]).seed.nunique()
    return f"{n.min()}" if n.min() == n.max() else f"{n.min()}--{n.max()}"


STAR_NOTE = ("Bold: best mean per dataset and metric; $^{*}$: significantly lower than the best model "
             "(two-sided paired Wilcoxon signed-rank test over seeds, Holm-adjusted $p<0.05$).")


# ----------------------------------------------------------------------------- main
def main():
    runs_csv, preds, gt_dir, out = parse()
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(runs_csv).drop_duplicates(
        subset=["dataset", "model", "split", "seed", "n_train_per_class", "n_pca"], keep="last")
    gts = {ds: load_gt(ds, gt_dir) for ds in DATASETS if ds in set(runs.dataset)}
    bmask = {ds: boundary_masks(g) for ds, g in gts.items()}
    dist_cache = {}

    per_run, checks, bnd, splits, preds_by_key = [], [], [], [], {}
    for r in runs.itertuples(index=False):
        f = pred_file(preds, r)
        if not f.exists():
            continue
        z = np.load(f)
        yt, yp, te = z["y_true"], z["y_pred"], z["test_idx"]
        m = metrics(yt, yp)
        setting = (r.split if (r.n_train_per_class == 30 and r.n_pca == 30) else
                   f"{r.split}_n{r.n_train_per_class}" if r.n_pca == 30 else f"{r.split}_allbands")
        row = {"dataset": r.dataset, "model": r.model, "split": r.split, "seed": int(r.seed),
               "n_train_per_class": int(r.n_train_per_class), "n_pca": int(r.n_pca), "setting": setting,
               **{k: m[k] for k in METRICS}}
        per_run.append(row)
        preds_by_key[(r.dataset, r.model, r.split, int(r.seed), int(r.n_train_per_class), int(r.n_pca))] = (yt, yp)
        checks.append({**{k: row[k] for k in ["dataset", "model", "split", "seed", "n_train_per_class", "n_pca"]},
                       **{f"d_{k}": m[k] - getattr(r, k) for k in ["OA", "AA", "mIoU", "MacroF1"]}})
        # boundary vs interior
        anyb, clsb = bmask[r.dataset]
        for name, mask in [("any_label", anyb), ("class_only", clsb)]:
            isb = mask[te]
            for grp, sel in [("boundary", isb), ("interior", ~isb)]:
                if sel.sum() == 0:
                    continue
                mm = metrics(yt[sel], yp[sel])
                bnd.append({**{k: row[k] for k in ["dataset", "model", "setting", "seed"]}, "definition": name,
                            "group": grp, "n_pixels": int(sel.sum()), "share_of_test": float(sel.mean()),
                            "OA": mm["OA"], "MacroF1": mm["MacroF1"]})
        # split statistics, once per scene x split x seed x labels
        skey = (r.dataset, r.split, int(r.seed), int(r.n_train_per_class))
        if skey not in dist_cache:
            gt = gts[r.dataset]
            H, W = gt.shape
            used = np.concatenate([z["train_idx"], z["val_idx"]])
            mask = np.ones(H * W, bool)
            mask[used] = False
            dist = distance_transform_cdt(mask.reshape(H, W), metric="chessboard").ravel()
            dte = dist[te]
            n_lab = int((gt > 0).sum())
            dist_cache[skey] = True
            splits.append({"dataset": r.dataset, "split": r.split, "seed": int(r.seed),
                           "n_train_per_class": int(r.n_train_per_class), "n_train": len(z["train_idx"]),
                           "n_val": len(z["val_idx"]), "n_test": len(te),
                           "n_classes": int(len(np.unique(gt[gt > 0]))), "n_test_classes": int(len(np.unique(yt))),
                           "test_share_of_labelled": len(te) / n_lab, "min_dist_train_test": int(dte.min()),
                           "test_within_1px_of_train": float(np.mean(dte <= 1)),
                           "test_inside_a_training_patch": float(np.mean(dte <= PATCH_RADIUS))})
    P = pd.DataFrame(per_run)
    if P.empty:
        raise SystemExit("no runs with saved predictions found")
    pd.DataFrame(checks).to_csv(out / "check_consistency.csv", index=False)
    pd.DataFrame(splits).sort_values(["dataset", "split", "n_train_per_class", "seed"]).to_csv(
        out / "T_splits.csv", index=False)

    tests = pairwise(P)
    tests.to_csv(out / "tests_pairwise.csv", index=False)
    notes = []

    # ---- main table (random, PCA-30, 30/class), same columns as the published table
    rows, tex = [], []
    for ds in DATASETS:
        g = P[(P.setting == "random") & (P.dataset == ds)]
        if g.empty:
            continue
        cells = {met: table_cells(P, tests, "random", ds, met) for met in METRICS}
        models = [m for m in MODELS if m in set(g.model)]
        for i, mo in enumerate(models):
            s = {met: summ(g[g.model == mo][met]) for met in METRICS}
            rows.append({"dataset": ds, "model": mo, "n_seeds": s["OA"]["n"],
                         **{f"{met}_{k}": s[met][k] for met in METRICS for k in ["mean", "sd", "ci95"]}})
            first = f"\\multirow{{{len(models)}}}{{*}}{{{NAMES[ds]}}}" if i == 0 else ""
            tex.append(f"{first} & {mo} & " + " & ".join(cells[met][mo] for met in METRICS) + r" \\")
        tex.append(r"\midrule")
    if rows:
        pd.DataFrame(rows).to_csv(out / "T_main_random.csv", index=False)
        tex[-1] = r"\bottomrule"
        head = [r"\begin{table*}[!t]", r"\centering",
                r"\caption{Classification accuracy metrics across datasets and models (random split, 30 labelled "
                r"pixels per class, PCA-30). Results are mean $\pm$ sample standard deviation over "
                f"{n_seed_text(P, 'random')} seeds. " + STAR_NOTE +
                r" OA: overall accuracy; AA: average accuracy; mIoU: mean intersection over union.}",
                r"\label{tab:benchmark_accuracy}", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
                r"\renewcommand{\arraystretch}{1.15}", r"\begin{tabular}{llccccc}", r"\toprule",
                r"\textbf{Dataset} & \textbf{Model} & \textbf{OA (\%)} & \textbf{AA (\%)} & \textbf{Kappa} & "
                r"\textbf{Macro F1 (\%)} & \textbf{mIoU (\%)} \\", r"\midrule"]
        (out / "T_main_random.tex").write_text("\n".join(head + tex + [r"\end{tabular}", r"\end{table*}"]) + "\n")

    # ---- random vs disjoint
    rows, tex = [], []
    show = ["OA", "mIoU"]
    for ds in DATASETS:
        gr, gd = P[(P.setting == "random") & (P.dataset == ds)], P[(P.setting == "disjoint") & (P.dataset == ds)]
        if gd.empty:
            continue
        models = [m for m in MODELS if m in set(gd.model)]
        cr = {met: table_cells(P, tests, "random", ds, met) for met in show}
        cd = {met: table_cells(P, tests, "disjoint", ds, met) for met in show}
        for i, mo in enumerate(models):
            line = {"dataset": ds, "model": mo}
            parts = []
            for met in show:
                sr, sd_ = summ(gr[gr.model == mo][met]), summ(gd[gd.model == mo][met])
                drop = summ(np.array([sr["mean"] - sd_["mean"]]))
                line.update({f"{met}_random_mean": sr["mean"], f"{met}_random_sd": sr["sd"], f"{met}_random_n": sr["n"],
                             f"{met}_disjoint_mean": sd_["mean"], f"{met}_disjoint_sd": sd_["sd"],
                             f"{met}_disjoint_n": sd_["n"], f"{met}_drop_points": 100 * drop["mean"]})
                parts += [cr[met].get(mo, "--"), cd[met].get(mo, "--"), f"${100 * drop['mean']:.1f}$"]
            rows.append(line)
            first = f"\\multirow{{{len(models)}}}{{*}}{{{NAMES[ds]}}}" if i == 0 else ""
            tex.append(f"{first} & {mo} & " + " & ".join(parts) + r" \\")
        tex.append(r"\midrule")
        # ranking change
        for met in show:
            rk = lambda g: [m for m, _ in sorted(((m, g[g.model == m][met].mean()) for m in models),
                                                 key=lambda t: -t[1])]
            a, b = rk(gr), rk(gd)
            if a != b:
                notes.append(f"- {NAMES[ds]}, {met}: model order changes from random ({' > '.join(a)}) "
                             f"to disjoint ({' > '.join(b)}); check the tests before calling it a change.")
    if rows:
        pd.DataFrame(rows).to_csv(out / "T_disjoint.csv", index=False)
        tex[-1] = r"\bottomrule"
        head = [r"\begin{table*}[!t]", r"\centering",
                r"\caption{Random versus spatially disjoint split (30 labelled pixels per class, PCA-30). "
                r"Mean $\pm$ sample standard deviation over " + n_seed_text(P, "disjoint") +
                r" seeds; Drop: random minus disjoint, in percentage points. Disjoint: block partition with a "
                r"guard band of 6 pixels, so no test pixel lies inside a training patch. " + STAR_NOTE +
                r" On Indian Pines, classes absent from a disjoint test set are excluded from mIoU.}",
                r"\label{tab:disjoint}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
                r"\renewcommand{\arraystretch}{1.15}", r"\begin{tabular}{llcccccc}", r"\toprule",
                r" & & \multicolumn{3}{c}{\textbf{OA (\%)}} & \multicolumn{3}{c}{\textbf{mIoU (\%)}} \\",
                r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
                r"\textbf{Dataset} & \textbf{Model} & Random & Disjoint & Drop & Random & Disjoint & Drop \\",
                r"\midrule"]
        (out / "T_disjoint.tex").write_text("\n".join(head + tex + [r"\end{tabular}", r"\end{table*}"]) + "\n")

    # ---- Indian Pines random scored on the disjoint test classes (same seed, same model)
    rows = []
    for (ds, mo, sp, seed, nt, npca), (yt, yp) in preds_by_key.items():
        if ds != "indian_pines" or sp != "disjoint" or nt != 30 or npca != 30:
            continue
        rnd = preds_by_key.get((ds, mo, "random", seed, 30, 30))
        if rnd is None:
            continue
        cls = np.unique(yt)
        sc = subset_scores(rnd[0], rnd[1], cls)
        rows.append({"model": mo, "seed": seed, "n_test_classes": len(cls),
                     **{f"random_sameclasses_{k}": v for k, v in sc.items()},
                     **{f"disjoint_{k}": metrics(yt, yp)[k] for k in ["OA", "AA", "mIoU"]}})
    if rows:
        pd.DataFrame(rows).sort_values(["model", "seed"]).to_csv(out / "T_ip_sameclasses.csv", index=False)

    # ---- scarcity (Indian Pines, OA): 30/class taken from the main random runs
    sc = P[(P.dataset == "indian_pines") & (P.split == "random") & (P.n_pca == 30)].copy()
    if sc.n_train_per_class.nunique() > 1:
        sc["setting"] = "scarcity_n" + sc.n_train_per_class.astype(str)
        st_tests = pairwise(sc)
        st_tests.to_csv(out / "tests_scarcity.csv", index=False)
        rows, tex = [], []
        models = [m for m in MODELS if m in set(sc.model)]
        for k in sorted(sc.n_train_per_class.unique()):
            st = f"scarcity_n{k}"
            cells = table_cells(sc, st_tests, st, "indian_pines", "OA")
            for mo in models:
                s = summ(sc[(sc.setting == st) & (sc.model == mo)]["OA"])
                rows.append({"n_per_class": k, "model": mo, "OA_mean": s["mean"], "OA_sd": s["sd"], "n_seeds": s["n"]})
            tex.append(f"{k} & " + " & ".join(cells.get(mo, "--") for mo in models) + r" \\")
        pd.DataFrame(rows).to_csv(out / "T_scarcity.csv", index=False)
        nseed = sc.groupby(["n_train_per_class", "model"]).seed.nunique()
        head = [r"\begin{table}[!t]", r"\centering",
                r"\caption{Label scarcity experiment on Indian Pines. Overall accuracy (\%) as mean $\pm$ sample "
                f"standard deviation over {nseed.min()}--{nseed.max()} seeds. " + STAR_NOTE.replace("dataset and metric", "sample size") + "}",
                r"\label{tab:scarcity_results}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
                r"\renewcommand{\arraystretch}{1.15}", r"\begin{tabular}{l" + "c" * len(models) + "}", r"\toprule",
                r"\textbf{Samples / class} & " + " & ".join(f"\\textbf{{{m}}}" for m in models) + r" \\", r"\midrule"]
        (out / "T_scarcity.tex").write_text("\n".join(head + tex + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]) + "\n")

    # ---- all bands vs PCA-30 (random split; same seed = same split, so paired)
    fb = P[(P.split == "random") & (P.n_train_per_class == 30)]
    rows, tex = [], []
    for ds in DATASETS:
        for mo in MODELS:
            A = fb[(fb.dataset == ds) & (fb.model == mo) & (fb.n_pca == 0)].set_index("seed")
            B = fb[(fb.dataset == ds) & (fb.model == mo) & (fb.n_pca == 30)].set_index("seed")
            seeds = sorted(set(A.index) & set(B.index))
            if not seeds:
                continue
            line = {"dataset": ds, "model": mo, "n_seeds": len(seeds)}
            parts = []
            for met in ["OA", "mIoU"]:
                d = A.loc[seeds, met].values - B.loc[seeds, met].values
                s = summ(d)
                line.update({f"{met}_pca30_mean": B.loc[seeds, met].mean(), f"{met}_allbands_mean": A.loc[seeds, met].mean(),
                             f"{met}_diff_points": 100 * s["mean"], f"{met}_diff_ci95_points": 100 * s["ci95"],
                             f"{met}_allbands_better": int((d > 0).sum()), f"{met}_p": wilcoxon_p(d)})
                ci = f" [{100 * (s['mean'] - s['ci95']):+.1f}, {100 * (s['mean'] + s['ci95']):+.1f}]" if len(seeds) > 1 else ""
                parts += [cell(summ(B.loc[seeds, met]), met), cell(summ(A.loc[seeds, met]), met),
                          f"${100 * s['mean']:+.1f}${ci}"]
            rows.append(line)
            tex.append(f"{NAMES[ds]} & {mo} & " + " & ".join(parts) + r" \\")
    if rows:
        pd.DataFrame(rows).to_csv(out / "T_fullband.csv", index=False)
        head = [r"\begin{table*}[!t]", r"\centering",
                r"\caption{All spectral bands versus PCA-30 (random split, 30 labelled pixels per class, same splits). "
                r"Mean $\pm$ sample standard deviation; Diff: all bands minus PCA-30 in percentage points with its 95\% "
                r"confidence interval over seeds.}", r"\label{tab:fullband}", r"\scriptsize",
                r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llcccccc}", r"\toprule",
                r" & & \multicolumn{3}{c}{\textbf{OA (\%)}} & \multicolumn{3}{c}{\textbf{mIoU (\%)}} \\",
                r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
                r"\textbf{Dataset} & \textbf{Model} & PCA-30 & All bands & Diff & PCA-30 & All bands & Diff \\", r"\midrule"]
        (out / "T_fullband.tex").write_text("\n".join(head + tex + [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]) + "\n")

    # ---- boundary vs interior (primary definition: any different label), random and disjoint, PCA-30, 30/class
    Bd = pd.DataFrame(bnd)
    rows, tex = [], []
    if not Bd.empty:
        for (ds, st, mo, de), g in Bd[Bd.setting.isin(["random", "disjoint"])].groupby(
                ["dataset", "setting", "model", "definition"]):
            b, i = g[g.group == "boundary"], g[g.group == "interior"]
            seeds = sorted(set(b.seed) & set(i.seed))
            b, i = b.set_index("seed").loc[seeds], i.set_index("seed").loc[seeds]
            line = {"dataset": ds, "setting": st, "model": mo, "definition": de, "n_seeds": len(seeds),
                    "boundary_share_of_test": b.share_of_test.mean()}
            for met in ["OA", "MacroF1"]:
                gap = summ(i[met].values - b[met].values)
                line.update({f"{met}_boundary": b[met].mean(), f"{met}_interior": i[met].mean(),
                             f"{met}_gap_points": 100 * gap["mean"], f"{met}_gap_ci95_points": 100 * gap["ci95"]})
            rows.append(line)
        T = pd.DataFrame(rows)
        T.to_csv(out / "T_boundary.csv", index=False)
        prim = T[T.definition == "any_label"]
        for ds in DATASETS:
            for st in ["random", "disjoint"]:
                g = prim[(prim.dataset == ds) & (prim.setting == st)]
                for _, r in g.iterrows():
                    tex.append(f"{NAMES[ds]} & {st} & {r.model} & ${100 * r.boundary_share_of_test:.1f}$ & "
                               f"${100 * r.OA_boundary:.2f}$ & ${100 * r.OA_interior:.2f}$ & ${r.OA_gap_points:.1f}$ & "
                               f"${100 * r.MacroF1_boundary:.2f}$ & ${100 * r.MacroF1_interior:.2f}$ \\\\")
        head = [r"\begin{table*}[!t]", r"\centering",
                r"\caption{Accuracy on boundary and interior test pixels (PCA-30, 30 labelled pixels per class). "
                r"Boundary pixels have at least one 8-neighbour with a different label, unlabelled pixels included. "
                r"Values are means over seeds; Gap: interior minus boundary OA in percentage points.}", r"\label{tab:boundary}",
                r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lllcccccc}", r"\toprule",
                r"\textbf{Dataset} & \textbf{Split} & \textbf{Model} & \textbf{Boundary (\%)} & \textbf{OA bnd.} & "
                r"\textbf{OA int.} & \textbf{Gap} & \textbf{F1 bnd.} & \textbf{F1 int.} \\", r"\midrule"]
        (out / "T_boundary.tex").write_text("\n".join(head + tex + [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]) + "\n")

    # ---- summary
    lines = ["# Survey revision: what the results support", "",
             f"Runs with predictions: {len(P)}. Seeds per cell: "
             + ", ".join(f"{st} {n_seed_text(P, st)}" for st in sorted(P.setting.unique())), ""]
    for st in ["random", "disjoint"]:
        lines += [f"## {st} split (PCA-30, 30/class)", ""]
        for ds in DATASETS:
            g = P[(P.setting == st) & (P.dataset == ds)]
            if g.empty:
                continue
            for met in ["OA", "mIoU"]:
                means = g.groupby("model")[met].mean().sort_values(ascending=False)
                t = tests[(tests.dataset == ds) & (tests.setting == st) & (tests.metric == met)] if not tests.empty else tests
                sig = [f"{r.model_a} {'>' if r.mean_diff > 0 else '<'} {r.model_b} (p_holm={r.p_holm:.3f})"
                       for r in t.itertuples() if r.significant] if not t.empty else []
                ns = [f"{r.model_a} vs {r.model_b}" for r in t.itertuples() if not r.significant] if not t.empty else []
                lines.append(f"- {NAMES[ds]} {met}: " + ", ".join(f"{m} {100 * v:.2f}" for m, v in means.items())
                             + (f". Significant: {'; '.join(sig)}" if sig else ". No significant difference")
                             + (f". Indistinguishable: {', '.join(ns)}" if ns and sig else "") + ".")
        lines.append("")
    if notes:
        lines += ["## Ranking changes between splits", ""] + notes + [""]
    S = pd.DataFrame(splits)
    if not S.empty:
        lines += ["## Split statistics", ""]
        for (ds, sp), g in S[S.n_train_per_class == 30].groupby(["dataset", "split"]):
            lines.append(f"- {NAMES[ds]} {sp}: test pixels {g.n_test.min()}-{g.n_test.max()}, "
                         f"test classes {g.n_test_classes.min()}-{g.n_test_classes.max()} of {g.n_classes.iloc[0]}, "
                         f"min train-test distance {g.min_dist_train_test.min()}, "
                         f"test pixels inside a training patch {100 * g.test_inside_a_training_patch.mean():.1f}%")
        lines.append("")
    C = pd.DataFrame(checks)
    bad = C[(C[["d_OA", "d_AA", "d_mIoU"]].abs() > 1e-6).any(axis=1)]
    lines += ["## Checks", "",
              f"- Metrics recomputed from predictions match runs.csv for OA, AA and mIoU in {len(C) - len(bad)}/{len(C)} runs."]
    low = P.groupby(["setting", "dataset", "model"]).seed.nunique()
    if (low[low.index.get_level_values(0).isin(["random", "disjoint"])] < 10).any():
        lines.append("- Some random/disjoint cells have fewer than 10 seeds: tests cannot reach significance "
                     "below 5 seeds (smallest p = 2/2^n).")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
