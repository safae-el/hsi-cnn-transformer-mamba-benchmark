import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
import warnings

warnings.filterwarnings("ignore")

# ─── Global conference-style aesthetic ─────────────────────────────────────────

rcParams.update({
    # White background everywhere
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.transparent": False,

    # Conference-friendly font
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,

    # Clean axes
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",

    # Light grid
    "grid.color": "0.85",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,

    # Better fonts in exported PDF
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Legend
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "0.8",
})

# Simple conference-friendly colors
PALETTE = {
    "HybridSN": "#4C78A8",        # blue
    "SpectralFormer": "#F58518",  # orange
    "S2Mamba": "#54A24B",         # green
}

HATCH = {
    "HybridSN": "",
    "SpectralFormer": "//",
    "S2Mamba": "..",
}

EDGE_COLOR = "black"
GRID_COLOR = "0.85"
TEXT_COLOR = "black"


# ─── Load revision results (10 runs) ───────────────────────────────────────────
import argparse, ast
import pandas as pd
ap = argparse.ArgumentParser(description="Revision figures from runs.csv and efficiency.csv")
ap.add_argument("--runs", default="results/revision/runs.csv")
ap.add_argument("--eff", default="results/revision/efficiency.csv")
ap.add_argument("--out", default="results/revision/figures")
args = ap.parse_args()

MODELS = ["HybridSN", "SpectralFormer", "S2Mamba"]
DATASETS = ["indian_pines", "pavia_u", "salinas"]
DATASET_LABELS = ["Indian Pines", "Pavia University", "Salinas"]
OUT = args.out
os.makedirs(OUT, exist_ok=True)

R = pd.read_csv(args.runs)
R = R[R.model.isin(MODELS)]
MAIN = R[(R.split == "random") & (R.n_pca == 30) & (R.n_train_per_class == 30)]
SCAR = R[(R.dataset == "indian_pines") & (R.split == "random") & (R.n_pca == 30)]
N_SEEDS = MAIN.groupby(["dataset", "model"]).seed.nunique()
print("seeds per dataset/model (main):"); print(N_SEEDS.to_string())

# ─── Helper functions ─────────────────────────────────────────────────────────

def _strip(ax):
    """Minimal clean axis style."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.yaxis.grid(True, alpha=0.9)
    ax.set_axisbelow(True)


def _suptitle(fig, text):
    fig.suptitle(
        text,
        fontsize=12,
        fontweight="bold",
        x=0.02,
        ha="left",
        color=TEXT_COLOR,
    )


def save_fig(fig, path_base):
    """
    Save each figure in both PDF and PNG formats.

    Example:
        save_fig(fig, "results/figures/plotA_benchmark_overview")
    """
    pdf_path = f"{path_base}.pdf"
    png_path = f"{path_base}.png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )

    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")



def stat(df, metric):
    v = df[metric].astype(float) * 100
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)

CLASS_NAMES = {
    "indian_pines": [
        "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
        "Grass-pasture", "Grass-trees", "Grass-mowed",
        "Hay-windrowed", "Oats", "Soybean-notill",
        "Soybean-mintill", "Soybean-clean", "Wheat",
        "Woods", "Buildings", "Stone-steel",
    ],
    "pavia_u": [
        "Asphalt", "Meadows", "Gravel", "Trees",
        "Painted metal", "Bare soil", "Bitumen",
        "Bricks", "Shadows",
    ],
    "salinas": [
        "Weeds-1", "Weeds-2", "Fallow", "Fallow-rough",
        "Fallow-smooth", "Stubble", "Celery", "Grapes",
        "Soil", "Corn", "Lettuce-4wk", "Lettuce-5wk",
        "Lettuce-6wk", "Lettuce-7wk", "Vinyard-un",
        "Vinyard-tr",
    ],
}



def plot_benchmark_overview():
    metrics = ["OA", "AA", "MacroF1", "mIoU"]
    metric_labels = ["OA (%)", "AA (%)", "Macro F1 (%)", "mIoU (%)"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    _suptitle(fig, "Benchmark overview across datasets and models")
    x = np.arange(len(DATASETS)); width = 0.25
    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        for i, model in enumerate(MODELS):
            ms = [stat(MAIN[(MAIN.dataset == ds) & (MAIN.model == model)], metric) for ds in DATASETS]
            ax.bar(x + i * width, [m for m, s in ms], width, yerr=[s for m, s in ms], capsize=3,
                   label=model, color=PALETTE[model], hatch=HATCH[model], alpha=0.95,
                   edgecolor=EDGE_COLOR, linewidth=0.6, error_kw={"linewidth": 0.9, "ecolor": EDGE_COLOR})
        ax.set_title(mlabel, pad=6)
        ax.set_xticks(x + width); ax.set_xticklabels(DATASET_LABELS, rotation=20, ha="right")
        ax.set_ylim(50, 108); ax.set_ylabel("Score (%)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g")); _strip(ax)
    handles = [mpatches.Patch(facecolor=PALETTE[m], edgecolor=EDGE_COLOR, hatch=HATCH[m], label=m, linewidth=0.6) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, f"{OUT}/plotA_benchmark_overview")


def plot_scalability_profile():
    E = pd.read_csv(args.eff); E = E[E.model.isin(MODELS)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    _suptitle(fig, "Efficiency profile averaged across datasets")
    metrics = [("params_K", "Parameters (K)", "{:.1f}"),
               ("peak_MB_b256_total", "Peak GPU memory, batch 256 (MB)", "{:.1f}"),
               ("train_step_ms_b64", "Training step, batch 64 (ms)", "{:.1f}"),
               ("latency_ms_b1", "Inference latency, batch 1 (ms)", "{:.2f}")]
    for ax, (key, label, fmt) in zip(axes, metrics):
        vals = [E[E.model == m][key].astype(float).mean() for m in MODELS]
        for j, (model, val) in enumerate(zip(MODELS, vals)):
            ax.bar(j, val, color=PALETTE[model], hatch=HATCH[model], alpha=0.95, edgecolor=EDGE_COLOR, linewidth=0.6, width=0.55)
            ax.text(j, val + max(vals) * 0.025, fmt.format(val), ha="center", va="bottom", fontsize=8)
        ax.set_title(label, pad=6); ax.set_ylim(0, max(vals) * 1.28)
        ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, rotation=15, ha="right"); _strip(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, f"{OUT}/plotB_scalability_profile")


def plot_scarcity_curve():
    sizes = sorted(SCAR.n_train_per_class.unique())
    fig, ax = plt.subplots(figsize=(7, 5))
    _suptitle(fig, "Label scarcity analysis on Indian Pines")
    markers = {"HybridSN": "o", "SpectralFormer": "s", "S2Mamba": "^"}
    for model in MODELS:
        ms = np.array([stat(SCAR[(SCAR.model == model) & (SCAR.n_train_per_class == n)], "OA") for n in sizes])
        ax.plot(sizes, ms[:, 0], marker=markers[model], markersize=5, linewidth=2, label=model,
                color=PALETTE[model], markeredgecolor=EDGE_COLOR, markeredgewidth=0.6)
        ax.fill_between(sizes, ms[:, 0] - ms[:, 1], ms[:, 0] + ms[:, 1], alpha=0.15, color=PALETTE[model], linewidth=0)
    ax.set_xlabel("Training samples per class"); ax.set_ylabel("Overall accuracy (%)")
    ax.set_xticks(sizes); ax.set_ylim(30, 100); ax.legend(); _strip(ax)
    plt.tight_layout()
    save_fig(fig, f"{OUT}/plotC_scarcity_curve")


def plot_classwise_f1():
    cmap = LinearSegmentedColormap.from_list("conference_heatmap", ["#F7FBFF", "#6BAED6", "#08306B"], N=256)
    for dataset, ds_label in zip(DATASETS, DATASET_LABELS):
        class_names = CLASS_NAMES[dataset]; n_classes = len(class_names)
        matrix = np.zeros((len(MODELS), n_classes))
        for i, model in enumerate(MODELS):
            rows = MAIN[(MAIN.dataset == dataset) & (MAIN.model == model)].PerClassF1
            f1 = np.mean([ast.literal_eval(v) for v in rows], axis=0)
            matrix[i, :len(f1)] = f1[:n_classes]
        fig, ax = plt.subplots(figsize=(max(10, n_classes * 0.75), 3.8))
        _suptitle(fig, f"Per-class F1 scores on {ds_label}")
        im = ax.imshow(matrix * 100, aspect="auto", cmap=cmap, vmin=0, vmax=100)
        ax.set_xticks(range(n_classes)); ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MODELS, fontsize=9)
        for i in range(len(MODELS)):
            for j in range(n_classes):
                val = matrix[i, j] * 100
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if val > 60 else "black", fontweight="bold")
        cb = plt.colorbar(im, ax=ax, shrink=0.8); cb.ax.tick_params(labelsize=8); cb.set_label("F1 score (%)", fontsize=9)
        for xx in np.arange(-0.5, n_classes, 1): ax.axvline(xx, color="white", linewidth=0.4)
        for yy in np.arange(-0.5, len(MODELS), 1): ax.axhline(yy, color="white", linewidth=0.4)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        save_fig(fig, f"{OUT}/plotD_classwise_{dataset.replace('_', '')}")


def plot_radar_profile():
    metrics = ["OA", "AA", "MacroF1", "mIoU", "Kappa"]; m_labels = ["OA", "AA", "Macro F1", "mIoU", "Kappa"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist(); angles += angles[:1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), subplot_kw=dict(polar=True))
    _suptitle(fig, "Multi-metric radar profile per dataset")
    for ax, dataset, ds_label in zip(axes, DATASETS, DATASET_LABELS):
        for model in MODELS:
            vals = [stat(MAIN[(MAIN.dataset == dataset) & (MAIN.model == model)], m)[0] for m in metrics]
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=1.8, label=model, color=PALETTE[model])
            ax.fill(angles, vals, alpha=0.08, color=PALETTE[model])
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(m_labels, fontsize=8.5); ax.set_ylim(0, 100)
        ax.set_title(ds_label, fontsize=10, fontweight="bold", pad=14)
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--"); ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
        ax.spines["polar"].set_edgecolor("0.75")
    handles = [mpatches.Patch(facecolor=PALETTE[m], edgecolor=EDGE_COLOR, label=m, linewidth=0.6) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, f"{OUT}/plotE_radar_profile")


if __name__ == "__main__":
    plot_benchmark_overview()
    if os.path.exists(args.eff): plot_scalability_profile()
    else: print("efficiency.csv not found: Fig. 6 skipped")
    plot_scarcity_curve()
    plot_classwise_f1()
    plot_radar_profile()
