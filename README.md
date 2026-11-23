# CNNs, Transformers, and State-Space Models for Hyperspectral Image Analysis: benchmark code

Code, run results and analysis scripts for the controlled benchmark in

> S. El Jarai, A. Elhassouny, and L. F. Mbo Nkoulou, "CNNs, Transformers, and State-Space Models for
> Hyperspectral Image Analysis: Constraints, Trade-Offs, and a Comparative Benchmark," submitted to *IEEE Access*.

The benchmark compares one representative model per family (HybridSN, SpectralFormer, S2Mamba) for
**patch-based pixel classification** on Indian Pines, Pavia University and Salinas, under one protocol:
PCA to 30 components and z-score normalisation fitted on training pixels only, 13 × 13 patches,
30 labelled pixels per class (rare classes: min(15, ⌊n/2⌋)), 10 % of the training pixels held out for
validation, AdamW (learning rate 5e-4, weight decay 1e-4), cosine schedule, 300 epochs, ten seeds
(42, 123, 456, 1–7). It reports random and spatially disjoint sampling, label scarcity (5–30 labels per class),
all bands versus PCA-30, boundary-pixel accuracy and computational efficiency.

## Repository layout

| Path | Content |
|---|---|
| `models/` | HybridSN and SpectralFormer (our implementations following the original papers); S2Mamba: see below |
| `utils/` | data loading, PCA and normalisation, patch extraction, training loop, metrics |
| `run_benchmark.py`, `run_scarcity_only.py` | original benchmark scripts (three seeds) |
| `revision/` | revised protocol: `revision_splits.py` (random and spatially disjoint splits), `run_revision.py` (one run), `plan.py` and `queue.sh` (all runs), `analyze.py` (tables and paired tests), `efficiency.py` (efficiency measurements) |
| `tools/` | `make_disjoint_sameclass_table.py` |
| `figures/` | figure scripts (`make_revision_figures.py`, `prisma_survey.py`) |
| `results/revision/` | `runs.csv` (315 runs), `preds/` (test predictions of every run), `analysis/` (all tables and test results), `efficiency.csv`, `environment.txt` |

## Setup

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Download the three scenes from the [GIC hyperspectral scenes page](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
and place them as:

```
data/indian_pines/Indian_pines_corrected.mat   data/indian_pines/Indian_pines_gt.mat
data/pavia_u/PaviaU.mat                        data/pavia_u/PaviaU_gt.mat
data/salinas/Salinas_corrected.mat             data/salinas/Salinas_gt.mat
```

## Reproducing the results

```bash
python revision/analyze.py                         # all tables and tests from results/revision/ (needs only the *_gt.mat files)
python tools/make_disjoint_sameclass_table.py      # random-vs-disjoint table, Indian Pines scored on the same classes
python figures/make_revision_figures.py            # benchmark figures
python revision/plan.py status                     # list of all runs
bash revision/queue.sh next 60                     # re-run missing runs for about 60 minutes (GPU; S2Mamba runs need models/s2mamba.py)
python revision/efficiency.py                      # efficiency measurements (GPU, nothing else running)
```

Statistics: mean ± sample standard deviation over seeds; two-sided exact Wilcoxon signed-rank test paired by
seed, Holm correction across the three model pairs per scene, sampling protocol and metric (α = 0.05).
Spatially disjoint split: 16 × 16 (Indian Pines) or 32 × 32 (Pavia University, Salinas) pixel blocks, 30 % of blocks
for training, test pixels more than 6 pixels from the training region.

## Credits

- HybridSN: S. K. Roy et al., IEEE GRSL, 2020 (our implementation).
- SpectralFormer: D. Hong et al., IEEE TGRS, 2022 (our re-implementation following the paper; the cross-layer
  adaptive fusion is implemented as a learned weighted residual).
- S2Mamba: G. Wang et al., IEEE TGRS, 2025. Our model file `models/s2mamba.py` is adapted from the official code
  (https://github.com/PURE-melo/S2Mamba) for the unified pipeline. Because the official repository carries no licence,
  the adapted file is not included yet; it will be added once the original authors agree, and is available from the
  corresponding author on request. All S2Mamba results (runs, predictions, tables) are included.

This work was carried out with the support of the National Center for Scientific and Technical Research (CNRST)
as part of the "PhD-Associate Scholarship (PASS)" program.

## Licence

MIT for the code in this repository (see `LICENSE`).
