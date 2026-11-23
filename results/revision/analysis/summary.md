# Survey revision: what the results support

Runs with predictions: 315. Seeds per cell: disjoint 10, random 10, random_allbands 5, random_n10 10, random_n20 10, random_n5 10

## random split (PCA-30, 30/class)

- Indian Pines OA: SpectralFormer 88.71, S2Mamba 87.90, HybridSN 81.85. Significant: HybridSN < SpectralFormer (p_holm=0.006); HybridSN < S2Mamba (p_holm=0.006). Indistinguishable: SpectralFormer vs S2Mamba.
- Indian Pines mIoU: S2Mamba 78.99, HybridSN 75.25, SpectralFormer 74.68. No significant difference.
- Pavia University OA: HybridSN 92.96, S2Mamba 92.42, SpectralFormer 91.32. No significant difference.
- Pavia University mIoU: S2Mamba 87.45, HybridSN 86.24, SpectralFormer 83.54. No significant difference.
- Salinas OA: SpectralFormer 93.40, HybridSN 93.33, S2Mamba 92.80. No significant difference.
- Salinas mIoU: SpectralFormer 94.83, S2Mamba 94.31, HybridSN 94.03. No significant difference.

## disjoint split (PCA-30, 30/class)

- Indian Pines OA: SpectralFormer 65.40, S2Mamba 57.11, HybridSN 53.82. Significant: HybridSN < SpectralFormer (p_holm=0.006); SpectralFormer > S2Mamba (p_holm=0.027). Indistinguishable: HybridSN vs S2Mamba.
- Indian Pines mIoU: SpectralFormer 54.42, S2Mamba 46.55, HybridSN 41.53. Significant: HybridSN < SpectralFormer (p_holm=0.006); HybridSN < S2Mamba (p_holm=0.027); SpectralFormer > S2Mamba (p_holm=0.006).
- Pavia University OA: HybridSN 78.38, SpectralFormer 74.82, S2Mamba 74.55. No significant difference.
- Pavia University mIoU: HybridSN 70.26, SpectralFormer 62.85, S2Mamba 61.82. Significant: HybridSN > S2Mamba (p_holm=0.018). Indistinguishable: HybridSN vs SpectralFormer, SpectralFormer vs S2Mamba.
- Salinas OA: HybridSN 81.90, SpectralFormer 80.52, S2Mamba 78.21. No significant difference.
- Salinas mIoU: SpectralFormer 81.48, HybridSN 79.83, S2Mamba 76.12. Significant: SpectralFormer > S2Mamba (p_holm=0.012). Indistinguishable: HybridSN vs SpectralFormer, HybridSN vs S2Mamba.

## Ranking changes between splits

- Indian Pines, mIoU: model order changes from random (S2Mamba > HybridSN > SpectralFormer) to disjoint (SpectralFormer > S2Mamba > HybridSN); check the tests before calling it a change.
- Pavia University, OA: model order changes from random (HybridSN > S2Mamba > SpectralFormer) to disjoint (HybridSN > SpectralFormer > S2Mamba); check the tests before calling it a change.
- Pavia University, mIoU: model order changes from random (S2Mamba > HybridSN > SpectralFormer) to disjoint (HybridSN > SpectralFormer > S2Mamba); check the tests before calling it a change.
- Salinas, OA: model order changes from random (SpectralFormer > HybridSN > S2Mamba) to disjoint (HybridSN > SpectralFormer > S2Mamba); check the tests before calling it a change.
- Salinas, mIoU: model order changes from random (SpectralFormer > S2Mamba > HybridSN) to disjoint (SpectralFormer > HybridSN > S2Mamba); check the tests before calling it a change.

## Split statistics

- Indian Pines disjoint: test pixels 2347-4463, test classes 12-14 of 16, min train-test distance 7, test pixels inside a training patch 0.0%
- Indian Pines random: test pixels 9820-9820, test classes 16-16 of 16, min train-test distance 1, test pixels inside a training patch 94.6%
- Pavia University disjoint: test pixels 28028-30908, test classes 9-9 of 9, min train-test distance 7, test pixels inside a training patch 0.0%
- Pavia University random: test pixels 42506-42506, test classes 9-9 of 9, min train-test distance 1, test pixels inside a training patch 35.3%
- Salinas disjoint: test pixels 35261-37339, test classes 16-16 of 16, min train-test distance 7, test pixels inside a training patch 0.0%
- Salinas random: test pixels 53649-53649, test classes 16-16 of 16, min train-test distance 1, test pixels inside a training patch 62.7%

## Checks

- Metrics recomputed from predictions match runs.csv for OA, AA and mIoU in 315/315 runs.
