"""Train/test splits for the IEEE Access survey revision.

random_fewshot_split   : the published protocol (utils.data_loader.stratified_split), unchanged.
disjoint_fewshot_split : spatially disjoint version of the same few-shot protocol.

Disjoint protocol
  1. The scene is cut into block x block cells; each cell is assigned to the training
     region with probability train_frac (assignment drawn from the run seed).
  2. Test pixels are the labelled pixels whose Chebyshev distance to the training region
     is larger than `guard`. With guard = patch_size // 2, no labelled training pixel lies
     inside any test patch and no test pixel lies inside any training patch.
  3. Training pixels are drawn per class from the labelled pixels inside the training
     region, with the same per-class budget as the random protocol
     (n if the class has >= 2n pixels, else min(15, n // 2), at least 1),
     capped by the number of pixels of that class inside the training region.
  4. For each seed, 200 cell assignments are drawn; among those that put every class in the
     training region, the one with the most classes in the test region is kept (ties: most
     test pixels). Classes whose whole field fits inside one block plus guard (e.g. Oats on
     Indian Pines) cannot appear on both sides and are reported as absent from the test set.
Only numpy and scipy are needed, so the split can be checked without a GPU.
"""
import numpy as np
from scipy.ndimage import binary_dilation

# block size per scene (pixels); guard defaults to patch_size // 2
DISJOINT_CFG = {
    "indian_pines": {"block": 16, "train_frac": 0.3},
    "pavia_u":      {"block": 32, "train_frac": 0.3},
    "salinas":      {"block": 32, "train_frac": 0.3},
}


def class_budget(n_class_pixels, n_samples):
    """Per-class training budget of the published protocol (utils.data_loader.stratified_split)."""
    n_train = n_samples if n_class_pixels >= 2 * n_samples else min(15, n_class_pixels // 2)
    return max(1, n_train)


def random_fewshot_split(gt, n_samples, seed):
    """Published protocol, imported unchanged so random-split results stay comparable."""
    from utils.data_loader import stratified_split
    return stratified_split(gt, n_samples=n_samples, seed=seed)


def _train_region(shape, block, train_frac, rng):
    H, W = shape
    bh, bw = -(-H // block), -(-W // block)
    cells = rng.random((bh, bw)) < train_frac
    region = np.kron(cells, np.ones((block, block), dtype=bool))[:H, :W]
    return region


def disjoint_fewshot_split(gt, n_samples, seed, block, train_frac=0.3, guard=6, n_draws=200):
    """Returns train_idx, test_idx (flat row-major indices) and an info dict.

    n_draws random cell assignments are drawn for the seed. Among those that put every class in
    the training region, the one with the most classes in the test region is kept (ties: most
    test pixels). The choice uses only the label map, never model results.
    """
    flat = gt.ravel()
    labelled = gt > 0
    classes = np.arange(1, gt.max() + 1)
    classes = classes[[np.any(flat == c) for c in classes]]
    kernel = np.ones((2 * guard + 1, 2 * guard + 1), dtype=bool)
    best, n_valid = None, 0
    for sub in range(n_draws):
        rng = np.random.default_rng([seed, sub])
        region = _train_region(gt.shape, block, train_frac, rng)
        cand_flat = (region & labelled).ravel()
        if not all(np.any(cand_flat & (flat == c)) for c in classes):
            continue
        n_valid += 1
        test_mask = labelled & ~binary_dilation(region, structure=kernel)
        score = (len(np.unique(gt[test_mask])), int(test_mask.sum()))
        if best is None or score > best[0]:
            best = (score, sub, cand_flat, test_mask)
    if best is None:
        raise RuntimeError(f"no cell assignment out of {n_draws} put every class in the training "
                           f"region (block={block}, train_frac={train_frac}); lower the block size")
    (n_test_cls, _), sub, cand_flat, test_mask = best
    pick = np.random.default_rng([seed, sub, 1])
    train_idx = []
    for c in classes:
        idx_c = np.flatnonzero(cand_flat & (flat == c))
        n_c = min(class_budget(int((flat == c).sum()), n_samples), len(idx_c))
        train_idx.extend(pick.permutation(idx_c)[:n_c].tolist())
    train_idx = np.array(sorted(train_idx))
    test_idx = np.flatnonzero(test_mask.ravel())
    info = {
        "tries": f"{sub}/{n_valid}",          # chosen draw / number of valid draws
        "block": block,
        "guard": guard,
        "train_frac": train_frac,
        "n_classes": int(len(classes)),
        "n_test_classes": int(n_test_cls),
        "test_fraction_of_labelled": float(len(test_idx) / labelled.sum()),
    }
    return train_idx, test_idx, info


def min_distance(gt_shape, idx_a, idx_b):
    """Smallest Chebyshev distance between two pixel sets (for checking the guard)."""
    from scipy.ndimage import distance_transform_cdt
    H, W = gt_shape
    mask_b = np.ones(H * W, dtype=bool)
    mask_b[idx_b] = False
    dist = distance_transform_cdt(mask_b.reshape(H, W), metric="chessboard")
    return int(dist.ravel()[idx_a].min())
