import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

# ─── Dataset configurations ───────────────────────────────────────────────────
DATASETS = {
    'indian_pines': {
        'data_path': 'data/indian_pines/Indian_pines_corrected.mat',
        'gt_path':   'data/indian_pines/Indian_pines_gt.mat',
        'data_key':  'indian_pines_corrected',
        'gt_key':    'indian_pines_gt',
        'n_classes': 16,
        'n_bands':   200,
    },
    'pavia_u': {
        'data_path': 'data/pavia_u/PaviaU.mat',
        'gt_path':   'data/pavia_u/PaviaU_gt.mat',
        'data_key':  'paviaU',
        'gt_key':    'paviaU_gt',
        'n_classes': 9,
        'n_bands':   103,
    },
    'salinas': {
        'data_path': 'data/salinas/Salinas_corrected.mat',
        'gt_path':   'data/salinas/Salinas_gt.mat',
        'data_key':  'salinas_corrected',
        'gt_key':    'salinas_gt',
        'n_classes': 16,
        'n_bands':   204,
    },
}

# ─── Load raw data ─────────────────────────────────────────────────────────────
def load_dataset(name):
    cfg = DATASETS[name]
    data = sio.loadmat(cfg['data_path'])[cfg['data_key']].astype(np.float32)
    gt   = sio.loadmat(cfg['gt_path'])[cfg['gt_key']].astype(np.int64)
    return data, gt

# ─── Stratified few-shot split ─────────────────────────────────────────────────
def stratified_split(gt, n_samples=30, seed=42):
    """
    For each class: take min(n_samples, min(15, n//2)) training samples.
    Returns train_indices, test_indices (1-D arrays of flat pixel indices).
    Background class 0 is excluded.
    """
    np.random.seed(seed)
    train_idx, test_idx = [], []
    flat_gt = gt.flatten()

    for cls in range(1, gt.max() + 1):
        cls_idx = np.where(flat_gt == cls)[0]
        n = len(cls_idx)
        if n == 0:
            continue
        # fallback for rare classes
        n_train = n_samples if n >= 2 * n_samples else min(15, n // 2)
        n_train = max(1, n_train)  # at least 1 sample

        np.random.shuffle(cls_idx)
        train_idx.extend(cls_idx[:n_train].tolist())
        test_idx.extend(cls_idx[n_train:].tolist())

    return np.array(train_idx), np.array(test_idx)

# ─── Normalisation ─────────────────────────────────────────────────────────────
def normalize(data, train_idx):
    """
    Z-score normalization per band.
    Statistics computed on training pixels only.
    data shape: (H, W, C)
    """
    H, W, C = data.shape
    flat = data.reshape(-1, C)          # (N, C)
    scaler = StandardScaler()
    scaler.fit(flat[train_idx])         # fit on train pixels only
    flat_norm = scaler.transform(flat)
    return flat_norm.reshape(H, W, C), scaler

# ─── PCA reduction ─────────────────────────────────────────────────────────────
def apply_pca(data, train_idx, n_components):
    """
    PCA dimensionality reduction.
    data shape: (H, W, C) → returns (H, W, n_components)
    """
    H, W, C = data.shape
    flat = data.reshape(-1, C)
    pca = PCA(n_components=n_components)
    pca.fit(flat[train_idx])
    flat_pca = pca.transform(flat)
    return flat_pca.reshape(H, W, n_components), pca

# ─── Patch extraction ──────────────────────────────────────────────────────────
def extract_patches(data, indices, patch_size):
    """
    Extract square patches centered on each pixel index.
    data: (H, W, C)
    indices: flat pixel indices
    Returns: (N, C, patch_size, patch_size)
    """
    H, W, C = data.shape
    pad = patch_size // 2
    padded = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    patches = []
    for idx in indices:
        r, c = divmod(idx, W)
        patch = padded[r:r+patch_size, c:c+patch_size, :]  # (P, P, C)
        patches.append(patch.transpose(2, 0, 1))            # (C, P, P)
    return np.array(patches, dtype=np.float32)

# ─── PyTorch Dataset ───────────────────────────────────────────────────────────
class HSIDataset(Dataset):
    def __init__(self, patches, labels):
        self.patches = torch.from_numpy(patches)   # (N, C, P, P)
        self.labels  = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]

# ─── Main loader function ──────────────────────────────────────────────────────
def get_loaders(dataset_name, patch_size, n_pca, batch_size, seed,
                n_train_samples=30, val_fraction=0.1):
    """
    Full pipeline: load → normalize → PCA → split → patch → DataLoader.

    Returns:
        train_loader, val_loader, test_loader, meta (dict with dataset info)
    """
    data, gt = load_dataset(dataset_name)
    flat_gt  = gt.flatten()

    # Split indices (before normalization to avoid leakage)
    train_idx, test_idx = stratified_split(gt, n_samples=n_train_samples, seed=seed)

   # Carve out validation from training set
    train_labels_for_split = flat_gt[train_idx]
    n_classes_present = len(np.unique(train_labels_for_split))
    min_val_size = n_classes_present  # need at least one per class

    val_size = max(min_val_size, int(len(train_idx) * val_fraction))

    if val_size < len(train_idx) and len(train_idx) - val_size >= n_classes_present:
        try:
            train_idx, val_idx = train_test_split(
                train_idx,
                test_size=val_size,
                stratify=train_labels_for_split,
                random_state=seed
            )
        except ValueError:
            # fallback if stratification still fails
            np.random.seed(seed)
            np.random.shuffle(train_idx)
            val_idx   = train_idx[:val_size]
            train_idx = train_idx[val_size:]
    else:
        # too few samples to split — use test set as validation
        val_idx = test_idx[:len(test_idx)//10]

    # Normalize using training pixels only
    data_norm, scaler = normalize(data, train_idx)

    # PCA if requested
    if n_pca is not None and n_pca < data_norm.shape[2]:
        data_proc, pca = apply_pca(data_norm, train_idx, n_pca)
        n_bands_final = n_pca
    else:
        data_proc = data_norm
        pca = None
        n_bands_final = data_norm.shape[2]

    # Extract patches
    train_patches = extract_patches(data_proc, train_idx, patch_size)
    val_patches   = extract_patches(data_proc, val_idx,   patch_size)
    test_patches  = extract_patches(data_proc, test_idx,  patch_size)

    # Labels (0-indexed for CrossEntropyLoss)
    train_labels = flat_gt[train_idx] - 1
    val_labels   = flat_gt[val_idx]   - 1
    test_labels  = flat_gt[test_idx]  - 1

    # DataLoaders
    train_loader = DataLoader(HSIDataset(train_patches, train_labels),
                              batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(HSIDataset(val_patches,   val_labels),
                              batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(HSIDataset(test_patches,  test_labels),
                              batch_size=batch_size, shuffle=False, num_workers=0)

    meta = {
        'dataset':      dataset_name,
        'n_classes':    DATASETS[dataset_name]['n_classes'],
        'n_bands':      n_bands_final,
        'patch_size':   patch_size,
        'n_train':      len(train_idx),
        'n_val':        len(val_idx),
        'n_test':       len(test_idx),
        'seed':         seed,
        'H':            gt.shape[0],
        'W':            gt.shape[1],
        'gt':           gt,
        'data_proc':    data_proc,
    }

    return train_loader, val_loader, test_loader, meta


# ─── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os
    os.chdir('/root/hsi_benchmark')

    for name in ['indian_pines', 'pavia_u', 'salinas']:
        print(f'\n--- {name} ---')
        train_loader, val_loader, test_loader, meta = get_loaders(
            dataset_name   = name,
            patch_size     = 13,
            n_pca          = 30,
            batch_size     = 64,
            seed           = 42,
            n_train_samples= 30,
        )
        print(f"  Shape:   {meta['H']}x{meta['W']}, {meta['n_bands']} bands after PCA")
        print(f"  Classes: {meta['n_classes']}")
        print(f"  Train:   {meta['n_train']} samples")
        print(f"  Val:     {meta['n_val']} samples")
        print(f"  Test:    {meta['n_test']} samples")
        batch = next(iter(train_loader))
        print(f"  Batch:   patches={batch[0].shape}, labels={batch[1].shape}")

    print('\ndata_loader.py OK')