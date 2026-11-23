import time
import numpy as np
import torch
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    confusion_matrix,
)


# ─── Core classification metrics ──────────────────────────────────────────────

def overall_accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))


def average_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)
    per_class = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            continue
        per_class.append(np.mean(y_pred[mask] == c))
    return np.mean(per_class)


def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def per_class_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None, zero_division=0)


# ─── mIoU ─────────────────────────────────────────────────────────────────────

def mean_iou(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)
    iou_list = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = tp + fp + fn
        iou_list.append(tp / denom if denom > 0 else 0.0)
    return np.mean(iou_list)


def per_class_iou(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)
    iou_dict = {}
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = tp + fp + fn
        iou_dict[int(c)] = tp / denom if denom > 0 else 0.0
    return iou_dict


# ─── Scalability metrics ───────────────────────────────────────────────────────

def count_parameters(model):
    """Total trainable parameters in thousands."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e3  # in K


def measure_gpu_memory(model, sample_input):
    """
    Peak GPU memory in MB during a single forward pass.
    sample_input: a tensor already on CUDA.
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with torch.no_grad():
        _ = model(sample_input)
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    return peak_mb


def measure_inference_time(model, sample_input, n_runs=50):
    """
    Average inference time in milliseconds over n_runs forward passes.
    sample_input: a tensor already on CUDA.
    """
    model.eval()
    # warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(sample_input)
    torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(sample_input)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / n_runs * 1000  # ms


def estimate_flops(model, sample_input):
    """
    Estimate FLOPs using torch hooks.
    Returns FLOPs in millions (MFLOPs).
    Only counts Conv2d, Conv3d, and Linear layers.
    """
    flops = [0]

    def conv2d_hook(module, inp, out):
        in_c  = module.in_channels
        k     = module.kernel_size
        out_h, out_w = out.shape[2], out.shape[3]
        flops[0] += 2 * in_c * k[0] * k[1] * out_h * out_w * out.shape[1]

    def conv3d_hook(module, inp, out):
        in_c  = module.in_channels
        k     = module.kernel_size
        od, oh, ow = out.shape[2], out.shape[3], out.shape[4]
        flops[0] += 2 * in_c * k[0] * k[1] * k[2] * od * oh * ow * out.shape[1]

    def linear_hook(module, inp, out):
        flops[0] += 2 * module.in_features * module.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv2d_hook))
        elif isinstance(m, torch.nn.Conv3d):
            hooks.append(m.register_forward_hook(conv3d_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        model(sample_input)

    for h in hooks:
        h.remove()

    return flops[0] / 1e6  # MFLOPs


# ─── Aggregate across seeds ────────────────────────────────────────────────────

def aggregate_runs(results_list):
    """
    Aggregate metrics across multiple seed runs.
    results_list: list of dicts.
    Returns dict with mean and std for scalar metrics.
    """
    if not results_list:
        return {}

    scalar_keys = ['OA', 'AA', 'Kappa', 'MacroF1', 'mIoU']
    aggregated = {}
    for k in scalar_keys:
        if k in results_list[0]:
            vals = [r[k] for r in results_list]
            aggregated[f'{k}_mean'] = np.mean(vals)
            aggregated[f'{k}_std']  = np.std(vals)

    # Per-class F1 and IoU: average across seeds
    if 'PerClassF1' in results_list[0]:
        stacked = np.stack([r['PerClassF1'] for r in results_list])
        aggregated['PerClassF1_mean'] = stacked.mean(axis=0)
        aggregated['PerClassF1_std']  = stacked.std(axis=0)

    # Scalability metrics: take mean
    for k in ['Params_K', 'FLOPs_M', 'GPU_MB',
              'TrainTime_per_epoch_s', 'InferenceTime_ms']:
        if k in results_list[0]:
            vals = [r[k] for r in results_list]
            aggregated[f'{k}_mean'] = np.mean(vals)

    return aggregated


# ─── Compute all metrics from predictions ─────────────────────────────────────

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        'OA':         overall_accuracy(y_true, y_pred),
        'AA':         average_accuracy(y_true, y_pred),
        'Kappa':      kappa(y_true, y_pred),
        'MacroF1':    macro_f1(y_true, y_pred),
        'mIoU':       mean_iou(y_true, y_pred),
        'PerClassF1': per_class_f1(y_true, y_pred),
        'PerClassIoU':per_class_iou(y_true, y_pred),
    }


# ─── Pretty print ─────────────────────────────────────────────────────────────

def print_metrics(metrics, prefix=''):
    if 'OA_mean' in metrics:
        print(f"{prefix}OA:       {metrics['OA_mean']*100:.2f} ± {metrics['OA_std']*100:.2f}%")
        print(f"{prefix}AA:       {metrics['AA_mean']*100:.2f} ± {metrics['AA_std']*100:.2f}%")
        print(f"{prefix}Kappa:    {metrics['Kappa_mean']:.4f} ± {metrics['Kappa_std']:.4f}")
        print(f"{prefix}MacroF1:  {metrics['MacroF1_mean']*100:.2f} ± {metrics['MacroF1_std']*100:.2f}%")
        print(f"{prefix}mIoU:     {metrics['mIoU_mean']*100:.2f} ± {metrics['mIoU_std']*100:.2f}%")
        if 'Params_K_mean' in metrics:
            print(f"{prefix}Params:   {metrics['Params_K_mean']:.1f} K")
            print(f"{prefix}FLOPs:    {metrics['FLOPs_M_mean']:.2f} MFLOPs")
            print(f"{prefix}GPU Mem:  {metrics['GPU_MB_mean']:.1f} MB")
            print(f"{prefix}Train/ep: {metrics['TrainTime_per_epoch_s_mean']:.2f} s")
            print(f"{prefix}Infer:    {metrics['InferenceTime_ms_mean']:.2f} ms")
    else:
        print(f"{prefix}OA:       {metrics['OA']*100:.2f}%")
        print(f"{prefix}AA:       {metrics['AA']*100:.2f}%")
        print(f"{prefix}Kappa:    {metrics['Kappa']:.4f}")
        print(f"{prefix}MacroF1:  {metrics['MacroF1']*100:.2f}%")
        print(f"{prefix}mIoU:     {metrics['mIoU']*100:.2f}%")


# ─── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(42)
    y_true = np.random.randint(0, 5, 200)
    y_pred = np.random.randint(0, 5, 200)

    m = compute_metrics(y_true, y_pred)
    print('Single run:')
    print_metrics(m)

    print('\nPer-class F1:', np.round(m['PerClassF1'], 3))
    print('Per-class IoU:', {k: round(v, 3) for k, v in m['PerClassIoU'].items()})

    results = [
        compute_metrics(
            np.random.randint(0, 5, 200),
            np.random.randint(0, 5, 200)
        )
        for _ in range(3)
    ]
    agg = aggregate_runs(results)
    print('\nAggregated across 3 seeds:')
    print_metrics(agg)
    print('\nmetrics.py OK')