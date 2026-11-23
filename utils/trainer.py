import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils.metrics import compute_metrics, measure_gpu_memory, measure_inference_time, estimate_flops, count_parameters


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for patches, labels in loader:
        patches = patches.to(device)
        labels  = labels.to(device)
        optimizer.zero_grad()
        out  = model(patches)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * patches.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for patches, labels in loader:
            patches = patches.to(device)
            out     = model(patches)
            preds   = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


def train_model(model, train_loader, val_loader, test_loader,
                n_classes, device, config):
    """
    Full training loop with early stopping.

    config keys:
        epochs, lr, weight_decay, patience, seed
    """
    set_seed(config['seed'])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )

    best_val_oa  = 0.0
    best_state   = None
    patience_ctr = 0
    epoch_times  = []

    for epoch in range(1, config['epochs'] + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        epoch_times.append(time.perf_counter() - t0)

        # Validation
        val_labels, val_preds = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_labels, val_preds)
        val_oa = val_metrics['OA']

        scheduler.step()

        # Early stopping
        if val_oa > best_val_oa:
            best_val_oa = val_oa
            best_state  = {k: v.cpu().clone()
                           for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{config['epochs']} "
                  f"loss={train_loss:.4f} "
                  f"val_OA={val_oa*100:.2f}% "
                  f"best={best_val_oa*100:.2f}%")

        if patience_ctr >= config['patience']:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(
            {k: v.to(device) for k, v in best_state.items()}
        )

    # Test evaluation
    test_labels, test_preds = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(test_labels, test_preds)

    # Scalability metrics
    sample_input = next(iter(test_loader))[0][:1].to(device)

    params_k       = count_parameters(model)
    flops_m        = estimate_flops(model, sample_input)
    gpu_mb         = measure_gpu_memory(model, sample_input)
    inference_ms   = measure_inference_time(model, sample_input)
    train_time_s   = np.mean(epoch_times)

    result = {
        **test_metrics,
        'Params_K':               params_k,
        'FLOPs_M':                flops_m,
        'GPU_MB':                 gpu_mb,
        'TrainTime_per_epoch_s':  train_time_s,
        'InferenceTime_ms':       inference_ms,
        'best_val_OA':            best_val_oa,
        'epochs_trained':         epoch,
    }

    return result


def run_label_scarcity_experiment(model_fn, dataset_name, device,
                                  config, seeds,
                                  sample_sizes=(5, 10, 20, 30)):
    """
    Train each model at different label densities.
    model_fn: callable that returns a fresh model instance.
    Returns dict: {n_samples: aggregated_metrics}
    """
    from utils.data_loader import get_loaders
    from utils.metrics import aggregate_runs

    results_by_size = {}

    for n_samples in sample_sizes:
        print(f"\n  Label scarcity: {n_samples} samples/class")
        seed_results = []

        for seed in seeds:
            train_loader, val_loader, test_loader, meta = get_loaders(
                dataset_name    = dataset_name,
                patch_size      = config['patch_size'],
                n_pca           = config['n_pca'],
                batch_size      = config['batch_size'],
                seed            = seed,
                n_train_samples = n_samples,
            )

            model = model_fn(
                n_bands   = meta['n_bands'],
                n_classes = meta['n_classes'],
                patch_size= meta['patch_size'],
            ).to(device)

            cfg = {**config, 'seed': seed}
            result = train_model(
                model, train_loader, val_loader, test_loader,
                meta['n_classes'], device, cfg
            )
            seed_results.append(result)
            print(f"    seed={seed} OA={result['OA']*100:.2f}%")

        results_by_size[n_samples] = aggregate_runs(seed_results)

    return results_by_size

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.metrics import compute_metrics, measure_gpu_memory, measure_inference_time, estimate_flops, count_parameters
    print("trainer.py loaded OK")
    print("Functions available:")
    print("  train_model()")
    print("  run_label_scarcity_experiment()")
    print("  evaluate()")
    print("  set_seed()")