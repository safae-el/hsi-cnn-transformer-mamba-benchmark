import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from utils.data_loader import get_loaders
from utils.metrics import aggregate_runs, print_metrics
from utils.trainer import train_model, run_label_scarcity_experiment, set_seed
from models.hybridsn import HybridSN
from models.spectralformer import SpectralFormer
try:
    from models.s2mamba import S2Mamba  # adapted official code; see README
except ImportError:
    S2Mamba = None


# ─── Configuration ─────────────────────────────────────────────────────────────

SEEDS    = [42, 123, 456]
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASETS = ['indian_pines', 'pavia_u', 'salinas']

TRAIN_CONFIG = {
    'epochs':          300,
    'lr':              5e-4,
    'weight_decay':    1e-4,
    'patience':        300,
    'batch_size':      64,
    'n_pca':           30,
    'n_train_samples': 30,
}

MODEL_CONFIGS = {
    'HybridSN': {
        'patch_size': 13,
        'n_pca':      30,
    },
    'SpectralFormer': {
        'patch_size': 13,
        'n_pca':      30,
    },
    'S2Mamba': {
        'patch_size': 13,
        'n_pca':      30,
    },
}

SCARCITY_SAMPLES = [5, 10, 20, 30]


# ─── Model factory ─────────────────────────────────────────────────────────────

def get_model(model_name, n_bands, n_classes, patch_size):
    if model_name == 'HybridSN':
        return HybridSN(
            n_bands=n_bands,
            n_classes=n_classes,
            patch_size=patch_size
        )
    elif model_name == 'SpectralFormer':
        return SpectralFormer(
            n_bands=n_bands,
            n_classes=n_classes,
            patch_size=patch_size,
            d_model=64,
            n_heads=4,
            n_layers=5,
            band_patches=3
        )
    elif model_name == 'S2Mamba':
        return S2Mamba(
            n_bands=n_bands,
            n_classes=n_classes,
            patch_size=patch_size,
            depths=[1],
            dims=[64],
            d_state=16,
            attn_drop_rate=0.1,
            drop_path_rate=0.1
        )
    else:
        raise ValueError(f'Unknown model: {model_name}')


# ─── Save utilities ────────────────────────────────────────────────────────────

def save_results(results, path):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj
    with open(path, 'w') as f:
        json.dump(results, f, default=convert, indent=2)
    print(f"  Saved: {path}")


def build_summary_table(all_results):
    rows = []
    for dataset, models in all_results['main'].items():
        for model_name, agg in models.items():
            row = {
                'Dataset':    dataset,
                'Model':      model_name,
                'OA':         f"{agg.get('OA_mean',0)*100:.2f}±{agg.get('OA_std',0)*100:.2f}",
                'AA':         f"{agg.get('AA_mean',0)*100:.2f}±{agg.get('AA_std',0)*100:.2f}",
                'Kappa':      f"{agg.get('Kappa_mean',0):.4f}±{agg.get('Kappa_std',0):.4f}",
                'MacroF1':    f"{agg.get('MacroF1_mean',0)*100:.2f}±{agg.get('MacroF1_std',0)*100:.2f}",
                'mIoU':       f"{agg.get('mIoU_mean',0)*100:.2f}±{agg.get('mIoU_std',0)*100:.2f}",
                'Params_K':   f"{agg.get('Params_K_mean',0):.1f}",
                'FLOPs_M':    f"{agg.get('FLOPs_M_mean',0):.2f}",
                'GPU_MB':     f"{agg.get('GPU_MB_mean',0):.1f}",
                'Train_s/ep': f"{agg.get('TrainTime_per_epoch_s_mean',0):.2f}",
                'Infer_ms':   f"{agg.get('InferenceTime_ms_mean',0):.2f}",
            }
            rows.append(row)
    return pd.DataFrame(rows)


# ─── Main benchmark ────────────────────────────────────────────────────────────

def run_main_benchmark():
    print("\n" + "="*60)
    print("MAIN BENCHMARK")
    print("="*60)

    all_results = {'main': {}, 'scarcity': {}}
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')

    for dataset in DATASETS:
        print(f"\n{'─'*50}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'─'*50}")
        all_results['main'][dataset] = {}

        for model_name in MODEL_CONFIGS:
            cfg = MODEL_CONFIGS[model_name]
            print(f"\n  Model: {model_name}")
            seed_results = []

            for seed in SEEDS:
                print(f"    Seed {seed}...")
                set_seed(seed)

                train_loader, val_loader, test_loader, meta = get_loaders(
                    dataset_name    = dataset,
                    patch_size      = cfg['patch_size'],
                    n_pca           = cfg['n_pca'],
                    batch_size      = TRAIN_CONFIG['batch_size'],
                    seed            = seed,
                    n_train_samples = TRAIN_CONFIG['n_train_samples'],
                )

                model = get_model(
                    model_name,
                    n_bands    = meta['n_bands'],
                    n_classes  = meta['n_classes'],
                    patch_size = meta['patch_size'],
                ).to(DEVICE)

                train_cfg = {
                    **TRAIN_CONFIG,
                    'patch_size': cfg['patch_size'],
                    'seed':       seed,
                }

                result = train_model(
                    model, train_loader, val_loader, test_loader,
                    meta['n_classes'], DEVICE, train_cfg
                )
                seed_results.append(result)
                print(f"      OA={result['OA']*100:.2f}% "
                      f"AA={result['AA']*100:.2f}% "
                      f"Kappa={result['Kappa']:.4f}")

            agg = aggregate_runs(seed_results)
            all_results['main'][dataset][model_name] = agg

            print(f"\n  [{model_name} on {dataset}] FINAL:")
            print_metrics(agg, prefix='    ')

        save_results(
            all_results,
            f'results/benchmark_{timestamp}_partial.json'
        )

    return all_results, timestamp


# ─── Scarcity benchmark ────────────────────────────────────────────────────────

def run_scarcity_benchmark(all_results, timestamp):
    print("\n" + "="*60)
    print("LABEL SCARCITY EXPERIMENT")
    print("="*60)

    dataset = 'indian_pines'
    all_results['scarcity'][dataset] = {}

    for model_name in MODEL_CONFIGS:
        cfg = MODEL_CONFIGS[model_name]
        print(f"\n  Model: {model_name}")

        def model_fn(n_bands, n_classes, patch_size):
            return get_model(model_name, n_bands, n_classes, patch_size)

        scarcity_cfg = {
            **TRAIN_CONFIG,
            'patch_size': cfg['patch_size'],
        }

        results = run_label_scarcity_experiment(
            model_fn     = model_fn,
            dataset_name = dataset,
            device       = DEVICE,
            config       = scarcity_cfg,
            seeds        = SEEDS,
            sample_sizes = SCARCITY_SAMPLES,
        )
        all_results['scarcity'][dataset][model_name] = results

        print(f"\n  Scarcity curve for {model_name} on {dataset}:")
        for n_samples, agg in results.items():
            oa  = agg.get('OA_mean', 0) * 100
            std = agg.get('OA_std',  0) * 100
            print(f"    {n_samples:2d} samples/class: OA={oa:.2f}±{std:.2f}%")

    save_results(
        all_results,
        f'results/benchmark_{timestamp}_partial.json'
    )
    return all_results


# ─── Save final ────────────────────────────────────────────────────────────────

def save_final(all_results, timestamp):
    json_path = f'results/benchmark_{timestamp}_final.json'
    save_results(all_results, json_path)

    df = build_summary_table(all_results)
    csv_path = f'results/benchmark_{timestamp}_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    print("\n" + "="*60)
    print("FINAL RESULTS TABLE")
    print("="*60)
    print(df.to_string(index=False))


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"Device:   {DEVICE}")
    print(f"Seeds:    {SEEDS}")
    print(f"Datasets: {DATASETS}")
    print(f"Models:   {list(MODEL_CONFIGS.keys())}")

    t_start = time.perf_counter()

    all_results, timestamp = run_main_benchmark()
    all_results = run_scarcity_benchmark(all_results, timestamp)
    save_final(all_results, timestamp)

    total_time = time.perf_counter() - t_start
    print(f"\nTotal benchmark time: {total_time/3600:.2f} hours")
    print("DONE")