import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import torch
import numpy as np
from datetime import datetime

from utils.metrics import aggregate_runs, print_metrics
from utils.trainer import run_label_scarcity_experiment, set_seed
from run_benchmark import get_model, MODEL_CONFIGS, TRAIN_CONFIG, DATASETS, SEEDS, DEVICE, save_results

SCARCITY_SAMPLES = [5, 10, 20, 30]

def run_scarcity_only():
    print("\n" + "="*60)
    print("LABEL SCARCITY EXPERIMENT ONLY")
    print("="*60)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = {'scarcity': {'indian_pines': {}}}
    dataset = 'indian_pines'

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

    save_results(all_results, f'results/scarcity_{timestamp}.json')
    print("\nScarcity experiment DONE")

if __name__ == '__main__':
    run_scarcity_only()