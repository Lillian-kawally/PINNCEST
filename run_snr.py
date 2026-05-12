"""
run_snr_eval.py
---------------
Evaluate the trained baseline model across multiple SNR levels, with
multiple noise realizations per SNR to enable mean +/- s.d. plotting.

Reuses the trained baseline checkpoint(s) from run_ablation.py
(e.g., checkpoints/PINNCEST/baseline_s42/final.pth).

Usage:
    python run_snr_eval.py \
        --datapath /your/data \
        --model PINNCEST \
        --train_seeds 42 123 2024 \
        --snr_list 20 30 40 50 60 80 100 \
        --noise_repeats 5
"""

import argparse
import os
import json
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from test import main as test_main


# Match the training override (so the model is constructed identically)
BASELINE_OVERRIDES = {
    'dropout': 0.2,
    'z_loss_weight': 30.0,
    'fit_range': [-5, 5],
}


def metrics_to_long_rows(metrics, train_seed, snr, noise_seed):
    rows = []
    if metrics is None:
        return rows
    for pool, m in metrics.items():
        if pool == 'extra_metrics':
            continue
        for var in ('f', 'k'):
            if var not in m:
                continue
            for metric_key in ('r', 'rmse', 'nrmse'):
                if metric_key not in m[var]:
                    continue
                rows.append({
                    'train_seed': train_seed,
                    'snr': snr,
                    'noise_seed': noise_seed,
                    'pool': pool,
                    'var': var,
                    'metric': metric_key,
                    'value': m[var][metric_key],
                })
    return rows


def run_one_eval(args, train_seed, snr, noise_seed):
    trial_name = f"baseline_s{train_seed}"
    ckpt_path = os.path.join('checkpoints', args.model, trial_name, 'final.pth')

    if not os.path.exists(ckpt_path):
        print(f"[Skip] checkpoint not found: {ckpt_path}")
        return None

    overrides = dict(BASELINE_OVERRIDES)
    overrides['trial_name'] = f"snr_eval_s{train_seed}_snr{int(snr)}_n{noise_seed}"
    overrides['noise_seed'] = int(noise_seed)   # ← 关键：传给 test.py

    # torch.manual_seed(noise_seed)
    # np.random.seed(noise_seed)

    test_args = argparse.Namespace(
        datapath=args.datapath,
        k_mode=args.k_mode,
        model=args.model,
        use_physics=args.use_physics,
        checkpoint=ckpt_path,
        snr=snr,
    )

    try:
        metrics = test_main(test_args, override_params=overrides)
    except Exception as e:
        print(f"[Eval Error] s{train_seed} snr={snr} ns={noise_seed}: {e}")
        traceback.print_exc()
        return None
    finally:
        torch.cuda.empty_cache()

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--model', type=str, default='PINN')
    parser.add_argument('--k_mode', type=str, default='log')
    parser.add_argument('--use_physics', action='store_true', default=True)
    parser.add_argument('--no_physics', action='store_false', dest='use_physics')

    parser.add_argument('--train_seeds', type=int, nargs='+', default=[42, 123, 2024],
                        help='Training seeds (must match what run_ablation.py used)')
    parser.add_argument('--snr_list', type=float, nargs='+',
                        default=[20, 30, 40, 50, 60, 80, 100],
                        help='SNR levels (dB) to evaluate at')
    parser.add_argument('--noise_repeats', type=int, default=5,
                        help='Noise realizations per (train_seed, snr) pair')
    args = parser.parse_args()

    out_dir = os.path.join('snr_eval', args.model)
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
    raw_csv = os.path.join(out_dir, f'snr_baseline_raw_{timestamp}.csv')
    json_path = os.path.join(out_dir, f'snr_baseline_{timestamp}.json')

    all_rows = []
    raw_dump = {}

    total = (len(args.train_seeds) * len(args.snr_list) * args.noise_repeats)
    counter = 0

    for train_seed in args.train_seeds:
        for snr in args.snr_list:
            for ns in range(args.noise_repeats):
                counter += 1
                print(f"\n[{counter}/{total}] train_seed={train_seed} | "
                      f"snr={snr} | noise={ns}")
                metrics = run_one_eval(args, train_seed, snr, ns)
                rows = metrics_to_long_rows(metrics, train_seed, snr, ns)
                all_rows.extend(rows)
                raw_dump[f"s{train_seed}_snr{snr}_n{ns}"] = (
                    metrics if metrics is not None else "FAILED"
                )

                # Save progress incrementally
                pd.DataFrame(all_rows).to_csv(raw_csv, index=False)
                with open(json_path, 'w') as f:
                    json.dump(raw_dump, f, indent=2, default=str)

    print(f"\n{'=' * 80}")
    print(f"All SNR evaluations finished.")
    print(f"Raw CSV : {raw_csv}")
    print(f"JSON    : {json_path}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
