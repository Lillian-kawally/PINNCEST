import argparse
import copy
import os
import json
import traceback
from datetime import datetime

import torch
import pandas as pd

from train import main as train_main
from test import main as test_main

# Common overrides applied to every ablation experiment.
# Match these to whatever you trained your baseline with.
COMMON_OVERRIDES = {
    'dropout': 0.2,
    'z_loss_weight': 30.0,
    'fit_range': [-5, 5],
}

# Ablation configurations
ABLATION_CONFIGS = {
    'baseline':         {},                                  # full model
    'single_b1_high':   {'single_b1_idx': -1},
    'single_b1_low':    {'single_b1_idx': 0},
    'no_cross_attn':    {'use_cross_attn': False},
    'no_b1_emb':        {'use_b1_emb': False},
    'no_pos_emb':       {'use_pos_emb': False},
    'shared_head':      {'decouple_heads': False},
    'no_physics':       {'use_physics': False},
    # 'skip_bmlayer':   {'skip_bmlayer': True},
}

# Default seeds
DEFAULT_SEEDS = [42, 123, 2024]


def run_one_experiment(base_args, exp_name, override_dict, seed):
    seed_tag = f"s{seed}"
    trial_name = f"{exp_name}_{seed_tag}"

    overrides = dict(COMMON_OVERRIDES)
    overrides.update(override_dict)
    overrides['trial_name'] = trial_name
    overrides['seed'] = seed

    # Sync args.use_physics with override
    # The model is built from override_params, but train.py / test.py also
    # read args.use_physics for loss weighting and other flags. Without this
    # sync, args still says True while the model is built with False.
    local_args = copy.deepcopy(base_args)
    if 'use_physics' in overrides:
        local_args.use_physics = bool(overrides['use_physics'])

    print(f"\n{'#' * 80}")
    print(f"# Ablation: {exp_name}  |  seed: {seed}  |  trial: {trial_name}")
    print(f"# Overrides: {overrides}")
    print(f"# args.use_physics = {local_args.use_physics}")
    print(f"{'#' * 80}\n")

    ckpt_path = os.path.join('checkpoints', local_args.model, trial_name, 'final.pth')

    # Train
    if local_args.skip_existing and os.path.exists(ckpt_path):
        print(f"[Skip Train] checkpoint already exists: {ckpt_path}")
    else:
        try:
            train_main(local_args, override_params=overrides)
        except Exception as e:
            print(f"[Training Error] {trial_name}: {e}")
            traceback.print_exc()
            return trial_name, None
        finally:
            torch.cuda.empty_cache()

    # Test
    if not os.path.exists(ckpt_path):
        print(f"[Test Skipped] checkpoint not found: {ckpt_path}")
        return trial_name, None

    test_args = argparse.Namespace(
        datapath=local_args.datapath,
        k_mode=local_args.k_mode,
        model=local_args.model,
        use_physics=local_args.use_physics,
        checkpoint=ckpt_path,
        snr=local_args.snr,
    )

    try:
        all_metrics = test_main(test_args, override_params=overrides)
    except Exception as e:
        print(f"[Test Error] {trial_name}: {e}")
        traceback.print_exc()
        return trial_name, None
    finally:
        torch.cuda.empty_cache()

    return trial_name, all_metrics


def metrics_to_row(exp_name, trial_name, seed, metrics):
    row = {'experiment': exp_name, 'trial_name': trial_name, 'seed': seed}
    if metrics is None:
        return row
    for pool, m in metrics.items():
        if pool == 'extra_metrics':
            continue
        for var in ('f', 'k'):
            if var in m:
                for metric_key in ('r', 'rmse', 'nrmse'):
                    if metric_key in m[var]:
                        row[f"{pool}_{var}_{metric_key}"] = m[var][metric_key]
    return row


def aggregate_seeds(rows):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns
                   if c not in ('experiment', 'trial_name', 'seed')]
    # Handle the case where no successful runs have produced metrics yet
    if not metric_cols:
        return pd.DataFrame({'experiment': df['experiment'].unique(),
                             'n_seeds': 0})
    g = df.groupby('experiment')[metric_cols]
    mean_df = g.mean().add_suffix('_mean')
    std_df = g.std(ddof=1).add_suffix('_std')
    n_df = g.count().iloc[:, [0]].rename(columns={metric_cols[0]: 'n_seeds'})
    out = pd.concat([n_df, mean_df, std_df], axis=1).reset_index()
    return out


def run_experiments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--model', type=str, default='PINN')
    parser.add_argument('--k_mode', type=str, default='log')
    parser.add_argument('--opt', type=str, default='adamw')
    parser.add_argument('--snr', type=float, default=60)
    parser.add_argument('--use_physics', action='store_true', default=True)
    parser.add_argument('--no_physics', action='store_false', dest='use_physics')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS,
                        help='List of training seeds to use')
    parser.add_argument('--only', type=str, default=None,
                        help='Comma-separated experiment names to run')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip training if checkpoint already exists')
    base_args = parser.parse_args()

    if base_args.only:
        wanted = [s.strip() for s in base_args.only.split(',')]
        configs = {k: v for k, v in ABLATION_CONFIGS.items() if k in wanted}
        if not configs:
            raise ValueError(f"None of {wanted} found in ABLATION_CONFIGS")
    else:
        configs = ABLATION_CONFIGS

    summary_dir = os.path.join('ablation_summary', base_args.model)
    os.makedirs(summary_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
    raw_csv = os.path.join(summary_dir, f'ablation_raw_{timestamp}.csv')
    agg_csv = os.path.join(summary_dir, f'ablation_agg_{timestamp}.csv')
    json_path = os.path.join(summary_dir, f'ablation_{timestamp}.json')

    rows = []
    raw_results = {}

    for exp_name, override in configs.items():
        for seed in base_args.seeds:
            trial_name, metrics = run_one_experiment(
                base_args, exp_name, override, seed
            )
            rows.append(metrics_to_row(exp_name, trial_name, seed, metrics))
            raw_results[trial_name] = metrics if metrics is not None else "FAILED"

            # Save progress after every run
            pd.DataFrame(rows).to_csv(raw_csv, index=False)
            with open(json_path, 'w') as f:
                json.dump(raw_results, f, indent=2, default=str)
            agg_df = aggregate_seeds(rows)
            agg_df.to_csv(agg_csv, index=False)

    print(f"\n{'=' * 80}")
    print("All ablation experiments finished.")
    print(f"Raw per-seed CSV : {raw_csv}")
    print(f"Aggregated CSV   : {agg_csv}")
    print(f"JSON dump        : {json_path}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    run_experiments()