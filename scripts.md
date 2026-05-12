# Scripts Usage Guide

This document describes how to use the training, testing, ablation, SNR
robustness, and plotting scripts in the **PINNCEST** repository.

Before running any script:

- Install dependencies: `pip install -r requirements.txt`
- Prepare your dataset directory so that it contains a `params.yaml`, a
  `CEST_SOP_OFFS.mat` file, and `train/` / `test/` subdirectories (see
  `data.py` for the exact format expected).
- Place the model hyperparameter YAML at `params/PINNCEST.yaml`.

> Tip: every script accepts `--help` to print its full argument list,
> e.g. `python train.py --help`.

---

## Table of Contents

1. [Training](#1-training)
2. [Testing & Evaluation](#2-testing--evaluation)
3. [Ablation Study](#3-ablation-study)
4. [SNR Robustness Evaluation](#4-snr-robustness-evaluation)
5. [End-to-end Reproducibility Pipeline](#6-end-to-end-reproducibility-pipeline)
6. [Output Directory Layout](#7-output-directory-layout)
7. [Tips & Troubleshooting](#8-tips--troubleshooting)

---

## 1. Training

Train a single PINNCEST model from scratch.

```bash
python train.py \
    --datapath /path/to/data \
    --model PINNCEST \
    --k_mode log \
    --opt adamw
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--datapath` | str | *(required)* | Path to the dataset root |
| `--model` | str | *(required)* | Model class name (matches `models/PINNCEST.py`) |
| `--k_mode` | str | `log` | Label transformation for *k*: `log` / `loglog` / `linear` |
| `--opt` | str | `adamw` | Optimizer: `adamw`, `adam`, or `sgd` |
| `--no_physics` | flag | off | If set, disables the BM solver and physics-consistency loss |
| `--seed` | int | `42` | Random seed for reproducibility |

### Hyperparameters

All model and training hyperparameters (learning rate, batch size,
dropout, loss weights, fitting range, etc.) live in
`params/PINNCEST.yaml`. To override at runtime, edit the YAML file or
pass an `override_params` dict if calling `train.main()` directly from
Python.

### Outputs

- **Checkpoints**: `checkpoints/PINNCEST/<timestamp_or_trial_name>/ckp_<epoch>.pth`
  saved every 50 epochs, plus `final.pth` at the end.
- **TensorBoard logs**: `runs/PINNCEST/<timestamp_or_trial_name>/`.

Launch TensorBoard with:

```bash
tensorboard --logdir runs/
```

---

## 2. Testing & Evaluation

Run a trained model on the held-out test set, generate per-voxel
predictions, and automatically compute Pearson *r*, RMSE, and NRMSE for
each CEST pool.

```bash
python test.py \
    --datapath /path/to/data \
    --model PINNCEST \
    --k_mode log \
    --checkpoint checkpoints/PINNCEST/<run>/final.pth \
    --snr 60
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--datapath` | str | *(required)* | Path to the dataset root |
| `--model` | str | *(required)* | Model class name |
| `--checkpoint` | str | *(required)* | Path to trained `.pth` file |
| `--k_mode` | str | `log` | Must match what the model was trained with |
| `--snr` | float | `60` | Test-time SNR in dB; set very high (e.g. `120`) for near-noiseless evaluation |
| `--no_physics` | flag | off | Bypass BM solver during inference |

### Outputs

- **Predictions**: `results/PINNCEST/<trial_name>/Result.mat`
  contains the per-sample predicted `f` and `k` values for all pools,
  along with the test sample indices.
- **Evaluation plots**: density scatter plots with marginal histograms
  for each (pool, parameter), saved as `.svg` under
  `results/PINNCEST/<trial_name>/plots/`.
- **Metrics summary**: `results/PINNCEST/<trial_name>/data/metrics_summary.txt`.

The `trial_name` defaults to `default` if testing is invoked manually,
or is set automatically by `run_ablation.py` / `run_snr_eval.py` when
run in batch.

---

## 3. Ablation Study

Sweep over a set of pre-defined architectural ablations, optionally
across multiple training seeds, and auto-aggregate the results.

```bash
python run_ablation.py \
    --datapath /path/to/data \
    --model PINNCEST \
    --opt adamw
```

By default this trains and evaluates **all configurations** in
`ABLATION_CONFIGS` (defined at the top of `run_ablation.py`) across
three seeds (`42, 123, 2024`).

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--datapath` | str | *(required)* | Path to the dataset root |
| `--model` | str | `PINNCEST` | Model class name |
| `--k_mode` | str | `log` | Label transformation for *k* |
| `--opt` | str | `adamw` | Optimizer |
| `--snr` | float | `60` | Test-time SNR for evaluation |
| `--seeds` | int+ | `42 123 2024` | One or more training seeds (space-separated) |
| `--only` | str | *(all)* | Comma-separated subset of experiments to run |
| `--skip_existing` | flag | off | Skip training when a checkpoint already exists |

### Available ablation configurations

| Name | Toggle | What it tests |
|------|--------|---------------|
| `baseline` | (none) | Full PINNCEST model — reference |
| `single_b1_high` | `single_b1_idx=-1` | Highest single B<sub>1</sub> input only |
| `single_b1_low` | `single_b1_idx=0` | Lowest single B<sub>1</sub> input only |
| `no_cross_attn` | `use_cross_attn=False` | Mean pooling across B<sub>1</sub> |
| `no_b1_emb` | `use_b1_emb=False` | Remove B<sub>1</sub> token |
| `no_pos_emb` | `use_pos_emb=False` | Remove positional embedding |
| `shared_head` | `decouple_heads=False` | Single shared regression head for all pools |

### Examples

**Quick smoke test** (single seed, two configs):
```bash
python run_ablation.py \
    --datapath /path/to/data \
    --model PINNCEST \
    --only baseline,shared_head \
    --seeds 42
```

**Resume an interrupted full sweep** (reuses existing checkpoints):
```bash
python run_ablation.py \
    --datapath /path/to/data \
    --model PINNCEST \
    --skip_existing
```

**Custom seeds** for variance estimation:
```bash
python run_ablation.py \
    --datapath /path/to/data \
    --model PINNCEST \
    --seeds 42 123 2024 7 999
```

### Outputs

- `ablation_summary/PINNCEST/ablation_raw_<timestamp>.csv` — one row per
  `(experiment, seed)` with all metrics
- `ablation_summary/PINNCEST/ablation_agg_<timestamp>.csv` — aggregated
  mean / std across seeds for each experiment
- `ablation_summary/PINNCEST/ablation_<timestamp>.json` — full nested
  metrics dump

---

## 4. SNR Robustness Evaluation

Evaluate trained baseline models across multiple SNR levels with
multiple noise realizations per level. Reuses checkpoints produced by
`run_ablation.py` (e.g., `baseline_s42`, `baseline_s123`, `baseline_s2024`).

```bash
python run_snr_eval.py \
    --datapath /path/to/data \
    --model PINNCEST \
    --train_seeds 42 123 2024 \
    --snr_list 20 30 40 50 60 80 100 \
    --noise_repeats 5
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--datapath` | str | *(required)* | Path to the dataset root |
| `--model` | str | `PINNCEST` | Model class name |
| `--k_mode` | str | `log` | Label transformation for *k* |
| `--train_seeds` | int+ | `[42]` | Training seeds of the checkpoints to evaluate |
| `--snr_list` | float+ | `20 30 40 50 60 80 100` | SNR levels (dB) to sweep |
| `--noise_repeats` | int | `5` | Number of independent noise realizations per (seed, SNR) pair |
| `--no_physics` | flag | off | Disable physics during inference |

Total evaluations = `len(train_seeds) × len(snr_list) × noise_repeats`.

### Outputs

- `snr_eval/PINNCEST/snr_baseline_raw_<timestamp>.csv` — long-form data
  (one row per `(train_seed, snr, noise_seed, pool, var, metric)`)
- `snr_eval/PINNCEST/snr_baseline_<timestamp>.json` — nested metrics dump

---

## 5. End-to-end Reproducibility Pipeline

The recommended order to reproduce the paper figures from scratch:

```bash
# 1. Train all ablation configurations across 3 seeds (~hours)
python run_ablation.py \
    --datapath /path/to/data \
    --model PINNCEST \
    --opt adamw \
    --seeds 42 123 2024

# 2. Evaluate baseline SNR robustness (~minutes; reuses checkpoints)
python run_snr.py \
    --datapath /path/to/data \
    --model PINNCEST \
    --train_seeds 42 123 2024 \
    --snr_list 20 30 40 50 60 80 100 \
    --noise_repeats 5
```

## 6. Output Directory Layout

After running the full pipeline, the repository will look like:

```
PINNCEST/
├── checkpoints/PINNCEST/<trial_name>/        # trained weights
│   └── final.pth
├── runs/PINNCEST/<trial_name>/               # TensorBoard logs
├── results/PINNCEST/<trial_name>/            # test-set predictions
│   ├── Result.mat 
│   └── data/metrics_summary.txt
├── ablation_summary/PINNCEST/                 # ablation CSVs & JSON
│   ├── ablation_raw_<timestamp>.csv
│   ├── ablation_agg_<timestamp>.csv
│   └── ablation_<timestamp>.json
├── snr_eval/PINNCEST/                         # SNR evaluation outputs
│   ├── snr_baseline_raw_<timestamp>.csv
└── └── snr_baseline_<timestamp>.json

```

---

## 7. Tips & Troubleshooting

### Common pitfalls

- **All ablation experiments return identical metrics**
  Check that `test.py` writes to `results/PINNCEST/<trial_name>/Result.mat`
  (per trial) and that `evaluate.py` reads from the same per-trial path.
  Both must support the `trial_name` argument; otherwise the last trial
  silently overwrites all earlier results.

- **`no_physics` ablation crashes**
  Ensure `run_ablation.py` syncs `args.use_physics` with the override
  dictionary (this is handled automatically in the provided version).

- **Error bars look misaligned in Adobe Illustrator** but fine in
  preview
  This is a known AI rendering artifact for thin stroked lines. The
  plotting scripts work around this by drawing error bars as filled
  `PathPatch` rectangles instead of stroked lines.

- **CUDA out of memory**
  Reduce `batch_size` in `params/PINNCEST.yaml`. The default assumes
  a single GPU with ≥12 GB VRAM.

- **TensorBoard logs not appearing**
  Confirm that `runs/PINNCEST/<trial_name>/` exists and contains
  `events.out.tfevents.*` files; otherwise inspect the training stdout
  for errors.

### Reproducibility checklist

- Fixed seed in `train.py` (default `42`)
- Fixed test-set sampling seed in `test.py` (hard-coded `42`)
- Fixed dataset split in `train.py` (`torch.Generator().manual_seed(42)`)
- Per-trial result directories prevent overwrites
- All metrics aggregated across multiple seeds in `run_ablation.py`

---
**Methods** section of the accompanying paper.