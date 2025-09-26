# Reproducible CIFAR-10/100 Training: Multi-loss & Multi-backbone

This repository contains a single self-contained training script that reproduces the experiments in your paper setup: multiple losses (CE, Brier, CE+0.1*Brier, Label Smoothing, Focal, Adaptive Focal, Dual Focal, AdaFocal) across multiple backbones (ResNet-50, ResNet-110 (CIFAR style), WideResNet-28x10, DenseNet-121). It also evaluates Temperature Scaling (TS), draws reliability diagrams, logs metrics to CSV, and uses cross-host lock files plus "done" markers so multiple workers won't duplicate the same job.

## Environment

- Python 3.8+
- PyTorch >= 2.0 (CUDA optional)
- torchvision >= 0.15
- matplotlib

You can use conda to create a clean environment:

```bash
conda create -y -n calib python=3.10
conda activate calib
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or cpu wheels
pip install matplotlib
```

> If you prefer CPU-only: `pip install torch torchvision` (without the CUDA index). The script auto-disables AMP on CPU.

## Files

- `many_losses_multibackbone_training.py` — main training/eval script (single file).
- Outputs (under `--out-dir`):
  - `*_methods_log.csv` — per-epoch metrics.
  - `reliability_<dataset>_<backbone>_<method>.png` — reliability diagrams (before vs after TS).
  - `done_<dataset>_<backbone>_<method>.json` — done markers (used to skip on next runs).
  - `error_*.txt` — traceback if a run crashes.
  - `val_bin_stats.txt` — AdaFocal per-epoch bin statistics.

## Quick Start

CIFAR-10 with default backbones and all methods:

```bash
python many_losses_multibackbone_training.py --dataset cifar10 --out-dir runs/c10
```

CIFAR-100:

```bash
python many_losses_multibackbone_training.py --dataset cifar100 --out-dir runs/c100   --backbones resnet50,resnet110,wideresnet28x10,densenet121
```

Run a subset of backbones (example: only ResNet-50):

```bash
python many_losses_multibackbone_training.py --dataset cifar10 --out-dir runs/c10 --backbones resnet50
```

### Important Flags

- `--skip-if-done` (default: on): will skip a specific (dataset, backbone, method) run if its `done_*.json` exists.
- `--ts-every` (default: 1): perform Temperature Scaling evaluation every `ts-every` epochs for logging; reliability diagrams are generated after training finishes per run.
- AdaFocal options:
  - `--adafocal-num-bins 15`
  - `--adafocal-lambda 2.0`
  - `--adafocal-gamma-initial 5.0`
  - `--adafocal-switch-pt 0.25`
  - `--adafocal-gamma-max 8.0`
  - `--adafocal-gamma-min -8.0`
  - `--adafocal-update-every 1`

## Cross-host Mutex and Safe Resume

- The script creates a lock file per `(dataset|backbone|method)` key at: `lock_<key>.lock` inside `--out-dir`. If a lock exists, other workers skip that run.
- Stale locks: if the lock's mtime is older than 48 hours, the script removes it once and retries.
- After a run finishes, a `done_<dataset>_<backbone>_<method>.json` is written to allow `--skip-if-done` to skip next time.

## Logged Metrics

For each epoch and method:
- `test_acc`, `test_ece`, `test_adaece`, `test_classwise_ece`, `test_nll`
- The same after applying Temperature Scaling: `*_ts` columns
- For FCL-Metaλ: `lambda_if_any`, `val_smooth_ece_if_any` are populated.

At the end, the script draws line plots over epochs for accuracy/ECE/NLL etc., and saves per-run reliability diagrams.

## Reproducibility Notes

- We set a global seed (`--seed`, default 42) and use standard CIFAR-10/100 data augmentation (random crop + horizontal flip).
- Learning rate schedule: MultiStepLR at epochs 150 and 250 with `gamma=0.1`.
- Default epochs: 350.
- AMP is automatically enabled on CUDA and disabled on CPU.

## Cite
If your paper needs to be cited in a blind review repo, please include your own reference here.

---

Good luck with your reviews! If reviewers only want a "single command to reproduce", point them to:

```bash
python many_losses_multibackbone_training.py --dataset cifar100 --out-dir runs/c100   --backbones resnet50,resnet110,wideresnet28x10,densenet121
```

This will download CIFAR-100 automatically and produce logs & figures under `runs/c100`.
