# Reproducible CIFAR Calibration (Anonymous)

This repository contains a **runnable** training script for CIFAR-10 / CIFAR-100 with multiple losses, temperature scaling, and reliability diagrams. It includes **cross-host run locking** to avoid duplicate training when multiple servers launch the same configuration.

## Contents
- `many_losses_multibackbone_training.py` – main training pipeline
- `environment.yml` – conda environment spec (CPU-friendly by default)
- This README

## Environment (Conda)
Create and activate the environment:
```bash
conda env create -f environment.yml
conda activate cifar-calib
```

> **GPU users:** The provided `environment.yml` installs the **CPU** build of PyTorch by default (portable). If you have NVIDIA GPUs and want CUDA acceleration, install the matching CUDA build after activation, e.g.:
> ```bash
> conda install -y -c nvidia -c pytorch pytorch-cuda=12.4
> ```
> or use the official `pytorch` instructions for your OS/CUDA.

## Quick Start
Run CIFAR-10 (default backbones: resnet50, resnet110, wrn28x10, densenet121):
```bash
python many_losses_multibackbone_training.py --dataset cifar10 --out-dir runs/c10
```

Run CIFAR-100:
```bash
python many_losses_multibackbone_training.py --dataset cifar100 --out-dir runs/c100
```

You can control the backbones:
```bash
python many_losses_multibackbone_training.py --backbones resnet50,resnet110
```

## What gets saved
- `runs/<dataset>/.../reliability_*.png` – reliability diagrams (before & after TS)
- `runs/<dataset>/*_methods_log.csv` – per-epoch metrics
- `done_<dataset>_<backbone>_<method>.json` – done markers (to skip on future runs)
- `lock_*.lock` – transient lock files (auto-removed on finish); stale locks (>48h) are auto-recovered

## Notes
- Script works on **CPU** or **GPU** automatically. AMP autocast is enabled only on CUDA.
- Default epochs in this package are kept **small (5)** for fast verification. Reviewers can increase via `--epochs`.
- CIFAR datasets are downloaded automatically into `--data` (default `./data`).

## Minimal Reproduction Checklist
1. `conda env create -f environment.yml && conda activate cifar-calib`
2. `python many_losses_multibackbone_training.py --dataset cifar10 --out-dir runs/c10`
3. Inspect the produced `reliability_*.png` and the CSV logs.

If anything fails to run, please share the console output (no system details required).
