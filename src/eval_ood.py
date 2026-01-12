#!/usr/bin/env python3
# calattn-repro/src/eval_ood.py
"""
OOD evaluation for CalAttn reproducibility.

Computes AUROC (%) for distribution shift detection using max softmax confidence:
  - ID: CIFAR-10 test (or whatever cfg.data.dataset specifies, but this script is intended for CIFAR-10)
  - OOD-1: SVHN test
  - OOD-2: CIFAR-10-C (severity 1..5), per corruption + mean

If CIFAR-10-C .npy files are missing, auto-download + extract into a user-writable cache:
  ~/.cache/calattn-repro/cifar10c

AUROC definition:
  score = -max_softmax_confidence  (higher => more OOD-like)
  y_true: 0 for ID, 1 for OOD

Example:
  python src/eval_ood.py \
    --config configs/cifar10_vit224_calattn.yaml \
    --ckpt outputs/cifar10_vit224_calattn/seed0/best.pt \
    --c10c_root /data/cifar10c \
    --c10c_severity 5 \
    --c10c_corruptions all \
    --save_json
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PIL import Image

try:
    from sklearn.metrics import roc_auc_score
except Exception as e:
    raise ImportError("scikit-learn is required: pip install scikit-learn") from e

from utils.io import load_yaml, ensure_dir
from data.loaders import build_datasets, build_transforms
from models.factory import build_model_and_wrapper


# ----------------------------
# Constants
# ----------------------------
C10C_CORRUPTIONS: List[str] = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

DEFAULT_C10C_URL = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"


# ----------------------------
# Logging helpers
# ----------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def _default_cache_root() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "calattn-repro", "cifar10c")


# ----------------------------
# CIFAR-10-C cache utils
# ----------------------------
def _download(url: str, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    log(f"[OOD] Downloading: {url}")
    log(f"[OOD] To: {out_path}")
    urllib.request.urlretrieve(url, out_path)
    log("[OOD] Download finished.")


def _extract_tar(tar_path: str, out_dir: str) -> None:
    ensure_dir(out_dir)
    log(f"[OOD] Extracting: {tar_path} -> {out_dir}")
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=out_dir)
    log("[OOD] Extraction finished.")


def _resolve_cifar10c_dir(root: str) -> str:
    """
    Return the directory that contains labels.npy and gaussian_noise.npy.
    Common extraction layouts:
      root/CIFAR-10-C/*.npy
      root/*.npy
    """
    candidates = [
        root,
        os.path.join(root, "CIFAR-10-C"),
        os.path.join(root, "cifar10c"),
        os.path.join(root, "CIFAR-10-C", "CIFAR-10-C"),
    ]
    candidates = [c for c in candidates if os.path.isdir(c)]

    def has_core(d: str) -> bool:
        return (
            os.path.isfile(os.path.join(d, "labels.npy"))
            and os.path.isfile(os.path.join(d, "gaussian_noise.npy"))
        )

    for c in candidates:
        if has_core(c):
            return c

    # walk one level deep
    for c in candidates:
        try:
            for name in os.listdir(c):
                d = os.path.join(c, name)
                if os.path.isdir(d) and has_core(d):
                    return d
        except Exception:
            pass

    raise FileNotFoundError(
        f"Could not locate CIFAR-10-C .npy files under root='{root}'. "
        "Expected labels.npy and corruption .npy files."
    )


def ensure_cifar10c(
    root_preferred: Optional[str],
    url: str = DEFAULT_C10C_URL,
) -> Tuple[str, str]:
    """
    Ensure CIFAR-10-C is available locally.

    Returns:
      (root_used, c10c_dir) where c10c_dir contains the .npy files.
    """
    # Prefer user-provided root if it already works
    if root_preferred is not None:
        if os.path.isdir(root_preferred):
            try:
                c10c_dir = _resolve_cifar10c_dir(root_preferred)
                return root_preferred, c10c_dir
            except Exception:
                pass
        log(f"[OOD] Provided --c10c_root='{root_preferred}' is not usable. Falling back to cache.")

    cache_root = _default_cache_root()
    ensure_dir(cache_root)

    # If already extracted, do not download
    try:
        c10c_dir = _resolve_cifar10c_dir(cache_root)
        log(f"[OOD] Found existing CIFAR-10-C at: {c10c_dir}")
        return cache_root, c10c_dir
    except Exception:
        pass

    tar_path = os.path.join(cache_root, "CIFAR-10-C.tar")

    if os.path.isfile(tar_path):
        log(f"[OOD] Found cached archive: {tar_path}")
    else:
        _download(url, tar_path)

    log("[OOD] CIFAR-10-C not found — extracting archive.")
    _extract_tar(tar_path, cache_root)

    c10c_dir = _resolve_cifar10c_dir(cache_root)
    return cache_root, c10c_dir


# ----------------------------
# Preprocess CIFAR-10-C batches for ViT/Swin (224 input)
# ----------------------------
@torch.no_grad()
def preprocess_c10c_batch_fast(batch_uint8_nhwc: np.ndarray, img_size: int, device: torch.device) -> torch.Tensor:
    """
    batch_uint8_nhwc: np.ndarray [B, 32, 32, 3] uint8
    returns: torch.FloatTensor [B, 3, img_size, img_size] normalized like loaders.py
    """
    x = torch.from_numpy(batch_uint8_nhwc).to(device=device)        # uint8 NHWC
    x = x.permute(0, 3, 1, 2).float() / 255.0                       # float NCHW in [0,1]

    if x.shape[-1] != img_size or x.shape[-2] != img_size:
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


# ----------------------------
# Model inference helpers
# ----------------------------
@torch.no_grad()
def gather_logits(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    outs = []
    for x, _y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        outs.append(logits.detach().cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def max_softmax_conf_from_logits(logits: torch.Tensor) -> np.ndarray:
    probs = F.softmax(logits, dim=1)
    conf = probs.max(dim=1).values
    return conf.cpu().numpy()


def auroc_from_confs(id_conf: np.ndarray, ood_conf: np.ndarray) -> float:
    """
    Returns AUROC in percentage (0..100).
    score = -confidence  (higher => more OOD-like)
    """
    y_true = np.concatenate([np.zeros_like(id_conf), np.ones_like(ood_conf)])  # 1=OOD
    scores = np.concatenate([-id_conf, -ood_conf])
    return float(roc_auc_score(y_true, scores) * 100.0)


def safe_torch_load(path: str, map_location="cpu"):
    """
    weights_only=True exists in newer PyTorch; fall back gracefully.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ----------------------------
# Args
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)

    ap.add_argument("--batch_size", type=int, default=None, help="Override cfg.data.batch_size for eval")
    ap.add_argument("--num_workers", type=int, default=None, help="Override cfg.data.num_workers for eval")

    ap.add_argument("--c10c_root", type=str, default=None, help="Path to CIFAR-10-C root (optional)")
    ap.add_argument("--c10c_url", type=str, default=DEFAULT_C10C_URL, help="Download URL if missing")
    ap.add_argument("--c10c_severity", type=int, default=5, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--c10c_corruptions", type=str, default="all",
                    help="Comma-separated corruption list or 'all'")

    ap.add_argument("--svhn_root", type=str, default=None,
                    help="Optional root for SVHN download; defaults to cfg.data.data_dir")

    ap.add_argument("--save_json", action="store_true")
    ap.add_argument("--out_json", type=str, default=None)

    return ap.parse_args()


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[OOD] Device: {device}")

    bs = int(args.batch_size) if args.batch_size is not None else int(cfg["data"]["batch_size"])
    nw = int(args.num_workers) if args.num_workers is not None else int(cfg["data"]["num_workers"])
    img_size = int(cfg["data"].get("img_size", 224))

    # Build transforms and ID dataset loader
    train_tf, test_tf = build_transforms(cfg)
    _ds_train, ds_id_test = build_datasets(cfg, train_tf, test_tf)
    id_loader = DataLoader(ds_id_test, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    log(f"[OOD] ID dataset: {cfg['data']['dataset']} test={len(ds_id_test)} | bs={bs} | workers={nw}")

    # Model + ckpt
    model = build_model_and_wrapper(cfg).to(device)
    ckpt = safe_torch_load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    log(f"[OOD] Loaded checkpoint: {args.ckpt}")

    # ID logits + ID confidence (this fixes your id_conf bug)
    id_logits = gather_logits(model, id_loader, device)
    id_conf = max_softmax_conf_from_logits(id_logits)

    out: Dict = {
        "id_dataset": str(cfg["data"]["dataset"]).lower(),
        "model_backbone": str(cfg["model"]["backbone"]),
        "img_size": img_size,
        "batch_size": bs,
        "num_workers": nw,
    }

    # -----------------------
    # SVHN AUROC
    # -----------------------
    svhn_root = args.svhn_root if args.svhn_root is not None else cfg["data"]["data_dir"]
    try:
        from torchvision import datasets as tv_datasets
        ds_svhn = tv_datasets.SVHN(svhn_root, split="test", download=True, transform=test_tf)
        svhn_loader = DataLoader(ds_svhn, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
        log(f"[OOD] SVHN test={len(ds_svhn)} | root={svhn_root}")

        svhn_logits = gather_logits(model, svhn_loader, device)
        svhn_conf = max_softmax_conf_from_logits(svhn_logits)
        out["auroc_c10_to_svhn"] = auroc_from_confs(id_conf, svhn_conf)
    except Exception as e:
        out["auroc_c10_to_svhn"] = {"error": f"{type(e).__name__}: {e}"}

    # -----------------------
    # CIFAR-10-C AUROC
    # -----------------------
    if args.c10c_corruptions.strip().lower() == "all":
        corruptions = list(C10C_CORRUPTIONS)
    else:
        corruptions = [c.strip() for c in args.c10c_corruptions.split(",") if c.strip()]
        for c in corruptions:
            if c not in C10C_CORRUPTIONS:
                raise ValueError(f"Unknown corruption '{c}'. Valid: {C10C_CORRUPTIONS}")

    c10c_root_used, c10c_dir = ensure_cifar10c(args.c10c_root, url=args.c10c_url)
    out["c10c_root_used"] = c10c_root_used
    out["c10c_dir_used"] = c10c_dir
    out["c10c_severity"] = int(args.c10c_severity)
    out["c10c_corruptions"] = corruptions

    # Load labels once, slice once
    labels_all = np.load(os.path.join(c10c_dir, "labels.npy"))
    if labels_all.shape[0] != 50000:
        raise ValueError(f"Expected labels.npy length 50000, got {labels_all.shape[0]}")

    sev = int(args.c10c_severity)
    start = (sev - 1) * 10000
    end = sev * 10000
    labels_sev = labels_all[start:end]  # currently unused for AUROC, but kept for sanity

    by_corr: Dict[str, Dict] = {}
    aurocs: List[float] = []

    from tqdm import tqdm

    for corr in corruptions:
        t0 = time.time()
        log(f"[OOD] Processing corruption: {corr}")
        try:
            npy_path = os.path.join(c10c_dir, f"{corr}.npy")
            if not os.path.isfile(npy_path):
                raise FileNotFoundError(f"Missing corruption file: {npy_path}")

            data_all = np.load(npy_path)  # shape [50000, 32, 32, 3], uint8
            if data_all.shape[0] != 50000:
                raise ValueError(f"{corr}.npy expected 50000 samples, got {data_all.shape[0]}")

            data = data_all[start:end]  # [10000, 32, 32, 3]
            if data.dtype != np.uint8:
                data = data.astype(np.uint8)

            # Run model, collect OOD confidences
            confs = []
            with torch.no_grad():
                for i in tqdm(range(0, len(data), bs), desc=f"[OOD] {corr} batches", ncols=100):
                    batch_np = data[i:i + bs]
                    batch = preprocess_c10c_batch_fast(batch_np, img_size=img_size, device=device)
                    logits = model(batch)
                    conf = F.softmax(logits, dim=1).max(dim=1).values.detach().cpu().numpy()
                    confs.append(conf)

            ood_conf = np.concatenate(confs, axis=0)
            auroc = auroc_from_confs(id_conf, ood_conf)

            by_corr[corr] = {"auroc": float(auroc)}
            aurocs.append(float(auroc))
            log(f"[OOD] {corr}: AUROC={auroc:.2f} | time={time.time() - t0:.1f}s")

        except Exception as e:
            by_corr[corr] = {"error": f"{type(e).__name__}: {e}"}
            log(f"[OOD][ERROR] {corr}: {type(e).__name__}: {e}")

    out["auroc_c10_to_c10c_by_corruption"] = by_corr
    out["auroc_c10_to_c10c_mean"] = float(np.mean(aurocs)) if len(aurocs) > 0 else None

    print(json.dumps(out, indent=2))

    if args.save_json:
        if args.out_json:
            out_path = args.out_json
            ensure_dir(os.path.dirname(out_path) if os.path.dirname(out_path) else ".")
        else:
            ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
            out_path = os.path.join(ckpt_dir, f"ood_sev{sev}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        log(f"[OOD] Saved: {out_path}")


if __name__ == "__main__":
    main()
