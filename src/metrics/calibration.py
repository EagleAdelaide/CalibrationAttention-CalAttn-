# calattn-repro/src/metrics/calibration.py
from __future__ import annotations

import torch
import torch.nn.functional as F

# new metric implementations (you already created these files earlier)
from .ece import ece_mce
from .adaece import adaece
from .classece import classwise_ece
from .smece import smece


# -----------------------------
# Basic helpers (keep old API)
# -----------------------------
@torch.no_grad()
def probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Returns accuracy in [0,1].
    """
    pred = logits.argmax(dim=1)
    return float((pred == targets).float().mean().item())


@torch.no_grad()
def nll_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> float:
    """
    Cross-entropy / NLL from logits. Returns scalar float.
    """
    loss = F.cross_entropy(logits, targets, reduction=reduction)
    return float(loss.item()) if reduction != "none" else loss


@torch.no_grad()
def brier_score_from_probs(probs: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> float:
    """
    Multi-class Brier score: mean over samples of sum_c (p_c - 1[y=c])^2.
    """
    n, c = probs.shape
    y_onehot = F.one_hot(targets, num_classes=c).float()
    per_sample = ((probs - y_onehot) ** 2).sum(dim=1)
    if reduction == "mean":
        return float(per_sample.mean().item())
    if reduction == "sum":
        return float(per_sample.sum().item())
    return per_sample


@torch.no_grad()
def brier_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> float:
    return brier_score_from_probs(probs_from_logits(logits), targets, reduction=reduction)


# ----------------------------------------
# Legacy ECE API (your eval.py uses this)
# ----------------------------------------
@torch.no_grad()
def ece_equal_width(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    """
    Legacy: returns ECE in [0,1] (not percent).
    """
    probs = probs_from_logits(logits)
    ece, _ = ece_mce(probs, targets, n_bins=n_bins)  # in percentage points
    return ece / 100.0


# -----------------------------
# New unified calibration report
# -----------------------------
@torch.no_grad()
def calibration_summary_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
    smece_bandwidth: float = 0.05,
    smece_max_points: int | None = 5000,
) -> dict[str, float]:
    """
    Returns a dict of calibration metrics in *percentage points* (0..100):
      ECE, MCE, AdaECE, ClassECE, smECE
    """
    probs = probs_from_logits(logits)

    ece, mce = ece_mce(probs, targets, n_bins=n_bins)
    out = {
        "ECE": ece,
        "MCE": mce,
        "AdaECE": adaece(probs, targets, n_bins=n_bins),
        "ClassECE": classwise_ece(probs, targets, n_bins=n_bins),
        "smECE": smece(probs, targets, bandwidth=smece_bandwidth, max_points=smece_max_points),
    }
    return out


@torch.no_grad()
def full_eval_summary_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
    smece_bandwidth: float = 0.05,
    smece_max_points: int | None = 5000,
) -> dict[str, float]:
    """
    One-stop summary suitable for tables/logging.
    - top1: percentage (0..100)
    - nll: scalar
    - brier: scalar
    - plus calibration metrics in percentage points
    """
    top1 = accuracy_top1(logits, targets) * 100.0
    nll = nll_loss_from_logits(logits, targets)
    brier = brier_score_from_logits(logits, targets)

    cal = calibration_summary_from_logits(
        logits, targets,
        n_bins=n_bins,
        smece_bandwidth=smece_bandwidth,
        smece_max_points=smece_max_points,
    )

    out = {"top1": float(top1), "nll": float(nll), "brier": float(brier)}
    out.update({k.lower(): float(v) for k, v in cal.items()})  # e.g., ece, adaece, classece, smece
    return out


import torch

@torch.no_grad()
def calibration_summary(
    logits: torch.Tensor,
    y: torch.Tensor,
    n_bins: int = 15,
    prefix: str = ""
):
    """
    Return a dict of standard metrics used in the paper.
    Assumes logits shape (N, C) and y shape (N,).
    """
    out = {}
    # Required basic metrics (must exist in this file)
    if "accuracy_top1" in globals():
        out[prefix + "top1"] = float(accuracy_top1(logits, y) * 100.0)
    if "nll_loss_from_logits" in globals():
        out[prefix + "nll"] = float(nll_loss_from_logits(logits, y))
    if "brier_score_from_logits" in globals():
        out[prefix + "brier"] = float(brier_score_from_logits(logits, y))

    if "ece_equal_width" in globals():
        out[prefix + "ece"] = float(ece_equal_width(logits, y, n_bins=n_bins) * 100.0)
    if "mce_equal_width" in globals():
        out[prefix + "mce"] = float(mce_equal_width(logits, y, n_bins=n_bins) * 100.0)
    if "adaece" in globals():
        out[prefix + "adaece"] = float(adaece(logits, y, n_bins=n_bins) * 100.0)
    if "classwise_ece" in globals():
        out[prefix + "classece"] = float(classwise_ece(logits, y, n_bins=n_bins) * 100.0)
    if "smooth_ece" in globals():
        out[prefix + "smece"] = float(smooth_ece(logits, y) * 100.0)

    return out
