# calattn-repro/src/metrics/adaece.py
from __future__ import annotations
import torch


@torch.no_grad()
def adaece(
    probs: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    Adaptive ECE via equal-count (quantile) bins over top-1 confidence.

    Returns percentage points (0..100).
    """
    if probs.ndim != 2:
        raise ValueError("probs must be [N,C]")
    conf, pred = probs.max(dim=1)                 # [N]
    acc = (pred == targets).float()

    # Sort by confidence and split into equal-count bins
    order = torch.argsort(conf)
    conf_s = conf[order]
    acc_s = acc[order]

    n = conf.numel()
    ece = torch.zeros((), device=probs.device)

    # bin edges by index (equal mass)
    # last bin takes remainder
    for b in range(n_bins):
        start = (b * n) // n_bins
        end = ((b + 1) * n) // n_bins
        if end <= start:
            continue
        conf_b = conf_s[start:end].mean()
        acc_b = acc_s[start:end].mean()
        gap = (acc_b - conf_b).abs()
        weight = (end - start) / n
        ece = ece + gap * weight

    return float(ece.item() * 100.0)
