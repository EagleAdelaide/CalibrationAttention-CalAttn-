# calattn-repro/src/metrics/ece.py
from __future__ import annotations
import torch


def _bin_indices(conf: torch.Tensor, n_bins: int) -> torch.Tensor:
    # conf in [0,1]
    # bins: [0,1/n),...,[(n-1)/n,1]
    idx = torch.clamp((conf * n_bins).long(), max=n_bins - 1)
    return idx


@torch.no_grad()
def ece_mce(
    probs: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
) -> tuple[float, float]:
    """
    Classic fixed-width ECE + MCE for top-1 confidence.

    probs: [N, C] softmax probabilities
    targets: [N] int64
    returns: (ECE, MCE) in percentage points (0..100)
    """
    if probs.ndim != 2:
        raise ValueError("probs must be [N,C]")
    if targets.ndim != 1:
        raise ValueError("targets must be [N]")

    conf, pred = probs.max(dim=1)              # [N]
    acc = (pred == targets).float()            # [N]

    bin_id = _bin_indices(conf, n_bins)        # [N]
    ece = torch.zeros((), device=probs.device)
    mce = torch.zeros((), device=probs.device)

    for b in range(n_bins):
        mask = (bin_id == b)
        if mask.any():
            conf_b = conf[mask].mean()
            acc_b = acc[mask].mean()
            gap = (acc_b - conf_b).abs()
            ece = ece + gap * (mask.float().mean())
            mce = torch.maximum(mce, gap)

    return float(ece.item() * 100.0), float(mce.item() * 100.0)
