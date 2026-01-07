# calattn-repro/src/metrics/classece.py
from __future__ import annotations
import torch


def _bin_indices(conf: torch.Tensor, n_bins: int) -> torch.Tensor:
    idx = torch.clamp((conf * n_bins).long(), max=n_bins - 1)
    return idx


@torch.no_grad()
def classwise_ece(
    probs: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    Classwise ECE:
      For each class k, consider samples where predicted class == k,
      compute ECE_k over their confidence, then average over classes
      with at least one predicted sample.

    Returns percentage points (0..100).
    """
    if probs.ndim != 2:
        raise ValueError("probs must be [N,C]")
    n_classes = probs.size(1)

    conf, pred = probs.max(dim=1)           # [N]
    correct = (pred == targets).float()     # [N]

    eces = []
    for k in range(n_classes):
        mask_k = (pred == k)
        if not mask_k.any():
            continue
        conf_k = conf[mask_k]
        corr_k = correct[mask_k]

        bin_id = _bin_indices(conf_k, n_bins)
        ece_k = torch.zeros((), device=probs.device)

        for b in range(n_bins):
            mb = (bin_id == b)
            if mb.any():
                conf_b = conf_k[mb].mean()
                acc_b = corr_k[mb].mean()
                gap = (acc_b - conf_b).abs()
                ece_k = ece_k + gap * (mb.float().mean())
        eces.append(ece_k)

    if len(eces) == 0:
        return 0.0
    return float(torch.stack(eces).mean().item() * 100.0)
