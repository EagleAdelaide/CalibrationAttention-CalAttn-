# calattn-repro/src/metrics/smece.py
from __future__ import annotations
import torch


@torch.no_grad()
def smece(
    probs: torch.Tensor,
    targets: torch.Tensor,
    bandwidth: float = 0.05,
    max_points: int | None = 5000,
) -> float:
    """
    Smooth-ECE (kernel smoothed accuracy vs confidence) for top-1 confidence.

    Notes:
    - O(N^2) kernel; for reviewer-friendliness we optionally subsample if N is large.
    - Returns percentage points (0..100).
    """
    if probs.ndim != 2:
        raise ValueError("probs must be [N,C]")

    device = probs.device
    conf, pred = probs.max(dim=1)                  # [N]
    correct = (pred == targets).float()            # [N]
    n = conf.numel()

    if max_points is not None and n > max_points:
        # deterministic subsample: take evenly spaced indices
        idx = torch.linspace(0, n - 1, steps=max_points, device=device).long()
        conf = conf[idx]
        correct = correct[idx]
        n = conf.numel()

    # Pairwise distances (N,N)
    # K_ij = exp(-(ci-cj)^2 / (2h^2))
    h = float(bandwidth)
    if h <= 0:
        raise ValueError("bandwidth must be > 0")

    c = conf.view(-1, 1)                            # [N,1]
    dist2 = (c - c.t()).pow(2)                      # [N,N]
    K = torch.exp(-dist2 / (2.0 * h * h))           # [N,N]

    # smoothed accuracy at each i
    num = (K * correct.view(1, -1)).sum(dim=1)      # [N]
    den = K.sum(dim=1).clamp_min(1e-12)             # [N]
    acc_hat = num / den                             # [N]

    val = (acc_hat - conf).abs().mean() * 100.0
    return float(val.item())
