# calattn-repro/src/losses/label_smooth.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Standard label-smoothed cross entropy for multi-class classification.

    Args:
        smoothing: epsilon in [0, 1). 0 => vanilla CE.
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"smoothing must be in [0,1), got {smoothing}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"invalid reduction: {reduction}")
        self.smoothing = float(smoothing)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C]
        target: [B] (int64 class indices)
        """
        log_probs = F.log_softmax(logits, dim=-1)  # [B,C]
        n_classes = log_probs.size(-1)

        # Negative log-likelihood for the true class
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)  # [B]
        # Uniform prior loss
        smooth = -log_probs.mean(dim=-1)  # [B]

        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
