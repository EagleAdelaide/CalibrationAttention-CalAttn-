import torch
from torch import nn


class MMCE(nn.Module):
    """
    Practical differentiable MMCE-like penalty.
    Reference: Kumar et al. (2018) Trainable Calibration Measures for Neural Networks.
    """
    def __init__(self, sigma: float = 0.4):
        super().__init__()
        self.sigma = sigma

    def forward(self, probs, y):
        # conf and correctness
        conf, pred = probs.max(dim=1)
        corr = (pred == y).float()
        diff = (corr - conf).unsqueeze(1)  # (B,1)

        # RBF kernel on confidence
        c = conf.unsqueeze(1)  # (B,1)
        dist2 = (c - c.t()) ** 2
        K = torch.exp(-dist2 / (2 * self.sigma ** 2))

        mmce = (diff @ diff.t()) * K
        return mmce.mean()
