import torch
from torch import nn


class DualFocalLoss(nn.Module):
    """
    Dual Focal Loss for Calibration (DFL): uses p_t and the largest non-target probability p_j.
    Concept: keep focal term to mitigate over-confidence, and add a dual term to enlarge separation.

    Implementation follows the paper's description (arXiv:2305.13665):
    - p_t: prob of ground-truth class
    - p_j: largest probability among non-ground-truth classes
    """
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, probs, y):
        B, C = probs.shape
        probs = probs.clamp_min(1e-8)

        pt = probs.gather(1, y.view(-1, 1)).squeeze(1)  # (B,)

        # p_j: maximum non-target prob
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask.scatter_(1, y.view(-1, 1), False)
        pj = probs.masked_fill(~mask, -1.0).max(dim=1).values.clamp_min(1e-8)

        # focal term on pt
        term1 = -((1 - pt) ** self.gamma) * torch.log(pt)

        # dual term penalises large competing probability
        term2 = -(pj ** self.gamma) * torch.log(1 - pj)

        return (term1 + term2).mean()
