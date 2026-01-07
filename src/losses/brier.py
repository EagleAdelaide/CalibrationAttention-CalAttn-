import torch
from torch import nn


class BrierLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs, y):
        # probs: (B,C)
        y_oh = torch.zeros_like(probs).scatter_(1, y.view(-1, 1), 1.0)
        return torch.mean(torch.sum((probs - y_oh) ** 2, dim=1))
