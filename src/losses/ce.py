import torch
from torch import nn
import torch.nn.functional as F


class CrossEntropyLS(nn.Module):
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.ls = float(label_smoothing)

    def forward(self, logits, y):
        return F.cross_entropy(logits, y, label_smoothing=self.ls)
