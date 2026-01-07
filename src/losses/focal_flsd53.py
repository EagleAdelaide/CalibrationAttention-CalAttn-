import torch
from torch import nn
import torch.nn.functional as F


class FocalFLSD53(nn.Module):
    """
    A practical "Focal family" calibration baseline (FLSD-53-style):
    - focal-style modulation on p_t
    - optional schedule hook via set_progress()
    This is implemented as a reproducible baseline; tune (gamma, beta) if your paper uses exact values.
    """
    def __init__(self, gamma_start=5.0, gamma_end=3.0, beta=0.0):
        super().__init__()
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.beta = beta
        self._progress = 0.0

    def set_progress(self, progress: float):
        self._progress = float(max(0.0, min(1.0, progress)))

    def forward(self, probs, y):
        # probs: (B,C)
        pt = probs.gather(1, y.view(-1, 1)).clamp_min(1e-8).squeeze(1)
        gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * self._progress

        # focal loss
        loss = -((1.0 - pt) ** gamma) * torch.log(pt)

        # optional entropy-ish term (beta) for calibration baselines; keep beta=0 unless required
        if self.beta != 0.0:
            ent = -torch.sum(probs * torch.log(probs.clamp_min(1e-8)), dim=1)
            loss = loss + self.beta * ent

        return loss.mean()
