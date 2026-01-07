import torch
from torch import nn


class RelaxedTempHead(nn.Module):
    """Feature-conditioned temperature, trained jointly (baseline)."""
    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Softplus()
        )

    def forward(self, feat):
        return self.net(feat) + 1e-6


class RelaxedSoftmaxWrapper(nn.Module):
    """
    A baseline similar in spirit to feature-dependent temperature (Relaxed Softmax).
    Uses pooled feature -> predicts T(x) -> logits/T(x). No special inductive constraints.
    """
    def __init__(self, backbone: nn.Module, hidden: int = 128):
        super().__init__()
        self.backbone = backbone
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            feat = self._forward_features(dummy)
        d_in = feat.shape[-1]
        self.temp = RelaxedTempHead(d_in=d_in, hidden=hidden)

    def _forward_features(self, x):
        feat = self.backbone.forward_features(x)
        if feat.dim() == 3:
            feat = feat[:, 0, :]
        return feat

    def forward(self, x):
        feat = self._forward_features(x)
        T = self.temp(feat)
        logits = self.backbone.forward_head(feat, pre_logits=False)
        return logits / T
