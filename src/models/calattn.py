import torch
from torch import nn


class CalAttnHead(nn.Module):
    """2-layer MLP -> Softplus to predict positive per-sample scale s(x)."""
    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Softplus()
        )

    def forward(self, feat):
        s = self.net(feat) + 1e-6
        return s


class CalAttnWrapper(nn.Module):
    """
    Wraps a timm classifier:
    - extracts representation (typically CLS pooled feature)
    - predicts scale s(x)
    - returns scaled logits: logits / s(x)
    """
    def __init__(self, backbone: nn.Module, hidden: int = 128, mode: str = "cls"):
        super().__init__()
        self.backbone = backbone
        self.mode = mode

        # timm: use forward_features() to get pooled tokens/features
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            feat = self._forward_features(dummy)
        d_in = feat.shape[-1]

        self.head = CalAttnHead(d_in=d_in, hidden=hidden)

    def _forward_features(self, x):
        # timm models typically implement forward_features returning (B, D) or (B, N, D)
        feat = self.backbone.forward_features(x)

        # Handle common cases:
        # - ViT/DeiT: often (B, N+1, D) tokens; CLS at index 0
        # - Swin: often (B, D) pooled
        if feat.dim() == 3:
            if self.mode == "cls":
                return feat[:, 0, :]
            elif self.mode == "patch_mean":
                return feat[:, 1:, :].mean(dim=1)
            elif self.mode == "concat":
                cls = feat[:, 0, :]
                pm = feat[:, 1:, :].mean(dim=1)
                return torch.cat([cls, pm], dim=-1)
            else:
                raise ValueError(f"Unknown calattn_on: {self.mode}")
        return feat

    def forward(self, x):
        feat = self._forward_features(x)

        # For concat mode, head input dim differs; handle by lazy rebuild not desired in paper.
        # We allow concat only if the head was built correctly (img size matches config).
        s = self.head(feat)  # (B,1)

        # timm classifier: forward_head(feat, pre_logits=False) expects raw features (B,D)
        # But if feat came from token tensor, we already pooled to (B,D) above.
        logits = self.backbone.forward_head(feat, pre_logits=False)
        logits = logits / s
        return logits
