# src/models/calattn.py
import torch
from torch import nn


class CalAttnHead(nn.Module):
    """2-layer MLP -> Softplus to predict positive per-sample scale s(x)."""
    def __init__(self, d_in: int, hidden: int = 128, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # returns [B, 1]
        return self.net(feat) + self.eps


class CalAttnWrapper(nn.Module):
    """
    Wraps a timm classifier and applies representation-conditioned scaling:
      logits_scaled = logits / s(x)

    mode:
      - "cls": use CLS token (ViT/DeiT)
      - "patch_mean": mean of patch tokens (ViT/DeiT)
      - "concat": concat [CLS, patch_mean] (ViT/DeiT)
    """

    def __init__(self, backbone: nn.Module, hidden: int = 128, mode: str = "cls", eps: float = 1e-6):
        super().__init__()
        self.backbone = backbone
        self.mode = str(mode).lower()
        self.eps = eps

        # timm convention: embedding dim
        d = getattr(backbone, "num_features", None)
        if d is None:
            raise ValueError("Backbone missing attribute `num_features` (timm models usually have it).")

        d_in = 2 * d if self.mode == "concat" else d
        self.cal_head = CalAttnHead(d_in=d_in, hidden=hidden, eps=eps)

    def _select_feature(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat can be:
          - [B, N, D] for ViT/DeiT token sequences
          - [B, D] for pooled features (e.g., Swin in timm commonly returns [B, D])
        """
        if isinstance(feat, (tuple, list)):
            feat = feat[-1]

        if feat.dim() == 3:
            # token sequence: [B, N, D], CLS at index 0
            cls = feat[:, 0, :]
            patches = feat[:, 1:, :] if feat.size(1) > 1 else feat
            patch_mean = patches.mean(dim=1)

            if self.mode == "cls":
                return cls
            if self.mode in ("patch_mean", "patch-mean", "patchmean"):
                return patch_mean
            if self.mode == "concat":
                return torch.cat([cls, patch_mean], dim=-1)
            raise ValueError(f"Unknown CalAttn mode: {self.mode}")

        # feat is [B, D] (no explicit tokens)
        if self.mode == "concat":
            raise ValueError("mode='concat' requires token features [B, N, D] (ViT/DeiT).")
        return feat  # for Swin etc.
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Get a representation for predicting s(x)
        feat = self.backbone.forward_features(x)
        z = self._select_feature(feat)  # [B, D] or [B, 2D] depending on mode

        # 2) Get raw logits using the model's native forward (guarantees correct dim into head)
        logits = self.backbone(x)  # [B, C]

        # 3) Predict per-sample scale and apply scaling
        s = self.cal_head(z)       # [B, 1]
        logits = logits / s
        return logits
