# import torch
# import numpy as np
# from sklearn.metrics import roc_auc_score


# @torch.no_grad()
# def ood_auroc_from_confs(id_logits, ood_logits):
#     # Use max softmax confidence as score; lower conf => more OOD-like
#     id_probs = torch.softmax(id_logits, dim=1)
#     ood_probs = torch.softmax(ood_logits, dim=1)
#     id_conf = id_probs.max(dim=1).values.cpu().numpy()
#     ood_conf = ood_probs.max(dim=1).values.cpu().numpy()

#     y_true = np.concatenate([np.zeros_like(id_conf), np.ones_like(ood_conf)])  # 1=OOD
#     scores = np.concatenate([-id_conf, -ood_conf])  # higher => more OOD
#     return float(roc_auc_score(y_true, scores))


# src/metrics/ood.py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

@torch.no_grad()
def ood_auroc_from_logits(id_logits: torch.Tensor, ood_logits: torch.Tensor) -> float:
    """
    AUROC for OoD detection using MSP (max softmax probability).
    Convention:
      y=0 -> ID, y=1 -> OoD
      score higher -> more OoD-like
    We use score = -max_softmax_confidence.
    """
    id_probs = torch.softmax(id_logits, dim=1)
    ood_probs = torch.softmax(ood_logits, dim=1)

    id_conf = id_probs.max(dim=1).values.detach().float().cpu().numpy()
    ood_conf = ood_probs.max(dim=1).values.detach().float().cpu().numpy()

    y_true = np.concatenate([
        np.zeros(id_conf.shape[0], dtype=np.int64),
        np.ones(ood_conf.shape[0], dtype=np.int64),
    ])
    scores = np.concatenate([-id_conf, -ood_conf]).astype(np.float64)

    return float(roc_auc_score(y_true, scores))

@torch.no_grad()
def ood_auroc_from_confs(id_logits, ood_logits):
    # Use max softmax confidence as score; lower conf => more OOD-like
    id_probs = torch.softmax(id_logits, dim=1)
    ood_probs = torch.softmax(ood_logits, dim=1)
    id_conf = id_probs.max(dim=1).values.cpu().numpy()
    ood_conf = ood_probs.max(dim=1).values.cpu().numpy()

    y_true = np.concatenate([np.zeros_like(id_conf), np.ones_like(ood_conf)])  # 1=OOD
    scores = np.concatenate([-id_conf, -ood_conf])  # higher => more OOD
    return float(roc_auc_score(y_true, scores))

