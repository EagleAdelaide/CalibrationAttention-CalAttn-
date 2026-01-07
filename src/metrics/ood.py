import torch
import numpy as np
from sklearn.metrics import roc_auc_score


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
