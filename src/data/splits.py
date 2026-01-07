from torch.utils.data import Subset
import numpy as np


def make_train_val_split(dataset, val_ratio=0.05, seed=0):
    n = len(dataset)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(round(n * val_ratio))
    val_idx = idx[:n_val].tolist()
    tr_idx = idx[n_val:].tolist()
    return Subset(dataset, tr_idx), Subset(dataset, val_idx)
