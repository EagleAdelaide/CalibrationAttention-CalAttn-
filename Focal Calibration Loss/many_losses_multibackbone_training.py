#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-loss & Multi-backbone training on CIFAR-10 / CIFAR-100
+ Temperature Scaling (TS) & Reliability Diagrams
+ Safe resume (skip finished runs) and cross-host run locking (avoid duplicate runs)
Author: Anonymous

Quick start:
    python many_losses_multibackbone_training.py --dataset cifar10 --out-dir runs/c10

Notes:
- Works on CPU or GPU. AMP autocast is enabled only if CUDA is available.
- Reliability diagram file is saved under out-dir.
- "Done markers" are JSON files used to skip finished runs on future executions.
"""

import os, time, random, argparse, csv, contextlib, json, traceback, math, socket
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from torch.func import functional_call
except Exception:
    from torch.nn.utils.stateless import functional_call

# ---------------------- Cross-host run locking ----------------------

def _run_key_to_fname(run_key: str) -> str:
    return run_key.replace("|", "_").replace("/", "-").replace(" ", "_")

def _lock_path(out_dir: str, run_key: str) -> str:
    return os.path.join(out_dir, f"lock_{_run_key_to_fname(run_key)}.lock")

def acquire_run_lock(out_dir: str, run_key: str, stale_hours: int = 48) -> str or None:
    """
    Try to create a lock file. Return the lock file path if acquired, else None.
    If the existing lock looks stale (mtime > stale_hours), remove it and retry once.
    """
    os.makedirs(out_dir, exist_ok=True)
    lp = _lock_path(out_dir, run_key)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(lp, flags)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "start_ts": time.time(),
                "run_key": run_key
            }, indent=2))
        return lp
    except FileExistsError:
        try:
            mtime = os.path.getmtime(lp)
            if (time.time() - mtime) > stale_hours * 3600:
                print(f"[LOCK] Stale lock (> {stale_hours}h) for {run_key}. Removing {lp}.")
                try:
                    os.remove(lp)
                except FileNotFoundError:
                    pass
                # retry once
                try:
                    fd = os.open(lp, flags)
                    with os.fdopen(fd, "w") as f:
                        f.write(json.dumps({
                            "host": socket.gethostname(),
                            "pid": os.getpid(),
                            "start_ts": time.time(),
                            "run_key": run_key,
                            "note": "recovered stale lock"
                        }, indent=2))
                    return lp
                except FileExistsError:
                    return None
        except FileNotFoundError:
            return None
        return None

def release_run_lock(lock_path: str):
    try:
        if lock_path and os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception as e:
        print(f"[LOCK] Failed to remove lock {lock_path}: {e}")

# ---------------------- Utils ----------------------

def fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds + 0.5))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0:   return f"{d}d {h:02d}:{m:02d}:{s:02d}"
    if h > 0:   return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def safe_softmax(logits, dim=1):
    z = logits - logits.max(dim=dim, keepdim=True).values
    return F.softmax(z, dim=dim)

# ---------------------- ECE metrics ----------------------

def ece_histogram(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    with torch.no_grad():
        conf, pred = probs.max(dim=1)
        correct = (pred == labels).float()
        bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        ece = torch.zeros(1, device=probs.device)
        for i in range(n_bins):
            m = (conf > bins[i]) & (conf <= bins[i+1])
            if m.any():
                bin_conf = conf[m].mean()
                bin_acc  = correct[m].mean()
                ece += (m.float().mean()) * torch.abs(bin_conf - bin_acc)
        return float(ece.item())

def adaece_equalfreq(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    with torch.no_grad():
        conf, pred = probs.max(dim=1)
        correct = (pred == labels).float()
        N = conf.numel()
        if N == 0: return 0.0
        conf_sorted, idx = torch.sort(conf)
        correct_sorted = correct[idx]
        bin_sizes = [N // n_bins + (1 if x < (N % n_bins) else 0) for x in range(n_bins)]
        start = 0
        ece = 0.0
        for bs in bin_sizes:
            if bs == 0: continue
            end = start + bs
            cbin = conf_sorted[start:end]
            abin = correct_sorted[start:end]
            bin_conf = cbin.mean()
            bin_acc  = abin.mean()
            weight = float(bs) / float(N)
            ece += weight * float(torch.abs(bin_conf - bin_acc))
            start = end
        return float(ece)

def classwise_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15, num_classes: int = 10) -> float:
    with torch.no_grad():
        conf, pred = probs.max(dim=1)
        N = labels.numel()
        total = 0.0
        bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        for c in range(num_classes):
            mask = (pred == c)
            Nc = int(mask.sum().item())
            if Nc == 0:
                continue
            conf_c = conf[mask]
            correct_c = (labels[mask] == c).float()
            ece_c = torch.zeros(1, device=probs.device)
            for i in range(n_bins):
                m = (conf_c > bins[i]) & (conf_c <= bins[i+1])
                if m.any():
                    bin_conf = conf_c[m].mean()
                    bin_acc  = correct_c[m].mean()
                    ece_c += (m.float().mean()) * torch.abs(bin_conf - bin_acc)
            total += (Nc / N) * float(ece_c.item())
        return float(total)

def smooth_ece(probs: torch.Tensor, labels_onehot: torch.Tensor, n_bins=15, sigma=0.05, eps=1e-8):
    conf = probs.max(dim=1).values
    soft_correct = (probs * labels_onehot).sum(dim=1)
    centers = torch.linspace(0., 1., steps=n_bins, device=probs.device)
    W = torch.exp(-0.5 * ((conf[:, None] - centers[None, :])**2) / (sigma**2))
    W = W / (W.sum(dim=1, keepdim=True) + eps)
    mass = W.mean(dim=0)
    bin_conf = (W * conf[:, None]).sum(dim=0)
    bin_acc  = (W * soft_correct[:, None]).sum(dim=0)
    gap = torch.sqrt((bin_conf - bin_acc)**2 + 1e-6)
    ece = (mass * gap).sum()
    return ece

# ---------------------- Data ----------------------

def get_cifar_loaders(dataset: str, data_root: str, batch_size=128, num_workers=4, val_size=5000, seed=42):
    dataset = dataset.lower()
    assert dataset in ("cifar10", "cifar100")
    normalize = T.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    test_tf = T.Compose([T.ToTensor(), normalize])

    if dataset == "cifar10":
        TrainDS = torchvision.datasets.CIFAR10
        TestDS  = torchvision.datasets.CIFAR10
        num_classes = 10
    else:
        TrainDS = torchvision.datasets.CIFAR100
        TestDS  = torchvision.datasets.CIFAR100
        num_classes = 100

    full_train = TrainDS(root=data_root, train=True, download=True, transform=train_tf)
    test_set   = TestDS(root=data_root, train=False, download=True, transform=test_tf)

    g = torch.Generator().manual_seed(seed)
    train_size = len(full_train) - val_size
    train_set, val_set = torch.utils.data.random_split(full_train, [train_size, val_size], generator=g)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, num_classes

# ---------------------- Backbones (CIFAR variants) ----------------------

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_CIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(64*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def make_resnet110(num_classes=10):
    return ResNet_CIFAR(BasicBlock, [18,18,18], num_classes=num_classes)

def make_resnet50_cifar(num_classes=10):
    m = torchvision.models.resnet50(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

class WideBasic(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes) and (stride == 1)
        self.convShortcut = None
        if not self.equalInOut:
            self.convShortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
    def forward(self, x):
        if not self.equalInOut:
            out = self.relu(self.bn1(x))
        else:
            out = self.bn1(x); out = self.relu(out)
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        residual = x if self.equalInOut else self.convShortcut(x)
        return out + residual

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super().__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        self.layer = nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropRate=0.0, num_classes=10):
        super().__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = WideBasic
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.size(2))
        out = out.view(-1, self.nChannels)
        return self.fc(out)

def make_wrn28x10(num_classes=10):
    return WideResNet(depth=28, widen_factor=10, dropRate=0.0, num_classes=num_classes)

def make_densenet121_cifar(num_classes=10):
    m = torchvision.models.densenet121(weights=None)
    m.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.features.pool0 = nn.Identity()
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m

# ---------------------- Losses ----------------------

class BrierScorePure(nn.Module):
    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1).transpose(1,2).contiguous().view(-1, input.size(1))
        target = target.view(-1,1)
        target_one_hot = torch.zeros_like(input)
        target_one_hot.scatter_(1, target, 1)
        pt = safe_softmax(input, dim=1)
        squared_diff = (target_one_hot - pt).pow(2)
        return squared_diff.sum(dim=1).mean()

class BrierScoreCombo(nn.Module):
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target)
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1).transpose(1,2).contiguous().view(-1, input.size(1))
        target = target.view(-1, 1)
        target_one_hot = torch.zeros_like(input)
        target_one_hot.scatter_(1, target, 1)
        pt = safe_softmax(input, dim=1)
        squared_diff = (target_one_hot - pt).pow(2)
        brier = squared_diff.sum(dim=1).mean()
        return ce_loss + self.gamma * brier

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=10, smoothing=0.1, device='cuda'):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.device = device
    def forward(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device).long()
        target_prob_dist = torch.full(size=(targets.size(0), self.cls),
                                      fill_value=self.smoothing / (self.cls - 1), device=self.device)
        target_prob_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        return F.kl_div(inputs.log_softmax(dim=-1), target_prob_dist, reduction='batchmean')

class FocalLossClassic(nn.Module):
    def __init__(self, gamma=5.0, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average
    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1).transpose(1,2).contiguous().view(-1, input.size(2))
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1).gather(1, target).view(-1)
        pt = logpt.exp().clamp_min(1e-6)
        loss = - (1-pt).pow(self.gamma) * logpt
        return loss.mean() if self.size_average else loss.sum()

# ---------------------- Temperature Scaling ----------------------

class ModelWithTemperature(nn.Module):
    def __init__(self, model, device='cuda'):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.device = device
    def set_temperature(self, val_loader, max_iter=50, lr=0.01):
        self.to(self.device)
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter) if hasattr(optim, 'LBFGS') else optim.Adam([self.temperature], lr=lr)
        self.temperature.data = torch.ones_like(self.temperature)
        logits_list = []; labels_list = []
        self.model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                logits_list.append(logits)
                labels_list.append(y)
        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        if isinstance(optimizer, optim.LBFGS):
            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss = nll_criterion(logits / self.temperature, labels)
                loss.backward()
                return loss
            optimizer.step(closure)
        else:
            for _ in range(max_iter):
                optimizer.zero_grad(set_to_none=True)
                loss = nll_criterion(logits / self.temperature, labels)
                loss.backward()
                optimizer.step()
        return float(self.temperature.detach().cpu())

# ---------------------- Eval helpers ----------------------

def eval_logits(model, loader, device):
    model.eval()
    logits_list = []; labels_list = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            logits_list.append(logits)
            labels_list.append(y)
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)

def metrics_from_logits(logits, labels, n_bins=15, num_classes=10, temperature=None):
    if temperature is not None:
        logits = logits / temperature
    probs = safe_softmax(logits, dim=1)
    acc = (probs.argmax(dim=1) == labels).float().mean().item()
    ece = ece_histogram(probs, labels, n_bins=n_bins)
    adae = adaece_equalfreq(probs, labels, n_bins=n_bins)
    cwe = classwise_ece(probs, labels, n_bins=n_bins, num_classes=num_classes)
    nll = F.cross_entropy(logits, labels, reduction='mean').item()
    return acc, ece, adae, cwe, nll, probs

def reliability_diagram(probs_before, probs_after, labels, out_path, n_bins=15, title="Reliability Diagram (Before vs After TS)"):
    import numpy as np
    def compute_bin_stats(probs):
        conf, pred = probs.max(dim=1)
        correct = (pred == labels).float()
        bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        centers, accs, confs, counts = [], [], [], []
        for i in range(n_bins):
            if i == 0:
                m = (conf <= bins[i+1])
            elif i == n_bins - 1:
                m = (conf > bins[i]) & (conf <= bins[i+1] + 1e-8)
            else:
                m = (conf > bins[i]) & (conf <= bins[i+1])
            num = int(m.sum().item())
            centers.append(((bins[i] + bins[i+1]) / 2).item())
            if num > 0:
                accs.append(correct[m].mean().item())
                confs.append(conf[m].mean().item())
            else:
                accs.append(float('nan')); confs.append(float('nan'))
            counts.append(num)
        return centers, accs, confs, counts

    cb_x, cb_acc, cb_conf, cb_cnt = compute_bin_stats(probs_before)
    ca_x, ca_acc, ca_conf, ca_cnt = compute_bin_stats(probs_after)

    def nan_to_val(a, val=0.0):
        import numpy as np
        a = np.array(a, dtype=float); a[np.isnan(a)] = val; return a
    cb_acc_n = nan_to_val(cb_acc); cb_conf_n = nan_to_val(cb_conf)
    ca_acc_n = nan_to_val(ca_acc); ca_conf_n = nan_to_val(ca_conf)
    xs = nan_to_val(cb_x)

    group_width = 0.24
    w = group_width / 4.0
    offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]

    def alpha_for_counts(cnt, base=0.85, empty=0.15):
        import numpy as np
        cnt = np.array(cnt, dtype=float)
        return np.where(cnt > 0, base, empty)

    a_cb = alpha_for_counts(cb_cnt)
    a_ca = alpha_for_counts(ca_cnt)

    plt.figure(figsize=(7.2, 5.2))
    bars_cb_acc  = plt.bar(xs + offsets[0], cb_acc_n,  width=w, alpha=0.85, label="Acc (Before)")
    bars_cb_conf = plt.bar(xs + offsets[1], cb_conf_n, width=w, alpha=0.60, label="Conf (Before)")
    bars_ca_acc  = plt.bar(xs + offsets[2], ca_acc_n,  width=w, alpha=0.85, label="Acc (After TS)")
    bars_ca_conf = plt.bar(xs + offsets[3], ca_conf_n, width=w, alpha=0.60, label="Conf (After TS)")

    # Per-bar alpha (empty bins are more transparent)
    for i, r in enumerate(bars_cb_acc.patches):  r.set_alpha(float(a_cb[i]))
    for i, r in enumerate(bars_cb_conf.patches): r.set_alpha(float(a_cb[i] * 0.8))
    for i, r in enumerate(bars_ca_acc.patches):  r.set_alpha(float(a_ca[i]))
    for i, r in enumerate(bars_ca_conf.patches): r.set_alpha(float(a_ca[i] * 0.8))

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    plt.xlim(0.0 - group_width, 1.0 + group_width)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Confidence (bin centers)")
    plt.ylabel("Accuracy / Confidence")
    plt.title(title)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ---------------------- Train/Eval ----------------------

@dataclass
class TrainConfig:
    dataset: str = "cifar10"
    num_classes: int = 10
    epochs: int = 5
    batch_size: int = 128
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    num_workers: int = 2
    seed: int = 42
    amp: bool = True
    ece_bins: int = 15
    out_dir: str = "."
    log_csv: str = "methods_log.csv"
    ts_every: int = 1
    skip_if_done: bool = True

def make_scheduler(opt, total_epochs=350):
    return optim.lr_scheduler.MultiStepLR(opt, milestones=[150, 250], gamma=0.1)

def evaluate_epoch(model, test_loader, device, cfg: TrainConfig, val_loader_for_ts=None, do_ts=True):
    logits_test, labels_test = eval_logits(model, test_loader, device)
    acc, ece, adae, cwe, nll, probs_before = metrics_from_logits(logits_test, labels_test, n_bins=cfg.ece_bins, num_classes=cfg.num_classes)
    if do_ts and val_loader_for_ts is not None:
        mwT = ModelWithTemperature(model, device=device)
        Tval = mwT.set_temperature(val_loader_for_ts, max_iter=50, lr=0.01)
        acc_t, ece_t, adae_t, cwe_t, nll_t, probs_after = metrics_from_logits(
            logits_test, labels_test, n_bins=cfg.ece_bins, num_classes=cfg.num_classes,
            temperature=mwT.temperature.detach())
    else:
        Tval = 1.0
        acc_t, ece_t, adae_t, cwe_t, nll_t = acc, ece, adae, cwe, nll
        probs_after = probs_before
    return (acc, ece, adae, cwe, nll), (acc_t, ece_t, adae_t, cwe_t, nll_t), (probs_before, probs_after, labels_test), Tval

def init_csv(path, header):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

def append_csv(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)

def make_backbone(name: str, num_classes=10):
    name = name.lower()
    if name == "resnet50":
        return make_resnet50_cifar(num_classes)
    elif name == "resnet110":
        return make_resnet110(num_classes)
    elif name in ("wideresnet", "wideresnet28x10", "wrn28x10"):
        return make_wrn28x10(num_classes)
    elif name == "densenet121":
        return make_densenet121_cifar(num_classes)
    else:
        raise ValueError(f"Unknown backbone: {name}")

def train_standard(backbone_name: str, model, train_loader, val_loader, test_loader, cfg: TrainConfig, criterion, device, method_name: str):
    # Adaptive AMP: enable only if CUDA is available
    amp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = bool(cfg.amp and amp_device == 'cuda')
    scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)

    opt = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    sched = make_scheduler(opt, total_epochs=cfg.epochs)
    best = {"acc": 0.0, "ece": 1.0, "adaece": 1.0, "classwise": 1.0, "nll": 10.0,
            "acc_t":0.0,"ece_t":1.0,"adaece_t":1.0,"classwise_t":1.0,"nll_t":10.0}
    last_T = 1.0
    start_all = time.time()

    for ep in range(1, cfg.epochs+1):
        start_time = time.time()
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(amp_device, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()

        sched.step()
        do_ts = ((ep % cfg.ts_every) == 0)
        (acc, ece, adae, cwe, nll), (acc_t, ece_t, adae_t, cwe_t, nll_t), _, Tval = evaluate_epoch(
            model, test_loader, device, cfg, val_loader_for_ts=val_loader, do_ts=do_ts
        )
        last_T = Tval if do_ts else last_T
        if acc >= best["acc"]:
            best.update({"acc": acc, "ece": ece, "adaece": adae, "classwise": cwe, "nll": nll})
        if acc_t >= best["acc_t"]:
            best.update({"acc_t": acc_t, "ece_t": ece_t, "adaece_t": adae_t, "classwise_t": cwe_t, "nll_t": nll_t})
        append_csv(cfg.log_csv, [cfg.dataset, cfg.num_classes, backbone_name, method_name, ep,
                                 acc, ece, adae, cwe, nll,
                                 acc_t, ece_t, adae_t, cwe_t, nll_t,
                                 "", "", last_T])
        epoch_time = time.time() - start_time
        elapsed = time.time() - start_all
        remaining = (cfg.epochs - ep) * (elapsed / ep)
        eta_str = fmt_eta(remaining)

        print(f"[{cfg.dataset}|{backbone_name}|{method_name}] "
              f"ep {ep:03d}/{cfg.epochs}  acc={acc*100:.2f}%  "
              f"ECE={ece:.4f}  AdaECE={adae:.4f}  ClsECE={cwe:.4f}  NLL={nll:.3f}  "
              f"| TS: T={last_T:.3f} ECE_T={ece_t:.4f} NLL_T={nll_t:.3f}  "
              f"[time/epoch={epoch_time:.1f}s | ETA={eta_str}]")
    return best, last_T

# ---------------------- Main pipeline ----------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10","cifar100"])
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=5)  # keep small for reviewers; they can increase
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-dir", type=str, default=".", help="Directory to save artifacts and markers.")
    p.add_argument("--log-csv", type=str, default="methods_log.csv")
    p.add_argument("--backbones", type=str, default="resnet50,resnet110,wideresnet28x10,densenet121")
    p.add_argument("--ts-every", type=int, default=1)
    p.add_argument("--skip-if-done", action="store_true", default=True, help="Skip a run if its done marker exists")
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    def expand_out(pth): return pth if os.path.isabs(pth) else os.path.join(args.out_dir, pth)
    default_log = args.log_csv if args.log_csv != "methods_log.csv" else f"{args.dataset}_methods_log.csv"

    cfg = TrainConfig(dataset=args.dataset.lower(), epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      weight_decay=args.weight_decay, num_workers=args.num_workers, skip_if_done=args.skip_if_done,
                      out_dir=args.out_dir, ts_every=max(1, args.ts_every))
    try:
        cfg.amp = bool(getattr(cfg, "amp", True)) and (device == "cuda")
    except Exception:
        cfg.amp = False

    train_loader, val_loader, test_loader, num_classes = get_cifar_loaders(
        cfg.dataset, args.data, batch_size=cfg.batch_size, num_workers=args.num_workers, val_size=5000, seed=args.seed)
    cfg.num_classes = num_classes

    cfg.log_csv = expand_out(default_log)
    init_csv(cfg.log_csv, header=[
        "dataset", "num_classes", "backbone", "method", "epoch",
        "test_acc", "test_ece", "test_adaece", "test_classwise_ece", "test_nll",
        "test_acc_ts", "test_ece_ts", "test_adaece_ts", "test_classwise_ece_ts", "test_nll_ts",
        "lambda_if_any", "val_smooth_ece_if_any", "temperature_T"
    ])

    def run_guard(bname, method_name, criterion):
        run_key = f"{cfg.dataset}|{bname}|{method_name}"
        done_path = os.path.join(cfg.out_dir, f"done_{cfg.dataset}_{bname}_{method_name}.json")
        if cfg.skip_if_done and os.path.exists(done_path):
            print(f"[SKIP-DONE] {run_key} — done marker found at {done_path}")
            with open(done_path, "r") as f:
                return json.load(f)

        lock_path = acquire_run_lock(cfg.out_dir, run_key, stale_hours=48)
        if lock_path is None:
            print(f"[SKIP-LOCK] {run_key} — another worker is training (lock present).")
            return {"skipped": "locked", "run_key": run_key}

        try:
            model = make_backbone(bname, num_classes=cfg.num_classes).to(device)
            print(f"\n=== [{cfg.dataset}|{bname}] Training: {method_name} ===")
            best, _ = train_standard(bname, model, train_loader, val_loader, test_loader, cfg, criterion, device, method_name)

            logits_test, labels_test = eval_logits(model, test_loader, device)
            mwT = ModelWithTemperature(model, device=device)
            _ = mwT.set_temperature(val_loader, max_iter=50, lr=0.01)
            _, _, _, _, _, probs_before = metrics_from_logits(logits_test, labels_test, n_bins=cfg.ece_bins, num_classes=cfg.num_classes, temperature=None)
            _, _, _, _, _, probs_after  = metrics_from_logits(logits_test, labels_test, n_bins=cfg.ece_bins, num_classes=cfg.num_classes, temperature=mwT.temperature.detach())
            out_rel = os.path.join(cfg.out_dir, f"reliability_{cfg.dataset}_{bname}_{method_name}.png")
            reliability_diagram(probs_before, probs_after, labels_test, out_rel, n_bins=cfg.ece_bins,
                                title=f"Reliability: {cfg.dataset}|{bname}|{method_name} (T={float(mwT.temperature.detach().cpu()):.3f})")
            summary = {"acc": best["acc"], "ece": best["ece"], "adaece": best["adaece"], "classwise": best["classwise"], "nll": best["nll"]}
            with open(done_path, "w") as f: json.dump(summary, f)
            return summary
        except Exception as e:
            err_path = os.path.join(cfg.out_dir, f"error_{cfg.dataset}_{bname}_{method_name}.txt")
            with open(err_path, "w") as f:
                f.write("".join(traceback.format_exc()))
            print(f"[ERROR] {run_key} crashed. Traceback saved to {err_path}. Continuing...")
            return {"error": str(e), "run_key": run_key}
        finally:
            release_run_lock(lock_path)

    # A compact set of methods to keep runtime modest for reviewers.
    results: Dict[str, Dict] = {}
    backbone_list = [s.strip() for s in args.backbones.split(",") if s.strip()]
    # One or two methods; add more if desired
    for bname in backbone_list:
        results[f"{bname}|CE+WD"]       = run_guard(bname, "CE+WD", nn.CrossEntropyLoss())
        results[f"{bname}|Brier"]       = run_guard(bname, "Brier", BrierScorePure())

    print("\nAll runs finished. Done markers live under:", args.out_dir)
    print("Per-epoch metrics logged to:", cfg.log_csv)

if __name__ == "__main__":
    main()
