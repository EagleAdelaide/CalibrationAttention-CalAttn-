import argparse, os, json, math
from copy import deepcopy
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.io import load_yaml, ensure_dir
from utils.seed import set_seed
from utils.logger import Logger
from utils.meters import AverageMeter

from data.loaders import build_datasets, build_transforms
from data.splits import make_train_val_split

from models.factory import build_model_and_wrapper
from losses.ce import CrossEntropyLS
from losses.brier import BrierLoss
from losses.mmce import MMCE
from losses.focal_flsd53 import FocalFLSD53
from losses.dual_focal import DualFocalLoss

from metrics.calibration import (
    accuracy_top1, nll_loss_from_logits, ece_equal_width
)


def build_optimizer(cfg, params):
    opt_name = cfg["optim"]["optimizer"].lower()
    lr = cfg["optim"]["lr"]
    wd = cfg["optim"]["weight_decay"]
    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=cfg["optim"].get("momentum", 0.9), weight_decay=wd, nesterov=True)
    elif opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(cfg, optimizer):
    sch = cfg["sched"]
    if sch["type"] == "step":
        milestones = sch["milestones"]
        gamma = sch["gamma"]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    raise ValueError(f"Unknown scheduler: {sch['type']}")


def build_criterion(cfg):
    method = cfg["method"]["name"].lower()
    lam = float(cfg["method"].get("lambda_brier", 0.1))
    ls = float(cfg["method"].get("label_smoothing", 0.0))

    ce = CrossEntropyLS(label_smoothing=ls)
    brier = BrierLoss()
    mmce = MMCE()
    focal = FocalFLSD53()
    dfl = DualFocalLoss(gamma=2.0)

    def loss_fn(logits, y, extra=None):
        # logits are already "final logits" from wrapper (may include scaling head effect)
        if method == "ce":
            return ce(logits, y)
        if method == "ce_ls":
            return ce(logits, y)
        if method == "brier":
            probs = torch.softmax(logits, dim=-1)
            return brier(probs, y)
        if method == "ce_brier":
            probs = torch.softmax(logits, dim=-1)
            return ce(logits, y) + lam * brier(probs, y)
        if method == "mmce":
            probs = torch.softmax(logits, dim=-1)
            return ce(logits, y) + 0.1 * mmce(probs, y)  # weight exposed if needed
        if method == "focal_flsd53":
            # focal uses probabilities internally
            probs = torch.softmax(logits, dim=-1)
            return focal(probs, y)
        if method == "dual_focal":
            probs = torch.softmax(logits, dim=-1)
            return dfl(probs, y)
        if method in ("relaxed_softmax", "calattn"):
            # wrapper already applied scaling; train with CE + lambda*Brier as in paper
            probs = torch.softmax(logits, dim=-1)
            return ce(logits, y) + lam * brier(probs, y)
        raise ValueError(f"Unknown method: {method}")

    return loss_fn


@torch.no_grad()
def evaluate(model, loader, device, ece_bins=15):
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)

    top1 = accuracy_top1(logits, y) * 100.0
    nll = nll_loss_from_logits(logits, y)
    ece = ece_equal_width(logits, y, n_bins=ece_bins) * 100.0
    return {"top1": float(top1), "nll": float(nll), "ece": float(ece)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_dir = os.path.join(cfg["run"]["out_dir"], cfg["run"]["name"], f"seed{cfg['run']['seed']}")
    ensure_dir(out_dir)
    logger = Logger(os.path.join(out_dir, "train.log"))

    set_seed(int(cfg["run"]["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Device: {device}")

    train_tf, test_tf = build_transforms(cfg)
    ds_train_full, ds_test = build_datasets(cfg, train_tf, test_tf)
    ds_train, ds_val = make_train_val_split(ds_train_full, val_ratio=float(cfg["data"]["val_split"]), seed=int(cfg["run"]["seed"]))

    train_loader = DataLoader(ds_train, batch_size=int(cfg["data"]["batch_size"]), shuffle=True,
                              num_workers=int(cfg["data"]["num_workers"]), pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=int(cfg["data"]["batch_size"]), shuffle=False,
                            num_workers=int(cfg["data"]["num_workers"]), pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=int(cfg["data"]["batch_size"]), shuffle=False,
                             num_workers=int(cfg["data"]["num_workers"]), pin_memory=True)

    model = build_model_and_wrapper(cfg).to(device)

    optimizer = build_optimizer(cfg, model.parameters())
    scheduler = build_scheduler(cfg, optimizer)
    loss_fn = build_criterion(cfg)

    best_val_ece = 1e9
    best_path = os.path.join(out_dir, "best.pt")

    epochs = int(cfg["sched"]["epochs"])
    ece_bins = int(cfg["eval"].get("ts_bins", 15))

    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=100)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_meter.update(loss.item(), n=x.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.3g}")

        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, ece_bins=ece_bins)
        logger.log(f"[Epoch {epoch}] Val: top1={val_metrics['top1']:.2f}  nll={val_metrics['nll']:.4f}  ece={val_metrics['ece']:.2f}")

        if val_metrics["ece"] < best_val_ece:
            best_val_ece = val_metrics["ece"]
            torch.save({"cfg": cfg, "model": model.state_dict()}, best_path)
            logger.log(f"Saved best checkpoint to {best_path} (val ECE={best_val_ece:.2f})")

    # Final test with best model
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device, ece_bins=ece_bins)
    logger.log(f"[Best] Test: top1={test_metrics['top1']:.2f}  nll={test_metrics['nll']:.4f}  ece={test_metrics['ece']:.2f}")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"best_val_ece": best_val_ece, "test": test_metrics}, f, indent=2)

    logger.log("Done.")


if __name__ == "__main__":
    main()
