import argparse, os, json, time
from datetime import timedelta

import torch
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

from metrics.calibration import accuracy_top1, nll_loss_from_logits, ece_equal_width


def build_optimizer(cfg, params):
    opt_name = cfg["optim"]["optimizer"].lower()
    lr = float(cfg["optim"]["lr"])
    wd = float(cfg["optim"]["weight_decay"])
    if opt_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(cfg["optim"].get("momentum", 0.9)),
            weight_decay=wd,
            nesterov=True,
        )
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

    def loss_fn(logits, y):
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
            return ce(logits, y) + 0.1 * mmce(probs, y)
        if method == "focal_flsd53":
            probs = torch.softmax(logits, dim=-1)
            return focal(probs, y)
        if method == "dual_focal":
            probs = torch.softmax(logits, dim=-1)
            return dfl(probs, y)
        if method in ("relaxed_softmax", "calattn"):
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


def _fmt_td(seconds: float) -> str:
    return str(timedelta(seconds=int(max(0, seconds))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--amp", action="store_true", help="Enable torch autocast mixed precision (CUDA only).")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg["run"]["seed"])
    run_name = cfg["run"]["name"]
    out_dir = os.path.join(cfg["run"]["out_dir"], run_name, f"seed{seed}")
    ensure_dir(out_dir)
    logger = Logger(os.path.join(out_dir, "train.log"))

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logger.log(f"Run: {run_name} | seed={seed}")
    logger.log(f"Device: {device} | AMP: {use_amp}")

    # Data
    train_tf, test_tf = build_transforms(cfg)
    ds_train_full, ds_test = build_datasets(cfg, train_tf, test_tf)
    ds_train, ds_val = make_train_val_split(
        ds_train_full,
        val_ratio=float(cfg["data"]["val_split"]),
        seed=seed
    )

    bs = int(cfg["data"]["batch_size"])
    nw = int(cfg["data"]["num_workers"])

    train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(ds_val,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(ds_test,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    steps_per_epoch = len(train_loader)
    logger.log(f"Data: train={len(ds_train)} | val={len(ds_val)} | test={len(ds_test)}")
    logger.log(f"Loader: batch_size={bs} | workers={nw} | steps/epoch={steps_per_epoch}")

    # Model / optim
    model = build_model_and_wrapper(cfg).to(device)
    optimizer = build_optimizer(cfg, model.parameters())
    scheduler = build_scheduler(cfg, optimizer)
    loss_fn = build_criterion(cfg)

    epochs = int(cfg["sched"]["epochs"])
    ece_bins = int(cfg["eval"].get("ts_bins", 15))
    grad_clip = float(cfg["optim"].get("grad_clip", 5.0))

    best_val_ece = 1e9
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")

    # Training loop
    t0_all = time.perf_counter()
    ema_epoch_sec = None  # for smoother ETA

    for epoch in range(1, epochs + 1):

        model.train()
        loss_meter = AverageMeter()
        data_time = AverageMeter()
        iter_time = AverageMeter()

        t0_epoch = time.perf_counter()
        end = time.perf_counter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=120, leave=True)
        for it, (x, y) in enumerate(pbar, start=1):
            data_time.update(time.perf_counter() - end)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # with torch.cuda.amp.autocast(enabled=use_amp):
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), n=x.size(0))

            # timing
            iter_dt = time.perf_counter() - end
            iter_time.update(iter_dt)
            end = time.perf_counter()

            # epoch ETA
            rem_steps = steps_per_epoch - it
            eta_epoch = rem_steps * iter_time.avg

            # GPU memory (optional)
            mem = ""
            if device.type == "cuda":
                mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                mem = f", mem={mem_mb:.0f}MB"

            pbar.set_postfix_str(
                f"loss={loss_meter.avg:.4f}, lr={optimizer.param_groups[0]['lr']:.3g}, "
                f"dt={iter_time.avg:.3f}s, data={data_time.avg:.3f}s, eta={_fmt_td(eta_epoch)}{mem}"
            )

        scheduler.step()

        # epoch timing + overall ETA
        epoch_sec = time.perf_counter() - t0_epoch
        if ema_epoch_sec is None:
            ema_epoch_sec = epoch_sec
        else:
            ema_epoch_sec = 0.7 * ema_epoch_sec + 0.3 * epoch_sec

        rem_epochs = epochs - epoch
        eta_total = rem_epochs * ema_epoch_sec

        val_metrics = evaluate(model, val_loader, device, ece_bins=ece_bins)
        logger.log(
            f"[Epoch {epoch:03d}/{epochs}] "
            f"train_loss={loss_meter.avg:.4f} | "
            f"val_top1={val_metrics['top1']:.2f} val_nll={val_metrics['nll']:.4f} val_ece={val_metrics['ece']:.2f} | "
            f"epoch_time={_fmt_td(epoch_sec)} eta_total={_fmt_td(eta_total)}"
        )

        # save last
        # torch.save({"cfg": cfg, "model": model.state_dict(), "epoch": epoch}, last_path)

        # save best
        if val_metrics["ece"] < best_val_ece:
            best_val_ece = val_metrics["ece"]
            torch.save({"cfg": cfg, "model": model.state_dict(), "epoch": epoch}, best_path)
            logger.log(f"Saved best checkpoint: {best_path} (val ECE={best_val_ece:.2f})")

    # Final test with best model
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device, ece_bins=ece_bins)

    total_sec = time.perf_counter() - t0_all
    logger.log(f"[Best] Test: top1={test_metrics['top1']:.2f}  nll={test_metrics['nll']:.4f}  ece={test_metrics['ece']:.2f}")
    logger.log(f"Total time: {_fmt_td(total_sec)}")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"best_val_ece": best_val_ece, "test": test_metrics}, f, indent=2)

    logger.log("Done.")


if __name__ == "__main__":
    main()
