# calattn-repro/src/eval.py
import argparse, json
import torch
from torch.utils.data import DataLoader

from utils.io import load_yaml
from data.loaders import build_datasets, build_transforms
from models.factory import build_model_and_wrapper

from metrics.calibration import (
    accuracy_top1,
    nll_loss_from_logits,
    ece_equal_width,
    full_eval_summary_from_logits,
)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_tf = build_transforms(cfg)
    _, ds_test = build_datasets(cfg, train_tf=None, test_tf=test_tf)
    loader = DataLoader(
        ds_test,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=True,
    )

    model = build_model_and_wrapper(cfg).to(device)
    # ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits = torch.cat(all_logits, 0)
    y = torch.cat(all_y, 0)

    # bins config (keep your existing key name)
    n_bins = int(cfg.get("eval", {}).get("ts_bins", 15))

    # keep legacy fields (so your scripts won't break)
    top1_legacy = accuracy_top1(logits, y) * 100.0
    nll_legacy = nll_loss_from_logits(logits, y)
    ece_legacy = ece_equal_width(logits, y, n_bins=n_bins) * 100.0

    # full metrics for paper tables
    smece_bw = float(cfg.get("eval", {}).get("smece_bandwidth", 0.05))
    smece_max_points = cfg.get("eval", {}).get("smece_max_points", 5000)
    if smece_max_points is not None:
        smece_max_points = int(smece_max_points)

    full = full_eval_summary_from_logits(
        logits,
        y,
        n_bins=n_bins,
        smece_bandwidth=smece_bw,
        smece_max_points=smece_max_points,
    )

    # merge (legacy keys first; full metrics include top1/nll/brier etc.)
    res = {
        "top1": float(top1_legacy),
        "nll": float(nll_legacy),
        "ece": float(ece_legacy),
        **full,  # includes: top1, nll, brier, ece, mce, adaece, classece, smece (lowercase keys)
    }

    # if you prefer legacy top1/nll/ece to override, swap order.
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
