# import argparse, json, math
# import numpy as np
# import torch
# from torch.utils.data import DataLoader

# from utils.io import load_yaml
# from data.loaders import build_datasets, build_transforms
# from data.splits import make_train_val_split
# from models.factory import build_model_and_wrapper
# from metrics.calibration import ece_equal_width, nll_loss_from_logits, accuracy_top1


# @torch.no_grad()
# def gather_logits(model, loader, device):
#     model.eval()
#     L, Y = [], []
#     for x, y in loader:
#         x, y = x.to(device), y.to(device)
#         logits = model(x)
#         L.append(logits.detach().cpu())
#         Y.append(y.detach().cpu())
#     return torch.cat(L, 0), torch.cat(Y, 0)


# def apply_temp(logits, T):
#     return logits / T


# def grid_T(cfg):
#     start, end, step = cfg["eval"]["ts_grid"]
#     Ts = []
#     t = start
#     while t <= end + 1e-9:
#         Ts.append(round(t, 10))
#         t += step
#     return Ts


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", required=True)
#     ap.add_argument("--ckpt", required=True)
#     args = ap.parse_args()

#     cfg = load_yaml(args.config)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     train_tf, test_tf = build_transforms(cfg)
#     ds_train_full, ds_test = build_datasets(cfg, train_tf, test_tf)
#     _, ds_val = make_train_val_split(ds_train_full, val_ratio=float(cfg["data"]["val_split"]), seed=int(cfg["run"]["seed"]))

#     val_loader = DataLoader(ds_val, batch_size=int(cfg["data"]["batch_size"]), shuffle=False,
#                             num_workers=int(cfg["data"]["num_workers"]), pin_memory=True)
#     test_loader = DataLoader(ds_test, batch_size=int(cfg["data"]["batch_size"]), shuffle=False,
#                              num_workers=int(cfg["data"]["num_workers"]), pin_memory=True)

#     model = build_model_and_wrapper(cfg).to(device)
#     ckpt = torch.load(args.ckpt, map_location="cpu")
#     model.load_state_dict(ckpt["model"])

#     val_logits, val_y = gather_logits(model, val_loader, device)
#     test_logits, test_y = gather_logits(model, test_loader, device)

#     n_bins = int(cfg["eval"].get("ts_bins", 15))
#     Ts = grid_T(cfg)

#     best_T, best_ece = None, 1e9
#     for T in Ts:
#         ece = ece_equal_width(apply_temp(val_logits, T), val_y, n_bins=n_bins)
#         if ece < best_ece:
#             best_ece = ece
#             best_T = T

#     # Report pre and post TS on test
#     pre = {
#         "top1": float(accuracy_top1(test_logits, test_y) * 100.0),
#         "nll": float(nll_loss_from_logits(test_logits, test_y)),
#         "ece": float(ece_equal_width(test_logits, test_y, n_bins=n_bins) * 100.0),
#     }
#     post_logits = apply_temp(test_logits, best_T)
#     post = {
#         "top1": float(accuracy_top1(post_logits, test_y) * 100.0),
#         "nll": float(nll_loss_from_logits(post_logits, test_y)),
#         "ece": float(ece_equal_width(post_logits, test_y, n_bins=n_bins) * 100.0),
#     }

#     out = {"best_T_ece": best_T, "val_ece": float(best_ece * 100.0), "test_pre": pre, "test_post": post}
#     print(json.dumps(out, indent=2))


# if __name__ == "__main__":
#     main()

import argparse, json
import torch
from torch.utils.data import DataLoader

from utils.io import load_yaml
from data.loaders import build_datasets, build_transforms
from data.splits import make_train_val_split
from models.factory import build_model_and_wrapper

from metrics.calibration import (
    ece_equal_width,                 # legacy ECE in [0,1]
    nll_loss_from_logits,            # scalar
    full_eval_summary_from_logits,   # returns top1/nll/brier + (ece,mce,adaece,classece,smece) in %
)


@torch.no_grad()
def gather_logits(model, loader, device):
    model.eval()
    L, Y = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        L.append(logits.detach().cpu())
        Y.append(y.detach().cpu())
    return torch.cat(L, 0), torch.cat(Y, 0)


def apply_temp(logits, T: float):
    return logits / float(T)


def grid_T(cfg):
    # expects: eval.ts_grid: [start, end, step]
    start, end, step = cfg["eval"]["ts_grid"]
    start, end, step = float(start), float(end), float(step)
    Ts, t = [], start
    while t <= end + 1e-12:
        Ts.append(round(t, 10))
        t += step
    return Ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    # keep your paper protocol by default: tune T via validation ECE (equal-width bins)
    ap.add_argument("--tune_on", default="ece", choices=["ece", "nll"])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- data ----------------
    train_tf, test_tf = build_transforms(cfg)
    ds_train_full, ds_test = build_datasets(cfg, train_tf, test_tf)
    _, ds_val = make_train_val_split(
        ds_train_full,
        val_ratio=float(cfg["data"]["val_split"]),
        seed=int(cfg["run"]["seed"]),
    )

    val_loader = DataLoader(
        ds_val,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=True,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=True,
    )

    # ---------------- model ----------------
    model = build_model_and_wrapper(cfg).to(device)
    # ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"], strict=True)

    val_logits, val_y = gather_logits(model, val_loader, device)
    test_logits, test_y = gather_logits(model, test_loader, device)

    n_bins = int(cfg["eval"].get("ts_bins", 15))
    smece_bw = float(cfg["eval"].get("smece_bw", 0.05))
    smece_max_points = cfg["eval"].get("smece_max_points", 5000)
    if smece_max_points is not None:
        smece_max_points = int(smece_max_points)

    Ts = grid_T(cfg)

    # --------------- select best T on val ---------------
    best_T, best_score = None, float("inf")
    for T in Ts:
        scaled = apply_temp(val_logits, T)

        if args.tune_on == "ece":
            # ece_equal_width returns [0,1]
            score = float(ece_equal_width(scaled, val_y, n_bins=n_bins))
        else:
            score = float(nll_loss_from_logits(scaled, val_y))

        if score < best_score:
            best_score = score
            best_T = float(T)

    # --------------- report test metrics (pre/post) ---------------
    pre = full_eval_summary_from_logits(
        test_logits, test_y,
        n_bins=n_bins,
        smece_bandwidth=smece_bw,
        smece_max_points=smece_max_points,
    )

    post_logits = apply_temp(test_logits, best_T)
    post = full_eval_summary_from_logits(
        post_logits, test_y,
        n_bins=n_bins,
        smece_bandwidth=smece_bw,
        smece_max_points=smece_max_points,
    )

    out = {
        "tuned_on": args.tune_on,
        "best_T": best_T,
        # for ECE we store as percent in the output for readability
        "val_best_score": best_score * (100.0 if args.tune_on == "ece" else 1.0),
        "test_pre": pre,
        "test_post": post,
        "settings": {
            "ts_bins": n_bins,
            "smece_bw": smece_bw,
            "smece_max_points": smece_max_points,
            "ts_grid": cfg["eval"]["ts_grid"],
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
