# import argparse, json
# import torch
# from torch import nn
# from torch.utils.data import DataLoader

# from utils.io import load_yaml
# from data.loaders import build_datasets, build_transforms
# from data.splits import make_train_val_split
# from models.factory import build_model_and_wrapper
# from metrics.calibration import ece_equal_width, nll_loss_from_logits, accuracy_top1


# class SATSRegressor(nn.Module):
#     """Post-hoc per-sample temperature regressor on frozen logits."""
#     def __init__(self, num_classes: int, hidden: int = 128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(num_classes, hidden),
#             nn.GELU(),
#             nn.Linear(hidden, 1),
#             nn.Softplus()
#         )

#     def forward(self, logits):
#         T = self.net(logits) + 1e-6
#         return T


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", required=True)
#     ap.add_argument("--ckpt", required=True)
#     ap.add_argument("--epochs", type=int, default=10)
#     ap.add_argument("--lr", type=float, default=1e-3)
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
#     model.eval()

#     # Gather logits
#     @torch.no_grad()
#     def gather(loader):
#         L, Y = [], []
#         for x, y in loader:
#             x, y = x.to(device), y.to(device)
#             logits = model(x)
#             L.append(logits.detach())
#             Y.append(y.detach())
#         return torch.cat(L, 0), torch.cat(Y, 0)

#     val_logits, val_y = gather(val_loader)
#     test_logits, test_y = gather(test_loader)

#     sats = SATSRegressor(num_classes=int(cfg["model"]["num_classes"]),
#                          hidden=int(cfg["method"].get("relaxed_hidden", 128))).to(device)
#     opt = torch.optim.AdamW(sats.parameters(), lr=args.lr, weight_decay=0.0)

#     best_ece = 1e9
#     best_state = None
#     n_bins = int(cfg["eval"].get("ts_bins", 15))

#     for ep in range(1, args.epochs + 1):
#         sats.train()
#         opt.zero_grad(set_to_none=True)
#         T = sats(val_logits)
#         scaled = val_logits / T
#         loss = nn.CrossEntropyLoss()(scaled, val_y)
#         loss.backward()
#         opt.step()

#         sats.eval()
#         with torch.no_grad():
#             T = sats(val_logits)
#             ece = ece_equal_width(val_logits / T, val_y, n_bins=n_bins)
#             if ece < best_ece:
#                 best_ece = ece
#                 best_state = {k: v.detach().cpu().clone() for k, v in sats.state_dict().items()}

#     sats.load_state_dict(best_state)
#     sats.eval()

#     # Report on test
#     pre = {
#         "top1": float(accuracy_top1(test_logits, test_y) * 100.0),
#         "nll": float(nll_loss_from_logits(test_logits, test_y)),
#         "ece": float(ece_equal_width(test_logits, test_y, n_bins=n_bins) * 100.0),
#     }
#     with torch.no_grad():
#         Tt = sats(test_logits)
#         post_logits = test_logits / Tt
#     post = {
#         "top1": float(accuracy_top1(post_logits, test_y) * 100.0),
#         "nll": float(nll_loss_from_logits(post_logits, test_y)),
#         "ece": float(ece_equal_width(post_logits, test_y, n_bins=n_bins) * 100.0),
#     }

#     out = {"val_best_ece": float(best_ece * 100.0), "test_pre": pre, "test_post": post}
#     print(json.dumps(out, indent=2))


# if __name__ == "__main__":
#     main()
# # 

import argparse, json
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.io import load_yaml
from data.loaders import build_datasets, build_transforms
from data.splits import make_train_val_split
from models.factory import build_model_and_wrapper

from metrics.calibration import (
    ece_equal_width,                 # legacy ECE in [0,1] for selection
    full_eval_summary_from_logits,   # full metrics in percentage points for reporting
)


class SATSRegressor(nn.Module):
    """
    Post-hoc per-sample temperature regressor on frozen logits (SATS-style).
    Predicts positive temperature T(x) via Softplus.
    """
    def __init__(self, num_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Softplus()
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [B, C] -> T: [B, 1]
        T = self.net(logits) + 1e-6
        return T


@torch.no_grad()
def gather_logits_to_cpu(model, loader, device):
    model.eval()
    L, Y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        L.append(logits.detach().cpu())
        Y.append(y.detach().cpu())
    return torch.cat(L, 0), torch.cat(Y, 0)


@torch.no_grad()
def sats_eval_ece(sats, logits_cpu, y_cpu, device, n_bins: int, batch_size: int = 4096):
    """
    Compute ECE (equal-width) on CPU-stored logits/y using SATS temperatures.
    Returns ECE in [0,1].
    """
    sats.eval()
    ds = TensorDataset(logits_cpu, y_cpu)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_scaled = []
    all_y = []
    for logits, y in loader:
        logits = logits.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        T = sats(logits)              # [B,1]
        scaled = logits / T           # [B,C]
        all_scaled.append(scaled.detach().cpu())
        all_y.append(y.detach().cpu())

    scaled_logits = torch.cat(all_scaled, 0)
    y_all = torch.cat(all_y, 0)
    return float(ece_equal_width(scaled_logits, y_all, n_bins=n_bins))


@torch.no_grad()
def sats_apply(model, sats, loader, device):
    """
    Run base model -> logits, then SATS -> temperature, return scaled logits and y on CPU.
    """
    model.eval()
    sats.eval()
    L, Y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)                 # [B,C]
        T = sats(logits)                  # [B,1]
        scaled = logits / T               # [B,C]
        L.append(scaled.detach().cpu())
        Y.append(y.detach().cpu())
    return torch.cat(L, 0), torch.cat(Y, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=None, help="Override SATS train batch size (on logits).")
    ap.add_argument("--hidden", type=int, default=None, help="Override SATS hidden width.")
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

    # ---------------- base model ----------------
    model = build_model_and_wrapper(cfg).to(device)
    # ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # Gather logits for SATS training (store on CPU)
    val_logits_cpu, val_y_cpu = gather_logits_to_cpu(model, val_loader, device)
    test_logits_cpu, test_y_cpu = gather_logits_to_cpu(model, test_loader, device)

    # ---------------- SATS regressor ----------------
    num_classes = int(cfg["model"]["num_classes"])
    hidden = int(args.hidden if args.hidden is not None else cfg["method"].get("sats_hidden", cfg["method"].get("relaxed_hidden", 128)))
    sats = SATSRegressor(num_classes=num_classes, hidden=hidden).to(device)

    opt = torch.optim.AdamW(sats.parameters(), lr=args.lr, weight_decay=0.0)
    ce = nn.CrossEntropyLoss()

    n_bins = int(cfg["eval"].get("ts_bins", 15))
    sats_bs = int(args.batch_size if args.batch_size is not None else cfg["eval"].get("sats_batch_size", 4096))

    # SATS trains post-hoc on (logits, y) from val split
    train_ds = TensorDataset(val_logits_cpu, val_y_cpu)
    train_loader = DataLoader(train_ds, batch_size=sats_bs, shuffle=True, num_workers=0, drop_last=False)

    best_ece = float("inf")
    best_state = None
    best_epoch = -1

    for ep in range(1, args.epochs + 1):
        sats.train()
        loss_meter = 0.0
        n_seen = 0

        for logits, y in tqdm(train_loader, desc=f"SATS epoch {ep}/{args.epochs}", ncols=100):
            logits = logits.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            T = sats(logits)            # [B,1]
            scaled = logits / T         # [B,C]
            loss = ce(scaled, y)
            loss.backward()
            opt.step()

            bs = y.size(0)
            loss_meter += float(loss.item()) * bs
            n_seen += bs

        # Selection metric: validation ECE after SATS scaling (ECE in [0,1])
        val_ece = sats_eval_ece(sats, val_logits_cpu, val_y_cpu, device, n_bins=n_bins, batch_size=sats_bs)
        avg_loss = loss_meter / max(1, n_seen)

        print(json.dumps({
            "epoch": ep,
            "train_nll": avg_loss,
            "val_ece_percent": val_ece * 100.0
        }, indent=2))

        if val_ece < best_ece:
            best_ece = val_ece
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in sats.state_dict().items()}

    # Restore best SATS
    assert best_state is not None, "SATS training produced no best_state (unexpected)."
    sats.load_state_dict(best_state, strict=True)
    sats.eval()

    # --------- Reporting: pre / post SATS on test (full metrics) ---------
    # Pre: base logits
    pre = full_eval_summary_from_logits(
        test_logits_cpu, test_y_cpu,
        n_bins=n_bins,
        smece_bandwidth=float(cfg["eval"].get("smece_bw", 0.05)),
        smece_max_points=int(cfg["eval"].get("smece_max_points", 5000)) if cfg["eval"].get("smece_max_points", 5000) is not None else None,
    )

    # Post: SATS-scaled logits (compute through model+sats on test_loader for correctness)
    post_logits_cpu, post_y_cpu = sats_apply(model, sats, test_loader, device)
    post = full_eval_summary_from_logits(
        post_logits_cpu, post_y_cpu,
        n_bins=n_bins,
        smece_bandwidth=float(cfg["eval"].get("smece_bw", 0.05)),
        smece_max_points=int(cfg["eval"].get("smece_max_points", 5000)) if cfg["eval"].get("smece_max_points", 5000) is not None else None,
    )

    out = {
        "sats": {
            "hidden": hidden,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": sats_bs,
            "best_epoch": best_epoch,
            "val_best_ece_percent": best_ece * 100.0,
        },
        "test_pre": pre,
        "test_post": post,
        "settings": {
            "ts_bins": n_bins,
            "smece_bw": float(cfg["eval"].get("smece_bw", 0.05)),
            "smece_max_points": cfg["eval"].get("smece_max_points", 5000),
        }
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
