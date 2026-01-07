import argparse, json
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.io import load_yaml
from data.loaders import build_datasets, build_transforms
from data.splits import make_train_val_split
from models.factory import build_model_and_wrapper
from metrics.calibration import ece_equal_width, nll_loss_from_logits, accuracy_top1


class SATSRegressor(nn.Module):
    """Post-hoc per-sample temperature regressor on frozen logits."""
    def __init__(self, num_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Softplus()
        )

    def forward(self, logits):
        T = self.net(logits) + 1e-6
        return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf, test_tf = build_transforms(cfg)
    ds_train_full, ds_test = build_datasets(cfg, train_tf, test_tf)
    _, ds_val = make_train_val_split(ds_train_full, val_ratio=float(cfg["data"]["val_split"]), seed=int(cfg["run"]["seed"]))

    val_loader = DataLoader(ds_val, batch_size=int(cfg["data"]["batch_size"]), shuffle=False,
                            num_workers=int(cfg["data"]["num_workers"]), pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=int(cfg["data"]["batch_size"]), shuffle=False,
                             num_workers=int(cfg["data"]["num_workers"]), pin_memory=True)

    model = build_model_and_wrapper(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Gather logits
    @torch.no_grad()
    def gather(loader):
        L, Y = [], []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            L.append(logits.detach())
            Y.append(y.detach())
        return torch.cat(L, 0), torch.cat(Y, 0)

    val_logits, val_y = gather(val_loader)
    test_logits, test_y = gather(test_loader)

    sats = SATSRegressor(num_classes=int(cfg["model"]["num_classes"]),
                         hidden=int(cfg["method"].get("relaxed_hidden", 128))).to(device)
    opt = torch.optim.AdamW(sats.parameters(), lr=args.lr, weight_decay=0.0)

    best_ece = 1e9
    best_state = None
    n_bins = int(cfg["eval"].get("ts_bins", 15))

    for ep in range(1, args.epochs + 1):
        sats.train()
        opt.zero_grad(set_to_none=True)
        T = sats(val_logits)
        scaled = val_logits / T
        loss = nn.CrossEntropyLoss()(scaled, val_y)
        loss.backward()
        opt.step()

        sats.eval()
        with torch.no_grad():
            T = sats(val_logits)
            ece = ece_equal_width(val_logits / T, val_y, n_bins=n_bins)
            if ece < best_ece:
                best_ece = ece
                best_state = {k: v.detach().cpu().clone() for k, v in sats.state_dict().items()}

    sats.load_state_dict(best_state)
    sats.eval()

    # Report on test
    pre = {
        "top1": float(accuracy_top1(test_logits, test_y) * 100.0),
        "nll": float(nll_loss_from_logits(test_logits, test_y)),
        "ece": float(ece_equal_width(test_logits, test_y, n_bins=n_bins) * 100.0),
    }
    with torch.no_grad():
        Tt = sats(test_logits)
        post_logits = test_logits / Tt
    post = {
        "top1": float(accuracy_top1(post_logits, test_y) * 100.0),
        "nll": float(nll_loss_from_logits(post_logits, test_y)),
        "ece": float(ece_equal_width(post_logits, test_y, n_bins=n_bins) * 100.0),
    }

    out = {"val_best_ece": float(best_ece * 100.0), "test_pre": pre, "test_post": post}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
