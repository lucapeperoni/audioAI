"""
train.py – ResNet-18 per GTZAN, ora multi-rappresentazione (mel / lin / mfcc / dmfcc)

Salva:
    • best_resnet18_<rep>_fold{fold}.pt
    • log_<rep>_fold{fold}.txt
"""

from __future__ import annotations
import os, time, json, random, argparse, torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from dataset import MelDataset

# ─── hyper ───
EPOCHS     = 120
BATCH_SIZE = 8
LR         = 1e-4
EARLY_STOP = 60      # epoche senza migliorare

# ───────────────────────────────────────── helpers ────────────────────────────
def _get_device() -> torch.device:
    """Prefer MPS ▸ CUDA ▸ CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_resnet18(device: torch.device, in_channels: int = 1) -> nn.Module:
    """ResNet-18 adattata a in_channels e 10 classi."""
    w = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=w)

    # conv1 a in_channels: media dei pesi RGB o duplica
    old_w = model.conv1.weight.clone()
    model.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
    if in_channels == 1:
        model.conv1.weight.data = old_w.mean(1, keepdim=True)
    elif in_channels == 2:
        model.conv1.weight.data = old_w[:, :2].clone()
    else:  # in_channels > 3 → ripeti/trimma
        k = in_channels
        rep = (old_w.mean(1, keepdim=True)).repeat(1, k, 1, 1)
        model.conv1.weight.data = rep[:, :k]

    model.fc = nn.Linear(512, 10)
    return model.to(device)


# ──────────────────────────────── API principale ─────────────────────────────
def train_one_fold(
    train_dl: DataLoader,
    val_dl:   DataLoader,
    *,
    fold: int = 0,
    rep: str = "mel",
    epochs: int = EPOCHS,
    lr: float = LR,
    early_stop: int = EARLY_STOP,
    device: torch.device | None = None,
    save_dir: str = "models",
) -> float:
    """
    Addestra/valida un singolo fold per la rappresentazione scelta.
    Ritorna la migliore accuracy sul validation set.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = device or _get_device()
    print(f"[{rep.upper()} Fold {fold}] Device: {device}")

    torch.manual_seed(42); random.seed(42)

    in_ch = 2 if rep == "dmfcc" else 1
    model = _make_resnet18(device, in_channels=in_ch)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(model.parameters(), lr=lr)
    scheduler  = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        epochs=epochs,
        steps_per_epoch=len(train_dl),
        pct_start=0.3,
        div_factor=10,
        final_div_factor=1e4,
    )

    best_acc, wait, history = 0.0, 0, []
    start = time.time()

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # ── validazione ──
        model.eval(); correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
        acc = correct / len(val_dl.dataset)
        history.append({"epoch": epoch, "val_acc": acc})
        print(f"[{rep} F{fold}] Epoch {epoch:03d}  Val acc = {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(),
                       f"{save_dir}/best_resnet18_{rep}_fold{fold}.pt")
            wait = 0
        else:
            wait += 1
        if wait >= early_stop:
            print(f"[{rep} F{fold}] Early stop dopo {early_stop} epoche.")
            break

    elapsed = (time.time() - start) / 60
    print(f"[{rep} F{fold}] Best val-acc: {best_acc:.3f} – {elapsed:.1f} min")

    with open(f"{save_dir}/log_{rep}_fold{fold}.txt", "w") as fp:
        json.dump(history, fp, indent=2)

    return best_acc


# ────────────────────── modalità single-split legacy ─────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", choices=["mel","lin","mfcc","dmfcc"],
                        default="mel", help="quale rappresentazione usare")
    args = parser.parse_args()

    device = _get_device()
    print("Device in uso:", device, "  | rep =", args.rep)

    train_ds = MelDataset("train", rep=args.rep, normalize=True, augment=True)
    val_ds   = MelDataset("val",   rep=args.rep, normalize=True)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    train_one_fold(train_dl, val_dl, fold=0, rep=args.rep, device=device)