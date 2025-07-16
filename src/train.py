"""
train.py – ResNet-18 sui Mel-spectrogrammi GTZAN (k-fold ready)
Salva:
    • best_resnet18_fold{fold}.pt
    • log_fold{fold}.txt
"""

from __future__ import annotations
import os, time, json, random, torch, torch.nn as nn
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


def _make_resnet18(device: torch.device) -> nn.Module:
    """ResNet-18 con 1 canale in input e 10 classi in output."""
    w = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=w)
    # conv1 a 1 canale: media dei pesi RGB
    old_w = model.conv1.weight.clone()
    model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    model.conv1.weight.data = old_w.mean(1, keepdim=True)
    model.fc = nn.Linear(512, 10)
    return model.to(device)


# ──────────────────────────────── API principale ─────────────────────────────
def train_one_fold(
    train_dl: DataLoader,
    val_dl:   DataLoader,
    *,
    fold: int = 0,
    epochs: int = EPOCHS,
    lr: float = LR,
    early_stop: int = EARLY_STOP,
    device: torch.device | None = None,
    save_dir: str = "models",
) -> float:
    """
    Addestra/valida **un** singolo fold e restituisce la best-accuracy sul validation set.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = device or _get_device()
    print(f"[Fold {fold}] Device: {device}")

    # riproducibilità
    torch.manual_seed(42); random.seed(42)

    model = _make_resnet18(device)
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
        print(f"[Fold {fold}] Epoch {epoch:03d}  Val acc = {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{save_dir}/best_resnet18_fold{fold}.pt")
            wait = 0
        else:
            wait += 1
        if wait >= early_stop:
            print(f"[Fold {fold}] Early stopping dopo {early_stop} epoche senza migliorare.")
            break

    elapsed = (time.time() - start) / 60
    print(f"[Fold {fold}] Best val-acc: {best_acc:.3f} – tempo {elapsed:.1f} min")

    with open(f"{save_dir}/log_fold{fold}.txt", "w") as fp:
        json.dump(history, fp, indent=2)

    return best_acc


# ────────────────────── modalità single-split legacy ─────────────────────────
if __name__ == "__main__":
    device = _get_device()
    print("Device in uso:", device)

    train_ds = MelDataset("train", normalize=True, augment=True)
    val_ds   = MelDataset("val",   normalize=True)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    train_one_fold(train_dl, val_dl, fold=0, device=device)