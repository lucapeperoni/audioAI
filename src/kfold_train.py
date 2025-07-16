"""
kfold_train.py – orchestrazione k-fold cross-validation per GTZAN

Esegue la cross-validation stratificata (K fold) usando StratifiedKFold
e la funzione train_one_fold del train refactor. Salva:

    • best_resnet18_fold{fold}.pt   (da train_one_fold)
    • log_fold{fold}.txt            (da train_one_fold)
    • results_kfold.json            (accuracy di ogni fold + media e std)

Uso rapido (default):
    python kfold_train.py

Override parametri, es.:
    python kfold_train.py --k 3 --batch 16 --epochs 60
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from dataset import MelDataset
from train import train_one_fold, _get_device, BATCH_SIZE as TRAIN_BATCH, EPOCHS as TRAIN_EPOCHS

# ─── default “globali” per la CV ───
K_FOLDS    = 5
BATCH_SIZE = TRAIN_BATCH    # 8 di default in train.py
EPOCHS     = TRAIN_EPOCHS   # 120 di default in train.py


# ──────────────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    device = _get_device()
    print(f"Device in uso (k-fold): {device}\n")

    # Dataset completo (split=None) + etichette per StratifiedKFold
    full_ds = MelDataset(split=None, normalize=True, augment=True)
    y = [label for _, label in full_ds]

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    fold_acc = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        train_dl = DataLoader(
            Subset(full_ds, tr_idx),
            batch_size=args.batch,
            shuffle=True,
        )
        val_dl = DataLoader(
            Subset(full_ds, va_idx),
            batch_size=args.batch,
        )

        print(f"\n─── Fold {fold}/{args.k}  (train={len(tr_idx)}, val={len(va_idx)}) ───")
        acc = train_one_fold(
            train_dl=train_dl,
            val_dl=val_dl,
            fold=fold,
            epochs=args.epochs,
            device=device,
        )
        fold_acc.append(acc)

    mean, std = float(np.mean(fold_acc)), float(np.std(fold_acc))
    print(f"\n=== Accuracy media sui {args.k} fold: {mean:.3f} ± {std:.3f} ===\n")

    with open("results_kfold.json", "w") as fp:
        json.dump({"fold_acc": fold_acc, "mean": mean, "std": std}, fp, indent=2)
    print("Salvato results_kfold.json")


# ─── entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-fold cross-validation trainer")
    parser.add_argument("--k",     type=int, default=K_FOLDS,    help="numero di fold")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="batch size")
    parser.add_argument("--epochs",type=int, default=EPOCHS,     help="epoche per fold")
    main(parser.parse_args())