"""
kfold_train.py – orchestrazione k-fold cross-validation per GTZAN

Esegue la CV stratificata usando StratifiedKFold e train_one_fold.
Salva per ogni rappresentazione:

    • best_resnet18_<rep>_fold{n}.pt   (da train_one_fold)
    • log_<rep>_fold{n}.txt            (da train_one_fold)
    • results_kfold_<rep>.json         (accuracy di ogni fold + media e std)

Uso base (Mel):
    python kfold_train.py

Altri formati:
    python kfold_train.py --rep lin
    python kfold_train.py --rep mfcc
    python kfold_train.py --rep dmfcc

Override parametri:
    python kfold_train.py --rep lin --k 3 --batch 16 --epochs 60
"""

from __future__ import annotations
import argparse, json, numpy as np, torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

from dataset import MelDataset
from train import train_one_fold, _get_device, BATCH_SIZE as DEF_BATCH, EPOCHS as DEF_EPOCHS

# ─── default “globali” CV ───
K_FOLDS    = 5
BATCH_SIZE = DEF_BATCH
EPOCHS     = DEF_EPOCHS

# ──────────────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    device = _get_device()
    rep    = args.rep.lower()
    print(f"\n=== {rep.upper()} – k-fold ({args.k} fold) ===")
    print(f"Device in uso: {device}\n")

    # dataset completo per la rappresentazione scelta
    full_ds = MelDataset(split=None, rep=rep, normalize=True, augment=True)
    y = [lbl for _, lbl in full_ds]

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
            rep=rep,
            epochs=args.epochs,
            device=device,
        )
        fold_acc.append(acc)

    mean, std = float(np.mean(fold_acc)), float(np.std(fold_acc))
    print(f"\n=== {rep.upper()} – accuracy media sui {args.k} fold: {mean:.3f} ± {std:.3f} ===\n")

    out_file = f"results_kfold_{rep}.json"
    with open(out_file, "w") as fp:
        json.dump({"fold_acc": fold_acc, "mean": mean, "std": std}, fp, indent=2)
    print(f"Salvato {out_file}")

# ─── entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="K-fold cross-validation trainer")
    p.add_argument("--rep",  choices=["mel","lin","mfcc","dmfcc"],
                   default="mel",      help="feature da usare")
    p.add_argument("--k",     type=int, default=K_FOLDS,    help="numero di fold")
    p.add_argument("--batch", type=int, default=BATCH_SIZE, help="batch size")
    p.add_argument("--epochs",type=int, default=EPOCHS,     help="epoche per fold")
    main(p.parse_args())