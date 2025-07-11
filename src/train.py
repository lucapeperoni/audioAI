"""
train.py – ResNet-18 sui Mel-spectrogrammi GTZAN
Salva:
    • best_resnet18.pt
    • log.txt
"""
import time, json, random, torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from dataset import MelDataset

# ─── hyper ───
EPOCHS      = 120
BATCH_SIZE  = 8
LR          = 1e-4
EARLY_STOP  = 60      # epoche senza migliorare
DEVICE      = (
    torch.device('mps')
    if torch.backends.mps.is_available()
    else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)
print("Device in uso:", DEVICE)

# riproducibilità
torch.manual_seed(42); random.seed(42)

# ─── dati ───
train_ds = MelDataset('train', normalize=True, augment=True)
val_ds   = MelDataset('val',   normalize=True)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# ─── modello ───
w = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=w)
# conv1 a 1 canale mantenendo la media dei pesi RGB
old_w = model.conv1.weight.clone()
model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
model.conv1.weight.data = old_w.mean(1, keepdim=True)
model.fc = nn.Linear(512, 10)
model.to(DEVICE)

criterion  = nn.CrossEntropyLoss()
optimizer  = optim.Adam(model.parameters(), lr=LR)
scheduler  = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    epochs=EPOCHS,
    steps_per_epoch=len(train_dl),
    pct_start=0.3,
    div_factor=10,
    final_div_factor=1e4
)

best_acc, wait, history = 0.0, 0, []
start = time.time()

for epoch in range(1, EPOCHS + 1):
    # ── train ──
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
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
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
    acc = correct / len(val_ds)
    history.append({'epoch': epoch, 'val_acc': acc})
    print(f"Epoch {epoch:03d}  Val acc = {acc:.3f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_resnet18.pt')
        wait = 0
    else:
        wait += 1
    if wait >= EARLY_STOP:
        print(f"Early stopping dopo {EARLY_STOP} epoche senza migliorare.")
        break

elapsed = (time.time() - start) / 60
print(f"Best val-acc: {best_acc:.3f}  –  tempo {elapsed:.1f} min")

with open('log.txt', 'w') as fp:
    json.dump(history, fp, indent=2)