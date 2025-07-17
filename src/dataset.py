import torch, pathlib, random
from torch.utils.data import Dataset

# ─────────────────────────────── util: SpecAugment ─────────────────────────────
def _spec_augment(x, time_mask=12, freq_mask=8):
    """Applica un time-mask e un freq-mask leggeri sullo spettrogramma."""
    _, n_mels, n_frames = x.shape
    # time-mask
    t0 = random.randint(0, max(0, n_frames - time_mask))
    x[:, :, t0 : t0 + time_mask] = 0
    # freq-mask
    f0 = random.randint(0, max(0, n_mels - freq_mask))
    x[:, f0 : f0 + freq_mask, :] = 0
    return x


# ────────────────────────────────── Dataset ────────────────────────────────────
class MelDataset(Dataset):
    """
    Dataset multi-feature:
        rep = 'mel' | 'lin' | 'mfcc' | 'dmfcc'
    """

    GENRES = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    ]

    def __init__(self, split="train", rep="mel", normalize=False, augment=False):
        """
        Args
        ----
        split:   'train' | 'val' | 'test' | None/'all' (per k-fold)
        rep:     quale rappresentazione caricare
                 'mel'   log-Mel 1×128×128
                 'lin'   STFT-dB 1×128×128
                 'mfcc'  MFCC    1×13×128
                 'dmfcc' MFCC+Δ  2×13×128
        normalize: z-score per tensor
        augment:   shift + SpecAugment (solo train / all)
        """

        rep = rep.lower()
        assert rep in {"mel", "lin", "mfcc", "dmfcc"}, "rep deve essere mel/lin/mfcc/dmfcc"

        # cartella dei .pt relativa alla root del progetto
        root = (
            pathlib.Path(__file__).resolve().parent.parent
            / "data"
            / "processed"
            / rep
        )
        files = sorted(root.glob("*.pt"))

        # ── split train / val / test (o all) ────────────────────────────────
        if split is None or split == "all":
            self.files = files
        else:
            torch.manual_seed(42)
            idx = torch.randperm(len(files))
            n = len(idx)
            if split == "train":
                self.files = [files[i] for i in idx[: int(0.8 * n)]]
            elif split == "val":
                self.files = [files[i] for i in idx[int(0.8 * n) : int(0.9 * n)]]
            else:  # test
                self.files = [files[i] for i in idx[int(0.9 * n) :]]

        self.normalize = normalize
        # augment solo in training (o k-fold all)
        self.augment = augment and (split == "train" or split in {None, "all"})
        self.rep = rep  # salvo per eventuali usi futuri

    # ──────────────────────────────── PyTorch API ──────────────────────────────
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x = torch.load(self.files[i])  # shape variabile: (C, H, W)

        # normalizzazione z-score
        if self.normalize:
            x = (x - x.mean()) / (x.std() + 1e-6)

        if self.augment:
            # time-shift ±10 frame (25 %)
            if random.random() < 0.25:
                shift = random.randint(-10, 10)
                x = torch.roll(x, shifts=shift, dims=2)
            # SpecAugment leggero (25 %)
            if random.random() < 0.25 and self.rep in {"mel", "lin"}:
                x = _spec_augment(x)

        label = self.GENRES.index(self.files[i].stem.split(".")[0])
        return x, label