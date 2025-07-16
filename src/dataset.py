import torch, pathlib, random
from torch.utils.data import Dataset

def _spec_augment(x, time_mask=12, freq_mask=8):
    """Applica un time-mask e un freq-mask leggeri sullo spettrogramma."""
    _, n_mels, n_frames = x.shape
    # time-mask
    t0 = random.randint(0, max(0, n_frames - time_mask))
    x[:, :, t0:t0 + time_mask] = 0
    # freq-mask
    f0 = random.randint(0, max(0, n_mels - freq_mask))
    x[:, f0:f0 + freq_mask, :] = 0
    return x

class MelDataset(Dataset):
    GENRES = ['blues','classical','country','disco','hiphop',
              'jazz','metal','pop','reggae','rock']

    def __init__(self, split='train', normalize=False, augment=False):
        # Path is now resolved relative to the project root, no matter where you launch Python
        root = (pathlib.Path(__file__).resolve().parent.parent / 'data' / 'processed')
        files = sorted(root.glob('*.pt'))

        # If split is None or 'all', return the full list of files (useful for k‑fold CV)
        if split is None or split == 'all':
            self.files = files
        else:
            torch.manual_seed(42)
            idx = torch.randperm(len(files))
            n   = len(idx)
            if split == 'train':
                self.files = [files[i] for i in idx[:int(0.8*n)]]
            elif split == 'val':
                self.files = [files[i] for i in idx[int(0.8*n):int(0.9*n)]]
            else:  # test
                self.files = [files[i] for i in idx[int(0.9*n):]]

        self.normalize = normalize
        # Apply augmentation when requested and the split corresponds to training
        # or when the user is using the full dataset (split None/'all' for k‑fold).
        self.augment   = augment and (split == 'train' or split is None or split == 'all')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x = torch.load(self.files[i])            # (1, 128, 128)

        # normalizzazione z-score
        if self.normalize:
            x = (x - x.mean()) / (x.std() + 1e-6)

        if self.augment:
            # time-shift ±10 frame (25 % dei campioni)
            if random.random() < 0.25:
                shift = random.randint(-10, 10)
                x = torch.roll(x, shifts=shift, dims=2)
            # SpecAugment leggero (25 %)
            if random.random() < 0.25:
                x = _spec_augment(x)

        label = self.GENRES.index(self.files[i].stem.split('.')[0])
        return x, label