"""
module2/dataset_2b.py
STELLARIS-DNet — Module 2B Data Pipeline
G2Net Gravitational Wave Detection
Binary classification: Signal (GW present) / Noise
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module2.config import *


# ─────────────────────────────────────────────
# 1. G2NET DATA FORMAT
# Each sample: 3 detectors × 4096 time points
# LIGO-Hanford (H1), LIGO-Livingston (L1), Virgo (V1)
# Files: .npy arrays of shape (3, 4096)
# Labels: training_labels.csv → id, target (0/1)
# ─────────────────────────────────────────────

def _get_file_path(file_id: str, data_dir: str) -> str:
    """
    G2Net stores files in subdirectories based on first 3 chars of id.
    e.g. id='00001a4b' → data_dir/0/0/0/00001a4b.npy
    """
    return os.path.join(
        data_dir,
        file_id[0], file_id[1], file_id[2],
        f"{file_id}.npy"
    )


def _bandpass_filter(signal: np.ndarray,
                     low: float  = 20.0,
                     high: float = 500.0,
                     fs: float   = 2048.0) -> np.ndarray:
    """
    Simple bandpass filter for GW signal.
    Removes frequencies outside LIGO sensitive band (20-500 Hz).
    Uses FFT-based filtering — no scipy dependency.
    """
    n    = signal.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    fft  = np.fft.rfft(signal, axis=-1)
    mask = (freqs >= low) & (freqs <= high)
    fft[..., ~mask] = 0
    return np.fft.irfft(fft, n=n, axis=-1)


def _normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize each detector channel independently."""
    for i in range(signal.shape[0]):
        ch   = signal[i]
        std  = ch.std()
        if std > 1e-10:
            signal[i] = (ch - ch.mean()) / std
    return signal


# ─────────────────────────────────────────────
# 2. DATASET CLASS — G2Net real data
# ─────────────────────────────────────────────
class G2NetDataset(Dataset):
    def __init__(
        self,
        file_ids:  list,
        labels:    np.ndarray,
        data_dir:  str,
        train:     bool = False
    ):
        self.file_ids = file_ids
        self.labels   = torch.tensor(labels, dtype=torch.long)
        self.data_dir = data_dir
        self.train    = train

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        fid  = self.file_ids[idx]
        path = _get_file_path(fid, self.data_dir)

        # Load signal
        signal = np.load(path).astype(np.float32)  # (3, 4096)

        # Preprocess
        signal = _bandpass_filter(signal)
        signal = _normalize_signal(signal)

        # Augment during training
        if self.train:
            signal = self._augment(signal)

        return torch.tensor(signal, dtype=torch.float32), self.labels[idx]

    def _augment(self, signal: np.ndarray) -> np.ndarray:
        """
        Physics-aware augmentation for GW signals.
        - Time shift: GW arrives at detectors at different times
        - Amplitude scaling: accounts for detector sensitivity variation
        """
        # Random time shift (max 50 samples ~ 25ms)
        shift = np.random.randint(-50, 50)
        signal = np.roll(signal, shift, axis=-1)

        # Random amplitude scaling ±10%
        scale  = np.random.uniform(0.9, 1.1)
        signal = signal * scale

        # Random channel flip (swap H1/L1) — physically valid
        if np.random.rand() > 0.5:
            signal = signal[[1, 0, 2], :]

        return signal


# ─────────────────────────────────────────────
# 3. SYNTHETIC DATASET — when G2Net not available
# Generates physically plausible GW-like signals
# ─────────────────────────────────────────────
class SyntheticG2NetDataset(Dataset):
    def __init__(
        self,
        n_samples: int,
        labels:    np.ndarray,
        train:     bool = False
    ):
        self.n       = n_samples
        self.labels  = torch.tensor(labels, dtype=torch.long)
        self.train   = train
        np.random.seed(SEED)
        self.signals = self._generate(n_samples, labels)

    def _generate(self, n: int, labels: np.ndarray) -> np.ndarray:
        """
        Generates synthetic signals:
        - Noise (label=0): Gaussian noise (colored)
        - Signal (label=1): Chirp signal + noise (GW-like)
        """
        signals = np.zeros((n, LIGO_N_DETECTORS, LIGO_SIGNAL_LEN),
                           dtype=np.float32)
        t = np.linspace(0, 2, LIGO_SIGNAL_LEN)

        for i in range(n):
            # Base colored noise for all channels
            noise = np.random.randn(LIGO_N_DETECTORS, LIGO_SIGNAL_LEN)
            # Color the noise (1/f spectrum)
            for ch in range(LIGO_N_DETECTORS):
                fft  = np.fft.rfft(noise[ch])
                freq = np.fft.rfftfreq(LIGO_SIGNAL_LEN)
                freq[0] = 1e-10  # avoid div by zero
                fft  = fft / np.sqrt(freq)
                noise[ch] = np.fft.irfft(fft, LIGO_SIGNAL_LEN)
                noise[ch] /= (noise[ch].std() + 1e-10)

            if labels[i] == 1:
                # GW chirp signal: frequency increases over time
                f0, f1  = 20.0, 300.0       # start/end frequency Hz
                chirp_t = np.linspace(0, 2, LIGO_SIGNAL_LEN)
                phase   = 2 * np.pi * (f0 * chirp_t +
                          (f1 - f0) / 4 * chirp_t**2)
                amp     = np.exp(chirp_t - 2)  # amplitude increases near merger
                chirp   = (amp * np.sin(phase)).astype(np.float32)
                chirp  /= (chirp.std() + 1e-10)

                # Inject into each detector with small time offset
                for ch in range(LIGO_N_DETECTORS):
                    offset = np.random.randint(-10, 10)
                    injected = np.roll(chirp, offset)
                    signals[i, ch] = noise[ch] + 0.5 * injected
            else:
                signals[i] = noise

        return signals

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        signal = self.signals[idx].copy()
        signal = _normalize_signal(signal)
        return torch.tensor(signal, dtype=torch.float32), self.labels[idx]


# ─────────────────────────────────────────────
# 4. DATALOADER FACTORY
# ─────────────────────────────────────────────
def load_g2net(data_dir: str = LIGO_DATA_DIR, max_samples: int = 10000):
    """
    Loads G2Net dataset if available, otherwise uses synthetic data.
    max_samples: limit for local testing (G2Net has 560k files)
    """
    labels_path = os.path.join(data_dir, "training_labels.csv")

    if os.path.exists(labels_path):
        # ── Real G2Net data ──
        print("✅ G2Net labels found — loading real data")
        df = pd.read_csv(labels_path)

        # Limit samples for memory
        if len(df) > max_samples:
            df = df.sample(max_samples, random_state=SEED)
            print(f"   Sampling {max_samples} from {len(df)} total")

        file_ids = df["id"].tolist()
        labels   = df["target"].values.astype(np.int64)

        # Verify files exist
        valid = [i for i, fid in enumerate(file_ids)
                 if os.path.exists(_get_file_path(fid, data_dir))]
        file_ids = [file_ids[i] for i in valid]
        labels   = labels[valid]
        print(f"   Valid files: {len(file_ids)}")
        print(f"   Signal: {labels.sum()} | Noise: {(labels==0).sum()}")

        # Split
        ids_train, ids_test, y_train, y_test = train_test_split(
            file_ids, labels,
            test_size=LIGO_TEST_SPLIT, random_state=SEED, stratify=labels
        )
        ids_train, ids_val, y_train, y_val = train_test_split(
            ids_train, y_train,
            test_size=LIGO_VAL_SPLIT/(1-LIGO_TEST_SPLIT),
            random_state=SEED, stratify=y_train
        )

        train_ds = G2NetDataset(ids_train, y_train, data_dir, train=True)
        val_ds   = G2NetDataset(ids_val,   y_val,   data_dir, train=False)
        test_ds  = G2NetDataset(ids_test,  y_test,  data_dir, train=False)

    else:
        # ── Synthetic fallback ──
        print("⚠️  G2Net not found — using synthetic GW signals")
        print(f"   Add G2Net data to: {data_dir}")
        print(f"   kaggle competitions download -c g2net-gravitational-wave-detection")

        n      = 2000
        labels = np.array([0]*(n//2) + [1]*(n//2))
        np.random.shuffle(labels)

        n_test  = int(n * LIGO_TEST_SPLIT)
        n_val   = int(n * LIGO_VAL_SPLIT)
        n_train = n - n_test - n_val

        idx = np.random.permutation(n)
        train_ds = SyntheticG2NetDataset(n_train, labels[idx[:n_train]],     train=True)
        val_ds   = SyntheticG2NetDataset(n_val,   labels[idx[n_train:n_train+n_val]], train=False)
        test_ds  = SyntheticG2NetDataset(n_test,  labels[idx[n_train+n_val:]], train=False)

    train_loader = DataLoader(train_ds, batch_size=LIGO_BATCH_SIZE,
                              shuffle=True,  drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=LIGO_BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=LIGO_BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Module 2B Dataset Sanity Check")
    print("=" * 50)

    tr, vl, te = load_g2net()
    X, y = next(iter(tr))
    print(f"\nBatch shape : {X.shape}")
    print(f"Label shape : {y.shape}")
    print(f"Signal/Noise: {(y==1).sum()}/{(y==0).sum()} in batch")
    assert X.shape[1:] == (LIGO_N_DETECTORS, LIGO_SIGNAL_LEN), \
        f"Wrong shape: {X.shape}"
    print("\n✅ dataset_2b.py OK")