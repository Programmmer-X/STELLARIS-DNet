"""
module2/dataset_2b.py
STELLARIS-DNet — Module 2B Data Pipeline
G2Net Gravitational Wave Detection
Binary: Signal (GW present=1) / Noise (0)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module2.config import (
    SEED, LIGO_DATA_DIR, LIGO_CLASSES, LIGO_NUM_CLASSES,
    LIGO_SIGNAL_LEN, LIGO_N_DETECTORS,
    LIGO_BATCH_SIZE, LIGO_TEST_SPLIT, LIGO_VAL_SPLIT,
    GW_FREQ_MIN, GW_FREQ_MAX
)


# ─────────────────────────────────────────────
# PATH HELPER — auto-detect Kaggle vs Local
# ─────────────────────────────────────────────
def _resolve_data_dir(data_dir: str) -> str:
    kaggle_path = f"/kaggle/working/STELLARIS-DNet/{data_dir}"
    if os.path.exists("/kaggle/input") and os.path.exists(kaggle_path):
        return kaggle_path
    return data_dir


# ─────────────────────────────────────────────
# 1. SIGNAL PROCESSING
# ─────────────────────────────────────────────
def _bandpass_filter(signal: np.ndarray,
                     low: float  = GW_FREQ_MIN,
                     high: float = GW_FREQ_MAX,
                     fs: float   = 2048.0) -> np.ndarray:
    """FFT-based bandpass filter. No scipy needed."""
    n     = signal.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    fft   = np.fft.rfft(signal, axis=-1)
    mask  = (freqs >= low) & (freqs <= high)
    fft[..., ~mask] = 0
    return np.fft.irfft(fft, n=n, axis=-1)


def _normalize_signal(signal: np.ndarray) -> np.ndarray:
    for i in range(signal.shape[0]):
        ch  = signal[i]
        std = ch.std()
        if std > 1e-10:
            signal[i] = (ch - ch.mean()) / std
    return signal


# ─────────────────────────────────────────────
# 2. G2NET FILE PATH HELPER
# ─────────────────────────────────────────────
def _get_file_path(file_id: str, data_dir: str) -> str:
    """G2Net stores files in nested subdirs by first 3 chars of id."""
    return os.path.join(
        data_dir, "train",
        file_id[0], file_id[1], file_id[2],
        f"{file_id}.npy"
    )


# ─────────────────────────────────────────────
# 3. REAL G2NET DATASET
# ─────────────────────────────────────────────
class G2NetDataset(Dataset):
    def __init__(self, file_ids: list, labels: np.ndarray,
                 data_dir: str, train: bool = False):
        self.file_ids = file_ids
        self.labels   = torch.tensor(labels, dtype=torch.long)
        self.data_dir = data_dir
        self.train    = train

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        fid    = self.file_ids[idx]
        path   = _get_file_path(fid, self.data_dir)
        signal = np.load(path).astype(np.float32)   # (3, 4096)
        signal = _bandpass_filter(signal)
        signal = _normalize_signal(signal)
        if self.train:
            signal = self._augment(signal)
        return torch.tensor(signal, dtype=torch.float32), self.labels[idx]

    def _augment(self, signal: np.ndarray) -> np.ndarray:
        shift  = np.random.randint(-50, 50)
        signal = np.roll(signal, shift, axis=-1)
        scale  = np.random.uniform(0.9, 1.1)
        signal = signal * scale
        if np.random.rand() > 0.5:
            signal = signal[[1, 0, 2], :]
        return signal


# ─────────────────────────────────────────────
# 4. SYNTHETIC DATASET — fallback
# ─────────────────────────────────────────────
class SyntheticG2NetDataset(Dataset):
    def __init__(self, n_samples: int, labels: np.ndarray,
                 train: bool = False):
        self.labels  = torch.tensor(labels, dtype=torch.long)
        self.train   = train
        np.random.seed(SEED)
        self.signals = self._generate(n_samples, labels)

    def _generate(self, n: int, labels: np.ndarray) -> np.ndarray:
        signals = np.zeros(
            (n, LIGO_N_DETECTORS, LIGO_SIGNAL_LEN), dtype=np.float32
        )
        for i in range(n):
            noise = np.random.randn(
                LIGO_N_DETECTORS, LIGO_SIGNAL_LEN
            ).astype(np.float32)
            for ch in range(LIGO_N_DETECTORS):
                fft  = np.fft.rfft(noise[ch])
                freq = np.fft.rfftfreq(LIGO_SIGNAL_LEN)
                freq[0] = 1e-10
                fft  = fft / np.sqrt(freq)
                noise[ch] = np.fft.irfft(fft, LIGO_SIGNAL_LEN)
                std = noise[ch].std()
                if std > 1e-10:
                    noise[ch] /= std

            if labels[i] == 1:
                t     = np.linspace(0, 2, LIGO_SIGNAL_LEN)
                phase = 2 * np.pi * (20.0 * t + 70.0 * t**2)
                amp   = np.exp(t - 2)
                chirp = (amp * np.sin(phase)).astype(np.float32)
                std   = chirp.std()
                if std > 1e-10:
                    chirp /= std
                for ch in range(LIGO_N_DETECTORS):
                    offset = np.random.randint(-10, 10)
                    signals[i, ch] = noise[ch] + 0.5 * np.roll(chirp, offset)
            else:
                signals[i] = noise

        return signals

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx].copy()
        signal = _normalize_signal(signal)
        return torch.tensor(signal, dtype=torch.float32), self.labels[idx]


# ─────────────────────────────────────────────
# 5. DATALOADER FACTORY
# ─────────────────────────────────────────────
def load_g2net(data_dir: str = None, max_samples: int = 10000):
    if data_dir is None:
        data_dir = _resolve_data_dir(LIGO_DATA_DIR)

    labels_path = os.path.join(data_dir, "training_labels.csv")

    if os.path.exists(labels_path):
        print("✅ G2Net labels found — loading real data")
        df = pd.read_csv(labels_path)

        if len(df) > max_samples:
            df = df.sample(max_samples, random_state=SEED)
            print(f"   Sampling {max_samples} from {len(df)} total")

        file_ids = df["id"].tolist()
        labels   = df["target"].values.astype(np.int64)

        valid    = [
            i for i, fid in enumerate(file_ids)
            if os.path.exists(_get_file_path(fid, data_dir))
        ]

        if len(valid) == 0:
            print("⚠️  No .npy files found — switching to synthetic")
            return _synthetic_loaders()

        file_ids = [file_ids[i] for i in valid]
        labels   = labels[valid]
        print(f"   Valid files: {len(file_ids)}")
        print(f"   Signal: {labels.sum()} | Noise: {(labels==0).sum()}")

        ids_tr, ids_te, y_tr, y_te = train_test_split(
            file_ids, labels,
            test_size=LIGO_TEST_SPLIT, random_state=SEED, stratify=labels
        )
        ids_tr, ids_vl, y_tr, y_vl = train_test_split(
            ids_tr, y_tr,
            test_size=LIGO_VAL_SPLIT / (1 - LIGO_TEST_SPLIT),
            random_state=SEED, stratify=y_tr
        )

        train_ds = G2NetDataset(ids_tr, y_tr, data_dir, train=True)
        val_ds   = G2NetDataset(ids_vl, y_vl, data_dir, train=False)
        test_ds  = G2NetDataset(ids_te, y_te, data_dir, train=False)

    else:
        print("⚠️  G2Net not found — using synthetic GW signals")
        return _synthetic_loaders()

    train_loader = DataLoader(train_ds, batch_size=LIGO_BATCH_SIZE,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=LIGO_BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=LIGO_BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


def _synthetic_loaders():
    """Fallback: synthetic GW signals."""
    n      = 2000
    labels = np.array([0]*(n//2) + [1]*(n//2))
    np.random.shuffle(labels)

    n_test  = int(n * LIGO_TEST_SPLIT)
    n_val   = int(n * LIGO_VAL_SPLIT)
    n_train = n - n_test - n_val
    idx     = np.random.permutation(n)

    train_ds = SyntheticG2NetDataset(
        n_train, labels[idx[:n_train]], train=True)
    val_ds   = SyntheticG2NetDataset(
        n_val, labels[idx[n_train:n_train+n_val]], train=False)
    test_ds  = SyntheticG2NetDataset(
        n_test, labels[idx[n_train+n_val:]], train=False)

    train_loader = DataLoader(train_ds, batch_size=LIGO_BATCH_SIZE,
                              shuffle=True, drop_last=True, num_workers=0)
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
    try:
        tr, vl, te = load_g2net()
        X, y = next(iter(tr))
        print(f"Batch shape : {X.shape}")
        print(f"Label shape : {y.shape}")
        print(f"Signal/Noise: {(y==1).sum()}/{(y==0).sum()} in batch")
        assert X.shape[1:] == (LIGO_N_DETECTORS, LIGO_SIGNAL_LEN), \
            f"Wrong shape: {X.shape}"
        print("\n✅ dataset_2b.py OK")
    except Exception as e:
        import traceback
        traceback.print_exc()