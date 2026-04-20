"""
module2/dataset_2b.py
STELLARIS-DNet — Module 2B Data Pipeline
G2Net Gravitational Wave Detection
Binary: Noise(0) / Signal(1)

Key fix: G2Net data is already whitened strain.
Use robust normalization, not bandpass filter.
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
    LIGO_N_DETECTORS, LIGO_SIGNAL_LEN, LIGO_BATCH_SIZE,
    LIGO_TEST_SPLIT, LIGO_VAL_SPLIT, LIGO_MAX_SAMPLES,
    GW_FREQ_MIN, GW_FREQ_MAX
)


# ─────────────────────────────────────────────
# PATH HELPER
# ─────────────────────────────────────────────
def _resolve_data_dir(data_dir: str) -> str:
    kaggle_path = f"/kaggle/working/STELLARIS-DNet/{data_dir}"
    if os.path.exists("/kaggle/input") and os.path.exists(kaggle_path):
        return kaggle_path
    return data_dir


# ─────────────────────────────────────────────
# 1. PREPROCESSING
# G2Net data is already whitened — use robust normalization
# ─────────────────────────────────────────────
def _preprocess_signal(signal: np.ndarray) -> np.ndarray:
    """
    Correct preprocessing for G2Net whitened strain data.
    Input shape: (3, 4096) — already whitened by LIGO pipeline
    
    Key insight: G2Net signals are pre-whitened with extreme values.
    Standard normalization fails. Use robust MAD-based scaling.
    """
    result = np.zeros_like(signal, dtype=np.float32)
    for i in range(signal.shape[0]):
        ch = signal[i].astype(np.float64)
        # Remove DC offset
        ch = ch - ch.mean()
        # Robust scale: median absolute deviation
        mad = np.median(np.abs(ch - np.median(ch)))
        if mad > 1e-10:
            ch = ch / (mad * 1.4826)  # normalize to unit variance
        else:
            std = ch.std()
            if std > 1e-10:
                ch = ch / std
        # Clip extreme outliers — preserves GW signal shape
        ch = np.clip(ch, -20.0, 20.0)
        # Final rescale to [-1, 1]
        mx = np.abs(ch).max()
        if mx > 1e-10:
            ch = ch / mx
        result[i] = ch.astype(np.float32)
    return result


# ─────────────────────────────────────────────
# 2. G2NET FILE PATH
# ─────────────────────────────────────────────
def _get_file_path(file_id: str, data_dir: str) -> str:
    """G2Net: nested dirs by first 3 chars of id."""
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
        signal = _preprocess_signal(signal)
        if self.train:
            signal = self._augment(signal)
        return torch.tensor(signal, dtype=torch.float32), self.labels[idx]

    def _augment(self, signal: np.ndarray) -> np.ndarray:
        """Physics-aware augmentation for GW signals."""
        # Small time shift — GW arrives at detectors at different times
        shift = np.random.randint(-20, 20)
        signal = np.roll(signal, shift, axis=-1)
        # Small amplitude scaling ±5%
        signal = signal * np.random.uniform(0.95, 1.05)
        # Gaussian noise injection (very small)
        signal = signal + np.random.randn(*signal.shape).astype(np.float32) * 0.01
        return np.clip(signal, -1.0, 1.0)


# ─────────────────────────────────────────────
# 4. PRELOADED DATASET — loads all data into RAM
# Much faster than loading .npy per sample
# ─────────────────────────────────────────────
class G2NetPreloadedDataset(Dataset):
    """
    Preloads all signals into RAM for fast training.
    Use when max_samples <= 10000 (fits in ~480MB RAM).
    """
    def __init__(self, file_ids: list, labels: np.ndarray,
                 data_dir: str, train: bool = False):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.train  = train

        print(f"   Preloading {len(file_ids)} signals into RAM...")
        signals = []
        valid_idx = []
        for i, fid in enumerate(file_ids):
            path = _get_file_path(fid, data_dir)
            try:
                sig = np.load(path).astype(np.float32)
                sig = _preprocess_signal(sig)
                signals.append(sig)
                valid_idx.append(i)
            except Exception:
                continue

        self.signals = np.array(signals, dtype=np.float32)
        self.labels  = self.labels[valid_idx]
        print(f"   Loaded: {len(self.signals)} signals | "
              f"Signal: {self.labels.sum().item()} | "
              f"Noise: {(self.labels==0).sum().item()}")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx].copy()
        if self.train:
            shift  = np.random.randint(-20, 20)
            signal = np.roll(signal, shift, axis=-1)
            signal = signal * np.random.uniform(0.95, 1.05)
            signal = signal + np.random.randn(*signal.shape).astype(np.float32) * 0.01
            signal = np.clip(signal, -1.0, 1.0)
        return torch.tensor(signal, dtype=torch.float32), self.labels[idx]


# ─────────────────────────────────────────────
# 5. SYNTHETIC DATASET — fallback
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
            # Colored noise base
            noise = np.zeros(
                (LIGO_N_DETECTORS, LIGO_SIGNAL_LEN), dtype=np.float32
            )
            for ch in range(LIGO_N_DETECTORS):
                raw  = np.random.randn(LIGO_SIGNAL_LEN)
                fft  = np.fft.rfft(raw)
                freq = np.fft.rfftfreq(LIGO_SIGNAL_LEN)
                freq[0] = 1e-10
                fft  = fft / np.sqrt(np.abs(freq))
                filtered = np.fft.irfft(fft, LIGO_SIGNAL_LEN)
                std  = filtered.std()
                if std > 1e-10:
                    filtered /= std
                noise[ch] = filtered.astype(np.float32)

            if labels[i] == 1:
                # GW chirp
                t     = np.linspace(0, 2, LIGO_SIGNAL_LEN)
                phase = 2 * np.pi * (20.0 * t + 70.0 * t**2)
                amp   = np.exp(3 * (t - 2))
                chirp = (amp * np.sin(phase)).astype(np.float32)
                std   = chirp.std()
                if std > 1e-10:
                    chirp = chirp / std
                for ch in range(LIGO_N_DETECTORS):
                    offset = np.random.randint(-15, 15)
                    signals[i, ch] = noise[ch] + 0.8 * np.roll(chirp, offset)
            else:
                signals[i] = noise

            # Apply same preprocessing
            signals[i] = _preprocess_signal(signals[i])

        return signals

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx].copy()
        if self.train:
            shift  = np.random.randint(-20, 20)
            signal = np.roll(signal, shift, axis=-1)
        return torch.tensor(signal, dtype=torch.float32), self.labels[idx]


# ─────────────────────────────────────────────
# 6. DATALOADER FACTORY
# ─────────────────────────────────────────────
def load_g2net(data_dir: str = None, max_samples: int = LIGO_MAX_SAMPLES):
    if data_dir is None:
        data_dir = _resolve_data_dir(LIGO_DATA_DIR)

    labels_path = os.path.join(data_dir, "training_labels.csv")

    if os.path.exists(labels_path):
        print("✅ G2Net labels found — loading real data")
        df = pd.read_csv(labels_path)

        if len(df) > max_samples:
            # Stratified sample to keep 50/50 balance
            df_signal = df[df["target"] == 1].sample(
                max_samples // 2, random_state=SEED)
            df_noise   = df[df["target"] == 0].sample(
                max_samples // 2, random_state=SEED)
            df = pd.concat([df_signal, df_noise]).sample(
                frac=1, random_state=SEED)
            print(f"   Balanced sample: {max_samples} total")

        file_ids = df["id"].tolist()
        labels   = df["target"].values.astype(np.int64)

        # Verify files exist
        valid = [
            i for i, fid in enumerate(file_ids)
            if os.path.exists(_get_file_path(fid, data_dir))
        ]

        if len(valid) == 0:
            print("⚠️  No .npy files found — using synthetic")
            return _synthetic_loaders()

        file_ids = [file_ids[i] for i in valid]
        labels   = labels[valid]
        print(f"   Valid: {len(file_ids)} | "
              f"Signal: {labels.sum()} | Noise: {(labels==0).sum()}")

        # Split
        ids_tr, ids_te, y_tr, y_te = train_test_split(
            file_ids, labels, test_size=LIGO_TEST_SPLIT,
            random_state=SEED, stratify=labels
        )
        ids_tr, ids_vl, y_tr, y_vl = train_test_split(
            ids_tr, y_tr,
            test_size=LIGO_VAL_SPLIT / (1 - LIGO_TEST_SPLIT),
            random_state=SEED, stratify=y_tr
        )

        # Use preloaded for speed
        train_ds = G2NetPreloadedDataset(ids_tr, y_tr, data_dir, train=True)
        val_ds   = G2NetPreloadedDataset(ids_vl, y_vl, data_dir, train=False)
        test_ds  = G2NetPreloadedDataset(ids_te, y_te, data_dir, train=False)

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

    print(f"   Train: {n_train} | Val: {n_val} | Test: {n_test}")
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
        print(f"\nBatch shape : {X.shape}")
        print(f"Label shape : {y.shape}")
        print(f"Signal/Noise: {(y==1).sum()}/{(y==0).sum()} in batch")
        print(f"Signal range: [{X.min():.3f}, {X.max():.3f}]")
        assert X.shape[1:] == (LIGO_N_DETECTORS, LIGO_SIGNAL_LEN)
        print("\n✅ dataset_2b.py OK")
    except Exception as e:
        import traceback
        traceback.print_exc()