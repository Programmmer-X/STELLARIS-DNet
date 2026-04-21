"""
module2/dataset_2b.py
STELLARIS-DNet — Module 2B Data Pipeline
G2Net Gravitational Wave Detection — REAL DATA ONLY
CQT Spectrogram → EfficientNet-B0
Binary: Noise(0) / Signal(1)

Strategy:
1. Load real G2Net .npy strain files
2. Compute CQT spectrograms
3. Cache to disk (one-time cost)
4. Load from cache for all subsequent training
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
    LIGO_CQT_BINS, LIGO_CQT_STEPS, GW_FREQ_MIN, GW_FREQ_MAX
)


# ─────────────────────────────────────────────
# PATH HELPER
# ─────────────────────────────────────────────
def _resolve_data_dir(data_dir: str) -> str:
    kaggle_path = f"/kaggle/working/STELLARIS-DNet/{data_dir}"
    if os.path.exists("/kaggle/input") and os.path.exists(kaggle_path):
        return kaggle_path
    return data_dir


def _get_file_path(file_id: str, data_dir: str) -> str:
    return os.path.join(
        data_dir, "train",
        file_id[0], file_id[1], file_id[2],
        f"{file_id}.npy"
    )


# ─────────────────────────────────────────────
# 1. SIGNAL PROCESSING PIPELINE
# ─────────────────────────────────────────────
def _whiten(signal: np.ndarray, fs: float = 2048.0) -> np.ndarray:
    """Whiten signal by dividing by estimated PSD."""
    n   = signal.shape[-1]
    fft = np.fft.rfft(signal, axis=-1)
    psd = np.abs(fft) ** 2
    # Smooth PSD
    kernel = np.ones(10) / 10
    for i in range(signal.shape[0]):
        psd[i] = np.convolve(psd[i], kernel, mode='same')
        psd[i] = np.maximum(psd[i], 1e-20)
    return np.fft.irfft(fft / np.sqrt(psd), n=n, axis=-1).astype(np.float32)


def _bandpass(signal: np.ndarray,
              low: float = GW_FREQ_MIN,
              high: float = GW_FREQ_MAX,
              fs: float = 2048.0) -> np.ndarray:
    """FFT bandpass: keep only GW sensitive band 20-500 Hz."""
    n     = signal.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    fft   = np.fft.rfft(signal, axis=-1)
    mask  = (freqs >= low) & (freqs <= high)
    fft[..., ~mask] = 0
    return np.fft.irfft(fft, n=n, axis=-1).astype(np.float32)


def _cqt(signal: np.ndarray,
         fs: float = 2048.0,
         n_bins: int = LIGO_CQT_BINS,
         n_steps: int = LIGO_CQT_STEPS,
         f_min: float = GW_FREQ_MIN,
         f_max: float = GW_FREQ_MAX) -> np.ndarray:
    """
    Constant-Q Transform — logarithmically spaced frequency bins.
    GW chirp sweeps from f_min to f_max — visible as curved track.
    Output: (n_detectors, n_bins, n_steps)
    """
    n_det = signal.shape[0]
    n_sig = signal.shape[1]
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_bins)
    hop   = max(1, n_sig // n_steps)
    times = np.linspace(0, n_sig - hop, n_steps, dtype=int)
    spec  = np.zeros((n_det, n_bins, len(times)), dtype=np.float32)
    t_arr = np.arange(n_sig) / fs

    for fi, freq in enumerate(freqs):
        Q      = 8.0
        sigma  = 1.0 / (2 * np.pi * freq / Q)
        t_ctr  = t_arr[n_sig // 2]
        window = np.exp(-0.5 * ((t_arr - t_ctr) / sigma) ** 2)
        wavelet_r = window * np.cos(2 * np.pi * freq * t_arr)
        wavelet_i = window * np.sin(2 * np.pi * freq * t_arr)

        for det in range(n_det):
            conv_r = np.convolve(signal[det], wavelet_r[::-1], mode='same')
            conv_i = np.convolve(signal[det], wavelet_i[::-1], mode='same')
            amp    = np.sqrt(conv_r ** 2 + conv_i ** 2)
            for ti, t_idx in enumerate(times):
                spec[det, fi, ti] = amp[t_idx]

    return spec


def _normalize_spec(spec: np.ndarray) -> np.ndarray:
    """Normalize to [0,1] then apply ImageNet stats for EfficientNet."""
    result = np.zeros_like(spec, dtype=np.float32)
    for i in range(spec.shape[0]):
        mn, mx = spec[i].min(), spec[i].max()
        if mx - mn > 1e-8:
            result[i] = (spec[i] - mn) / (mx - mn)
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return (result - mean) / std


def signal_to_spectrogram(signal: np.ndarray) -> np.ndarray:
    """
    Full pipeline: raw strain → CQT spectrogram
    Input:  (3, 4096)
    Output: (3, LIGO_CQT_BINS, LIGO_CQT_STEPS)
    """
    signal = _whiten(signal)
    signal = _bandpass(signal)
    spec   = _cqt(signal)
    spec   = _normalize_spec(spec)
    return spec


# ─────────────────────────────────────────────
# 2. PRECOMPUTE + CACHE
# Compute CQT once, save to disk, load fast after
# ─────────────────────────────────────────────
def precompute_cqt_cache(file_ids: list, labels: np.ndarray,
                          data_dir: str, cache_path: str) -> bool:
    """
    Precomputes CQT spectrograms for all files and saves to .npz cache.
    Returns True if successful.
    """
    print(f"   Computing CQT for {len(file_ids)} samples...")
    print(f"   This runs ONCE then loads from cache.")

    specs     = []
    valid_idx = []

    for i, fid in enumerate(file_ids):
        try:
            path   = _get_file_path(fid, data_dir)
            signal = np.load(path).astype(np.float32)
            spec   = signal_to_spectrogram(signal)
            specs.append(spec)
            valid_idx.append(i)
            if (i + 1) % 200 == 0:
                print(f"   [{i+1}/{len(file_ids)}] done...")
        except Exception as e:
            continue

    if len(specs) == 0:
        print("❌ No files could be processed")
        return False

    specs      = np.array(specs, dtype=np.float32)
    labels_out = labels[valid_idx]

    np.savez_compressed(cache_path, specs=specs, labels=labels_out)
    print(f"✅ Cache saved: {cache_path}")
    print(f"   Shape: {specs.shape} | "
          f"Signal: {(labels_out==1).sum()} | "
          f"Noise: {(labels_out==0).sum()}")
    return True


def load_cqt_cache(cache_path: str):
    """Load precomputed CQT cache."""
    data   = np.load(cache_path)
    specs  = data["specs"]
    labels = data["labels"]
    print(f"✅ Cache loaded: {cache_path}")
    print(f"   Shape: {specs.shape} | "
          f"Signal: {(labels==1).sum()} | "
          f"Noise: {(labels==0).sum()}")
    return specs, labels


# ─────────────────────────────────────────────
# 3. DATASET CLASS
# ─────────────────────────────────────────────
class G2NetCQTDataset(Dataset):
    def __init__(self, specs: np.ndarray, labels: np.ndarray,
                 train: bool = False):
        self.specs  = specs
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.train  = train

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        spec = self.specs[idx].copy()
        if self.train:
            # Time shift in spectrogram domain
            shift = np.random.randint(-3, 3)
            spec  = np.roll(spec, shift, axis=2)
            # Amplitude scale
            spec  = spec * np.random.uniform(0.95, 1.05)
            # Horizontal flip (time reversal)
            if np.random.rand() > 0.5:
                spec = spec[:, :, ::-1].copy()
        return torch.tensor(spec, dtype=torch.float32), self.labels[idx]


# ─────────────────────────────────────────────
# 4. DATALOADER FACTORY
# ─────────────────────────────────────────────
def load_g2net(data_dir: str = None, max_samples: int = LIGO_MAX_SAMPLES):
    if data_dir is None:
        data_dir = _resolve_data_dir(LIGO_DATA_DIR)

    labels_path = os.path.join(data_dir, "training_labels.csv")
    cache_path  = os.path.join(data_dir, f"cqt_cache_{max_samples}.npz")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"G2Net labels not found at: {labels_path}\n"
            f"Add G2Net competition data to Kaggle notebook inputs."
        )

    # ── Load or build cache ────────────────────
    if os.path.exists(cache_path):
        print(f"✅ Loading from cache: {cache_path}")
        specs, labels = load_cqt_cache(cache_path)
    else:
        print("📡 Building CQT cache (one-time)...")
        df = pd.read_csv(labels_path)

        # Balanced stratified sample
        half = min(
            max_samples // 2,
            (df["target"] == 0).sum(),
            (df["target"] == 1).sum()
        )
        df = pd.concat([
            df[df["target"] == 0].sample(half, random_state=SEED),
            df[df["target"] == 1].sample(half, random_state=SEED)
        ]).sample(frac=1, random_state=SEED).reset_index(drop=True)

        file_ids = df["id"].tolist()
        labels   = df["target"].values.astype(np.int64)

        # Filter to existing files
        valid    = [i for i, fid in enumerate(file_ids)
                    if os.path.exists(_get_file_path(fid, data_dir))]

        if len(valid) == 0:
            raise FileNotFoundError(
                f"G2Net .npy files not found in {data_dir}/train/\n"
                f"Ensure G2Net train folder is symlinked correctly."
            )

        file_ids = [file_ids[i] for i in valid]
        labels   = labels[valid]

        success = precompute_cqt_cache(file_ids, labels,
                                        data_dir, cache_path)
        if not success:
            raise RuntimeError("CQT precomputation failed.")

        specs, labels = load_cqt_cache(cache_path)

    # ── Split ──────────────────────────────────
    n       = len(specs)
    idx_arr = np.arange(n)
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        idx_arr, labels, test_size=LIGO_TEST_SPLIT,
        random_state=SEED, stratify=labels
    )
    idx_tr, idx_vl, y_tr, y_vl = train_test_split(
        idx_tr, y_tr,
        test_size=LIGO_VAL_SPLIT / (1 - LIGO_TEST_SPLIT),
        random_state=SEED, stratify=y_tr
    )

    train_ds = G2NetCQTDataset(specs[idx_tr], y_tr, train=True)
    val_ds   = G2NetCQTDataset(specs[idx_vl], y_vl, train=False)
    test_ds  = G2NetCQTDataset(specs[idx_te], y_te, train=False)

    print(f"   Train: {len(train_ds)} | "
          f"Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=LIGO_BATCH_SIZE,
        shuffle=True, drop_last=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=LIGO_BATCH_SIZE,
        shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=LIGO_BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Module 2B — CQT Pipeline Test (single sample)")
    print("=" * 50)

    import numpy as np

    # Test CQT on one random signal
    print("\n── Single signal CQT test ──")
    sig  = np.random.randn(3, 4096).astype(np.float32)
    spec = signal_to_spectrogram(sig)
    print(f"Input  shape: {sig.shape}")
    print(f"Output shape: {spec.shape}")
    print(f"Value  range: [{spec.min():.3f}, {spec.max():.3f}]")
    assert spec.shape == (LIGO_N_DETECTORS, LIGO_CQT_BINS, LIGO_CQT_STEPS), \
        f"Wrong shape: {spec.shape}"
    print("\n✅ CQT pipeline OK")
    print("✅ dataset_2b.py ready — real G2Net only")
    print("\nNote: Full dataloader test requires G2Net data.")
    print("Run on Kaggle with G2Net competition dataset added.")