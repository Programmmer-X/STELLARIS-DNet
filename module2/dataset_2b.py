"""
module2/dataset_2b.py
STELLARIS-DNet — Module 2B Data Pipeline
G2Net Gravitational Wave Detection — Real Data Only
CQT Spectrogram → EfficientNet-B2

Kept from original: vectorized FFT CQT (0.05s/sample), multiprocessing
                    cache builder, whitening + bandpass preprocessing
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module2.config import (
    SEED, LIGO_DATA_DIR, LIGO_CLASSES, LIGO_NUM_CLASSES,
    LIGO_N_DETECTORS, LIGO_SIGNAL_LEN, LIGO_SAMPLE_RATE,
    LIGO_BATCH_SIZE, LIGO_TEST_SPLIT, LIGO_VAL_SPLIT,
    LIGO_MAX_SAMPLES, LIGO_CQT_BINS, LIGO_CQT_STEPS,
    GW_FREQ_MIN, GW_FREQ_MAX
)


# ─────────────────────────────────────────────
# PATH HELPER
# ─────────────────────────────────────────────
def _resolve_data_dir(data_dir: str) -> str:
    kaggle = f"/kaggle/working/STELLARIS-DNet/{data_dir}"
    if os.path.exists("/kaggle/input") and os.path.exists(kaggle):
        return kaggle
    return data_dir


def _get_file_path(file_id: str, data_dir: str) -> str:
    return os.path.join(
        data_dir, "train",
        file_id[0], file_id[1], file_id[2],
        f"{file_id}.npy"
    )


# ─────────────────────────────────────────────
# 1. FAST VECTORIZED CQT
# Key insight: FFT convolution — 180x faster than loops
# Original per-sample time: ~9s → ~0.05s
# ─────────────────────────────────────────────
def _fast_cqt(signal: np.ndarray,
              fs: float      = LIGO_SAMPLE_RATE,
              n_bins: int    = LIGO_CQT_BINS,
              n_steps: int   = LIGO_CQT_STEPS,
              f_min: float   = GW_FREQ_MIN,
              f_max: float   = GW_FREQ_MAX,
              Q: float       = 8.0) -> np.ndarray:
    """
    Vectorized CQT using FFT convolution.
    Input:  (n_detectors, signal_len)
    Output: (n_detectors, n_bins, n_steps)
    """
    n_det = signal.shape[0]
    n_sig = signal.shape[1]
    t_arr = np.arange(n_sig) / fs
    t_ctr = t_arr[n_sig // 2]

    freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_bins)
    t_idx = np.linspace(0, n_sig - 1, n_steps, dtype=int)

    # FFT of all detector signals
    sig_fft = np.fft.rfft(signal, axis=-1)   # (n_det, n_freq)

    spec = np.zeros((n_det, n_bins, n_steps), dtype=np.float32)

    for fi, freq in enumerate(freqs):
        sigma = 1.0 / (2 * np.pi * freq / Q)

        gauss     = np.exp(-0.5 * ((t_arr - t_ctr) / sigma) ** 2)
        wavelet_r = gauss * np.cos(2 * np.pi * freq * t_arr)
        wavelet_i = gauss * np.sin(2 * np.pi * freq * t_arr)

        wav_fft_r = np.fft.rfft(wavelet_r[::-1], n=n_sig)
        wav_fft_i = np.fft.rfft(wavelet_i[::-1], n=n_sig)

        # Convolve all detectors simultaneously
        conv_r = np.fft.irfft(sig_fft * wav_fft_r, n=n_sig)
        conv_i = np.fft.irfft(sig_fft * wav_fft_i, n=n_sig)

        amp = np.sqrt(conv_r[:, t_idx] ** 2 + conv_i[:, t_idx] ** 2)
        spec[:, fi, :] = amp.astype(np.float32)

    return spec


def _whiten(signal: np.ndarray, fs: float = LIGO_SAMPLE_RATE) -> np.ndarray:
    """Frequency-domain whitening — removes colored noise floor."""
    n   = signal.shape[-1]
    fft = np.fft.rfft(signal, axis=-1)
    psd = np.abs(fft) ** 2
    k   = np.ones(10) / 10
    for i in range(signal.shape[0]):
        psd[i] = np.convolve(psd[i], k, mode='same')
        psd[i] = np.maximum(psd[i], 1e-20)
    return np.fft.irfft(fft / np.sqrt(psd), n=n, axis=-1).astype(np.float32)


def _bandpass(signal: np.ndarray,
              low: float  = GW_FREQ_MIN,
              high: float = GW_FREQ_MAX,
              fs: float   = LIGO_SAMPLE_RATE) -> np.ndarray:
    """Zero-phase bandpass filter in frequency domain."""
    n     = signal.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    fft   = np.fft.rfft(signal, axis=-1)
    mask  = (freqs >= low) & (freqs <= high)
    fft[..., ~mask] = 0
    return np.fft.irfft(fft, n=n, axis=-1).astype(np.float32)


def _normalize_spec(spec: np.ndarray) -> np.ndarray:
    """Per-channel min-max → ImageNet statistics."""
    result = np.zeros_like(spec, dtype=np.float32)
    for i in range(spec.shape[0]):
        mn, mx = spec[i].min(), spec[i].max()
        if mx - mn > 1e-8:
            result[i] = (spec[i] - mn) / (mx - mn)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return (result - mean) / std


def signal_to_spectrogram(signal: np.ndarray) -> np.ndarray:
    """
    Full pipeline: raw strain → CQT spectrogram
    Input:  (3, 4096)
    Output: (3, LIGO_CQT_BINS, LIGO_CQT_STEPS) — normalized
    """
    signal = _whiten(signal)
    signal = _bandpass(signal)
    spec   = _fast_cqt(signal)
    spec   = _normalize_spec(spec)
    return spec


# ─────────────────────────────────────────────
# 2. PARALLEL CACHE BUILDER
# Multiprocessing for maximum throughput
# ~2-5 min for 4000 samples on Kaggle (4 cores)
# ─────────────────────────────────────────────
def _process_one(args):
    file_id, data_dir = args
    try:
        path   = _get_file_path(file_id, data_dir)
        signal = np.load(path).astype(np.float32)
        spec   = signal_to_spectrogram(signal)
        return spec, True
    except Exception:
        return None, False


def build_cqt_cache(file_ids: list, labels: np.ndarray,
                    data_dir: str, cache_path: str,
                    n_workers: int = None) -> bool:
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    print(f"   Building CQT cache: {len(file_ids)} samples")
    print(f"   Workers: {n_workers} CPU cores")
    print(f"   Est. time: ~{len(file_ids) * 0.05 / n_workers / 60:.1f} min")

    args      = [(fid, data_dir) for fid in file_ids]
    specs     = []
    valid_idx = []

    batch_size = 100
    for batch_start in range(0, len(args), batch_size):
        batch = args[batch_start:batch_start + batch_size]
        with Pool(n_workers) as pool:
            results = pool.map(_process_one, batch)
        for i, (spec, ok) in enumerate(results):
            if ok:
                specs.append(spec)
                valid_idx.append(batch_start + i)
        done = min(batch_start + batch_size, len(args))
        print(f"   [{done}/{len(args)}] processed | {len(specs)} valid")

    if not specs:
        print("❌ No files processed successfully")
        return False

    specs_arr  = np.array(specs, dtype=np.float32)
    labels_out = labels[valid_idx]

    np.savez_compressed(cache_path, specs=specs_arr, labels=labels_out)
    print(f"✅ Cache saved → {cache_path}")
    print(f"   Shape: {specs_arr.shape}")
    print(f"   Signal: {(labels_out==1).sum()} | Noise: {(labels_out==0).sum()}")
    return True


# ─────────────────────────────────────────────
# 3. DATASET CLASSES
# ─────────────────────────────────────────────
class G2NetCQTDataset(Dataset):
    """CQT spectrogram dataset — primary mode for EfficientNet."""
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
            # Time-shift augmentation (physically valid for GW signals)
            shift = np.random.randint(-3, 3)
            spec  = np.roll(spec, shift, axis=2)
            # Mild amplitude jitter
            spec  = spec * np.random.uniform(0.95, 1.05)
            # Time flip (GW physics is time-symmetric pre-merger)
            if np.random.rand() > 0.5:
                spec = spec[:, :, ::-1].copy()
        return torch.tensor(spec, dtype=torch.float32), self.labels[idx]


class G2NetRawDataset(Dataset):
    """Raw signal dataset — for CQT vs Raw comparison in evaluate.py."""
    def __init__(self, signals: np.ndarray, labels: np.ndarray,
                 train: bool = False):
        self.signals = signals
        self.labels  = torch.tensor(labels, dtype=torch.long)
        self.train   = train

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        sig = self.signals[idx].copy()
        if self.train:
            sig = np.roll(sig, np.random.randint(-20, 20), axis=-1)
            sig = sig * np.random.uniform(0.95, 1.05)
        return torch.tensor(sig, dtype=torch.float32), self.labels[idx]


# ─────────────────────────────────────────────
# 4. DATALOADER FACTORY
# ─────────────────────────────────────────────
def load_g2net(data_dir: str = None,
               max_samples: int = LIGO_MAX_SAMPLES,
               use_cqt: bool = True):
    """
    Loads G2Net with CQT or raw signal preprocessing.
    use_cqt=True  → G2NetCQTDataset (EfficientNet)
    use_cqt=False → G2NetRawDataset (1D CNN, for comparison)
    """
    if data_dir is None:
        data_dir = _resolve_data_dir(LIGO_DATA_DIR)

    labels_path = os.path.join(data_dir, "training_labels.csv")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"G2Net labels not found: {labels_path}\n"
            f"Add G2Net competition dataset to notebook inputs.\n"
            f"Kaggle: https://www.kaggle.com/c/g2net-gravitational-wave-detection"
        )

    tag        = "cqt" if use_cqt else "raw"
    cache_path = os.path.join(data_dir, f"cache_{tag}_{max_samples}.npz")

    if os.path.exists(cache_path):
        print(f"✅ Loading cache: {cache_path}")
        data   = np.load(cache_path)
        specs  = data["specs"]
        labels = data["labels"]
        print(f"   Shape: {specs.shape} | "
              f"Signal: {(labels==1).sum()} | "
              f"Noise: {(labels==0).sum()}")
    else:
        print(f"📡 Building {tag.upper()} cache (first run only)...")
        df = pd.read_csv(labels_path)

        # Balanced sampling: equal signal + noise
        half = min(
            max_samples // 2,
            (df["target"]==0).sum(),
            (df["target"]==1).sum()
        )
        df = pd.concat([
            df[df["target"]==0].sample(half, random_state=SEED),
            df[df["target"]==1].sample(half, random_state=SEED)
        ]).sample(frac=1, random_state=SEED).reset_index(drop=True)

        file_ids = df["id"].tolist()
        labels   = df["target"].values.astype(np.int64)

        # Validate files exist
        valid = [i for i, fid in enumerate(file_ids)
                 if os.path.exists(_get_file_path(fid, data_dir))]
        if not valid:
            raise FileNotFoundError(
                "G2Net .npy files not found. "
                "Check that train/ folder is linked correctly."
            )

        file_ids = [file_ids[i] for i in valid]
        labels   = labels[valid]

        if use_cqt:
            ok = build_cqt_cache(file_ids, labels, data_dir, cache_path)
        else:
            ok = _build_raw_cache(file_ids, labels, data_dir, cache_path)

        if not ok:
            raise RuntimeError("Cache build failed.")

        data   = np.load(cache_path)
        specs  = data["specs"]
        labels = data["labels"]

    # Stratified split
    idx_all = np.arange(len(specs))
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        idx_all, labels, test_size=LIGO_TEST_SPLIT,
        random_state=SEED, stratify=labels
    )
    idx_tr, idx_vl, y_tr, y_vl = train_test_split(
        idx_tr, y_tr,
        test_size=LIGO_VAL_SPLIT / (1 - LIGO_TEST_SPLIT),
        random_state=SEED, stratify=y_tr
    )

    DS = G2NetCQTDataset if use_cqt else G2NetRawDataset

    train_ds = DS(specs[idx_tr], y_tr, train=True)
    val_ds   = DS(specs[idx_vl], y_vl, train=False)
    test_ds  = DS(specs[idx_te], y_te, train=False)

    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=LIGO_BATCH_SIZE,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=LIGO_BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=LIGO_BATCH_SIZE,
                              shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def _build_raw_cache(file_ids, labels, data_dir, cache_path):
    """Builds whitened+bandpassed raw signal cache for comparison."""
    print(f"   Building raw signal cache: {len(file_ids)} samples")
    signals, valid_idx = [], []
    for i, fid in enumerate(file_ids):
        try:
            signal = np.load(_get_file_path(fid, data_dir)).astype(np.float32)
            signal = _whiten(signal)
            signal = _bandpass(signal)
            for ch in range(signal.shape[0]):
                std = signal[ch].std()
                if std > 1e-10:
                    signal[ch] /= std
            signals.append(signal)
            valid_idx.append(i)
        except: continue
        if (i+1) % 200 == 0:
            print(f"   [{i+1}/{len(file_ids)}]...")

    signals    = np.array(signals, dtype=np.float32)
    labels_out = labels[valid_idx]
    np.savez_compressed(cache_path, specs=signals, labels=labels_out)
    print(f"✅ Raw cache saved → {cache_path}")
    return True


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Module 2B — Fast CQT Pipeline Test")
    print("=" * 55)
    import time

    sig  = np.random.randn(3, 4096).astype(np.float32)
    t0   = time.time()
    spec = signal_to_spectrogram(sig)
    t1   = time.time()

    print(f"Input  : {sig.shape}")
    print(f"Output : {spec.shape}")
    print(f"Time   : {(t1-t0)*1000:.1f} ms per sample")
    print(f"Range  : [{spec.min():.3f}, {spec.max():.3f}]")

    assert spec.shape == (LIGO_N_DETECTORS, LIGO_CQT_BINS, LIGO_CQT_STEPS)
    assert (t1-t0) < 5.0, f"Too slow: {t1-t0:.1f}s (should be <5s)"
    print(f"\n✅ CQT pipeline OK ({(t1-t0)*1000:.0f}ms/sample)")
    print(f"   4000 samples: ~{4000*(t1-t0)/60/max(1,cpu_count()-1):.1f} min "
          f"({max(1,cpu_count()-1)} cores)")