"""
module1/dataset.py
STELLARIS-DNet — Module 1 Data Pipeline
Handles: HTRU2 (MLP) + Raw pulse profiles (1D CNN) + Autoencoder data

Upgrades:
  - Noise injection + RFI/glitch simulation
  - FFT-based frequency feature extraction
  - Optional CQT spectrogram generation
  - Multi-modal outputs (time-domain + frequency-domain)
  - Data augmentation (shift, scale, distortion)
  - Signal energy feature computation
  - All features gated by config.py toggles
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module1.config import *


# ═════════════════════════════════════════════
# SECTION 1 — SIGNAL PROCESSING UTILITIES
# All functions are stateless and config-driven
# ═════════════════════════════════════════════

def inject_noise(profile: np.ndarray, std: float = NOISE_STD) -> np.ndarray:
    """Add Gaussian noise to a normalized pulse profile."""
    return profile + np.random.normal(0.0, std, size=profile.shape).astype(np.float32)


def inject_rfi(
    profile:   np.ndarray,
    prob:      float = RFI_PROB,
    width_min: int   = RFI_WIDTH_MIN,
    width_max: int   = RFI_WIDTH_MAX,
    amp_scale: float = RFI_AMP_SCALE,
) -> np.ndarray:
    """
    Simulate RFI (Radio Frequency Interference) glitch with probability `prob`.
    Injects a narrow spike of random width and amplitude.
    """
    if np.random.rand() > prob:
        return profile
    profile  = profile.copy()
    width    = np.random.randint(width_min, width_max + 1)
    start    = np.random.randint(0, max(1, len(profile) - width))
    amp      = amp_scale * profile.max()
    profile[start : start + width] = amp
    return profile


def augment_profile(profile: np.ndarray) -> np.ndarray:
    """
    Apply three augmentations in sequence (all optional via config):
      1. Circular shift  — preserves profile shape, changes phase
      2. Amplitude scale — simulates distance/flux variation
      3. Gaussian noise  — simulates receiver noise
    """
    profile = profile.copy()

    # 1. Circular shift
    shift   = np.random.randint(-SHIFT_MAX, SHIFT_MAX + 1)
    profile = np.roll(profile, shift)

    # 2. Amplitude scaling
    scale   = np.random.uniform(SCALE_MIN, SCALE_MAX)
    profile = profile * scale

    # 3. Noise
    profile = inject_noise(profile, NOISE_STD)

    return profile.astype(np.float32)


def extract_fft_features(profile: np.ndarray, n_bins: int = FFT_BINS) -> np.ndarray:
    """
    Compute real FFT of a 1D profile and return the magnitude spectrum.
    Returns first `n_bins` bins (positive frequencies only).
    Output is log-scaled and normalized to [0, 1].
    """
    fft_mag  = np.abs(np.fft.rfft(profile))[:n_bins]
    fft_log  = np.log1p(fft_mag)                # log scale: suppresses dynamic range
    max_val  = fft_log.max()
    if max_val > 1e-8:
        fft_log /= max_val
    return fft_log.astype(np.float32)


def compute_energy(profile: np.ndarray) -> float:
    """
    Signal energy proxy: mean squared amplitude.
    Physically related to pulse luminosity flux.
    Returned as scalar; appended to feature vectors where relevant.
    """
    return float(np.mean(profile ** 2))


def extract_cqt(
    profile:  np.ndarray,
    n_bins:   int = CQT_BINS,
    n_steps:  int = CQT_STEPS,
    sr:       float = SAMPLE_RATE,
) -> np.ndarray:
    """
    Compute a Constant-Q Transform (CQT) approximation via FFT convolution.
    Returns a (n_bins, n_steps) magnitude spectrogram normalized to [0, 1].

    Uses log-spaced frequency bins from fmin → Nyquist.
    Lightweight implementation — no external library required.
    """
    N       = len(profile)
    freqs   = np.logspace(np.log10(1.0), np.log10(sr / 2.0), n_bins)
    times   = np.linspace(0, N - 1, n_steps).astype(int)

    spec    = np.zeros((n_bins, n_steps), dtype=np.float32)
    t_idx   = np.arange(N)

    for i, f0 in enumerate(freqs):
        # Gaussian-windowed complex sinusoid (Gabor atom)
        sigma  = N / (8.0 * f0 + 1e-8)
        window = np.exp(-0.5 * ((t_idx - N // 2) / (sigma + 1e-8)) ** 2)
        kernel = window * np.exp(2j * np.pi * f0 * t_idx / N)
        conv   = np.abs(np.fft.ifft(np.fft.fft(profile) * np.conj(np.fft.fft(kernel))))
        spec[i] = conv[times]

    # Normalize to [0, 1]
    s_min, s_max = spec.min(), spec.max()
    if s_max - s_min > 1e-8:
        spec = (spec - s_min) / (s_max - s_min)
    return spec


# ═════════════════════════════════════════════
# SECTION 2 — PROFILE UTILITIES (unchanged API)
# ═════════════════════════════════════════════

def pad_or_trim(signal: np.ndarray, length: int = SIGNAL_LENGTH) -> np.ndarray:
    """Pad with zeros or trim to fixed length."""
    if len(signal) >= length:
        return signal[:length]
    return np.pad(signal, (0, length - len(signal)), mode="constant")


def normalize_profile(profile: np.ndarray) -> np.ndarray:
    """Min-max normalize a single pulse profile to [0, 1]."""
    mn, mx = profile.min(), profile.max()
    if mx - mn < 1e-8:
        return np.zeros_like(profile)
    return (profile - mn) / (mx - mn)


def preprocess_profile(
    raw:     np.ndarray,
    length:  int = SIGNAL_LENGTH
) -> np.ndarray:
    """Full preprocessing pipeline: pad/trim → normalize."""
    return normalize_profile(pad_or_trim(raw, length))


# ═════════════════════════════════════════════
# SECTION 3 — HTRU2 DATASET (MLP, unchanged)
# 8 statistical features → pulsar / non-pulsar
# ═════════════════════════════════════════════

class HTRU2Dataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels,   dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_htru2(path: str = HTRU2_PATH):
    """
    Loads HTRU2 CSV, splits train/val/test, normalizes features.
    Returns: train_loader, val_loader, test_loader, scaler, pos_weight
    API unchanged from original.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"HTRU2 not found at {path}\n"
            f"Download: https://archive.ics.uci.edu/dataset/372/htru2\n"
            f"Place HTRU_2.csv in data/module1/"
        )

    df = pd.read_csv(path, header=None)
    X  = df.iloc[:, :8].values.astype(np.float32)
    y  = df.iloc[:,  8].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        random_state=SEED, stratify=y_train
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    classes       = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    pos_weight    = torch.tensor(
        [class_weights[1] / class_weights[0]], dtype=torch.float32
    )

    train_loader = DataLoader(
        HTRU2Dataset(X_train, y_train),
        batch_size=MLP_BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        HTRU2Dataset(X_val, y_val),
        batch_size=MLP_BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        HTRU2Dataset(X_test, y_test),
        batch_size=MLP_BATCH_SIZE, shuffle=False
    )

    print(f"✅ HTRU2 loaded")
    print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"   Pulsars in train: {int(y_train.sum())} / {len(y_train)}")
    print(f"   Pos weight: {pos_weight.item():.2f}")

    return train_loader, val_loader, test_loader, scaler, pos_weight


# ═════════════════════════════════════════════
# SECTION 4 — PULSE PROFILE DATASET (1D CNN)
# Upgraded: multi-modal output, augmentation,
# FFT features, RFI injection, energy proxy
# ═════════════════════════════════════════════

class PulseProfileDataset(Dataset):
    """
    Original single-modal dataset.
    Returns (X_time, y) — (B,1,64), (B,)
    Used when USE_FFT=False and USE_AUGMENTATION=False.
    """
    def __init__(self, profiles: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(
            profiles[:, np.newaxis, :], dtype=torch.float32
        )
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EnhancedPulseProfileDataset(Dataset):
    """
    Upgraded multi-modal dataset.
    Returns dict with keys: 'time', 'freq', 'energy', 'label'
      - time:   (1, SIGNAL_LENGTH)   always present
      - freq:   (FFT_BINS,)          present if USE_FFT=True
      - cqt:    (CQT_BINS, CQT_STEPS) present if USE_CQT=True
      - energy: scalar float         present if USE_PHYSICS_FEATURES=True
      - label:  long scalar

    Applies augmentation + RFI injection at __getitem__ time (training only).
    """
    def __init__(
        self,
        profiles:    np.ndarray,
        labels:      np.ndarray,
        augment:     bool = False,
        inject_rfi_: bool = False,
    ):
        self.profiles    = profiles             # (N, SIGNAL_LENGTH) float32
        self.labels      = labels               # (N,) int64
        self.augment     = augment
        self.inject_rfi_ = inject_rfi_

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, idx):
        profile = self.profiles[idx].copy()

        # ── Augmentation (training only) ──────────────
        if self.augment and USE_AUGMENTATION:
            profile = augment_profile(profile)
        if self.inject_rfi_ and USE_AUGMENTATION:
            profile = inject_rfi(profile)

        # Clip to [0,1] after augmentation
        profile = np.clip(profile, 0.0, 1.0).astype(np.float32)

        out = {}

        # Time-domain — always included
        out["time"] = torch.tensor(
            profile[np.newaxis, :], dtype=torch.float32  # (1, L)
        )

        # FFT frequency features
        if USE_FFT:
            out["freq"] = torch.tensor(
                extract_fft_features(profile, FFT_BINS), dtype=torch.float32
            )

        # CQT spectrogram
        if USE_CQT:
            cqt = extract_cqt(profile, CQT_BINS, CQT_STEPS, SAMPLE_RATE)
            out["cqt"] = torch.tensor(cqt, dtype=torch.float32)  # (B, H, W)

        # Energy proxy feature
        if USE_PHYSICS_FEATURES:
            out["energy"] = torch.tensor(
                [compute_energy(profile)], dtype=torch.float32
            )

        out["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return out


def _build_pulse_loaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    enhanced: bool
):
    """
    Internal helper — builds DataLoaders from split arrays.
    enhanced=True  → EnhancedPulseProfileDataset (multi-modal)
    enhanced=False → PulseProfileDataset (original)
    """
    if enhanced:
        train_ds = EnhancedPulseProfileDataset(
            X_train, y_train, augment=True, inject_rfi_=True
        )
        val_ds = EnhancedPulseProfileDataset(
            X_val, y_val, augment=False, inject_rfi_=False
        )
        test_ds = EnhancedPulseProfileDataset(
            X_test, y_test, augment=False, inject_rfi_=False
        )
    else:
        train_ds = PulseProfileDataset(X_train, y_train)
        val_ds   = PulseProfileDataset(X_val,   y_val)
        test_ds  = PulseProfileDataset(X_test,  y_test)

    return (
        DataLoader(train_ds, batch_size=CNN_BATCH_SIZE, shuffle=True,  drop_last=True),
        DataLoader(val_ds,   batch_size=CNN_BATCH_SIZE, shuffle=False),
        DataLoader(test_ds,  batch_size=CNN_BATCH_SIZE, shuffle=False),
    )


def load_pulse_profiles(
    path:     str  = LOTAAS_PATH,
    enhanced: bool = None,          # None → reads USE_FFT or USE_CQT from config
):
    """
    Loads LOTAAS pulse profiles or falls back to synthetic.
    enhanced=True  → returns EnhancedPulseProfileDataset loaders (multi-modal dicts)
    enhanced=False → returns original PulseProfileDataset loaders (X, y) tuples
    enhanced=None  → auto-selects based on USE_FFT / USE_CQT / USE_AUGMENTATION

    Returns: train_loader, val_loader, test_loader
    API is backward-compatible; callers that unpack 3 values still work.
    """
    if enhanced is None:
        enhanced = USE_FFT or USE_CQT or USE_AUGMENTATION

    if os.path.exists(path):
        data     = np.load(path, allow_pickle=True).item()
        profiles = data["profiles"]
        labels   = data["labels"]
        print(f"✅ LOTAAS profiles loaded: {len(profiles)} samples")
    else:
        print("⚠️  LOTAAS data not found — using synthetic profiles")
        print(f"   Download: https://www.astron.nl/lotaas/")
        profiles, labels = _generate_synthetic_profiles(n=2000)

    profiles = np.array([preprocess_profile(p) for p in profiles], dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        profiles, labels,
        test_size=TEST_SPLIT, random_state=SEED, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        random_state=SEED, stratify=y_train
    )

    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"   Mode: {'enhanced (multi-modal)' if enhanced else 'standard'}")

    return _build_pulse_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, enhanced
    )


# ═════════════════════════════════════════════
# SECTION 5 — AUTOENCODER DATASET (Magnetar)
# Upgraded: multi-modal, energy proxy, RFI
# ═════════════════════════════════════════════

class AutoencoderDataset(Dataset):
    """
    Original flat dataset — no labels, reconstruction only.
    Returns X: (SIGNAL_LENGTH,) tensor.
    """
    def __init__(self, profiles: np.ndarray):
        self.X = torch.tensor(profiles, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class EnhancedAutoencoderDataset(Dataset):
    """
    Upgraded AE dataset.
    Returns dict: {'input': (L,), 'freq': (FFT_BINS,), 'energy': (1,)}
    Augmentation applied at train time (augment=True).
    """
    def __init__(self, profiles: np.ndarray, augment: bool = False):
        self.profiles = profiles
        self.augment  = augment

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, idx):
        profile = self.profiles[idx].copy()

        if self.augment and USE_AUGMENTATION:
            profile = augment_profile(profile)
            profile = inject_rfi(profile)
            profile = np.clip(profile, 0.0, 1.0).astype(np.float32)

        out = {"input": torch.tensor(profile, dtype=torch.float32)}

        if USE_FFT:
            out["freq"] = torch.tensor(
                extract_fft_features(profile, FFT_BINS), dtype=torch.float32
            )
        if USE_PHYSICS_FEATURES:
            out["energy"] = torch.tensor(
                [compute_energy(profile)], dtype=torch.float32
            )
        return out


def load_autoencoder_data(
    path:     str  = LOTAAS_PATH,
    enhanced: bool = None,
):
    """
    Loads ONLY normal pulsar profiles (label=0) for AE training.
    enhanced=None → auto-selects based on config flags.
    Returns: train_loader, val_loader
    """
    if enhanced is None:
        enhanced = USE_FFT or USE_AUGMENTATION

    if os.path.exists(path):
        data     = np.load(path, allow_pickle=True).item()
        profiles = data["profiles"]
        labels   = data["labels"]
        mask     = labels == 0
        profiles = profiles[mask]
        print(f"✅ AE data: {len(profiles)} normal pulsar profiles loaded")
    else:
        print("⚠️  LOTAAS not found — using synthetic normal profiles")
        profiles, _ = _generate_synthetic_profiles(n=1500, normal_only=True)

    profiles = np.array([preprocess_profile(p) for p in profiles], dtype=np.float32)

    X_train, X_val = train_test_split(
        profiles, test_size=VAL_SPLIT, random_state=SEED
    )

    if enhanced:
        train_ds = EnhancedAutoencoderDataset(X_train, augment=True)
        val_ds   = EnhancedAutoencoderDataset(X_val,   augment=False)
    else:
        train_ds = AutoencoderDataset(X_train)
        val_ds   = AutoencoderDataset(X_val)

    train_loader = DataLoader(
        train_ds, batch_size=AE_BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=AE_BATCH_SIZE, shuffle=False
    )

    print(f"   AE Train: {len(X_train)} | AE Val: {len(X_val)}")
    print(f"   Mode: {'enhanced' if enhanced else 'standard'}")
    return train_loader, val_loader


# ═════════════════════════════════════════════
# SECTION 6 — SYNTHETIC PROFILE GENERATOR
# Fallback when LOTAAS data is unavailable
# Generates 4 subtype shapes with noise
# ═════════════════════════════════════════════

def _generate_synthetic_profiles(
    n:           int  = 2000,
    normal_only: bool = False
) -> tuple:
    """
    Generates synthetic pulsar profiles with physically distinct shapes.
    Subtypes:
      0 — Normal:     single broad Gaussian
      1 — Millisecond: narrow peak + interpulse
      2 — Binary:     distorted profile with drift
      3 — Recycled:   sharp narrow peak
    """
    np.random.seed(SEED)
    t        = np.linspace(0, 1, SIGNAL_LENGTH)
    profiles = []
    labels   = []

    n_classes     = 1 if normal_only else NUM_PULSAR_CLASSES
    n_per_class   = n // n_classes

    def gaussian(t, mu, sigma, amp=1.0):
        return amp * np.exp(-0.5 * ((t - mu) / (sigma + 1e-8)) ** 2)

    for cls in range(n_classes):
        for _ in range(n_per_class):
            noise = np.random.normal(0, NOISE_STD, SIGNAL_LENGTH)
            if cls == 0:
                p = gaussian(t, 0.5, 0.08) + noise
            elif cls == 1:
                p = (gaussian(t, 0.5, 0.02) +
                     gaussian(t, 0.0, 0.015, 0.4) + noise)
            elif cls == 2:
                drift = 0.05 * np.sin(2 * np.pi * t)
                p = gaussian(t, 0.5 + drift.mean(), 0.06) + noise
            else:
                p = gaussian(t, 0.5, 0.015, 1.2) + noise

            profiles.append(p.astype(np.float32))
            labels.append(cls)

    return np.array(profiles), np.array(labels, dtype=np.int64)


# ═════════════════════════════════════════════
# SANITY CHECK
# ═════════════════════════════════════════════
if __name__ == "__main__":
    import torch
    print("=" * 60)
    print("Module 1 Dataset Sanity Check — Enhanced Pipeline")
    print("=" * 60)

    # ── HTRU2 ──
    try:
        tr, vl, te, sc, pw = load_htru2()
        X_b, y_b = next(iter(tr))
        print(f"[HTRU2]  batch: {X_b.shape} | labels: {y_b.shape}")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")

    # ── Signal utilities ──
    print()
    dummy = np.random.rand(SIGNAL_LENGTH).astype(np.float32)
    noisy = inject_noise(dummy)
    rfi   = inject_rfi(dummy, prob=1.0)
    aug   = augment_profile(dummy)
    fft_f = extract_fft_features(dummy)
    en    = compute_energy(dummy)
    print(f"[Utils]  inject_noise: {noisy.shape} | inject_rfi: {rfi.shape}")
    print(f"[Utils]  augment: {aug.shape} | fft: {fft_f.shape} | energy: {en:.6f}")

    if USE_CQT:
        cqt = extract_cqt(dummy)
        print(f"[Utils]  CQT shape: {cqt.shape}")

    # ── Enhanced pulse profiles ──
    print()
    tr, vl, te = load_pulse_profiles(enhanced=True)
    batch = next(iter(tr))
    print(f"[CNN Enhanced] keys: {list(batch.keys())}")
    print(f"   time:  {batch['time'].shape}")
    if "freq" in batch:
        print(f"   freq:  {batch['freq'].shape}")
    if "energy" in batch:
        print(f"   energy:{batch['energy'].shape}")
    print(f"   label: {batch['label'].shape}")

    # ── Standard pulse profiles (backward compat) ──
    print()
    tr_std, _, _ = load_pulse_profiles(enhanced=False)
    X_b, y_b = next(iter(tr_std))
    print(f"[CNN Standard] time: {X_b.shape} | labels: {y_b.shape}")

    # ── Enhanced AE ──
    print()
    tr_ae, vl_ae = load_autoencoder_data(enhanced=True)
    ae_batch = next(iter(tr_ae))
    if isinstance(ae_batch, dict):
        print(f"[AE Enhanced] keys: {list(ae_batch.keys())}")
        print(f"   input: {ae_batch['input'].shape}")
    else:
        print(f"[AE Standard] input: {ae_batch.shape}")

    # ── Standard AE (backward compat) ──
    tr_ae_std, _ = load_autoencoder_data(enhanced=False)
    ae_b = next(iter(tr_ae_std))
    print(f"[AE Standard] input: {ae_b.shape}")

    print()
    print("✅ dataset.py — All checks passed")