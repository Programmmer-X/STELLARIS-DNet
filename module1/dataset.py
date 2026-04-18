"""
module1/dataset.py
STELLARIS-DNet — Module 1 Data Pipeline
Handles: HTRU2 (MLP) + Raw pulse profiles (1D CNN) + Autoencoder data
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module1.config import *


# ─────────────────────────────────────────────
# 1. HTRU2 DATASET (MLP Binary Classifier)
# 8 statistical features → pulsar / non-pulsar
# ─────────────────────────────────────────────
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
    Loads HTRU2 CSV, splits into train/val/test,
    normalizes features, returns DataLoaders + class weights.

    HTRU2 columns (no header in file):
    0-3: integrated profile stats (mean, std, kurtosis, skewness)
    4-7: DM-SNR curve stats      (mean, std, kurtosis, skewness)
    8:   label (0=noise, 1=pulsar)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"HTRU2 not found at {path}\n"
            f"Download from: https://archive.ics.uci.edu/dataset/372/htru2\n"
            f"Place HTRU_2.csv in data/module1/"
        )

    df = pd.read_csv(path, header=None)
    X  = df.iloc[:, :8].values.astype(np.float32)
    y  = df.iloc[:,  8].values.astype(np.float32)

    # Train / val / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        random_state=SEED, stratify=y_train
    )

    # Normalize — fit on train only
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Class weights for imbalance (HTRU2 ~9:1)
    classes        = np.unique(y_train)
    class_weights  = compute_class_weight("balanced", classes=classes, y=y_train)
    pos_weight     = torch.tensor([class_weights[1] / class_weights[0]],
                                   dtype=torch.float32)

    train_loader = DataLoader(
        HTRU2Dataset(X_train, y_train),
        batch_size=MLP_BATCH_SIZE, shuffle=True,  drop_last=True
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
    print(f"   Pos weight (imbalance correction): {pos_weight.item():.2f}")

    return train_loader, val_loader, test_loader, scaler, pos_weight


# ─────────────────────────────────────────────
# 2. PULSE PROFILE DATASET (1D CNN)
# Raw 1D signal → pulsar subtype classification
# ─────────────────────────────────────────────
class PulseProfileDataset(Dataset):
    def __init__(self, profiles: np.ndarray, labels: np.ndarray):
        # profiles shape: (N, SIGNAL_LENGTH)
        # Add channel dim → (N, 1, SIGNAL_LENGTH) for Conv1d
        self.X = torch.tensor(
            profiles[:, np.newaxis, :], dtype=torch.float32
        )
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def pad_or_trim(signal: np.ndarray, length: int = SIGNAL_LENGTH) -> np.ndarray:
    """Pads with zeros or trims to fixed length."""
    if len(signal) >= length:
        return signal[:length]
    return np.pad(signal, (0, length - len(signal)), mode="constant")


def normalize_profile(profile: np.ndarray) -> np.ndarray:
    """Min-max normalize a single pulse profile to [0, 1]."""
    mn, mx = profile.min(), profile.max()
    if mx - mn < 1e-8:
        return np.zeros_like(profile)
    return (profile - mn) / (mx - mn)


def load_pulse_profiles(path: str = LOTAAS_PATH):
    """
    Loads LOTAAS pulse profile .npy file.
    Expected format: dict with keys 'profiles' and 'labels'
    profiles: (N, variable_length) or (N, SIGNAL_LENGTH)
    labels:   (N,) int array of subtype indices

    If LOTAAS data unavailable, generates synthetic profiles for testing.
    """
    if os.path.exists(path):
        data     = np.load(path, allow_pickle=True).item()
        profiles = data["profiles"]
        labels   = data["labels"]
        print(f"✅ LOTAAS profiles loaded: {len(profiles)} samples")
    else:
        print("⚠️  LOTAAS data not found — using synthetic profiles for dev/test")
        print(f"   Download from: https://www.astron.nl/lotaas/")
        profiles, labels = _generate_synthetic_profiles(n=2000)

    # Pad/trim + normalize each profile
    profiles = np.array([
        normalize_profile(pad_or_trim(p, SIGNAL_LENGTH))
        for p in profiles
    ], dtype=np.float32)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        profiles, labels,
        test_size=TEST_SPLIT, random_state=SEED, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        random_state=SEED, stratify=y_train
    )

    train_loader = DataLoader(
        PulseProfileDataset(X_train, y_train),
        batch_size=CNN_BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        PulseProfileDataset(X_val, y_val),
        batch_size=CNN_BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        PulseProfileDataset(X_test, y_test),
        batch_size=CNN_BATCH_SIZE, shuffle=False
    )

    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# 3. AUTOENCODER DATASET (Magnetar Anomaly)
# Trained ONLY on normal pulsar signals
# ─────────────────────────────────────────────
class AutoencoderDataset(Dataset):
    def __init__(self, profiles: np.ndarray):
        self.X = torch.tensor(profiles, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]   # no label — reconstruction task


def load_autoencoder_data(path: str = LOTAAS_PATH):
    """
    Loads ONLY normal pulsar profiles (label=0) for autoencoder training.
    At inference, any signal with high reconstruction error is flagged
    as a potential magnetar candidate.
    """
    if os.path.exists(path):
        data     = np.load(path, allow_pickle=True).item()
        profiles = data["profiles"]
        labels   = data["labels"]
        # Keep only normal pulsars (label 0)
        mask     = labels == 0
        profiles = profiles[mask]
        print(f"✅ AE data: {len(profiles)} normal pulsar profiles")
    else:
        print("⚠️  LOTAAS not found — using synthetic normal profiles")
        profiles, _ = _generate_synthetic_profiles(n=1500, normal_only=True)

    profiles = np.array([
        normalize_profile(pad_or_trim(p, SIGNAL_LENGTH))
        for p in profiles
    ], dtype=np.float32)

    X_train, X_val = train_test_split(
        profiles, test_size=VAL_SPLIT, random_state=SEED
    )

    train_loader = DataLoader(
        AutoencoderDataset(X_train),
        batch_size=AE_BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        AutoencoderDataset(X_val),
        batch_size=AE_BATCH_SIZE, shuffle=False
    )

    print(f"   AE Train: {len(X_train)} | AE Val: {len(X_val)}")
    return train_loader, val_loader


# ─────────────────────────────────────────────
# 4. SYNTHETIC PROFILE GENERATOR
# Used when LOTAAS data is unavailable
# Generates physically plausible pulse shapes
# ─────────────────────────────────────────────
def _generate_synthetic_profiles(
    n: int = 2000,
    normal_only: bool = False
) -> tuple:
    """
    Generates synthetic pulsar profiles for development/testing.
    Each subtype has distinct pulse shape characteristics:
    - Normal:      single broad Gaussian peak
    - Millisecond: narrow peak + interpulse
    - Binary:      distorted profile with drift
    - Recycled:    sharp narrow peak
    """
    np.random.seed(SEED)
    t        = np.linspace(0, 1, SIGNAL_LENGTH)
    profiles = []
    labels   = []

    n_per_class = n // (1 if normal_only else NUM_PULSAR_CLASSES)

    def gaussian(t, mu, sigma, amp=1.0):
        return amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)

    for cls in range(1 if normal_only else NUM_PULSAR_CLASSES):
        for _ in range(n_per_class):
            noise = np.random.normal(0, 0.02, SIGNAL_LENGTH)
            if cls == 0:   # Normal — broad single peak
                p = gaussian(t, 0.5, 0.08) + noise
            elif cls == 1: # Millisecond — narrow + interpulse
                p = (gaussian(t, 0.5, 0.02) +
                     gaussian(t, 0.0, 0.015, 0.4) + noise)
            elif cls == 2: # Binary — drifting distorted
                drift = 0.05 * np.sin(2 * np.pi * t)
                p = gaussian(t, 0.5 + drift.mean(), 0.06) + noise
            else:          # Recycled — sharp narrow
                p = gaussian(t, 0.5, 0.015, 1.2) + noise

            profiles.append(p.astype(np.float32))
            labels.append(cls)

    return np.array(profiles), np.array(labels, dtype=np.int64)


# ─────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Module 1 Dataset Sanity Check")
    print("=" * 50)

    # Test HTRU2 (will fail gracefully if file missing)
    try:
        tr, vl, te, sc, pw = load_htru2()
        X_batch, y_batch = next(iter(tr))
        print(f"HTRU2 batch shape: {X_batch.shape}, labels: {y_batch.shape}")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")

    # Test pulse profiles (uses synthetic if LOTAAS missing)
    print()
    tr, vl, te = load_pulse_profiles()
    X_batch, y_batch = next(iter(tr))
    print(f"Pulse profile batch: {X_batch.shape}, labels: {y_batch.shape}")

    # Test autoencoder data
    print()
    tr, vl = load_autoencoder_data()
    X_batch = next(iter(tr))
    print(f"AE batch shape: {X_batch.shape}")

    print()
    print("✅ dataset.py OK")