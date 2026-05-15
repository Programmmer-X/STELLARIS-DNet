"""
unified/dataset.py
STELLARIS-DNet — Unified Multi-Modal Dataset

Wraps all module datasets into a single training format.
Each sample: (inputs_dict, labels_dict, modality_mask)

Memory-efficient: stores ONLY active modality data per sample.
Zero tensors for absent modalities are created on-the-fly in __getitem__.

Handles:
  - Single-modality samples (most data)
  - Cross-modal M1+M3 samples (ATNF neutron stars)
  - Missing modules (graceful skip if data unavailable)
  - Balanced sampling across modalities
"""

import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified.config import (
    MODALITY_ORDER, MODALITY_INDEX, NUM_ENCODERS,
    ENCODER_DIMS, PROJ_DIM, NUM_STELLAR_CLASSES,
    NUM_PULSAR_SUBTYPES, NUM_RADIO_CLASSES, NUM_REG_TARGETS,
    SCALER_PATHS, SEED, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    VAL_SPLIT, TEST_SPLIT,
)


# ═════════════════════════════════════════════════════════════
# 1. SAMPLE FORMAT CONSTANTS
# ═════════════════════════════════════════════════════════════

# Label sentinel: -1 means "do not compute loss for this head"
IGNORE_LABEL = -1

# Input shapes per modality (for zero-fill when absent)
INPUT_SHAPES = {
    "m1_mlp":      (8,),
    "m1_cnn_time": (1, 64),
    "m1_cnn_freq": (32,),
    "m1_ae":       (64,),
    "m2_rgc":      (3, 224, 224),
    "m2_gwd":      (3, 128, 128),
    "m3":          (7,),
}


def _default_labels() -> dict:
    """Create ignore-sentinel labels for all heads."""
    return {
        "stellar_cls":      IGNORE_LABEL,
        "pulsar_det":       IGNORE_LABEL,
        "pulsar_subtype":   IGNORE_LABEL,
        "radio_morphology": IGNORE_LABEL,
        "gw_det":           IGNORE_LABEL,
        "anomaly":          IGNORE_LABEL,
        "regression":       np.full(NUM_REG_TARGETS, float("nan"), dtype=np.float32),
        "reg_mask":         np.zeros(NUM_REG_TARGETS, dtype=np.float32),
    }


# ═════════════════════════════════════════════════════════════
# 2. PER-MODULE DATA LOADERS
# ═════════════════════════════════════════════════════════════

def _load_m1_mlp_data():
    """
    Load HTRU2 dataset for M1-MLP.
    Returns: list of (features_8dim, is_pulsar_label)
    """
    try:
        import pandas as pd
        from module1.config import CHECKPOINT_DIR as M1_CKPT

        # Find HTRU2 CSV
        candidates = [
            os.path.join("data", "module1", "HTRU_2.csv"),
            "/kaggle/input/htru2-pulsar-dataset/HTRU_2.csv",
            "/kaggle/input/predicting-pulsar-star/pulsar_stars.csv",
        ]
        # Also try module1 config path
        try:
            from module1.config import HTRU2_PATH
            candidates.insert(0, HTRU2_PATH)
        except ImportError:
            pass

        csv_path = None
        for p in candidates:
            if os.path.exists(p):
                csv_path = p
                break

        if csv_path is None:
            print("  ⚠️  HTRU2 CSV not found — skipping M1-MLP")
            return None

        df = pd.read_csv(csv_path, header=None)
        X = df.iloc[:, :8].values.astype(np.float32)
        y = df.iloc[:, 8].values.astype(np.float32)

        # Load or fit scaler
        scaler_path = SCALER_PATHS.get("m1_mlp")
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            X = scaler.transform(X)
        else:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            print("  ⚠️  M1-MLP scaler not found — fit on full data")

        print(f"  ✅ M1-MLP: {len(X)} samples from {os.path.basename(csv_path)}")
        return list(zip(X.astype(np.float32), y))

    except Exception as e:
        print(f"  ⚠️  M1-MLP load failed: {e}")
        return None


def _load_m1_cnn_data():
    """
    Load pulse profiles for M1-CNN and M1-AE.
    Returns: list of (profile_64, fft_32, time_1x64, subtype_label)
    All stored as numpy arrays to save memory.
    """
    try:
        from module1.config import FFT_BINS, SIGNAL_LENGTH
        from module1.dataset import load_pulse_profiles

        train_loader, val_loader, _ = load_pulse_profiles(enhanced=True)

        samples = []
        for loader in [train_loader, val_loader]:
            for batch in loader:
                if isinstance(batch, dict):
                    times = batch["time"]
                    freqs = batch.get("freq", torch.zeros(times.size(0), FFT_BINS))
                    labels = batch["label"]
                else:
                    times, labels = batch[0], batch[1]
                    freqs = torch.zeros(times.size(0), FFT_BINS)

                for i in range(times.size(0)):
                    t = times[i].numpy()             # (1, 64) float32
                    f = freqs[i].numpy()             # (32,) float32
                    profile = t.squeeze(0)           # (64,) float32
                    lbl = int(labels[i].item())
                    samples.append((profile, f, t, lbl))

        print(f"  ✅ M1-CNN/AE: {len(samples)} samples (synthetic profiles)")
        return samples

    except Exception as e:
        print(f"  ⚠️  M1-CNN/AE load failed: {e}")
        return None


def _load_m2_rgc_data():
    """
    Load MiraBest radio galaxy images for M2-RGC.
    Returns: list of (image_3x224x224_numpy, fri_frii_label)
    """
    try:
        from module2.dataset_2a import MiraBestDataset
        ds = MiraBestDataset(split="all")
        samples = []
        for i in range(len(ds)):
            img, lbl = ds[i]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            samples.append((img.astype(np.float32), int(lbl)))
        print(f"  ✅ M2-RGC: {len(samples)} samples (MiraBest)")
        return samples

    except Exception as e:
        print(f"  ⚠️  M2-RGC load failed: {e}")
        return None


def _load_m2_gwd_data():
    """
    Load G2Net CQT spectrograms for M2-GWD.
    Returns: list of (cqt_3x128x128_numpy, signal_label)
    """
    try:
        from module2.dataset_2b import G2NetDataset
        ds = G2NetDataset(split="all")
        samples = []
        for i in range(len(ds)):
            cqt, lbl = ds[i]
            if isinstance(cqt, torch.Tensor):
                cqt = cqt.numpy()
            samples.append((cqt.astype(np.float32), int(lbl)))
        print(f"  ✅ M2-GWD: {len(samples)} samples (G2Net CQT)")
        return samples

    except Exception as e:
        print(f"  ⚠️  M2-GWD load failed: {e}")
        return None


def _load_m3_data():
    """
    Load stellar classification data for M3.
    Returns: list of (features_7dim, class_label, reg_targets_4, reg_mask_4)
    """
    try:
        from module3.dataset import load_stellar_data
        data = load_stellar_data()

        if data is None:
            print("  ⚠️  M3 data returned None — skipping")
            return None

        if isinstance(data, tuple) and len(data) >= 3:
            train_loader, val_loader, test_loader = data[:3]
        else:
            print("  ⚠️  M3 data format unexpected — skipping")
            return None

        samples = []
        for loader in [train_loader, val_loader]:
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    features   = batch[0]
                    cls_labels = batch[1]
                    reg_targets = batch[2] if len(batch) > 2 else None
                    reg_mask    = batch[3] if len(batch) > 3 else None
                elif isinstance(batch, dict):
                    features   = batch["features"]
                    cls_labels = batch["class_label"]
                    reg_targets = batch.get("reg_targets")
                    reg_mask    = batch.get("reg_mask")
                else:
                    continue

                for i in range(features.size(0)):
                    feat    = features[i].numpy().astype(np.float32)
                    cls_lbl = int(cls_labels[i].item())

                    if reg_targets is not None:
                        reg_t = reg_targets[i].numpy().astype(np.float32)
                        reg_m = reg_mask[i].numpy().astype(np.float32) \
                                if reg_mask is not None \
                                else np.ones(NUM_REG_TARGETS, dtype=np.float32)
                    else:
                        reg_t = np.full(NUM_REG_TARGETS, float("nan"),
                                        dtype=np.float32)
                        reg_m = np.zeros(NUM_REG_TARGETS, dtype=np.float32)

                    samples.append((feat, cls_lbl, reg_t, reg_m))

        print(f"  ✅ M3: {len(samples)} samples (Gaia+SDSS+ATNF+MWDD)")
        return samples

    except Exception as e:
        print(f"  ⚠️  M3 load failed: {e}")
        return None


# ═════════════════════════════════════════════════════════════
# 3. UNIFIED DATASET CLASS
# ═════════════════════════════════════════════════════════════

class UnifiedDataset(Dataset):
    """
    Unified multi-modal dataset.

    MEMORY-EFFICIENT: stores only active modality data per sample
    as numpy arrays. Zero-fill tensors created in __getitem__.

    Each __getitem__ returns:
        inputs:  dict of float32 tensors (zero-filled for absent modalities)
        labels:  dict of labels (IGNORE_LABEL for invalid heads)
        mask:    (6,) float32 tensor
    """
    def __init__(self, split: str = "train", verbose: bool = True):
        super().__init__()
        self.split = split
        self.samples = []        # list of (sparse_inputs, labels, mask_list)
        self.source_counts = {}

        if verbose:
            print(f"\n{'═' * 50}")
            print(f"Loading Unified Dataset — split: {split}")
            print(f"{'═' * 50}")

        self._build_from_modules(verbose)

        if verbose:
            print(f"\n── Summary ──")
            total = len(self.samples)
            for src, cnt in sorted(self.source_counts.items()):
                pct = cnt / total * 100 if total > 0 else 0
                print(f"  {src:<16s}: {cnt:>7,}  ({pct:.1f}%)")
            print(f"  {'TOTAL':<16s}: {total:>7,}")
            print(f"{'═' * 50}\n")

    def _build_from_modules(self, verbose: bool):
        """Load each module's data and create unified samples."""

        # ── M3 (largest, most important) ─────────────────────────────
        m3_raw = _load_m3_data()
        if m3_raw:
            m3_split = self._split_data(m3_raw, self.split)
            for feat, cls_lbl, reg_t, reg_m in m3_split:
                sparse = {"m3": feat}
                labels = _default_labels()
                labels["stellar_cls"]  = cls_lbl
                labels["regression"]   = reg_t
                labels["reg_mask"]     = reg_m
                mask_bits = [0, 0, 0, 0, 0, 1]  # only m3 active
                self.samples.append((sparse, labels, mask_bits))
            self.source_counts["m3"] = len(m3_split)

        # ── M1-MLP (HTRU2) ──────────────────────────────────────────
        m1_mlp_raw = _load_m1_mlp_data()
        if m1_mlp_raw:
            m1_mlp_split = self._split_data(m1_mlp_raw, self.split)
            for feat, is_pulsar in m1_mlp_split:
                sparse = {"m1_mlp": feat}
                labels = _default_labels()
                labels["pulsar_det"] = int(is_pulsar)
                mask_bits = [1, 0, 0, 0, 0, 0]  # only m1_mlp active
                self.samples.append((sparse, labels, mask_bits))
            self.source_counts["m1_mlp"] = len(m1_mlp_split)

        # ── M1-CNN + M1-AE (pulse profiles) ─────────────────────────
        m1_cnn_raw = _load_m1_cnn_data()
        if m1_cnn_raw:
            m1_cnn_split = self._split_data(m1_cnn_raw, self.split)
            for profile, fft_feat, time_feat, subtype in m1_cnn_split:
                sparse = {
                    "m1_cnn_time": time_feat,    # (1, 64) numpy
                    "m1_cnn_freq": fft_feat,     # (32,) numpy
                    "m1_ae":       profile,      # (64,) numpy
                }
                labels = _default_labels()
                labels["pulsar_subtype"] = subtype
                labels["pulsar_det"]     = 1   # all profiles are pulsars
                labels["anomaly"]        = 0   # synthetic = non-anomalous
                mask_bits = [0, 1, 1, 0, 0, 0]  # m1_cnn + m1_ae
                self.samples.append((sparse, labels, mask_bits))
            self.source_counts["m1_cnn_ae"] = len(m1_cnn_split)

        # ── M2-RGC (radio galaxies) ─────────────────────────────────
        m2_rgc_raw = _load_m2_rgc_data()
        if m2_rgc_raw:
            m2_rgc_split = self._split_data(m2_rgc_raw, self.split)
            for img, lbl in m2_rgc_split:
                sparse = {"m2_rgc": img}         # (3, 224, 224) numpy
                labels = _default_labels()
                labels["radio_morphology"] = lbl
                mask_bits = [0, 0, 0, 1, 0, 0]  # only m2_rgc
                self.samples.append((sparse, labels, mask_bits))
            self.source_counts["m2_rgc"] = len(m2_rgc_split)

        # ── M2-GWD (gravitational waves) ─────────────────────────────
        m2_gwd_raw = _load_m2_gwd_data()
        if m2_gwd_raw:
            m2_gwd_split = self._split_data(m2_gwd_raw, self.split)
            for cqt, lbl in m2_gwd_split:
                sparse = {"m2_gwd": cqt}         # (3, 128, 128) numpy
                labels = _default_labels()
                labels["gw_det"] = lbl
                mask_bits = [0, 0, 0, 0, 1, 0]  # only m2_gwd
                self.samples.append((sparse, labels, mask_bits))
            self.source_counts["m2_gwd"] = len(m2_gwd_split)

        # ── Cross-modal M1+M3 (neutron star pairs) ──────────────────
        if m3_raw and m1_mlp_raw:
            self._create_cross_modal_ns(m3_raw, m1_mlp_raw, m1_cnn_raw)

    def _create_cross_modal_ns(self, m3_data, m1_mlp_data, m1_cnn_data):
        """
        Create cross-modal samples for neutron stars.
        M3 NS samples paired with M1 pulsar-positive HTRU2 features.
        """
        ns_idx = 3  # Neutron_Star class index
        m3_ns = [(f, c, r, m) for f, c, r, m in m3_data if c == ns_idx]
        m1_pulsars = [(f, l) for f, l in m1_mlp_data if l == 1.0]

        if not m3_ns or not m1_pulsars:
            return

        m3_ns_split = self._split_data(m3_ns, self.split)
        if not m3_ns_split:
            return

        rng = np.random.RandomState(SEED + 99)
        m1_indices = rng.choice(len(m1_pulsars), size=len(m3_ns_split),
                                replace=True)

        count = 0
        for (m3_feat, cls_lbl, reg_t, reg_m), m1_idx in \
                zip(m3_ns_split, m1_indices):
            m1_feat, _ = m1_pulsars[m1_idx]

            sparse = {
                "m3":     m3_feat,
                "m1_mlp": m1_feat,
            }
            labels = _default_labels()
            labels["stellar_cls"]  = cls_lbl
            labels["regression"]   = reg_t
            labels["reg_mask"]     = reg_m
            labels["pulsar_det"]   = 1
            mask_bits = [1, 0, 0, 0, 0, 1]  # m1_mlp + m3

            # Optionally add M1-CNN/AE
            if m1_cnn_data:
                cnn_idx = rng.randint(0, len(m1_cnn_data))
                profile, fft_feat, time_feat, subtype = m1_cnn_data[cnn_idx]
                sparse["m1_cnn_time"] = time_feat
                sparse["m1_cnn_freq"] = fft_feat
                sparse["m1_ae"]       = profile
                labels["pulsar_subtype"] = subtype
                labels["anomaly"]        = 0
                mask_bits[1] = 1  # m1_cnn
                mask_bits[2] = 1  # m1_ae

            self.samples.append((sparse, labels, mask_bits))
            count += 1

        self.source_counts["cross_m1_m3"] = count
        print(f"  ✅ Cross-modal M1+M3: {count} NS samples")

    def _split_data(self, data: list, split: str) -> list:
        """Deterministic 70/15/15 split."""
        n = len(data)
        if n < 10:
            return data if split == "train" else []

        test_ratio = TEST_SPLIT
        val_ratio  = VAL_SPLIT / (1 - TEST_SPLIT)

        train_val, test = train_test_split(
            data, test_size=test_ratio, random_state=SEED
        )
        train, val = train_test_split(
            train_val, test_size=val_ratio, random_state=SEED
        )

        return {"train": train, "val": val, "test": test}[split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns tensors with lazy zero-fill for absent modalities.
        Only active modality data is stored in memory.
        """
        sparse_inputs, labels_raw, mask_bits = self.samples[idx]

        # ── Build input tensors (zero-fill absent) ───────────────
        inputs = {}
        for key, shape in INPUT_SHAPES.items():
            if key in sparse_inputs:
                inputs[key] = torch.as_tensor(
                    sparse_inputs[key], dtype=torch.float32
                )
            else:
                inputs[key] = torch.zeros(shape, dtype=torch.float32)

        # ── Build label tensors ──────────────────────────────────
        labels = {}
        for key in ["stellar_cls", "pulsar_det", "pulsar_subtype",
                     "radio_morphology", "gw_det", "anomaly"]:
            labels[key] = labels_raw[key]

        labels["regression"] = torch.as_tensor(
            labels_raw["regression"], dtype=torch.float32
        )
        labels["reg_mask"] = torch.as_tensor(
            labels_raw["reg_mask"], dtype=torch.float32
        )

        # ── Build mask tensor ────────────────────────────────────
        mask = torch.tensor(mask_bits, dtype=torch.float32)

        return inputs, labels, mask


# ═════════════════════════════════════════════════════════════
# 4. COLLATE FUNCTION
# ═════════════════════════════════════════════════════════════

def unified_collate_fn(batch):
    """
    Custom collate for UnifiedDataset.
    Stacks per-modality inputs, per-head labels, and masks.
    """
    inputs_list, labels_list, masks_list = zip(*batch)
    B = len(batch)

    # ── Stack inputs ─────────────────────────────────────────────
    inputs = {}
    for key in INPUT_SHAPES:
        inputs[key] = torch.stack([inp[key] for inp in inputs_list])

    # ── Stack labels ─────────────────────────────────────────────
    labels = {}
    for key in ["stellar_cls", "pulsar_det", "pulsar_subtype",
                "radio_morphology", "gw_det", "anomaly"]:
        labels[key] = torch.tensor(
            [lbl[key] for lbl in labels_list], dtype=torch.long
        )

    labels["regression"] = torch.stack(
        [lbl["regression"] for lbl in labels_list]
    )
    labels["reg_mask"] = torch.stack(
        [lbl["reg_mask"] for lbl in labels_list]
    )

    # ── Stack masks ──────────────────────────────────────────────
    masks = torch.stack(masks_list)

    return inputs, labels, masks


# ═════════════════════════════════════════════════════════════
# 5. DATALOADER FACTORY
# ═════════════════════════════════════════════════════════════

def get_unified_loaders(
    batch_size:  int  = BATCH_SIZE,
    num_workers: int  = NUM_WORKERS,
    pin_memory:  bool = PIN_MEMORY,
    verbose:     bool = True,
) -> tuple:
    """Create train/val/test DataLoaders with balanced sampling."""

    train_ds = UnifiedDataset(split="train", verbose=verbose)
    val_ds   = UnifiedDataset(split="val",   verbose=False)
    test_ds  = UnifiedDataset(split="test",  verbose=False)

    # ── Balanced sampling ────────────────────────────────────────
    # Weight inversely by source module count so small datasets
    # (MiraBest 800) get sampled as often as large ones (M3 200K)
    sampler = None
    shuffle = True

    if len(train_ds) > 0 and train_ds.source_counts:
        n_sources = len(train_ds.source_counts)
        total = len(train_ds)

        # Build per-sample weight based on source
        source_weight = {}
        for src, cnt in train_ds.source_counts.items():
            source_weight[src] = total / (cnt * n_sources)

        # Assign weight to each sample by tracking source order
        weights = []
        for src in sorted(train_ds.source_counts.keys()):
            cnt = train_ds.source_counts[src]
            w = source_weight[src]
            weights.extend([w] * cnt)

        if len(weights) == total:
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=total,
                replacement=True,
            )
            shuffle = False

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle,
        sampler=sampler, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=unified_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=unified_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=unified_collate_fn,
    )

    return train_loader, val_loader, test_loader


# ═════════════════════════════════════════════════════════════
# 6. SANITY CHECK
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Unified Dataset — Sanity Check")
    print("=" * 60)

    ds = UnifiedDataset(split="train", verbose=True)

    if len(ds) == 0:
        print("\n⚠️  No data loaded — attach datasets and retry")
    else:
        # Check first sample
        inputs, labels, mask = ds[0]

        print("\n── Sample 0 ──")
        print("Inputs:")
        for k, v in inputs.items():
            nz = "★ ACTIVE" if v.abs().sum() > 0 else "  (zeros)"
            print(f"  {k:<16s}: {list(v.shape)}  {nz}")
        print("Labels:")
        for k, v in labels.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k:<16s}: {list(v.shape)}")
            else:
                val = v if v != IGNORE_LABEL else "(ignore)"
                print(f"  {k:<16s}: {val}")
        print(f"Mask: {mask.tolist()}")
        active = [MODALITY_ORDER[i] for i in range(NUM_ENCODERS)
                  if mask[i] > 0]
        print(f"Active: {active}")

        # Memory estimate
        sample_bytes = sum(v.nelement() * 4 for v in inputs.values())
        print(f"\nPer-sample tensor size: {sample_bytes / 1024:.1f} KB"
              f" (created on-the-fly, NOT stored)")

        # Collate test
        print("\n── Collate Test ──")
        batch = [ds[i] for i in range(min(4, len(ds)))]
        inp_b, lbl_b, msk_b = unified_collate_fn(batch)
        print(f"Batch size: {msk_b.shape[0]}")
        for k, v in inp_b.items():
            print(f"  inputs[{k:<16s}]: {list(v.shape)}")
        for k, v in lbl_b.items():
            print(f"  labels[{k:<16s}]: {list(v.shape)}")
        print(f"  masks: {list(msk_b.shape)}")

        # Memory footprint of stored data
        import sys as _sys
        raw_bytes = _sys.getsizeof(ds.samples)
        print(f"\nDataset memory: ~{raw_bytes / 1024 / 1024:.1f} MB"
              f" (sparse storage, {len(ds)} samples)")

    print("\n" + "=" * 60)
    print("✅ Dataset sanity check complete")
    print("=" * 60)