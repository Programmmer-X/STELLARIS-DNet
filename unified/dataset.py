"""
unified/dataset.py
STELLARIS-DNet — Unified Multi-Modal Dataset

Wraps all module datasets into a single training format.
Each sample: (inputs_dict, labels_dict, modality_mask)

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
# 1. UNIFIED SAMPLE FORMAT
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


def _empty_inputs() -> dict:
    """Create zero-filled input tensors for all modalities."""
    return {k: torch.zeros(v) for k, v in INPUT_SHAPES.items()}


def _empty_labels() -> dict:
    """Create ignore-sentinel labels for all heads."""
    return {
        "stellar_cls":      IGNORE_LABEL,
        "pulsar_det":       IGNORE_LABEL,
        "pulsar_subtype":   IGNORE_LABEL,
        "radio_morphology": IGNORE_LABEL,
        "gw_det":           IGNORE_LABEL,
        "anomaly":          IGNORE_LABEL,
        "regression":       torch.full((NUM_REG_TARGETS,), float("nan")),
        "reg_mask":         torch.zeros(NUM_REG_TARGETS),
    }


def _empty_mask() -> torch.Tensor:
    """Create zero modality mask."""
    return torch.zeros(NUM_ENCODERS)


# ═════════════════════════════════════════════════════════════
# 2. PER-MODULE DATA LOADERS
# ═════════════════════════════════════════════════════════════

def _load_m1_mlp_data():
    """
    Load HTRU2 dataset for M1-MLP.
    Returns: list of (features_8dim, is_pulsar_label)
    """
    try:
        from module1.config import HTRU2_PATH, CHECKPOINT_DIR as M1_CKPT
        import pandas as pd

        # Find HTRU2 CSV
        candidates = [
            HTRU2_PATH,
            os.path.join("data", "module1", "HTRU_2.csv"),
            "/kaggle/input/htru2-pulsar-dataset/HTRU_2.csv",
            "/kaggle/input/predicting-pulsar-star/pulsar_stars.csv",
        ]
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
            print("  ⚠️  M1-MLP scaler not found — fit on full data (no leakage control)")

        print(f"  ✅ M1-MLP: {len(X)} samples loaded from {csv_path}")
        return list(zip(X.astype(np.float32), y))

    except Exception as e:
        print(f"  ⚠️  M1-MLP load failed: {e}")
        return None


def _load_m1_cnn_data():
    """
    Load pulse profiles for M1-CNN and M1-AE.
    Returns: list of (profile_64, fft_32, subtype_label)
    """
    try:
        from module1.config import FFT_BINS, SIGNAL_LENGTH
        from module1.dataset import load_pulse_profiles

        train_loader, val_loader, _ = load_pulse_profiles(enhanced=True)

        samples = []
        for loader in [train_loader, val_loader]:
            for batch in loader:
                if isinstance(batch, dict):
                    times = batch["time"]       # (B, 1, 64)
                    freqs = batch.get("freq", torch.zeros(times.size(0), FFT_BINS))
                    labels = batch["label"]     # (B,)
                else:
                    times, labels = batch[0], batch[1]
                    freqs = torch.zeros(times.size(0), FFT_BINS)

                for i in range(times.size(0)):
                    t = times[i].numpy()                     # (1, 64)
                    f = freqs[i].numpy()                     # (32,)
                    profile = t.squeeze(0)                    # (64,)
                    lbl = int(labels[i].item())
                    samples.append((profile, f, t, lbl))

        print(f"  ✅ M1-CNN/AE: {len(samples)} samples loaded (synthetic profiles)")
        return samples

    except Exception as e:
        print(f"  ⚠️  M1-CNN/AE load failed: {e}")
        return None


def _load_m2_rgc_data():
    """
    Load MiraBest radio galaxy images for M2-RGC.
    Returns: list of (image_3x224x224, fri_frii_label)
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

        print(f"  ✅ M2-RGC: {len(samples)} samples loaded (MiraBest)")
        return samples

    except Exception as e:
        print(f"  ⚠️  M2-RGC load failed: {e}")
        return None


def _load_m2_gwd_data():
    """
    Load G2Net CQT spectrograms for M2-GWD.
    Returns: list of (cqt_3x128x128, signal_label)
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

        print(f"  ✅ M2-GWD: {len(samples)} samples loaded (G2Net CQT)")
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
        from module3.config import NUM_FEATURES as M3_FEATURES
        from module3.dataset import load_stellar_data

        data = load_stellar_data()

        if data is None:
            print("  ⚠️  M3 data returned None — skipping")
            return None

        # Unpack based on return format
        if isinstance(data, tuple) and len(data) >= 3:
            train_loader, val_loader, test_loader = data[:3]
        else:
            print("  ⚠️  M3 data format unexpected — skipping")
            return None

        samples = []
        for loader in [train_loader, val_loader]:
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    features = batch[0]         # (B, 7)
                    cls_labels = batch[1]        # (B,)
                    reg_targets = batch[2] if len(batch) > 2 else None
                    reg_mask = batch[3] if len(batch) > 3 else None
                elif isinstance(batch, dict):
                    features = batch["features"]
                    cls_labels = batch["class_label"]
                    reg_targets = batch.get("reg_targets")
                    reg_mask = batch.get("reg_mask")
                else:
                    continue

                for i in range(features.size(0)):
                    feat = features[i].numpy()
                    cls_lbl = int(cls_labels[i].item())

                    if reg_targets is not None:
                        reg_t = reg_targets[i].numpy()
                        reg_m = reg_mask[i].numpy() if reg_mask is not None \
                                else np.ones(NUM_REG_TARGETS)
                    else:
                        reg_t = np.full(NUM_REG_TARGETS, float("nan"))
                        reg_m = np.zeros(NUM_REG_TARGETS)

                    samples.append((feat.astype(np.float32), cls_lbl,
                                   reg_t.astype(np.float32),
                                   reg_m.astype(np.float32)))

        print(f"  ✅ M3: {len(samples)} samples loaded (Gaia+SDSS+ATNF+MWDD)")
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

    Each sample is a tuple: (inputs_dict, labels_dict, mask_tensor)

    inputs_dict:  modality tensors (zero-filled if absent)
    labels_dict:  per-head labels (IGNORE_LABEL if head invalid)
    mask_tensor:  (6,) binary — which encoders are active
    """
    def __init__(self, split: str = "train", verbose: bool = True):
        """
        Args:
            split: "train", "val", or "test"
            verbose: print loading progress
        """
        super().__init__()
        self.split = split
        self.samples = []       # list of (inputs, labels, mask) tuples
        self.source_counts = {} # track samples per module

        if verbose:
            print(f"\n{'═' * 50}")
            print(f"Loading Unified Dataset — split: {split}")
            print(f"{'═' * 50}")

        # Load all available modules
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
        m3_raw = _load_m3_data() if verbose else _load_m3_data()
        if m3_raw:
            m3_split = self._split_data(m3_raw, self.split)
            for feat, cls_lbl, reg_t, reg_m in m3_split:
                inputs = _empty_inputs()
                labels = _empty_labels()
                mask = _empty_mask()

                inputs["m3"] = torch.tensor(feat)
                labels["stellar_cls"] = cls_lbl
                labels["regression"] = torch.tensor(reg_t)
                labels["reg_mask"] = torch.tensor(reg_m)
                mask[MODALITY_INDEX["m3"]] = 1.0

                self.samples.append((inputs, labels, mask))
            self.source_counts["m3"] = len(m3_split)

        # ── M1-MLP (HTRU2) ──────────────────────────────────────────
        m1_mlp_raw = _load_m1_mlp_data()
        if m1_mlp_raw:
            m1_mlp_split = self._split_data(m1_mlp_raw, self.split)
            for feat, is_pulsar in m1_mlp_split:
                inputs = _empty_inputs()
                labels = _empty_labels()
                mask = _empty_mask()

                inputs["m1_mlp"] = torch.tensor(feat)
                labels["pulsar_det"] = int(is_pulsar)
                mask[MODALITY_INDEX["m1_mlp"]] = 1.0

                self.samples.append((inputs, labels, mask))
            self.source_counts["m1_mlp"] = len(m1_mlp_split)

        # ── M1-CNN + M1-AE (pulse profiles) ─────────────────────────
        m1_cnn_raw = _load_m1_cnn_data()
        if m1_cnn_raw:
            m1_cnn_split = self._split_data(m1_cnn_raw, self.split)
            for profile, fft_feat, time_feat, subtype in m1_cnn_split:
                inputs = _empty_inputs()
                labels = _empty_labels()
                mask = _empty_mask()

                inputs["m1_cnn_time"] = torch.tensor(time_feat)
                inputs["m1_cnn_freq"] = torch.tensor(fft_feat)
                inputs["m1_ae"] = torch.tensor(profile)
                labels["pulsar_subtype"] = subtype
                labels["pulsar_det"] = 1  # all profile samples are pulsars
                # Anomaly: subtype 0 = normal, 1+ = could be anomalous
                # For training, label all synthetic as non-anomalous (0)
                labels["anomaly"] = 0
                mask[MODALITY_INDEX["m1_cnn"]] = 1.0
                mask[MODALITY_INDEX["m1_ae"]] = 1.0

                self.samples.append((inputs, labels, mask))
            self.source_counts["m1_cnn_ae"] = len(m1_cnn_split)

        # ── M2-RGC (radio galaxies) ─────────────────────────────────
        m2_rgc_raw = _load_m2_rgc_data()
        if m2_rgc_raw:
            m2_rgc_split = self._split_data(m2_rgc_raw, self.split)
            for img, lbl in m2_rgc_split:
                inputs = _empty_inputs()
                labels = _empty_labels()
                mask = _empty_mask()

                inputs["m2_rgc"] = torch.tensor(img)
                labels["radio_morphology"] = lbl
                mask[MODALITY_INDEX["m2_rgc"]] = 1.0

                self.samples.append((inputs, labels, mask))
            self.source_counts["m2_rgc"] = len(m2_rgc_split)

        # ── M2-GWD (gravitational waves) ─────────────────────────────
        m2_gwd_raw = _load_m2_gwd_data()
        if m2_gwd_raw:
            m2_gwd_split = self._split_data(m2_gwd_raw, self.split)
            for cqt, lbl in m2_gwd_split:
                inputs = _empty_inputs()
                labels = _empty_labels()
                mask = _empty_mask()

                inputs["m2_gwd"] = torch.tensor(cqt)
                labels["gw_det"] = lbl
                mask[MODALITY_INDEX["m2_gwd"]] = 1.0

                self.samples.append((inputs, labels, mask))
            self.source_counts["m2_gwd"] = len(m2_gwd_split)

        # ── Cross-modal M1+M3 (neutron star augmentation) ───────────
        if m3_raw and m1_mlp_raw:
            self._create_cross_modal_ns(m3_raw, m1_mlp_raw, m1_cnn_raw)

    def _create_cross_modal_ns(self, m3_data, m1_mlp_data, m1_cnn_data):
        """
        Create cross-modal samples for neutron stars.

        M3 NS samples (class=3) get paired with M1 pulsar-positive
        HTRU2 features. This teaches the fusion MLP that m1_mlp + m3
        can co-activate for neutron star objects.

        The pairing is random (not sky-matched), but physically valid:
        both represent neutron star observations from different modalities.
        """
        # Filter M3 NS samples
        ns_idx = 3  # Neutron_Star class index
        m3_ns = [(f, c, r, m) for f, c, r, m in m3_data if c == ns_idx]

        # Filter M1-MLP pulsar-positive samples
        m1_pulsars = [(f, l) for f, l in m1_mlp_data if l == 1.0]

        if not m3_ns or not m1_pulsars:
            return

        # Split cross-modal samples using same seed
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

            inputs = _empty_inputs()
            labels = _empty_labels()
            mask = _empty_mask()

            # M3 features
            inputs["m3"] = torch.tensor(m3_feat)
            labels["stellar_cls"] = cls_lbl
            labels["regression"] = torch.tensor(reg_t)
            labels["reg_mask"] = torch.tensor(reg_m)
            mask[MODALITY_INDEX["m3"]] = 1.0

            # M1-MLP features (cross-modal)
            inputs["m1_mlp"] = torch.tensor(m1_feat)
            labels["pulsar_det"] = 1  # confirmed pulsar
            mask[MODALITY_INDEX["m1_mlp"]] = 1.0

            # Optionally add M1-CNN/AE if available
            if m1_cnn_data:
                cnn_idx = rng.randint(0, len(m1_cnn_data))
                profile, fft_feat, time_feat, subtype = m1_cnn_data[cnn_idx]
                inputs["m1_cnn_time"] = torch.tensor(time_feat)
                inputs["m1_cnn_freq"] = torch.tensor(fft_feat)
                inputs["m1_ae"] = torch.tensor(profile)
                labels["pulsar_subtype"] = subtype
                labels["anomaly"] = 0
                mask[MODALITY_INDEX["m1_cnn"]] = 1.0
                mask[MODALITY_INDEX["m1_ae"]] = 1.0

            self.samples.append((inputs, labels, mask))
            count += 1

        self.source_counts["cross_m1_m3"] = count
        print(f"  ✅ Cross-modal M1+M3: {count} NS samples created")

    def _split_data(self, data: list, split: str) -> list:
        """
        Deterministic train/val/test split.
        70% train, 15% val, 15% test.
        """
        n = len(data)
        if n < 10:
            return data if split == "train" else []

        test_ratio = TEST_SPLIT
        val_ratio = VAL_SPLIT / (1 - TEST_SPLIT)

        train_val, test = train_test_split(
            data, test_size=test_ratio, random_state=SEED
        )
        train, val = train_test_split(
            train_val, test_size=val_ratio, random_state=SEED
        )

        if split == "train":
            return train
        elif split == "val":
            return val
        elif split == "test":
            return test
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inputs, labels, mask = self.samples[idx]
        return inputs, labels, mask


# ═════════════════════════════════════════════════════════════
# 4. COLLATE FUNCTION
# ═════════════════════════════════════════════════════════════

def unified_collate_fn(batch):
    """
    Custom collate for UnifiedDataset.

    Stacks per-modality input tensors, per-head labels, and masks
    into batched tensors.

    Returns:
        inputs:  dict of (B, *) tensors
        labels:  dict of (B,) or (B, 4) tensors
        masks:   (B, 6) tensor
    """
    inputs_list, labels_list, masks_list = zip(*batch)
    B = len(batch)

    # ── Stack inputs ─────────────────────────────────────────────
    inputs = {}
    for key in INPUT_SHAPES.keys():
        inputs[key] = torch.stack([inp[key] for inp in inputs_list])

    # ── Stack labels ─────────────────────────────────────────────
    labels = {}

    # Integer labels (classification + binary heads)
    for key in ["stellar_cls", "pulsar_det", "pulsar_subtype",
                "radio_morphology", "gw_det", "anomaly"]:
        vals = [lbl[key] for lbl in labels_list]
        labels[key] = torch.tensor(vals, dtype=torch.long)

    # Regression targets + mask
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
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    verbose: bool = True,
) -> tuple:
    """
    Create train/val/test DataLoaders for unified training.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = UnifiedDataset(split="train", verbose=verbose)
    val_ds   = UnifiedDataset(split="val",   verbose=verbose and False)
    test_ds  = UnifiedDataset(split="test",  verbose=verbose and False)

    # ── Balanced sampling for training ───────────────────────────
    # Weight each sample inversely by its source module count
    if len(train_ds) > 0 and train_ds.source_counts:
        total = len(train_ds)
        weights = []
        idx = 0
        for src, cnt in sorted(train_ds.source_counts.items()):
            w = total / (cnt * len(train_ds.source_counts))
            for _ in range(cnt):
                weights.append(w)
            idx += cnt
        # Pad if mismatch
        while len(weights) < total:
            weights.append(1.0)
        weights = weights[:total]

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=total,
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

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

    # Build a small dataset (only loads available modules)
    ds = UnifiedDataset(split="train", verbose=True)

    if len(ds) == 0:
        print("\n⚠️  No data loaded — attach datasets and retry")
    else:
        # Check first sample
        inputs, labels, mask = ds[0]

        print("\n── Sample 0 ──")
        print("Inputs:")
        for k, v in inputs.items():
            print(f"  {k:<16s}: {list(v.shape)}")
        print("Labels:")
        for k, v in labels.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k:<16s}: {list(v.shape)}  val={v}")
            else:
                print(f"  {k:<16s}: {v}")
        print(f"Mask: {mask}")
        print(f"Active modalities: "
              f"{[MODALITY_ORDER[i] for i in range(NUM_ENCODERS) if mask[i] > 0]}")

        # Test collate
        print("\n── Collate Test ──")
        batch = [ds[i] for i in range(min(4, len(ds)))]
        inputs_b, labels_b, masks_b = unified_collate_fn(batch)
        print(f"Batch size: {masks_b.shape[0]}")
        for k, v in inputs_b.items():
            print(f"  inputs[{k:<16s}]: {list(v.shape)}")
        for k, v in labels_b.items():
            print(f"  labels[{k:<16s}]: {list(v.shape)}")
        print(f"  masks: {list(masks_b.shape)}")

    print("\n" + "=" * 60)
    print("✅ Dataset sanity check complete")
    print("=" * 60)