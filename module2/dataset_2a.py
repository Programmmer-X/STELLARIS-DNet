"""
module2/dataset_2a.py
STELLARIS-DNet — Module 2A Data Pipeline
MiraBest Radio Galaxy Classification: FRI vs FRII

Kept from original: direct pickle loading, multi-format label handling
Upgraded:           percentile normalization for morphological clarity
"""

import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module2.config import (
    SEED, RGZ_DATA_DIR, RGZ_CLASSES, RGZ_NUM_CLASSES,
    RGZ_IMG_SIZE, RGZ_AUGMENT, RGZ_BATCH_SIZE,
    RGZ_TEST_SPLIT, RGZ_VAL_SPLIT
)


# ─────────────────────────────────────────────
# PATH HELPER
# ─────────────────────────────────────────────
def _resolve_data_dir(data_dir: str) -> str:
    kaggle = f"/kaggle/working/STELLARIS-DNet/{data_dir}"
    if os.path.exists("/kaggle/input") and os.path.exists(kaggle):
        return kaggle
    return data_dir


# ─────────────────────────────────────────────
# 1. LOAD SINGLE BATCH
# ─────────────────────────────────────────────
def _load_batch(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")

    data = None
    for key in [b"data", "data"]:
        if key in d:
            data = np.array(d[key])
            break

    labels = None
    for key in [b"labels", "labels", b"fine_labels",
                "fine_labels", b"coarse_labels", "coarse_labels"]:
        if key in d:
            labels = np.array(d[key])
            break

    if data is None or labels is None:
        raise KeyError(f"Cannot find data/labels in {path}. "
                       f"Keys: {list(d.keys())}")
    return data, labels


# ─────────────────────────────────────────────
# 2. LOAD ALL MIRABEST BATCHES
# Handles all known MiraBest label encodings:
#   binary (0=FRI, 1=FRII)
#   5-class  (confident+uncertain FRI/FRII + hybrid)
#   10-class (median split)
# ─────────────────────────────────────────────
def _load_mirabest(data_dir: str):
    all_images, all_labels = [], []

    for i in range(1, 9):
        path = os.path.join(data_dir, f"data_batch_{i}")
        if not os.path.exists(path):
            continue
        imgs, lbls = _load_batch(path)
        all_images.append(imgs)
        all_labels.append(lbls)

    test_path = os.path.join(data_dir, "test_batch")
    if os.path.exists(test_path):
        imgs, lbls = _load_batch(test_path)
        all_images.append(imgs)
        all_labels.append(lbls)

    if not all_images:
        raise FileNotFoundError(
            f"No batch files found in {data_dir}\n"
            f"Expected: data_batch_1 ... data_batch_8, test_batch\n"
            f"Download: https://zenodo.org/record/4288837"
        )

    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"Raw labels: {np.unique(labels, return_counts=True)}")
    unique = np.unique(labels)

    if set(unique).issubset({0, 1}):
        print("Label encoding: binary (0=FRI, 1=FRII)")

    elif set(unique).issubset({0, 1, 2, 3, 4}):
        print("Label encoding: 5-class MiraBest")
        label_map = {0: 0, 1: 0, 2: 1, 3: 1, 4: -1}  # 4=Hybrid → discard
        labels    = np.array([label_map[int(l)] for l in labels])
        keep      = labels >= 0
        images    = images[keep]
        labels    = labels[keep]

    elif set(unique).issubset(set(range(10))):
        print("Label encoding: 10-class MiraBest")
        labels = (labels >= 5).astype(int)

    else:
        print(f"Unknown encoding {unique} — median split")
        labels = (labels > np.median(labels)).astype(int)

    n = len(images)
    try:
        images = images.reshape(n, 3, 150, 150).astype(np.float32)
    except ValueError:
        try:
            images = images.reshape(n, 1, 150, 150)
            images = np.repeat(images, 3, axis=1).astype(np.float32)
        except ValueError:
            total  = images.shape[1]
            side   = int(np.sqrt(total // 3))
            images = images.reshape(n, 3, side, side).astype(np.float32)

    images = images / 255.0

    # ── UPGRADE: Percentile normalization for morphological clarity ──
    # Clips each image to [p1, p99] and rescales to [0, 1]
    # Makes FRI/FRII jet structures significantly more visible
    images = _percentile_normalize_batch(images)

    print(f"✅ MiraBest loaded: {n} samples")
    print(f"   FRI:  {(labels == 0).sum()}")
    print(f"   FRII: {(labels == 1).sum()}")
    print(f"   Image shape: {images.shape}")

    return images, labels.astype(np.int64)


def _percentile_normalize_batch(images: np.ndarray,
                                 lo: float = 1.0,
                                 hi: float = 99.0) -> np.ndarray:
    """
    Per-image percentile normalization.
    Clips to [p1, p99] then rescales to [0, 1].
    Applied after /255 normalization.
    Significantly improves jet morphology visibility.
    """
    result = np.zeros_like(images)
    for i in range(len(images)):
        img  = images[i]                  # (3, H, W)
        p_lo = np.percentile(img, lo)
        p_hi = np.percentile(img, hi)
        if p_hi > p_lo:
            result[i] = np.clip((img - p_lo) / (p_hi - p_lo), 0.0, 1.0)
        else:
            result[i] = img
    return result.astype(np.float32)


# ─────────────────────────────────────────────
# 3. TRANSFORMS
# ─────────────────────────────────────────────
def _augment(img: torch.Tensor) -> torch.Tensor:
    """
    Physics-valid augmentations for radio galaxy images.
    Flip + rotation: radio galaxies have no preferred orientation.
    """
    if torch.rand(1) > 0.5:
        img = torch.flip(img, dims=[2])         # horizontal flip
    if torch.rand(1) > 0.5:
        img = torch.flip(img, dims=[1])         # vertical flip
    k = torch.randint(0, 4, (1,)).item()
    img = torch.rot90(img, k, dims=[1, 2])      # 90° rotations
    return img


def _normalize(img: torch.Tensor) -> torch.Tensor:
    """ImageNet statistics normalization (for pretrained backbone)."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img - mean) / std


def _resize(img: torch.Tensor, size: int = RGZ_IMG_SIZE) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        img.unsqueeze(0), size=(size, size),
        mode="bilinear", align_corners=False
    ).squeeze(0)


# ─────────────────────────────────────────────
# 4. DATASET CLASS
# ─────────────────────────────────────────────
class MiraBestDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 train: bool = False, img_size: int = RGZ_IMG_SIZE):
        self.images   = torch.tensor(images, dtype=torch.float32)
        self.labels   = torch.tensor(labels, dtype=torch.long)
        self.train    = train
        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].clone()
        img = _resize(img, self.img_size)
        if self.train and RGZ_AUGMENT:
            img = _augment(img)
        img = _normalize(img)
        return img, self.labels[idx]


# ─────────────────────────────────────────────
# 5. DATALOADER FACTORY
# ─────────────────────────────────────────────
def load_mirabest(data_dir: str = None):
    """
    Loads MiraBest dataset with stratified train/val/test split.
    Returns: train_loader, val_loader, test_loader
    """
    if data_dir is None:
        data_dir = _resolve_data_dir(RGZ_DATA_DIR)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"MiraBest not found at: {data_dir}\n"
            f"Download from: https://zenodo.org/record/4288837"
        )

    images, labels = _load_mirabest(data_dir)
    n = len(images)

    # Stratified split — preserves FRI/FRII ratio
    from sklearn.model_selection import train_test_split
    idx = np.arange(n)

    idx_tr, idx_te, y_tr, y_te = train_test_split(
        idx, labels, test_size=RGZ_TEST_SPLIT,
        random_state=SEED, stratify=labels
    )
    idx_tr, idx_vl, y_tr, y_vl = train_test_split(
        idx_tr, y_tr,
        test_size=RGZ_VAL_SPLIT / (1 - RGZ_TEST_SPLIT),
        random_state=SEED, stratify=y_tr
    )

    train_ds = MiraBestDataset(images[idx_tr], y_tr, train=True)
    val_ds   = MiraBestDataset(images[idx_vl], y_vl, train=False)
    test_ds  = MiraBestDataset(images[idx_te], y_te, train=False)

    train_loader = DataLoader(train_ds, batch_size=RGZ_BATCH_SIZE,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=RGZ_BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=RGZ_BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    _print_split_balance("Train", y_tr)
    _print_split_balance("Val",   y_vl)
    return train_loader, val_loader, test_loader


def _print_split_balance(split: str, labels: np.ndarray):
    n_fri  = (labels == 0).sum()
    n_frii = (labels == 1).sum()
    total  = len(labels)
    print(f"   {split}: FRI={n_fri} ({n_fri/total:.1%}) | "
          f"FRII={n_frii} ({n_frii/total:.1%})")


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Module 2A Dataset Sanity Check")
    print("=" * 50)
    try:
        tr, vl, te = load_mirabest()
        X, y = next(iter(tr))
        print(f"Batch shape : {X.shape}")
        print(f"Label shape : {y.shape}")
        print(f"Classes     : {[RGZ_CLASSES[i] for i in y[:8].tolist()]}")
        print(f"Pixel range : [{X.min():.3f}, {X.max():.3f}]")
        assert X.shape[1:] == (3, RGZ_IMG_SIZE, RGZ_IMG_SIZE)
        assert set(y.tolist()).issubset({0, 1})
        print("\n✅ dataset_2a.py OK")
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
    except Exception:
        import traceback
        traceback.print_exc()