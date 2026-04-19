"""
module2/dataset_2a.py
STELLARIS-DNet — Module 2A Data Pipeline
MiraBest Radio Galaxy Classification
Classes: FRI (0), FRII (1)
"""

import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module2.config import *


# ─────────────────────────────────────────────
# 1. INSPECT BATCH — find exact keys and labels
# ─────────────────────────────────────────────
def inspect_batch(path: str):
    """Prints all keys and label info from a batch file."""
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    print(f"\nKeys in batch: {list(d.keys())}")
    for k, v in d.items():
        if isinstance(v, (list, np.ndarray)):
            arr = np.array(v)
            print(f"  {k}: shape={arr.shape}, "
                  f"unique={np.unique(arr) if arr.ndim==1 else 'N-D array'}")
        else:
            print(f"  {k}: {v}")


# ─────────────────────────────────────────────
# 2. LOAD SINGLE BATCH — robust key detection
# ─────────────────────────────────────────────
def _load_batch(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")

    # Try all possible data keys
    data = None
    for key in [b"data", "data"]:
        if key in d:
            data = np.array(d[key])
            break

    # Try all possible label keys
    labels = None
    for key in [b"labels", "labels", b"fine_labels", "fine_labels",
                b"coarse_labels", "coarse_labels"]:
        if key in d:
            labels = np.array(d[key])
            break

    if data is None or labels is None:
        raise KeyError(f"Cannot find data/labels in {path}. Keys: {list(d.keys())}")

    return data, labels


# ─────────────────────────────────────────────
# 3. LOAD ALL MIRABEST BATCHES
# ─────────────────────────────────────────────
def _load_mirabest(data_dir: str):
    """
    Loads all MiraBest CIFAR batches.
    Handles all label encoding variants automatically.
    """
    all_images, all_labels = [], []

    # Load train batches
    for i in range(1, 9):
        path = os.path.join(data_dir, f"data_batch_{i}")
        if not os.path.exists(path):
            continue
        imgs, lbls = _load_batch(path)
        all_images.append(imgs)
        all_labels.append(lbls)

    # Load test batch
    test_path = os.path.join(data_dir, "test_batch")
    if os.path.exists(test_path):
        imgs, lbls = _load_batch(test_path)
        all_images.append(imgs)
        all_labels.append(lbls)

    if not all_images:
        raise FileNotFoundError(
            f"No batch files found in {data_dir}\n"
            f"Expected: data_batch_1 ... data_batch_8, test_batch"
        )

    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"Raw labels: {np.unique(labels, return_counts=True)}")

    # ── Detect label encoding ──────────────────
    unique = np.unique(labels)

    if set(unique).issubset({0, 1}):
        # Already binary: 0=FRI, 1=FRII
        print("Label encoding: binary (0=FRI, 1=FRII)")
        keep = np.ones(len(labels), dtype=bool)

    elif set(unique).issubset({0, 1, 2, 3, 4}):
        # MiraBest 5-class encoding
        # 0,1 = FRI (confident, uncertain)
        # 2,3 = FRII (confident, uncertain)
        # 4   = Hybrid → discard
        print("Label encoding: 5-class MiraBest")
        label_map = {0: 0, 1: 0, 2: 1, 3: 1, 4: -1}
        labels    = np.array([label_map[l] for l in labels])
        keep      = labels >= 0
        labels    = labels[keep]
        images    = images[keep]

    elif set(unique).issubset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}):
        # MiraBest 10-class encoding
        # 0-4 = FRI variants, 5-9 = FRII variants
        print("Label encoding: 10-class MiraBest")
        labels = (labels >= 5).astype(int)  # 0=FRI, 1=FRII
        keep   = np.ones(len(labels), dtype=bool)

    else:
        # Unknown — treat as binary by median split
        print(f"Unknown encoding {unique} — using median split")
        median = np.median(labels)
        labels = (labels > median).astype(int)
        keep   = np.ones(len(labels), dtype=bool)

    images = images[keep] if keep.sum() < len(images) else images

    # ── Reshape images ─────────────────────────
    n = len(images)
    try:
        # Try 3-channel 150x150
        images = images.reshape(n, 3, 150, 150).astype(np.float32)
    except ValueError:
        try:
            # Try 1-channel 150x150
            images = images.reshape(n, 1, 150, 150)
            images = np.repeat(images, 3, axis=1).astype(np.float32)
        except ValueError:
            # Try square root for image size
            total  = images.shape[1]
            side   = int(np.sqrt(total // 3))
            images = images.reshape(n, 3, side, side).astype(np.float32)

    # Normalize to [0, 1]
    images = images / 255.0

    print(f"✅ MiraBest loaded: {n} samples")
    print(f"   FRI:  {(labels == 0).sum()}")
    print(f"   FRII: {(labels == 1).sum()}")
    print(f"   Image shape: {images.shape}")

    return images, labels.astype(np.int64)


# ─────────────────────────────────────────────
# 4. TRANSFORMS — manual (no PIL dependency)
# ─────────────────────────────────────────────
def augment(img: torch.Tensor) -> torch.Tensor:
    """
    Simple augmentation for radio galaxy images.
    Radio galaxies have no preferred orientation.
    """
    # Random horizontal flip
    if torch.rand(1) > 0.5:
        img = torch.flip(img, dims=[2])
    # Random vertical flip
    if torch.rand(1) > 0.5:
        img = torch.flip(img, dims=[1])
    # Random 90/180/270 rotation
    k = torch.randint(0, 4, (1,)).item()
    img = torch.rot90(img, k, dims=[1, 2])
    return img


def normalize_imagenet(img: torch.Tensor) -> torch.Tensor:
    """ImageNet normalization for EfficientNet-B0 transfer learning."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img - mean) / std


def resize_img(img: torch.Tensor, size: int = RGZ_IMG_SIZE) -> torch.Tensor:
    """Resize using interpolation."""
    return torch.nn.functional.interpolate(
        img.unsqueeze(0), size=(size, size),
        mode="bilinear", align_corners=False
    ).squeeze(0)


# ─────────────────────────────────────────────
# 5. DATASET CLASS
# ─────────────────────────────────────────────
class MiraBestDataset(Dataset):
    def __init__(
        self,
        images:  np.ndarray,
        labels:  np.ndarray,
        train:   bool = False,
        img_size: int = RGZ_IMG_SIZE
    ):
        self.images   = torch.tensor(images,  dtype=torch.float32)
        self.labels   = torch.tensor(labels,  dtype=torch.long)
        self.train    = train
        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].clone()          # (3, H, W)
        img = resize_img(img, self.img_size)    # → (3, 224, 224)
        if self.train and RGZ_AUGMENT:
            img = augment(img)
        img = normalize_imagenet(img)
        return img, self.labels[idx]


# ─────────────────────────────────────────────
# 6. DATALOADER FACTORY
# ─────────────────────────────────────────────
def load_mirabest(data_dir: str = RGZ_DATA_DIR):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"MiraBest not found at: {data_dir}\n"
            f"Download from: https://zenodo.org/record/4288837"
        )

    images, labels = _load_mirabest(data_dir)
    n = len(images)

    # Shuffle
    np.random.seed(SEED)
    idx     = np.random.permutation(n)
    n_test  = max(1, int(n * RGZ_TEST_SPLIT))
    n_val   = max(1, int(n * RGZ_VAL_SPLIT))
    n_train = n - n_test - n_val

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    train_ds = MiraBestDataset(images[train_idx], labels[train_idx], train=True)
    val_ds   = MiraBestDataset(images[val_idx],   labels[val_idx],   train=False)
    test_ds  = MiraBestDataset(images[test_idx],  labels[test_idx],  train=False)

    train_loader = DataLoader(train_ds, batch_size=RGZ_BATCH_SIZE,
                              shuffle=True,  drop_last=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=RGZ_BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=RGZ_BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"   Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Module 2A Dataset Sanity Check")
    print("=" * 50)

    # First inspect one batch to understand structure
    batch1 = os.path.join(RGZ_DATA_DIR, "data_batch_1")
    if os.path.exists(batch1):
        print("\n── Batch inspection ──")
        inspect_batch(batch1)

    print("\n── Loading dataset ──")
    try:
        tr, vl, te = load_mirabest()
        X, y = next(iter(tr))
        print(f"\nBatch shape : {X.shape}")
        print(f"Label shape : {y.shape}")
        print(f"Classes     : {[RGZ_CLASSES[i] for i in y[:8].tolist()]}")
        print(f"Label dist  : FRI={( y==0).sum()} FRII={(y==1).sum()}")
        assert X.shape[1:] == (3, RGZ_IMG_SIZE, RGZ_IMG_SIZE), \
            f"Wrong shape: {X.shape}"
        assert set(y.tolist()).issubset({0, 1}), \
            f"Unexpected labels: {y.unique()}"
        print("\n✅ dataset_2a.py OK")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()