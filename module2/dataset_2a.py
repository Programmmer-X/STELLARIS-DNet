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
from module2.config import (
    SEED, RGZ_DATA_DIR, RGZ_CLASSES, RGZ_NUM_CLASSES,
    RGZ_IMG_SIZE, RGZ_AUGMENT, RGZ_BATCH_SIZE,
    RGZ_TEST_SPLIT, RGZ_VAL_SPLIT
)


# ─────────────────────────────────────────────
# PATH HELPER — auto-detect Kaggle vs Local
# ─────────────────────────────────────────────
def _resolve_data_dir(data_dir: str) -> str:
    """Returns correct path for Kaggle or local environment."""
    # Kaggle: repo cloned to /kaggle/working/STELLARIS-DNet
    kaggle_path = f"/kaggle/working/STELLARIS-DNet/{data_dir}"
    if os.path.exists("/kaggle/input") and os.path.exists(kaggle_path):
        return kaggle_path
    # Local
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
            f"Expected: data_batch_1 ... data_batch_8, test_batch"
        )

    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"Raw labels: {np.unique(labels, return_counts=True)}")

    unique = np.unique(labels)

    if set(unique).issubset({0, 1}):
        print("Label encoding: binary (0=FRI, 1=FRII)")

    elif set(unique).issubset({0, 1, 2, 3, 4}):
        print("Label encoding: 5-class MiraBest")
        label_map = {0: 0, 1: 0, 2: 1, 3: 1, 4: -1}
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
            total = images.shape[1]
            side  = int(np.sqrt(total // 3))
            images = images.reshape(n, 3, side, side).astype(np.float32)

    images = images / 255.0

    print(f"✅ MiraBest loaded: {n} samples")
    print(f"   FRI:  {(labels == 0).sum()}")
    print(f"   FRII: {(labels == 1).sum()}")
    print(f"   Image shape: {images.shape}")

    return images, labels.astype(np.int64)


# ─────────────────────────────────────────────
# 3. TRANSFORMS
# ─────────────────────────────────────────────
def _augment(img: torch.Tensor) -> torch.Tensor:
    if torch.rand(1) > 0.5:
        img = torch.flip(img, dims=[2])
    if torch.rand(1) > 0.5:
        img = torch.flip(img, dims=[1])
    k = torch.randint(0, 4, (1,)).item()
    img = torch.rot90(img, k, dims=[1, 2])
    return img


def _normalize(img: torch.Tensor) -> torch.Tensor:
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
    if data_dir is None:
        data_dir = _resolve_data_dir(RGZ_DATA_DIR)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"MiraBest not found at: {data_dir}\n"
            f"Download from: https://zenodo.org/record/4288837"
        )

    images, labels = _load_mirabest(data_dir)
    n = len(images)

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
                              shuffle=True, drop_last=True, num_workers=0)
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
    try:
        tr, vl, te = load_mirabest()
        X, y = next(iter(tr))
        print(f"Batch shape : {X.shape}")
        print(f"Label shape : {y.shape}")
        print(f"Classes     : {[RGZ_CLASSES[i] for i in y[:8].tolist()]}")
        assert X.shape[1:] == (3, RGZ_IMG_SIZE, RGZ_IMG_SIZE)
        assert set(y.tolist()).issubset({0, 1})
        print("\n✅ dataset_2a.py OK")
    except Exception as e:
        import traceback
        traceback.print_exc()