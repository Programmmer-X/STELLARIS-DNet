"""
core/utils.py
STELLARIS-DNet — Shared Utilities
Checkpointing, logging, seeding, metrics helpers.
Used across all modules.
"""

import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# 1. REPRODUCIBILITY
# Call this at the top of every train.py
# ─────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────
# 2. DEVICE SELECTION
# ─────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU found — running on CPU")
    return device


# ─────────────────────────────────────────────
# 3. LOGGING SETUP
# ─────────────────────────────────────────────
def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# ─────────────────────────────────────────────
# 4. CHECKPOINTING
# ─────────────────────────────────────────────
def save_checkpoint(
    model:      nn.Module,
    optimizer:  torch.optim.Optimizer,
    epoch:      int,
    loss:       float,
    save_dir:   str,
    filename:   str = "checkpoint.pt"
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss":                 loss,
    }, path)
    print(f"💾 Checkpoint saved → {path}")


def load_checkpoint(
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    path:      str,
    device:    torch.device
):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"✅ Checkpoint loaded ← {path}")
    return checkpoint["epoch"], checkpoint["loss"]


# ─────────────────────────────────────────────
# 5. SAVE ENCODER ONLY
# Critical for unification — saves just the
# encoder weights, discards classification head
# ─────────────────────────────────────────────
def save_encoder(
    encoder:  nn.Module,
    save_dir: str,
    filename: str = "encoder.pt"
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(encoder.state_dict(), path)
    print(f"🔗 Encoder saved → {path}")


def load_encoder(
    encoder:  nn.Module,
    path:     str,
    device:   torch.device,
    freeze:   bool = True
) -> nn.Module:
    encoder.load_state_dict(torch.load(path, map_location=device))
    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False
        print(f"🔒 Encoder loaded + frozen ← {path}")
    else:
        print(f"🔓 Encoder loaded + unfrozen ← {path}")
    return encoder


# ─────────────────────────────────────────────
# 6. EARLY STOPPING
# ─────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = float("inf")
        self.stop       = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"⛔ Early stopping triggered at patience={self.patience}")
                self.stop = True
        return self.stop


# ─────────────────────────────────────────────
# 7. MODEL PARAMETER COUNT
# ─────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total params:     {total:,}")
    print(f"📊 Trainable params: {trainable:,}")
    return trainable


# ─────────────────────────────────────────────
# 8. TRAINING PROGRESS PRINTER
# ─────────────────────────────────────────────
def print_epoch(
    epoch:      int,
    total:      int,
    train_loss: float,
    val_loss:   float,
    val_acc:    float = None,
    extra:      dict  = None
):
    msg = (f"Epoch [{epoch:03d}/{total}] "
           f"Train Loss: {train_loss:.4f} | "
           f"Val Loss: {val_loss:.4f}")
    if val_acc is not None:
        msg += f" | Val Acc: {val_acc:.4f}"
    if extra:
        for k, v in extra.items():
            msg += f" | {k}: {v:.4f}"
    print(msg)


# ─────────────────────────────────────────────
# 9. REQUIREMENTS FREEZE HELPER
# Run once to save exact package versions
# ─────────────────────────────────────────────
def save_requirements(path: str = "requirements.txt"):
    import subprocess
    result = subprocess.run(
        ["pip", "freeze"],
        capture_output=True, text=True
    )
    with open(path, "w") as f:
        f.write(result.stdout)
    print(f"📦 Requirements saved → {path}")


if __name__ == "__main__":
    print("Testing utils...")

    set_seed(42)
    device = get_device()

    print("Device:", device)