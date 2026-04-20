"""
module2/train.py
STELLARIS-DNet — Module 2 Training
2A: RadioGalaxyClassifier (EfficientNet-B0) on MiraBest
2B: GravWaveDetector (1D CNN) on G2Net
Run: python module2/train.py
"""

import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module2.config     import *
from module2.dataset_2a import load_mirabest
from module2.dataset_2b import load_g2net
from module2.model      import RadioGalaxyClassifier, GravWaveDetector
from core.utils         import (set_seed, get_device, get_logger,
                                 save_checkpoint, save_encoder,
                                 EarlyStopping, count_parameters,
                                 print_epoch)


# ─────────────────────────────────────────────
# 1. TRAIN RADIO GALAXY CLASSIFIER (2A)
# Staged: freeze backbone → train head → unfreeze
# ─────────────────────────────────────────────
def train_radio_galaxy(device, logger):
    logger.info("=" * 50)
    logger.info("Training 2A — Radio Galaxy Classifier (EfficientNet-B0)")
    logger.info("=" * 50)

    train_loader, val_loader, _ = load_mirabest()
    model = RadioGalaxyClassifier(pretrained=True).to(device)
    count_parameters(model)

    head_params     = list(model.encoder.parameters()) + \
                      list(model.head.parameters())
    backbone_params = list(model.backbone.parameters())

    optimizer = torch.optim.AdamW([
        {"params": head_params,     "lr": RGZ_LR},
        {"params": backbone_params, "lr": RGZ_LR_BACKBONE}
    ], weight_decay=RGZ_WEIGHT_DECAY)

    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=RGZ_EPOCHS
    )
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
    early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
    best_val_loss = float("inf")
    best_val_acc  = 0.0

    for epoch in range(1, RGZ_EPOCHS + 1):
        if epoch == RGZ_FREEZE_EPOCHS + 1:
            model.unfreeze_last_blocks(n_blocks=2)
            logger.info(f"Epoch {epoch}: Unfreezing last 2 backbone blocks")

        # Train
        model.train()
        train_loss = 0.0
        for X, y in tqdm(train_loader, desc=f"RGC {epoch}", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y  = X.to(device), y.to(device)
                out   = model(X)
                val_loss += criterion(out, y).item()
                correct  += (out.argmax(1) == y).sum().item()
                total    += y.size(0)
        val_loss /= len(val_loader)
        val_acc   = correct / total

        scheduler.step()
        print_epoch(epoch, RGZ_EPOCHS, train_loss, val_loss, val_acc)
        logger.info(f"Epoch {epoch} | train={train_loss:.4f} "
                    f"val={val_loss:.4f} acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss,
                            CHECKPOINT_DIR, "rgc_best.pt")
            save_encoder(model.encoder, CHECKPOINT_DIR, "rgc_encoder.pt")

        if early_stop(val_loss):
            break

    logger.info(f"2A done. Best val loss: {best_val_loss:.4f} "
                f"acc: {best_val_acc:.4f}")
    return model


# ─────────────────────────────────────────────
# 2. TRAIN GRAVITATIONAL WAVE DETECTOR (2B)
# Fixed: ReduceLROnPlateau instead of OneCycleLR
# Fixed: lower LR (1e-4), class weights for balance
# ─────────────────────────────────────────────
def train_grav_wave(device, logger):
    logger.info("=" * 50)
    logger.info("Training 2B — Gravitational Wave Detector (1D CNN)")
    logger.info("=" * 50)

    train_loader, val_loader, _ = load_g2net()
    model = GravWaveDetector().to(device)
    count_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LIGO_LR,                 # 1e-4 — fixed from 1e-3
        weight_decay=LIGO_WEIGHT_DECAY
    )

    # ReduceLROnPlateau — stable, won't explode like OneCycleLR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5,
    factor=0.5, min_lr=1e-6
    )

    # Balanced class weights — Signal and Noise should be equal
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    early_stop    = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

    best_val_loss = float("inf")
    best_val_acc  = 0.0

    for epoch in range(1, LIGO_EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X, y in tqdm(train_loader, desc=f"GWD {epoch}", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = correct = total = 0
        pred_counts = {0: 0, 1: 0}
        with torch.no_grad():
            for X, y in val_loader:
                X, y  = X.to(device), y.to(device)
                out   = model(X)
                val_loss += criterion(out, y).item()
                preds = out.argmax(1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
                for p in preds.cpu().tolist():
                    pred_counts[p] = pred_counts.get(p, 0) + 1
        val_loss /= len(val_loader)
        val_acc   = correct / total

        scheduler.step(val_loss)
        print_epoch(epoch, LIGO_EPOCHS, train_loss, val_loss, val_acc,
                    extra={"N_pred": pred_counts.get(0, 0),
                           "S_pred": pred_counts.get(1, 0)})
        logger.info(f"Epoch {epoch} | train={train_loss:.4f} "
                    f"val={val_loss:.4f} acc={val_acc:.4f} "
                    f"preds={pred_counts}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss,
                            CHECKPOINT_DIR, "gwd_best.pt")
            save_encoder(model.conv_encoder, CHECKPOINT_DIR,
                         "gwd_encoder.pt")

        if early_stop(val_loss):
            break

    logger.info(f"2B done. Best val loss: {best_val_loss:.4f} "
                f"acc: {best_val_acc:.4f}")
    return model


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    set_seed(SEED)
    device = get_device()
    logger = get_logger("module2", LOG_DIR)

    logger.info("STELLARIS-DNet | Module 2 Training Started")
    logger.info(f"Device: {device}")

    rgc_model = train_radio_galaxy(device, logger)
    gwd_model = train_grav_wave(device, logger)

    logger.info("=" * 50)
    logger.info("Module 2 Training Complete")
    logger.info(f"Checkpoints: {CHECKPOINT_DIR}")
    logger.info("=" * 50)
    print("\n✅ Module 2 training complete. Run module2/evaluate.py next.")