"""
module2/train.py
STELLARIS-DNet — Module 2 Training
2A: RadioGalaxyClassifier on MiraBest images
2B: GravWaveDetector on CQT spectrograms
Both use EfficientNet-B0 — same staged training strategy
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
# SHARED TRAINING LOOP
# Both 2A and 2B use identical training strategy
# Only data loaders and model differ
# ─────────────────────────────────────────────
def _train_efficientnet(
    model, train_loader, val_loader,
    epochs, freeze_epochs, lr, lr_backbone,
    weight_decay, checkpoint_dir,
    checkpoint_name, encoder_name,
    device, logger, task_name
):
    count_parameters(model)

    head_params     = list(model.encoder.parameters()) + \
                      list(model.head.parameters())
    backbone_params = list(model.backbone.parameters())

    optimizer = torch.optim.AdamW([
        {"params": head_params,     "lr": lr},
        {"params": backbone_params, "lr": lr_backbone}
    ], weight_decay=weight_decay)

    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
    early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

    best_val_loss = float("inf")
    best_val_acc  = 0.0

    for epoch in range(1, epochs + 1):

        # Unfreeze after freeze_epochs
        if epoch == freeze_epochs + 1:
            model.unfreeze_last_blocks(n=2)
            logger.info(f"{task_name} Epoch {epoch}: Unfreeze last 2 blocks")

        # ── Train ──
        model.train()
        train_loss = 0.0
        for X, y in tqdm(train_loader, desc=f"{task_name} {epoch}", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ──
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
        print_epoch(epoch, epochs, train_loss, val_loss, val_acc)
        logger.info(f"{task_name} Epoch {epoch} | "
                    f"train={train_loss:.4f} val={val_loss:.4f} "
                    f"acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss,
                            checkpoint_dir, checkpoint_name)
            save_encoder(model.encoder, checkpoint_dir, encoder_name)

        if early_stop(val_loss):
            break

    logger.info(f"{task_name} done. "
                f"Best val loss: {best_val_loss:.4f} "
                f"acc: {best_val_acc:.4f}")
    return model


# ─────────────────────────────────────────────
# 2A: RADIO GALAXY
# ─────────────────────────────────────────────
def train_radio_galaxy(device, logger):
    logger.info("=" * 50)
    logger.info("Training 2A — Radio Galaxy (EfficientNet-B0)")
    logger.info("=" * 50)

    train_loader, val_loader, _ = load_mirabest()
    model = RadioGalaxyClassifier(pretrained=True).to(device)

    return _train_efficientnet(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        epochs       = RGZ_EPOCHS,
        freeze_epochs= RGZ_FREEZE_EPOCHS,
        lr           = RGZ_LR,
        lr_backbone  = RGZ_LR_BACKBONE,
        weight_decay = RGZ_WEIGHT_DECAY,
        checkpoint_dir  = CHECKPOINT_DIR,
        checkpoint_name = "rgc_best.pt",
        encoder_name    = "rgc_encoder.pt",
        device       = device,
        logger       = logger,
        task_name    = "2A-RGC"
    )


# ─────────────────────────────────────────────
# 2B: GRAVITATIONAL WAVE
# ─────────────────────────────────────────────
def train_grav_wave(device, logger):
    logger.info("=" * 50)
    logger.info("Training 2B — GW Detector (EfficientNet-B0 + CQT)")
    logger.info("=" * 50)

    train_loader, val_loader, _ = load_g2net()
    model = GravWaveDetector(pretrained=True).to(device)

    return _train_efficientnet(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        epochs       = LIGO_EPOCHS,
        freeze_epochs= LIGO_FREEZE_EPOCHS,
        lr           = LIGO_LR,
        lr_backbone  = LIGO_LR_BACKBONE,
        weight_decay = LIGO_WEIGHT_DECAY,
        checkpoint_dir  = CHECKPOINT_DIR,
        checkpoint_name = "gwd_best.pt",
        encoder_name    = "gwd_encoder.pt",
        device       = device,
        logger       = logger,
        task_name    = "2B-GWD"
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    set_seed(SEED)
    device = get_device()
    logger = get_logger("module2", LOG_DIR)

    logger.info("STELLARIS-DNet | Module 2 Training Started")
    logger.info(f"Device: {device}")

    train_radio_galaxy(device, logger)
    train_grav_wave(device, logger)

    logger.info("Module 2 Training Complete")
    print("\n✅ Module 2 complete. Run module2/evaluate.py next.")