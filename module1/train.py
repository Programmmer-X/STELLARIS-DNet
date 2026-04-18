"""
module1/train.py
STELLARIS-DNet — Module 1 Training
Trains: MLP (HTRU2) + 1D CNN (pulse profiles) + Autoencoder (magnetar anomaly)
Run: python module1/train.py
"""

import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module1.config  import *
from module1.dataset import load_htru2, load_pulse_profiles, load_autoencoder_data
from module1.model   import PulsarMLP, PulsarCNN, PulsarAutoencoder
from core.utils      import (set_seed, get_device, get_logger,
                              save_checkpoint, save_encoder,
                              EarlyStopping, count_parameters, print_epoch)


# ─────────────────────────────────────────────
# 1. TRAIN MLP (HTRU2 Binary Classifier)
# ─────────────────────────────────────────────
def train_mlp(device, logger):
    logger.info("=" * 50)
    logger.info("Training MLP — HTRU2 Binary Classifier")
    logger.info("=" * 50)

    # Data
    train_loader, val_loader, _, scaler, pos_weight = load_htru2()
    pos_weight = pos_weight.to(device)

    # Model
    model     = PulsarMLP().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    criterion    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    early_stop   = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
    count_parameters(model)

    best_val_loss = float("inf")

    for epoch in range(1, MLP_EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for X, y in tqdm(train_loader, desc=f"MLP Epoch {epoch}", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(X).squeeze(1)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y  = X.to(device), y.to(device)
                out   = model(X).squeeze(1)
                loss  = criterion(out, y)
                val_loss += loss.item()
                preds    = (torch.sigmoid(out) > 0.5).float()
                correct  += (preds == y).sum().item()
                total    += y.size(0)
        val_loss /= len(val_loader)
        val_acc   = correct / total

        scheduler.step(val_loss)
        print_epoch(epoch, MLP_EPOCHS, train_loss, val_loss, val_acc)
        logger.info(f"Epoch {epoch} | train={train_loss:.4f} "
                    f"val={val_loss:.4f} acc={val_acc:.4f}")

        # ── Checkpoint best ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            CHECKPOINT_DIR, "mlp_best.pt")
            save_encoder(model.encoder, CHECKPOINT_DIR, "mlp_encoder.pt")

        if early_stop(val_loss):
            break

    logger.info(f"MLP training done. Best val loss: {best_val_loss:.4f}")
    return model


# ─────────────────────────────────────────────
# 2. TRAIN 1D CNN (Pulsar Subtype Classifier)
# ─────────────────────────────────────────────
def train_cnn(device, logger):
    logger.info("=" * 50)
    logger.info("Training 1D CNN — Pulsar Subtype Classifier")
    logger.info("=" * 50)

    # Data
    train_loader, val_loader, _ = load_pulse_profiles()

    # Model
    model     = PulsarCNN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CNN_LR, weight_decay=CNN_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CNN_EPOCHS
    )
    criterion  = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
    count_parameters(model)

    best_val_loss = float("inf")

    for epoch in range(1, CNN_EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for X, y in tqdm(train_loader, desc=f"CNN Epoch {epoch}", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y  = X.to(device), y.to(device)
                out   = model(X)
                loss  = criterion(out, y)
                val_loss += loss.item()
                preds    = out.argmax(dim=1)
                correct  += (preds == y).sum().item()
                total    += y.size(0)
        val_loss /= len(val_loader)
        val_acc   = correct / total

        scheduler.step()
        print_epoch(epoch, CNN_EPOCHS, train_loss, val_loss, val_acc)
        logger.info(f"Epoch {epoch} | train={train_loss:.4f} "
                    f"val={val_loss:.4f} acc={val_acc:.4f}")

        # ── Checkpoint best ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            CHECKPOINT_DIR, "cnn_best.pt")
            save_encoder(model.conv_encoder, CHECKPOINT_DIR, "cnn_encoder.pt")

        if early_stop(val_loss):
            break

    logger.info(f"CNN training done. Best val loss: {best_val_loss:.4f}")
    return model


# ─────────────────────────────────────────────
# 3. TRAIN AUTOENCODER (Magnetar Anomaly)
# ─────────────────────────────────────────────
def train_autoencoder(device, logger):
    logger.info("=" * 50)
    logger.info("Training Autoencoder — Magnetar Anomaly Detector")
    logger.info("=" * 50)

    # Data — normal pulsars only
    train_loader, val_loader = load_autoencoder_data()

    # Model
    model     = PulsarAutoencoder().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=AE_LR
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    criterion  = nn.MSELoss()
    early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
    count_parameters(model)

    best_val_loss = float("inf")

    for epoch in range(1, AE_EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for X in tqdm(train_loader, desc=f"AE Epoch {epoch}", leave=False):
            X = X.to(device)
            optimizer.zero_grad()
            recon = model(X)
            loss  = criterion(recon, X)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X in val_loader:
                X     = X.to(device)
                recon = model(X)
                loss  = criterion(recon, X)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        print_epoch(epoch, AE_EPOCHS, train_loss, val_loss)
        logger.info(f"Epoch {epoch} | train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            CHECKPOINT_DIR, "ae_best.pt")
            save_encoder(model.encoder, CHECKPOINT_DIR, "ae_encoder.pt")

        if early_stop(val_loss):
            break

    logger.info(f"AE training done. Best val loss: {best_val_loss:.6f}")
    return model


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    set_seed(SEED)
    device = get_device()
    logger = get_logger("module1", LOG_DIR)

    logger.info("STELLARIS-DNet | Module 1 Training Started")
    logger.info(f"Device: {device}")

    # Train all 3 models in sequence
    mlp_model = train_mlp(device, logger)
    cnn_model = train_cnn(device, logger)
    ae_model  = train_autoencoder(device, logger)

    logger.info("=" * 50)
    logger.info("Module 1 Training Complete")
    logger.info(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    logger.info("=" * 50)
    print("\n✅ Module 1 training complete. Run module1/evaluate.py next.")