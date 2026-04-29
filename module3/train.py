"""
module3/train.py
STELLARIS-DNet — Module 3 Training
FT-Transformer: 5-class classification + 4-param regression + physics loss
Run on Kaggle: python module3/train.py
"""

import os
import sys
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module3.config  import *
from module3.dataset import load_stellar_data
from module3.model   import StellarFTTransformer
from core.utils      import (set_seed, get_device, get_logger,
                              save_checkpoint, save_encoder,
                              EarlyStopping, count_parameters, print_epoch)
from core.physics_loss import stellar_physics_loss


# ─────────────────────────────────────────────
# 1. SCHEDULER — Linear Warmup + Cosine Decay
# ─────────────────────────────────────────────
def get_scheduler(
    optimizer:      torch.optim.Optimizer,
    warmup_epochs:  int,
    total_epochs:   int
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup for warmup_epochs, then cosine decay to ~0.
    More stable than ReduceLROnPlateau for transformers.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
# 2. COMBINED LOSS
# ─────────────────────────────────────────────
def compute_loss(
    model:        StellarFTTransformer,
    X:            torch.Tensor,
    y_class:      torch.Tensor,
    y_reg:        torch.Tensor,
    is_synthetic: torch.Tensor,
    cls_criterion: nn.CrossEntropyLoss,
    device:        torch.device
) -> tuple[torch.Tensor, dict]:
    """
    Combined loss = CLASS + REG + PHYSICS

    Soft physics masking:
        probs[:,0] * MS_loss + probs[:,2] * WD_loss
        (Chandrasekhar), probs[:,0] * ML_loss (mass-luminosity).
        SB law applied to classes 0,1,2 only (not NS, not QSO).

    Synthetic down-weighting:
        All loss terms multiplied by 0.7 for synthetic samples.

    Returns total_loss (scalar) and dict of component losses for logging.
    """
    class_logits, reg_out, _ = model(X)

    # ── Classification loss ───────────────────
    # CrossEntropyLoss with class weights, reduction='none' for per-sample weighting
    cls_loss = cls_criterion(class_logits, y_class)   # (B,)

    # ── Regression loss ───────────────────────
    reg_loss = nn.functional.mse_loss(
        reg_out, y_reg, reduction='none'
    ).mean(dim=1)                                      # (B,)

    # ── Soft physics masking ──────────────────
    probs = torch.softmax(class_logits, dim=1)         # (B, 5)

    # Extract predicted log-scale params
    log_mass   = reg_out[:, 0]   # log10(M/M_sun)
    log_lum    = reg_out[:, 1]   # log10(L/L_sun)
    log_teff   = reg_out[:, 2]   # log10(Teff/K)
    log_radius = reg_out[:, 3]   # log10(R/R_sun)

    # Convert to linear for physics loss functions
    # Clamp to avoid exploding gradients
    L_pred = 10 ** log_lum.clamp(LOG_LUM_MIN,    LOG_LUM_MAX)
    R_pred = 10 ** log_radius.clamp(LOG_RADIUS_MIN, LOG_RADIUS_MAX)
    T_pred = 10 ** log_teff.clamp(LOG_TEFF_MIN,  LOG_TEFF_MAX)
    M_pred = 10 ** log_mass.clamp(LOG_MASS_MIN,  LOG_MASS_MAX)

    # Class probability masks for each constraint
    # SB loss: MS(0) + RG(1) + WD(2) — NOT QSO or NS
    sb_weight = probs[:, 0] + probs[:, 1] + probs[:, 2]   # (B,)
    # ML loss: MS(0) only
    ml_weight = probs[:, 0]                                 # (B,)
    # Chandrasekhar: WD(2) only
    ch_weight = probs[:, 2]                                 # (B,)

    # Stefan-Boltzmann: L = 4π R² σ T⁴
    # Expected L in solar units from R (solar) and T (K)
    R_si       = R_pred * 6.957e8
    L_expected = (4 * math.pi * R_si**2 * SIGMA_SB * T_pred**4) / L_SUN
    sb_loss_per = nn.functional.mse_loss(
        L_pred, L_expected.detach(), reduction='none'
    ).clamp(max=1e6) / (L_SUN ** 2 + 1e-10)
    sb_loss_per = sb_weight * sb_loss_per.clamp(max=10.0)  # (B,)

    # Mass-Luminosity: L ∝ M^3.5 (MS only)
    M_clamped  = M_pred.clamp(0.1, 100)
    L_ml_exp   = M_clamped ** 3.5
    ml_loss_per = ml_weight * nn.functional.mse_loss(
        L_pred, L_ml_exp.detach(), reduction='none'
    ).clamp(max=1e6) / (L_SUN + 1e-10)                    # (B,)

    # Chandrasekhar: M < 1.44 M_sun for WD
    ch_loss_per = ch_weight * nn.functional.relu(
        M_pred - CHANDRASEKHAR_LIMIT
    )                                                      # (B,)

    physics_loss = (sb_loss_per + ml_loss_per + ch_loss_per)  # (B,)

    # ── Per-sample total loss ─────────────────
    total_per = (
        CLASS_LOSS_WEIGHT   * cls_loss    +
        REG_LOSS_WEIGHT     * reg_loss    +
        PHYSICS_LOSS_WEIGHT * physics_loss
    )                                                      # (B,)

    # ── Synthetic down-weighting ──────────────
    weights = torch.where(
        is_synthetic,
        torch.full_like(total_per, SYNTHETIC_LOSS_WEIGHT),
        torch.ones_like(total_per)
    )
    total_loss = (total_per * weights).mean()

    return total_loss, {
        "cls":     cls_loss.mean().item(),
        "reg":     reg_loss.mean().item(),
        "physics": physics_loss.mean().item(),
    }


# ─────────────────────────────────────────────
# 3. EVAL PASS
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model:        StellarFTTransformer,
    loader:       torch.utils.data.DataLoader,
    cls_criterion: nn.CrossEntropyLoss,
    device:        torch.device
) -> dict:
    model.eval()
    total_loss = cls_t = reg_t = phy_t = 0.0
    correct = total = 0

    for X, y_class, y_reg, is_syn in loader:
        X, y_class, y_reg, is_syn = (
            X.to(device), y_class.to(device),
            y_reg.to(device), is_syn.to(device)
        )
        loss, parts = compute_loss(
            model, X, y_class, y_reg, is_syn, cls_criterion, device
        )
        total_loss += loss.item()
        cls_t += parts["cls"]
        reg_t += parts["reg"]
        phy_t += parts["physics"]

        logits, _, _ = model(X)
        preds    = logits.argmax(dim=1)
        correct += (preds == y_class).sum().item()
        total   += y_class.size(0)

    n = len(loader)
    return {
        "loss":    total_loss / n,
        "cls":     cls_t / n,
        "reg":     reg_t / n,
        "physics": phy_t / n,
        "acc":     correct / total,
    }


# ─────────────────────────────────────────────
# 4. TRAIN
# ─────────────────────────────────────────────
def train(device, logger):
    logger.info("=" * 55)
    logger.info("Module 3 Training — StellarFTTransformer")
    logger.info("=" * 55)

    # ── Data ──────────────────────────────────
    train_loader, val_loader, _, scaler, class_weights = load_stellar_data()

    # ── Model ─────────────────────────────────
    model = StellarFTTransformer().to(device)
    count_parameters(model)

    # ── Loss ──────────────────────────────────
    cw_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    cls_criterion = nn.CrossEntropyLoss(
        weight=cw_tensor, reduction='none'
    )

    # ── Optimiser ─────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    # ── Scheduler: warmup + cosine ────────────
    scheduler = get_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)

    # ── AMP scaler ────────────────────────────
    amp_scaler = GradScaler(enabled=USE_AMP)

    # ── Early stopping ────────────────────────
    early_stop    = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
    best_val_loss = float("inf")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    logger.info(f"Device: {device} | AMP: {USE_AMP}")
    logger.info(f"LR: {LR} | Warmup: {WARMUP_EPOCHS} | Epochs: {EPOCHS}")
    logger.info(f"Effective batch: {ACTUAL_BATCH_SIZE * ACCUMULATE_STEPS}")

    # ─────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        train_loss = cls_sum = reg_sum = phy_sum = 0.0
        step_count = 0

        for step, (X, y_class, y_reg, is_syn) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)
        ):
            X        = X.to(device)
            y_class  = y_class.to(device)
            y_reg    = y_reg.to(device)
            is_syn   = is_syn.to(device)

            # ── Forward + loss ────────────────
            with autocast(enabled=USE_AMP):
                loss, parts = compute_loss(
                    model, X, y_class, y_reg,
                    is_syn, cls_criterion, device
                )
                loss = loss / ACCUMULATE_STEPS

            amp_scaler.scale(loss).backward()

            # ── Gradient accumulation step ────
            if (step + 1) % ACCUMULATE_STEPS == 0:
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=GRAD_CLIP
                )
                amp_scaler.step(optimizer)
                amp_scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * ACCUMULATE_STEPS
            cls_sum    += parts["cls"]
            reg_sum    += parts["reg"]
            phy_sum    += parts["physics"]
            step_count += 1

        # Handle leftover steps not divisible by ACCUMULATE_STEPS
        if step_count % ACCUMULATE_STEPS != 0:
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            optimizer.zero_grad()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        train_loss /= step_count
        cls_sum    /= step_count
        reg_sum    /= step_count
        phy_sum    /= step_count

        # ── Validation ────────────────────────
        val_metrics = evaluate(model, val_loader, cls_criterion, device)
        model.train()

        # ── Logging ───────────────────────────
        print_epoch(
            epoch, EPOCHS,
            train_loss, val_metrics["loss"],
            val_acc=val_metrics["acc"],
            extra={"LR": current_lr}
        )
        logger.info(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.4f} "
            f"(cls={cls_sum:.4f} reg={reg_sum:.4f} phy={phy_sum:.4f}) | "
            f"val={val_metrics['loss']:.4f} "
            f"(cls={val_metrics['cls']:.4f} "
            f"reg={val_metrics['reg']:.4f} "
            f"phy={val_metrics['physics']:.4f}) | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        # ── Checkpoint best model ──────────────
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                model, optimizer, epoch,
                val_metrics["loss"],
                CHECKPOINT_DIR, "module3_best.pt"
            )
            # Encoder only — for unified model
            save_encoder(
                model,          # full model: encode() is the encoder interface
                CHECKPOINT_DIR, "module3_encoder.pt"
            )
            # Save scaler for inference
            scaler_path = os.path.join(CHECKPOINT_DIR, "module3_scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

            # Save learned feature scales (useful for analysis)
            scale_path = os.path.join(CHECKPOINT_DIR, "feature_scales.npy")
            np.save(scale_path,
                    model.feature_scale.detach().cpu().numpy())

            logger.info(
                f"  ✅ Best model saved — val_loss={best_val_loss:.4f}"
            )

        if early_stop(val_metrics["loss"]):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return model


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    set_seed(SEED)
    device = get_device()
    logger = get_logger("module3_train", LOG_DIR)

    logger.info("STELLARIS-DNet | Module 3 Training Started")

    model = train(device, logger)

    print("\n" + "=" * 55)
    print("Module 3 Training Complete")
    print(f"Checkpoints: {CHECKPOINT_DIR}/module3_best.pt")
    print(f"Encoder:     {CHECKPOINT_DIR}/module3_encoder.pt")
    print(f"Scaler:      {CHECKPOINT_DIR}/module3_scaler.pkl")
    print("Run evaluate.py next.")
    print("=" * 55)