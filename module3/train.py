"""
module3/train.py
STELLARIS-DNet — Module 3 Training (v3)
FT-Transformer: 5-class classification + 4-param regression + physics loss

v3 upgrades:
  - GPU noise injection during training (breaks constant-fill shortcuts)
  - 7 features only (validity flags removed in dataset.py v3)
  - reg_mask retained for per-class regression supervision
  - Curriculum physics masking (true labels → soft probs)
  - Log-space physics loss (FP16 stable)

Run on Kaggle: import train; train.train(device, logger)
"""

import os
import sys
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module3.config  import *
from module3.dataset import load_stellar_data
from module3.model   import StellarFTTransformer
from core.utils      import (set_seed, get_device, get_logger,
                              save_checkpoint, save_encoder,
                              EarlyStopping, count_parameters, print_epoch)


# ─────────────────────────────────────────────
# 1. SCHEDULER — Linear Warmup + Cosine Decay
# ─────────────────────────────────────────────
def get_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
# 2. CURRICULUM SCHEDULE
# alpha = 0 → use ground truth labels (early)
# alpha = 1 → use predicted softmax (late)
# ─────────────────────────────────────────────
def curriculum_alpha(epoch: int) -> float:
    if epoch <= CURRICULUM_HARD_END:
        return 0.0
    if epoch >= CURRICULUM_SOFT_START:
        return 1.0
    span = CURRICULUM_SOFT_START - CURRICULUM_HARD_END
    return float(epoch - CURRICULUM_HARD_END) / float(span)


# ─────────────────────────────────────────────
# 3. NOISE INJECTION (v3 — GPU augmentation)
# Pre-built noise std vector for the 7 features after StandardScaler.
# After scaling, features are roughly N(0, 1).
# Noise applied here is in STANDARDISED space.
# Order: [teff, log_g, feh, abs_mag, bp_rp, redshift, period_ms]
# ─────────────────────────────────────────────
def build_noise_std(device: torch.device) -> torch.Tensor:
    """
    Returns (NUM_FEATURES,) tensor of noise stds in standardised space.
    Roughly std=0.05-0.10 in standardised units = small but meaningful jitter.
    """
    noise = torch.tensor([
        NOISE_TEFF_FRAC,        # teff (relative — but scaled is dimensionless)
        NOISE_LOGG_STD,         # log_g
        NOISE_FEH_STD,          # feh
        NOISE_ABSMAG_STD,       # abs_mag
        NOISE_BPRP_STD,         # bp_rp
        NOISE_REDSHIFT_STD,     # redshift
        NOISE_PERIODMS_FRAC,    # period_ms
    ], dtype=torch.float32, device=device)
    return noise


def apply_noise(X: torch.Tensor, noise_std: torch.Tensor) -> torch.Tensor:
    """
    Training-only feature augmentation:
    1. Gaussian noise on every feature (existing v3 behaviour)
    2. Sparse-feature dropout on redshift + period_ms (Path C — v4)
       Forces the model to NOT rely on the deterministic NS/QSO identifier.
    """
    # 1. Gaussian noise
    noise = torch.randn_like(X) * noise_std.unsqueeze(0)
    X = X + noise

    # 2. Sparse-feature dropout (zero with probability SPARSE_DROPOUT_PROB)
    if SPARSE_DROPOUT_PROB > 0:
        for idx in SPARSE_DROPOUT_IDX:
            keep = (torch.rand(X.size(0), device=X.device) > SPARSE_DROPOUT_PROB)
            X[:, idx] = X[:, idx] * keep.float()

    return X
# ─────────────────────────────────────────────
# 4. COMBINED LOSS (v3 — log-space physics, mask-aware reg)
# ─────────────────────────────────────────────
def compute_loss(
    model:        StellarFTTransformer,
    X:            torch.Tensor,
    y_class:      torch.Tensor,
    y_reg:        torch.Tensor,
    reg_mask:     torch.Tensor,
    cls_criterion: nn.CrossEntropyLoss,
    epoch:        int,
    device:       torch.device
) -> tuple[torch.Tensor, dict]:
    class_logits, reg_out, _ = model(X)

    # ── Classification loss ──
    cls_loss = cls_criterion(class_logits, y_class)        # (B,)

    # ── Regression loss — masked per-target ──
    reg_se = (reg_out - y_reg) ** 2                         # (B, 4)
    reg_se = reg_se * reg_mask                              # mask out unsupervised
    n_supervised   = reg_mask.sum().clamp(min=1.0)
    reg_loss_total = reg_se.sum() / n_supervised
    per_sample_sup = reg_mask.sum(dim=1).clamp(min=1.0)
    reg_loss_per   = (reg_se.sum(dim=1) / per_sample_sup)   # (B,)

    # ── Physics loss with curriculum + reg_mask gating ──
    alpha       = curriculum_alpha(epoch)
    probs_pred  = torch.softmax(class_logits, dim=1)
    probs_true  = nn.functional.one_hot(
        y_class, num_classes=NUM_STELLAR_CLASSES
    ).float()
    probs       = alpha * probs_pred + (1.0 - alpha) * probs_true

    # Log-scale predictions (no exp → FP16 stable)
    log_mass   = reg_out[:, 0]
    log_lum    = reg_out[:, 1]
    log_teff   = reg_out[:, 2]
    log_radius = reg_out[:, 3]

    # Class-probability gates — which physical law applies to this class
    sb_weight = probs[:, 0] + probs[:, 1] + probs[:, 2]   # MS + RG + WD
    ml_weight = probs[:, 0]                                # MS only
    ch_weight = probs[:, 2]                                # WD only

    # reg_mask gates — only enforce when the targets in question are supervised
    # REGRESSION_TARGETS = [log_mass, log_lum, log_teff, log_radius]  (idx 0,1,2,3)
    sb_valid = reg_mask[:, 1] * reg_mask[:, 2] * reg_mask[:, 3]   # needs L, T, R
    ml_valid = reg_mask[:, 0] * reg_mask[:, 1]                    # needs M, L
    ch_valid = reg_mask[:, 0]                                     # needs M

    # Stefan-Boltzmann (log space)
    log_L_expected = 2 * log_radius + 4 * (log_teff - LOG_TEFF_SUN)
    sb_diff        = (log_lum - log_L_expected.detach()) ** 2
    sb_loss        = sb_weight * sb_diff.clamp(max=100.0) * sb_valid

    # Mass-Luminosity (log space)
    log_L_ml_exp = 3.5 * log_mass.clamp(-1.0, 2.0)
    ml_diff      = (log_lum - log_L_ml_exp.detach()) ** 2
    ml_loss      = ml_weight * ml_diff.clamp(max=100.0) * ml_valid

    # Chandrasekhar (log space)
    log_chandra = math.log10(CHANDRASEKHAR_LIMIT)
    ch_loss     = ch_weight * nn.functional.relu(log_mass - log_chandra) * ch_valid

    physics_loss_per = sb_loss + ml_loss + ch_loss

    total_per = (
        CLASS_LOSS_WEIGHT   * cls_loss +
        REG_LOSS_WEIGHT     * reg_loss_per +
        PHYSICS_LOSS_WEIGHT * physics_loss_per
    )
    total_loss = total_per.mean()

    return total_loss, {
        "cls":     cls_loss.mean().item(),
        "reg":     reg_loss_total.item(),
        "physics": physics_loss_per.mean().item(),
        "alpha":   alpha,
    }


# ─────────────────────────────────────────────
# 5. EVAL (no noise — clean evaluation)
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, cls_criterion, epoch, device) -> dict:
    model.eval()
    total_loss = cls_t = reg_t = phy_t = 0.0
    correct = total = 0

    for X, y_class, y_reg, reg_mask in loader:
        X        = X.to(device)
        y_class  = y_class.to(device)
        y_reg    = y_reg.to(device)
        reg_mask = reg_mask.to(device)

        loss, parts = compute_loss(
            model, X, y_class, y_reg, reg_mask,
            cls_criterion, epoch, device
        )
        total_loss += loss.item()
        cls_t += parts["cls"]
        reg_t += parts["reg"]
        phy_t += parts["physics"]

        logits, _, _ = model(X)
        preds   = logits.argmax(dim=1)
        correct += (preds == y_class).sum().item()
        total   += y_class.size(0)

    n = len(loader)
    return {
        "loss": total_loss / n, "cls": cls_t / n,
        "reg": reg_t / n,       "physics": phy_t / n,
        "acc": correct / total,
    }


# ─────────────────────────────────────────────
# 6. TRAIN
# ─────────────────────────────────────────────
def train(device, logger):
    logger.info("=" * 55)
    logger.info("Module 3 Training v3 — StellarFTTransformer")
    logger.info("=" * 55)

    train_loader, val_loader, _, scaler, class_weights = load_stellar_data()

    model = StellarFTTransformer().to(device)
    count_parameters(model)

    cw_tensor      = torch.tensor(class_weights, dtype=torch.float32).to(device)
    cls_criterion  = nn.CrossEntropyLoss(weight=cw_tensor, reduction='none')
    optimizer      = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler      = get_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)
    amp_scaler = GradScaler('cuda', enabled=USE_AMP)
    early_stop     = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
    best_val_loss  = float("inf")

    # GPU noise std vector (built once)
    noise_std = build_noise_std(device)
    logger.info(f"GPU noise std: {noise_std.cpu().numpy().round(3)}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    logger.info(f"Device: {device} | AMP: {USE_AMP}")
    logger.info(f"LR: {LR} | Warmup: {WARMUP_EPOCHS} | Epochs: {EPOCHS}")
    logger.info(f"Effective batch: {ACTUAL_BATCH_SIZE * ACCUMULATE_STEPS}")
    logger.info(f"Curriculum: hard until ep {CURRICULUM_HARD_END}, "
                f"soft from ep {CURRICULUM_SOFT_START}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        train_loss = cls_sum = reg_sum = phy_sum = 0.0
        step_count = 0
        alpha_log  = 0.0

        for step, (X, y_class, y_reg, reg_mask) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)
        ):
            X        = X.to(device, non_blocking=True)
            y_class  = y_class.to(device, non_blocking=True)
            y_reg    = y_reg.to(device, non_blocking=True)
            reg_mask = reg_mask.to(device, non_blocking=True)

            # ── GPU noise injection (training only) ──
            X = apply_noise(X, noise_std)

            with autocast('cuda', enabled=USE_AMP):
                loss, parts = compute_loss(
                    model, X, y_class, y_reg, reg_mask,
                    cls_criterion, epoch, device
                )
                loss = loss / ACCUMULATE_STEPS

            amp_scaler.scale(loss).backward()

            if (step + 1) % ACCUMULATE_STEPS == 0:
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * ACCUMULATE_STEPS
            cls_sum    += parts["cls"]
            reg_sum    += parts["reg"]
            phy_sum    += parts["physics"]
            alpha_log   = parts["alpha"]
            step_count += 1

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

        val_metrics = evaluate(model, val_loader, cls_criterion, epoch, device)
        model.train()

        print_epoch(
            epoch, EPOCHS,
            train_loss, val_metrics["loss"],
            val_acc=val_metrics["acc"],
            extra={"LR": current_lr, "α": alpha_log}
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
            f"lr={current_lr:.2e} | α={alpha_log:.2f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                model, optimizer, epoch,
                val_metrics["loss"],
                CHECKPOINT_DIR, "module3_best.pt"
            )
            save_encoder(model, CHECKPOINT_DIR, "module3_encoder.pt")

            with open(os.path.join(CHECKPOINT_DIR, "module3_scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)

            np.save(os.path.join(CHECKPOINT_DIR, "feature_scales.npy"),
                    model.feature_scale.detach().cpu().numpy())

            logger.info(f"  ✅ Best — val_loss={best_val_loss:.4f}")

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

    logger.info("STELLARIS-DNet | Module 3 Training v3 Started")
    model = train(device, logger)

    print("\n" + "=" * 55)
    print("Module 3 Training v3 Complete")
    print(f"Checkpoint: {CHECKPOINT_DIR}/module3_best.pt")
    print(f"Encoder:    {CHECKPOINT_DIR}/module3_encoder.pt")
    print(f"Scaler:     {CHECKPOINT_DIR}/module3_scaler.pkl")
    print("Run evaluate.py next.")
    print("=" * 55)