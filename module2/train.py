"""
module2/train.py
STELLARIS-DNet — Module 2 Training (Upgraded)
Physics losses + warmup scheduler added to original _train_loop structure.

Usage:
  python module2/train.py                   # trains both 2A and 2B
  python module2/train.py --task 2a         # only 2A
  python module2/train.py --task 2b         # only 2B
  python module2/train.py --use_cqt False   # 2B with raw signal
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module2.config     import *
from module2.dataset_2a import load_mirabest
from module2.dataset_2b import load_g2net
from module2.model      import (RadioGalaxyClassifier,
                                 GravWaveDetector, GravWave1DCNN)
from core.utils         import (set_seed, get_device, get_logger,
                                 save_checkpoint, save_encoder,
                                 EarlyStopping, count_parameters, print_epoch)


# ─────────────────────────────────────────────
# OUTPUT UNPACKING
# RadioGalaxyClassifier returns (logits, aux)
# GravWaveDetector returns logits
# _train_loop handles both transparently
# ─────────────────────────────────────────────
def _unpack(output):
    """Returns (logits, aux_or_None) regardless of model type."""
    if isinstance(output, tuple):
        return output[0], output[1]
    return output, None


# ─────────────────────────────────────────────
# PHYSICS LOSSES
# ─────────────────────────────────────────────
def jet_power_physics_loss(
    jet_power:  torch.Tensor,   # (B, 1) — predicted log10(P / W·Hz⁻¹)
    class_pred: torch.Tensor    # (B,) — argmax class prediction
) -> torch.Tensor:
    """
    Enforces the Fanaroff-Riley luminosity boundary:
      FRI  → log10(jet_power) < 25.0  (< 10^25 W/Hz)
      FRII → log10(jet_power) > 25.0  (> 10^25 W/Hz)

    Uses ReLU to penalize ONLY violations, not compliant predictions.
    Weight: JET_POWER_LOSS_WEIGHT (default 0.05)
    """
    if jet_power is None:
        return torch.tensor(0.0, device=class_pred.device)

    pred     = jet_power.squeeze(1)                 # (B,)
    boundary = FRI_FRII_BOUNDARY_LOG                # 25.0

    fri_mask  = (class_pred == 0)
    frii_mask = (class_pred == 1)
    loss      = torch.tensor(0.0, device=pred.device)

    if fri_mask.sum() > 0:
        # FRI predicted above boundary → penalize
        loss = loss + F.relu(pred[fri_mask] - boundary).mean()

    if frii_mask.sum() > 0:
        # FRII predicted below boundary → penalize
        loss = loss + F.relu(boundary - pred[frii_mask]).mean()

    return loss


def chirp_slope_loss(
    cqt:    torch.Tensor,   # (B, 3, H, W) — CQT spectrograms
    logits: torch.Tensor    # (B, 2) — class logits
) -> torch.Tensor:
    """
    Chirp slope consistency for predicted Signal samples.
    GW chirp: f(t) ∝ (t_c − t)^(−3/8) → rising frequency centroid.

    For each sample predicted as Signal (P(Signal) > 0.5):
    - Compute frequency centroid per time bin from CQT
    - Penalize if centroid DECREASES over time (non-physical)

    Weight: CHIRP_LOSS_WEIGHT (default 0.1)
    """
    probs       = F.softmax(logits, dim=1)[:, 1]   # P(Signal) — (B,)
    signal_mask = (probs > 0.5)

    if signal_mask.sum() == 0:
        return torch.tensor(0.0, device=cqt.device)

    # Average over detector channels → (N_sig, H, W)
    cqt_sig = cqt[signal_mask].mean(dim=1)
    H       = cqt_sig.shape[1]

    # Frequency bin indices (0 = low freq, H-1 = high freq)
    freq_bins = torch.arange(H, dtype=torch.float32, device=cqt.device)

    # Softmax weights over frequency axis → (N_sig, H, W)
    weights   = F.softmax(cqt_sig, dim=1)

    # Weighted frequency centroid per time step → (N_sig, W)
    centroids = (weights * freq_bins.view(1, -1, 1)).sum(dim=1)

    # First difference of centroids → (N_sig, W-1)
    diff      = centroids[:, 1:] - centroids[:, :-1]

    # Penalize negative differences (centroid dropping → non-physical)
    return F.relu(-diff).mean()


# ─────────────────────────────────────────────
# SCHEDULER WITH WARMUP
# ─────────────────────────────────────────────
def _make_scheduler(optimizer, total_epochs: int,
                    warmup_epochs: int, lr_min: float):
    """
    Linear warmup → CosineAnnealingLR.
    Warmup: critical for Transformer stability.
    """
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=lr_min
    )
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs]
        )
    return cosine


# ─────────────────────────────────────────────
# SHARED TRAINING LOOP
# ─────────────────────────────────────────────
def _train_loop(model, train_loader, val_loader,
                epochs, freeze_epochs, lr, lr_backbone,
                weight_decay, warmup_epochs, lr_min,
                ckpt_dir, ckpt_name, enc_name,
                device, logger, tag,
                physics_loss_fn=None):
    """
    Shared training loop for 2A and 2B models.

    Key upgrades over original:
    - Differential LR: non-backbone params use lr, backbone uses lr_backbone
    - Warmup + CosineAnnealing scheduler
    - Physics loss support via physics_loss_fn callback
    - Dynamic output unpacking: (logits, aux) or plain logits

    physics_loss_fn signature:
        fn(logits, aux, X) → scalar tensor
    """
    count_parameters(model)
    has_backbone = hasattr(model, "backbone")

    # ── Differential LR Parameter Groups ──────
    if has_backbone:
        backbone_params = set(model.backbone.parameters())
        other_params    = [p for p in model.parameters()
                           if p not in backbone_params]
        optimizer = torch.optim.AdamW([
            {"params": other_params,    "lr": lr},
            {"params": list(backbone_params), "lr": lr_backbone}
        ], weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    scheduler  = _make_scheduler(optimizer, epochs, warmup_epochs, lr_min)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
    early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

    best_val_loss = float("inf")
    best_val_acc  = 0.0

    for epoch in range(1, epochs + 1):
        # Progressive backbone unfreezing
        if has_backbone and epoch == freeze_epochs + 1:
            model.unfreeze_last_blocks(n=2)
            logger.info(f"{tag} Epoch {epoch}: Backbone last 2 blocks unfrozen")

        # ── Train ────────────────────────────
        model.train()
        train_loss = 0.0

        for X, y in tqdm(train_loader, desc=f"{tag} {epoch}/{epochs}", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            output          = model(X)
            logits, aux     = _unpack(output)
            cls_loss        = criterion(logits, y)

            # Physics loss (optional, task-specific)
            phys_loss = torch.tensor(0.0, device=device)
            if USE_PHYSICS_LOSS and physics_loss_fn is not None:
                phys_loss = physics_loss_fn(logits, aux, X)

            loss = cls_loss + phys_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ─────────────────────────
        model.eval()
        val_loss = 0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y        = X.to(device), y.to(device)
                logits, _   = _unpack(model(X))
                val_loss   += criterion(logits, y).item()
                correct    += (logits.argmax(1) == y).sum().item()
                total      += y.size(0)

        val_loss /= len(val_loader)
        val_acc   = correct / total

        scheduler.step()
        print_epoch(epoch, epochs, train_loss, val_loss, val_acc)
        logger.info(f"{tag} [{epoch}/{epochs}] "
                    f"train={train_loss:.4f} val={val_loss:.4f} "
                    f"acc={val_acc:.4f}")

        # ── Checkpoint ───────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss,
                            ckpt_dir, ckpt_name)
            if has_backbone:
                save_encoder(model.encoder, ckpt_dir, enc_name)
            else:
                save_encoder(model.encoder, ckpt_dir, enc_name)

        if early_stop(val_loss):
            break

    logger.info(f"{tag} done | best_loss={best_val_loss:.4f} "
                f"best_acc={best_val_acc:.4f}")
    return model, best_val_acc


# ─────────────────────────────────────────────
# PHYSICS LOSS CALLBACKS
# ─────────────────────────────────────────────
def _physics_loss_2a(logits, aux, X):
    """2A: FRI/FRII jet power boundary constraint."""
    class_pred = logits.argmax(dim=1)
    return JET_POWER_LOSS_WEIGHT * jet_power_physics_loss(aux, class_pred)


def _physics_loss_2b(logits, aux, X):
    """2B: GW chirp slope consistency constraint."""
    if not USE_CHIRP_LOSS:
        return torch.tensor(0.0, device=logits.device)
    return CHIRP_LOSS_WEIGHT * chirp_slope_loss(X, logits)


# ─────────────────────────────────────────────
# TASK TRAINERS
# ─────────────────────────────────────────────
def train_radio_galaxy(device, logger):
    logger.info("=" * 50)
    logger.info("2A — RadioGalaxyClassifier (EfficientNet-B2 + CBAM + GeM)")
    logger.info("=" * 50)

    train_loader, val_loader, _ = load_mirabest()
    model = RadioGalaxyClassifier(pretrained=True).to(device)

    return _train_loop(
        model, train_loader, val_loader,
        epochs=RGZ_EPOCHS,
        freeze_epochs=RGZ_FREEZE_EPOCHS,
        lr=RGZ_LR,
        lr_backbone=RGZ_LR_BACKBONE,
        weight_decay=RGZ_WEIGHT_DECAY,
        warmup_epochs=RGZ_WARMUP_EPOCHS,
        lr_min=RGZ_LR_MIN,
        ckpt_dir=CHECKPOINT_DIR,
        ckpt_name="rgc_best.pt",
        enc_name="rgc_encoder.pt",
        device=device,
        logger=logger,
        tag="2A-RGC",
        physics_loss_fn=_physics_loss_2a if USE_PHYSICS_LOSS else None
    )


def train_grav_wave(device, logger, use_cqt: bool = True):
    mode = "CQT+EfficientNet-B2" if use_cqt else "Raw+1DCNN"
    logger.info("=" * 50)
    logger.info(f"2B — GW Detector ({mode})")
    logger.info("=" * 50)

    train_loader, val_loader, _ = load_g2net(use_cqt=use_cqt)

    if use_cqt:
        model     = GravWaveDetector(pretrained=True).to(device)
        ckpt_name = "gwd_best.pt"
        enc_name  = "gwd_encoder.pt"
        lr        = LIGO_LR
        lr_bb     = LIGO_LR_BACKBONE
        wd        = LIGO_WEIGHT_DECAY
        ep        = LIGO_EPOCHS
        freeze_ep = LIGO_FREEZE_EPOCHS
        warmup_ep = LIGO_WARMUP_EPOCHS
        lr_min    = LIGO_LR_MIN
        phys_fn   = _physics_loss_2b if USE_PHYSICS_LOSS else None
    else:
        model     = GravWave1DCNN().to(device)
        ckpt_name = "gwd_raw_best.pt"
        enc_name  = "gwd_raw_encoder.pt"
        lr        = 1e-3
        lr_bb     = 1e-3
        wd        = 1e-4
        ep        = LIGO_EPOCHS
        freeze_ep = 0
        warmup_ep = 0
        lr_min    = 1e-6
        phys_fn   = None

    return _train_loop(
        model, train_loader, val_loader,
        epochs=ep,
        freeze_epochs=freeze_ep,
        lr=lr,
        lr_backbone=lr_bb,
        weight_decay=wd,
        warmup_epochs=warmup_ep,
        lr_min=lr_min,
        ckpt_dir=CHECKPOINT_DIR,
        ckpt_name=ckpt_name,
        enc_name=enc_name,
        device=device,
        logger=logger,
        tag=f"2B-{'CQT' if use_cqt else 'RAW'}",
        physics_loss_fn=phys_fn
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",    default="both",
                        choices=["both", "2a", "2b"])
    parser.add_argument("--use_cqt", default="True",
                        type=lambda x: x.lower() != "false")
    args = parser.parse_args()

    set_seed(SEED)
    device = get_device()
    logger = get_logger("module2", LOG_DIR)
    logger.info(f"Module 2 | task={args.task} | use_cqt={args.use_cqt}")
    logger.info(f"Upgrades: EfficientNet-B2 | CBAM | GeM | "
                f"Transformer={'2B only'} | Physics={'ON' if USE_PHYSICS_LOSS else 'OFF'}")

    if args.task in ("both", "2a"):
        train_radio_galaxy(device, logger)

    if args.task in ("both", "2b"):
        train_grav_wave(device, logger, use_cqt=args.use_cqt)

    print("\n✅ Module 2 training complete. Run module2/evaluate.py next.")