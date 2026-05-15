"""
unified/train.py
STELLARIS-DNet — Unified Fusion Training

Staged training:
  Stage 1: All encoders frozen, train fusion + heads only
  Stage 2: Unfreeze M1-MLP + M1-AE, lower LR
  Stage 3: Conditional, data-driven (extend unfreeze list)

Per-head masked losses: each head's loss computed ONLY on samples
where that head's labels are valid (modality present + label exists).

Physics consistency loss with curriculum warmup.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified.config import (
    SEED, CHECKPOINT_DIR, LOG_DIR, STAGES, LOSS_WEIGHTS,
    MODALITY_INDEX, NUM_ENCODERS, FUSED_DIM,
    MAX_GRAD_NORM, PATIENCE, MIN_DELTA, EXPERIMENT_TAG,
    PHYSICS_LOSS_START_EPOCH, PHYSICS_LOSS_RAMP_EPOCHS,
    PHYSICS_LOSS_TARGET_WEIGHT, NUM_STELLAR_CLASSES,
    REG_SUPERVISION, STELLAR_CLASS_NAMES,
)
from unified.fusion_model import UnifiedModel
from unified.dataset import (
    get_unified_loaders, IGNORE_LABEL,
)
from core.utils import set_seed, get_device, get_logger, EarlyStopping


# ═════════════════════════════════════════════════════════════
# 1. LOSS COMPUTATION
# ═════════════════════════════════════════════════════════════

def compute_losses(outputs: dict, labels: dict, masks: torch.Tensor,
                   epoch: int, device: torch.device) -> dict:
    """
    Compute per-head losses with masking.

    Only samples where the head is valid contribute to that head's loss.
    Invalid samples (label == IGNORE_LABEL) are excluded.

    Returns:
        dict of {head_name: scalar loss tensor}
        All losses are mean-reduced over valid samples only.
    """
    losses = {}

    # ── Stellar Classification (CE) ──────────────────────────────
    valid = labels["stellar_cls"] != IGNORE_LABEL
    if valid.any():
        pred = outputs["stellar_cls"][valid]
        tgt  = labels["stellar_cls"][valid].to(device)
        losses["stellar_cls"] = F.cross_entropy(pred, tgt)

    # ── Pulsar Detection (BCE) ───────────────────────────────────
    valid = labels["pulsar_det"] != IGNORE_LABEL
    if valid.any():
        pred = outputs["pulsar_det"][valid].squeeze(-1)
        tgt  = labels["pulsar_det"][valid].float().to(device)
        losses["pulsar_det"] = F.binary_cross_entropy_with_logits(pred, tgt)

    # ── Pulsar Subtype (CE) ──────────────────────────────────────
    valid = labels["pulsar_subtype"] != IGNORE_LABEL
    if valid.any():
        pred = outputs["pulsar_subtype"][valid]
        tgt  = labels["pulsar_subtype"][valid].to(device)
        losses["pulsar_subtype"] = F.cross_entropy(pred, tgt)

    # ── Radio Morphology (CE) ────────────────────────────────────
    valid = labels["radio_morphology"] != IGNORE_LABEL
    if valid.any():
        pred = outputs["radio_morphology"][valid]
        tgt  = labels["radio_morphology"][valid].to(device)
        losses["radio_morphology"] = F.cross_entropy(pred, tgt)

    # ── GW Detection (BCE) ───────────────────────────────────────
    valid = labels["gw_det"] != IGNORE_LABEL
    if valid.any():
        pred = outputs["gw_det"][valid].squeeze(-1)
        tgt  = labels["gw_det"][valid].float().to(device)
        losses["gw_det"] = F.binary_cross_entropy_with_logits(pred, tgt)

    # ── Anomaly Score (BCE) ──────────────────────────────────────
    valid = labels["anomaly"] != IGNORE_LABEL
    if valid.any():
        pred = outputs["anomaly"][valid].squeeze(-1)
        tgt  = labels["anomaly"][valid].float().to(device)
        losses["anomaly"] = F.binary_cross_entropy_with_logits(pred, tgt)

    # ── Regression (MSE, mask-aware) ─────────────────────────────
    reg_mask = labels["reg_mask"].to(device)           # (B, 4)
    has_any_reg = reg_mask.sum(dim=1) > 0              # (B,)
    if has_any_reg.any():
        pred = outputs["regression"][has_any_reg]       # (N, 4)
        tgt  = labels["regression"][has_any_reg].to(device)
        m    = reg_mask[has_any_reg]                    # (N, 4)

        # Replace NaN targets with 0 (masked out anyway)
        tgt  = torch.where(torch.isnan(tgt), torch.zeros_like(tgt), tgt)

        mse  = (pred - tgt) ** 2 * m                   # (N, 4)
        if m.sum() > 0:
            losses["regression"] = mse.sum() / m.sum()

    # ── Physics Consistency (curriculum) ─────────────────────────
    physics_weight = _physics_curriculum_weight(epoch)
    if physics_weight > 0 and "stellar_cls" in losses:
        try:
            from core.physics_loss import stellar_physics_loss
            valid_m3 = labels["stellar_cls"] != IGNORE_LABEL
            if valid_m3.any():
                reg_pred = outputs["regression"][valid_m3]
                cls_pred = outputs["stellar_cls"][valid_m3]
                cls_hard = labels["stellar_cls"][valid_m3].to(device)

                phys = stellar_physics_loss(
                    reg_pred, cls_hard,
                    class_names=STELLAR_CLASS_NAMES,
                )
                if phys is not None and not torch.isnan(phys):
                    losses["physics"] = phys
        except Exception:
            pass  # Physics loss optional — don't crash training

    return losses


def _physics_curriculum_weight(epoch: int) -> float:
    """
    Curriculum warmup for physics loss.
    Epochs < START: weight = 0
    START → START+RAMP: linear ramp 0 → target
    After: target weight
    """
    if epoch < PHYSICS_LOSS_START_EPOCH:
        return 0.0
    ramp_epoch = epoch - PHYSICS_LOSS_START_EPOCH
    if ramp_epoch < PHYSICS_LOSS_RAMP_EPOCHS:
        return PHYSICS_LOSS_TARGET_WEIGHT * (ramp_epoch / PHYSICS_LOSS_RAMP_EPOCHS)
    return PHYSICS_LOSS_TARGET_WEIGHT


def compute_total_loss(losses: dict, epoch: int) -> torch.Tensor:
    """
    Weighted sum of per-head losses.
    Physics loss uses curriculum weight instead of fixed λ.
    """
    total = torch.tensor(0.0, device=next(iter(losses.values())).device,
                         requires_grad=True)

    for head_name, loss_val in losses.items():
        if head_name == "physics":
            w = _physics_curriculum_weight(epoch)
        else:
            w = LOSS_WEIGHTS.get(head_name, 1.0)
        total = total + w * loss_val

    return total


# ═════════════════════════════════════════════════════════════
# 2. METRICS
# ═════════════════════════════════════════════════════════════

class MetricTracker:
    """Tracks per-head metrics across batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = {}
        self.correct = {}
        self.losses = {}
        self.loss_counts = {}

    def update_cls(self, head: str, logits: torch.Tensor,
                   targets: torch.Tensor, valid_mask: torch.Tensor):
        """Update classification accuracy for a head."""
        if not valid_mask.any():
            return
        preds = logits[valid_mask].argmax(dim=1)
        tgts  = targets[valid_mask]
        n = tgts.size(0)
        correct = (preds == tgts).sum().item()
        self.counts[head]  = self.counts.get(head, 0) + n
        self.correct[head] = self.correct.get(head, 0) + correct

    def update_bin(self, head: str, logits: torch.Tensor,
                   targets: torch.Tensor, valid_mask: torch.Tensor):
        """Update binary accuracy for a head."""
        if not valid_mask.any():
            return
        preds = (logits[valid_mask].squeeze(-1) > 0).long()
        tgts  = targets[valid_mask]
        n = tgts.size(0)
        correct = (preds == tgts).sum().item()
        self.counts[head]  = self.counts.get(head, 0) + n
        self.correct[head] = self.correct.get(head, 0) + correct

    def update_loss(self, head: str, loss_val: float, n: int = 1):
        """Accumulate loss for averaging."""
        self.losses[head]      = self.losses.get(head, 0.0) + loss_val * n
        self.loss_counts[head] = self.loss_counts.get(head, 0) + n

    def get_accuracies(self) -> dict:
        return {h: self.correct[h] / max(self.counts[h], 1)
                for h in self.counts}

    def get_avg_losses(self) -> dict:
        return {h: self.losses[h] / max(self.loss_counts[h], 1)
                for h in self.losses}


# ═════════════════════════════════════════════════════════════
# 3. TRAINING LOOP (SINGLE STAGE)
# ═════════════════════════════════════════════════════════════

def train_stage(
    model:        UnifiedModel,
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    stage_cfg:    dict,
    device:       torch.device,
    logger,
    global_epoch: int = 0,
) -> tuple:
    """
    Train one stage of the unified model.

    Returns:
        (model, history, final_global_epoch)
    """
    label = stage_cfg["label"]
    epochs = stage_cfg["epochs"]
    logger.info(f"{'═' * 50}")
    logger.info(f"Stage: {label}")
    logger.info(f"Epochs: {epochs} | Starting from global epoch {global_epoch}")

    # ── Unfreeze specified encoders ──────────────────────────────
    for enc_name in stage_cfg.get("unfreeze", []):
        try:
            model.unfreeze_encoder(enc_name)
        except ValueError as e:
            logger.warning(f"Cannot unfreeze {enc_name}: {e}")

    # ── Optimizer: separate param groups ─────────────────────────
    param_groups = [
        {
            "params": list(model.projections.parameters()) +
                      list(model.fusion.parameters()),
            "lr": stage_cfg["lr_fusion"],
            "weight_decay": stage_cfg["weight_decay"],
        },
        {
            "params": list(model.stellar_cls_head.parameters()) +
                      list(model.pulsar_det_head.parameters()) +
                      list(model.pulsar_subtype_head.parameters()) +
                      list(model.radio_morphology_head.parameters()) +
                      list(model.gw_det_head.parameters()) +
                      list(model.anomaly_head.parameters()) +
                      list(model.regression_head.parameters()),
            "lr": stage_cfg["lr_fusion"],
            "weight_decay": stage_cfg["weight_decay"],
        },
    ]

    # Encoder params (if any unfrozen)
    enc_params = model.get_encoder_params()
    if enc_params:
        param_groups.append({
            "params": enc_params,
            "lr": stage_cfg.get("lr_encoder", 1e-5),
            "weight_decay": stage_cfg["weight_decay"],
        })
        logger.info(f"Encoder params in optimizer: {len(enc_params)}")

    optimizer = torch.optim.AdamW(param_groups)

    # ── Scheduler: warmup + cosine ───────────────────────────────
    warmup = stage_cfg.get("warmup_epochs", 0)
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Early stopping ───────────────────────────────────────────
    early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

    # ── History ──────────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "per_head_train_loss": [], "per_head_val_loss": [],
        "per_head_val_acc": [], "epoch_time_s": [],
        "lr": [],
    }

    best_val_loss = float("inf")

    for epoch_idx in range(epochs):
        epoch = global_epoch + epoch_idx
        epoch_start = time.time()

        # ── Train ────────────────────────────────────────────────
        model.train()
        # Keep frozen encoders in eval mode
        for name, enc in model._encoders.items():
            if not any(p.requires_grad for p in enc.parameters()):
                enc.eval()

        train_metrics = MetricTracker()
        train_total_loss = 0.0
        n_batches = 0

        for batch_inputs, batch_labels, batch_masks in tqdm(
            train_loader, desc=f"[{label}] Epoch {epoch+1}/{global_epoch+epochs}",
            leave=False
        ):
            # Move labels to device
            for k in batch_labels:
                if isinstance(batch_labels[k], torch.Tensor):
                    batch_labels[k] = batch_labels[k].to(device)
            batch_masks = batch_masks.to(device)

            # Move inputs to device
            for k in batch_inputs:
                batch_inputs[k] = batch_inputs[k].to(device)

            # Forward
            outputs = model(batch_inputs, batch_masks)

            # Loss
            losses = compute_losses(outputs, batch_labels, batch_masks,
                                    epoch, device)
            if not losses:
                continue

            total_loss = compute_total_loss(losses, epoch)

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                MAX_GRAD_NORM,
            )
            optimizer.step()
            scheduler.step()

            # Track
            train_total_loss += total_loss.item()
            n_batches += 1
            for h, lv in losses.items():
                train_metrics.update_loss(h, lv.item())

        avg_train_loss = train_total_loss / max(n_batches, 1)

        # ── Validate ─────────────────────────────────────────────
        model.eval()
        val_metrics = MetricTracker()
        val_total_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_inputs, batch_labels, batch_masks in val_loader:
                for k in batch_labels:
                    if isinstance(batch_labels[k], torch.Tensor):
                        batch_labels[k] = batch_labels[k].to(device)
                batch_masks = batch_masks.to(device)
                for k in batch_inputs:
                    batch_inputs[k] = batch_inputs[k].to(device)

                outputs = model(batch_inputs, batch_masks)
                losses = compute_losses(outputs, batch_labels, batch_masks,
                                        epoch, device)
                if not losses:
                    continue

                total_loss = compute_total_loss(losses, epoch)
                val_total_loss += total_loss.item()
                val_batches += 1

                for h, lv in losses.items():
                    val_metrics.update_loss(h, lv.item())

                # Accuracy tracking
                lbl = batch_labels
                valid_sc = lbl["stellar_cls"] != IGNORE_LABEL
                val_metrics.update_cls("stellar_cls",
                                       outputs["stellar_cls"], lbl["stellar_cls"],
                                       valid_sc)

                valid_pd = lbl["pulsar_det"] != IGNORE_LABEL
                val_metrics.update_bin("pulsar_det",
                                       outputs["pulsar_det"], lbl["pulsar_det"],
                                       valid_pd)

                valid_ps = lbl["pulsar_subtype"] != IGNORE_LABEL
                val_metrics.update_cls("pulsar_subtype",
                                       outputs["pulsar_subtype"],
                                       lbl["pulsar_subtype"], valid_ps)

                valid_rm = lbl["radio_morphology"] != IGNORE_LABEL
                val_metrics.update_cls("radio_morphology",
                                       outputs["radio_morphology"],
                                       lbl["radio_morphology"], valid_rm)

                valid_gw = lbl["gw_det"] != IGNORE_LABEL
                val_metrics.update_bin("gw_det",
                                       outputs["gw_det"], lbl["gw_det"],
                                       valid_gw)

        avg_val_loss = val_total_loss / max(val_batches, 1)
        epoch_time = time.time() - epoch_start

        # ── Log ──────────────────────────────────────────────────
        val_accs = val_metrics.get_accuracies()
        val_losses = val_metrics.get_avg_losses()
        train_losses = train_metrics.get_avg_losses()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["per_head_train_loss"].append(train_losses)
        history["per_head_val_loss"].append(val_losses)
        history["per_head_val_acc"].append(val_accs)
        history["epoch_time_s"].append(epoch_time)
        history["lr"].append(current_lr)

        # Print summary
        acc_str = " | ".join(f"{h}: {a:.3f}" for h, a in val_accs.items())
        logger.info(
            f"Epoch {epoch+1:>3d} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{epoch_time:.0f}s"
        )
        if acc_str:
            logger.info(f"  Acc: {acc_str}")

        # ── Checkpoint ───────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            ckpt_path = os.path.join(CHECKPOINT_DIR,
                                     f"unified_best_{label}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "val_accs": val_accs,
                "stage": label,
            }, ckpt_path)
            logger.info(f"  💾 Best checkpoint → {ckpt_path}")

        # ── Early stopping ───────────────────────────────────────
        if early_stop(avg_val_loss):
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    # ── Save history ─────────────────────────────────────────────
    os.makedirs(LOG_DIR, exist_ok=True)
    hist_path = os.path.join(LOG_DIR, f"unified_{label}_history.json")

    # Convert numpy/tensor values for JSON serialization
    def _jsonable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2, default=_jsonable)
    logger.info(f"📊 History saved → {hist_path}")

    final_epoch = global_epoch + epoch_idx + 1
    return model, history, final_epoch


# ═════════════════════════════════════════════════════════════
# 4. TRAINING CURVES
# ═════════════════════════════════════════════════════════════

def save_training_curves(all_histories: list, labels: list):
    """Save combined training curves across all stages."""
    os.makedirs(LOG_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("STELLARIS-DNet Unified — Training Curves", fontsize=14)

    all_train = []
    all_val = []
    for hist in all_histories:
        all_train.extend(hist["train_loss"])
        all_val.extend(hist["val_loss"])

    epochs = range(1, len(all_train) + 1)
    axes[0].plot(epochs, all_train, lw=1.5, label="Train", color="steelblue")
    axes[0].plot(epochs, all_val,   lw=1.5, label="Val",   color="darkorange")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # Stage boundaries
    ep = 0
    for hist, label in zip(all_histories, labels):
        ep += len(hist["train_loss"])
        axes[0].axvline(ep, color="gray", linestyle="--", alpha=0.5)
        axes[0].text(ep, axes[0].get_ylim()[1] * 0.95, label,
                     fontsize=7, ha="right", rotation=90)

    # Per-head val accuracy
    all_accs = []
    for hist in all_histories:
        all_accs.extend(hist["per_head_val_acc"])

    if all_accs:
        heads = set()
        for a in all_accs:
            heads.update(a.keys())
        for head in sorted(heads):
            vals = [a.get(head, float("nan")) for a in all_accs]
            axes[1].plot(range(1, len(vals)+1), vals, lw=1.5, label=head)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Per-Head Val Accuracy")
        axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(LOG_DIR, "unified_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 Training curves → {path}")


# ═════════════════════════════════════════════════════════════
# 5. MAIN TRAINING ENTRY POINT
# ═════════════════════════════════════════════════════════════

def run_training(stages_to_run: list = None):
    """
    Run the full unified training pipeline.

    Args:
        stages_to_run: list of stage indices [0, 1, 2] or None for all.
                       Stage 0 = frozen, Stage 1 = partial, Stage 2 = finetune
    """
    set_seed(SEED)
    device = get_device()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = get_logger("unified_train", LOG_DIR)

    logger.info("═" * 60)
    logger.info("STELLARIS-DNet — Unified Fusion Training")
    logger.info("═" * 60)

    # ── Load data ────────────────────────────────────────────────
    logger.info("Loading unified dataset...")
    train_loader, val_loader, test_loader = get_unified_loaders(verbose=True)
    logger.info(f"Train batches: {len(train_loader)} | "
                f"Val batches: {len(val_loader)}")

    if len(train_loader) == 0:
        logger.error("No training data — cannot proceed")
        return

    # ── Build model ──────────────────────────────────────────────
    logger.info("Building unified model...")
    model = UnifiedModel(device=device, load_encoders=True)
    model = model.to(device)

    counts = model.count_params()
    logger.info(f"Trainable params: {counts['grand_total']['trainable']:,}")
    logger.info(f"Total params:     {counts['grand_total']['total']:,}")

    # ── Run stages ───────────────────────────────────────────────
    if stages_to_run is None:
        stages_to_run = list(range(len(STAGES)))

    all_histories = []
    all_labels = []
    global_epoch = 0

    for stage_idx in stages_to_run:
        if stage_idx >= len(STAGES):
            logger.warning(f"Stage {stage_idx} not defined — skipping")
            continue

        stage_cfg = STAGES[stage_idx]
        logger.info(f"\n{'▶' * 3} Starting Stage {stage_idx}: {stage_cfg['label']}")

        model, history, global_epoch = train_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            stage_cfg=stage_cfg,
            device=device,
            logger=logger,
            global_epoch=global_epoch,
        )
        all_histories.append(history)
        all_labels.append(stage_cfg["label"])

    # ── Save final model ─────────────────────────────────────────
    final_path = os.path.join(CHECKPOINT_DIR, "unified_final.pt")
    torch.save({
        "epoch": global_epoch,
        "model_state_dict": model.state_dict(),
        "stages_completed": [STAGES[i]["label"] for i in stages_to_run],
    }, final_path)
    logger.info(f"💾 Final model → {final_path}")

    # ── Save curves ──────────────────────────────────────────────
    if all_histories:
        save_training_curves(all_histories, all_labels)

    logger.info("═" * 60)
    logger.info("Unified training complete")
    logger.info("═" * 60)

    return model, all_histories


# ═════════════════════════════════════════════════════════════
# 6. CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="STELLARIS-DNet Unified Training"
    )
    parser.add_argument(
        "--stages", type=int, nargs="+", default=None,
        help="Stage indices to run (e.g. --stages 0 1). Default: all"
    )
    args = parser.parse_args()

    run_training(stages_to_run=args.stages)