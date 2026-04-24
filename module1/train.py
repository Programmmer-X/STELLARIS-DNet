"""
module1/train.py
STELLARIS-DNet — Module 1 Training

Trains: MLP (HTRU2) + 1D CNN (pulse profiles) + Autoencoder (magnetar anomaly)

Upgrades:
  - Focal loss for MLP (fixes F1 gap on imbalanced HTRU2)
  - Physics-informed spindown loss term (optional)
  - Per-epoch precision, recall, F1 tracking
  - Training/validation curve saving (PNG)
  - Baseline vs enhanced comparison runs
  - Time-per-epoch logging
  - Enhanced multi-modal batch handling (time + freq + energy)
  - Reproducibility enforced per run
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
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module1.config  import *
from module1.dataset import load_htru2, load_pulse_profiles, load_autoencoder_data
from module1.model   import PulsarMLP, PulsarCNN, PulsarAutoencoder
from core.utils      import (set_seed, get_device, get_logger,
                              save_checkpoint, save_encoder,
                              EarlyStopping, count_parameters, print_epoch)
from core.physics_loss import spindown_energy_loss


# ═════════════════════════════════════════════
# SECTION 1 — LOSS FUNCTIONS
# ═════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Addresses class imbalance by down-weighting easy (confident) examples.
    FL(p) = -alpha * (1-p)^gamma * log(p)

    Reference: Lin et al., 2017 — RetinaNet
    Used when USE_FOCAL_LOSS=True in config.
    """
    def __init__(
        self,
        alpha:     float = FOCAL_ALPHA,
        gamma:     float = FOCAL_GAMMA,
        pos_weight: torch.Tensor = None,
    ):
        super().__init__()
        self.alpha      = alpha
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,   # (B,) raw logits
        targets: torch.Tensor,  # (B,) float binary labels
    ) -> torch.Tensor:
        # BCE loss per sample (no reduction)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probs    = torch.sigmoid(logits)
        p_t      = probs * targets + (1 - probs) * (1 - targets)
        focal_w  = (1 - p_t) ** self.gamma
        alpha_t  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal_w * bce).mean()


# ═════════════════════════════════════════════
# SECTION 2 — CURVE UTILITIES
# ═════════════════════════════════════════════

def _save_training_curves(
    history:   dict,
    name:      str,
    save_dir:  str = LOG_DIR,
):
    """
    Saves train/val loss + metric curves to PNG.
    history keys: 'train_loss', 'val_loss', and optionally
                  'val_acc', 'val_f1', 'val_precision', 'val_recall'
    """
    if not SAVE_CURVES:
        return

    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Loss curve ──
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=1.5)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"{name} — Loss Curves")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{name}_loss_curve.png"), dpi=100)
    plt.close(fig)

    # ── Metrics curve (if present) ──
    metric_keys = [k for k in history if k not in ("train_loss", "val_loss")]
    if metric_keys:
        fig, ax = plt.subplots(figsize=(8, 4))
        for k in metric_keys:
            if history[k]:
                ax.plot(epochs, history[k], label=k, linewidth=1.5)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
        ax.set_title(f"{name} — Metric Curves")
        ax.legend(); ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{name}_metric_curve.png"), dpi=100)
        plt.close(fig)

    print(f"📈 Curves saved → {save_dir}/{name}_*.png")


def _save_comparison(
    baseline_history: dict,
    enhanced_history: dict,
    name:     str,
    metric:   str = "val_f1",
    save_dir: str = LOG_DIR,
):
    """Plots baseline vs enhanced metric curve on same axes."""
    if not EVAL_COMPARISON:
        return
    os.makedirs(save_dir, exist_ok=True)
    ep_b = range(1, len(baseline_history.get(metric, [])) + 1)
    ep_e = range(1, len(enhanced_history.get(metric, [])) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    if baseline_history.get(metric):
        ax.plot(ep_b, baseline_history[metric],
                label=f"Baseline {metric}", linewidth=1.5, linestyle="--")
    if enhanced_history.get(metric):
        ax.plot(ep_e, enhanced_history[metric],
                label=f"Enhanced {metric}", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel(metric)
    ax.set_title(f"{name} — Baseline vs Enhanced ({metric})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, f"{name}_comparison_{metric}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"📊 Comparison saved → {path}")


def _save_history_json(history: dict, name: str, save_dir: str = LOG_DIR):
    """Persists training history as JSON for later analysis."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}_history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


# ═════════════════════════════════════════════
# SECTION 3 — BATCH HELPERS
# Handle both standard (X, y) and enhanced
# multi-modal dict batches from dataset.py
# ═════════════════════════════════════════════

def _unpack_cnn_batch(batch, device):
    """
    Unpacks a CNN batch — handles both modes:
      Standard:  (X_time, y) tuple
      Enhanced:  dict with keys 'time', 'freq', 'energy', 'label'
    Returns: x_time, x_freq (or None), y
    """
    if isinstance(batch, dict):
        x_time = batch["time"].to(device)
        x_freq = batch["freq"].to(device)  if "freq"   in batch else None
        y      = batch["label"].to(device)
    else:
        x_time, y = batch
        x_time = x_time.to(device)
        x_freq = None
        y      = y.to(device)
    return x_time, x_freq, y


def _unpack_ae_batch(batch, device):
    """
    Unpacks AE batch — handles both modes:
      Standard:  X tensor (B, L)
      Enhanced:  dict with key 'input'
    Returns: x (B, L)
    """
    if isinstance(batch, dict):
        return batch["input"].to(device)
    return batch.to(device)


# ═════════════════════════════════════════════
# SECTION 4 — TRAIN MLP
# ═════════════════════════════════════════════

def train_mlp(device, logger, tag: str = EXPERIMENT_TAG):
    logger.info("=" * 60)
    logger.info(f"Training MLP — HTRU2 | tag={tag} | "
                f"focal={USE_FOCAL_LOSS} | physics={USE_PHYSICS_LOSS}")
    logger.info("=" * 60)

    train_loader, val_loader, _, scaler, pos_weight = load_htru2()
    pos_weight = pos_weight.to(device)

    model     = PulsarMLP().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # Loss — focal or standard BCE
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(
            alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, pos_weight=pos_weight
        )
        logger.info(f"Loss: FocalLoss (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info("Loss: BCEWithLogitsLoss")

    early_stop    = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
    best_val_loss = float("inf")
    count_parameters(model)

    history = {
        "train_loss": [], "val_loss": [],
        "val_acc": [], "val_f1": [],
        "val_precision": [], "val_recall": [],
        "epoch_time_s": [],
    }

    for epoch in range(1, MLP_EPOCHS + 1):
        t_start = time.time()

        # ── Train ──────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X, y in tqdm(train_loader, desc=f"MLP {epoch}", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X).squeeze(1)
            loss   = criterion(logits, y)

            # Optional physics spindown loss
            # Note: HTRU2 features are statistical — no raw P/Ṗ available.
            # We use feature[0] (mean IP) as period proxy only for loss shape.
            if USE_PHYSICS_LOSS:
                with torch.no_grad():          # compute proxy without gradient risk
                    P_proxy    = torch.sigmoid(X[:, 0]).clamp(PERIOD_MIN, PERIOD_MAX)
                    Pdot_proxy = torch.sigmoid(X[:, 1]) * 1e-15
                E_proxy   = model.get_features(X).norm(dim=1)   # (B,)
                # Normalize I to float32-safe range before computing loss
                # Use dimensionless ratio: E_pred / E_expected
                E_expected = (4 * torch.pi**2 * torch.abs(Pdot_proxy)) / (P_proxy**3)
                # Both are now O(1) — no 1e45 overflow
                E_proxy_norm    = E_proxy    / (E_proxy.detach().mean().clamp(min=1e-8))
                E_expected_norm = E_expected / (E_expected.detach().mean().clamp(min=1e-8))
                phys_loss = F.mse_loss(E_proxy_norm, E_expected_norm.detach())
                if torch.isfinite(phys_loss):
                    loss = loss + SPINDOWN_LOSS_WEIGHT * phys_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ───────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X, y in val_loader:
                X, y   = X.to(device), y.to(device)
                logits = model(X).squeeze(1)
                loss   = criterion(logits, y)
                val_loss += loss.item()
                preds  = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.long().cpu().numpy())
        val_loss /= len(val_loader)

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        val_acc    = (all_preds == all_labels).mean()
        val_f1     = f1_score(all_labels, all_preds, zero_division=0)
        val_prec   = precision_score(all_labels, all_preds, zero_division=0)
        val_rec    = recall_score(all_labels, all_preds, zero_division=0)

        epoch_time = time.time() - t_start
        scheduler.step(val_loss)

        # ── Log ────────────────────────────────────────
        print_epoch(epoch, MLP_EPOCHS, train_loss, val_loss, val_acc,
                    extra={"F1": val_f1, "Prec": val_prec, "Rec": val_rec})
        if LOG_TIME:
            print(f"   ⏱  {epoch_time:.1f}s/epoch")

        logger.info(
            f"Epoch {epoch:03d} | train={train_loss:.4f} val={val_loss:.4f} "
            f"acc={val_acc:.4f} f1={val_f1:.4f} "
            f"prec={val_prec:.4f} rec={val_rec:.4f} t={epoch_time:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)
        history["epoch_time_s"].append(epoch_time)

        # ── Checkpoint ─────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            CHECKPOINT_DIR, f"mlp_best_{tag}.pt")
            save_encoder(model.encoder, CHECKPOINT_DIR, "mlp_encoder.pt")

        if early_stop(val_loss):
            break

    _save_training_curves(history, f"mlp_{tag}")
    _save_history_json(history, f"mlp_{tag}")
    logger.info(f"MLP done. Best val loss: {best_val_loss:.4f}")
    return model, history


# ═════════════════════════════════════════════
# SECTION 5 — TRAIN 1D CNN
# ═════════════════════════════════════════════

def train_cnn(device, logger, tag: str = EXPERIMENT_TAG):
    logger.info("=" * 60)
    logger.info(f"Training CNN — Subtypes | tag={tag} | "
                f"attn={USE_ATTENTION} | freq_fusion={USE_FREQ_FUSION}")
    logger.info("=" * 60)

    # Enhanced loaders if features enabled
    enhanced      = USE_FFT or USE_CQT or USE_AUGMENTATION
    train_loader, val_loader, _ = load_pulse_profiles(enhanced=enhanced)

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
    history = {
        "train_loss": [], "val_loss": [],
        "val_acc": [], "val_f1": [],
        "val_precision": [], "val_recall": [],
        "epoch_time_s": [],
    }

    for epoch in range(1, CNN_EPOCHS + 1):
        t_start = time.time()

        # ── Train ──────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"CNN {epoch}", leave=False):
            x_time, x_freq, y = _unpack_cnn_batch(batch, device)
            optimizer.zero_grad()
            out  = model(x_time, x_freq)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ───────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                x_time, x_freq, y = _unpack_cnn_batch(batch, device)
                out      = model(x_time, x_freq)
                loss     = criterion(out, y)
                val_loss += loss.item()
                preds    = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.cpu().numpy())
        val_loss /= len(val_loader)

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        val_acc    = (all_preds == all_labels).mean()
        val_f1     = f1_score(all_labels, all_preds,
                              average="macro", zero_division=0)
        val_prec   = precision_score(all_labels, all_preds,
                                     average="macro", zero_division=0)
        val_rec    = recall_score(all_labels, all_preds,
                                  average="macro", zero_division=0)

        epoch_time = time.time() - t_start
        scheduler.step()

        print_epoch(epoch, CNN_EPOCHS, train_loss, val_loss, val_acc,
                    extra={"F1": val_f1, "Prec": val_prec, "Rec": val_rec})
        if LOG_TIME:
            print(f"   ⏱  {epoch_time:.1f}s/epoch")

        logger.info(
            f"Epoch {epoch:03d} | train={train_loss:.4f} val={val_loss:.4f} "
            f"acc={val_acc:.4f} f1={val_f1:.4f} t={epoch_time:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)
        history["epoch_time_s"].append(epoch_time)

        # ── Checkpoint ─────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            CHECKPOINT_DIR, f"cnn_best_{tag}.pt")
            # Save full encoder (conv_encoder + time_proj + freq_fusion)
            # This is the fix: save the encode() subgraph, not just conv_encoder
            save_encoder(
                nn.Sequential(
                    model.conv_encoder,
                    model.pool,
                    model.time_proj,
                ),
                CHECKPOINT_DIR, "cnn_encoder.pt"
            )

        if early_stop(val_loss):
            break

    _save_training_curves(history, f"cnn_{tag}")
    _save_history_json(history, f"cnn_{tag}")
    logger.info(f"CNN done. Best val loss: {best_val_loss:.4f}")
    return model, history


# ═════════════════════════════════════════════
# SECTION 6 — TRAIN AUTOENCODER
# ═════════════════════════════════════════════

def train_autoencoder(device, logger, tag: str = EXPERIMENT_TAG):
    logger.info("=" * 60)
    logger.info(f"Training AE — Magnetar Anomaly | tag={tag}")
    logger.info("=" * 60)

    enhanced     = USE_FFT or USE_AUGMENTATION
    train_loader, val_loader = load_autoencoder_data(enhanced=enhanced)

    model     = PulsarAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    criterion  = nn.MSELoss()
    early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
    count_parameters(model)

    best_val_loss = float("inf")
    history = {
        "train_loss": [], "val_loss": [],
        "epoch_time_s": [],
    }

    for epoch in range(1, AE_EPOCHS + 1):
        t_start = time.time()

        # ── Train ──────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"AE {epoch}", leave=False):
            X = _unpack_ae_batch(batch, device)
            optimizer.zero_grad()
            recon = model(X)
            loss  = criterion(recon, X)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss /= len(train_loader)

        # ── Validate ───────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                X     = _unpack_ae_batch(batch, device)
                recon = model(X)
                val_loss += criterion(recon, X).item()
        val_loss /= len(val_loader)

        epoch_time = time.time() - t_start
        scheduler.step(val_loss)

        print_epoch(epoch, AE_EPOCHS, train_loss, val_loss)
        if LOG_TIME:
            print(f"   ⏱  {epoch_time:.1f}s/epoch")

        logger.info(
            f"Epoch {epoch:03d} | train={train_loss:.6f} "
            f"val={val_loss:.6f} t={epoch_time:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_time_s"].append(epoch_time)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            CHECKPOINT_DIR, f"ae_best_{tag}.pt")
            save_encoder(model.encoder, CHECKPOINT_DIR, "ae_encoder.pt")

        if early_stop(val_loss):
            break

    _save_training_curves(history, f"ae_{tag}")
    _save_history_json(history, f"ae_{tag}")
    logger.info(f"AE done. Best val loss: {best_val_loss:.6f}")
    return model, history


# ═════════════════════════════════════════════
# SECTION 7 — COMPARISON RUN
# Baseline (no enhancements) vs Enhanced
# Runs both configs, saves comparison plots
# ═════════════════════════════════════════════

def run_comparison(device, logger):
    """
    Trains MLP and CNN twice:
      1. Baseline — all USE_* flags effectively disabled
      2. Enhanced — current config flags active
    Saves side-by-side comparison plots.
    Only runs if RUN_BASELINE=True AND RUN_ENHANCED=True.
    """
    if not (RUN_BASELINE and RUN_ENHANCED and EVAL_COMPARISON):
        return

    logger.info("=" * 60)
    logger.info("Comparison Run: Baseline vs Enhanced")
    logger.info("=" * 60)

    results = {}

    # ── Baseline: force standard loaders + no focal/physics loss ──
    # We temporarily monkey-patch via direct calls
    logger.info("── Baseline MLP ──")
    train_loader, val_loader, _, _, pos_weight = load_htru2()
    pos_weight = pos_weight.to(device)
    mlp_b = PulsarMLP().to(device)
    opt_b = torch.optim.Adam(mlp_b.parameters(), lr=MLP_LR,
                             weight_decay=MLP_WEIGHT_DECAY)
    crit_b = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    es_b   = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)
    hist_b = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_b = float("inf")

    for epoch in range(1, MLP_EPOCHS + 1):
        mlp_b.train()
        tl = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt_b.zero_grad()
            loss = crit_b(mlp_b(X).squeeze(1), y)
            loss.backward(); opt_b.step()
            tl += loss.item()
        tl /= len(train_loader)

        mlp_b.eval()
        vl, preds_b, labels_b = 0.0, [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = mlp_b(X).squeeze(1)
                vl    += crit_b(logits, y).item()
                preds_b.extend((torch.sigmoid(logits) > 0.5).long().cpu().numpy())
                labels_b.extend(y.long().cpu().numpy())
        vl /= len(val_loader)
        f1 = f1_score(np.array(labels_b), np.array(preds_b), zero_division=0)
        hist_b["train_loss"].append(tl)
        hist_b["val_loss"].append(vl)
        hist_b["val_f1"].append(f1)
        if vl < best_b:
            best_b = vl
            save_checkpoint(mlp_b, opt_b, epoch, vl,
                            CHECKPOINT_DIR, "mlp_best_baseline.pt")
        if es_b(vl):
            break

    results["mlp_baseline"] = hist_b
    logger.info(f"Baseline MLP — best val_loss={best_b:.4f} | "
                f"final F1={hist_b['val_f1'][-1]:.4f}")

    # ── Enhanced MLP ──
    logger.info("── Enhanced MLP ──")
    _, hist_e = train_mlp(device, logger, tag="enhanced")
    results["mlp_enhanced"] = hist_e

    # ── Comparison plot ──
    _save_comparison(hist_b, hist_e, "mlp", metric="val_f1")
    _save_comparison(hist_b, hist_e, "mlp", metric="val_loss")

    # ── Summary ──
    final_b_f1 = hist_b["val_f1"][-1]  if hist_b["val_f1"]  else 0.0
    final_e_f1 = hist_e["val_f1"][-1]  if hist_e["val_f1"]  else 0.0
    delta      = final_e_f1 - final_b_f1

    logger.info("─" * 40)
    logger.info(f"Baseline F1 : {final_b_f1:.4f}")
    logger.info(f"Enhanced F1 : {final_e_f1:.4f}")
    logger.info(f"Δ F1        : {delta:+.4f}")
    logger.info("─" * 40)

    print(f"\n── Comparison Summary ──")
    print(f"Baseline MLP F1 : {final_b_f1:.4f}")
    print(f"Enhanced MLP F1 : {final_e_f1:.4f}")
    print(f"Δ F1            : {delta:+.4f}")

    return results


# ═════════════════════════════════════════════
# SECTION 8 — MAIN
# ═════════════════════════════════════════════

if __name__ == "__main__":
    set_seed(SEED)
    device = get_device()
    logger = get_logger("module1", LOG_DIR)

    logger.info("STELLARIS-DNet | Module 1 Training Started")
    logger.info(f"Device       : {device}")
    logger.info(f"Experiment   : {EXPERIMENT_TAG}")
    logger.info(f"Focal Loss   : {USE_FOCAL_LOSS}")
    logger.info(f"Physics Loss : {USE_PHYSICS_LOSS}")
    logger.info(f"Attention    : {USE_ATTENTION}")
    logger.info(f"Freq Fusion  : {USE_FREQ_FUSION}")
    logger.info(f"Augmentation : {USE_AUGMENTATION}")

    # ── MLP: comparison run (baseline vs enhanced) ──
    if RUN_BASELINE and RUN_ENHANCED and EVAL_COMPARISON:
        run_comparison(device, logger)
    else:
        train_mlp(device, logger)

    # ── CNN: always trains ──
    cnn_model, cnn_history = train_cnn(device, logger)

    # ── AE: always trains ──
    ae_model, ae_history = train_autoencoder(device, logger)

    logger.info("=" * 60)
    logger.info("Module 1 Training Complete")
    logger.info(f"Checkpoints : {CHECKPOINT_DIR}")
    logger.info(f"Logs/Curves : {LOG_DIR}")
    logger.info("=" * 60)

    print("\n✅ Module 1 training complete.")
    print(f"   Checkpoints → {CHECKPOINT_DIR}")
    print(f"   Curves      → {LOG_DIR}")
    print("   Run module1/evaluate.py next.")