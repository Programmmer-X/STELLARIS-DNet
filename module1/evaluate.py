"""
module1/evaluate.py
STELLARIS-DNet — Module 1 Evaluation

Evaluates: MLP + 1D CNN + Autoencoder

Upgrades:
  - ROC curve + Precision-Recall curve (MLP)
  - Misclassification analysis (visualize wrong predictions)
  - Confidence distribution plots (correct vs incorrect)
  - Normal vs magnetar reconstruction visualization
  - Separate error distributions (normal vs anomaly)
  - Latent space PCA visualization (AE)
  - Class-wise confusion insights (CNN)
  - Inference speed benchmark (all models)
  - Baseline vs enhanced comparison report
  - Dual threshold reporting (percentile + z-score)
  - All plots gated by config eval toggles
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score,
)
from sklearn.decomposition import PCA

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module1.config  import *
from module1.dataset import (load_htru2, load_pulse_profiles,
                              load_autoencoder_data,
                              _generate_synthetic_profiles,
                              preprocess_profile)
from module1.model   import PulsarMLP, PulsarCNN, PulsarAutoencoder
from module1.train   import _unpack_cnn_batch, _unpack_ae_batch
from core.utils      import get_device, get_logger


# ═════════════════════════════════════════════
# SECTION 1 — SHARED HELPERS
# ═════════════════════════════════════════════

def _load_model(
    model:  nn.Module,
    path:   str,
    device: torch.device,
    tag:    str = EXPERIMENT_TAG,
) -> nn.Module:
    """
    Loads checkpoint into model. Tries tag-specific path first,
    then falls back to the base path (original naming).
    """
    # Try tag-specific first
    tag_path = path.replace(".pt", f"_{tag}.pt")
    load_path = tag_path if os.path.exists(tag_path) else path

    if not os.path.exists(load_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {load_path}\n"
            f"Run module1/train.py first."
        )
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"✅ Loaded: {load_path}")
    return model


def _save_plot(fig, name: str, subdir: str = ""):
    """Saves figure to LOG_DIR (optionally in a subdirectory)."""
    save_dir = os.path.join(LOG_DIR, subdir) if subdir else LOG_DIR
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, name)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Plot saved → {path}")


def _confusion_matrix_plot(
    labels:      np.ndarray,
    preds:       np.ndarray,
    class_names: list,
    title:       str,
    filename:    str,
):
    """Reusable annotated confusion matrix plot."""
    n   = len(class_names)
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(max(5, n * 1.4), max(4, n * 1.2)))
    im  = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color=color, fontsize=10)
    fig.colorbar(im)
    fig.tight_layout()
    _save_plot(fig, filename)
    return cm


# ═════════════════════════════════════════════
# SECTION 2 — EVALUATE MLP
# ═════════════════════════════════════════════

def evaluate_mlp(device, logger):
    logger.info("=" * 60)
    logger.info("Evaluating MLP — HTRU2 Binary Classifier")
    logger.info("=" * 60)

    _, _, test_loader, _, pos_weight = load_htru2()
    model = _load_model(
        PulsarMLP(),
        os.path.join(CHECKPOINT_DIR, "mlp_best.pt"),
        device
    )

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for X, y in test_loader:
            X      = X.to(device)
            logits = model(X).squeeze(1)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.numpy().astype(int))

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    # ── Core metrics ──
    auc  = roc_auc_score(all_labels, all_probs)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    acc  = (all_preds == all_labels).mean()
    ap   = average_precision_score(all_labels, all_probs)

    logger.info(f"Accuracy  : {acc:.4f}")
    logger.info(f"ROC-AUC   : {auc:.4f}")
    logger.info(f"Avg Prec  : {ap:.4f}")
    logger.info(f"F1        : {f1:.4f}")
    logger.info(f"Precision : {prec:.4f}")
    logger.info(f"Recall    : {rec:.4f}")
    logger.info("\n" + classification_report(
        all_labels, all_preds, target_names=HTRU2_CLASSES
    ))

    print(f"\n── MLP Results ──")
    print(f"Accuracy  : {acc:.4f} | ROC-AUC: {auc:.4f} | AP: {ap:.4f}")
    print(f"F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    print(classification_report(all_labels, all_preds,
                                 target_names=HTRU2_CLASSES))

    # ── Confusion matrix ──
    _confusion_matrix_plot(
        all_labels, all_preds, HTRU2_CLASSES,
        "MLP — Confusion Matrix", "mlp_confusion_matrix.png"
    )

    # ── ROC Curve ──
    if EVAL_ROC_CURVE:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC={auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("MLP — ROC Curve")
        ax.legend(); ax.grid(alpha=0.3)
        _save_plot(fig, "mlp_roc_curve.png")

    # ── Precision-Recall Curve ──
    if EVAL_PR_CURVE:
        precision_pts, recall_pts, _ = precision_recall_curve(
            all_labels, all_probs
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall_pts, precision_pts, lw=2,
                label=f"AP={ap:.4f}", color="darkorange")
        ax.axhline(all_labels.mean(), color="navy", linestyle="--",
                   label=f"Baseline (prevalence={all_labels.mean():.3f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("MLP — Precision-Recall Curve")
        ax.legend(); ax.grid(alpha=0.3)
        _save_plot(fig, "mlp_pr_curve.png")

    # ── Confidence distribution ──
    if EVAL_CONFIDENCE:
        correct_mask = (all_preds == all_labels)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(all_probs[correct_mask],  bins=40, alpha=0.65,
                color="steelblue", label="Correct")
        ax.hist(all_probs[~correct_mask], bins=40, alpha=0.65,
                color="tomato",    label="Incorrect")
        ax.axvline(0.5, color="black", linestyle="--", label="Threshold=0.5")
        ax.set_xlabel("Predicted Probability (Pulsar)")
        ax.set_ylabel("Count")
        ax.set_title("MLP — Confidence Distribution")
        ax.legend(); ax.grid(alpha=0.3)
        _save_plot(fig, "mlp_confidence_dist.png")

    # ── Misclassification analysis ──
    if EVAL_ERROR_ANALYSIS:
        wrong_idx    = np.where(all_preds != all_labels)[0]
        wrong_probs  = all_probs[wrong_idx]
        wrong_labels = all_labels[wrong_idx]
        wrong_preds  = all_preds[wrong_idx]

        # Most confident wrong predictions
        sorted_idx  = np.argsort(-wrong_probs)[:20]
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["tomato" if wrong_preds[i] == 1 else "steelblue"
                  for i in sorted_idx]
        ax.bar(range(len(sorted_idx)),
               wrong_probs[sorted_idx], color=colors)
        ax.axhline(0.5, color="black", linestyle="--")
        ax.set_xlabel("Sample rank (most confident wrong)")
        ax.set_ylabel("Predicted probability")
        ax.set_title("MLP — Most Confident Misclassifications\n"
                     "(Red=predicted Pulsar, Blue=predicted Non-Pulsar)")
        ax.grid(alpha=0.3)
        _save_plot(fig, "mlp_misclassifications.png")

        logger.info(f"Total misclassifications: {len(wrong_idx)} / {len(all_labels)}")
        logger.info(f"FP (noise predicted as pulsar): "
                    f"{((wrong_preds==1) & (wrong_labels==0)).sum()}")
        logger.info(f"FN (pulsar predicted as noise): "
                    f"{((wrong_preds==0) & (wrong_labels==1)).sum()}")

    # ── Speed benchmark ──
    if EVAL_SPEED_BENCH:
        _benchmark_speed(model, torch.randn(256, MLP_INPUT_DIM).to(device),
                         "MLP", logger)

    return {"acc": acc, "auc": auc, "ap": ap, "f1": f1,
            "precision": prec, "recall": rec}


# ═════════════════════════════════════════════
# SECTION 3 — EVALUATE CNN
# ═════════════════════════════════════════════

def evaluate_cnn(device, logger):
    logger.info("=" * 60)
    logger.info("Evaluating CNN — Pulsar Subtype Classifier")
    logger.info("=" * 60)

    enhanced = USE_FFT or USE_CQT or USE_AUGMENTATION
    _, _, test_loader = load_pulse_profiles(enhanced=enhanced)
    model = _load_model(
        PulsarCNN(),
        os.path.join(CHECKPOINT_DIR, "cnn_best.pt"),
        device
    )

    all_preds, all_labels = [], []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            x_time, x_freq, y = _unpack_cnn_batch(batch, device)
            out    = model(x_time, x_freq)
            probs  = torch.softmax(out, dim=1).cpu().numpy()
            preds  = out.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)   # (N, num_classes)

    acc = (all_preds == all_labels).mean()
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    logger.info(f"Accuracy  : {acc:.4f}")
    logger.info(f"Macro F1  : {f1:.4f}")
    logger.info("\n" + classification_report(
        all_labels, all_preds, target_names=PULSAR_CLASSES
    ))

    print(f"\n── CNN Results ──")
    print(f"Accuracy : {acc:.4f} | Macro F1 : {f1:.4f}")
    print(classification_report(all_labels, all_preds,
                                 target_names=PULSAR_CLASSES))

    # ── Confusion matrix ──
    cm = _confusion_matrix_plot(
        all_labels, all_preds, PULSAR_CLASSES,
        "CNN — Pulsar Subtype Confusion Matrix",
        "cnn_confusion_matrix.png"
    )

    # ── Class-wise breakdown ──
    if EVAL_CLASSWISE:
        per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(PULSAR_CLASSES, per_class_acc,
                      color=["steelblue", "darkorange", "seagreen", "tomato"])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Per-Class Accuracy")
        ax.set_title("CNN — Per-Class Accuracy")
        for bar, val in zip(bars, per_class_acc):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.3f}", ha="center", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        _save_plot(fig, "cnn_classwise_accuracy.png")

    # ── Confidence distribution ──
    if EVAL_CONFIDENCE:
        correct_mask  = (all_preds == all_labels)
        max_probs     = all_probs.max(axis=1)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(max_probs[correct_mask],  bins=40, alpha=0.65,
                color="steelblue", label="Correct")
        ax.hist(max_probs[~correct_mask], bins=40, alpha=0.65,
                color="tomato",    label="Incorrect")
        ax.set_xlabel("Max Softmax Probability")
        ax.set_ylabel("Count")
        ax.set_title("CNN — Confidence Distribution")
        ax.legend(); ax.grid(alpha=0.3)
        _save_plot(fig, "cnn_confidence_dist.png")

    # ── Misclassification analysis ──
    if EVAL_ERROR_ANALYSIS:
        wrong_mask  = ~correct_mask
        wrong_idx   = np.where(wrong_mask)[0]
        if len(wrong_idx) > 0:
            logger.info(f"Total misclassifications: {wrong_mask.sum()} / {len(all_labels)}")
            # Per-class error counts
            for ci, cls_name in enumerate(PULSAR_CLASSES):
                cls_mask  = all_labels == ci
                cls_wrong = (wrong_mask & cls_mask).sum()
                cls_total = cls_mask.sum()
                logger.info(f"  {cls_name}: {cls_wrong}/{cls_total} wrong "
                            f"({100*cls_wrong/max(cls_total,1):.1f}%)")

    # ── Speed benchmark ──
    if EVAL_SPEED_BENCH:
        dummy_time = torch.randn(256, 1, SIGNAL_LENGTH).to(device)
        dummy_freq = torch.randn(256, FFT_BINS).to(device) if USE_FFT else None
        _benchmark_speed_cnn(model, dummy_time, dummy_freq, logger)

    return {"acc": acc, "f1": f1}


# ═════════════════════════════════════════════
# SECTION 4 — EVALUATE AUTOENCODER
# ═════════════════════════════════════════════

def evaluate_autoencoder(device, logger):
    logger.info("=" * 60)
    logger.info("Evaluating AE — Setting Magnetar Threshold")
    logger.info("=" * 60)

    enhanced = USE_FFT or USE_AUGMENTATION
    _, val_loader = load_autoencoder_data(enhanced=enhanced)
    model = _load_model(
        PulsarAutoencoder(),
        os.path.join(CHECKPOINT_DIR, "ae_best.pt"),
        device
    )

    # ── Collect val errors (normal pulsars only) ──
    all_errors, all_inputs, all_recons, all_latents = [], [], [], []

    with torch.no_grad():
        for batch in val_loader:
            X      = _unpack_ae_batch(batch, device)
            errors = model.reconstruction_error(X)
            recon  = model(X)
            z      = model.get_latent(X)
            all_errors.extend(errors.cpu().numpy())
            all_inputs.extend(X.cpu().numpy())
            all_recons.extend(recon.cpu().numpy())
            all_latents.extend(z.cpu().numpy())

    all_errors  = np.array(all_errors)
    all_inputs  = np.array(all_inputs)
    all_recons  = np.array(all_recons)
    all_latents = np.array(all_latents)

    # ── Compute both thresholds ──
    thresh_pct = model.compute_threshold(all_errors, method="percentile")
    thresh_z   = model.compute_threshold(all_errors, method="zscore")
    # Use config-selected method as primary
    threshold  = thresh_pct if AE_THRESHOLD_METHOD == "percentile" else thresh_z

    logger.info(f"Val errors — mean: {all_errors.mean():.6f} "
                f"std: {all_errors.std():.6f}")
    logger.info(f"Threshold (percentile {AE_ANOMALY_PERCENTILE}th): {thresh_pct:.6f}")
    logger.info(f"Threshold (z-score  {AE_ZSCORE_SIGMA}σ)         : {thresh_z:.6f}")
    logger.info(f"Active threshold ({AE_THRESHOLD_METHOD})          : {threshold:.6f}")

    print(f"\n── Autoencoder Results ──")
    print(f"Val errors — mean: {all_errors.mean():.6f} | std: {all_errors.std():.6f}")
    print(f"Threshold ({AE_ANOMALY_PERCENTILE}th pct)  : {thresh_pct:.6f}")
    print(f"Threshold ({AE_ZSCORE_SIGMA}σ z-score): {thresh_z:.6f}")
    print(f"Active ({AE_THRESHOLD_METHOD})         : {threshold:.6f}")

    # ── Save threshold ──
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    np.save(os.path.join(CHECKPOINT_DIR, "ae_threshold.npy"),
            np.array([threshold]))
    print(f"💾 Threshold saved → {CHECKPOINT_DIR}/ae_threshold.npy")

    # ── Error distribution (normal only) ──
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_errors, bins=50, color="steelblue", alpha=0.7,
            label="Normal pulsar val errors")
    ax.axvline(thresh_pct, color="red",    linestyle="--",
               label=f"Pct threshold ({AE_ANOMALY_PERCENTILE}th) = {thresh_pct:.4f}")
    ax.axvline(thresh_z,   color="orange", linestyle="-.",
               label=f"Z-score threshold ({AE_ZSCORE_SIGMA}σ) = {thresh_z:.4f}")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Count")
    ax.set_title("AE — Reconstruction Error Distribution (Normal Pulsars)")
    ax.legend(); ax.grid(alpha=0.3)
    _save_plot(fig, "ae_error_distribution.png")

    # ── Normal vs simulated anomaly error comparison ──
    if EVAL_ANOMALY_VIZ:
        # Generate synthetic magnetar-like profiles (high scatter, complex)
        mag_profiles, _ = _generate_synthetic_profiles(n=200, normal_only=False)
        # Corrupt with extreme scatter to simulate magnetar anomaly
        rng = np.random.default_rng(SEED)
        magnetar_like = []
        for p in mag_profiles[:100]:
            p  = p + rng.normal(0, 0.15, SIGNAL_LENGTH)   # heavy noise
            p  = np.clip(p, 0, 1).astype(np.float32)
            magnetar_like.append(p)
        magnetar_like = np.array(magnetar_like)

        mag_tensor  = torch.tensor(magnetar_like).to(device)
        mag_errors  = model.reconstruction_error(mag_tensor).cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(all_errors,  bins=40, alpha=0.65, color="steelblue",
                label=f"Normal (n={len(all_errors)})")
        ax.hist(mag_errors,  bins=40, alpha=0.65, color="tomato",
                label=f"Magnetar-like (n={len(mag_errors)})")
        ax.axvline(threshold, color="black", linestyle="--",
                   label=f"Threshold = {threshold:.4f}")
        ax.set_xlabel("Reconstruction Error (MSE)")
        ax.set_ylabel("Count")
        ax.set_title("AE — Normal vs Magnetar-like Error Distributions")
        ax.legend(); ax.grid(alpha=0.3)
        _save_plot(fig, "ae_normal_vs_anomaly.png")

        flagged_normal  = (all_errors > threshold).sum()
        flagged_mag     = (mag_errors > threshold).sum()
        logger.info(f"Normal flagged as anomaly     : "
                    f"{flagged_normal}/{len(all_errors)} "
                    f"({100*flagged_normal/len(all_errors):.1f}%)")
        logger.info(f"Magnetar-like correctly flagged: "
                    f"{flagged_mag}/{len(mag_errors)} "
                    f"({100*flagged_mag/len(mag_errors):.1f}%)")
        print(f"Normal flagged (FP rate)  : "
              f"{flagged_normal}/{len(all_errors)} "
              f"({100*flagged_normal/len(all_errors):.1f}%)")
        print(f"Magnetar-like detected    : "
              f"{flagged_mag}/{len(mag_errors)} "
              f"({100*flagged_mag/len(mag_errors):.1f}%)")

        # ── Reconstruction visualization ──
        n_show = 6
        fig, axes = plt.subplots(2, n_show, figsize=(14, 5))
        fig.suptitle("AE — Normal vs Magnetar-like Reconstructions", fontsize=12)
        with torch.no_grad():                          # ← FIX: wrap in no_grad
            for i in range(n_show):
            # Normal
                axes[0, i].plot(all_inputs[i], lw=1.5, color="steelblue")
                axes[0, i].plot(all_recons[i], lw=1.5, linestyle="--",
                               color="navy")
                axes[0, i].set_title(f"Normal\nerr={all_errors[i]:.4f}",
                                      fontsize=8)
                axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
                # Magnetar-like
                mag_r = model(mag_tensor[i:i+1]).cpu().numpy()[0]
                axes[1, i].plot(magnetar_like[i], lw=1.5, color="tomato")
                axes[1, i].plot(mag_r, lw=1.5, linestyle="--", color="navy")
                axes[1, i].set_title(f"Magnetar-like\nerr={mag_errors[i]:.4f}",
                                      fontsize=8)
                axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
        if i == 0:
                    axes[0, 0].legend(["Input", "Recon"], fontsize=6)
        fig.tight_layout()
        _save_plot(fig, "ae_reconstruction_comparison.png")

    # ── Latent space PCA ──
    if EVAL_LATENT_VIZ and all_latents.shape[0] >= 10:
        # Mix in anomaly latents for contrast
        # Build anomaly latents independently — no scope dependency
        rng = np.random.default_rng(SEED + 1)
        # Generate fresh anomaly-like inputs for latent comparison
        anomaly_inputs = np.clip(
        np.random.default_rng(SEED + 2).normal(0.5, 0.25,
        (50, AE_INPUT_DIM)), 0, 1
        ).astype(np.float32)
        anomaly_tensor = torch.tensor(anomaly_inputs).to(device)
        with torch.no_grad():                          # ← FIX: wrap in no_grad
            mag_lat = model.get_latent(anomaly_tensor).cpu().numpy()

        combined   = np.vstack([all_latents[:200], mag_lat])
        labels_pca = np.array(
        ["Normal"] * min(200, len(all_latents)) + ["Anomaly"] * len(mag_lat)
        )

        pca   = PCA(n_components=2, random_state=SEED)
        z_2d  = pca.fit_transform(combined)
        var_r = pca.explained_variance_ratio_
        
        fig, ax = plt.subplots(figsize=(7, 6))
        for grp, color, marker in [("Normal",  "steelblue", "o"),
                                    ("Anomaly", "tomato",    "^")]:
            mask = labels_pca == grp
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                       c=color, marker=marker, alpha=0.6, s=20, label=grp)
            
        ax.set_xlabel(f"PC1 ({100*var_r[0]:.1f}% var)")
        ax.set_ylabel(f"PC2 ({100*var_r[1]:.1f}% var)")
        ax.set_title("AE — Latent Space PCA (Normal vs Anomaly)")
        ax.legend(); ax.grid(alpha=0.3)
        _save_plot(fig, "ae_latent_pca.png")   


    # ── Speed benchmark ──
    if EVAL_SPEED_BENCH:
        _benchmark_speed(
            model,
            torch.randn(256, AE_INPUT_DIM).to(device),
            "Autoencoder", logger
        )

    return {
        "threshold_pct": thresh_pct,
        "threshold_z":   thresh_z,
        "threshold":     threshold,
        "mean_error":    float(all_errors.mean()),
        "std_error":     float(all_errors.std()),
    }


# ═════════════════════════════════════════════
# SECTION 5 — SPEED BENCHMARKS
# ═════════════════════════════════════════════

def _benchmark_speed(
    model:   nn.Module,
    x_dummy: torch.Tensor,
    name:    str,
    logger,
    n_runs:  int = 100,
):
    """
    Measures inference latency and throughput.
    Warms up GPU for 10 runs, then times n_runs forward passes.
    """
    model.eval()
    batch_size = x_dummy.shape[0]

    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(x_dummy)

        # Timed runs
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(x_dummy)
        elapsed = time.perf_counter() - t0

    latency_ms   = (elapsed / n_runs) * 1000
    throughput   = (batch_size * n_runs) / elapsed

    msg = (f"{name} | Latency: {latency_ms:.2f} ms/batch "
           f"| Throughput: {throughput:.0f} samples/s "
           f"| Batch size: {batch_size}")
    print(f"⚡ {msg}")
    logger.info(f"Speed | {msg}")
    return {"latency_ms": latency_ms, "throughput": throughput}


def _benchmark_speed_cnn(
    model:      nn.Module,
    x_time:     torch.Tensor,
    x_freq:     torch.Tensor,
    logger,
    n_runs:     int = 100,
):
    """CNN-specific benchmark — handles optional freq input."""
    model.eval()
    batch_size = x_time.shape[0]

    with torch.no_grad():
        for _ in range(10):
            _ = model(x_time, x_freq)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(x_time, x_freq)
        elapsed = time.perf_counter() - t0

    latency_ms = (elapsed / n_runs) * 1000
    throughput = (batch_size * n_runs) / elapsed
    msg = (f"CNN | Latency: {latency_ms:.2f} ms/batch "
           f"| Throughput: {throughput:.0f} samples/s")
    print(f"⚡ {msg}")
    logger.info(f"Speed | {msg}")
    return {"latency_ms": latency_ms, "throughput": throughput}


# ═════════════════════════════════════════════
# SECTION 6 — BASELINE vs ENHANCED COMPARISON
# ═════════════════════════════════════════════

def compare_baseline_enhanced(logger):
    """
    Loads saved history JSONs for baseline and enhanced runs.
    Prints final metric comparison table.
    Only runs if both JSON files exist.
    """
    if not EVAL_COMPARISON:
        return

    b_path = os.path.join(LOG_DIR, "mlp_baseline_history.json")
    e_path = os.path.join(LOG_DIR, f"mlp_{EXPERIMENT_TAG}_history.json")

    if not os.path.exists(b_path) or not os.path.exists(e_path):
        print("⚠️  Comparison skipped — history JSONs not found. "
              "Run with RUN_BASELINE=True first.")
        return

    with open(b_path) as f: baseline = json.load(f)
    with open(e_path) as f: enhanced = json.load(f)

    def last(d, k): return d[k][-1] if d.get(k) else float("nan")

    rows = [
        ("Val Loss",      last(baseline, "val_loss"),  last(enhanced, "val_loss")),
        ("Val Accuracy",  last(baseline, "val_acc"),   last(enhanced, "val_acc")),
        ("Val F1",        last(baseline, "val_f1"),    last(enhanced, "val_f1")),
        ("Val Precision", last(baseline, "val_precision"), last(enhanced, "val_precision")),
        ("Val Recall",    last(baseline, "val_recall"),    last(enhanced, "val_recall")),
    ]

    print("\n" + "=" * 52)
    print(f"{'Metric':<18} {'Baseline':>10} {'Enhanced':>10} {'Δ':>10}")
    print("=" * 52)
    for metric, b_val, e_val in rows:
        delta = e_val - b_val
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"{metric:<18} {b_val:>10.4f} {e_val:>10.4f} {delta:>+9.4f}{arrow}")
    print("=" * 52)
    logger.info("Baseline vs Enhanced comparison printed.")


# ═════════════════════════════════════════════
# SECTION 7 — MAIN
# ═════════════════════════════════════════════

if __name__ == "__main__":
    device = get_device()
    logger = get_logger("module1_eval", LOG_DIR)

    logger.info("STELLARIS-DNet | Module 1 Evaluation Started")
    logger.info(f"Experiment tag : {EXPERIMENT_TAG}")

    results = {}
    results["mlp"] = evaluate_mlp(device, logger)
    results["cnn"] = evaluate_cnn(device, logger)
    results["ae"]  = evaluate_autoencoder(device, logger)

    # Baseline vs enhanced comparison table
    compare_baseline_enhanced(logger)

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("MODULE 1 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"MLP  — Acc: {results['mlp']['acc']:.4f} | "
          f"AUC: {results['mlp']['auc']:.4f} | "
          f"AP: {results['mlp']['ap']:.4f} | "
          f"F1: {results['mlp']['f1']:.4f}")
    print(f"CNN  — Acc: {results['cnn']['acc']:.4f} | "
          f"Macro F1: {results['cnn']['f1']:.4f}")
    print(f"AE   — Threshold (pct): {results['ae']['threshold_pct']:.6f} | "
          f"Threshold (z): {results['ae']['threshold_z']:.6f}")
    print(f"       Active ({AE_THRESHOLD_METHOD}): {results['ae']['threshold']:.6f}")
    print("=" * 60)
    print(f"✅ Evaluation complete. All plots → {LOG_DIR}")

    logger.info("Module 1 Evaluation Complete")