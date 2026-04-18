"""
module1/evaluate.py
STELLARIS-DNet — Module 1 Evaluation
Evaluates: MLP + 1D CNN + Autoencoder (sets anomaly threshold)
Run AFTER train.py: python module1/evaluate.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module1.config  import *
from module1.dataset import load_htru2, load_pulse_profiles, load_autoencoder_data
from module1.model   import PulsarMLP, PulsarCNN, PulsarAutoencoder
from core.utils      import get_device, get_logger


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_model(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Run module1/train.py first."
        )
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)   # ← this line was missing
    model.eval()
    print(f"✅ Loaded: {path}")
    return model


def save_plot(fig, name: str, log_dir: str = LOG_DIR):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, name)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Plot saved → {path}")


# ─────────────────────────────────────────────
# 1. EVALUATE MLP
# ─────────────────────────────────────────────
def evaluate_mlp(device, logger):
    logger.info("=" * 50)
    logger.info("Evaluating MLP — HTRU2 Binary Classifier")
    logger.info("=" * 50)

    _, _, test_loader, _, pos_weight = load_htru2()
    model = load_model(PulsarMLP(), 
                       os.path.join(CHECKPOINT_DIR, "mlp_best.pt"), device)

    all_preds  = []
    all_probs  = []
    all_labels = []

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

    # Metrics
    auc = roc_auc_score(all_labels, all_probs)
    f1  = f1_score(all_labels, all_preds)
    acc = (all_preds == all_labels).mean()

    logger.info(f"MLP Test Accuracy : {acc:.4f}")
    logger.info(f"MLP ROC-AUC       : {auc:.4f}")
    logger.info(f"MLP F1 Score      : {f1:.4f}")
    logger.info("\n" + classification_report(
        all_labels, all_preds,
        target_names=HTRU2_CLASSES
    ))

    print(f"\n── MLP Results ──")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(classification_report(all_labels, all_preds,
                                 target_names=HTRU2_CLASSES))

    # Confusion matrix plot
    cm  = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(HTRU2_CLASSES); ax.set_yticklabels(HTRU2_CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("MLP — Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    save_plot(fig, "mlp_confusion_matrix.png")

    return {"acc": acc, "auc": auc, "f1": f1}


# ─────────────────────────────────────────────
# 2. EVALUATE 1D CNN
# ─────────────────────────────────────────────
def evaluate_cnn(device, logger):
    logger.info("=" * 50)
    logger.info("Evaluating 1D CNN — Pulsar Subtype Classifier")
    logger.info("=" * 50)

    _, _, test_loader = load_pulse_profiles()
    model = load_model(PulsarCNN(),
                       os.path.join(CHECKPOINT_DIR, "cnn_best.pt"), device)

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X     = X.to(device)
            out   = model(X)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    acc = (all_preds == all_labels).mean()
    f1  = f1_score(all_labels, all_preds, average="macro")

    logger.info(f"CNN Test Accuracy : {acc:.4f}")
    logger.info(f"CNN Macro F1      : {f1:.4f}")
    logger.info("\n" + classification_report(
        all_labels, all_preds,
        target_names=PULSAR_CLASSES
    ))

    print(f"\n── CNN Results ──")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print(classification_report(all_labels, all_preds,
                                 target_names=PULSAR_CLASSES))

    # Confusion matrix
    cm  = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_PULSAR_CLASSES))
    ax.set_yticks(range(NUM_PULSAR_CLASSES))
    ax.set_xticklabels(PULSAR_CLASSES, rotation=30)
    ax.set_yticklabels(PULSAR_CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("CNN — Pulsar Subtype Confusion Matrix")
    for i in range(NUM_PULSAR_CLASSES):
        for j in range(NUM_PULSAR_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    save_plot(fig, "cnn_confusion_matrix.png")

    return {"acc": acc, "f1": f1}


# ─────────────────────────────────────────────
# 3. EVALUATE AUTOENCODER + SET THRESHOLD
# ─────────────────────────────────────────────
def evaluate_autoencoder(device, logger):
    logger.info("=" * 50)
    logger.info("Evaluating Autoencoder — Setting Magnetar Threshold")
    logger.info("=" * 50)

    _, val_loader = load_autoencoder_data()
    model = load_model(PulsarAutoencoder(),
                       os.path.join(CHECKPOINT_DIR, "ae_best.pt"), device)

    # Collect reconstruction errors on normal pulsar val set
    all_errors = []
    with torch.no_grad():
        for X in val_loader:
            X      = X.to(device)
            errors = model.reconstruction_error(X)
            all_errors.extend(errors.cpu().numpy())

    all_errors = np.array(all_errors)

    # Set threshold at 95th percentile
    threshold = float(np.percentile(all_errors, AE_ANOMALY_PERCENTILE))

    logger.info(f"AE Val errors — mean: {all_errors.mean():.6f} "
                f"std: {all_errors.std():.6f}")
    logger.info(f"Magnetar threshold ({AE_ANOMALY_PERCENTILE}th pct): {threshold:.6f}")

    print(f"\n── Autoencoder Results ──")
    print(f"Val errors  — mean: {all_errors.mean():.6f} | "
          f"std: {all_errors.std():.6f}")
    print(f"Magnetar threshold ({AE_ANOMALY_PERCENTILE}th percentile): "
          f"{threshold:.6f}")
    print(f"Signals above threshold flagged as magnetar candidates")

    # Save threshold for inference
    threshold_path = os.path.join(CHECKPOINT_DIR, "ae_threshold.npy")
    np.save(threshold_path, np.array([threshold]))
    print(f"💾 Threshold saved → {threshold_path}")

    # Error distribution plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_errors, bins=50, color="steelblue", alpha=0.7,
            label="Normal pulsar errors")
    ax.axvline(threshold, color="red", linestyle="--",
               label=f"Threshold ({AE_ANOMALY_PERCENTILE}th pct) = {threshold:.4f}")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Count")
    ax.set_title("Autoencoder — Reconstruction Error Distribution")
    ax.legend()
    save_plot(fig, "ae_error_distribution.png")

    return {"threshold": threshold, "mean_error": all_errors.mean()}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = get_device()
    logger = get_logger("module1_eval", LOG_DIR)

    logger.info("STELLARIS-DNet | Module 1 Evaluation Started")

    results = {}
    results["mlp"] = evaluate_mlp(device, logger)
    results["cnn"] = evaluate_cnn(device, logger)
    results["ae"]  = evaluate_autoencoder(device, logger)

    # Final summary
    print("\n" + "=" * 50)
    print("MODULE 1 EVALUATION SUMMARY")
    print("=" * 50)
    print(f"MLP  — Accuracy: {results['mlp']['acc']:.4f} | "
          f"AUC: {results['mlp']['auc']:.4f} | "
          f"F1: {results['mlp']['f1']:.4f}")
    print(f"CNN  — Accuracy: {results['cnn']['acc']:.4f} | "
          f"Macro F1: {results['cnn']['f1']:.4f}")
    print(f"AE   — Magnetar threshold: {results['ae']['threshold']:.6f}")
    print("=" * 50)
    print("✅ Module 1 evaluation complete.")
    print(f"   Plots saved to: {LOG_DIR}")

    logger.info("Module 1 Evaluation Complete")