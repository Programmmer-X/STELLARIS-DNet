"""
module3/evaluate.py
STELLARIS-DNet — Module 3 Evaluation
Metrics: classification + regression + physics consistency
Plots:   confusion matrix, HR diagram, regression scatter,
         physics residuals, confidence analysis
Run AFTER train.py: python module3/evaluate.py
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module3.config  import *
from module3.dataset import load_stellar_data
from module3.model   import StellarFTTransformer
from core.utils      import get_device, get_logger
from core.physics_loss import physics_consistency_score


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _save(fig, name: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, name)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved → {path}")


def _load_model(device: torch.device) -> StellarFTTransformer:
    path = os.path.join(CHECKPOINT_DIR, "module3_best.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Run module3/train.py first."
        )
    ckpt  = torch.load(path, map_location=device)
    model = StellarFTTransformer().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✅ Loaded checkpoint ← {path}")
    print(f"   Epoch: {ckpt['epoch']} | Val loss: {ckpt['loss']:.4f}")
    return model


def _collect_predictions(
    model:  StellarFTTransformer,
    loader: torch.utils.data.DataLoader,
    device: torch.device
) -> dict:
    """
    Run full test set through model.
    Returns dict of numpy arrays for all metrics.
    """
    all_logits   = []
    all_reg      = []
    all_labels   = []
    all_reg_true = []
    all_enc      = []

    with torch.no_grad():
        for X, y_class, y_reg, _ in loader:
            X       = X.to(device)
            logits, reg_out, enc = model(X)
            all_logits.append(logits.cpu())
            all_reg.append(reg_out.cpu())
            all_labels.append(y_class)
            all_reg_true.append(y_reg)
            all_enc.append(enc.cpu())

    logits   = torch.cat(all_logits).numpy()
    reg_pred = torch.cat(all_reg).numpy()
    labels   = torch.cat(all_labels).numpy()
    reg_true = torch.cat(all_reg_true).numpy()
    enc      = torch.cat(all_enc).numpy()

    probs       = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    pred_labels = logits.argmax(axis=1)

    return {
        "logits":    logits,
        "probs":     probs,
        "pred":      pred_labels,
        "true":      labels,
        "reg_pred":  reg_pred,
        "reg_true":  reg_true,
        "enc":       enc,
    }


# ─────────────────────────────────────────────
# 1. CLASSIFICATION METRICS
# ─────────────────────────────────────────────
def eval_classification(results: dict, logger) -> dict:
    pred  = results["pred"]
    true  = results["true"]
    probs = results["probs"]

    acc      = accuracy_score(true, pred)
    macro_f1 = f1_score(true, pred, average="macro")

    # Per-class AUC (one-vs-rest)
    try:
        auc = roc_auc_score(
            np.eye(NUM_STELLAR_CLASSES)[true],
            probs, average="macro", multi_class="ovr"
        )
    except Exception:
        auc = float("nan")

    report = classification_report(
        true, pred, target_names=STELLAR_CLASSES
    )

    print("\n── Classification Results ──")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"  Macro AUC: {auc:.4f}")
    print(report)

    logger.info(f"Accuracy={acc:.4f} | F1={macro_f1:.4f} | AUC={auc:.4f}")
    logger.info("\n" + report)

    return {"acc": acc, "f1": macro_f1, "auc": auc}


# ─────────────────────────────────────────────
# 2. CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion_matrix(results: dict):
    cm   = confusion_matrix(results["true"], results["pred"])
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_STELLAR_CLASSES))
    ax.set_yticks(range(NUM_STELLAR_CLASSES))
    short = ["MS", "RG", "WD", "NS", "QSO"]
    ax.set_xticklabels(short, fontsize=10)
    ax.set_yticklabels(short, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Module 3 — Confusion Matrix (row-normalised)", fontsize=13)

    for i in range(NUM_STELLAR_CLASSES):
        for j in range(NUM_STELLAR_CLASSES):
            color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({norm[i,j]:.2f})",
                    ha="center", va="center",
                    fontsize=8, color=color)
    fig.colorbar(im, ax=ax, label="Fraction")
    fig.tight_layout()
    _save(fig, "confusion_matrix.png")


# ─────────────────────────────────────────────
# 3. REGRESSION METRICS
# ─────────────────────────────────────────────
def eval_regression(results: dict, logger) -> dict:
    reg_pred = results["reg_pred"]
    reg_true = results["reg_true"]

    print("\n── Regression Results (log10-scale) ──")
    metrics = {}
    for i, name in enumerate(REGRESSION_TARGETS):
        mae = np.abs(reg_pred[:, i] - reg_true[:, i]).mean()
        ss_res = ((reg_pred[:, i] - reg_true[:, i]) ** 2).sum()
        ss_tot = ((reg_true[:, i] - reg_true[:, i].mean()) ** 2).sum()
        r2  = 1 - ss_res / (ss_tot + 1e-10)
        print(f"  {name:12s} | MAE={mae:.4f} | R²={r2:.4f}")
        logger.info(f"  {name}: MAE={mae:.4f} R²={r2:.4f}")
        metrics[name] = {"mae": mae, "r2": r2}

    return metrics


# ─────────────────────────────────────────────
# 4. REGRESSION SCATTER PLOTS
# ─────────────────────────────────────────────
def plot_regression_scatter(results: dict):
    reg_pred = results["reg_pred"]
    reg_true = results["reg_true"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_STELLAR_CLASSES))
    class_colors = colors[results["true"]]

    for i, (name, ax) in enumerate(zip(REGRESSION_TARGETS, axes)):
        ax.scatter(
            reg_true[:, i], reg_pred[:, i],
            c=class_colors, alpha=0.3, s=8
        )
        lo = min(reg_true[:, i].min(), reg_pred[:, i].min())
        hi = max(reg_true[:, i].max(), reg_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect")
        ax.set_xlabel(f"True {name}", fontsize=10)
        ax.set_ylabel(f"Predicted {name}", fontsize=10)
        ax.set_title(name, fontsize=11)
        mae = np.abs(reg_pred[:, i] - reg_true[:, i]).mean()
        ax.text(0.05, 0.92, f"MAE={mae:.3f}",
                transform=ax.transAxes, fontsize=9)

    # Legend for classes
    from matplotlib.patches import Patch
    handles = [Patch(color=colors[c], label=STELLAR_CLASSES[c].replace("_", " "))
               for c in range(NUM_STELLAR_CLASSES)]
    fig.legend(handles=handles, loc="lower center",
               ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Module 3 — Regression: Predicted vs True", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "regression_scatter.png")


# ─────────────────────────────────────────────
# 5. HR DIAGRAM
# Predicted log_teff (x, reversed) vs log_lum (y)
# coloured by predicted class
# ─────────────────────────────────────────────
def plot_hr_diagram(results: dict):
    reg_pred = results["reg_pred"]
    pred_cls = results["pred"]

    log_teff_pred   = reg_pred[:, 2]   # log10(Teff)
    log_lum_pred    = reg_pred[:, 1]   # log10(L)

    colors = plt.cm.tab10(np.linspace(0, 1, NUM_STELLAR_CLASSES))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, cls_arr, title in zip(
        axes,
        [pred_cls, results["true"]],
        ["Predicted class", "True class"]
    ):
        for cls in range(NUM_STELLAR_CLASSES):
            mask = cls_arr == cls
            if mask.sum() == 0:
                continue
            ax.scatter(
                log_teff_pred[mask],
                log_lum_pred[mask],
                c=[colors[cls]], alpha=0.3, s=6,
                label=STELLAR_CLASSES[cls].replace("_", " ")
            )
        ax.invert_xaxis()   # HR diagram: hot stars on left
        ax.set_xlabel("log₁₀(Teff / K)", fontsize=11)
        ax.set_ylabel("log₁₀(L / L☉)", fontsize=11)
        ax.set_title(f"HR Diagram — {title}", fontsize=12)
        ax.legend(fontsize=8, markerscale=2)

    fig.suptitle("Module 3 — Hertzsprung-Russell Diagram", fontsize=13)
    fig.tight_layout()
    _save(fig, "hr_diagram.png")


# ─────────────────────────────────────────────
# 6. PHYSICS RESIDUALS
# Compare L_pred vs L from SB law (R, T)
# Only for MS, RG, WD samples
# ─────────────────────────────────────────────
def plot_physics_residuals(results: dict):
    reg_pred = results["reg_pred"]
    true_cls = results["true"]

    # Reconstruct linear-scale predictions
    L_pred = 10 ** reg_pred[:, 1]
    T_pred = 10 ** reg_pred[:, 2]
    R_pred = 10 ** reg_pred[:, 3]

    # Expected L from Stefan-Boltzmann
    R_si     = R_pred * R_SUN
    L_sb_si  = 4 * np.pi * R_si**2 * SIGMA_SB * T_pred**4
    L_sb     = L_sb_si / L_SUN

    log_L_pred = np.log10(L_pred.clip(1e-10, 1e20))
    log_L_sb   = np.log10(L_sb.clip(1e-10, 1e20))
    residual   = log_L_pred - log_L_sb

    # Only plot MS, RG, WD (classes 0, 1, 2) — SB valid for these
    mask   = np.isin(true_cls, [0, 1, 2])
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_STELLAR_CLASSES))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter: L_pred vs L_SB
    ax = axes[0]
    for cls in [0, 1, 2]:
        m = (true_cls == cls) & mask
        ax.scatter(log_L_sb[m], log_L_pred[m],
                   c=[colors[cls]], alpha=0.4, s=8,
                   label=STELLAR_CLASSES[cls].replace("_", " "))
    lo = min(log_L_sb[mask].min(), log_L_pred[mask].min())
    hi = max(log_L_sb[mask].max(), log_L_pred[mask].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect")
    ax.set_xlabel("log L from SB law (R, T)", fontsize=10)
    ax.set_ylabel("log L predicted", fontsize=10)
    ax.set_title("Physics Residual: L_pred vs L_SB", fontsize=11)
    ax.legend(fontsize=8)

    # Residual histogram
    ax = axes[1]
    ax.hist(residual[mask], bins=60, color="steelblue",
            alpha=0.8, edgecolor="none")
    ax.axvline(0, color="red", lw=1.5, linestyle="--", label="Zero residual")
    ax.axvline(residual[mask].mean(), color="orange",
               lw=1.5, linestyle="--", label=f"Mean={residual[mask].mean():.3f}")
    ax.set_xlabel("log L_pred − log L_SB", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("SB Residual Distribution (MS+RG+WD)", fontsize=11)
    ax.legend(fontsize=9)

    fig.suptitle("Module 3 — Physics Residuals (Stefan-Boltzmann)", fontsize=13)
    fig.tight_layout()
    _save(fig, "physics_residuals.png")

    mae_residual = np.abs(residual[mask]).mean()
    print(f"\n── Physics Residuals (MS+RG+WD) ──")
    print(f"  Mean SB residual: {residual[mask].mean():.4f}")
    print(f"  MAE  SB residual: {mae_residual:.4f}")
    print("  (Ideal: mean ≈ 0, MAE < 0.5)")


# ─────────────────────────────────────────────
# 7. CONFIDENCE ANALYSIS
# ─────────────────────────────────────────────
def plot_confidence_analysis(results: dict):
    probs     = results["probs"]
    pred      = results["pred"]
    true      = results["true"]
    confidence = probs.max(axis=1)
    correct    = (pred == true)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confidence histogram — correct vs incorrect
    ax = axes[0]
    ax.hist(confidence[correct],  bins=40, alpha=0.7,
            color="steelblue", label=f"Correct ({correct.sum():,})")
    ax.hist(confidence[~correct], bins=40, alpha=0.7,
            color="tomato",    label=f"Wrong ({(~correct).sum():,})")
    ax.set_xlabel("Max Softmax Confidence", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Confidence Distribution", fontsize=12)
    ax.legend(fontsize=10)

    # Calibration curve (reliability diagram)
    ax = axes[1]
    bin_edges = np.linspace(0, 1, 11)
    bin_acc, bin_conf = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() > 0:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(confidence[mask].mean())

    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="Perfect calibration")
    ax.plot(bin_conf, bin_acc, "bo-", markersize=6, label="Model")
    ax.fill_between(bin_conf, bin_acc, bin_conf, alpha=0.15, color="red")
    ax.set_xlabel("Mean Confidence", fontsize=11)
    ax.set_ylabel("Fraction Correct", fontsize=11)
    ax.set_title("Reliability Diagram (Calibration)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle("Module 3 — Confidence Analysis", fontsize=13)
    fig.tight_layout()
    _save(fig, "confidence_analysis.png")

    # Flag overconfident wrong predictions
    overconfident = (~correct) & (confidence > 0.9)
    print(f"\n── Confidence Analysis ──")
    print(f"  High-conf correct:   {(correct & (confidence > 0.9)).sum():,}")
    print(f"  High-conf WRONG:     {overconfident.sum():,}  ← dangerous overconfidence")
    print(f"  Mean conf (correct): {confidence[correct].mean():.3f}")
    print(f"  Mean conf (wrong):   {confidence[~correct].mean():.3f}")


# ─────────────────────────────────────────────
# 8. FEATURE SCALE ANALYSIS
# Show what the model learned to weight
# ─────────────────────────────────────────────
def plot_feature_scales(model: StellarFTTransformer):
    scales = model.feature_scale.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(NUM_FEATURES), scales,
                  color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(NUM_FEATURES))
    ax.set_xticklabels(FEATURE_NAMES, rotation=30, ha="right", fontsize=10)
    ax.axhline(1.0, color="red", lw=1.2, linestyle="--", label="Init=1.0")
    ax.set_ylabel("Learned Scale", fontsize=11)
    ax.set_title("Module 3 — Learned Feature Scaling", fontsize=12)
    ax.legend(fontsize=9)

    for bar, val in zip(bars, scales):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _save(fig, "feature_scales.png")

    print("\n── Learned Feature Scales ──")
    for name, val in zip(FEATURE_NAMES, scales):
        bar = "█" * int(abs(val) * 10)
        print(f"  {name:12s}: {val:.3f}  {bar}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Module 3 Evaluation")
    print("=" * 55)

    device = get_device()
    logger = get_logger("module3_eval", LOG_DIR)
    logger.info("Module 3 Evaluation Started")

    # Load model
    model = _load_model(device)

    # Load data — only test loader needed
    _, _, test_loader, _, _ = load_stellar_data()

    # Collect all predictions
    print("\nRunning inference on test set...")
    results = _collect_predictions(model, test_loader, device)
    print(f"Test samples: {len(results['true']):,}")

    # 1. Classification
    cls_metrics = eval_classification(results, logger)

    # 2. Confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(results)

    # 3. Regression
    reg_metrics = eval_regression(results, logger)

    # 4. Regression scatter
    print("\nGenerating regression scatter plots...")
    plot_regression_scatter(results)

    # 5. HR diagram
    print("\nGenerating HR diagram...")
    plot_hr_diagram(results)

    # 6. Physics residuals
    print("\nGenerating physics residual plots...")
    plot_physics_residuals(results)

    # 7. Confidence analysis
    print("\nGenerating confidence analysis...")
    plot_confidence_analysis(results)

    # 8. Feature scales
    print("\nGenerating feature scale plot...")
    plot_feature_scales(model)

    # ── Final Summary ─────────────────────────
    print("\n" + "=" * 55)
    print("MODULE 3 EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Accuracy:     {cls_metrics['acc']:.4f}")
    print(f"Macro F1:     {cls_metrics['f1']:.4f}")
    print(f"Macro AUC:    {cls_metrics['auc']:.4f}")
    print()
    for name, m in reg_metrics.items():
        print(f"{name:12s}: MAE={m['mae']:.4f} | R²={m['r2']:.4f}")
    print()
    print(f"Plots saved to: {LOG_DIR}/")
    print("=" * 55)

    logger.info(f"Evaluation complete. "
                f"Acc={cls_metrics['acc']:.4f} "
                f"F1={cls_metrics['f1']:.4f}")