"""
module3/evaluate.py
STELLARIS-DNet — Module 3 Evaluation (v2)
Metrics: classification + regression (mask-aware) + physics consistency

Plots: confusion matrix, HR diagram, regression scatter,
       physics residuals, confidence analysis, feature scales

v2 changes:
  - reg_mask applied to all regression metrics
  - Per-class regression breakdown (skips unsupervised target/class pairs)

Run AFTER train.py: python module3/evaluate.py
"""

import os
import sys
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
            f"Checkpoint not found: {path}\nRun module3/train.py first."
        )
    ckpt  = torch.load(path, map_location=device)
    model = StellarFTTransformer().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✅ Loaded checkpoint ← {path}")
    print(f"   Epoch: {ckpt['epoch']} | Val loss: {ckpt['loss']:.4f}")
    return model


def _collect_predictions(model, loader, device) -> dict:
    """Run full test set through model. Returns dict of numpy arrays."""
    all_logits, all_reg, all_labels, all_reg_true, all_mask = [], [], [], [], []

    with torch.no_grad():
        for X, y_class, y_reg, reg_mask in loader:
            X       = X.to(device)
            logits, reg_out, _ = model(X)
            all_logits.append(logits.cpu())
            all_reg.append(reg_out.cpu())
            all_labels.append(y_class)
            all_reg_true.append(y_reg)
            all_mask.append(reg_mask)

    logits   = torch.cat(all_logits).numpy()
    reg_pred = torch.cat(all_reg).numpy()
    labels   = torch.cat(all_labels).numpy()
    reg_true = torch.cat(all_reg_true).numpy()
    mask     = torch.cat(all_mask).numpy()

    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    pred  = logits.argmax(axis=1)

    return {
        "logits":    logits,
        "probs":     probs,
        "pred":      pred,
        "true":      labels,
        "reg_pred":  reg_pred,
        "reg_true":  reg_true,
        "reg_mask":  mask,
    }


# ─────────────────────────────────────────────
# 1. CLASSIFICATION METRICS
# ─────────────────────────────────────────────
def eval_classification(results, logger) -> dict:
    pred, true, probs = results["pred"], results["true"], results["probs"]

    acc      = accuracy_score(true, pred)
    macro_f1 = f1_score(true, pred, average="macro")
    try:
        auc = roc_auc_score(
            np.eye(NUM_STELLAR_CLASSES)[true], probs,
            average="macro", multi_class="ovr"
        )
    except Exception:
        auc = float("nan")

    report = classification_report(true, pred, target_names=STELLAR_CLASSES)

    print("\n── Classification Results ──")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(f"  Macro AUC: {auc:.4f}")
    print(report)
    logger.info(f"Acc={acc:.4f} F1={macro_f1:.4f} AUC={auc:.4f}")

    return {"acc": acc, "f1": macro_f1, "auc": auc}


# ─────────────────────────────────────────────
# 2. CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion_matrix(results):
    cm   = confusion_matrix(results["true"], results["pred"])
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_STELLAR_CLASSES))
    ax.set_yticks(range(NUM_STELLAR_CLASSES))
    short = ["MS", "RG", "WD", "NS", "QSO"]
    ax.set_xticklabels(short); ax.set_yticklabels(short)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Module 3 — Confusion Matrix (row-normalised)")

    for i in range(NUM_STELLAR_CLASSES):
        for j in range(NUM_STELLAR_CLASSES):
            color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({norm[i,j]:.2f})",
                    ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, label="Fraction")
    fig.tight_layout()
    _save(fig, "confusion_matrix.png")


# ─────────────────────────────────────────────
# 3. REGRESSION METRICS — mask-aware, per-class
# ─────────────────────────────────────────────
def eval_regression(results, logger) -> dict:
    reg_pred = results["reg_pred"]
    reg_true = results["reg_true"]
    mask     = results["reg_mask"]
    true_cls = results["true"]

    print("\n── Regression Results (log10-scale, mask-aware) ──")
    print(f"{'Target':12s} {'Coverage':>10s} {'MAE':>8s} {'R²':>8s}")
    print("-" * 42)

    metrics = {}
    for i, name in enumerate(REGRESSION_TARGETS):
        m = mask[:, i].astype(bool)
        if m.sum() == 0:
            print(f"  {name:12s} {0:>10.0f} {'-':>8s} {'-':>8s}")
            metrics[name] = {"coverage": 0, "mae": np.nan, "r2": np.nan}
            continue
        diff = reg_pred[m, i] - reg_true[m, i]
        mae = np.abs(diff).mean()
        ss_res = (diff ** 2).sum()
        ss_tot = ((reg_true[m, i] - reg_true[m, i].mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        cov = m.sum() / len(m) * 100
        print(f"  {name:12s} {cov:>9.1f}% {mae:>8.4f} {r2:>8.4f}")
        logger.info(f"  {name}: coverage={cov:.1f}% MAE={mae:.4f} R²={r2:.4f}")
        metrics[name] = {"coverage": cov, "mae": mae, "r2": r2}

    # ── Per-class breakdown ──
    print("\nPer-class regression MAE (where supervised):")
    print(f"{'Class':15s}", end="")
    for name in REGRESSION_TARGETS:
        short = name.replace("log_", "")
        print(f" {short:>10s}", end="")
    print()
    print("-" * (15 + 11 * NUM_REGRESSION))

    for cls in range(NUM_STELLAR_CLASSES):
        c_mask = (true_cls == cls)
        if c_mask.sum() == 0:
            continue
        print(f"  {STELLAR_CLASSES[cls]:13s}", end="")
        for j in range(NUM_REGRESSION):
            cm_mask = c_mask & mask[:, j].astype(bool)
            if cm_mask.sum() == 0:
                print(f" {'-':>10s}", end="")
            else:
                mae_c = np.abs(reg_pred[cm_mask, j] - reg_true[cm_mask, j]).mean()
                print(f" {mae_c:>10.4f}", end="")
        print()

    return metrics


# ─────────────────────────────────────────────
# 4. REGRESSION SCATTER PLOTS — supervised only
# ─────────────────────────────────────────────
def plot_regression_scatter(results):
    reg_pred = results["reg_pred"]
    reg_true = results["reg_true"]
    mask     = results["reg_mask"]
    true_cls = results["true"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_STELLAR_CLASSES))

    for i, (name, ax) in enumerate(zip(REGRESSION_TARGETS, axes)):
        m = mask[:, i].astype(bool)
        if m.sum() == 0:
            ax.text(0.5, 0.5, "No supervised samples",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name)
            continue

        cls_arr   = true_cls[m]
        cls_color = colors[cls_arr]
        ax.scatter(reg_true[m, i], reg_pred[m, i],
                   c=cls_color, alpha=0.3, s=8)
        lo = min(reg_true[m, i].min(), reg_pred[m, i].min())
        hi = max(reg_true[m, i].max(), reg_pred[m, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect")
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f"{name}  (n={m.sum():,})")
        mae = np.abs(reg_pred[m, i] - reg_true[m, i]).mean()
        ax.text(0.05, 0.92, f"MAE={mae:.3f}",
                transform=ax.transAxes, fontsize=9)

    from matplotlib.patches import Patch
    handles = [Patch(color=colors[c], label=STELLAR_CLASSES[c].replace("_", " "))
               for c in range(NUM_STELLAR_CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Module 3 — Regression: Predicted vs True (supervised only)")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "regression_scatter.png")


# ─────────────────────────────────────────────
# 5. HR DIAGRAM
# ─────────────────────────────────────────────
def plot_hr_diagram(results):
    reg_pred = results["reg_pred"]
    pred_cls = results["pred"]

    log_teff_pred = reg_pred[:, 2]
    log_lum_pred  = reg_pred[:, 1]

    colors = plt.cm.tab10(np.linspace(0, 1, NUM_STELLAR_CLASSES))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, cls_arr, title in zip(
        axes, [pred_cls, results["true"]],
        ["Predicted class", "True class"]
    ):
        for cls in range(NUM_STELLAR_CLASSES):
            m = cls_arr == cls
            if m.sum() == 0:
                continue
            ax.scatter(log_teff_pred[m], log_lum_pred[m],
                       c=[colors[cls]], alpha=0.3, s=6,
                       label=STELLAR_CLASSES[cls].replace("_", " "))
        ax.invert_xaxis()
        ax.set_xlabel("log₁₀(Teff / K)")
        ax.set_ylabel("log₁₀(L / L☉)")
        ax.set_title(f"HR Diagram — {title}")
        ax.legend(fontsize=8, markerscale=2)

    fig.suptitle("Module 3 — Hertzsprung-Russell Diagram")
    fig.tight_layout()
    _save(fig, "hr_diagram.png")


# ─────────────────────────────────────────────
# 6. PHYSICS RESIDUALS (SB law) — MS+RG+WD only
# ─────────────────────────────────────────────
def plot_physics_residuals(results):
    reg_pred = results["reg_pred"]
    true_cls = results["true"]
    mask     = results["reg_mask"]

    L_pred = 10 ** reg_pred[:, 1]
    T_pred = 10 ** reg_pred[:, 2]
    R_pred = 10 ** reg_pred[:, 3]

    R_si    = R_pred * R_SUN
    L_sb_si = 4 * np.pi * R_si**2 * SIGMA_SB * T_pred**4
    L_sb    = L_sb_si / L_SUN

    log_L_pred = np.log10(L_pred.clip(1e-10, 1e20))
    log_L_sb   = np.log10(L_sb.clip(1e-10, 1e20))
    residual   = log_L_pred - log_L_sb

    # SB only valid for MS, RG, WD — AND requires all 3 of L, T, R supervised
    sb_supervised = (
        mask[:, 1].astype(bool) &  # log_lum
        mask[:, 2].astype(bool) &  # log_teff
        mask[:, 3].astype(bool)    # log_radius
    )
    valid = sb_supervised & np.isin(true_cls, [0, 1, 2])

    if valid.sum() == 0:
        print("\n⚠️  No valid samples for SB physics check (need L,T,R all supervised)")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, NUM_STELLAR_CLASSES))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for cls in [0, 1, 2]:
        m = (true_cls == cls) & valid
        if m.sum() == 0:
            continue
        ax.scatter(log_L_sb[m], log_L_pred[m], c=[colors[cls]],
                   alpha=0.4, s=8,
                   label=STELLAR_CLASSES[cls].replace("_", " "))
    lo = min(log_L_sb[valid].min(), log_L_pred[valid].min())
    hi = max(log_L_sb[valid].max(), log_L_pred[valid].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect")
    ax.set_xlabel("log L from SB law (R, T)")
    ax.set_ylabel("log L predicted")
    ax.set_title(f"L_pred vs L_SB (n={valid.sum():,})")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(residual[valid], bins=60, color="steelblue",
            alpha=0.8, edgecolor="none")
    ax.axvline(0, color="red", lw=1.5, ls="--", label="Zero")
    ax.axvline(residual[valid].mean(), color="orange", lw=1.5, ls="--",
               label=f"Mean={residual[valid].mean():.3f}")
    ax.set_xlabel("log L_pred − log L_SB")
    ax.set_ylabel("Count")
    ax.set_title("SB Residual Distribution")
    ax.legend(fontsize=9)

    fig.suptitle("Module 3 — Physics Residuals (Stefan-Boltzmann)")
    fig.tight_layout()
    _save(fig, "physics_residuals.png")

    print(f"\n── Physics Residuals (MS+RG+WD, n={valid.sum():,}) ──")
    print(f"  Mean SB residual: {residual[valid].mean():.4f}")
    print(f"  MAE  SB residual: {np.abs(residual[valid]).mean():.4f}")
    print(f"  (Ideal: mean ≈ 0, MAE < 0.5)")


# ─────────────────────────────────────────────
# 7. CONFIDENCE ANALYSIS
# ─────────────────────────────────────────────
def plot_confidence_analysis(results):
    probs    = results["probs"]
    pred     = results["pred"]
    true     = results["true"]
    confidence = probs.max(axis=1)
    correct    = (pred == true)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(confidence[correct],  bins=40, alpha=0.7, color="steelblue",
            label=f"Correct ({correct.sum():,})")
    ax.hist(confidence[~correct], bins=40, alpha=0.7, color="tomato",
            label=f"Wrong ({(~correct).sum():,})")
    ax.set_xlabel("Max Softmax Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()

    ax = axes[1]
    bin_edges = np.linspace(0, 1, 11)
    bin_acc, bin_conf = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (confidence >= lo) & (confidence < hi)
        if m.sum() > 0:
            bin_acc.append(correct[m].mean())
            bin_conf.append(confidence[m].mean())
    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="Perfect calibration")
    ax.plot(bin_conf, bin_acc, "bo-", markersize=6, label="Model")
    ax.fill_between(bin_conf, bin_acc, bin_conf, alpha=0.15, color="red")
    ax.set_xlabel("Mean Confidence")
    ax.set_ylabel("Fraction Correct")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle("Module 3 — Confidence Analysis")
    fig.tight_layout()
    _save(fig, "confidence_analysis.png")

    overconfident = (~correct) & (confidence > 0.9)
    print(f"\n── Confidence Analysis ──")
    print(f"  High-conf correct:   {(correct & (confidence > 0.9)).sum():,}")
    print(f"  High-conf WRONG:     {overconfident.sum():,}  ← overconfident")
    print(f"  Mean conf (correct): {confidence[correct].mean():.3f}")
    print(f"  Mean conf (wrong):   {confidence[~correct].mean():.3f}")


# ─────────────────────────────────────────────
# 8. FEATURE SCALE ANALYSIS
# ─────────────────────────────────────────────
def plot_feature_scales(model):
    scales = model.feature_scale.detach().cpu().numpy()

    # Two colours: physical features vs validity flags
    colors = ["steelblue"] * NUM_PHYSICAL + ["darkorange"] * len(VALIDITY_FLAGS)

    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(range(NUM_FEATURES), scales,
                  color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(NUM_FEATURES))
    ax.set_xticklabels(FEATURE_NAMES, rotation=40, ha="right", fontsize=8)
    ax.axhline(1.0, color="red", lw=1.2, ls="--", label="Init=1.0")
    ax.set_ylabel("Learned Scale")
    ax.set_title("Module 3 — Learned Feature Scaling (blue=physical, orange=validity)")
    ax.legend(fontsize=9)

    for bar, val in zip(bars, scales):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    _save(fig, "feature_scales.png")

    print("\n── Learned Feature Scales ──")
    print("Physical features:")
    for name, val in zip(PHYSICAL_FEATURES, scales[:NUM_PHYSICAL]):
        bar = "█" * max(1, int(abs(val) * 10))
        print(f"  {name:13s}: {val:.3f}  {bar}")
    print("Validity flags:")
    for name, val in zip(VALIDITY_FLAGS, scales[NUM_PHYSICAL:]):
        bar = "█" * max(1, int(abs(val) * 10))
        print(f"  {name:15s}: {val:.3f}  {bar}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Module 3 Evaluation v2")
    print("=" * 55)

    device = get_device()
    logger = get_logger("module3_eval", LOG_DIR)
    logger.info("Module 3 Evaluation v2 Started")

    model = _load_model(device)
    _, _, test_loader, _, _ = load_stellar_data()

    print("\nRunning inference on test set...")
    results = _collect_predictions(model, test_loader, device)
    print(f"Test samples: {len(results['true']):,}")

    cls_metrics = eval_classification(results, logger)
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(results)

    reg_metrics = eval_regression(results, logger)
    print("\nGenerating regression scatter plots...")
    plot_regression_scatter(results)

    print("\nGenerating HR diagram...")
    plot_hr_diagram(results)

    print("\nGenerating physics residual plots...")
    plot_physics_residuals(results)

    print("\nGenerating confidence analysis...")
    plot_confidence_analysis(results)

    print("\nGenerating feature scale plot...")
    plot_feature_scales(model)

    print("\n" + "=" * 55)
    print("MODULE 3 EVALUATION v2 SUMMARY")
    print("=" * 55)
    print(f"Accuracy:     {cls_metrics['acc']:.4f}")
    print(f"Macro F1:     {cls_metrics['f1']:.4f}")
    print(f"Macro AUC:    {cls_metrics['auc']:.4f}")
    print()
    for name, m in reg_metrics.items():
        if not np.isnan(m['mae']):
            print(f"{name:12s}: cov={m['coverage']:.1f}% | "
                  f"MAE={m['mae']:.4f} | R²={m['r2']:.4f}")
    print()
    print(f"Plots saved to: {LOG_DIR}/")
    print("=" * 55)

    logger.info(f"Done. Acc={cls_metrics['acc']:.4f} F1={cls_metrics['f1']:.4f}")