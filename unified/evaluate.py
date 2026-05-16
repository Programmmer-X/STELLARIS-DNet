"""
unified/evaluate.py
STELLARIS-DNet — Unified Fusion Evaluation

Per-head evaluation:
  - Stellar classification: accuracy, F1, confusion matrix, AUC
  - Pulsar detection: accuracy, F1, ROC-AUC
  - Pulsar subtype: accuracy, macro F1
  - Radio morphology: accuracy, F1
  - GW detection: accuracy, F1, ROC-AUC
  - Anomaly: AUC (if labels available)
  - Regression: MAE, R² per target
  - Physics consistency: Stefan-Boltzmann residual

Generates: per-head plots + summary table + JSON report
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_auc_score, classification_report, r2_score,
    mean_absolute_error,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified.config import (
    CHECKPOINT_DIR, LOG_DIR, FUSED_DIM,
    STELLAR_CLASS_NAMES, PULSAR_SUBTYPE_NAMES, RADIO_CLASS_NAMES,
    REG_TARGET_NAMES, NUM_STELLAR_CLASSES, NUM_REG_TARGETS,
    MODALITY_ORDER, NUM_ENCODERS, SEED,
    HEAD_VALIDITY, MODALITY_INDEX,
)
from unified.fusion_model import UnifiedModel
from unified.dataset import get_unified_loaders, IGNORE_LABEL
from core.utils import set_seed, get_device


# ═════════════════════════════════════════════════════════════
# 1. COLLECT PREDICTIONS
# ═════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_predictions(model: UnifiedModel, loader, device) -> dict:
    """
    Run model on full loader, collect per-head predictions + labels.

    Returns dict with per-head arrays, filtered to valid samples only.
    """
    model.eval()

    # Accumulators
    acc = {
        "stellar_cls_pred": [], "stellar_cls_true": [], "stellar_cls_prob": [],
        "pulsar_det_pred": [], "pulsar_det_true": [], "pulsar_det_prob": [],
        "pulsar_sub_pred": [], "pulsar_sub_true": [],
        "radio_pred": [], "radio_true": [],
        "gw_pred": [], "gw_true": [], "gw_prob": [],
        "anomaly_score": [], "anomaly_true": [],
        "reg_pred": [], "reg_true": [], "reg_mask": [],
        "fused_embeddings": [], "masks": [],
    }

    for batch_inputs, batch_labels, batch_masks in loader:
        for k in batch_inputs:
            batch_inputs[k] = batch_inputs[k].to(device)
        for k in batch_labels:
            if isinstance(batch_labels[k], torch.Tensor):
                batch_labels[k] = batch_labels[k].to(device)
        batch_masks = batch_masks.to(device)

        outputs = model(batch_inputs, batch_masks)
        lbl = batch_labels

        # ── Stellar classification ───────────────────────────────
        valid = lbl["stellar_cls"] != IGNORE_LABEL
        if valid.any():
            probs = F.softmax(outputs["stellar_cls"][valid], dim=1)
            preds = probs.argmax(dim=1)
            acc["stellar_cls_pred"].append(preds.cpu())
            acc["stellar_cls_true"].append(lbl["stellar_cls"][valid].cpu())
            acc["stellar_cls_prob"].append(probs.cpu())

        # ── Pulsar detection ─────────────────────────────────────
        valid = lbl["pulsar_det"] != IGNORE_LABEL
        if valid.any():
            logits = outputs["pulsar_det"][valid].squeeze(-1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            acc["pulsar_det_pred"].append(preds.cpu())
            acc["pulsar_det_true"].append(lbl["pulsar_det"][valid].cpu())
            acc["pulsar_det_prob"].append(probs.cpu())

        # ── Pulsar subtype ───────────────────────────────────────
        valid = lbl["pulsar_subtype"] != IGNORE_LABEL
        if valid.any():
            preds = outputs["pulsar_subtype"][valid].argmax(dim=1)
            acc["pulsar_sub_pred"].append(preds.cpu())
            acc["pulsar_sub_true"].append(lbl["pulsar_subtype"][valid].cpu())

        # ── Radio morphology ─────────────────────────────────────
        valid = lbl["radio_morphology"] != IGNORE_LABEL
        if valid.any():
            preds = outputs["radio_morphology"][valid].argmax(dim=1)
            acc["radio_pred"].append(preds.cpu())
            acc["radio_true"].append(lbl["radio_morphology"][valid].cpu())

        # ── GW detection ─────────────────────────────────────────
        valid = lbl["gw_det"] != IGNORE_LABEL
        if valid.any():
            logits = outputs["gw_det"][valid].squeeze(-1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            acc["gw_pred"].append(preds.cpu())
            acc["gw_true"].append(lbl["gw_det"][valid].cpu())
            acc["gw_prob"].append(probs.cpu())

        # ── Anomaly ──────────────────────────────────────────────
        valid = lbl["anomaly"] != IGNORE_LABEL
        if valid.any():
            scores = torch.sigmoid(
                outputs["anomaly"][valid].squeeze(-1)
            )
            acc["anomaly_score"].append(scores.cpu())
            acc["anomaly_true"].append(lbl["anomaly"][valid].cpu())

        # ── Regression ───────────────────────────────────────────
        rmask = lbl["reg_mask"]
        has_reg = rmask.sum(dim=1) > 0
        if has_reg.any():
            acc["reg_pred"].append(outputs["regression"][has_reg].cpu())
            acc["reg_true"].append(lbl["regression"][has_reg].cpu())
            acc["reg_mask"].append(rmask[has_reg].cpu())

        # ── Fused embeddings + masks ─────────────────────────────
        acc["fused_embeddings"].append(outputs["fused"].cpu())
        acc["masks"].append(batch_masks.cpu())

    # Concatenate
    result = {}
    for key, arrays in acc.items():
        if arrays:
            result[key] = torch.cat(arrays, dim=0).numpy()
        else:
            result[key] = np.array([])

    return result


# ═════════════════════════════════════════════════════════════
# 2. PER-HEAD EVALUATION
# ═════════════════════════════════════════════════════════════

def evaluate_stellar_cls(preds, report):
    """Evaluate stellar classification head."""
    if len(preds.get("stellar_cls_pred", [])) == 0:
        return report

    y_true = preds["stellar_cls_true"]
    y_pred = preds["stellar_cls_pred"]
    y_prob = preds["stellar_cls_prob"]

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report["stellar_cls"] = {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "n_samples": int(len(y_true)),
        "per_class": {},
    }

    # Per-class metrics
    cr = classification_report(y_true, y_pred,
                               target_names=STELLAR_CLASS_NAMES,
                               output_dict=True, zero_division=0)
    for cls_name in STELLAR_CLASS_NAMES:
        if cls_name in cr:
            report["stellar_cls"]["per_class"][cls_name] = {
                "f1": float(cr[cls_name]["f1-score"]),
                "precision": float(cr[cls_name]["precision"]),
                "recall": float(cr[cls_name]["recall"]),
                "support": int(cr[cls_name]["support"]),
            }

    # AUC (one-vs-rest)
    try:
        if y_prob.shape[1] == NUM_STELLAR_CLASSES:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                average="macro")
            report["stellar_cls"]["macro_auc"] = float(auc)
    except Exception:
        pass

    # Confusion matrix
    _plot_confusion_matrix(y_true, y_pred, STELLAR_CLASS_NAMES,
                           "Stellar Classification", "stellar_cls_cm.png")

    print(f"  Stellar  | Acc: {acc:.4f} | F1: {f1:.4f} | N: {len(y_true)}")
    return report


def evaluate_binary_head(preds, pred_key, true_key, prob_key,
                         head_name, report):
    """Evaluate a binary detection head."""
    if len(preds.get(pred_key, [])) == 0:
        return report

    y_true = preds[true_key]
    y_pred = preds[pred_key]

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="binary", zero_division=0)

    entry = {
        "accuracy": float(acc),
        "f1": float(f1),
        "n_samples": int(len(y_true)),
    }

    if len(preds.get(prob_key, [])) > 0:
        try:
            auc = roc_auc_score(y_true, preds[prob_key])
            entry["roc_auc"] = float(auc)
        except Exception:
            pass

    report[head_name] = entry
    auc_str = f" | AUC: {entry.get('roc_auc', 0):.4f}" if "roc_auc" in entry else ""
    print(f"  {head_name:<10s} | Acc: {acc:.4f} | F1: {f1:.4f}{auc_str}"
          f" | N: {len(y_true)}")
    return report


def evaluate_multiclass_head(preds, pred_key, true_key,
                             class_names, head_name, report):
    """Evaluate a multi-class head."""
    if len(preds.get(pred_key, [])) == 0:
        return report

    y_true = preds[true_key]
    y_pred = preds[pred_key]

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report[head_name] = {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "n_samples": int(len(y_true)),
    }

    _plot_confusion_matrix(y_true, y_pred, class_names,
                           head_name, f"{head_name}_cm.png")

    print(f"  {head_name:<10s} | Acc: {acc:.4f} | F1: {f1:.4f}"
          f" | N: {len(y_true)}")
    return report


def evaluate_regression(preds, report):
    """Evaluate regression head."""
    if len(preds.get("reg_pred", [])) == 0:
        return report

    y_pred = preds["reg_pred"]
    y_true = preds["reg_true"]
    mask   = preds["reg_mask"]

    report["regression"] = {"per_target": {}}

    for i, name in enumerate(REG_TARGET_NAMES):
        valid = mask[:, i] > 0
        if valid.sum() == 0:
            continue

        yt = y_true[valid, i]
        yp = y_pred[valid, i]

        # Filter NaN
        finite = np.isfinite(yt) & np.isfinite(yp)
        if finite.sum() < 2:
            continue

        yt, yp = yt[finite], yp[finite]

        mae = mean_absolute_error(yt, yp)
        r2  = r2_score(yt, yp)

        report["regression"]["per_target"][name] = {
            "mae": float(mae),
            "r2":  float(r2),
            "n_samples": int(finite.sum()),
        }
        print(f"  Reg {name:<12s} | MAE: {mae:.4f} | R²: {r2:.4f}"
              f" | N: {int(finite.sum())}")

    # Scatter plots
    _plot_regression_scatter(y_pred, y_true, mask)

    return report


# ═════════════════════════════════════════════════════════════
# 3. PHYSICS CONSISTENCY
# ═════════════════════════════════════════════════════════════

def evaluate_physics(preds, report):
    """
    Evaluate Stefan-Boltzmann consistency on regression outputs.
    SB: L = 4πR²σT⁴ → log_lum ≈ 2·log_radius + 4·log_teff + const
    """
    if len(preds.get("reg_pred", [])) == 0:
        return report

    y_pred = preds["reg_pred"]
    mask   = preds["reg_mask"]

    # Need lum, teff, radius all valid
    has_all = (mask[:, 1] > 0) & (mask[:, 2] > 0) & (mask[:, 3] > 0)
    if has_all.sum() < 10:
        return report

    log_lum  = y_pred[has_all, 1]
    log_teff = y_pred[has_all, 2]
    log_rad  = y_pred[has_all, 3]

    # SB constant: log(4π) + log(σ) expressed in solar units ≈ -0.009
    sb_const = -0.009
    sb_expected = 2 * log_rad + 4 * log_teff + sb_const
    sb_residual = log_lum - sb_expected

    mean_res = float(np.mean(sb_residual))
    mae_res  = float(np.mean(np.abs(sb_residual)))
    std_res  = float(np.std(sb_residual))

    report["physics"] = {
        "stefan_boltzmann": {
            "mean_residual": mean_res,
            "mae_residual":  mae_res,
            "std_residual":  std_res,
            "n_samples":     int(has_all.sum()),
        }
    }
    print(f"  Physics SB   | Mean res: {mean_res:.4f} | "
          f"MAE: {mae_res:.4f} | N: {int(has_all.sum())}")

    return report


# ═════════════════════════════════════════════════════════════
# 4. MODALITY ANALYSIS
# ═════════════════════════════════════════════════════════════

def evaluate_modality_stats(preds, report):
    """Analyze modality distribution in test set."""
    if len(preds.get("masks", [])) == 0:
        return report

    masks = preds["masks"]
    n = masks.shape[0]

    stats = {}
    for i, name in enumerate(MODALITY_ORDER):
        active = masks[:, i].sum()
        stats[name] = {"active": int(active), "pct": float(active / n * 100)}

    # Multi-modal samples
    multi = (masks.sum(axis=1) > 1).sum()
    stats["multi_modal"] = {"count": int(multi), "pct": float(multi / n * 100)}

    report["modality_stats"] = stats

    print(f"\n── Modality Distribution (N={n}) ──")
    for name in MODALITY_ORDER:
        s = stats[name]
        print(f"  {name:<8s}: {s['active']:>6d} ({s['pct']:.1f}%)")
    print(f"  {'multi':<8s}: {stats['multi_modal']['count']:>6d} "
          f"({stats['multi_modal']['pct']:.1f}%)")

    return report


# ═════════════════════════════════════════════════════════════
# 5. PLOTTING HELPERS
# ═════════════════════════════════════════════════════════════

def _plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Save confusion matrix plot."""
    os.makedirs(LOG_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(max(6, n_classes * 1.2),
                                    max(5, n_classes)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"{title} — Confusion Matrix", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    path = os.path.join(LOG_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_regression_scatter(y_pred, y_true, mask):
    """Save regression scatter plots (pred vs true)."""
    os.makedirs(LOG_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, NUM_REG_TARGETS, figsize=(16, 4))
    fig.suptitle("Regression — Predicted vs True", fontsize=13)

    for i, (ax, name) in enumerate(zip(axes, REG_TARGET_NAMES)):
        valid = mask[:, i] > 0
        if valid.sum() < 2:
            ax.set_title(f"{name}\n(no data)"); continue

        yt = y_true[valid, i]
        yp = y_pred[valid, i]
        finite = np.isfinite(yt) & np.isfinite(yp)
        if finite.sum() < 2:
            ax.set_title(f"{name}\n(no valid)"); continue

        yt, yp = yt[finite], yp[finite]
        ax.scatter(yt, yp, s=3, alpha=0.3, color="steelblue")
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("True"); ax.set_ylabel("Predicted")
        r2 = r2_score(yt, yp)
        ax.set_title(f"{name}\nR²={r2:.4f}", fontsize=10)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(LOG_DIR, "regression_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ═════════════════════════════════════════════════════════════
# 6. MAIN EVALUATION ENTRY POINT
# ═════════════════════════════════════════════════════════════

def run_evaluation(checkpoint_path: str = None, split: str = "test"):
    """
    Run full evaluation of the unified model.

    Args:
        checkpoint_path: path to .pt checkpoint. If None, uses best Stage 1.
        split: "test" or "val"
    """
    set_seed(SEED)
    device = get_device()
    os.makedirs(LOG_DIR, exist_ok=True)

    print("═" * 60)
    print("STELLARIS-DNet — Unified Evaluation")
    print("═" * 60)

    # ── Load model ───────────────────────────────────────────────
    if checkpoint_path is None:
        candidates = [
            os.path.join(CHECKPOINT_DIR, "unified_final.pt"),
            os.path.join(CHECKPOINT_DIR, "unified_best_stage1_frozen.pt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                checkpoint_path = c
                break

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("❌ No checkpoint found — run training first")
        return None

    print(f"Loading: {checkpoint_path}")
    model = UnifiedModel(device=device, load_encoders=True)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", "?")
    print(f"Epoch: {epoch} | Val loss: {val_loss}")

    # ── Load data ────────────────────────────────────────────────
    _, val_loader, test_loader = get_unified_loaders(verbose=False)
    loader = test_loader if split == "test" else val_loader
    print(f"Evaluating on {split} set ({len(loader.dataset)} samples)")

    # ── Collect predictions ──────────────────────────────────────
    print("\nCollecting predictions...")
    preds = collect_predictions(model, loader, device)

    # ── Evaluate each head ───────────────────────────────────────
    report = {"checkpoint": checkpoint_path, "split": split, "epoch": epoch}

    print("\n── Per-Head Results ──")
    report = evaluate_stellar_cls(preds, report)
    report = evaluate_binary_head(preds, "pulsar_det_pred", "pulsar_det_true",
                                  "pulsar_det_prob", "pulsar_det", report)
    report = evaluate_multiclass_head(preds, "pulsar_sub_pred", "pulsar_sub_true",
                                      PULSAR_SUBTYPE_NAMES, "pulsar_sub", report)
    report = evaluate_multiclass_head(preds, "radio_pred", "radio_true",
                                      RADIO_CLASS_NAMES, "radio_morph", report)
    report = evaluate_binary_head(preds, "gw_pred", "gw_true",
                                  "gw_prob", "gw_det", report)
    report = evaluate_regression(preds, report)
    report = evaluate_physics(preds, report)
    report = evaluate_modality_stats(preds, report)

    # ── Save report ──────────────────────────────────────────────
    report_path = os.path.join(LOG_DIR, "unified_eval_report.json")

    def _jsonable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_jsonable)
    print(f"\n📊 Report → {report_path}")

    # ── Print summary table ──────────────────────────────────────
    print("\n" + "═" * 60)
    print("UNIFIED EVALUATION SUMMARY")
    print("═" * 60)
    for head in ["stellar_cls", "pulsar_det", "pulsar_sub",
                 "radio_morph", "gw_det"]:
        if head in report:
            r = report[head]
            acc = r.get("accuracy", r.get("macro_f1", 0))
            f1  = r.get("macro_f1", r.get("f1", 0))
            n   = r.get("n_samples", 0)
            print(f"  {head:<16s}: Acc={acc:.4f}  F1={f1:.4f}  N={n}")
    if "regression" in report:
        for tgt, m in report["regression"].get("per_target", {}).items():
            print(f"  reg_{tgt:<12s}: MAE={m['mae']:.4f}  R²={m['r2']:.4f}")
    if "physics" in report:
        sb = report["physics"]["stefan_boltzmann"]
        print(f"  physics_SB     : MAE_res={sb['mae_residual']:.4f}")
    print("═" * 60)

    return report


# ═════════════════════════════════════════════════════════════
# 7. CLI
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="STELLARIS-DNet Unified Evaluation"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "val"])
    args = parser.parse_args()

    run_evaluation(checkpoint_path=args.checkpoint, split=args.split)