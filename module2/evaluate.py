"""
module2/evaluate.py
STELLARIS-DNet — Module 2 Full Evaluation Suite (Upgraded)

Kept from original:
  1.  Metrics + Confusion Matrix
  2.  ROC Curves
  3.  GradCAM Saliency Maps      [FIXED: handles (logits, aux) tuple]
  4.  Failure Analysis
  5.  Confidence Histogram + Calibration
  6.  Speed Benchmark
  7.  CQT vs Raw Comparison
  8.  Cleaned Spectrograms
  9.  Radio Galaxy Catalogue
  10. LIGO Event Triggers

Added:
  11. Precision-Recall Curves
  12. Physics Consistency Report (jet power / chirp slope)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, auc,
    precision_recall_curve, average_precision_score
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module2.config     import *
from module2.dataset_2a import load_mirabest
from module2.dataset_2b import (load_g2net, signal_to_spectrogram,
                                 _whiten, _bandpass,
                                 _resolve_data_dir, _get_file_path)
from module2.model      import (RadioGalaxyClassifier, GravWaveDetector,
                                 GravWave1DCNN)
from core.utils         import get_device, get_logger


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _unpack(output):
    """Handle both (logits, aux) and plain logits."""
    if isinstance(output, tuple):
        return output[0], output[1]
    return output, None


def load_model(model, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Run module2/train.py first."
        )
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"✅ Loaded: {path}")
    return model


def save_plot(fig, name, log_dir=LOG_DIR):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved → {path}")


def _get_predictions(model, loader, device):
    all_preds, all_probs, all_labels = [], [], []
    all_aux = []
    with torch.no_grad():
        for X, y in loader:
            X           = X.to(device)
            logits, aux = _unpack(model(X))
            probs       = F.softmax(logits, dim=1)
            preds       = logits.argmax(1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            if aux is not None:
                all_aux.extend(aux.squeeze(1).cpu().numpy())
    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
            np.array(all_aux) if all_aux else None)


# ─────────────────────────────────────────────
# 1. METRICS + CONFUSION MATRIX
# ─────────────────────────────────────────────
def evaluate_metrics(model, loader, device, logger,
                     class_names, tag, cmap="Blues"):
    logger.info(f"Evaluating {tag} metrics")
    y_true, y_pred, y_prob, y_aux = _get_predictions(model, loader, device)

    acc       = (y_pred == y_true).mean()
    f1        = f1_score(y_true, y_pred, average="macro")
    auc_score = roc_auc_score(
        y_true,
        y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob,
        multi_class="ovr"
    )

    print(f"\n── {tag} Results ──")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print(f"ROC-AUC  : {auc_score:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    logger.info(f"{tag} acc={acc:.4f} f1={f1:.4f} auc={auc_score:.4f}")

    # Confusion matrix
    cm_arr = confusion_matrix(y_true, y_pred)
    n      = len(class_names)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, cmap=cmap)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{tag} — Confusion Matrix")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm_arr[i, j], ha="center", va="center",
                    color="white" if cm_arr[i,j] > cm_arr.max()/2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    save_plot(fig, f"{tag.lower()}_confusion_matrix.png")

    return {"acc": acc, "f1": f1, "auc": auc_score,
            "y_true": y_true, "y_pred": y_pred,
            "y_prob": y_prob, "y_aux": y_aux}


# ─────────────────────────────────────────────
# 2. ROC + PR CURVES (upgraded: both on one figure)
# ─────────────────────────────────────────────
def plot_roc_curves(results_2a, results_2b, logger):
    logger.info("Plotting ROC + PR curves")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for col, (res, tag, color, classes) in enumerate([
        (results_2a, "2A Radio Galaxy", "#FF6B35", RGZ_CLASSES),
        (results_2b, "2B GW Detection", "#4ECDC4", LIGO_CLASSES),
    ]):
        y_true = res["y_true"]
        y_prob = res["y_prob"]

        # ── ROC ──
        ax = axes[0, col]
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5)
        ax.fill_between(fpr, tpr, alpha=0.15, color=color)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{tag}\nROC Curve"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

        # ── Precision-Recall ──
        ax = axes[1, col]
        prec, rec, _ = precision_recall_curve(y_true, y_prob[:, 1])
        ap           = average_precision_score(y_true, y_prob[:, 1])
        ax.plot(rec, prec, color=color, lw=2.5, label=f"AP = {ap:.4f}")
        ax.fill_between(rec, prec, alpha=0.15, color=color)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"{tag}\nPrecision-Recall Curve")
        ax.legend(); ax.grid(alpha=0.3)
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

    fig.suptitle("ROC + PR Curves — STELLARIS-DNet Module 2",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, "roc_pr_curves.png")


# ─────────────────────────────────────────────
# 3. GRADCAM (FIXED for tuple returns)
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach()))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def generate(self, x, class_idx=None):
        self.model.eval()
        x      = x.clone().requires_grad_(True)
        output = self.model(x)
        # FIX: unpack tuple if model returns (logits, aux)
        logits = output[0] if isinstance(output, tuple) else output

        if class_idx is None:
            class_idx = logits.argmax(1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam     = F.interpolate(cam, x.shape[2:],
                                mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def generate_gradcam(model, loader, device, logger,
                     class_names, tag, n=6):
    logger.info(f"Generating GradCAM for {tag}")
    target_layer = model.backbone[-1]
    gc = GradCAM(model, target_layer)

    fig, axes = plt.subplots(2, n, figsize=(3*n, 7))
    fig.patch.set_facecolor("#0d0d0d")
    shown = 0

    for X, y in loader:
        for i in range(len(X)):
            if shown >= n: break
            x_s     = X[i:i+1].to(device)
            cam, pred = gc.generate(x_s)

            # Denormalize
            img  = X[i].cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])[:, None, None]
            std  = np.array([0.229, 0.224, 0.225])[:, None, None]
            img  = np.clip(img * std + mean, 0, 1).transpose(1, 2, 0)
            heat = cm.jet(cam)[:, :, :3]
            over = 0.55 * img + 0.45 * heat

            col = shown
            axes[0, col].imshow(img)
            axes[0, col].set_title(f"True: {class_names[y[i]]}",
                                    fontsize=8, color="white")
            axes[0, col].axis("off")
            axes[1, col].imshow(over)
            axes[1, col].set_title(
                f"Pred: {class_names[pred]}",
                fontsize=8,
                color="lime" if pred == y[i].item() else "red"
            )
            axes[1, col].axis("off")
            shown += 1
        if shown >= n: break

    fig.suptitle(f"GradCAM Saliency Maps — {tag}\n"
                 "Top: Original | Bottom: Heatmap Overlay",
                 color="white", fontsize=11)
    save_plot(fig, f"{tag.lower()}_gradcam.png")


# ─────────────────────────────────────────────
# 4. FAILURE ANALYSIS
# ─────────────────────────────────────────────
def failure_analysis(model, loader, device, logger,
                     class_names, tag, n_worst=8):
    logger.info(f"Failure analysis for {tag}")
    model.eval()
    failures = []

    with torch.no_grad():
        for X, y in loader:
            X_d         = X.to(device)
            logits, _   = _unpack(model(X_d))
            prob        = F.softmax(logits, dim=1).cpu().numpy()
            pred        = prob.argmax(1)
            for i in range(len(X)):
                if pred[i] != y[i].item():
                    failures.append((prob[i, pred[i]], y[i].item(),
                                     pred[i], X[i].cpu().numpy()))

    if not failures:
        print(f"✅ {tag}: No failures found!")
        return

    failures.sort(key=lambda x: x[0], reverse=True)
    failures = failures[:n_worst]

    cols      = min(n_worst, 4)
    rows      = (len(failures) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    fig.patch.set_facecolor("#0d0d0d")
    axes_flat = np.array(axes).flatten()

    for i, (conf, true, pred, img) in enumerate(failures):
        ax   = axes_flat[i]
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std  = np.array([0.229, 0.224, 0.225])[:, None, None]
        img_show = np.clip(img * std + mean, 0, 1).transpose(1, 2, 0)
        ax.imshow(img_show[:, :, 0], cmap="hot")
        ax.set_title(f"True: {class_names[true]}\n"
                     f"Pred: {class_names[pred]} ({conf:.1%})",
                     fontsize=8, color="red")
        ax.axis("off")

    for ax in axes_flat[len(failures):]:
        ax.axis("off")

    fig.suptitle(f"Failure Analysis — {tag}\nMost confident wrong predictions",
                 color="white", fontsize=11)
    save_plot(fig, f"{tag.lower()}_failure_analysis.png")


# ─────────────────────────────────────────────
# 5. CONFIDENCE HISTOGRAM + CALIBRATION
# ─────────────────────────────────────────────
def confidence_histogram(results, tag, class_names, logger):
    logger.info(f"Confidence histogram for {tag}")
    y_true     = results["y_true"]
    y_pred     = results["y_pred"]
    y_prob     = results["y_prob"]
    confidence = y_prob.max(axis=1)
    correct    = (y_pred == y_true)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(confidence[correct],  bins=30, alpha=0.7,
                 color="#2ecc71", label="Correct", density=True)
    axes[0].hist(confidence[~correct], bins=30, alpha=0.7,
                 color="#e74c3c", label="Incorrect", density=True)
    axes[0].axvline(0.5, color="black", linestyle="--", label="50% threshold")
    axes[0].set_xlabel("Confidence"); axes[0].set_ylabel("Density")
    axes[0].set_title(f"{tag}\nConfidence Distribution")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Calibration
    bins      = np.linspace(0, 1, 11)
    bin_accs, bin_confs = [], []
    for i in range(len(bins)-1):
        mask = (confidence >= bins[i]) & (confidence < bins[i+1])
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confidence[mask].mean())

    axes[1].plot([0,1],[0,1], "k--", label="Perfect calibration")
    axes[1].plot(bin_confs, bin_accs, "o-", color="#3498db", lw=2, label="Model")
    axes[1].fill_between(bin_confs, bin_accs, bin_confs, alpha=0.2, color="#3498db")
    axes[1].set_xlabel("Mean Confidence"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{tag}\nCalibration Curve")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_xlim([0,1]); axes[1].set_ylim([0,1])

    print(f"\n── {tag} Confidence ──")
    print(f"Mean (correct):   {confidence[correct].mean():.4f}")
    print(f"Mean (incorrect): {confidence[~correct].mean():.4f}")
    print(f"High-conf errors: {((confidence > 0.9) & ~correct).sum()}")

    fig.suptitle(f"Confidence Analysis — {tag}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, f"{tag.lower()}_confidence_histogram.png")


# ─────────────────────────────────────────────
# 6. SPEED BENCHMARK
# ─────────────────────────────────────────────
def speed_benchmark(model, device, logger, tag, input_shape, n_runs=100):
    logger.info(f"Speed benchmark for {tag}")
    model.eval()
    dummy = torch.randn(1, *input_shape).to(device)

    with torch.no_grad():
        for _ in range(10):
            _unpack(model(dummy))

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _unpack(model(dummy))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms  = (t1 - t0) * 1000 / n_runs
    fps = 1000 / ms

    print(f"\n── {tag} Speed ──")
    print(f"Latency  : {ms:.2f} ms/sample")
    print(f"Throughput: {fps:.0f} samples/sec")
    logger.info(f"{tag} benchmark: {ms:.2f}ms | {fps:.0f} fps")
    return ms, fps


def plot_speed_comparison(benchmarks, logger):
    logger.info("Plotting speed comparison")
    names  = list(benchmarks.keys())
    ms     = [benchmarks[n][0] for n in names]
    fps_v  = [benchmarks[n][1] for n in names]
    colors = ["#FF6B35", "#4ECDC4", "#9B59B6", "#F39C12"][:len(names)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(names, ms, color=colors, edgecolor="white")
    axes[0].set_xlabel("Latency (ms/sample)")
    axes[0].set_title("Inference Latency (lower = better)")
    for i, v in enumerate(ms):
        axes[0].text(v + 0.1, i, f"{v:.1f}ms", va="center")

    axes[1].barh(names, fps_v, color=colors, edgecolor="white")
    axes[1].set_xlabel("Samples/second")
    axes[1].set_title("Throughput (higher = better)")
    for i, v in enumerate(fps_v):
        axes[1].text(v + 1, i, f"{v:.0f}", va="center")

    fig.suptitle("Speed Benchmark — STELLARIS-DNet Module 2",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, "speed_benchmark.png")


# ─────────────────────────────────────────────
# 7. CQT vs RAW COMPARISON
# ─────────────────────────────────────────────
def cqt_vs_raw_comparison(device, logger, quick_epochs=5):
    logger.info("CQT vs Raw signal comparison")
    print("\n── CQT vs Raw Signal Comparison ──")

    results, acc_hist = {}, {}

    def quick_train(model, loader_tr, loader_vl, tag, epochs=quick_epochs):
        opt  = torch.optim.AdamW(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        hist = []
        for ep in range(epochs):
            model.train()
            for X, y in loader_tr:
                X, y = X.to(device), y.to(device)
                opt.zero_grad()
                logits, _ = _unpack(model(X))
                crit(logits, y).backward()
                opt.step()
            model.eval()
            c = t = 0
            with torch.no_grad():
                for X, y in loader_vl:
                    X, y    = X.to(device), y.to(device)
                    logits, _ = _unpack(model(X))
                    c      += (logits.argmax(1) == y).sum().item()
                    t      += y.size(0)
            acc = c / t
            hist.append(acc)
            print(f"   {tag} Epoch {ep+1}: {acc:.4f}")
        return max(hist), hist

    try:
        tr, vl, _ = load_g2net(use_cqt=True)
        m = GravWaveDetector(pretrained=True).to(device)
        best, hist = quick_train(m, tr, vl, "CQT+EfficientNet-B2")
        results["With CQT\n(EfficientNet-B2)"]  = best
        acc_hist["With CQT\n(EfficientNet-B2)"] = hist
    except Exception as e:
        print(f"CQT comparison failed: {e}")

    try:
        tr, vl, _ = load_g2net(use_cqt=False)
        m = GravWave1DCNN().to(device)
        best, hist = quick_train(m, tr, vl, "Raw+1DCNN")
        results["Without CQT\n(1D CNN)"]  = best
        acc_hist["Without CQT\n(1D CNN)"] = hist
    except Exception as e:
        print(f"Raw comparison failed: {e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2ecc71", "#e74c3c"]

    names  = list(results.keys())
    values = [v * 100 for v in results.values()]
    bars   = axes[0].bar(names, values, color=colors[:len(names)],
                          width=0.4, edgecolor="white")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Validation Accuracy (%)")
    axes[0].set_title(f"CQT vs Raw ({quick_epochs} epochs)")
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 1,
                     f"{val:.1f}%", ha="center",
                     fontsize=13, fontweight="bold")

    for (name, hist), color in zip(acc_hist.items(), colors):
        axes[1].plot(range(1, len(hist)+1), hist, "o-",
                     color=color, lw=2, label=name)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Accuracy")
    axes[1].set_title("Learning Curves"); axes[1].legend()
    axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 1)

    fig.suptitle("Why CQT Spectrograms Beat Raw Signals\nfor GW Detection",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, "cqt_vs_raw_comparison.png")

    if len(results) == 2:
        vals = list(results.values())
        print(f"\n   CQT improvement: +{(vals[0]-vals[1])*100:.1f}%")
    return results


# ─────────────────────────────────────────────
# 8. CLEANED SPECTROGRAMS
# ─────────────────────────────────────────────
def plot_cleaned_spectrograms(logger, n=3):
    logger.info("Generating cleaned spectrograms")
    fig, axes = plt.subplots(n, 3, figsize=(15, 4*n))
    fig.suptitle("LIGO Strain: Raw → Whitened → Bandpass Cleaned",
                 fontsize=11)
    np.random.seed(0)
    for i in range(n):
        raw     = np.random.randn(LIGO_SIGNAL_LEN).astype(np.float32)
        t       = np.linspace(0, 2, LIGO_SIGNAL_LEN)
        chirp   = (np.sin(2*np.pi*(20*t + 70*t**2)) *
                   np.exp(3*(t-2))).astype(np.float32)
        raw    += 0.3 * chirp
        whitened = _whiten(raw[None])[0]
        cleaned  = _bandpass(whitened[None])[0]
        for j, (sig, title) in enumerate([
            (raw,      "Raw Strain"),
            (whitened, "Whitened"),
            (cleaned,  "Bandpass 20-500Hz")
        ]):
            axes[i, j].specgram(sig, NFFT=128, Fs=2048, noverlap=64, cmap="inferno")
            axes[i, j].set_ylim(0, 512)
            axes[i, j].set_title(title, fontsize=9)
            if j == 0:
                axes[i, j].set_ylabel(f"Sample {i+1}\nFreq (Hz)")
    fig.tight_layout()
    save_plot(fig, "cleaned_spectrograms.png")


# ─────────────────────────────────────────────
# 9. RADIO GALAXY CATALOGUE
# ─────────────────────────────────────────────
def build_catalogue(model, loader, device, logger):
    logger.info("Building radio galaxy catalogue")
    records = []
    sid     = 0
    with torch.no_grad():
        for X, y in loader:
            X           = X.to(device)
            logits, aux = _unpack(model(X))
            probs       = F.softmax(logits, dim=1).cpu().numpy()
            preds       = probs.argmax(1)
            aux_vals    = aux.squeeze(1).cpu().numpy() if aux is not None else [None]*len(X)
            for i in range(len(X)):
                records.append({
                    "source_id":       f"STELLARIS_{sid:05d}",
                    "predicted_class": RGZ_CLASSES[preds[i]],
                    "true_class":      RGZ_CLASSES[y[i].item()],
                    "FRI_prob":        round(float(probs[i, 0]), 4),
                    "FRII_prob":       round(float(probs[i, 1]), 4),
                    "confidence":      round(float(probs[i].max()), 4),
                    "log10_jet_power": round(float(aux_vals[i]), 4) if aux_vals[i] is not None else None,
                    "physics_boundary":">10^25 W/Hz" if preds[i]==1 else "<10^25 W/Hz",
                    "correct":         preds[i] == y[i].item()
                })
                sid += 1

    df   = pd.DataFrame(records)
    path = os.path.join(LOG_DIR, "radio_galaxy_catalogue.csv")
    os.makedirs(LOG_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n── Radio Galaxy Catalogue ──")
    print(f"Total: {len(df)} | FRI: {(df['predicted_class']=='FRI').sum()} | "
          f"FRII: {(df['predicted_class']=='FRII').sum()}")
    print(f"Accuracy: {df['correct'].mean():.4f} | Saved → {path}")
    return df


# ─────────────────────────────────────────────
# 10. LIGO EVENT TRIGGERS
# ─────────────────────────────────────────────
def detect_event_triggers(model, device, logger, threshold=0.85, duration_s=40):
    logger.info("LIGO event trigger detection")
    model.eval()
    total  = LIGO_SIGNAL_LEN * 20
    t_arr  = np.linspace(0, duration_s, total)
    strain = np.random.randn(LIGO_N_DETECTORS, total).astype(np.float32)
    event_times = [5.0, 18.5, 32.0]
    for et in event_times:
        idx   = int(et / duration_s * total)
        t_ch  = np.linspace(0, 2, LIGO_SIGNAL_LEN)
        chirp = (np.sin(2*np.pi*(20*t_ch + 70*t_ch**2)) *
                 np.exp(3*(t_ch-2))).astype(np.float32)
        end   = min(idx + LIGO_SIGNAL_LEN, total)
        strain[:, idx:end] += 0.8 * chirp[:end-idx]

    step, triggers, probs_tl = LIGO_SIGNAL_LEN // 4, [], np.zeros(total)
    with torch.no_grad():
        for i in range(0, total - LIGO_SIGNAL_LEN, step):
            win  = strain[:, i:i+LIGO_SIGNAL_LEN].copy()
            spec = signal_to_spectrogram(win)
            x    = torch.tensor(spec[None], dtype=torch.float32).to(device)
            prob = F.softmax(model(x), dim=1)[0, 1].item()
            probs_tl[i:i+step] = prob
            if prob > threshold:
                triggers.append({"timestamp_s": round(i/total*duration_s, 3),
                                  "signal_prob": round(prob, 4),
                                  "SNR_est":     round(prob*15, 2),
                                  "type":        "GW_candidate"})

    df   = pd.DataFrame(triggers).drop_duplicates("timestamp_s") if triggers else pd.DataFrame()
    path = os.path.join(LOG_DIR, "ligo_event_triggers.csv")
    df.to_csv(path, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    axes[0].plot(t_arr, strain[0], lw=0.4, color="steelblue", alpha=0.7)
    axes[0].set_title("LIGO H1 Strain (synthetic)"); axes[0].set_ylabel("Strain")
    axes[1].plot(t_arr, probs_tl, color="orangered", lw=1)
    axes[1].axhline(threshold, color="red", ls="--", label=f"Threshold {threshold}")
    for et in event_times:
        axes[1].axvline(et, color="lime", ls=":", alpha=0.8)
    axes[1].set_title("GW Signal Probability"); axes[1].legend()
    axes[1].set_ylabel("P(Signal)"); axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    save_plot(fig, "ligo_event_triggers.png")

    print(f"\n── Event Triggers ──")
    print(f"Triggers found: {len(df)} | Saved → {path}")
    return df


# ─────────────────────────────────────────────
# 11. PHYSICS CONSISTENCY REPORT (NEW)
# ─────────────────────────────────────────────
def physics_consistency_report(results_2a, logger):
    """
    Reports what fraction of predictions respect the FR boundary.
    FRI predicted jet power < 10^25 W/Hz, FRII > 10^25 W/Hz.
    """
    logger.info("Physics consistency report")
    y_aux = results_2a.get("y_aux")
    if y_aux is None:
        print("⚠️  No jet power predictions — USE_JET_POWER_HEAD=False")
        return

    y_true = results_2a["y_true"]
    y_pred = results_2a["y_pred"]

    fri_mask  = (y_true == 0)
    frii_mask = (y_true == 1)
    boundary  = FRI_FRII_BOUNDARY_LOG

    fri_consistent  = (y_aux[fri_mask]  < boundary).mean() if fri_mask.sum() > 0 else 0
    frii_consistent = (y_aux[frii_mask] >= boundary).mean() if frii_mask.sum() > 0 else 0

    logger.info(f"Physics — FRI below boundary:  {fri_consistent:.3f}")
    logger.info(f"Physics — FRII above boundary: {frii_consistent:.3f}")

    print(f"\n── Physics Consistency (FR Boundary) ──")
    print(f"FRI  predicted power < 10^25 W/Hz : {fri_consistent:.3f} ({fri_consistent:.1%})")
    print(f"FRII predicted power > 10^25 W/Hz : {frii_consistent:.3f} ({frii_consistent:.1%})")

    # Plot jet power distributions
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(y_aux[fri_mask],  bins=30, alpha=0.6, color="#FF6B35", label="FRI (true)")
    ax.hist(y_aux[frii_mask], bins=30, alpha=0.6, color="#4ECDC4", label="FRII (true)")
    ax.axvline(boundary, color="red", linestyle="--",
               label=f"FR boundary log₁₀(P)={boundary}")
    ax.set_xlabel("Predicted log₁₀(Jet Power / W·Hz⁻¹)")
    ax.set_ylabel("Count")
    ax.set_title("Physics Head: Predicted Jet Power Distribution\n"
                 f"FRI boundary adherence: {fri_consistent:.1%} | "
                 f"FRII: {frii_consistent:.1%}")
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "physics_jet_power_distribution.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = get_device()
    logger = get_logger("module2_eval", LOG_DIR)
    logger.info("Module 2 Evaluation Started")

    # Load models
    rgc = load_model(RadioGalaxyClassifier(),
                     os.path.join(CHECKPOINT_DIR, "rgc_best.pt"), device)
    gwd = load_model(GravWaveDetector(),
                     os.path.join(CHECKPOINT_DIR, "gwd_best.pt"), device)

    _, _, test_2a = load_mirabest()
    _, _, test_2b = load_g2net(use_cqt=True)

    # 1. Metrics
    res_2a = evaluate_metrics(rgc, test_2a, device, logger,
                               RGZ_CLASSES,  "2A-RGC", "Blues")
    res_2b = evaluate_metrics(gwd, test_2b, device, logger,
                               LIGO_CLASSES, "2B-GWD", "Greens")

    # 2. ROC + PR Curves
    plot_roc_curves(res_2a, res_2b, logger)

    # 3. GradCAM
    generate_gradcam(rgc, test_2a, device, logger, RGZ_CLASSES,  "2A-RGC")
    generate_gradcam(gwd, test_2b, device, logger, LIGO_CLASSES, "2B-GWD")

    # 4. Failure Analysis
    failure_analysis(rgc, test_2a, device, logger, RGZ_CLASSES,  "2A-RGC")
    failure_analysis(gwd, test_2b, device, logger, LIGO_CLASSES, "2B-GWD")

    # 5. Confidence Histograms
    confidence_histogram(res_2a, "2A-RGC", RGZ_CLASSES,  logger)
    confidence_histogram(res_2b, "2B-GWD", LIGO_CLASSES, logger)

    # 6. Speed Benchmark
    benchmarks = {}
    ms, fps = speed_benchmark(rgc, device, logger, "2A-RGC",
                               (3, RGZ_IMG_SIZE, RGZ_IMG_SIZE))
    benchmarks["2A RadioGalaxy\n(EfficientNet-B2)"] = (ms, fps)
    ms, fps = speed_benchmark(gwd, device, logger, "2B-GWD",
                               (LIGO_N_DETECTORS, LIGO_CQT_BINS, LIGO_CQT_STEPS))
    benchmarks["2B GravWave\n(EfficientNet-B2+CQT)"] = (ms, fps)
    plot_speed_comparison(benchmarks, logger)

    # 7. CQT vs Raw
    cqt_vs_raw_comparison(device, logger, quick_epochs=5)

    # 8. Cleaned Spectrograms
    plot_cleaned_spectrograms(logger)

    # 9. Radio Galaxy Catalogue
    build_catalogue(rgc, test_2a, device, logger)

    # 10. LIGO Event Triggers
    detect_event_triggers(gwd, device, logger)

    # 11. Physics Consistency (NEW)
    physics_consistency_report(res_2a, logger)

    # Summary
    print("\n" + "=" * 60)
    print("MODULE 2 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"2A Radio Galaxy — Acc: {res_2a['acc']:.4f} | "
          f"F1: {res_2a['f1']:.4f} | AUC: {res_2a['auc']:.4f}")
    print(f"2B GW Detection — Acc: {res_2b['acc']:.4f} | "
          f"F1: {res_2b['f1']:.4f} | AUC: {res_2b['auc']:.4f}")
    print(f"\nOutputs: {LOG_DIR}/")
    print("=" * 60)
    print("✅ Module 2 evaluation complete.")
    logger.info("Module 2 Evaluation Complete")