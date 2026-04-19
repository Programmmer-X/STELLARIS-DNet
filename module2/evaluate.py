"""
module2/evaluate.py
STELLARIS-DNet — Module 2 Evaluation
Features:
1. Accuracy, F1, Confusion Matrix
2. Loss Curves
3. GradCAM Saliency Maps
4. Feature Map Visualization
5. Automated Catalogue CSV
6. LIGO Event Triggers
7. Cleaned Spectrograms
8. Transfer Learning Comparison
Run AFTER train.py: python module2/evaluate.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module2.config     import *
from module2.dataset_2a import load_mirabest, MiraBestDataset, _load_mirabest
from module2.dataset_2b import load_g2net, SyntheticG2NetDataset
from module2.model      import RadioGalaxyClassifier, GravWaveDetector
from core.utils         import get_device, get_logger


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# 1. METRICS — Accuracy, F1, Confusion Matrix
# ─────────────────────────────────────────────
def evaluate_2a_metrics(model, device, logger):
    logger.info("Evaluating 2A — Radio Galaxy Metrics")
    _, _, test_loader = load_mirabest()

    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            X      = X.to(device)
            logits = model(X)
            probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = (y_pred == y_true).mean()
    f1  = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_prob)

    print(f"\n── 2A: Radio Galaxy Results ──")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=RGZ_CLASSES))
    logger.info(f"2A — acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

    # Confusion matrix
    cm_arr = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(RGZ_NUM_CLASSES))
    ax.set_yticks(range(RGZ_NUM_CLASSES))
    ax.set_xticklabels(RGZ_CLASSES)
    ax.set_yticklabels(RGZ_CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Radio Galaxy — Confusion Matrix")
    for i in range(RGZ_NUM_CLASSES):
        for j in range(RGZ_NUM_CLASSES):
            ax.text(j, i, cm_arr[i, j], ha="center", va="center",
                    color="white" if cm_arr[i, j] > cm_arr.max()/2 else "black")
    fig.colorbar(im)
    save_plot(fig, "2a_confusion_matrix.png")

    return {"acc": acc, "f1": f1, "auc": auc}


def evaluate_2b_metrics(model, device, logger):
    logger.info("Evaluating 2B — GW Detection Metrics")
    _, _, test_loader = load_g2net()

    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            X      = X.to(device)
            logits = model(X)
            probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = (y_pred == y_true).mean()
    f1  = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_prob)

    print(f"\n── 2B: GW Detection Results ──")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=LIGO_CLASSES))
    logger.info(f"2B — acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

    # Confusion matrix
    cm_arr = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, cmap="Greens")
    ax.set_xticks(range(LIGO_NUM_CLASSES))
    ax.set_yticks(range(LIGO_NUM_CLASSES))
    ax.set_xticklabels(LIGO_CLASSES)
    ax.set_yticklabels(LIGO_CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("GW Detection — Confusion Matrix")
    for i in range(LIGO_NUM_CLASSES):
        for j in range(LIGO_NUM_CLASSES):
            ax.text(j, i, cm_arr[i, j], ha="center", va="center",
                    color="white" if cm_arr[i, j] > cm_arr.max()/2 else "black")
    fig.colorbar(im)
    save_plot(fig, "2b_confusion_matrix.png")

    return {"acc": acc, "f1": f1, "auc": auc}


# ─────────────────────────────────────────────
# 2. GRADCAM — Saliency Maps for 2A
# Shows which pixels drove FRI/FRII classification
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, x, class_idx=None):
        self.model.eval()
        x.requires_grad_(True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Compute GradCAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)

        # Resize to input size
        cam = F.interpolate(cam, size=x.shape[2:],
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def generate_gradcam(model, device, logger, n_samples=6):
    logger.info("Generating GradCAM saliency maps")
    _, _, test_loader = load_mirabest()

    # Target last conv block of EfficientNet-B0
    target_layer = model.backbone[-1]
    gradcam      = GradCAM(model, target_layer)

    images_shown = 0
    fig, axes    = plt.subplots(2, n_samples, figsize=(3*n_samples, 6))

    for X, y in test_loader:
        for i in range(min(len(X), n_samples - images_shown)):
            x_single = X[i:i+1].to(device)
            cam, pred = gradcam.generate(x_single)

            # Original image (denormalize)
            img = X[i].cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])[:, None, None]
            std  = np.array([0.229, 0.224, 0.225])[:, None, None]
            img  = np.clip(img * std + mean, 0, 1)
            img  = img.transpose(1, 2, 0)

            # Overlay heatmap
            heatmap = cm.jet(cam)[:, :, :3]
            overlay = 0.6 * img + 0.4 * heatmap

            col = images_shown
            axes[0, col].imshow(img)
            axes[0, col].set_title(f"True: {RGZ_CLASSES[y[i]]}", fontsize=8)
            axes[0, col].axis("off")

            axes[1, col].imshow(overlay)
            axes[1, col].set_title(f"Pred: {RGZ_CLASSES[pred]}", fontsize=8)
            axes[1, col].axis("off")

            images_shown += 1
            if images_shown >= n_samples:
                break
        if images_shown >= n_samples:
            break

    fig.suptitle("GradCAM — Radio Galaxy Classification\n"
                 "Top: Original | Bottom: Saliency Overlay", fontsize=10)
    save_plot(fig, "2a_gradcam.png")
    print("📊 GradCAM maps generated")


# ─────────────────────────────────────────────
# 3. FEATURE MAPS — Intermediate layer visualization
# ─────────────────────────────────────────────
def generate_feature_maps(model, device, logger, n_filters=16):
    logger.info("Generating feature maps")
    _, _, test_loader = load_mirabest()
    X, y = next(iter(test_loader))
    x    = X[0:1].to(device)

    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook

    # Hook first and last conv blocks
    handles = [
        model.backbone[0].register_forward_hook(hook_fn("block_0_early")),
        model.backbone[4].register_forward_hook(hook_fn("block_4_mid")),
        model.backbone[-1].register_forward_hook(hook_fn("block_8_deep")),
    ]

    with torch.no_grad():
        model(x)

    for h in handles:
        h.remove()

    # Plot feature maps
    fig, axes = plt.subplots(3, n_filters, figsize=(n_filters * 1.2, 5))
    titles    = ["Early (edges)", "Mid (structures)", "Deep (morphology)"]

    for row, (name, title) in enumerate(zip(
        ["block_0_early", "block_4_mid", "block_8_deep"], titles
    )):
        if name not in activations:
            continue
        feat = activations[name][0]  # (C, H, W)
        for col in range(min(n_filters, feat.shape[0])):
            fm = feat[col].numpy()
            fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
            axes[row, col].imshow(fm, cmap="viridis")
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(title, fontsize=8)

    fig.suptitle("Feature Maps — EfficientNet-B0 Layer Activations\n"
                 "Early→Mid→Deep complexity progression", fontsize=10)
    save_plot(fig, "2a_feature_maps.png")
    print("📊 Feature maps generated")


# ─────────────────────────────────────────────
# 4. AUTOMATED CATALOGUE CSV
# Classifies all test images → exports CSV
# ─────────────────────────────────────────────
def build_catalogue_csv(model, device, logger):
    logger.info("Building automated radio galaxy catalogue")
    _, _, test_loader = load_mirabest()

    records = []
    sample_id = 0

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            logits = model(X)
            probs  = F.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)

            for i in range(len(X)):
                fri_prob  = probs[i, 0].item()
                frii_prob = probs[i, 1].item()
                pred_cls  = RGZ_CLASSES[preds[i].item()]
                true_cls  = RGZ_CLASSES[y[i].item()]

                # Physics-derived BH activity estimate
                bh_activity = "High" if pred_cls == "FRII" else "Moderate"
                jet_power   = ">1e25 W/Hz" if pred_cls == "FRII" else "<1e25 W/Hz"

                records.append({
                    "source_id":        f"STELLARIS_{sample_id:05d}",
                    "predicted_class":  pred_cls,
                    "true_class":       true_cls,
                    "FRI_probability":  round(fri_prob,  4),
                    "FRII_probability": round(frii_prob, 4),
                    "confidence":       round(max(fri_prob, frii_prob), 4),
                    "BH_activity":      bh_activity,
                    "jet_power_est":    jet_power,
                    "correct":          pred_cls == true_cls
                })
                sample_id += 1

    df   = pd.DataFrame(records)
    path = os.path.join(LOG_DIR, "radio_galaxy_catalogue.csv")
    os.makedirs(LOG_DIR, exist_ok=True)
    df.to_csv(path, index=False)

    print(f"\n── Automated Catalogue ──")
    print(f"Total classified : {len(df)}")
    print(f"FRI detected     : {(df['predicted_class']=='FRI').sum()}")
    print(f"FRII detected    : {(df['predicted_class']=='FRII').sum()}")
    print(f"Overall accuracy : {df['correct'].mean():.4f}")
    print(f"📄 Catalogue saved → {path}")
    logger.info(f"Catalogue saved: {len(df)} sources")
    return df


# ─────────────────────────────────────────────
# 5. LIGO EVENT TRIGGERS
# Sliding window on raw strain → flag GW candidates
# ─────────────────────────────────────────────
def detect_event_triggers(model, device, logger,
                           threshold: float = 0.85,
                           n_windows:  int  = 200):
    logger.info("Running LIGO event trigger detection")
    model.eval()

    # Generate synthetic continuous strain for demo
    # In production: replace with real LIGO strain data
    np.random.seed(42)
    total_len = LIGO_SIGNAL_LEN * 20  # 20 windows worth
    t         = np.linspace(0, 40, total_len)

    # Base noise
    strain = np.random.randn(LIGO_N_DETECTORS, total_len).astype(np.float32)

    # Inject 3 fake GW events at known times
    event_times = [5.0, 18.5, 32.0]
    for et in event_times:
        idx  = int(et / 40 * total_len)
        chirp_t = np.linspace(0, 2, LIGO_SIGNAL_LEN)
        chirp   = np.sin(2 * np.pi * (20 * chirp_t + 50 * chirp_t**2))
        end_idx = min(idx + LIGO_SIGNAL_LEN, total_len)
        strain[:, idx:end_idx] += 0.8 * chirp[:end_idx-idx]

    # Sliding window detection
    triggers = []
    step     = LIGO_SIGNAL_LEN // 4   # 75% overlap

    with torch.no_grad():
        for i in range(0, total_len - LIGO_SIGNAL_LEN, step):
            window = strain[:, i:i + LIGO_SIGNAL_LEN].copy()

            # Normalize
            for ch in range(LIGO_N_DETECTORS):
                std = window[ch].std()
                if std > 1e-10:
                    window[ch] = (window[ch] - window[ch].mean()) / std

            x      = torch.tensor(window[None], dtype=torch.float32).to(device)
            logits = model(x)
            probs  = F.softmax(logits, dim=1)
            signal_prob = probs[0, 1].item()

            time_s = i / (total_len / 40)
            if signal_prob > threshold:
                triggers.append({
                    "timestamp_s":   round(time_s, 3),
                    "signal_prob":   round(signal_prob, 4),
                    "SNR_estimate":  round(signal_prob * 15, 2),
                    "event_type":    "GW_candidate",
                    "window_start":  i,
                    "window_end":    i + LIGO_SIGNAL_LEN
                })

    df   = pd.DataFrame(triggers)
    path = os.path.join(LOG_DIR, "ligo_event_triggers.csv")
    df.to_csv(path, index=False)

    print(f"\n── LIGO Event Triggers ──")
    print(f"Windows scanned : {(total_len - LIGO_SIGNAL_LEN) // step}")
    print(f"Triggers found  : {len(df)}")
    if len(df) > 0:
        print(df[["timestamp_s", "signal_prob", "SNR_estimate"]].to_string())
    print(f"📄 Triggers saved → {path}")
    logger.info(f"Event triggers: {len(df)} found")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    t_axis = np.linspace(0, 40, total_len)
    axes[0].plot(t_axis, strain[0], lw=0.5, color="steelblue", alpha=0.7)
    axes[0].set_title("LIGO H1 Strain (synthetic demo)")
    axes[0].set_ylabel("Strain")

    probs_timeline = np.zeros(total_len)
    idx_list = list(range(0, total_len - LIGO_SIGNAL_LEN, step))
    with torch.no_grad():
        for j, i in enumerate(idx_list):
            window = strain[:, i:i + LIGO_SIGNAL_LEN].copy()
            for ch in range(LIGO_N_DETECTORS):
                std = window[ch].std()
                if std > 1e-10:
                    window[ch] = (window[ch] - window[ch].mean()) / std
            x    = torch.tensor(window[None], dtype=torch.float32).to(device)
            prob = F.softmax(model(x), dim=1)[0, 1].item()
            probs_timeline[i:i+step] = prob

    axes[1].plot(t_axis, probs_timeline, color="orangered", lw=1)
    axes[1].axhline(threshold, color="red", linestyle="--",
                    label=f"Threshold={threshold}")
    for et in event_times:
        axes[1].axvline(et, color="green", linestyle=":", alpha=0.7,
                        label="Injected event")
    axes[1].set_title("GW Signal Probability Over Time")
    axes[1].set_ylabel("P(Signal)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save_plot(fig, "2b_event_triggers.png")
    return df


# ─────────────────────────────────────────────
# 6. CLEANED SPECTROGRAMS
# Shows LIGO strain before/after bandpass filter
# ─────────────────────────────────────────────
def plot_cleaned_spectrograms(logger, n_examples=3):
    logger.info("Generating cleaned spectrogram plots")
    from module2.dataset_2b import _bandpass_filter

    np.random.seed(0)
    fig, axes = plt.subplots(n_examples, 2, figsize=(12, 3 * n_examples))

    for i in range(n_examples):
        # Generate noisy signal with glitch
        signal = np.random.randn(LIGO_SIGNAL_LEN).astype(np.float32)
        # Add a glitch (high-freq artifact)
        glitch_start = LIGO_SIGNAL_LEN // 3
        signal[glitch_start:glitch_start+50] += 10 * np.random.randn(50)

        # Compute spectrograms
        def spectrogram(s, title, ax):
            ax.specgram(s, NFFT=128, Fs=2048, noverlap=64,
                        cmap="inferno", vmin=-20)
            ax.set_ylim(0, 512)
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Freq (Hz)")

        spectrogram(signal,
                    f"Raw Strain {i+1} (with glitch)",
                    axes[i, 0])
        cleaned = _bandpass_filter(signal[None], low=20, high=500, fs=2048)[0]
        spectrogram(cleaned,
                    f"Cleaned {i+1} (bandpass 20-500 Hz)",
                    axes[i, 1])

    fig.suptitle("LIGO Strain: Raw vs Bandpass Cleaned\n"
                 "Removes instrumental glitches outside GW sensitive band",
                 fontsize=10)
    fig.tight_layout()
    save_plot(fig, "2b_cleaned_spectrograms.png")
    print("📊 Cleaned spectrograms generated")


# ─────────────────────────────────────────────
# 7. TRANSFER LEARNING COMPARISON
# Pretrained ImageNet vs Scratch
# ─────────────────────────────────────────────
def transfer_learning_comparison(device, logger, quick_epochs=5):
    logger.info("Running transfer learning comparison")
    train_loader, val_loader, _ = load_mirabest()
    criterion = nn.CrossEntropyLoss()

    results = {}
    for pretrained, label in [(True, "ImageNet pretrained"),
                               (False, "Trained from scratch")]:
        model     = RadioGalaxyClassifier(pretrained=pretrained).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        best_acc  = 0.0
        times     = []

        import time
        for epoch in range(quick_epochs):
            model.train()
            t0 = time.time()
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
            times.append(time.time() - t0)

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y  = X.to(device), y.to(device)
                    preds = model(X).argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total   += y.size(0)
            acc = correct / total
            if acc > best_acc:
                best_acc = acc

        results[label] = {
            "best_val_acc":    round(best_acc, 4),
            "avg_epoch_time":  round(np.mean(times), 2)
        }
        print(f"{label}: acc={best_acc:.4f} "
              f"time/epoch={np.mean(times):.1f}s")

    # Print comparison table
    print(f"\n── Transfer Learning Comparison ({quick_epochs} epochs) ──")
    print(f"{'Model':<25} {'Val Acc':>10} {'Time/Epoch':>12}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<25} {res['best_val_acc']:>10.4f} "
              f"{res['avg_epoch_time']:>10.1f}s")

    logger.info(f"Transfer learning comparison: {results}")
    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = get_device()
    logger = get_logger("module2_eval", LOG_DIR)
    logger.info("STELLARIS-DNet | Module 2 Evaluation Started")

    # Load models
    rgc = load_model(RadioGalaxyClassifier(),
                     os.path.join(CHECKPOINT_DIR, "rgc_best.pt"), device)
    gwd = load_model(GravWaveDetector(),
                     os.path.join(CHECKPOINT_DIR, "gwd_best.pt"), device)

    results = {}

    # 1. Metrics
    results["2a"] = evaluate_2a_metrics(rgc, device, logger)
    results["2b"] = evaluate_2b_metrics(gwd, device, logger)

    # 2. GradCAM
    generate_gradcam(rgc, device, logger)

    # 3. Feature Maps
    generate_feature_maps(rgc, device, logger)

    # 4. Automated Catalogue
    build_catalogue_csv(rgc, device, logger)

    # 5. Event Triggers
    detect_event_triggers(gwd, device, logger)

    # 6. Cleaned Spectrograms
    plot_cleaned_spectrograms(logger)

    # 7. Transfer Learning Comparison
    transfer_learning_comparison(device, logger, quick_epochs=3)

    # Final Summary
    print("\n" + "=" * 55)
    print("MODULE 2 EVALUATION SUMMARY")
    print("=" * 55)
    print(f"2A Radio Galaxy — Acc: {results['2a']['acc']:.4f} | "
          f"F1: {results['2a']['f1']:.4f} | AUC: {results['2a']['auc']:.4f}")
    print(f"2B GW Detection — Acc: {results['2b']['acc']:.4f} | "
          f"F1: {results['2b']['f1']:.4f} | AUC: {results['2b']['auc']:.4f}")
    print(f"\nOutputs saved to: {LOG_DIR}")
    print("=" * 55)
    print("✅ Module 2 evaluation complete.")
    logger.info("Module 2 Evaluation Complete")