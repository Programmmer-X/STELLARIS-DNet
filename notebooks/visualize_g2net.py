"""
notebooks/visualize_g2net.py
STELLARIS-DNet — G2Net Gravitational Wave Visualization
Run: python notebooks/visualize_g2net.py
Saves visualizations to notebooks/g2net_visuals/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module2.config import *
from module2.dataset_2b import (
    signal_to_spectrogram, _whiten, _bandpass,
    _fast_cqt, _normalize_spec,
    _resolve_data_dir, _get_file_path
)

OUTPUT_DIR = "notebooks/g2net_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# LOAD SAMPLE SIGNALS FROM G2NET
# ─────────────────────────────────────────────
def load_sample_signals(data_dir: str, n_each: int = 5):
    """Load n_each signal + n_each noise samples."""
    labels_path = os.path.join(data_dir, "training_labels.csv")
    if not os.path.exists(labels_path):
        print("⚠️  G2Net not found — using synthetic signals")
        return _generate_synthetic(n_each)

    df    = pd.read_csv(labels_path)
    noise = df[df["target"] == 0].sample(n_each, random_state=42)
    sig   = df[df["target"] == 1].sample(n_each, random_state=42)

    signals, labels = [], []
    for _, row in pd.concat([noise, sig]).iterrows():
        try:
            path = _get_file_path(row["id"], data_dir)
            s    = np.load(path).astype(np.float32)
            signals.append(s)
            labels.append(row["target"])
        except: continue

    print(f"✅ Loaded {len(signals)} G2Net signals")
    return np.array(signals), np.array(labels)


def _generate_synthetic(n_each: int = 5):
    """Generate synthetic signals when G2Net not available."""
    print("   Generating synthetic signals for visualization")
    signals, labels = [], []
    fs = LIGO_SAMPLE_RATE
    n  = LIGO_SIGNAL_LEN
    t  = np.linspace(0, 2, n)

    for _ in range(n_each):
        # Noise only
        noise = np.zeros((LIGO_N_DETECTORS, n), dtype=np.float32)
        for ch in range(LIGO_N_DETECTORS):
            raw  = np.random.randn(n).astype(np.float32)
            fft  = np.fft.rfft(raw)
            freq = np.fft.rfftfreq(n)
            freq[0] = 1e-10
            noise[ch] = np.fft.irfft(fft/np.sqrt(np.abs(freq)), n)
        signals.append(noise); labels.append(0)

    for _ in range(n_each):
        # Signal + noise (GW chirp)
        noise = signals[0].copy()
        chirp = (np.sin(2*np.pi*(20*t + 70*t**2)) *
                 np.exp(3*(t-2))).astype(np.float32)
        chirp /= (chirp.std() + 1e-10)
        s = noise.copy()
        for ch in range(LIGO_N_DETECTORS):
            offset = np.random.randint(-20, 20)
            s[ch]  = noise[ch] + 0.7 * np.roll(chirp, offset)
        signals.append(s); labels.append(1)

    return np.array(signals), np.array(labels)


# ─────────────────────────────────────────────
# PLOT 1: RAW STRAIN TIME SERIES
# ─────────────────────────────────────────────
def plot_raw_strain(signals, labels):
    print("Generating raw strain comparison...")
    noise_idx  = np.where(labels == 0)[0][:3]
    signal_idx = np.where(labels == 1)[0][:3]

    fig, axes = plt.subplots(6, LIGO_N_DETECTORS,
                              figsize=(15, 12))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle("G2Net Gravitational Wave Strain Data\n"
                 "Raw whitened strain from 3 LIGO/Virgo detectors",
                 color="white", fontsize=14, fontweight="bold")

    det_names = ["LIGO Hanford (H1)",
                 "LIGO Livingston (L1)",
                 "Virgo (V1)"]
    t_axis = np.linspace(0, LIGO_SIGNAL_LEN / LIGO_SAMPLE_RATE,
                          LIGO_SIGNAL_LEN)

    for row, idx in enumerate(list(noise_idx) + list(signal_idx)):
        label  = labels[idx]
        color  = "#e74c3c" if label == 1 else "#3498db"
        lstr   = "Signal (GW)" if label == 1 else "Noise"

        for ch in range(LIGO_N_DETECTORS):
            ax = axes[row, ch]
            ax.set_facecolor("#0d0d1a")
            ax.plot(t_axis, signals[idx, ch],
                    color=color, lw=0.6, alpha=0.8)
            ax.tick_params(colors="white", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")

            if row == 0:
                ax.set_title(det_names[ch],
                             color="white", fontsize=9)
            if ch == 0:
                ax.set_ylabel(f"{lstr}\n#{row%3+1}",
                              color=color, fontsize=8)
            if row == 5:
                ax.set_xlabel("Time (s)", color="white", fontsize=8)

    # Legend
    fig.text(0.1, 0.01, "■ Noise",  color="#3498db", fontsize=11)
    fig.text(0.2, 0.01, "■ Signal", color="#e74c3c", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    path = os.path.join(OUTPUT_DIR, "1_raw_strain_comparison.png")
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   ✅ Saved: {path}")


# ─────────────────────────────────────────────
# PLOT 2: PROCESSING PIPELINE
# Raw → Whitened → Bandpass → CQT Spectrogram
# ─────────────────────────────────────────────
def plot_processing_pipeline(signals, labels):
    print("Generating processing pipeline...")

    sig_idx   = np.where(labels == 1)[0][0]
    signal    = signals[sig_idx]
    det_names = ["H1", "L1", "V1"]

    whitened  = _whiten(signal)
    cleaned   = _bandpass(whitened)
    spec      = signal_to_spectrogram(signal)

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle("GW Signal Processing Pipeline\n"
                 "Raw Strain → CQT Spectrogram",
                 color="white", fontsize=15, fontweight="bold")

    gs   = gridspec.GridSpec(3, 4, figure=fig,
                              hspace=0.4, wspace=0.3)
    t_ax = np.linspace(0, LIGO_SIGNAL_LEN / LIGO_SAMPLE_RATE,
                        LIGO_SIGNAL_LEN)
    titles = ["1. Raw Strain", "2. Whitened",
              "3. Bandpass 20-500Hz", "4. CQT Spectrogram"]
    cmaps  = [None, None, None, "inferno"]

    for ch in range(LIGO_N_DETECTORS):
        for step, (data, title) in enumerate(zip(
            [signal[ch], whitened[ch], cleaned[ch], None], titles
        )):
            ax = fig.add_subplot(gs[ch, step])
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(colors="white", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#333355")

            if step < 3:
                ax.plot(t_ax, data, color="#4ECDC4", lw=0.6)
                ax.set_xlim(0, t_ax[-1])
                if ch == 0:
                    ax.set_title(title, color="white", fontsize=9)
                if step == 0:
                    ax.set_ylabel(det_names[ch],
                                  color="white", fontsize=9)
            else:
                # Spectrogram
                im = ax.imshow(spec[ch], aspect="auto",
                               origin="lower", cmap="inferno",
                               extent=[0, LIGO_SIGNAL_LEN/LIGO_SAMPLE_RATE,
                                       GW_FREQ_MIN, GW_FREQ_MAX])
                ax.set_ylabel("Freq (Hz)", color="white", fontsize=7)
                if ch == 0:
                    ax.set_title(title, color="white", fontsize=9)
                    # Annotate chirp
                    ax.annotate("GW Chirp\n(curved track)",
                                xy=(0.5, 100), color="yellow",
                                fontsize=7, fontweight="bold",
                                arrowprops=dict(
                                    arrowstyle="->",
                                    color="yellow"))

    path = os.path.join(OUTPUT_DIR, "2_processing_pipeline.png")
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   ✅ Saved: {path}")


# ─────────────────────────────────────────────
# PLOT 3: CQT SPECTROGRAM GALLERY
# Signal vs Noise side by side
# ─────────────────────────────────────────────
def plot_cqt_gallery(signals, labels, n=4):
    print("Generating CQT spectrogram gallery...")

    noise_idx  = np.where(labels == 0)[0][:n]
    signal_idx = np.where(labels == 1)[0][:n]

    fig, axes = plt.subplots(2, n, figsize=(5*n, 10))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle("CQT Spectrograms: Noise vs GW Signal\n"
                 "GW chirp appears as curved track sweeping from low → high freq",
                 color="white", fontsize=13, fontweight="bold")

    for col, (n_idx, s_idx) in enumerate(
        zip(noise_idx, signal_idx)
    ):
        for row, (idx, label, color) in enumerate([
            (n_idx, "Noise",    "#3498db"),
            (s_idx, "Signal",   "#e74c3c")
        ]):
            ax   = axes[row, col]
            ax.set_facecolor("#0d0d1a")
            spec = signal_to_spectrogram(signals[idx])

            # Show H1 channel
            im = ax.imshow(spec[0], aspect="auto", origin="lower",
                           cmap="inferno",
                           extent=[0, 2, GW_FREQ_MIN, GW_FREQ_MAX])

            ax.set_title(f"{label} #{col+1}",
                         color=color, fontsize=10, fontweight="bold")
            ax.tick_params(colors="white", labelsize=7)
            ax.set_xlabel("Time (s)", color="white", fontsize=8)
            if col == 0:
                ax.set_ylabel("Frequency (Hz)", color="white", fontsize=8)

            # Border color
            for sp in ax.spines.values():
                sp.set_edgecolor(color)
                sp.set_linewidth(2)
                sp.set_visible(True)

    path = os.path.join(OUTPUT_DIR, "3_cqt_gallery.png")
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   ✅ Saved: {path}")


# ─────────────────────────────────────────────
# PLOT 4: FREQUENCY ANALYSIS
# PSD of signal vs noise
# ─────────────────────────────────────────────
def plot_frequency_analysis(signals, labels):
    print("Generating frequency analysis...")

    noise_sigs  = signals[labels == 0]
    signal_sigs = signals[labels == 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle("Gravitational Wave Signal Frequency Analysis",
                 color="white", fontsize=13, fontweight="bold")

    freqs = np.fft.rfftfreq(LIGO_SIGNAL_LEN, d=1.0/LIGO_SAMPLE_RATE)
    mask  = (freqs >= GW_FREQ_MIN) & (freqs <= GW_FREQ_MAX)

    for ax in axes:
        ax.set_facecolor("#111122")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")

    # PSD comparison
    for sigs, label, color in [
        (noise_sigs,  "Noise",  "#3498db"),
        (signal_sigs, "Signal", "#e74c3c")
    ]:
        psds = []
        for s in sigs[:5]:
            fft = np.fft.rfft(s[0])
            psds.append(np.abs(fft) ** 2)
        psd_mean = np.mean(psds, axis=0)
        axes[0].semilogy(freqs[mask], psd_mean[mask],
                         color=color, lw=2, label=label, alpha=0.8)

    axes[0].axvspan(GW_FREQ_MIN, GW_FREQ_MAX, alpha=0.1,
                    color="yellow", label="GW band")
    axes[0].set_xlabel("Frequency (Hz)", color="white")
    axes[0].set_ylabel("Power Spectral Density", color="white")
    axes[0].set_title("PSD: Noise vs GW Signal", color="white")
    axes[0].legend(facecolor="#111122", labelcolor="white")

    # Chirp in time domain
    t = np.linspace(0, 2, LIGO_SIGNAL_LEN)
    chirp = (np.sin(2*np.pi*(20*t + 70*t**2)) *
             np.exp(3*(t-2))).astype(np.float32)
    axes[1].plot(t[-512:],
                 chirp[-512:], color="#e74c3c", lw=2)
    axes[1].fill_between(t[-512:], chirp[-512:],
                          alpha=0.3, color="#e74c3c")
    axes[1].set_xlabel("Time (s)", color="white")
    axes[1].set_ylabel("Strain", color="white")
    axes[1].set_title("GW Chirp Signal (merger moment)\n"
                       "Frequency increases as BHs spiral inward",
                       color="white")
    axes[1].tick_params(colors="white")

    path = os.path.join(OUTPUT_DIR, "4_frequency_analysis.png")
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   ✅ Saved: {path}")


# ─────────────────────────────────────────────
# PLOT 5: DETECTOR CORRELATION
# Cross-correlation between H1, L1, V1
# ─────────────────────────────────────────────
def plot_detector_correlation(signals, labels):
    print("Generating detector correlation...")

    sig_idx = np.where(labels == 1)[0][0]
    signal  = signals[sig_idx]
    cleaned = _bandpass(_whiten(signal))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle("Multi-Detector Correlation\n"
                 "GW signal arrives at all detectors with small time offset",
                 color="white", fontsize=13, fontweight="bold")

    det_names = ["H1", "L1", "V1"]
    t_axis    = np.linspace(0, LIGO_SIGNAL_LEN/LIGO_SAMPLE_RATE,
                             LIGO_SIGNAL_LEN)
    colors    = ["#FF6B35", "#4ECDC4", "#9B59B6"]

    # Individual channels
    for ch in range(LIGO_N_DETECTORS):
        ax = axes[0, ch]
        ax.set_facecolor("#0d0d1a")
        ax.plot(t_axis, cleaned[ch],
                color=colors[ch], lw=0.8, alpha=0.9)
        ax.set_title(det_names[ch], color=colors[ch],
                     fontsize=10, fontweight="bold")
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")
        ax.set_xlabel("Time (s)", color="white", fontsize=8)
        if ch == 0:
            ax.set_ylabel("Strain (cleaned)", color="white", fontsize=8)

    # Cross-correlations
    pairs  = [(0, 1, "H1 × L1"), (0, 2, "H1 × V1"), (1, 2, "L1 × V1")]
    lags   = np.arange(-100, 101)
    lag_ms = lags / LIGO_SAMPLE_RATE * 1000  # ms

    for i, (a, b, title) in enumerate(pairs):
        ax = axes[1, i]
        ax.set_facecolor("#0d0d1a")

        # Cross-correlation
        xcorr = np.correlate(
            cleaned[a] - cleaned[a].mean(),
            cleaned[b] - cleaned[b].mean(),
            mode="full"
        )
        center = len(xcorr) // 2
        xcorr  = xcorr[center-100:center+101]
        xcorr  = xcorr / (xcorr.std() + 1e-10)

        ax.plot(lag_ms, xcorr, color="#F39C12", lw=1.5)
        ax.axvline(0, color="white", ls="--", lw=0.5, alpha=0.5)
        ax.set_title(f"Cross-corr: {title}",
                     color="white", fontsize=9)
        ax.set_xlabel("Lag (ms)", color="white", fontsize=8)
        if i == 0:
            ax.set_ylabel("Correlation", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")

    path = os.path.join(OUTPUT_DIR, "5_detector_correlation.png")
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   ✅ Saved: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 55)
    print("STELLARIS-DNet — G2Net Visualization")
    print("=" * 55)

    data_dir = _resolve_data_dir(LIGO_DATA_DIR)
    signals, labels = load_sample_signals(data_dir, n_each=5)

    print(f"\nGenerating visualizations → {OUTPUT_DIR}/")
    print("-" * 55)

    plot_raw_strain(signals, labels)
    plot_processing_pipeline(signals, labels)
    plot_cqt_gallery(signals, labels, n=4)
    plot_frequency_analysis(signals, labels)
    plot_detector_correlation(signals, labels)

    print("\n" + "=" * 55)
    print("✅ All G2Net visualizations saved!")
    print(f"   Location: {OUTPUT_DIR}/")
    print("\nFiles generated:")
    print("  1_raw_strain_comparison.png  ← show this first")
    print("  2_processing_pipeline.png    ← CQT pipeline steps")
    print("  3_cqt_gallery.png            ← noise vs signal")
    print("  4_frequency_analysis.png     ← PSD + chirp")
    print("  5_detector_correlation.png   ← H1/L1/V1 correlation")
    print("=" * 55)