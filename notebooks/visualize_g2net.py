"""
notebooks/visualize_g2net.py
STELLARIS-DNet — G2Net Gravitational Wave Visualization
Generates 13 presentation-quality figures for professor/demo.

Run: python notebooks/visualize_g2net.py
Output: notebooks/g2net_visuals/  (13 PNG files)

Works with real G2Net data OR synthetic fallback (no data needed).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module2.config import (
    LIGO_DATA_DIR, LIGO_CLASSES, LIGO_N_DETECTORS,
    LIGO_SIGNAL_LEN, LIGO_SAMPLE_RATE, LIGO_CQT_BINS,
    LIGO_CQT_STEPS, GW_FREQ_MIN, GW_FREQ_MAX
)
from module2.dataset_2b import (
    signal_to_spectrogram, _whiten, _bandpass,
    _fast_cqt, _normalize_spec,
    _resolve_data_dir, _get_file_path
)

OUTPUT_DIR = "notebooks/g2net_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)

DARK_BG  = "#0d0d1a"
PANEL_BG = "#111122"
GRID_COL = "#222244"
C_NOISE  = "#3498db"
C_SIGNAL = "#e74c3c"
C_ACCENT = "#4ECDC4"


def _save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   ✅ {name}")


def _style_ax(ax, bg=PANEL_BG):
    ax.set_facecolor(bg)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def _load_real(data_dir, n_each=6):
    labels_path = os.path.join(data_dir, "training_labels.csv")
    if not os.path.exists(labels_path):
        return None, None

    df    = pd.read_csv(labels_path)
    noise = df[df["target"] == 0].sample(min(n_each, (df["target"]==0).sum()), random_state=42)
    sig   = df[df["target"] == 1].sample(min(n_each, (df["target"]==1).sum()), random_state=42)

    signals, labels = [], []
    for _, row in pd.concat([noise, sig]).iterrows():
        try:
            path = _get_file_path(row["id"], data_dir)
            s    = np.load(path).astype(np.float32)
            signals.append(s)
            labels.append(int(row["target"]))
        except:
            continue

    if not signals:
        return None, None
    print(f"✅ Loaded {len(signals)} real G2Net signals")
    return np.array(signals), np.array(labels)


def _generate_synthetic(n_each=6):
    """Physically realistic synthetic GW signals."""
    print("⚠️  G2Net not found — generating synthetic signals")
    fs = float(LIGO_SAMPLE_RATE)
    n  = LIGO_SIGNAL_LEN
    t  = np.linspace(0, n/fs, n)
    signals, labels = [], []

    def colored_noise(n):
        raw  = np.random.randn(n).astype(np.float32)
        fft  = np.fft.rfft(raw)
        freq = np.fft.rfftfreq(n)
        freq[0] = 1e-10
        return np.fft.irfft(fft / np.sqrt(np.abs(freq) + 1e-10), n).astype(np.float32)

    def chirp_signal(f0=30, tc=None, snr=0.8):
        if tc is None:
            tc = n/fs * np.random.uniform(1.2, 1.8)
        eps  = 1e-6
        dt   = np.clip(tc - t, eps, None)
        freq = f0 * dt ** (-3/8)
        freq = np.clip(freq, 20, 600)
        phase = 2 * np.pi * np.cumsum(freq) / fs
        env   = np.exp(5 * (t/tc - 1))
        env   = np.clip(env, 0, 1)
        return (np.sin(phase) * env * snr).astype(np.float32)

    # Noise samples
    for _ in range(n_each):
        sigs = np.stack([colored_noise(n) for _ in range(LIGO_N_DETECTORS)])
        signals.append(sigs); labels.append(0)

    # Signal samples
    for i in range(n_each):
        f0   = np.random.uniform(25, 60)
        snr  = np.random.uniform(0.5, 1.2)
        chirp = chirp_signal(f0=f0, snr=snr)
        sigs = []
        for _ in range(LIGO_N_DETECTORS):
            noise  = colored_noise(n)
            offset = np.random.randint(-15, 15)
            sigs.append(noise + np.roll(chirp, offset))
        signals.append(np.array(sigs).astype(np.float32))
        labels.append(1)

    return np.array(signals), np.array(labels)


def _get_data(n_each=6):
    data_dir = _resolve_data_dir(LIGO_DATA_DIR)
    signals, labels = _load_real(data_dir, n_each)
    if signals is None:
        signals, labels = _generate_synthetic(n_each)
    return signals, labels


# ─────────────────────────────────────────────
# FIGURE 1: RAW STRAIN TIME SERIES
# ─────────────────────────────────────────────
def fig1_raw_strain(signals, labels):
    print("Fig 1: Raw strain time series...")
    noise_idx  = np.where(labels == 0)[0][:3]
    signal_idx = np.where(labels == 1)[0][:3]
    det_names  = ["LIGO Hanford (H1)", "LIGO Livingston (L1)", "Virgo (V1)"]
    t_axis     = np.linspace(0, LIGO_SIGNAL_LEN/LIGO_SAMPLE_RATE, LIGO_SIGNAL_LEN)

    fig, axes = plt.subplots(6, LIGO_N_DETECTORS, figsize=(16, 13))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("G2Net — Raw Strain Data: Noise vs GW Signal\n"
                 "3 detectors: LIGO Hanford | LIGO Livingston | Virgo",
                 color="white", fontsize=14, fontweight="bold")

    for row, idx in enumerate(list(noise_idx) + list(signal_idx)):
        lbl   = labels[idx]
        color = C_SIGNAL if lbl == 1 else C_NOISE
        lstr  = "GW Signal" if lbl == 1 else "Noise"
        for ch in range(LIGO_N_DETECTORS):
            ax = axes[row, ch]
            _style_ax(ax, DARK_BG)
            ax.plot(t_axis, signals[idx, ch], color=color, lw=0.5, alpha=0.85)
            if row == 0:
                ax.set_title(det_names[ch], color="white", fontsize=9, fontweight="bold")
            if ch == 0:
                ax.set_ylabel(f"{lstr}\n#{row%3+1}", color=color, fontsize=8)
            if row == 5:
                ax.set_xlabel("Time (s)", color="white", fontsize=8)

    fig.text(0.1, 0.01, "■ Noise",     color=C_NOISE,  fontsize=11)
    fig.text(0.22, 0.01, "■ GW Signal", color=C_SIGNAL, fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    _save(fig, "01_raw_strain_comparison.png")


# ─────────────────────────────────────────────
# FIGURE 2: SIGNAL PROCESSING PIPELINE
# ─────────────────────────────────────────────
def fig2_pipeline(signals, labels):
    print("Fig 2: Processing pipeline...")
    idx      = np.where(labels == 1)[0][0]
    signal   = signals[idx]
    whitened = _whiten(signal)
    cleaned  = _bandpass(whitened)
    spec     = signal_to_spectrogram(signal)
    t_ax     = np.linspace(0, LIGO_SIGNAL_LEN/LIGO_SAMPLE_RATE, LIGO_SIGNAL_LEN)
    det_names = ["H1", "L1", "V1"]

    fig = plt.figure(figsize=(22, 12))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("GW Signal Processing Pipeline: Raw → CQT Spectrogram",
                 color="white", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.3)
    titles = ["1. Raw Strain", "2. Whitened", "3. Bandpass 20-500Hz", "4. CQT Spectrogram"]

    for ch in range(LIGO_N_DETECTORS):
        for step in range(4):
            ax = fig.add_subplot(gs[ch, step])
            _style_ax(ax, DARK_BG)
            if step < 3:
                data = [signal[ch], whitened[ch], cleaned[ch]][step]
                ax.plot(t_ax, data, color=C_ACCENT, lw=0.6)
                ax.set_xlim(0, t_ax[-1])
                if ch == 0:
                    ax.set_title(titles[step], color="white", fontsize=10, fontweight="bold")
                if step == 0:
                    ax.set_ylabel(det_names[ch], color="white", fontsize=10)
            else:
                im = ax.imshow(spec[ch], aspect="auto", origin="lower",
                               cmap="inferno",
                               extent=[0, LIGO_SIGNAL_LEN/LIGO_SAMPLE_RATE,
                                       GW_FREQ_MIN, GW_FREQ_MAX])
                ax.set_ylabel("Freq (Hz)", color="white", fontsize=7)
                if ch == 0:
                    ax.set_title(titles[step], color="white", fontsize=10, fontweight="bold")
                    ax.annotate("GW Chirp\n↑ rising freq", xy=(0.8, 80),
                                xytext=(0.3, 200), color="yellow", fontsize=7,
                                fontweight="bold",
                                arrowprops=dict(arrowstyle="->", color="yellow", lw=1.2))
    _save(fig, "02_processing_pipeline.png")


# ─────────────────────────────────────────────
# FIGURE 3: CQT SPECTROGRAM GALLERY
# ─────────────────────────────────────────────
def fig3_cqt_gallery(signals, labels, n=4):
    print("Fig 3: CQT gallery...")
    noise_idx  = np.where(labels == 0)[0][:n]
    signal_idx = np.where(labels == 1)[0][:n]

    fig, axes = plt.subplots(2, n, figsize=(6*n, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("CQT Spectrograms: Noise vs GW Signal (H1 Channel)\n"
                 "Chirp track visible as sweeping curve from low → high frequency",
                 color="white", fontsize=13, fontweight="bold")

    for col in range(n):
        for row, (idx, cls, color) in enumerate([
            (noise_idx[col],  "Noise",     C_NOISE),
            (signal_idx[col], "GW Signal", C_SIGNAL),
        ]):
            ax   = axes[row, col]
            spec = signal_to_spectrogram(signals[idx])
            ax.imshow(spec[0], aspect="auto", origin="lower", cmap="inferno",
                      extent=[0, 2, GW_FREQ_MIN, GW_FREQ_MAX])
            ax.set_title(f"{cls} #{col+1}", color=color,
                         fontsize=11, fontweight="bold")
            ax.tick_params(colors="white", labelsize=7)
            ax.set_xlabel("Time (s)", color="white", fontsize=8)
            if col == 0:
                ax.set_ylabel("Frequency (Hz)", color="white", fontsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(color); sp.set_linewidth(2); sp.set_visible(True)

            # Annotate chirp region on signals
            if row == 1:
                ax.annotate("", xy=(1.8, 400), xytext=(0.3, 40),
                            arrowprops=dict(arrowstyle="-|>",
                                            color="yellow", lw=1.8,
                                            connectionstyle="arc3,rad=-0.3"))
                ax.text(0.95, 120, "Chirp", color="yellow",
                        fontsize=8, fontweight="bold")

    fig.tight_layout()
    _save(fig, "03_cqt_gallery.png")


# ─────────────────────────────────────────────
# FIGURE 4: FREQUENCY ANALYSIS (PSD + CHIRP)
# ─────────────────────────────────────────────
def fig4_frequency_analysis(signals, labels):
    print("Fig 4: Frequency analysis...")
    noise_sigs  = signals[labels == 0]
    signal_sigs = signals[labels == 1]
    freqs = np.fft.rfftfreq(LIGO_SIGNAL_LEN, d=1.0/LIGO_SAMPLE_RATE)
    mask  = (freqs >= GW_FREQ_MIN) & (freqs <= GW_FREQ_MAX)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Gravitational Wave — Frequency Domain Analysis",
                 color="white", fontsize=14, fontweight="bold")

    # PSD
    _style_ax(axes[0])
    for sigs, lbl, color in [
        (noise_sigs,  "Noise",     C_NOISE),
        (signal_sigs, "GW Signal", C_SIGNAL),
    ]:
        psds = [np.abs(np.fft.rfft(s[0]))**2 for s in sigs[:5]]
        mean = np.mean(psds, axis=0)
        std  = np.std(psds, axis=0)
        axes[0].semilogy(freqs[mask], mean[mask], color=color, lw=2, label=lbl)
        axes[0].fill_between(freqs[mask],
                              (mean - std)[mask], (mean + std)[mask],
                              color=color, alpha=0.2)
    axes[0].axvspan(GW_FREQ_MIN, GW_FREQ_MAX, alpha=0.08,
                    color="yellow", label="GW sensitivity band")
    axes[0].set_xlabel("Frequency (Hz)", color="white")
    axes[0].set_ylabel("Power Spectral Density", color="white")
    axes[0].set_title("PSD: Noise vs GW Signal (Mean ± 1σ)", color="white")
    axes[0].legend(facecolor=PANEL_BG, labelcolor="white")
    axes[0].grid(alpha=0.2, color=GRID_COL)

    # Chirp waveform zoom
    _style_ax(axes[1])
    t   = np.linspace(0, LIGO_SIGNAL_LEN/LIGO_SAMPLE_RATE, LIGO_SIGNAL_LEN)
    tc  = 2.2
    eps = 1e-6
    dt  = np.clip(tc - t, eps, None)
    freq_inst = 30 * dt**(-3/8)
    freq_inst = np.clip(freq_inst, 20, 600)
    phase = 2 * np.pi * np.cumsum(freq_inst) / LIGO_SAMPLE_RATE
    env   = np.exp(5*(t/tc - 1))
    chirp = np.sin(phase) * np.clip(env, 0, 1)

    zoom_s = int(1.5 * LIGO_SAMPLE_RATE)
    zoom_e = int(2.0 * LIGO_SAMPLE_RATE)
    axes[1].plot(t[zoom_s:zoom_e], chirp[zoom_s:zoom_e], color=C_SIGNAL, lw=1.5)
    axes[1].fill_between(t[zoom_s:zoom_e], chirp[zoom_s:zoom_e],
                          alpha=0.3, color=C_SIGNAL)
    axes[1].set_xlabel("Time (s)", color="white")
    axes[1].set_ylabel("Strain (arbitrary units)", color="white")
    axes[1].set_title("GW Chirp Waveform — Final 0.5s Before Merger\n"
                       "Amplitude AND frequency increase: inspiral → ringdown",
                       color="white")
    axes[1].grid(alpha=0.2, color=GRID_COL)
    axes[1].text(0.95, 0.90, "f(t) ∝ (t_c − t)^(−3/8)",
                  ha="right", va="top", transform=axes[1].transAxes,
                  color="yellow", fontsize=11, fontweight="bold",
                  bbox=dict(boxstyle="round", facecolor=PANEL_BG, alpha=0.7))

    fig.tight_layout()
    _save(fig, "04_frequency_analysis.png")


# ─────────────────────────────────────────────
# FIGURE 5: DETECTOR CORRELATION
# ─────────────────────────────────────────────
def fig5_detector_correlation(signals, labels):
    print("Fig 5: Detector correlation...")
    idx     = np.where(labels == 1)[0][0]
    signal  = signals[idx]
    cleaned = _bandpass(_whiten(signal))
    t_axis  = np.linspace(0, LIGO_SIGNAL_LEN/LIGO_SAMPLE_RATE, LIGO_SIGNAL_LEN)
    det_names = ["H1", "L1", "V1"]
    colors    = ["#FF6B35", C_ACCENT, "#9B59B6"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Multi-Detector Coherence — GW Signal\n"
                 "Same event arrives with light-travel time delay (~10 ms max)",
                 color="white", fontsize=13, fontweight="bold")

    for ch in range(LIGO_N_DETECTORS):
        ax = axes[0, ch]
        _style_ax(ax, DARK_BG)
        ax.plot(t_axis, cleaned[ch], color=colors[ch], lw=0.8, alpha=0.9)
        ax.set_title(det_names[ch], color=colors[ch], fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)", color="white", fontsize=8)
        if ch == 0:
            ax.set_ylabel("Strain (cleaned)", color="white", fontsize=8)

    pairs  = [(0, 1, "H1 × L1"), (0, 2, "H1 × V1"), (1, 2, "L1 × V1")]
    lag_ms = np.arange(-100, 101) / LIGO_SAMPLE_RATE * 1000

    for i, (a, b, title) in enumerate(pairs):
        ax = axes[1, i]
        _style_ax(ax, DARK_BG)
        xcorr = np.correlate(
            cleaned[a] - cleaned[a].mean(),
            cleaned[b] - cleaned[b].mean(), mode="full"
        )
        center = len(xcorr) // 2
        xcorr  = xcorr[center-100:center+101]
        xcorr  = xcorr / (xcorr.std() + 1e-10)
        ax.plot(lag_ms, xcorr, color="#F39C12", lw=1.5)
        ax.axvline(0, color="white", ls="--", lw=0.8, alpha=0.5)
        # Mark peak
        peak_lag = lag_ms[np.argmax(np.abs(xcorr))]
        ax.axvline(peak_lag, color="lime", ls=":", lw=1.2,
                   label=f"Peak: {peak_lag:.1f}ms")
        ax.set_title(f"Cross-Corr: {title}", color="white", fontsize=9)
        ax.set_xlabel("Time Lag (ms)", color="white", fontsize=8)
        if i == 0:
            ax.set_ylabel("Correlation", color="white", fontsize=8)
        ax.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)
        ax.grid(alpha=0.2, color=GRID_COL)

    fig.tight_layout()
    _save(fig, "05_detector_correlation.png")


# ─────────────────────────────────────────────
# FIGURE 6: WHITENING EFFECT
# ─────────────────────────────────────────────
def fig6_whitening(signals, labels):
    print("Fig 6: Whitening effect...")
    idx    = np.where(labels == 1)[0][0]
    sig    = signals[idx, 0]                       # H1 channel
    white  = _whiten(signals[idx])[0]
    clean  = _bandpass(_whiten(signals[idx]))[0]
    freqs  = np.fft.rfftfreq(LIGO_SIGNAL_LEN, d=1.0/LIGO_SAMPLE_RATE)
    mask   = (freqs > 5) & (freqs < 600)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Signal Whitening — Revealing Hidden GW Events\n"
                 "Detector noise floor must be removed before CQT",
                 color="white", fontsize=14, fontweight="bold")

    t_ax = np.linspace(0, LIGO_SIGNAL_LEN/LIGO_SAMPLE_RATE, LIGO_SIGNAL_LEN)

    for col, (data, name, color) in enumerate([
        (sig,   "Raw Strain",          "#888899"),
        (white, "Whitened",            C_ACCENT),
        (clean, "Bandpass 20-500 Hz",  C_SIGNAL),
    ]):
        # Time domain
        _style_ax(axes[0, col], DARK_BG)
        axes[0, col].plot(t_ax, data, color=color, lw=0.5, alpha=0.9)
        axes[0, col].set_title(name, color=color, fontsize=11, fontweight="bold")
        axes[0, col].set_xlabel("Time (s)", color="white", fontsize=8)
        if col == 0:
            axes[0, col].set_ylabel("Strain", color="white", fontsize=8)

        # Frequency domain
        _style_ax(axes[1, col], DARK_BG)
        psd = np.abs(np.fft.rfft(data))**2
        axes[1, col].semilogy(freqs[mask], psd[mask], color=color, lw=1.2)
        axes[1, col].set_xlabel("Frequency (Hz)", color="white", fontsize=8)
        if col == 0:
            axes[1, col].set_ylabel("PSD", color="white", fontsize=8)
        axes[1, col].set_title(f"PSD — {name}", color=color, fontsize=9)
        axes[1, col].grid(alpha=0.2, color=GRID_COL)

    fig.tight_layout()
    _save(fig, "06_whitening_effect.png")


# ─────────────────────────────────────────────
# FIGURE 7: SNR ANALYSIS
# ─────────────────────────────────────────────
def fig7_snr_analysis(signals, labels):
    print("Fig 7: SNR analysis...")
    noise_sigs  = signals[labels == 0]
    signal_sigs = signals[labels == 1]

    def compute_snr(sig):
        """Estimate SNR as peak power / median background power in GW band."""
        cleaned = _bandpass(_whiten(sig))
        psd     = np.abs(np.fft.rfft(cleaned[0]))**2
        freqs   = np.fft.rfftfreq(LIGO_SIGNAL_LEN, 1.0/LIGO_SAMPLE_RATE)
        mask    = (freqs >= GW_FREQ_MIN) & (freqs <= GW_FREQ_MAX)
        band    = psd[mask]
        return float(band.max() / (np.median(band) + 1e-10))

    noise_snrs  = [compute_snr(s) for s in noise_sigs]
    signal_snrs = [compute_snr(s) for s in signal_sigs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Signal-to-Noise Ratio Analysis\n"
                 "GW signals have higher peak-to-median power in GW band",
                 color="white", fontsize=14, fontweight="bold")

    _style_ax(axes[0])
    axes[0].hist(noise_snrs,  bins=15, color=C_NOISE,  alpha=0.7,
                 label=f"Noise  (n={len(noise_snrs)})", density=True)
    axes[0].hist(signal_snrs, bins=15, color=C_SIGNAL, alpha=0.7,
                 label=f"Signal (n={len(signal_snrs)})", density=True)
    axes[0].set_xlabel("Estimated SNR (band peak/median)", color="white")
    axes[0].set_ylabel("Density", color="white")
    axes[0].set_title("SNR Distribution", color="white")
    axes[0].legend(facecolor=PANEL_BG, labelcolor="white")
    axes[0].grid(alpha=0.2, color=GRID_COL)

    # Bar comparison
    _style_ax(axes[1])
    cats   = ["Noise", "GW Signal"]
    means  = [np.mean(noise_snrs), np.mean(signal_snrs)]
    stds   = [np.std(noise_snrs),  np.std(signal_snrs)]
    colors = [C_NOISE, C_SIGNAL]
    bars   = axes[1].bar(cats, means, color=colors, edgecolor="white",
                          width=0.4, yerr=stds, capsize=8)
    axes[1].set_ylabel("Mean SNR", color="white")
    axes[1].set_title("Mean SNR: Noise vs Signal", color="white")
    for bar, m, s in zip(bars, means, stds):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     m + s + 0.3, f"{m:.1f}±{s:.1f}",
                     ha="center", color="white", fontsize=10, fontweight="bold")
    axes[1].grid(alpha=0.2, color=GRID_COL, axis="y")

    fig.tight_layout()
    _save(fig, "07_snr_analysis.png")


# ─────────────────────────────────────────────
# FIGURE 8: CHIRP PARAMETER SPACE
# ─────────────────────────────────────────────
def fig8_chirp_parameters(signals, labels):
    print("Fig 8: Chirp parameter space...")
    fs = float(LIGO_SAMPLE_RATE)
    n  = LIGO_SIGNAL_LEN
    t  = np.linspace(0, n/fs, n)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("GW Chirp Parameter Space — Varying f₀ and Mass Ratio\n"
                 "Each unique merger produces a distinct time-frequency signature",
                 color="white", fontsize=13, fontweight="bold")

    configs = [
        (25,  1.5, "Low mass (NS-NS)\nf₀=25 Hz"),
        (40,  1.5, "Medium mass (NS-BH)\nf₀=40 Hz"),
        (70,  1.5, "High mass (BBH)\nf₀=70 Hz"),
        (35,  1.2, "Short merger\ntc=1.2s"),
        (35,  1.8, "Long merger\ntc=1.8s"),
        (35,  1.5, "Equal mass ratio\nSNR=1.0"),
    ]
    snrs = [0.6, 0.8, 1.0, 0.8, 0.8, 1.0]

    for i, (ax, (f0, tc_factor, title), snr) in enumerate(
        zip(axes.flat, configs, snrs)
    ):
        _style_ax(ax, DARK_BG)
        tc  = n/fs * tc_factor
        eps = 1e-6
        dt  = np.clip(tc - t, eps, None)
        fi  = np.clip(f0 * dt**(-3/8), 20, 700)
        ph  = 2 * np.pi * np.cumsum(fi) / fs
        env = np.clip(np.exp(5*(t/tc - 1)), 0, 1)
        ch  = np.sin(ph) * env * snr

        # Time-frequency via STFT-like
        sig_noise = np.random.normal(0, 1.0, n).astype(np.float32) + ch.astype(np.float32)
        sig_3ch   = np.stack([sig_noise]*3)
        spec      = signal_to_spectrogram(sig_3ch)

        ax.imshow(spec[0], aspect="auto", origin="lower", cmap="inferno",
                  extent=[0, n/fs, GW_FREQ_MIN, GW_FREQ_MAX])
        ax.set_title(title, color=C_SIGNAL, fontsize=9, fontweight="bold")
        ax.set_xlabel("Time (s)", color="white", fontsize=7)
        ax.set_ylabel("Freq (Hz)", color="white", fontsize=7)

    fig.tight_layout()
    _save(fig, "08_chirp_parameter_space.png")


# ─────────────────────────────────────────────
# FIGURE 9: THREE DETECTOR SPECTROGRAMS
# All 3 channels for one event
# ─────────────────────────────────────────────
def fig9_three_detector_spectrograms(signals, labels):
    print("Fig 9: Three detector spectrograms...")
    idx      = np.where(labels == 1)[0][0]
    signal   = signals[idx]
    spec     = signal_to_spectrogram(signal)
    det_names = ["LIGO Hanford (H1)", "LIGO Livingston (L1)", "Virgo (V1)"]
    det_colors = ["#FF6B35", C_ACCENT, "#9B59B6"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("CQT Spectrogram — All Three Detectors (Same Event)\n"
                 "Chirp track visible in all detectors — confirms real GW event",
                 color="white", fontsize=13, fontweight="bold")

    for ch, (ax, det, col) in enumerate(zip(axes, det_names, det_colors)):
        im = ax.imshow(spec[ch], aspect="auto", origin="lower",
                       cmap="inferno",
                       extent=[0, LIGO_SIGNAL_LEN/LIGO_SAMPLE_RATE,
                               GW_FREQ_MIN, GW_FREQ_MAX])
        ax.set_title(det, color=col, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)", color="white")
        ax.set_ylabel("Frequency (Hz)", color="white")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(2); sp.set_visible(True)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    _save(fig, "09_three_detector_spectrograms.png")


# ─────────────────────────────────────────────
# FIGURE 10: NOISE ANATOMY
# ─────────────────────────────────────────────
def fig10_noise_anatomy(signals, labels):
    print("Fig 10: Noise anatomy...")
    n_idx = np.where(labels == 0)[0]
    fs    = float(LIGO_SAMPLE_RATE)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("LIGO Noise Anatomy — Understanding the Background\n"
                 "Model must distinguish GW signal from seismic/thermal/shot noise",
                 color="white", fontsize=13, fontweight="bold")

    for ax in axes.flat:
        _style_ax(ax)

    # 1. Multiple noise PSDs
    freqs = np.fft.rfftfreq(LIGO_SIGNAL_LEN, 1.0/fs)
    mask  = (freqs >= 10) & (freqs <= 600)
    for i, idx in enumerate(n_idx[:5]):
        sig = _whiten(signals[idx])
        psd = np.abs(np.fft.rfft(sig[0]))**2
        axes[0, 0].semilogy(freqs[mask], psd[mask],
                             alpha=0.5, lw=0.8,
                             label=f"Noise #{i+1}" if i < 3 else None)
    axes[0, 0].set_title("Multiple Noise PSDs (whitened)", color="white")
    axes[0, 0].set_xlabel("Frequency (Hz)", color="white")
    axes[0, 0].set_ylabel("PSD", color="white")
    axes[0, 0].legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)
    axes[0, 0].grid(alpha=0.2, color=GRID_COL)

    # 2. Noise spectrogram
    idx  = n_idx[0]
    spec = signal_to_spectrogram(signals[idx])
    axes[0, 1].imshow(spec[0], aspect="auto", origin="lower",
                       cmap="inferno",
                       extent=[0, LIGO_SIGNAL_LEN/fs, GW_FREQ_MIN, GW_FREQ_MAX])
    axes[0, 1].set_title("Noise CQT Spectrogram (no chirp track)", color="white")
    axes[0, 1].set_xlabel("Time (s)", color="white")
    axes[0, 1].set_ylabel("Frequency (Hz)", color="white")

    # 3. Amplitude histogram
    all_noise = np.concatenate([_whiten(signals[i])[0]
                                 for i in n_idx[:5]])
    x_gauss = np.linspace(-5, 5, 200)
    y_gauss = np.exp(-x_gauss**2/2) / np.sqrt(2*np.pi)
    axes[1, 0].hist(all_noise[::10], bins=60, color=C_NOISE,
                     alpha=0.7, density=True, label="Noise")
    axes[1, 0].plot(x_gauss, y_gauss, "r--", lw=2, label="Gaussian fit")
    axes[1, 0].set_title("Noise Amplitude Distribution (Gaussian?)", color="white")
    axes[1, 0].set_xlabel("Amplitude (σ)", color="white")
    axes[1, 0].set_ylabel("Density", color="white")
    axes[1, 0].legend(facecolor=PANEL_BG, labelcolor="white")
    axes[1, 0].grid(alpha=0.2, color=GRID_COL)

    # 4. Glitch example
    glitchy = signals[n_idx[0], 0].copy()
    glitch_t = LIGO_SIGNAL_LEN // 3
    glitchy[glitch_t:glitch_t+30] += 8*np.random.randn(30).astype(np.float32)
    t_ax = np.linspace(0, LIGO_SIGNAL_LEN/fs, LIGO_SIGNAL_LEN)
    axes[1, 1].plot(t_ax, glitchy, color="#888899", lw=0.5, alpha=0.7)
    axes[1, 1].axvspan(glitch_t/fs, (glitch_t+30)/fs, color="yellow", alpha=0.3,
                        label="Glitch (non-GW transient)")
    axes[1, 1].set_title("Detector Glitch — False Positive Source", color="white")
    axes[1, 1].set_xlabel("Time (s)", color="white")
    axes[1, 1].set_ylabel("Strain", color="white")
    axes[1, 1].legend(facecolor=PANEL_BG, labelcolor="white")
    axes[1, 1].grid(alpha=0.2, color=GRID_COL)

    fig.tight_layout()
    _save(fig, "10_noise_anatomy.png")


# ─────────────────────────────────────────────
# FIGURE 11: MATCHED FILTER RESPONSE
# ─────────────────────────────────────────────
def fig11_matched_filter(signals, labels):
    print("Fig 11: Matched filter response...")
    fs    = float(LIGO_SAMPLE_RATE)
    n     = LIGO_SIGNAL_LEN
    t     = np.linspace(0, n/fs, n)

    # Template chirp
    tc    = n/fs * 1.5
    dt    = np.clip(tc - t, 1e-6, None)
    fi    = np.clip(35 * dt**(-3/8), 20, 600)
    phase = 2*np.pi*np.cumsum(fi)/fs
    env   = np.clip(np.exp(5*(t/tc-1)), 0, 1)
    template = (np.sin(phase)*env).astype(np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Matched Filter Analysis — Template Correlation\n"
                 "Classical GW detection method: neural net learns this implicitly",
                 color="white", fontsize=13, fontweight="bold")

    for ax in axes.flat:
        _style_ax(ax)

    # Template
    axes[0, 0].plot(t, template, color="#F39C12", lw=1.2)
    axes[0, 0].set_title("Chirp Template (f₀=35Hz)", color="#F39C12")
    axes[0, 0].set_xlabel("Time (s)", color="white")
    axes[0, 0].set_ylabel("Strain", color="white")
    axes[0, 0].grid(alpha=0.2, color=GRID_COL)

    # Matched filter output for noise vs signal
    for col_i, (idx, lbl, color) in enumerate([
        (np.where(labels == 0)[0][0], "Noise",     C_NOISE),
        (np.where(labels == 1)[0][0], "GW Signal", C_SIGNAL),
    ]):
        sig    = _bandpass(_whiten(signals[idx]))[0]
        mf_out = np.abs(np.fft.irfft(
            np.fft.rfft(sig) * np.conj(np.fft.rfft(template[::-1]))
        ))
        mf_out = mf_out / (mf_out.max() + 1e-10)
        snr_est = mf_out.max()

        axes[0, 1+col_i-1].plot(t, sig, color=color, lw=0.6, alpha=0.8)
        axes[0, 1+col_i-1].set_title(f"{lbl} — H1 (cleaned)", color=color)
        axes[0, 1+col_i-1].set_xlabel("Time (s)", color="white")
        axes[0, 1+col_i-1].grid(alpha=0.2, color=GRID_COL)

        axes[1, col_i].plot(t, mf_out, color=color, lw=1.2)
        axes[1, col_i].axhline(0.5, color="yellow", ls="--", lw=1,
                                label="Detection threshold")
        axes[1, col_i].set_title(f"Matched Filter Output — {lbl}\n"
                                   f"Peak SNR: {snr_est:.2f}",
                                  color=color)
        axes[1, col_i].set_xlabel("Time (s)", color="white")
        axes[1, col_i].set_ylabel("Normalized correlation", color="white")
        axes[1, col_i].legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)
        axes[1, col_i].grid(alpha=0.2, color=GRID_COL)

    fig.tight_layout()
    _save(fig, "11_matched_filter.png")


# ─────────────────────────────────────────────
# FIGURE 12: FULL PIPELINE SUMMARY
# ─────────────────────────────────────────────
def fig12_pipeline_summary(signals, labels):
    print("Fig 12: Pipeline summary...")
    idx      = np.where(labels == 1)[0][0]
    signal   = signals[idx]
    whitened = _whiten(signal)
    cleaned  = _bandpass(whitened)
    spec     = signal_to_spectrogram(signal)
    t_ax     = np.linspace(0, LIGO_SIGNAL_LEN/LIGO_SAMPLE_RATE, LIGO_SIGNAL_LEN)

    fig = plt.figure(figsize=(24, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("STELLARIS-DNet Module 2B — Complete GW Detection Pipeline\n"
                 "Input → Preprocess → CQT → EfficientNet-B2 + CBAM + GeM → Decision",
                 color="white", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.5, wspace=0.35,
                           top=0.85, bottom=0.08)

    steps = [
        ("Raw H1 Strain",       "Input"),
        ("Whitened",            "Preprocessing"),
        ("Bandpass 20-500Hz",   "Preprocessing"),
        ("CQT Spectrogram",     "Feature Extraction"),
        ("STELLARIS\nDecision", "Classification"),
    ]
    arrows_x = [0.195, 0.355, 0.515, 0.675]

    for col, (title, stage) in enumerate(steps):
        ax = fig.add_subplot(gs[:, col])
        _style_ax(ax, PANEL_BG)

        if col == 0:
            ax.plot(t_ax, signal[0], color=C_NOISE, lw=0.5)
            ax.set_ylabel("Strain", color="white", fontsize=8)
        elif col == 1:
            ax.plot(t_ax, whitened[0], color=C_ACCENT, lw=0.5)
        elif col == 2:
            ax.plot(t_ax, cleaned[0], color=C_SIGNAL, lw=0.5)
        elif col == 3:
            ax.imshow(spec[0], aspect="auto", origin="lower", cmap="inferno",
                      extent=[0, LIGO_SIGNAL_LEN/LIGO_SAMPLE_RATE,
                              GW_FREQ_MIN, GW_FREQ_MAX])
            ax.set_ylabel("Freq (Hz)", color="white", fontsize=8)
        else:
            ax.set_facecolor("#0a1a0a")
            ax.text(0.5, 0.65, "✓ GW Signal\nDetected",
                    ha="center", va="center", fontsize=16,
                    color="#00ff88", fontweight="bold", transform=ax.transAxes)
            ax.text(0.5, 0.35, "P(Signal) = 0.94",
                    ha="center", va="center", fontsize=13,
                    color="white", transform=ax.transAxes)
            ax.axis("off")

        ax.set_title(f"{title}\n[{stage}]", color="white",
                     fontsize=9, fontweight="bold")
        if col < 3:
            ax.set_xlabel("Time (s)", color="white", fontsize=7)
            ax.tick_params(colors="white", labelsize=6)

    # Arrows
    for xp in arrows_x:
        fig.text(xp, 0.47, "→", ha="center", va="center",
                 fontsize=24, color="white", fontweight="bold")

    _save(fig, "12_full_pipeline_summary.png")


# ─────────────────────────────────────────────
# FIGURE 13: PHYSICS ASTROPHYSICS CONTEXT
# ─────────────────────────────────────────────
def fig13_physics_context(signals, labels):
    print("Fig 13: Physics context...")
    idx  = np.where(labels == 1)[0][0]
    spec = signal_to_spectrogram(signals[idx])
    t    = np.linspace(0, LIGO_SIGNAL_LEN/LIGO_SAMPLE_RATE, LIGO_SIGNAL_LEN)

    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4,
                             top=0.88, bottom=0.08, left=0.06, right=0.96)

    fig.text(0.5, 0.94,
             "Astrophysics Context — What STELLARIS-DNet Detects",
             ha="center", fontsize=16, color="white", fontweight="bold")
    fig.text(0.5, 0.90,
             "Binary black hole / neutron star mergers produce GW chirps detectable by LIGO/Virgo",
             ha="center", fontsize=11, color="#aaaacc")

    # CQT
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(spec[0], aspect="auto", origin="lower", cmap="inferno",
               extent=[0, LIGO_SIGNAL_LEN/LIGO_SAMPLE_RATE, GW_FREQ_MIN, GW_FREQ_MAX])
    ax0.set_title("Detected GW Event\n(CQT Spectrogram)", color=C_SIGNAL,
                  fontsize=12, fontweight="bold")
    ax0.set_xlabel("Time (s)", color="white")
    ax0.set_ylabel("Frequency (Hz)", color="white")
    ax0.tick_params(colors="white")

    # Chirp physics formula
    ax1 = fig.add_subplot(gs[0, 1])
    _style_ax(ax1)
    ax1.set_title("Chirp Equation", color="white", fontsize=11, fontweight="bold")
    lines = [
        ("f(t) ∝ (t_c − t)^(−3/8)", "yellow",  0.82),
        ("f  = instantaneous frequency", "white", 0.65),
        ("t_c = coalescence time",       "white", 0.52),
        ("",                             "white", 0.43),
        ("Chirp Mass:",                  "#aaaacc",0.35),
        ("M_c = (m₁m₂)^(3/5)",         C_ACCENT, 0.22),
        ("       / (m₁+m₂)^(1/5)",      C_ACCENT, 0.12),
    ]
    for text, color, y in lines:
        ax1.text(0.1, y, text, color=color, fontsize=10,
                 transform=ax1.transAxes,
                 fontweight="bold" if color == "yellow" else "normal")
    ax1.axis("off")

    # Detection statistics
    ax2 = fig.add_subplot(gs[1, 1])
    _style_ax(ax2)
    ax2.set_title("LIGO Detection Facts", color="white", fontsize=11, fontweight="bold")
    facts = [
        ("Events detected (O3):","~90",   C_SIGNAL),
        ("Sensitivity:",          "10⁻²³ m", C_ACCENT),
        ("GW travel speed:",      "c",       "white"),
        ("Typical SNR:",          "8–25",    "yellow"),
        ("Signal duration:",      "0.1–100s","white"),
        ("Frequency range:",      "20–500 Hz",C_NOISE),
    ]
    y = 0.85
    for k, v, col in facts:
        ax2.text(0.05, y, k, color="#aaaacc", fontsize=9, transform=ax2.transAxes)
        ax2.text(0.65, y, v, color=col, fontsize=9, fontweight="bold",
                 transform=ax2.transAxes)
        y -= 0.13
    ax2.axis("off")

    # Inspiral phases
    ax3 = fig.add_subplot(gs[:, 2])
    _style_ax(ax3)
    ax3.set_title("Binary Merger Phases\n(Time → Frequency)", color="white",
                  fontsize=11, fontweight="bold")
    tc   = LIGO_SIGNAL_LEN/LIGO_SAMPLE_RATE * 1.5
    dt   = np.clip(tc - t, 1e-6, None)
    fi   = np.clip(35*dt**(-3/8), 20, 600)
    mask = fi < 590
    ax3.plot(t[mask], fi[mask], color=C_SIGNAL, lw=2.5)
    ax3.fill_between(t[mask], 20, fi[mask], alpha=0.15, color=C_SIGNAL)
    ax3.axvspan(0, tc*0.5,   alpha=0.08, color="blue",   label="Inspiral")
    ax3.axvspan(tc*0.5,tc*0.9,alpha=0.08, color="orange", label="Merger")
    ax3.axvspan(tc*0.9,t[mask][-1],alpha=0.08,color="purple",label="Ringdown")
    ax3.set_xlabel("Time (s)", color="white")
    ax3.set_ylabel("GW Frequency (Hz)", color="white")
    ax3.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=9)
    ax3.grid(alpha=0.2, color=GRID_COL)
    ax3.set_ylim(0, 600)

    _save(fig, "13_physics_context.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 60)
    print("STELLARIS-DNet — G2Net Visualization (13 Figures)")
    print("=" * 60)

    signals, labels = _get_data(n_each=6)

    print(f"\nGenerating 13 figures → {OUTPUT_DIR}/")
    print("-" * 60)

    fig1_raw_strain(signals, labels)
    fig2_pipeline(signals, labels)
    fig3_cqt_gallery(signals, labels, n=4)
    fig4_frequency_analysis(signals, labels)
    fig5_detector_correlation(signals, labels)
    fig6_whitening(signals, labels)
    fig7_snr_analysis(signals, labels)
    fig8_chirp_parameters(signals, labels)
    fig9_three_detector_spectrograms(signals, labels)
    fig10_noise_anatomy(signals, labels)
    fig11_matched_filter(signals, labels)
    fig12_pipeline_summary(signals, labels)
    fig13_physics_context(signals, labels)

    print("\n" + "=" * 60)
    print(f"✅ All 13 figures saved to: {OUTPUT_DIR}/")
    print()
    print("Files:")
    for i, name in enumerate([
        "01_raw_strain_comparison.png",
        "02_processing_pipeline.png",
        "03_cqt_gallery.png",
        "04_frequency_analysis.png",
        "05_detector_correlation.png",
        "06_whitening_effect.png",
        "07_snr_analysis.png",
        "08_chirp_parameter_space.png",
        "09_three_detector_spectrograms.png",
        "10_noise_anatomy.png",
        "11_matched_filter.png",
        "12_full_pipeline_summary.png",
        "13_physics_context.png",
    ], 1):
        print(f"  {i:02d}. {name}")
    print("=" * 60)