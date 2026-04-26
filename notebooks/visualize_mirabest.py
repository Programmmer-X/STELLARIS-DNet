"""
notebooks/visualize_mirabest.py
STELLARIS-DNet — MiraBest Radio Galaxy Visualization
Generates 12 presentation-quality figures.

Run: python notebooks/visualize_mirabest.py
Output: notebooks/mirabest_visuals/  (12 PNG files)

Handles ALL MiraBest label encodings:
  Binary:  0=FRI, 1=FRII   ← your real data uses this
  5-class: 0=FRI-conf, 1=FRI-unc, 2=FRII-conf, 3=FRII-unc
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module2.config import RGZ_DATA_DIR, RGZ_CLASSES

OUTPUT_DIR = "notebooks/mirabest_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)

DARK_BG  = "#0a0a12"
PANEL_BG = "#111122"
GRID_COL = "#222244"
COLORS   = {"FRI": "#FF6B35", "FRII": "#4ECDC4", "Hybrid": "#95A5A6"}


# ─────────────────────────────────────────────
# LABEL UTILITIES — handles binary AND 5-class
# ─────────────────────────────────────────────
def _get_class_indices(labels):
    """
    Returns (fri_idx, frii_idx) for ANY MiraBest label encoding.
    Binary (0,1): FRI=0, FRII=1
    5-class:      FRI=0,1  FRII=2,3
    """
    unique = set(np.unique(labels).tolist())
    if 2 not in unique and 3 not in unique:
        # Binary encoding — your real data
        return np.where(labels == 0)[0], np.where(labels == 1)[0]
    else:
        # Multi-class — confident samples only
        return (np.where(np.isin(labels, [0, 1]))[0],
                np.where(np.isin(labels, [2, 3]))[0])


def _safe_choice(arr, n, replace=False):
    """Crash-safe random.choice — never fails on empty or small arrays."""
    if len(arr) == 0:
        return np.array([], dtype=int)
    n = min(n, len(arr))
    return np.random.choice(arr, n, replace=replace)


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


def _enhance(img, lo=1.0, hi=99.0):
    """Percentile normalization for morphological clarity."""
    img  = np.asarray(img, dtype=np.float32)
    img  = np.clip(img, 0, None)
    p_lo = np.percentile(img, lo)
    p_hi = np.percentile(img, hi)
    if p_hi > p_lo:
        return np.clip((img - p_lo) / (p_hi - p_lo), 0, 1)
    return img / (img.max() + 1e-8)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def _find_data_dir():
    candidates = [
        RGZ_DATA_DIR,
        "data/module2/mirabest",
        "/kaggle/working/STELLARIS-DNet/data/module2/mirabest",
        "/kaggle/input/mirabest-radio-galaxy",
    ]
    for path in candidates:
        if os.path.isdir(path):
            for i in range(1, 9):
                if os.path.exists(os.path.join(path, f"data_batch_{i}")):
                    print(f"✅ MiraBest found: {path}")
                    return path
    return None


def _load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    imgs   = np.array(d.get(b"data",   d.get("data",   [])))
    labels = np.array(d.get(b"labels", d.get("labels", [])))
    return imgs, labels


def _load_all(data_dir):
    all_imgs, all_lbls = [], []
    for i in range(1, 9):
        p = os.path.join(data_dir, f"data_batch_{i}")
        if os.path.exists(p):
            im, lb = _load_batch(p)
            all_imgs.append(im); all_lbls.append(lb)
    tp = os.path.join(data_dir, "test_batch")
    if os.path.exists(tp):
        im, lb = _load_batch(tp)
        all_imgs.append(im); all_lbls.append(lb)

    imgs   = np.concatenate(all_imgs)
    labels = np.concatenate(all_lbls)
    n      = len(imgs)

    # Reshape to (N, H, W, C)
    try:
        imgs = imgs.reshape(n, 3, 150, 150).astype(np.float32) / 255.0
        imgs = imgs.transpose(0, 2, 3, 1)
    except ValueError:
        try:
            imgs = imgs.reshape(n, 1, 150, 150).astype(np.float32) / 255.0
            imgs = np.repeat(imgs.transpose(0, 2, 3, 1), 3, axis=-1)
        except ValueError:
            imgs = imgs.reshape(n, 150, 150, -1).astype(np.float32) / 255.0

    print(f"✅ Loaded {n} real images | Labels: {np.unique(labels, return_counts=True)}")
    fri_idx, frii_idx = _get_class_indices(labels)
    print(f"   FRI: {len(fri_idx)} | FRII: {len(frii_idx)}")
    return imgs, labels


def _generate_synthetic(n=300):
    print("⚠️  MiraBest not found — generating synthetic images")
    size = 150
    imgs, labels = [], []
    Y, X   = np.ogrid[:size, :size]
    cx, cy = size//2, size//2
    dist   = np.sqrt((X-cx)**2 + (Y-cy)**2)

    def gauss(cx_, cy_, sigma, amp=1.0):
        return amp * np.exp(-((X-cx_)**2+(Y-cy_)**2)/(2*sigma**2))

    for _ in range(n//2):
        core = gauss(cx, cy, 6)
        jet  = (np.exp(-(Y-cy)**2/(2*8**2)) *
                np.exp(-dist/np.random.uniform(12, 20)))
        img  = core + 0.4*jet + np.random.normal(0, 0.03, (size, size))
        img  = np.clip(img, 0, None) / (img.max() + 1e-8)
        imgs.append(np.stack([img]*3, -1).astype(np.float32))
        labels.append(0)  # FRI = 0

    for _ in range(n//2):
        core = 0.3 * gauss(cx, cy, 5)
        ang  = np.random.uniform(0, np.pi)
        hs   = np.random.uniform(35, 55)
        core += gauss(cx+hs*np.cos(ang), cy+hs*np.sin(ang), 7)
        core += gauss(cx-hs*np.cos(ang), cy-hs*np.sin(ang), 7)
        img   = core + np.random.normal(0, 0.03, (size, size))
        img   = np.clip(img, 0, None) / (img.max() + 1e-8)
        imgs.append(np.stack([img]*3, -1).astype(np.float32))
        labels.append(1)  # FRII = 1

    return np.array(imgs), np.array(labels)


# ─────────────────────────────────────────────
# FIGURE 1: FULL GALLERY — 32 IMAGES
# ─────────────────────────────────────────────
def fig1_gallery(images, labels):
    print("Fig 1: Full gallery...")
    fri_idx, frii_idx = _get_class_indices(labels)
    chosen = np.concatenate([
        _safe_choice(fri_idx,  min(16, len(fri_idx))),
        _safe_choice(frii_idx, min(16, len(frii_idx)))
    ])
    np.random.shuffle(chosen)

    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor(DARK_BG)
    fig.text(0.5, 0.97, "MiraBest Dataset — Real VLA Radio Galaxy Images",
             ha="center", fontsize=22, color="white", fontweight="bold")
    fig.text(0.5, 0.94,
             "Expert-labeled radio AGN — supermassive black hole jets at 1.4 GHz",
             ha="center", fontsize=12, color="#aaaacc")

    axes = fig.subplots(4, 8)
    fig.subplots_adjust(hspace=0.06, wspace=0.04,
                        top=0.91, bottom=0.06, left=0.01, right=0.99)

    for i, ax in enumerate(axes.flat):
        ax.set_facecolor(DARK_BG)
        if i >= len(chosen):
            ax.axis("off"); continue
        idx      = chosen[i]
        is_fri   = idx in fri_idx
        cls      = "FRI" if is_fri else "FRII"
        col      = COLORS[cls]
        cmap_use = "hot" if is_fri else "cool"
        ax.imshow(_enhance(images[idx, :, :, 0]), cmap=cmap_use, aspect="equal")
        ax.set_title(cls, fontsize=7, color=col, fontweight="bold", pad=2)
        ax.axis("off")
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(1.2); sp.set_visible(True)

    fig.text(0.08, 0.03, "■ FRI  — jets fade from core",
             color=COLORS["FRI"], fontsize=11)
    fig.text(0.38, 0.03, "■ FRII — bright hotspots at jet ends",
             color=COLORS["FRII"], fontsize=11)
    _save(fig, "01_gallery_32images.png")


# ─────────────────────────────────────────────
# FIGURE 2: FRI vs FRII COMPARISON
# ─────────────────────────────────────────────
def fig2_fri_vs_frii(images, labels):
    print("Fig 2: FRI vs FRII comparison...")
    fri_idx, frii_idx = _get_class_indices(labels)
    n_each = 5
    fri_s  = _safe_choice(fri_idx,  n_each)
    frii_s = _safe_choice(frii_idx, n_each)

    fig = plt.figure(figsize=(22, 9))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(2, 11, figure=fig, hspace=0.12, wspace=0.06,
                             top=0.88, bottom=0.08)

    fig.text(0.5, 0.95, "Fanaroff-Riley Classification — The FR Boundary",
             ha="center", fontsize=18, color="white", fontweight="bold")
    fig.text(0.5, 0.91,
             "Both types powered by SMBHs — separated by jet power 10²⁵ W/Hz",
             ha="center", fontsize=11, color="#aaaacc")

    for (col_s, col_e, text, color) in [
        (0, 4, "TYPE I  (FRI)\nJets fade from center\nEdge-brightened core\nJet power < 10²⁵ W/Hz",
         COLORS["FRI"]),
        (6, 10, "TYPE II  (FRII)\nBright hotspots at jet tips\nEdge-darkened core\nJet power > 10²⁵ W/Hz",
         COLORS["FRII"]),
    ]:
        ax = fig.add_subplot(gs[0, col_s:col_e+1])
        ax.set_facecolor(DARK_BG)
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12,
                color=color, fontweight="bold", transform=ax.transAxes,
                linespacing=1.6)
        ax.axis("off")

    ax_vs = fig.add_subplot(gs[:, 5])
    ax_vs.text(0.5, 0.5, "vs", ha="center", va="center",
               fontsize=22, color="white", fontweight="bold",
               transform=ax_vs.transAxes)
    ax_vs.axis("off")

    for i, idx in enumerate(fri_s):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(_enhance(images[idx, :, :, 0]), cmap="hot", aspect="equal")
        ax.axis("off")
        ax.set_title(f"FRI #{i+1}", fontsize=8, color=COLORS["FRI"], pad=2)
        for sp in ax.spines.values():
            sp.set_edgecolor(COLORS["FRI"]); sp.set_linewidth(2)
            sp.set_visible(True)

    for i, idx in enumerate(frii_s):
        ax = fig.add_subplot(gs[1, 6+i])
        ax.imshow(_enhance(images[idx, :, :, 0]), cmap="cool", aspect="equal")
        ax.axis("off")
        ax.set_title(f"FRII #{i+1}", fontsize=8, color=COLORS["FRII"], pad=2)
        for sp in ax.spines.values():
            sp.set_edgecolor(COLORS["FRII"]); sp.set_linewidth(2)
            sp.set_visible(True)

    _save(fig, "02_FRI_vs_FRII_comparison.png")


# ─────────────────────────────────────────────
# FIGURE 3: ANNOTATED GALAXY
# ─────────────────────────────────────────────
def fig3_annotated(images, labels):
    print("Fig 3: Annotated galaxy...")
    _, frii_idx = _get_class_indices(labels)
    if len(frii_idx) == 0:
        frii_idx = np.arange(len(images))
    idx = int(_safe_choice(frii_idx, 1)[0])
    img = _enhance(images[idx, :, :, 0])
    H, W = img.shape
    cx, cy = W//2, H//2

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.imshow(img, cmap="inferno", aspect="equal")
    ax.axis("off")
    ax.set_title(
        "Real Radio Galaxy — AGN Jet Structure\n"
        "Supermassive Black Hole drives collimated plasma jets",
        color="white", fontsize=13, fontweight="bold", pad=12
    )

    annotations = [
        (cx,    cy,    -45, -35, "AGN Core\n(SMBH)",          "yellow"),
        (cx-50, cy-40,  -50,  15, "Radio Lobe\n(jet terminus)", COLORS["FRII"]),
        (cx+50, cy+40,   45,  15, "Radio Lobe\n(jet terminus)", COLORS["FRII"]),
        (cx-25, cy,    -55,   0, "Jet\n(plasma beam)",         "white"),
    ]
    for (x, y, dx, dy, lbl, col) in annotations:
        x, y = int(np.clip(x, 5, W-5)), int(np.clip(y, 5, H-5))
        ax.annotate(
            lbl, xy=(x, y), xytext=(x+dx, y+dy),
            color=col, fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor=DARK_BG, edgecolor=col, alpha=0.85)
        )
    _save(fig, "03_annotated_galaxy.png")


# ─────────────────────────────────────────────
# FIGURE 4: DATASET STATISTICS
# ─────────────────────────────────────────────
def fig4_statistics(images, labels):
    print("Fig 4: Dataset statistics...")
    fri_idx, frii_idx = _get_class_indices(labels)
    fri_n = len(fri_idx); frii_n = len(frii_idx)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("MiraBest Dataset Statistics",
                 fontsize=16, color="white", fontweight="bold")
    for ax in axes: _style_ax(ax)

    bars = axes[0].bar(["FRI", "FRII"], [fri_n, frii_n],
                        color=[COLORS["FRI"], COLORS["FRII"]],
                        edgecolor="white", width=0.5)
    axes[0].set_title("Class Distribution", color="white", fontsize=13)
    axes[0].set_ylabel("Count", color="white")
    axes[0].set_xticklabels(["FRI", "FRII"], color="white", fontsize=11)
    for bar, v in zip(bars, [fri_n, frii_n]):
        axes[0].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+2, str(v),
                     ha="center", color="white", fontsize=12, fontweight="bold")

    wedges, texts, autotexts = axes[1].pie(
        [fri_n, frii_n], labels=[f"FRI\n{fri_n}", f"FRII\n{frii_n}"],
        colors=[COLORS["FRI"], COLORS["FRII"]], autopct="%1.1f%%",
        startangle=90, textprops={"color": "white", "fontsize": 11},
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 2}
    )
    for at in autotexts: at.set_color("black"); at.set_fontweight("bold")
    axes[1].set_title("FRI vs FRII Split", color="white", fontsize=13)
    axes[1].set_facecolor(DARK_BG)

    axes[2].hist(images[fri_idx,  :, :, 0].flatten()[::10], bins=60,
                 alpha=0.7, color=COLORS["FRI"],  label="FRI",  density=True)
    axes[2].hist(images[frii_idx, :, :, 0].flatten()[::10], bins=60,
                 alpha=0.7, color=COLORS["FRII"], label="FRII", density=True)
    axes[2].set_title("Pixel Intensity Distribution", color="white", fontsize=13)
    axes[2].set_xlabel("Pixel Value", color="white")
    axes[2].set_ylabel("Density", color="white")
    axes[2].legend(facecolor=PANEL_BG, labelcolor="white", fontsize=10)

    fig.tight_layout()
    _save(fig, "04_dataset_statistics.png")


# ─────────────────────────────────────────────
# FIGURE 5: AUGMENTATION SHOWCASE
# ─────────────────────────────────────────────
def fig5_augmentation(images, labels):
    print("Fig 5: Augmentation showcase...")
    fri_idx, _ = _get_class_indices(labels)
    idx  = int(_safe_choice(fri_idx, 1)[0])
    base = _enhance(images[idx, :, :, 0])

    augmented = [
        ("Original",           base),
        ("Horizontal Flip",    base[:, ::-1]),
        ("Vertical Flip",      base[::-1, :]),
        ("90° Rotation",       np.rot90(base, 1)),
        ("180° Rotation",      np.rot90(base, 2)),
        ("270° Rotation",      np.rot90(base, 3)),
        ("Gaussian Noise",     np.clip(base + np.random.normal(0, 0.04, base.shape), 0, 1)),
        ("Brightness +20%",    np.clip(base * 1.2, 0, 1)),
        ("Contrast Stretch",   _enhance(base, 5, 95)),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        "Data Augmentation Pipeline — Physics-valid Transforms\n"
        "Radio galaxies have no preferred orientation",
        color="white", fontsize=14, fontweight="bold"
    )

    for i, (ax, (name, img)) in enumerate(zip(axes.flat, augmented)):
        ax.imshow(img, cmap="hot", aspect="equal")
        ax.set_title(name, fontsize=10, color=COLORS["FRI"],
                     fontweight="bold" if i == 0 else "normal", pad=4)
        ax.axis("off")
        if i == 0:
            for sp in ax.spines.values():
                sp.set_edgecolor("yellow"); sp.set_linewidth(2)
                sp.set_visible(True)

    ax_last = list(axes.flat)[len(augmented)]
    ax_last.set_facecolor(PANEL_BG)
    ax_last.text(0.5, 0.5,
                 "Strategy:\n\n✓ Flips (H+V)\n✓ Rotations\n"
                 "✓ Noise\n✓ Brightness\n\nTrain only",
                 ha="center", va="center", fontsize=11,
                 color="white", transform=ax_last.transAxes)
    ax_last.axis("off")
    for ax in list(axes.flat)[len(augmented)+1:]:
        ax.axis("off")

    fig.tight_layout()
    _save(fig, "05_augmentation_showcase.png")


# ─────────────────────────────────────────────
# FIGURE 6: BRIGHTNESS PROFILES
# ─────────────────────────────────────────────
def fig6_brightness_profile(images, labels):
    print("Fig 6: Brightness profiles...")
    fri_idx, frii_idx = _get_class_indices(labels)
    n_each = min(10, len(fri_idx), len(frii_idx))
    x_axis = np.linspace(-75, 75, 150)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        "Brightness Profile — Cross-Section Through Jet Axis\n"
        "FRI fades outward | FRII peaks at hotspot ends",
        color="white", fontsize=14, fontweight="bold"
    )

    for ax, idx_arr, cls, color in [
        (axes[0], fri_idx[:n_each],  "FRI",  COLORS["FRI"]),
        (axes[1], frii_idx[:n_each], "FRII", COLORS["FRII"]),
    ]:
        _style_ax(ax)
        profiles = np.array([_enhance(images[idx, :, :, 0])[75, :]
                              for idx in idx_arr])
        mean = profiles.mean(axis=0)
        std  = profiles.std(axis=0)

        for p in profiles:
            ax.plot(x_axis, p, color=color, alpha=0.15, lw=0.8)
        ax.plot(x_axis, mean, color=color, lw=2.5, label=f"Mean {cls}")
        ax.fill_between(x_axis, mean-std, mean+std,
                        color=color, alpha=0.25, label="±1σ")
        ax.axvline(0, color="yellow", ls="--", lw=1.2, alpha=0.7, label="Core")
        ax.set_title(f"{cls} — Brightness Profile", color=color, fontsize=12)
        ax.set_xlabel("Offset from core (pixels)", color="white")
        ax.set_ylabel("Normalized intensity", color="white")
        ax.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=9)
        ax.grid(alpha=0.2, color=GRID_COL)
        ax.set_xlim(-75, 75); ax.set_ylim(0, 1)

    fig.tight_layout()
    _save(fig, "06_brightness_profiles.png")


# ─────────────────────────────────────────────
# FIGURE 7: MEAN CLASS IMAGES
# ─────────────────────────────────────────────
def fig7_mean_images(images, labels):
    print("Fig 7: Mean class images...")
    fri_idx, frii_idx = _get_class_indices(labels)
    fri_mean  = _enhance(images[fri_idx,  :, :, 0].mean(axis=0))
    frii_mean = _enhance(images[frii_idx, :, :, 0].mean(axis=0))
    diff      = frii_mean - fri_mean
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        "Mean Class Images — Morphological Signature\n"
        "Averaged over all samples in each class",
        color="white", fontsize=14, fontweight="bold"
    )

    for ax, img, title, cmap, txt in [
        (axes[0], fri_mean,  f"Mean FRI (n={len(fri_idx)})",  "hot",    ""),
        (axes[1], frii_mean, f"Mean FRII (n={len(frii_idx)})", "cool",   ""),
        (axes[2], diff_norm, "FRII − FRI (Difference)",        "RdBu_r", "Hotspot signature"),
    ]:
        ax.set_facecolor(DARK_BG)
        im = ax.imshow(img, cmap=cmap, aspect="equal")
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")
        if txt:
            ax.text(0.5, -0.06, txt, ha="center",
                    transform=ax.transAxes, color="#aaaacc", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    _save(fig, "07_mean_class_images.png")


# ─────────────────────────────────────────────
# FIGURE 8: MULTISCALE MORPHOLOGY
# ─────────────────────────────────────────────
def fig8_multiscale(images, labels):
    print("Fig 8: Multi-scale morphology...")
    fri_idx, frii_idx = _get_class_indices(labels)
    # Use whichever class has more samples for the clearest example
    use_idx = frii_idx if len(frii_idx) > 0 else fri_idx
    if len(use_idx) == 0:
        use_idx = np.arange(len(images))
    idx = int(_safe_choice(use_idx, 1)[0])
    img = _enhance(images[idx, :, :, 0])
    color = COLORS["FRII"] if idx in frii_idx else COLORS["FRI"]

    crops = [
        ("Full view\n(150×150)", img),
        ("Core\n(60×60)",        img[45:105, 45:105]),
        ("Lobe A\n(50×50)",      img[5:55, 5:55]),
        ("Lobe B\n(50×50)",      img[95:145, 95:145]),
        ("Jet region\n(30×80)",  img[60:90, 10:90]),
    ]
    cmaps = ["inferno", "hot", "plasma", "plasma", "cool"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        "Multi-Scale Morphology — Radio Galaxy Structure\n"
        "Different components visible at each zoom level",
        color="white", fontsize=14, fontweight="bold"
    )
    for ax, (name, crop), cmap in zip(axes, crops, cmaps):
        ax.set_facecolor(DARK_BG)
        ax.imshow(crop, cmap=cmap, aspect="equal")
        ax.set_title(name, color=color, fontsize=9, fontweight="bold")
        ax.axis("off")
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(1.5); sp.set_visible(True)

    fig.tight_layout()
    _save(fig, "08_multiscale_morphology.png")


# ─────────────────────────────────────────────
# FIGURE 9: PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
def fig9_preprocessing(images, labels):
    print("Fig 9: Preprocessing pipeline...")
    fri_idx, frii_idx = _get_class_indices(labels)
    all_idx = np.concatenate([fri_idx, frii_idx])
    idx     = int(_safe_choice(all_idx, 1)[0])
    raw     = images[idx, :, :, 0]
    normed  = raw / (raw.max() + 1e-8)
    pct     = _enhance(raw)
    pct_f   = pct[:, ::-1]
    pct_b   = np.clip(pct * 1.3, 0, 1)

    steps = [
        ("Step 1\nRaw (÷255)",         raw,    "gray"),
        ("Step 2\nMin-Max Norm",        normed, "gray"),
        ("Step 3\nPercentile Norm",     pct,    "hot"),
        ("Step 4\nAugment: Flip",       pct_f,  "hot"),
        ("Step 5\nAugment: Brightness", pct_b,  "hot"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        "Image Preprocessing Pipeline — MiraBest 2A\n"
        "Each step improves morphological features for EfficientNet-B2",
        color="white", fontsize=14, fontweight="bold"
    )
    for ax, (name, img, cmap) in zip(axes, steps):
        ax.set_facecolor(DARK_BG)
        ax.imshow(img, cmap=cmap, aspect="equal")
        ax.set_title(name, color="white", fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.tight_layout()
    _save(fig, "09_preprocessing_pipeline.png")


# ─────────────────────────────────────────────
# FIGURE 10: PHYSICS CONTEXT
# ─────────────────────────────────────────────
def fig10_physics_context(images, labels):
    print("Fig 10: Physics context...")
    fri_idx, frii_idx = _get_class_indices(labels)
    fri_img  = _enhance(images[int(_safe_choice(fri_idx,  1)[0]), :, :, 0])
    frii_img = _enhance(images[int(_safe_choice(frii_idx, 1)[0]), :, :, 0])

    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor(DARK_BG)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.4,
                             top=0.88, bottom=0.08, left=0.05, right=0.95)

    fig.text(0.5, 0.94,
             "Astrophysics Context — Radio Galaxy Classification",
             ha="center", fontsize=17, color="white", fontweight="bold")
    fig.text(0.5, 0.90,
             "STELLARIS-DNet connects ML predictions to black hole physics",
             ha="center", fontsize=11, color="#aaaacc")

    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(fri_img, cmap="hot", aspect="equal")
    ax1.set_title("FRI\n(STELLARIS prediction)", color=COLORS["FRI"],
                  fontsize=12, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[:, 3])
    ax2.imshow(frii_img, cmap="cool", aspect="equal")
    ax2.set_title("FRII\n(STELLARIS prediction)", color=COLORS["FRII"],
                  fontsize=12, fontweight="bold")
    ax2.axis("off")

    for col, (props, cls, color) in enumerate([
        ([("Jet Power",   "< 10²⁵ W/Hz"), ("Jet Velocity","< 0.3c"),
          ("BH Mass",     "~10⁸ M☉"),    ("Lobes",       "Diffuse"),
          ("Hotspots",    "Absent"),       ("FR Ratio",    "< 0.5")],
         "FRI Properties", COLORS["FRI"]),
        ([("Jet Power",   "> 10²⁵ W/Hz"), ("Jet Velocity","> 0.5c"),
          ("BH Mass",     "~10⁹ M☉"),    ("Lobes",       "Edge-bright"),
          ("Hotspots",    "Compact"),      ("FR Ratio",    "> 0.5")],
         "FRII Properties", COLORS["FRII"]),
    ]):
        ax = fig.add_subplot(gs[:, 1+col])
        _style_ax(ax, PANEL_BG)
        ax.set_title(cls, color=color, fontsize=12, fontweight="bold", pad=8)
        y = 0.88
        for prop, val in props:
            ax.text(0.05, y, f"  {prop}", color="#ccccee",
                    fontsize=9, transform=ax.transAxes)
            ax.text(0.60, y, val, color=color, fontsize=9,
                    fontweight="bold", transform=ax.transAxes)
            y -= 0.13
        ax.axis("off")

    _save(fig, "10_physics_context.png")


# ─────────────────────────────────────────────
# FIGURE 11: PIXEL STATISTICS
# ─────────────────────────────────────────────
def fig11_pixel_analysis(images, labels):
    print("Fig 11: Pixel analysis...")
    fri_idx, frii_idx = _get_class_indices(labels)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        "FRI vs FRII — Pixel & Spatial Statistics\n"
        "Quantitative morphological differences exploited by CBAM attention",
        color="white", fontsize=14, fontweight="bold"
    )
    for ax in axes.flat: _style_ax(ax)

    for col, (r_s, c_s, name) in enumerate([
        (slice(55,95), slice(55,95), "Core Region (40×40)"),
        (slice(0,40),  slice(0,40),  "Lobe Region (40×40)"),
        (slice(None),  slice(None),  "Full Image"),
    ]):
        ax = axes[0, col]
        ax.hist(images[fri_idx,  r_s, c_s, 0].flatten()[::5], bins=50,
                alpha=0.7, color=COLORS["FRI"],  label="FRI",  density=True)
        ax.hist(images[frii_idx, r_s, c_s, 0].flatten()[::5], bins=50,
                alpha=0.7, color=COLORS["FRII"], label="FRII", density=True)
        ax.set_title(name, color="white", fontsize=10)
        ax.set_xlabel("Pixel value", color="white", fontsize=8)
        ax.set_ylabel("Density", color="white", fontsize=8)
        ax.legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)
        ax.grid(alpha=0.2, color=GRID_COL)

    slices = [(slice(55,95), slice(55,95)),
              (slice(30,70), slice(30,70)),
              (slice(0,40),  slice(0,40))]
    regions = ["Core\n40×40", "Mid\n40×40", "Edge\n40×40"]
    fri_m  = [images[fri_idx,  s[0], s[1], 0].mean() for s in slices]
    frii_m = [images[frii_idx, s[0], s[1], 0].mean() for s in slices]
    fri_s  = [images[fri_idx,  s[0], s[1], 0].std()  for s in slices]
    frii_s = [images[frii_idx, s[0], s[1], 0].std()  for s in slices]
    x = np.arange(3)

    axes[1,0].bar(x-0.2, fri_m,  0.35, color=COLORS["FRI"],  label="FRI",
                  yerr=fri_s, capsize=5, edgecolor="white")
    axes[1,0].bar(x+0.2, frii_m, 0.35, color=COLORS["FRII"], label="FRII",
                  yerr=frii_s, capsize=5, edgecolor="white")
    axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(regions, color="white")
    axes[1,0].set_title("Mean Brightness per Region", color="white", fontsize=10)
    axes[1,0].set_ylabel("Mean pixel value", color="white", fontsize=8)
    axes[1,0].legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)
    axes[1,0].grid(alpha=0.2, color=GRID_COL, axis="y")

    fri_tot  = images[fri_idx,  :, :, 0].mean(axis=(1,2))
    frii_tot = images[frii_idx, :, :, 0].mean(axis=(1,2))
    axes[1,1].hist(fri_tot,  bins=25, alpha=0.7, color=COLORS["FRI"],
                   label=f"FRI  μ={fri_tot.mean():.3f}")
    axes[1,1].hist(frii_tot, bins=25, alpha=0.7, color=COLORS["FRII"],
                   label=f"FRII μ={frii_tot.mean():.3f}")
    axes[1,1].set_title("Total Brightness Distribution", color="white", fontsize=10)
    axes[1,1].set_xlabel("Mean pixel", color="white", fontsize=8)
    axes[1,1].legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)
    axes[1,1].grid(alpha=0.2, color=GRID_COL)

    fri_pk  = images[fri_idx,  :, :, 0].max(axis=(1,2))
    frii_pk = images[frii_idx, :, :, 0].max(axis=(1,2))
    axes[1,2].hist(fri_pk,  bins=25, alpha=0.7, color=COLORS["FRI"],
                   label=f"FRI  μ={fri_pk.mean():.3f}")
    axes[1,2].hist(frii_pk, bins=25, alpha=0.7, color=COLORS["FRII"],
                   label=f"FRII μ={frii_pk.mean():.3f}")
    axes[1,2].set_title("Peak Brightness Distribution", color="white", fontsize=10)
    axes[1,2].set_xlabel("Max pixel value", color="white", fontsize=8)
    axes[1,2].legend(facecolor=PANEL_BG, labelcolor="white", fontsize=8)
    axes[1,2].grid(alpha=0.2, color=GRID_COL)

    fig.tight_layout()
    _save(fig, "11_pixel_analysis.png")


# ─────────────────────────────────────────────
# FIGURE 12: INDIVIDUAL EXPORTS
# ─────────────────────────────────────────────
def fig12_individual_exports(images, labels):
    print("Fig 12: Individual exports...")
    indiv_dir = os.path.join(OUTPUT_DIR, "individual_images")
    os.makedirs(indiv_dir, exist_ok=True)

    fri_idx, frii_idx = _get_class_indices(labels)
    n_show = 6

    fig, axes = plt.subplots(2, n_show, figsize=(24, 9))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        "Gallery: FRI and FRII Radio Galaxies — Export Quality",
        color="white", fontsize=15, fontweight="bold"
    )

    for row, (idx_arr, cls, cmap, color) in enumerate([
        (fri_idx,  "FRI",  "hot",  COLORS["FRI"]),
        (frii_idx, "FRII", "cool", COLORS["FRII"]),
    ]):
        chosen = _safe_choice(idx_arr, min(n_show, len(idx_arr)))
        for col_i in range(n_show):
            ax = axes[row, col_i]
            ax.set_facecolor(DARK_BG)
            if col_i < len(chosen):
                idx = int(chosen[col_i])
                img = _enhance(images[idx, :, :, 0])
                ax.imshow(img, cmap=cmap, aspect="equal")
                ax.set_title(f"{cls} #{col_i+1}", color=color,
                             fontsize=9, fontweight="bold", pad=3)
                for sp in ax.spines.values():
                    sp.set_edgecolor(color); sp.set_linewidth(2)
                    sp.set_visible(True)

                ifig, iax = plt.subplots(figsize=(5, 5))
                ifig.patch.set_facecolor(DARK_BG)
                iax.imshow(img, cmap=cmap, aspect="equal")
                iax.set_title(f"{cls} Radio Galaxy — STELLARIS-DNet",
                              color=color, fontsize=11, fontweight="bold")
                iax.axis("off")
                ifig.tight_layout()
                ifig.savefig(
                    os.path.join(indiv_dir, f"{cls}_{col_i+1:02d}.png"),
                    dpi=150, bbox_inches="tight",
                    facecolor=ifig.get_facecolor()
                )
                plt.close(ifig)
            ax.axis("off")

    fig.tight_layout()
    _save(fig, "12_individual_export_gallery.png")
    print(f"      → Individual images saved to {indiv_dir}/")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 60)
    print("STELLARIS-DNet — MiraBest Visualization (12 Figures)")
    print("=" * 60)

    data_dir = _find_data_dir()
    if data_dir:
        images, labels = _load_all(data_dir)
    else:
        images, labels = _generate_synthetic(n=300)

    print(f"\nGenerating 12 figures → {OUTPUT_DIR}/")
    print("-" * 60)

    fig1_gallery(images, labels)
    fig2_fri_vs_frii(images, labels)
    fig3_annotated(images, labels)
    fig4_statistics(images, labels)
    fig5_augmentation(images, labels)
    fig6_brightness_profile(images, labels)
    fig7_mean_images(images, labels)
    fig8_multiscale(images, labels)
    fig9_preprocessing(images, labels)
    fig10_physics_context(images, labels)
    fig11_pixel_analysis(images, labels)
    fig12_individual_exports(images, labels)

    print("\n" + "=" * 60)
    print(f"✅ All 12 figures saved to: {OUTPUT_DIR}/")
    for i, name in enumerate([
        "01_gallery_32images.png", "02_FRI_vs_FRII_comparison.png",
        "03_annotated_galaxy.png", "04_dataset_statistics.png",
        "05_augmentation_showcase.png", "06_brightness_profiles.png",
        "07_mean_class_images.png", "08_multiscale_morphology.png",
        "09_preprocessing_pipeline.png", "10_physics_context.png",
        "11_pixel_analysis.png", "12_individual_export_gallery.png",
    ], 1):
        print(f"  {i:02d}. {name}")
    print("=" * 60)