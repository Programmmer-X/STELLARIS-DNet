"""
notebooks/visualize_mirabest.py
STELLARIS-DNet — MiraBest Radio Galaxy Visualization

LOCAL (VS Code): Uses real MiraBest batch files
KAGGLE:          Uses real MiraBest batch files from input dataset

Run locally:  python notebooks/visualize_mirabest.py
Run on Kaggle: !python notebooks/visualize_mirabest.py

Saves to: notebooks/mirabest_visuals/
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

LABEL_MAP = {
    0: ("FRI",    "Confident"),
    1: ("FRI",    "Uncertain"),
    2: ("FRII",   "Confident"),
    3: ("FRII",   "Uncertain"),
    4: ("Hybrid", "Uncertain"),
}
CLASS_COLORS = {
    "FRI":    "#FF6B35",
    "FRII":   "#4ECDC4",
    "Hybrid": "#95A5A6",
}


def _find_data_dir():
    candidates = [
        RGZ_DATA_DIR,
        "data/module2/mirabest",
        "/kaggle/working/STELLARIS-DNet/data/module2/mirabest",
        "/kaggle/input/datasets/programmmerasx/mirabest-radio-galaxy",
    ]
    for path in candidates:
        if os.path.exists(path) and any(
            os.path.exists(os.path.join(path, f"data_batch_{i}"))
            for i in range(1, 9)
        ):
            print(f"✅ MiraBest found at: {path}")
            return path
    raise FileNotFoundError(
        "MiraBest not found.\n"
        "Locally: data/module2/mirabest/\n"
        "Kaggle:  add mirabest-radio-galaxy to inputs"
    )


def load_all_batches(data_dir):
    all_images, all_labels = [], []
    for i in range(1, 9):
        path = os.path.join(data_dir, f"data_batch_{i}")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        all_images.append(np.array(d.get(b"data",   d.get("data"))))
        all_labels.append(np.array(d.get(b"labels", d.get("labels"))))

    test_path = os.path.join(data_dir, "test_batch")
    if os.path.exists(test_path):
        with open(test_path, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        all_images.append(np.array(d.get(b"data",   d.get("data"))))
        all_labels.append(np.array(d.get(b"labels", d.get("labels"))))

    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    n = len(images)
    try:
        images = images.reshape(n, 3, 150, 150).astype(np.float32) / 255.0
        images = images.transpose(0, 2, 3, 1)
    except ValueError:
        images = images.reshape(n, 150, 150, -1).astype(np.float32) / 255.0

    print(f"✅ Loaded {n} real MiraBest images")
    print(f"   FRI:  {(labels <= 1).sum()}")
    print(f"   FRII: {((labels == 2) | (labels == 3)).sum()}")
    return images, labels


def _enhance(img):
    img = np.clip(img, 0, 1)
    p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
    if p98 > p2:
        img = np.clip((img - p2) / (p98 - p2), 0, 1)
    return img


# ─────────────────────────────────────────────
# PLOT 1: GALLERY 32 REAL IMAGES
# ─────────────────────────────────────────────
def plot_gallery(images, labels, n=32):
    print("Generating real image gallery...")
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0a0a0a")
    fig.text(0.5, 0.97,
             "MiraBest Dataset — Real Radio Galaxy Images",
             ha="center", fontsize=22, color="white", fontweight="bold")
    fig.text(0.5, 0.94,
             "1,256 real VLA radio telescope observations — "
             "AGN footprints of supermassive black holes",
             ha="center", fontsize=11, color="#aaaaaa")

    axes = fig.subplots(4, 8)
    fig.subplots_adjust(hspace=0.08, wspace=0.05,
                        top=0.91, bottom=0.05,
                        left=0.02, right=0.98)

    fri_idx  = np.where(labels == 0)[0]
    frii_idx = np.where(labels == 2)[0]
    chosen   = np.concatenate([
        np.random.choice(fri_idx,  min(n//2, len(fri_idx)),  replace=False),
        np.random.choice(frii_idx, min(n//2, len(frii_idx)), replace=False)
    ])
    np.random.shuffle(chosen)
    chosen = chosen[:n]

    cmaps = {0: "hot", 1: "hot", 2: "cool", 3: "cool", 4: "gray"}

    for i, ax in enumerate(axes.flat):
        ax.set_facecolor("#0a0a0a")
        if i >= len(chosen):
            ax.axis("off")
            continue
        idx   = chosen[i]
        lbl   = int(labels[idx])
        cls   = LABEL_MAP.get(lbl, ("?", ""))[0]
        color = CLASS_COLORS.get(cls, "white")
        ax.imshow(_enhance(images[idx])[:, :, 0],
                  cmap=cmaps.get(lbl, "hot"), aspect="auto")
        ax.set_title(cls, fontsize=7, color=color,
                     fontweight="bold", pad=2)
        ax.axis("off")
        for sp in ax.spines.values():
            sp.set_edgecolor(color)
            sp.set_linewidth(1.5)
            sp.set_visible(True)

    fig.text(0.08, 0.02, "■ FRI (jets fade from center)",
             color="#FF6B35", fontsize=11)
    fig.text(0.38, 0.02, "■ FRII (bright hotspots at jet ends)",
             color="#4ECDC4", fontsize=11)

    path = os.path.join(OUTPUT_DIR, "1_real_gallery_32images.png")
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   ✅ {path}")


# ─────────────────────────────────────────────
# PLOT 2: FRI vs FRII SIDE BY SIDE
# ─────────────────────────────────────────────
def plot_fri_vs_frii(images, labels):
    print("Generating FRI vs FRII comparison...")
    fri_idx  = np.where(labels == 0)[0]
    frii_idx = np.where(labels == 2)[0]
    n_each   = 5
    fri_s    = np.random.choice(fri_idx,  min(n_each, len(fri_idx)),  replace=False)
    frii_s   = np.random.choice(frii_idx, min(n_each, len(frii_idx)), replace=False)

    fig = plt.figure(figsize=(20, 9))
    fig.patch.set_facecolor("#0d0d1a")
    fig.text(0.5, 0.97,
             "Fanaroff-Riley Classification — Real Radio Galaxy Images",
             ha="center", fontsize=18, color="white", fontweight="bold")
    fig.text(0.5, 0.93,
             "Both AGN types powered by supermassive black holes — "
             "separated by jet power boundary 10²⁵ W/Hz",
             ha="center", fontsize=11, color="#aaaaaa")

    gs = gridspec.GridSpec(2, n_each*2+1, figure=fig,
                           hspace=0.15, wspace=0.08,
                           top=0.90, bottom=0.08)

    for ax_pos, text, color in [
        (gs[0, :n_each],
         "TYPE I  (FRI)\nJets fade from center\nLower BH power  < 10²⁵ W/Hz",
         "#FF6B35"),
        (gs[0, n_each+1:],
         "TYPE II  (FRII)\nBright hotspots at jet ends\nHigher BH power  > 10²⁵ W/Hz",
         "#4ECDC4"),
    ]:
        ax = fig.add_subplot(ax_pos)
        ax.text(0.5, 0.5, text, ha="center", va="center",
                fontsize=12, color=color, fontweight="bold",
                transform=ax.transAxes)
        ax.axis("off")

    ax_d = fig.add_subplot(gs[:, n_each])
    ax_d.text(0.5, 0.5, "vs", ha="center", va="center",
              fontsize=18, color="white", fontweight="bold",
              transform=ax_d.transAxes)
    ax_d.axis("off")

    for i, idx in enumerate(fri_s):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(_enhance(images[idx])[:, :, 0], cmap="hot")
        ax.axis("off")
        for sp in ax.spines.values():
            sp.set_edgecolor("#FF6B35"); sp.set_linewidth(2); sp.set_visible(True)

    for i, idx in enumerate(frii_s):
        ax = fig.add_subplot(gs[1, n_each+1+i])
        ax.imshow(_enhance(images[idx])[:, :, 0], cmap="cool")
        ax.axis("off")
        for sp in ax.spines.values():
            sp.set_edgecolor("#4ECDC4"); sp.set_linewidth(2); sp.set_visible(True)

    fig.text(0.5, 0.02,
             "MiraBest DR1  |  VLA 1.4 GHz  |  Expert-labeled",
             ha="center", fontsize=9, color="#555555", style="italic")

    path = os.path.join(OUTPUT_DIR, "2_real_FRI_vs_FRII.png")
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   ✅ {path}")


# ─────────────────────────────────────────────
# PLOT 3: ANNOTATED IMAGE
# ─────────────────────────────────────────────
def plot_annotated(images, labels):
    print("Generating annotated image...")
    frii_idx = np.where(labels == 2)[0]
    idx      = frii_idx[np.random.randint(len(frii_idx))]
    img      = _enhance(images[idx])

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")
    ax.imshow(img[:, :, 0], cmap="inferno")
    ax.axis("off")
    ax.set_title(
        "Real FRII Radio Galaxy — AGN Jet Structure\n"
        "Supermassive Black Hole at center drives relativistic plasma jets",
        color="white", fontsize=12, fontweight="bold", pad=10
    )
    for x, y, dx, dy, label, color in [
        (75,  75,  -40, -35, "Core\n(SMBH)",       "yellow"),
        (25,  35,  -40,  15, "Hotspot\n(jet end)",  "#4ECDC4"),
        (125, 115,  35,  15, "Hotspot\n(jet end)",  "#4ECDC4"),
        (45,  75,  -40,   0, "Radio Lobe",           "#FF6B35"),
        (105, 75,   35,   0, "Radio Lobe",           "#FF6B35"),
    ]:
        if x < img.shape[1] and y < img.shape[0]:
            ax.annotate(label, xy=(x, y), xytext=(x+dx, y+dy),
                        color=color, fontsize=9, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="#0d0d1a",
                                  edgecolor=color, alpha=0.8))

    fig.text(0.5, 0.01,
             "MiraBest DR1  |  STELLARIS-DNet Module 2A",
             ha="center", fontsize=9, color="#555555", style="italic")

    path = os.path.join(OUTPUT_DIR, "3_annotated_real_FRII.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   ✅ {path}")


# ─────────────────────────────────────────────
# PLOT 4: STATISTICS
# ─────────────────────────────────────────────
def plot_statistics(images, labels):
    print("Generating statistics...")
    fri_n  = (labels == 0).sum()
    frii_n = (labels == 2).sum()
    fri_u  = (labels == 1).sum()
    frii_u = (labels == 3).sum()
    hyb_n  = (labels == 4).sum()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d0d1a")
    fig.suptitle("MiraBest Dataset Statistics — Real Data",
                 fontsize=16, color="white", fontweight="bold")

    for ax in axes:
        ax.set_facecolor("#111122")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")

    cats   = ["FRI\n(conf)", "FRI\n(unc)", "FRII\n(conf)", "FRII\n(unc)", "Hybrid"]
    counts = [fri_n, fri_u, frii_n, frii_u, hyb_n]
    colors = ["#FF6B35", "#FF9966", "#4ECDC4", "#88DDDD", "#95A5A6"]
    bars   = axes[0].bar(cats, counts, color=colors, edgecolor="white")
    axes[0].set_title("Class Distribution", color="white", fontsize=13)
    axes[0].set_ylabel("Count", color="white")
    for bar, cnt in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 2, str(cnt),
                     ha="center", color="white", fontsize=10)
    axes[0].set_xticklabels(cats, color="white", fontsize=9)

    pie_v = [fri_n+fri_u, frii_n+frii_u]
    pie_l = [f"FRI\n{fri_n+fri_u}", f"FRII\n{frii_n+frii_u}"]
    if hyb_n > 0:
        pie_v.append(hyb_n); pie_l.append(f"Hybrid\n{hyb_n}")
    _, _, autotexts = axes[1].pie(
        pie_v, labels=pie_l,
        colors=["#FF6B35", "#4ECDC4", "#95A5A6"][:len(pie_v)],
        autopct="%1.1f%%", startangle=90,
        textprops={"color": "white", "fontsize": 11}
    )
    for at in autotexts:
        at.set_color("black"); at.set_fontweight("bold")
    axes[1].set_title("FRI vs FRII Split", color="white", fontsize=13)

    fri_imgs  = images[labels == 0, :, :, 0].flatten()
    frii_imgs = images[labels == 2, :, :, 0].flatten()
    axes[2].hist(fri_imgs,  bins=50, alpha=0.7,
                 color="#FF6B35", label="FRI",  density=True)
    axes[2].hist(frii_imgs, bins=50, alpha=0.7,
                 color="#4ECDC4", label="FRII", density=True)
    axes[2].set_title("Pixel Intensity Distribution",
                       color="white", fontsize=13)
    axes[2].set_xlabel("Pixel Value", color="white")
    axes[2].set_ylabel("Density",     color="white")
    axes[2].legend(facecolor="#111122", labelcolor="white")

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "4_dataset_statistics.png")
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"   ✅ {path}")


# ─────────────────────────────────────────────
# PLOT 5: INDIVIDUAL EXPORTS
# ─────────────────────────────────────────────
def export_individual(images, labels, n=10):
    print("Exporting individual images...")
    indiv_dir = os.path.join(OUTPUT_DIR, "individual_images")
    os.makedirs(indiv_dir, exist_ok=True)

    for cls, idxs, cmap, color in [
        ("FRI",  np.where(labels == 0)[0], "hot",  "#FF6B35"),
        ("FRII", np.where(labels == 2)[0], "cool", "#4ECDC4"),
    ]:
        chosen = np.random.choice(idxs, min(n//2, len(idxs)), replace=False)
        for i, idx in enumerate(chosen):
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor("#0d0d1a")
            ax.set_facecolor("#0d0d1a")
            ax.imshow(_enhance(images[idx])[:, :, 0], cmap=cmap)
            ax.set_title(f"Real {cls} Radio Galaxy\nAGN jet — SMBH powered",
                         color=color, fontsize=9, fontweight="bold")
            ax.axis("off")
            plt.savefig(os.path.join(indiv_dir, f"{cls}_{i+1:02d}.png"),
                        dpi=120, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close()

    print(f"   ✅ {n} images → {indiv_dir}/")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 60)
    print("STELLARIS-DNet — MiraBest Real Image Visualization")
    print("=" * 60)

    data_dir       = _find_data_dir()
    images, labels = load_all_batches(data_dir)

    print(f"\nGenerating → {OUTPUT_DIR}/")
    print("-" * 60)

    plot_gallery(images, labels)
    plot_fri_vs_frii(images, labels)
    plot_annotated(images, labels)
    plot_statistics(images, labels)
    export_individual(images, labels, n=10)

    print("\n" + "=" * 60)
    print("✅ Done! Files:")
    print("   1_real_gallery_32images.png")
    print("   2_real_FRI_vs_FRII.png")
    print("   3_annotated_real_FRII.png")
    print("   4_dataset_statistics.png")
    print("   individual_images/  (10 PNGs)")
    print("=" * 60)