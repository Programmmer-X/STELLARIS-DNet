"""
module2/config.py
STELLARIS-DNet — Module 2 Hyperparameters
Sub-task 2A: Radio Galaxy Zoo (EfficientNet-B3)
Sub-task 2B: LIGO Gravitational Wave (1D CNN)
"""

# ─────────────────────────────────────────────
# GENERAL
# ─────────────────────────────────────────────
SEED           = 42
DEVICE         = "auto"
CHECKPOINT_DIR = "checkpoints/module2"
LOG_DIR        = "logs/module2"

# ─────────────────────────────────────────────
# DATASET PATHS
# ─────────────────────────────────────────────
RGZ_DATA_DIR   = "data/module2/rgz"          # Radio Galaxy Zoo images
LIGO_DATA_DIR  = "data/module2/ligo"         # LIGO strain .hdf5 files

# ─────────────────────────────────────────────
# SUB-TASK 2A — RADIO GALAXY ZOO
# EfficientNet-B3 image classifier
# Classes: radio galaxy morphologies caused by SMBH jets
# ─────────────────────────────────────────────
RGZ_CLASSES = [
    "FRI",        # 0 — Fanaroff-Riley Type I  (jets fade from center)
    "FRII",       # 1 — Fanaroff-Riley Type II (hotspots at jet ends)
    "Compact",    # 2 — Compact steep spectrum (young/confined BH jet)
    "Bent",       # 3 — Bent-tailed (jets bent by environment)
]
RGZ_NUM_CLASSES  = len(RGZ_CLASSES)

# Image settings
RGZ_IMG_SIZE     = 224        # EfficientNet-B3 input size
RGZ_CHANNELS     = 3          # RGB
RGZ_AUGMENT      = True       # augmentation during training

# Train settings
RGZ_EPOCHS       = 30
RGZ_BATCH_SIZE   = 32
RGZ_LR           = 1e-4       # low LR for transfer learning
RGZ_LR_BACKBONE  = 1e-5       # even lower for pretrained backbone
RGZ_WEIGHT_DECAY = 1e-4
RGZ_DROPOUT      = 0.4
RGZ_FREEZE_EPOCHS = 5         # freeze backbone for first 5 epochs

# Split
RGZ_TEST_SPLIT   = 0.15
RGZ_VAL_SPLIT    = 0.15

# Encoder output dim for unified model
RGZ_ENCODER_DIM  = 256

# ─────────────────────────────────────────────
# SUB-TASK 2B — LIGO GRAVITATIONAL WAVES
# 1D CNN on strain time-series
# Classes: type of GW event detected
# ─────────────────────────────────────────────
LIGO_CLASSES = [
    "BBH",        # 0 — Binary Black Hole merger
    "BNS",        # 1 — Binary Neutron Star merger
    "Noise",      # 2 — Glitch / non-astrophysical noise
]
LIGO_NUM_CLASSES  = len(LIGO_CLASSES)

# Signal settings
LIGO_SAMPLE_RATE  = 4096      # Hz — LIGO standard sample rate
LIGO_DURATION     = 4         # seconds of data per sample
LIGO_SIGNAL_LEN   = LIGO_SAMPLE_RATE * LIGO_DURATION  # 16384 points

# Spectrogram settings (Q-transform → 2D image for CNN)
LIGO_USE_SPECTROGRAM = True   # convert strain to spectrogram
LIGO_SPEC_HEIGHT     = 64     # spectrogram height
LIGO_SPEC_WIDTH      = 64     # spectrogram width

# Train settings
LIGO_EPOCHS       = 40
LIGO_BATCH_SIZE   = 32
LIGO_LR           = 1e-3
LIGO_WEIGHT_DECAY = 1e-4
LIGO_DROPOUT      = 0.3

# 1D CNN architecture
LIGO_CHANNELS     = [32, 64, 128, 256]
LIGO_KERNEL_SIZES = [15, 11, 7, 5]
LIGO_ENCODER_DIM  = 256       # matches RGZ for unified fusion

# Split
LIGO_TEST_SPLIT   = 0.15
LIGO_VAL_SPLIT    = 0.15

# ─────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────
PATIENCE  = 10
MIN_DELTA = 1e-4

# ─────────────────────────────────────────────
# PHYSICS NOTES (enforced in evaluate.py)
# FRI:    jet power < 10^25 W/Hz  → lower BH mass/spin
# FRII:   jet power > 10^25 W/Hz  → higher BH mass/spin
# BBH:    total mass 5-150 M_sun  → stellar BH range
# BNS:    total mass 1-3  M_sun   → neutron star range
# ─────────────────────────────────────────────
FRI_FRII_POWER_BOUNDARY = 1e25   # W/Hz — Fanaroff-Riley boundary
BBH_MASS_RANGE  = (5,   150)     # solar masses
BNS_MASS_RANGE  = (1.0,   3.0)   # solar masses