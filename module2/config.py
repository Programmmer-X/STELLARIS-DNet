"""
module2/config.py
STELLARIS-DNet — Module 2 Hyperparameters
Sub-task 2A: MiraBest Radio Galaxies (EfficientNet-B0)
Sub-task 2B: G2Net Gravitational Wave (1D CNN)
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
RGZ_DATA_DIR  = "data/module2/mirabest"   # MiraBest CIFAR batches
LIGO_DATA_DIR = "data/module2/ligo"       # G2Net .npy files

# ─────────────────────────────────────────────
# SUB-TASK 2A — MIRABEST RADIO GALAXIES
# EfficientNet-B0 image classifier
# FRI/FRII morphologies directly caused by SMBH jets
# ─────────────────────────────────────────────
RGZ_CLASSES      = ["FRI", "FRII"]
RGZ_NUM_CLASSES  = 2

# Image settings
RGZ_IMG_SIZE     = 224        # EfficientNet-B0 input size
RGZ_CHANNELS     = 3          # RGB
RGZ_AUGMENT      = True

# Train settings — tuned for small dataset (~1200 samples)
RGZ_EPOCHS        = 50
RGZ_BATCH_SIZE    = 32
RGZ_LR            = 1e-4      # low LR for transfer learning
RGZ_LR_BACKBONE   = 1e-5     # very low for pretrained backbone
RGZ_WEIGHT_DECAY  = 1e-3     # strong regularization for small dataset
RGZ_DROPOUT       = 0.5      # heavy dropout to prevent overfitting
RGZ_FREEZE_EPOCHS = 10       # freeze backbone for first 10 epochs

# Split
RGZ_TEST_SPLIT   = 0.15
RGZ_VAL_SPLIT    = 0.15

# Encoder output dim — must match LIGO and unified model
RGZ_ENCODER_DIM  = 256

# ─────────────────────────────────────────────
# SUB-TASK 2B — G2NET GRAVITATIONAL WAVES
# 1D CNN on strain time-series
# Binary: Signal (GW present) / Noise
# ─────────────────────────────────────────────
LIGO_CLASSES     = ["Noise", "Signal"]
LIGO_NUM_CLASSES = 2

# Signal settings — G2Net format
LIGO_SIGNAL_LEN  = 4096      # samples per observation (G2Net standard)
LIGO_N_DETECTORS = 3         # LIGO Hanford + Livingston + Virgo

# Train settings
LIGO_EPOCHS       = 40
LIGO_BATCH_SIZE   = 64
LIGO_LR           = 1e-3
LIGO_WEIGHT_DECAY = 1e-4
LIGO_DROPOUT      = 0.3

# 1D CNN architecture
LIGO_CHANNELS     = [32, 64, 128, 256]
LIGO_KERNEL_SIZES = [15, 11, 7, 5]
LIGO_ENCODER_DIM  = 256      # matches RGZ for unified fusion

# Split
LIGO_TEST_SPLIT  = 0.15
LIGO_VAL_SPLIT   = 0.15

# ─────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────
PATIENCE  = 10
MIN_DELTA = 1e-4

# ─────────────────────────────────────────────
# PHYSICS BOUNDARIES
# FRI:    jet power < 10^25 W/Hz → lower BH mass/spin
# FRII:   jet power > 10^25 W/Hz → higher BH mass/spin
# Signal: GW frequency 20-500 Hz (LIGO sensitive band)
# ─────────────────────────────────────────────
FRI_FRII_POWER_BOUNDARY = 1e25   # W/Hz
GW_FREQ_MIN = 20                 # Hz
GW_FREQ_MAX = 500                # Hz