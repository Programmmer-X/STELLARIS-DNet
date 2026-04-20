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
# DATASET PATHS — simple constants only
# Kaggle path resolution happens in dataset files
# ─────────────────────────────────────────────
RGZ_DATA_DIR  = "data/module2/mirabest"
LIGO_DATA_DIR = "data/module2/ligo"

# ─────────────────────────────────────────────
# SUB-TASK 2A — MIRABEST RADIO GALAXIES
# EfficientNet-B0, transfer learning
# ─────────────────────────────────────────────
RGZ_CLASSES       = ["FRI", "FRII"]
RGZ_NUM_CLASSES   = 2
RGZ_IMG_SIZE      = 224
RGZ_CHANNELS      = 3
RGZ_AUGMENT       = True
RGZ_EPOCHS        = 50
RGZ_BATCH_SIZE    = 32
RGZ_LR            = 1e-4
RGZ_LR_BACKBONE   = 1e-5
RGZ_WEIGHT_DECAY  = 1e-3
RGZ_DROPOUT       = 0.5
RGZ_FREEZE_EPOCHS = 10
RGZ_TEST_SPLIT    = 0.15
RGZ_VAL_SPLIT     = 0.15
RGZ_ENCODER_DIM   = 256

# ─────────────────────────────────────────────
# SUB-TASK 2B — G2NET GRAVITATIONAL WAVES
# 1D CNN on whitened strain data
# Binary: Noise(0) / Signal(1)
# ─────────────────────────────────────────────
LIGO_CLASSES      = ["Noise", "Signal"]
LIGO_NUM_CLASSES  = 2
LIGO_N_DETECTORS  = 3          # H1, L1, V1
LIGO_SIGNAL_LEN   = 4096       # samples per observation
LIGO_EPOCHS       = 50
LIGO_BATCH_SIZE   = 64
LIGO_LR           = 1e-4       # FIXED: was 1e-3, too high
LIGO_WEIGHT_DECAY = 1e-4
LIGO_DROPOUT      = 0.3
LIGO_CHANNELS     = [64, 128, 256, 256]
LIGO_KERNEL_SIZES = [15, 11, 7, 5]
LIGO_ENCODER_DIM  = 256
LIGO_TEST_SPLIT   = 0.15
LIGO_VAL_SPLIT    = 0.15
LIGO_MAX_SAMPLES  = 10000      # limit for memory

# ─────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────
PATIENCE  = 15
MIN_DELTA = 1e-4

# ─────────────────────────────────────────────
# PHYSICS BOUNDARIES
# ─────────────────────────────────────────────
FRI_FRII_POWER_BOUNDARY = 1e25   # W/Hz
GW_FREQ_MIN = 20                 # Hz
GW_FREQ_MAX = 500                # Hz