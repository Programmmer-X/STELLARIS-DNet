"""
module2/config.py
STELLARIS-DNet — Module 2 Hyperparameters
Both 2A and 2B use EfficientNet-B0
2A: Radio galaxy images
2B: GW CQT spectrograms
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
RGZ_DATA_DIR  = "data/module2/mirabest"
LIGO_DATA_DIR = "data/module2/ligo"

# ─────────────────────────────────────────────
# SHARED EFFICIENTNET SETTINGS
# Both 2A and 2B use these for the backbone
# ─────────────────────────────────────────────
ENCODER_DIM   = 256    # unified encoder dim for both tasks

# ─────────────────────────────────────────────
# SUB-TASK 2A — MIRABEST RADIO GALAXIES
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
RGZ_ENCODER_DIM   = ENCODER_DIM

# ─────────────────────────────────────────────
# SUB-TASK 2B — G2NET GRAVITATIONAL WAVES
# EfficientNet-B0 on CQT spectrograms
# ─────────────────────────────────────────────
LIGO_CLASSES       = ["Noise", "Signal"]
LIGO_NUM_CLASSES   = 2
LIGO_N_DETECTORS   = 3          # H1, L1, V1
LIGO_SIGNAL_LEN    = 4096       # samples per observation

# CQT Spectrogram output shape
LIGO_CQT_BINS      = 64         # frequency bins (height)
LIGO_CQT_STEPS     = 64         # time steps (width)
LIGO_IMG_SIZE      = 224        # resize to EfficientNet input

LIGO_EPOCHS        = 50
LIGO_BATCH_SIZE    = 32
LIGO_LR            = 1e-4
LIGO_LR_BACKBONE   = 1e-5
LIGO_WEIGHT_DECAY  = 1e-3
LIGO_DROPOUT       = 0.5
LIGO_FREEZE_EPOCHS = 10
LIGO_TEST_SPLIT    = 0.15
LIGO_VAL_SPLIT     = 0.15
LIGO_ENCODER_DIM   = ENCODER_DIM
LIGO_MAX_SAMPLES   = 4000       # keep manageable for CQT compute

# ─────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────
PATIENCE  = 15
MIN_DELTA = 1e-4

# ─────────────────────────────────────────────
# PHYSICS BOUNDARIES
# ─────────────────────────────────────────────
FRI_FRII_POWER_BOUNDARY = 1e25   # W/Hz
GW_FREQ_MIN = 20                 # Hz — LIGO lower cutoff
GW_FREQ_MAX = 500                # Hz — LIGO upper cutoff