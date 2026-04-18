"""
module1/config.py
STELLARIS-DNet — Module 1 Hyperparameters
All tunable values in one place. Never hardcode in model/train files.
"""

# ─────────────────────────────────────────────
# GENERAL
# ─────────────────────────────────────────────
SEED        = 42
DEVICE      = "auto"          # "auto" | "cpu" | "cuda"
CHECKPOINT_DIR = "checkpoints/module1"
LOG_DIR        = "logs/module1"

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
HTRU2_PATH      = "data/module1/HTRU_2.csv"
LOTAAS_PATH     = "data/module1/lotaas_profiles.npy"   # raw pulse profiles
SIGNAL_LENGTH   = 64        # fixed length for 1D pulse profiles
TEST_SPLIT      = 0.15      # 15% test
VAL_SPLIT       = 0.15      # 15% validation
NORMALIZE       = True

# ─────────────────────────────────────────────
# CLASS DEFINITIONS
# ─────────────────────────────────────────────
# HTRU2: Binary (pulsar=1, non-pulsar=0)
HTRU2_CLASSES   = ["Non-Pulsar", "Pulsar"]

# Pulsar subtypes (for 1D CNN on raw profiles)
PULSAR_CLASSES  = [
    "Normal",           # 0 — canonical pulsar
    "Millisecond",      # 1 — recycled, fast spin
    "Binary",           # 2 — in binary system
    "Recycled",         # 3 — spun up by companion
]
NUM_PULSAR_CLASSES = len(PULSAR_CLASSES)

# ─────────────────────────────────────────────
# MLP CONFIG (HTRU2 binary classifier)
# ─────────────────────────────────────────────
MLP_INPUT_DIM   = 8         # HTRU2 has 8 statistical features
MLP_HIDDEN_DIMS = [64, 32]  # layer sizes
MLP_DROPOUT     = 0.3
MLP_LR          = 1e-3
MLP_EPOCHS      = 50
MLP_BATCH_SIZE  = 128
MLP_WEIGHT_DECAY = 1e-4

# ─────────────────────────────────────────────
# 1D CNN CONFIG (Pulsar subtype classifier)
# ─────────────────────────────────────────────
CNN_IN_CHANNELS  = 1
CNN_CHANNELS     = [64, 128, 256]   # conv layer output channels
CNN_KERNEL_SIZES = [7, 5, 3]        # kernel per layer
CNN_DROPOUT      = 0.3
CNN_LR           = 1e-3
CNN_EPOCHS       = 60
CNN_BATCH_SIZE   = 64
CNN_WEIGHT_DECAY = 1e-4

# Encoder output dimension (used in unified model)
CNN_ENCODER_DIM  = 256

# ─────────────────────────────────────────────
# AUTOENCODER CONFIG (Magnetar anomaly detector)
# Trained ONLY on normal pulsar signals
# High reconstruction error = magnetar candidate
# ─────────────────────────────────────────────
AE_INPUT_DIM     = 64       # matches SIGNAL_LENGTH
AE_LATENT_DIM    = 16       # bottleneck size
AE_HIDDEN_DIMS   = [48, 32] # encoder hidden layers
AE_DROPOUT       = 0.2
AE_LR            = 1e-3
AE_EPOCHS        = 50
AE_BATCH_SIZE    = 64

# Anomaly threshold — predictions above this = magnetar flag
# Set automatically during evaluate.py using 95th percentile of val errors
AE_ANOMALY_PERCENTILE = 95

# ─────────────────────────────────────────────
# PHYSICS CONFIG (Module 1 specific)
# ─────────────────────────────────────────────
# Spin-down energy loss weight in combined loss
SPINDOWN_LOSS_WEIGHT = 0.1

# Typical pulsar period range (seconds)
PERIOD_MIN = 0.001          # 1ms  (millisecond pulsars)
PERIOD_MAX = 10.0           # 10s  (slowest known pulsars)

# ─────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────
PATIENCE    = 10
MIN_DELTA   = 1e-4

# ─────────────────────────────────────────────
# CLASS IMBALANCE (HTRU2 is ~9:1)
# ─────────────────────────────────────────────
USE_CLASS_WEIGHTS = True    # weight loss by inverse class frequency
USE_OVERSAMPLE    = False   # SMOTE oversampling (slower, try if weights fail)