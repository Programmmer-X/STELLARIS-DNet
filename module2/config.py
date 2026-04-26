"""
module2/config.py
STELLARIS-DNet — Module 2 Hyperparameters (Upgraded)
2A: MiraBest Radio Galaxies  → EfficientNet-B2 + CBAM + GeM
2B: G2Net Gravitational Waves → EfficientNet-B2 + CBAM + GeM + Transformer
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
# BACKBONE UPGRADE: B0 → B2
# torchvision >= 0.13 required for B2
# Auto-falls back to B0 (1280) if unavailable
# ─────────────────────────────────────────────
BACKBONE              = "efficientnet_b2"
EFFICIENTNET_FEAT_DIM = 1408      # B2 final conv channels (B0=1280)

# ─────────────────────────────────────────────
# SHARED ENCODER DIM
# ─────────────────────────────────────────────
ENCODER_DIM = 256

# ─────────────────────────────────────────────
# CBAM — Convolutional Block Attention Module
# Channel + spatial attention on CNN feature map
# Focuses on jet structures (2A) and chirp regions (2B)
# ─────────────────────────────────────────────
USE_CBAM       = True
CBAM_REDUCTION = 16               # channel squeeze ratio in CBAM

# ─────────────────────────────────────────────
# GeM POOLING — Generalised Mean Pooling
# Replaces AdaptiveAvgPool2d
# Higher p → sharper focus on peak activations
# Better for fine-grained detection (hotspots, chirp peaks)
# ─────────────────────────────────────────────
USE_GEM       = True
GEM_P         = 3.0
GEM_EPS       = 1e-6
GEM_LEARNABLE = True              # p is a trainable parameter

# ─────────────────────────────────────────────
# TRANSFORMER — CNN Feature Map → Seq → Transformer → GeM
# 2A: OFF — MiraBest ~1000 samples (overfitting risk)
# 2B: ON  — G2Net 4000+ samples (global dependencies useful)
# ─────────────────────────────────────────────
USE_TRANSFORMER_2A  = False
USE_TRANSFORMER_2B  = True
TRANSFORMER_DIM     = 256         # project feat_dim → this before transformer
TRANSFORMER_HEADS   = 8
TRANSFORMER_LAYERS  = 2
TRANSFORMER_FF_DIM  = 512
TRANSFORMER_DROPOUT = 0.1

# ─────────────────────────────────────────────
# PHYSICS CONSTRAINTS
# ─────────────────────────────────────────────
USE_PHYSICS_LOSS = True           # master switch

# 2A — Fanaroff-Riley Boundary
# FRI predicted jet power < 10^25 W/Hz < FRII
FRI_FRII_POWER_BOUNDARY = 1e25   # W/Hz
FRI_FRII_BOUNDARY_LOG   = 25.0   # log10(W/Hz) for loss computation
USE_JET_POWER_HEAD      = True   # auxiliary regression head on encoder
JET_POWER_LOSS_WEIGHT   = 0.05

# 2B — GW Chirp Slope Consistency
# Frequency centroid of predicted signals must rise over time
# Enforces chirp: f(t) ∝ (t_c − t)^(−3/8)
USE_CHIRP_LOSS    = True
CHIRP_LOSS_WEIGHT = 0.1
GW_FREQ_MIN       = 20           # Hz
GW_FREQ_MAX       = 500          # Hz

# ─────────────────────────────────────────────
# SUB-TASK 2A — MIRABEST RADIO GALAXIES
# ─────────────────────────────────────────────
RGZ_CLASSES       = ["FRI", "FRII"]
RGZ_NUM_CLASSES   = 2
RGZ_IMG_SIZE      = 224          # EfficientNet input (upsampled from 150×150)
RGZ_CHANNELS      = 3
RGZ_AUGMENT       = True

RGZ_EPOCHS        = 50
RGZ_BATCH_SIZE    = 32
RGZ_LR            = 1e-4
RGZ_LR_BACKBONE   = 1e-5         # lower LR for pretrained weights
RGZ_LR_MIN        = 1e-6         # cosine annealing floor
RGZ_WEIGHT_DECAY  = 1e-3
RGZ_DROPOUT       = 0.4          # reduced from 0.5 (B2 is more regularized)
RGZ_FREEZE_EPOCHS = 10           # epochs before backbone unfreezing
RGZ_WARMUP_EPOCHS = 3            # linear LR warmup epochs
RGZ_TEST_SPLIT    = 0.15
RGZ_VAL_SPLIT     = 0.15
RGZ_ENCODER_DIM   = ENCODER_DIM

# ─────────────────────────────────────────────
# SUB-TASK 2B — G2NET GRAVITATIONAL WAVES
# ─────────────────────────────────────────────
LIGO_CLASSES       = ["Noise", "Signal"]
LIGO_NUM_CLASSES   = 2
LIGO_N_DETECTORS   = 3
LIGO_SIGNAL_LEN    = 4096
LIGO_SAMPLE_RATE   = 2048.0      # Hz

LIGO_CQT_BINS      = 64          # CQT frequency bins
LIGO_CQT_STEPS     = 64          # CQT time steps
LIGO_IMG_SIZE      = 224         # EfficientNet input

LIGO_EPOCHS        = 50
LIGO_BATCH_SIZE    = 32
LIGO_LR            = 1e-4
LIGO_LR_BACKBONE   = 1e-5
LIGO_LR_MIN        = 1e-6
LIGO_WEIGHT_DECAY  = 1e-3
LIGO_DROPOUT       = 0.4
LIGO_FREEZE_EPOCHS = 10
LIGO_WARMUP_EPOCHS = 3
LIGO_TEST_SPLIT    = 0.15
LIGO_VAL_SPLIT     = 0.15
LIGO_ENCODER_DIM   = ENCODER_DIM
LIGO_MAX_SAMPLES   = 4000        # balanced per-class cap for cache build

# ─────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────
PATIENCE  = 15
MIN_DELTA = 1e-4