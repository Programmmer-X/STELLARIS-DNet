"""
module1/config.py
STELLARIS-DNet — Module 1 Hyperparameters & Feature Toggles
All tunable values in one place. PURE CONSTANTS — no imports, no logic.
Upgraded: feature flags, signal processing, anomaly methods, eval toggles.
"""

# ─────────────────────────────────────────────
# GENERAL
# ─────────────────────────────────────────────
SEED           = 42
DEVICE         = "auto"               # "auto" | "cpu" | "cuda"
CHECKPOINT_DIR = "checkpoints/module1"
LOG_DIR        = "logs/module1"
EXPERIMENT_TAG = "enhanced_v2"        # used in filenames for experiment tracking

# ─────────────────────────────────────────────
# DATASET PATHS
# ─────────────────────────────────────────────
HTRU2_PATH  = "data/module1/HTRU_2.csv"
LOTAAS_PATH = "data/module1/lotaas_profiles.npy"

# ─────────────────────────────────────────────
# SIGNAL CONFIG
# ─────────────────────────────────────────────
SIGNAL_LENGTH = 64          # fixed length for all 1D pulse profiles
TEST_SPLIT    = 0.15        # 15% held-out test
VAL_SPLIT     = 0.15        # 15% validation
NORMALIZE     = True        # min-max normalize profiles to [0, 1]

# ─────────────────────────────────────────────
# CLASS DEFINITIONS
# ─────────────────────────────────────────────
HTRU2_CLASSES      = ["Non-Pulsar", "Pulsar"]
PULSAR_CLASSES     = ["Normal", "Millisecond", "Binary", "Recycled"]
NUM_PULSAR_CLASSES = 4

# ─────────────────────────────────────────────
# ═══════════════════════════════════════════
# FEATURE TOGGLES  ← new
# All downstream code reads these flags.
# Toggle independently; no other file changes.
# ═══════════════════════════════════════════
# ─────────────────────────────────────────────

# Signal Processing Toggles
USE_FFT         = True    # extract FFT magnitude features from profiles
USE_CQT         = False   # extract CQT spectrogram (64×64); heavier, optional
USE_AUGMENTATION = True   # apply shift/scale/noise augmentation during training

# Model Feature Toggles
USE_ATTENTION          = True   # self-attention in CNN encoder
USE_FREQ_FUSION        = True   # fuse time-domain + frequency-domain (CNN)
USE_PHYSICS_FEATURES   = True   # append energy proxy as extra feature
USE_HYBRID_FUSION      = False  # fuse MLP + CNN features (experimental)

# Loss Toggles
USE_PHYSICS_LOSS   = True    # add spindown_energy_loss term to MLP training
USE_FOCAL_LOSS     = True    # focal loss for MLP (fixes class imbalance F1 gap)

# Unified Model
USE_UNIFIED_MODE   = False   # expose encoder-only outputs; set True for unified

# Experiment Tracking
RUN_BASELINE  = True    # also train/eval without any enhancements
RUN_ENHANCED  = True    # train/eval with all active toggles
SAVE_CURVES   = True    # save loss/metric curves per epoch as PNG
LOG_TIME      = True    # log seconds per epoch

# ─────────────────────────────────────────────
# SIGNAL PROCESSING PARAMETERS  ← new
# ─────────────────────────────────────────────
FFT_BINS        = 32          # number of FFT bins kept (half of SIGNAL_LENGTH)
CQT_BINS        = 64          # frequency bins for CQT spectrogram
CQT_STEPS       = 64          # time steps for CQT spectrogram
SAMPLE_RATE     = 1.0         # normalized sample rate for profile data

# ─────────────────────────────────────────────
# NOISE & RFI SIMULATION  ← new
# ─────────────────────────────────────────────
NOISE_STD       = 0.02        # Gaussian noise std added during augmentation
RFI_PROB        = 0.15        # probability of injecting RFI glitch per sample
RFI_WIDTH_MIN   = 2           # min width (samples) of RFI spike
RFI_WIDTH_MAX   = 8           # max width (samples) of RFI spike
RFI_AMP_SCALE   = 3.0         # RFI amplitude as multiple of profile max
SHIFT_MAX       = 8           # max sample shift for augmentation
SCALE_MIN       = 0.85        # min amplitude scale factor
SCALE_MAX       = 1.15        # max amplitude scale factor

# ─────────────────────────────────────────────
# MLP CONFIG (HTRU2 binary classifier)
# ─────────────────────────────────────────────
MLP_INPUT_DIM    = 8          # 8 HTRU2 statistical features (base)
MLP_HIDDEN_DIMS  = [64, 32]   # encoder hidden layers
MLP_DROPOUT      = 0.3
MLP_LR           = 1e-3
MLP_EPOCHS       = 50
MLP_BATCH_SIZE   = 128
MLP_WEIGHT_DECAY = 1e-4

# Focal Loss (replaces BCEWithLogitsLoss when USE_FOCAL_LOSS=True)
FOCAL_ALPHA = 0.25            # weighting factor for rare class
FOCAL_GAMMA = 2.0             # focusing parameter

# Physics loss weighting (wired into MLP when USE_PHYSICS_LOSS=True)
SPINDOWN_LOSS_WEIGHT = 0.05   # kept low — MLP features are statistical, not raw
PERIOD_MIN = 0.001            # 1ms (millisecond pulsars)
PERIOD_MAX = 10.0             # 10s (slowest known pulsars)

# ─────────────────────────────────────────────
# 1D CNN CONFIG (Pulsar subtype classifier)
# ─────────────────────────────────────────────
CNN_IN_CHANNELS  = 1
CNN_CHANNELS     = [64, 128, 256]
CNN_KERNEL_SIZES = [7, 5, 3]
CNN_DROPOUT      = 0.3
CNN_LR           = 1e-3
CNN_EPOCHS       = 60
CNN_BATCH_SIZE   = 64
CNN_WEIGHT_DECAY = 1e-4
CNN_ENCODER_DIM  = 256        # encoder output dim → unified model input

# Attention (active when USE_ATTENTION=True)
CNN_ATTN_HEADS   = 4          # number of attention heads
CNN_ATTN_DROPOUT = 0.1

# Freq fusion (active when USE_FREQ_FUSION=True)
# FFT features are projected to match CNN encoder before concat
CNN_FREQ_DIM     = 64         # FFT feature projection dim
CNN_FUSED_DIM    = 256        # post-fusion projection → same as CNN_ENCODER_DIM

# ─────────────────────────────────────────────
# AUTOENCODER CONFIG (Magnetar anomaly detector)
# ─────────────────────────────────────────────
AE_INPUT_DIM   = 64           # matches SIGNAL_LENGTH
AE_LATENT_DIM  = 16           # bottleneck (latent representation)
AE_HIDDEN_DIMS = [48, 32]
AE_DROPOUT     = 0.2
AE_LR          = 1e-3
AE_EPOCHS      = 50
AE_BATCH_SIZE  = 64

# Anomaly Threshold Methods  ← new
# "percentile" — 95th pct of val errors (original)
# "zscore"     — mean + N*std of val errors
AE_THRESHOLD_METHOD   = "percentile"   # "percentile" | "zscore"
AE_ANOMALY_PERCENTILE = 95             # used when method = percentile
AE_ZSCORE_SIGMA       = 3.0            # used when method = zscore

# ─────────────────────────────────────────────
# EVALUATION TOGGLES  ← new
# ─────────────────────────────────────────────
EVAL_ROC_CURVE      = True    # generate and save ROC curve (MLP)
EVAL_PR_CURVE       = True    # precision-recall curve (MLP, imbalanced data)
EVAL_ANOMALY_VIZ    = True    # visualize normal vs magnetar AE reconstructions
EVAL_COMPARISON     = True    # baseline vs enhanced side-by-side report
EVAL_ERROR_ANALYSIS = True    # misclassification analysis plots
EVAL_CONFIDENCE     = True    # confidence distribution plots (correct vs wrong)
EVAL_LATENT_VIZ     = True    # PCA of AE latent space
EVAL_SPEED_BENCH    = True    # inference latency + throughput benchmark
EVAL_CLASSWISE      = True    # per-class breakdown in confusion matrix

# ─────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────
PATIENCE  = 10
MIN_DELTA = 1e-4

# ─────────────────────────────────────────────
# CLASS IMBALANCE (HTRU2 ~9:1)
# ─────────────────────────────────────────────
USE_CLASS_WEIGHTS = True      # pos_weight in BCEWithLogitsLoss
USE_OVERSAMPLE    = False      # SMOTE — slower, try if focal loss insufficient