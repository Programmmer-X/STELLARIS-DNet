"""
module3/config.py
STELLARIS-DNet — Module 3 Hyperparameters (v2 upgraded)
PURE CONSTANTS ONLY — no imports, no logic.
"""

# ─────────────────────────────────────────────
# GENERAL
# ─────────────────────────────────────────────
SEED           = 42
DEVICE         = "auto"
CHECKPOINT_DIR = "checkpoints/module3"
LOG_DIR        = "logs/module3"

# ─────────────────────────────────────────────
# DATASET PATHS
# ─────────────────────────────────────────────
GAIA_PATH = "data/module3/gaia_dr3.csv"
SDSS_PATH = "data/module3/sdss_stars.csv"
ATNF_PATH = "data/module3/atnf_catalog.csv"
MWDD_PATH = "data/module3/montreal_wd.csv"

KAGGLE_GAIA_PATH = "/kaggle/input/stellaris-module3-data/gaia_dr3.csv"
KAGGLE_SDSS_PATH = "/kaggle/input/stellaris-module3-data/sdss_stars.csv"
KAGGLE_ATNF_PATH = "/kaggle/input/stellaris-module3-data/atnf_catalog.csv"
KAGGLE_MWDD_PATH = "/kaggle/input/stellaris-module3-data/montreal_wd.csv"

# ─────────────────────────────────────────────
# CLASS DEFINITIONS
# ─────────────────────────────────────────────
STELLAR_CLASSES = [
    "Main_Sequence",   # 0
    "Red_Giant",       # 1
    "White_Dwarf",     # 2
    "Neutron_Star",    # 3
    "Quasar",          # 4
]
NUM_STELLAR_CLASSES = 5

# ─────────────────────────────────────────────
# FEATURES — v3: 7 physical features only
# Validity flags removed (caused domain shortcut)
# Missing features filled with class-conditional + noise
# ─────────────────────────────────────────────
PHYSICAL_FEATURES = [
    "teff", "log_g", "feh", "abs_mag", "bp_rp", "redshift", "period_ms",
]
FEATURE_NAMES = PHYSICAL_FEATURES                # 7 only
NUM_FEATURES  = len(FEATURE_NAMES)               # 7
NUM_PHYSICAL  = NUM_FEATURES

# ─────────────────────────────────────────────
# NOISE INJECTION (GPU augmentation in train.py)
# Applied only during training — not val/test
# Forces model to learn distributions, not exact values
# Breaks "constant fill = domain identifier" shortcut
# ─────────────────────────────────────────────
NOISE_TEFF_FRAC     = 0.05    # relative noise on teff (5% of value)
NOISE_LOGG_STD      = 0.10    # absolute std on log_g
NOISE_FEH_STD       = 0.10    # absolute std on [Fe/H]
NOISE_ABSMAG_STD    = 0.20    # absolute std on abs_mag
NOISE_BPRP_STD      = 0.05    # absolute std on bp_rp
NOISE_REDSHIFT_STD  = 0.02    # absolute std on redshift
NOISE_PERIODMS_FRAC = 0.05    # relative noise on period_ms
# ─────────────────────────────────────────────
# REGRESSION TARGETS — 4 log10-scale
# Per-class supervision via reg_mask (computed in dataset.py)
# ─────────────────────────────────────────────
REGRESSION_TARGETS = ["log_mass", "log_lum", "log_teff", "log_radius"]
NUM_REGRESSION     = len(REGRESSION_TARGETS)    # 4

# Per-class regression supervision policy
# Index: [log_mass, log_lum, log_teff, log_radius]
# 1 = supervise, 0 = mask out (no loss)
REG_SUPERVISION_BY_CLASS = {
    0: [1, 1, 1, 1],   # MS:  all targets supervised
    1: [1, 1, 1, 1],   # RG:  all targets supervised
    2: [1, 1, 1, 1],   # WD:  all targets supervised
    3: [1, 0, 0, 1],   # NS:  only log_mass + log_radius (Teff/Lum from cooling = synthetic)
    4: [0, 1, 0, 0],   # QSO: only log_lum (from bolometric correction)
}

# ─────────────────────────────────────────────
# REGRESSION OUTPUT BOUNDS (sigmoid-scaled)
# ─────────────────────────────────────────────
LOG_MASS_MIN   = -1.1;  LOG_MASS_MAX   = 10.0
LOG_LUM_MIN    = -4.0;  LOG_LUM_MAX    = 14.0
LOG_TEFF_MIN   =  3.2;  LOG_TEFF_MAX   =  7.5
LOG_RADIUS_MIN = -4.9;  LOG_RADIUS_MAX =  6.0

# ─────────────────────────────────────────────
# HR DIAGRAM LABELLING (Gaia, joint criteria)
# Drops subgiant branch (ambiguous) → cleaner labels
# ─────────────────────────────────────────────
LOG_G_MS_MIN     = 4.0     # log_g > 4.0 + cool enough → MS
TEFF_MS_MIN      = 3500.0
TEFF_MS_MAX      = 50000.0
LOG_LUM_MS_MAX   = 2.0     # MS doesn't exceed log_L = 2

LOG_G_RG_MAX     = 3.5     # log_g < 3.5 → RG
TEFF_RG_MAX      = 5500.0
LOG_LUM_RG_MIN   = 1.0     # RG must be luminous

WD_PROB_THRESHOLD = 0.9    # Gaia PWD column

# ─────────────────────────────────────────────
# FT-TRANSFORMER ARCHITECTURE
# ─────────────────────────────────────────────
TRANSFORMER_DIM      = 128
TRANSFORMER_HEADS    = 8
TRANSFORMER_LAYERS   = 4
TRANSFORMER_FFN_MULT = 4
TRANSFORMER_DROPOUT  = 0.1

HEAD_HIDDEN_DIMS = [256, 128]
HEAD_DROPOUT     = 0.3
ENCODER_DIM      = 256

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
EPOCHS            = 100
ACTUAL_BATCH_SIZE = 128
ACCUMULATE_STEPS  = 4
WARMUP_EPOCHS     = 10
LR                = 1e-4
WEIGHT_DECAY      = 1e-4
GRAD_CLIP         = 1.0
USE_AMP           = False

# ─────────────────────────────────────────────
# LOSS WEIGHTS
# ─────────────────────────────────────────────
CLASS_LOSS_WEIGHT   = 1.0
REG_LOSS_WEIGHT     = 0.5
PHYSICS_LOSS_WEIGHT = 0.1

# ─────────────────────────────────────────────
# CURRICULUM PHYSICS MASKING (v2 upgrade)
# Epochs 1 to CURRICULUM_HARD_END  → use true class labels (hard mask)
# Epochs CURRICULUM_HARD_END to CURRICULUM_SOFT_START → linear blend
# Epochs > CURRICULUM_SOFT_START → pure soft probs
# ─────────────────────────────────────────────
CURRICULUM_HARD_END   = 10   # last epoch using pure true labels
CURRICULUM_SOFT_START = 30   # first epoch using pure soft probs

# ─────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────
PATIENCE  = 15
MIN_DELTA = 1e-4

# ─────────────────────────────────────────────
# DATA SPLITS + SAMPLING
# ─────────────────────────────────────────────
TEST_SPLIT    = 0.15
VAL_SPLIT     = 0.15
MAX_PER_CLASS = 50000

# Synthetic injection — only when class has zero real samples
# Capped at 10% of mean real class size
SYNTHETIC_FALLBACK_FRACTION = 0.10
SYNTHETIC_MIN_PER_CLASS     = 500    # if injecting, at least this many

# ─────────────────────────────────────────────
# DATA CLEANING BOUNDS
# ─────────────────────────────────────────────
TEFF_MIN     = 1500.0
TEFF_MAX     = 6.0e5
LOGG_MIN     = -1.0
LOGG_MAX     = 16.0
FEH_MIN      = -5.0
FEH_MAX      =  1.0
MASS_WD_MIN  =  0.17
MASS_WD_MAX  =  1.43

# ─────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────
CHANDRASEKHAR_LIMIT = 1.44
NS_MASS_MIN         = 1.1
NS_MASS_MAX         = 2.5
TEFF_SUN            = 5778.0
LOG_TEFF_SUN        = 3.7617
L_SUN               = 3.828e26
R_SUN               = 6.957e8
SIGMA_SB            = 5.6704e-8

# ─────────────────────────────────────────────
# QSO BOLOMETRIC CORRECTION (Richards et al. 2006)
# ─────────────────────────────────────────────
QSO_M_I_SUN     = 4.54     # solar absolute i-band magnitude
QSO_BOL_CORR    = 0.95     # bolometric correction in log space

# v3.1: feature dropout for sparse features (redshift, period_ms)
SPARSE_DROPOUT_PROB    = 0.3   # probability of zeroing per batch
SPARSE_FEATURE_INDICES = [5, 6]   # redshift, period_ms in standardised X

