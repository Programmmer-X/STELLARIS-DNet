"""
module3/config.py
STELLARIS-DNet — Module 3 Hyperparameters
Stellar object classification + physical parameter regression.
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
# DATASET PATHS — Local
# ─────────────────────────────────────────────
GAIA_PATH    = "data/module3/gaia_dr3.csv"
SDSS_PATH    = "data/module3/sdss_stars.csv"
ATNF_PATH    = "data/module3/atnf_catalog.csv"
MWDD_PATH    = "data/module3/montreal_wd.csv"

# ─────────────────────────────────────────────
# DATASET PATHS — Kaggle
# ─────────────────────────────────────────────
KAGGLE_GAIA_PATH = "/kaggle/input/stellaris-module3-data/gaia_dr3.csv"
KAGGLE_SDSS_PATH = "/kaggle/input/stellaris-module3-data/sdss_stars.csv"
KAGGLE_ATNF_PATH = "/kaggle/input/stellaris-module3-data/atnf_catalog.csv"
KAGGLE_MWDD_PATH = "/kaggle/input/stellaris-module3-data/montreal_wd.csv"

# ─────────────────────────────────────────────
# CLASS DEFINITIONS
# ─────────────────────────────────────────────
STELLAR_CLASSES = [
    "Main_Sequence",    # 0 — H-burning dwarfs
    "Red_Giant",        # 1 — evolved, expanded
    "White_Dwarf",      # 2 — degenerate remnant, M < 1.44 M_sun
    "Neutron_Star",     # 3 — ultra-compact, post-supernova
    "Quasar",           # 4 — AGN, supermassive BH accretion
]
NUM_STELLAR_CLASSES = 5

# ─────────────────────────────────────────────
# DOMAIN IDs  (assigned per data source)
# ─────────────────────────────────────────────
DOMAIN_GAIA      = 0
DOMAIN_SDSS      = 1
DOMAIN_ATNF      = 2
DOMAIN_MWDD      = 3
DOMAIN_SYNTHETIC = 4

# ─────────────────────────────────────────────
# FEATURES  — 8 total (7 physical + domain_id)
# ─────────────────────────────────────────────
FEATURE_NAMES = [
    "teff",        # Effective temperature (K)
    "log_g",       # Surface gravity log10(g / cm s^-2)
    "feh",         # Metallicity [Fe/H]
    "abs_mag",     # Absolute magnitude (G-band / V-band)
    "bp_rp",       # Colour index (Gaia BP-RP or proxy)
    "redshift",    # Cosmological redshift (0 for galactic objects)
    "period_ms",   # Spin period in milliseconds (0 for non-pulsars)
    "domain_id",   # Source domain: 0=Gaia,1=SDSS,2=ATNF,3=MWDD,4=Synthetic
]
NUM_FEATURES = len(FEATURE_NAMES)    # 8

# ─────────────────────────────────────────────
# REGRESSION TARGETS — 4 total, all log10-scale
# ─────────────────────────────────────────────
REGRESSION_TARGETS = [
    "log_mass",    # log10(M / M_sun)
    "log_lum",     # log10(L / L_sun)
    "log_teff",    # log10(Teff / K)
    "log_radius",  # log10(R / R_sun)
]
NUM_REGRESSION = len(REGRESSION_TARGETS)    # 4

# ─────────────────────────────────────────────
# REGRESSION OUTPUT BOUNDS  (sigmoid-scaled)
# MIN + (MAX - MIN) * sigmoid(raw) — gradients
# flow at boundaries unlike clamp()
# ─────────────────────────────────────────────
LOG_MASS_MIN   = -1.1;  LOG_MASS_MAX   = 10.0   # 0.08 M_sun → BH/QSO
LOG_LUM_MIN    = -4.0;  LOG_LUM_MAX    = 14.0   # dim WD → bright QSO
LOG_TEFF_MIN   =  3.2;  LOG_TEFF_MAX   =  7.5   # 1500 K → 30M K (NS)
LOG_RADIUS_MIN = -4.9;  LOG_RADIUS_MAX =  6.0   # 10 km NS → QSO scale

# ─────────────────────────────────────────────
# HR DIAGRAM BOUNDARIES (label assignment)
# ─────────────────────────────────────────────
LOG_G_MS_BOUNDARY  = 3.5    # log_g > 3.5 → Main Sequence
LOG_G_WD_BOUNDARY  = 7.0    # log_g > 7.0 → White Dwarf (from Gaia PWD)
TEFF_RG_MAX        = 5500.0 # K — Red Giants cooler than this
WD_PROB_THRESHOLD  = 0.9    # Gaia PWD column threshold for WD label

# ─────────────────────────────────────────────
# FT-TRANSFORMER ARCHITECTURE
# ─────────────────────────────────────────────
TRANSFORMER_DIM      = 128   # d_token — embedding dimension per feature
TRANSFORMER_HEADS    = 8     # multi-head attention heads
TRANSFORMER_LAYERS   = 4     # transformer encoder depth
TRANSFORMER_FFN_MULT = 4     # FFN hidden = DIM * FFN_MULT = 512
TRANSFORMER_DROPOUT  = 0.1

# ─────────────────────────────────────────────
# HEAD ARCHITECTURE
# ─────────────────────────────────────────────
HEAD_HIDDEN_DIMS = [256, 128]
HEAD_DROPOUT     = 0.3
ENCODER_DIM      = 256       # CLS projection → unified model compatible

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
EPOCHS            = 100
ACTUAL_BATCH_SIZE = 128      # per forward pass
ACCUMULATE_STEPS  = 4        # effective batch = 128 × 4 = 512
WARMUP_EPOCHS     = 10       # linear LR warmup before cosine decay
LR                = 1e-4
WEIGHT_DECAY      = 1e-4
GRAD_CLIP         = 1.0
USE_AMP           = False    # set True on Kaggle for faster training

# ─────────────────────────────────────────────
# LOSS WEIGHTS
# ─────────────────────────────────────────────
CLASS_LOSS_WEIGHT     = 1.0
REG_LOSS_WEIGHT       = 0.5
PHYSICS_LOSS_WEIGHT   = 0.1
SYNTHETIC_LOSS_WEIGHT = 0.7  # down-weight synthetic samples in loss

# Physics loss: Stefan-Boltzmann applied to classes 0,1,2 ONLY
# Not Quasar (accretion disk) and not Neutron Star (different emission)
SB_LOSS_CLASSES = [0, 1, 2]

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
SYNTHETIC_N   = 10000    # per class, only when real data unavailable
MAX_PER_CLASS = 50000    # cap per class — prevents Gaia MS domination

# ─────────────────────────────────────────────
# DATA CLEANING BOUNDS
# ─────────────────────────────────────────────
TEFF_MIN     = 1500.0    # K
TEFF_MAX     = 6.0e5     # K
LOGG_MIN     = -1.0
LOGG_MAX     = 16.0
FEH_MIN      = -5.0
FEH_MAX      =  1.0
MASS_WD_MIN  =  0.17     # M_sun — lowest confirmed WD
MASS_WD_MAX  =  1.43     # M_sun — just below Chandrasekhar

# ─────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────
CHANDRASEKHAR_LIMIT = 1.44   # M_sun
NS_MASS_MIN         = 1.1    # M_sun
NS_MASS_MAX         = 2.5    # M_sun
TEFF_SUN            = 5778.0 # K
LOG_TEFF_SUN        = 3.7617 # log10(5778)
L_SUN               = 3.828e26   # W
R_SUN               = 6.957e8    # m
SIGMA_SB            = 5.6704e-8  # W m^-2 K^-4