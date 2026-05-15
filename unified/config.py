"""
unified/config.py
STELLARIS-DNet — Unified Fusion Configuration

All hyperparameters, paths, and training stage configs for the
multi-modal fusion model. References module-level configs for
preprocessing constants — does NOT duplicate them.
"""

import os

# ═════════════════════════════════════════════════════════════
# 1. PATHS
# ═════════════════════════════════════════════════════════════

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", "unified")
LOG_DIR        = os.path.join(BASE_DIR, "logs", "unified")

# ── Module encoder checkpoints ───────────────────────────────────────
# CNN/RGC/GWD: full model state_dict (instantiate model, load, call .encode())
# MLP/AE:      encoder-only state_dict (load into model.encoder)
ENCODER_PATHS = {
    "m1_mlp": os.path.join(BASE_DIR, "checkpoints", "module1", "mlp_encoder.pt"),
    "m1_cnn": os.path.join(BASE_DIR, "checkpoints", "module1", "cnn_encoder.pt"),
    "m1_ae":  os.path.join(BASE_DIR, "checkpoints", "module1", "ae_encoder.pt"),
    "m2_rgc": os.path.join(BASE_DIR, "checkpoints", "module2", "rgc_best.pt"),
    "m2_gwd": os.path.join(BASE_DIR, "checkpoints", "module2", "gwd_best.pt"),
    "m3":     os.path.join(BASE_DIR, "checkpoints", "module3", "module3_best.pt"),
}

# ── Scaler artifacts (MANDATORY for inference) ───────────────────────
SCALER_PATHS = {
    "m1_mlp": os.path.join(BASE_DIR, "checkpoints", "module1", "mlp_scaler.pkl"),
    "m3":     os.path.join(BASE_DIR, "checkpoints", "module3", "module3_scaler.pkl"),
}

# ── AE anomaly threshold ─────────────────────────────────────────────
AE_THRESHOLD_PATH = os.path.join(BASE_DIR, "checkpoints", "module1", "ae_threshold.npy")


# ═════════════════════════════════════════════════════════════
# 2. ENCODER DIMENSIONS (locked post-training)
# ═════════════════════════════════════════════════════════════

ENCODER_DIMS = {
    "m1_mlp": 32,
    "m1_cnn": 256,
    "m1_ae":  16,
    "m2_rgc": 256,
    "m2_gwd": 256,
    "m3":     256,
}

# Common projection target for all encoders
PROJ_DIM = 256

# Number of encoder streams
NUM_ENCODERS = len(ENCODER_DIMS)  # 6


# ═════════════════════════════════════════════════════════════
# 3. FUSION ARCHITECTURE
# ═════════════════════════════════════════════════════════════

# Fusion MLP: concat(6 × PROJ_DIM + NUM_ENCODERS mask) → FUSED_DIM
FUSION_INPUT_DIM  = NUM_ENCODERS * PROJ_DIM + NUM_ENCODERS  # 1542
FUSION_HIDDEN_DIM = 768
FUSED_DIM         = 512

FUSION_DROPOUT_1  = 0.3     # after first FC
FUSION_DROPOUT_2  = 0.2     # after second FC

# Projection layer regularization (higher for expansion projections)
PROJ_DROPOUT_EXPAND = 0.4   # for M1-MLP (32→256) and M1-AE (16→256)
PROJ_DROPOUT_PASS   = 0.0   # for 256-dim encoders (identity + LayerNorm)


# ═════════════════════════════════════════════════════════════
# 4. OUTPUT HEADS
# ═════════════════════════════════════════════════════════════

# Head 1: Stellar Classification (M3 taxonomy)
NUM_STELLAR_CLASSES  = 5
STELLAR_CLASS_NAMES  = ["Main_Sequence", "Red_Giant", "White_Dwarf",
                        "Neutron_Star", "Quasar"]

# Head 2: Pulsar Detection (binary)
# — no extra config needed, sigmoid output

# Head 3: Pulsar Subtype
NUM_PULSAR_SUBTYPES  = 4
PULSAR_SUBTYPE_NAMES = ["Normal", "Millisecond", "Binary", "Recycled"]

# Head 4: Radio Morphology
NUM_RADIO_CLASSES    = 2
RADIO_CLASS_NAMES    = ["FRI", "FRII"]

# Head 5: GW Detection (binary)
# — no extra config needed, sigmoid output

# Head 6: Pulse Profile Anomaly Score (scalar sigmoid)
# — no extra config needed

# Head 7: Physical Parameter Regression
NUM_REG_TARGETS      = 4
REG_TARGET_NAMES     = ["log_mass", "log_lum", "log_teff", "log_radius"]
REG_BOUNDS = {
    "log_mass":   (-1.1, 10.0),
    "log_lum":    (-4.0, 14.0),
    "log_teff":   (3.2,  7.5),
    "log_radius": (-4.9, 6.0),
}

# Head 8: Physics Consistency (computed, not learned)
# — uses core/physics_loss.py; no trainable params

# ── Head hidden dims ─────────────────────────────────────────────────
HEAD_HIDDEN_DIM = 256


# ═════════════════════════════════════════════════════════════
# 5. TRAINING — GENERAL
# ═════════════════════════════════════════════════════════════

SEED             = 42
BATCH_SIZE       = 256
NUM_WORKERS      = 2
PIN_MEMORY       = True

# Data split
VAL_SPLIT        = 0.15
TEST_SPLIT       = 0.15

# Gradient clipping
MAX_GRAD_NORM    = 1.0

# Early stopping
PATIENCE         = 12
MIN_DELTA        = 1e-4

# Experiment tag
EXPERIMENT_TAG   = "v1"


# ═════════════════════════════════════════════════════════════
# 6. TRAINING — STAGE CONFIGS
# ═════════════════════════════════════════════════════════════

# Stage 1: Frozen encoders, fusion head only
STAGE1 = {
    "epochs":      20,
    "lr_fusion":   5e-4,
    "lr_proj":     5e-4,
    "weight_decay": 1e-3,
    "warmup_epochs": 3,
    "unfreeze":    [],                  # all encoders frozen
    "label":       "stage1_frozen",
}

# Stage 2: Unfreeze M1-MLP + M1-AE (smallest encoders)
STAGE2 = {
    "epochs":      15,
    "lr_fusion":   1e-4,
    "lr_proj":     1e-4,
    "lr_encoder":  1e-5,
    "weight_decay": 1e-3,
    "warmup_epochs": 0,
    "unfreeze":    ["m1_mlp", "m1_ae"], # small encoders only
    "label":       "stage2_partial",
}

# Stage 3: Conditional, data-driven
STAGE3 = {
    "epochs":      10,
    "lr_fusion":   5e-5,
    "lr_proj":     5e-5,
    "lr_encoder":  5e-6,
    "weight_decay": 1e-3,
    "warmup_epochs": 0,
    "unfreeze":    ["m1_mlp", "m1_ae", "m1_cnn"],  # extend as data improves
    "label":       "stage3_finetune",
}

STAGES = [STAGE1, STAGE2, STAGE3]


# ═════════════════════════════════════════════════════════════
# 7. LOSS WEIGHTS
# ═════════════════════════════════════════════════════════════

# Per-head loss weights (λ values)
# Tuned to compensate for dataset size imbalance:
#   M3: ~200K samples, M2-GWD: ~50K, M1: ~2K, M2-RGC: ~800
LOSS_WEIGHTS = {
    "stellar_cls":     1.0,     # base weight (largest dataset)
    "pulsar_det":      5.0,     # 100× fewer samples than M3
    "pulsar_subtype":  5.0,     # synthetic data, ~2K samples
    "radio_morphology":10.0,    # 792 samples — highest per-sample weight
    "gw_det":          2.0,     # 50K samples
    "anomaly":         3.0,     # evaluated on synthetic data
    "regression":      1.0,     # same samples as stellar_cls
    "physics":         0.0,     # starts at 0, curriculum warmup
}

# Physics loss curriculum: off → ramp → target
PHYSICS_LOSS_START_EPOCH  = 5    # disabled for first N epochs
PHYSICS_LOSS_RAMP_EPOCHS  = 10   # linear ramp from 0 → target
PHYSICS_LOSS_TARGET_WEIGHT = 0.1  # final λ_physics


# ═════════════════════════════════════════════════════════════
# 8. MODALITY MASK INDICES
# ═════════════════════════════════════════════════════════════

# Fixed order — must match dataset and model
MODALITY_ORDER = ["m1_mlp", "m1_cnn", "m1_ae", "m2_rgc", "m2_gwd", "m3"]

MODALITY_INDEX = {name: i for i, name in enumerate(MODALITY_ORDER)}

# Head validity: which modalities must be present for each head
HEAD_VALIDITY = {
    "stellar_cls":     ["m3"],
    "pulsar_det":      ["m1_mlp"],
    "pulsar_subtype":  ["m1_cnn"],
    "radio_morphology":["m2_rgc"],
    "gw_det":          ["m2_gwd"],
    "anomaly":         ["m1_ae"],
    "regression":      ["m3"],
    "physics":         ["m3"],
}


# ═════════════════════════════════════════════════════════════
# 9. REGRESSION SUPERVISION POLICY (inherited from M3)
# ═════════════════════════════════════════════════════════════

# Which regression targets are supervised for each stellar class
# Same as Module 3's REG_SUPERVISION_BY_CLASS
REG_SUPERVISION = {
    "Main_Sequence": [True, True, True, True],     # all 4
    "Red_Giant":     [True, True, True, True],     # all 4
    "White_Dwarf":   [True, True, True, True],     # all 4
    "Neutron_Star":  [True, False, False, True],   # mass + radius only
    "Quasar":        [False, True, False, False],  # luminosity only
}


# ═════════════════════════════════════════════════════════════
# 10. KAGGLE COMPATIBILITY
# ═════════════════════════════════════════════════════════════

def is_kaggle():
    return os.path.exists("/kaggle/working")

if is_kaggle():
    BASE_DIR       = "/kaggle/working/STELLARIS-DNet"
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", "unified")
    LOG_DIR        = os.path.join(BASE_DIR, "logs", "unified")
    # Re-resolve all paths
    for key in ENCODER_PATHS:
        ENCODER_PATHS[key] = os.path.join(
            BASE_DIR, "checkpoints",
            ENCODER_PATHS[key].split("checkpoints/")[-1]
        )
    for key in SCALER_PATHS:
        SCALER_PATHS[key] = os.path.join(
            BASE_DIR, "checkpoints",
            SCALER_PATHS[key].split("checkpoints/")[-1]
        )
    AE_THRESHOLD_PATH = os.path.join(
        BASE_DIR, "checkpoints", "module1", "ae_threshold.npy"
    )