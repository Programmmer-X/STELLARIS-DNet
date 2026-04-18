"""
core/physics_loss.py
STELLARIS-DNet — Physics-Informed Loss Functions
All physical constraints used across Module 1, 2, 3 and Unified.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
SIGMA_SB      = 5.6704e-8   # Stefan-Boltzmann constant (W/m²/K⁴)
CHANDRASEKHAR = 1.44         # Chandrasekhar limit (solar masses)
L_SUN         = 3.828e26     # Solar luminosity (W)
R_SUN         = 6.957e8      # Solar radius (m)


# ─────────────────────────────────────────────
# 1. STEFAN-BOLTZMANN LAW
# L = 4π R² σ T⁴
# Used in: Module 3 (all stellar objects)
# ─────────────────────────────────────────────
def stefan_boltzmann_loss(
    L_pred: torch.Tensor,   # predicted luminosity (L_sun units)
    R_pred: torch.Tensor,   # predicted radius     (R_sun units)
    T_pred: torch.Tensor    # predicted temperature (Kelvin)
) -> torch.Tensor:
    """
    Penalizes predictions that violate L = 4πR²σT⁴.
    All inputs in solar units / Kelvin.
    """
    # Convert to SI
    R_si = R_pred * R_SUN
    L_si = L_pred * L_SUN

    L_expected_si = 4 * torch.pi * R_si**2 * SIGMA_SB * T_pred**4
    L_expected     = L_expected_si / L_SUN   # back to solar units

    return F.mse_loss(L_si, L_expected_si) / (L_SUN**2)  # normalized


# ─────────────────────────────────────────────
# 2. MASS-LUMINOSITY RELATION
# L ∝ M^3.5  (valid for main sequence stars)
# Used in: Module 3 (Main Sequence class only)
# ─────────────────────────────────────────────
def mass_luminosity_loss(
    L_pred: torch.Tensor,   # predicted luminosity (L_sun units)
    M_pred: torch.Tensor,   # predicted mass       (M_sun units)
    class_pred: torch.Tensor,  # predicted class indices
    main_seq_idx: int = 0   # index of Main Sequence class
) -> torch.Tensor:
    """
    Enforces L = M^3.5 only for main sequence star predictions.
    Ignores all other classes.
    """
    mask = (class_pred == main_seq_idx)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=L_pred.device)

    L_ms = L_pred[mask]
    M_ms = M_pred[mask]

    # Clamp to avoid exploding gradients on extreme mass values
    M_clamped    = torch.clamp(M_ms, min=0.1, max=100.0)
    L_expected   = M_clamped ** 3.5

    return F.mse_loss(L_ms, L_expected)


# ─────────────────────────────────────────────
# 3. CHANDRASEKHAR LIMIT
# White dwarf mass must be < 1.44 M_sun
# Used in: Module 3 (White Dwarf class only)
# ─────────────────────────────────────────────
def chandrasekhar_loss(
    M_pred: torch.Tensor,      # predicted mass (M_sun units)
    class_pred: torch.Tensor,  # predicted class indices
    wd_idx: int = 2            # index of White Dwarf class
) -> torch.Tensor:
    """
    Penalizes any White Dwarf prediction with mass >= 1.44 M_sun.
    Uses ReLU to penalize only violations, not compliant predictions.
    """
    mask = (class_pred == wd_idx)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=M_pred.device)

    wd_masses  = M_pred[mask]
    violation  = F.relu(wd_masses - CHANDRASEKHAR)  # 0 if ok, >0 if violating

    return violation.mean()


# ─────────────────────────────────────────────
# 4. SPIN-DOWN ENERGY LOSS (Pulsar)
# E_dot = -4π²I * P_dot / P³
# Used in: Module 1
# ─────────────────────────────────────────────
def spindown_energy_loss(
    E_pred: torch.Tensor,    # predicted spin-down luminosity
    P_pred: torch.Tensor,    # predicted spin period (seconds)
    Pdot_pred: torch.Tensor  # predicted period derivative
) -> torch.Tensor:
    """
    Enforces E_dot = 4π²I * |Ṗ| / P³
    Moment of inertia I ≈ 10^45 g·cm² (canonical neutron star)
    """
    I = 1e45  # g·cm²
    E_expected = (4 * torch.pi**2 * I * torch.abs(Pdot_pred)) / (P_pred**3)

    # Normalize to avoid scale issues
    E_expected_norm = E_expected / 1e30
    E_pred_norm     = E_pred    / 1e30

    return F.mse_loss(E_pred_norm, E_expected_norm)


# ─────────────────────────────────────────────
# 5. COMBINED PHYSICS LOSS (Module 3)
# Weighted sum of all stellar physics constraints
# ─────────────────────────────────────────────
def stellar_physics_loss(
    L_pred: torch.Tensor,
    R_pred: torch.Tensor,
    T_pred: torch.Tensor,
    M_pred: torch.Tensor,
    class_pred: torch.Tensor,
    weights: dict = None
) -> torch.Tensor:
    """
    Combined physics loss for Module 3.
    weights: dict with keys 'sb', 'ml', 'ch' and float values.
    """
    if weights is None:
        weights = {'sb': 0.1, 'ml': 0.1, 'ch': 0.2}

    sb_loss = stefan_boltzmann_loss(L_pred, R_pred, T_pred)
    ml_loss = mass_luminosity_loss(L_pred, M_pred, class_pred)
    ch_loss = chandrasekhar_loss(M_pred, class_pred)

    return (
        weights['sb'] * sb_loss +
        weights['ml'] * ml_loss +
        weights['ch'] * ch_loss
    )


# ─────────────────────────────────────────────
# 6. PHYSICS CONSISTENCY SCORE (Inference)
# Returns 0.0 (bad) → 1.0 (physically consistent)
# Used in: unified/infer.py
# ─────────────────────────────────────────────
def physics_consistency_score(
    L_pred: torch.Tensor,
    R_pred: torch.Tensor,
    T_pred: torch.Tensor,
    M_pred: torch.Tensor,
    class_pred: torch.Tensor
) -> torch.Tensor:
    """
    Normalized score indicating how physically consistent
    the model's predictions are. Used in final output display.
    """
    with torch.no_grad():
        loss = stellar_physics_loss(
            L_pred, R_pred, T_pred, M_pred, class_pred,
            weights={'sb': 1.0, 'ml': 1.0, 'ch': 1.0}
        )
        # Convert loss → score: lower loss = higher score
        score = torch.exp(-loss)
    return score.clamp(0.0, 1.0)

if __name__ == "__main__":
    print("Physics loss module loaded successfully")