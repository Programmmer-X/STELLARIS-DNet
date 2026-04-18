# core/physics_loss.py
# All physics-informed loss functions for STELLARIS-DNet
# Used across Module 1, 2, 3 and Unified model

import torch
import torch.nn.functional as F

# ── Constants ────────────────────────────────────────────
STEFAN_BOLTZMANN = 5.6704e-8   # W m^-2 K^-4
CHANDRASEKHAR_LIMIT = 1.44     # Solar masses
SOLAR_LUMINOSITY = 3.828e26    # Watts
SOLAR_RADIUS = 6.957e8         # Meters


# ── 1. Stefan-Boltzmann Loss ─────────────────────────────
# Enforces: L = 4πR²σT⁴
# Used in: Module 3 (all stellar objects)
def stefan_boltzmann_loss(L_pred, R_pred, T_pred):
    """
    L_pred : predicted luminosity (solar units)
    R_pred : predicted radius (solar units)
    T_pred : predicted temperature (Kelvin)
    """
    L_expected = (
        4 * torch.pi
        * (R_pred * SOLAR_RADIUS) ** 2
        * STEFAN_BOLTZMANN
        * T_pred ** 4
    ) / SOLAR_LUMINOSITY  # convert back to solar units
    return F.mse_loss(L_pred, L_expected)


# ── 2. Mass-Luminosity Loss ──────────────────────────────
# Enforces: L ∝ M^3.5 (valid for main sequence stars only)
# Used in: Module 3 (main sequence class only)
def mass_luminosity_loss(L_pred, M_pred, class_mask=None):
    """
    L_pred     : predicted luminosity (solar units)
    M_pred     : predicted mass (solar units)
    class_mask : boolean tensor, True where object is main sequence
    """
    if class_mask is not None:
        if class_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)
        L_pred = L_pred[class_mask]
        M_pred = M_pred[class_mask]

    L_expected = M_pred ** 3.5
    return F.mse_loss(L_pred, L_expected)


# ── 3. Chandrasekhar Limit Loss ──────────────────────────
# Enforces: white dwarf mass < 1.44 M☉
# Used in: Module 3 (white dwarf class only)
def chandrasekhar_loss(M_pred, class_pred, wd_class_idx=2):
    """
    M_pred       : predicted mass (solar units)
    class_pred   : predicted class indices (argmax output)
    wd_class_idx : index of white dwarf in your class list
    """
    wd_mask = (class_pred == wd_class_idx)
    if wd_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    wd_masses = M_pred[wd_mask]
    violation = F.relu(wd_masses - CHANDRASEKHAR_LIMIT)
    return violation.mean()


# ── 4. Pulsar Spin-Down Loss ─────────────────────────────
# Enforces: E_dot = -4π²I * P_dot / P³
# Used in: Module 1 (pulsar subtype classification)
def spindown_loss(P_pred, Pdot_pred, Edot_pred,
                  I=1e45):
    """
    P_pred    : predicted spin period (seconds)
    Pdot_pred : predicted period derivative (s/s)
    Edot_pred : predicted spin-down luminosity (ergs/s)
    I         : moment of inertia (default neutron star ~ 1e45 g cm²)
    """
    Edot_expected = (
        -4 * torch.pi ** 2 * I * Pdot_pred / P_pred ** 3
    )
    return F.mse_loss(Edot_pred, Edot_expected)


# ── 5. Combined Physics Loss ─────────────────────────────
# Single function called in training loops
def physics_loss(
    # Stellar params (Module 3)
    L_pred=None, R_pred=None, T_pred=None,
    M_pred=None, class_pred=None,
    ms_mask=None, wd_class_idx=2,
    # Pulsar params (Module 1)
    P_pred=None, Pdot_pred=None, Edot_pred=None,
    # Loss weights
    w_sb=0.1, w_ml=0.1, w_ch=0.2, w_sd=0.1
):
    loss = torch.tensor(0.0, requires_grad=True)

    # Stefan-Boltzmann
    if all(v is not None for v in [L_pred, R_pred, T_pred]):
        loss = loss + w_sb * stefan_boltzmann_loss(
            L_pred, R_pred, T_pred)

    # Mass-Luminosity
    if all(v is not None for v in [L_pred, M_pred]):
        loss = loss + w_ml * mass_luminosity_loss(
            L_pred, M_pred, ms_mask)

    # Chandrasekhar
    if all(v is not None for v in [M_pred, class_pred]):
        loss = loss + w_ch * chandrasekhar_loss(
            M_pred, class_pred, wd_class_idx)

    # Spin-down
    if all(v is not None for v in [P_pred, Pdot_pred, Edot_pred]):
        loss = loss + w_sd * spindown_loss(
            P_pred, Pdot_pred, Edot_pred)

    return loss