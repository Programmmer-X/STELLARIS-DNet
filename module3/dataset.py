"""
module3/dataset.py
STELLARIS-DNet — Module 3 Data Pipeline (v2 upgraded)
Sources: Gaia DR3 + SDSS (QSO) + ATNF (NS) + Montreal WD

Upgrades:
  - Validity flags per feature (replaces domain_id)
  - reg_mask per sample (per-class regression supervision)
  - QSO log_lum from bolometric correction (no more random)
  - Better Gaia HR labelling (drops subgiant branch)
  - Synthetic injection only for fully missing classes
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from module3.config import *


# ─────────────────────────────────────────────
# PATH RESOLVER
# ─────────────────────────────────────────────
def _resolve_paths() -> dict:
    paths = {}
    mapping = {
        "gaia": (KAGGLE_GAIA_PATH, GAIA_PATH),
        "sdss": (KAGGLE_SDSS_PATH, SDSS_PATH),
        "atnf": (KAGGLE_ATNF_PATH, ATNF_PATH),
        "mwdd": (KAGGLE_MWDD_PATH, MWDD_PATH),
    }
    for key, (k, l) in mapping.items():
        paths[key] = k if os.path.exists(k) else (l if os.path.exists(l) else None)
    return paths


# ─────────────────────────────────────────────
# 1. PYTORCH DATASET
# Returns: (X, y_class, y_reg, reg_mask)
# reg_mask: (4,) bool — which targets to supervise
# ─────────────────────────────────────────────
class StellarDataset(Dataset):
    def __init__(self, X, y_class, y_reg, reg_mask):
        self.X        = torch.tensor(X,        dtype=torch.float32)
        self.y_class  = torch.tensor(y_class,  dtype=torch.long)
        self.y_reg    = torch.tensor(y_reg,    dtype=torch.float32)
        self.reg_mask = torch.tensor(reg_mask, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx], self.y_reg[idx], self.reg_mask[idx]


# ─────────────────────────────────────────────
# HELPER — build full unified row schema
# All loaders must produce these columns
# ─────────────────────────────────────────────
_UNIFIED_COLS = (
    PHYSICAL_FEATURES + VALIDITY_FLAGS + REGRESSION_TARGETS + ['label']
)


def _empty_row_template(n: int) -> dict:
    """Defaults for missing features — physical zero + validity flag = 0."""
    return {
        # physical features (default fills)
        'teff':      np.full(n, 5778.0),       # solar
        'log_g':     np.zeros(n),
        'feh':       np.zeros(n),
        'abs_mag':   np.zeros(n),
        'bp_rp':     np.zeros(n),
        'redshift':  np.zeros(n),
        'period_ms': np.zeros(n),
        # validity flags — all 0 by default; loaders override per source
        'valid_teff':     np.zeros(n, dtype=np.int8),
        'valid_logg':     np.zeros(n, dtype=np.int8),
        'valid_feh':      np.zeros(n, dtype=np.int8),
        'valid_absmag':   np.zeros(n, dtype=np.int8),
        'valid_bprp':     np.zeros(n, dtype=np.int8),
        'valid_redshift': np.zeros(n, dtype=np.int8),
        'valid_periodms': np.zeros(n, dtype=np.int8),
        # regression targets — NaN by default
        'log_mass':   np.full(n, np.nan),
        'log_lum':    np.full(n, np.nan),
        'log_teff':   np.full(n, np.nan),
        'log_radius': np.full(n, np.nan),
    }


# ─────────────────────────────────────────────
# 2. GAIA DR3 LOADER (v2)
# Classes: MS (0), RG (1), WD (2)
# Joint HR criteria — drops ambiguous subgiants
# ─────────────────────────────────────────────
def _load_gaia(path: str) -> pd.DataFrame | None:
    if not path or not os.path.exists(path):
        return None
    print("Loading Gaia DR3...")

    df_raw = pd.read_csv(path, usecols=[
        'Teff', 'logg', '[Fe/H]', 'GMAG',
        'BPmag', 'RPmag', 'Rad', 'Lum-Flame',
        'Mass-Flame', 'PWD'
    ])
    df_raw = df_raw.dropna(subset=['Teff', 'logg'])
    n = len(df_raw)

    out = _empty_row_template(n)

    # Physical features — all from Gaia, all valid
    out['teff']      = df_raw['Teff'].values
    out['log_g']     = df_raw['logg'].values
    out['feh']       = df_raw['[Fe/H]'].fillna(0.0).values
    out['abs_mag']   = df_raw['GMAG'].fillna(0.0).values
    out['bp_rp']     = (df_raw['BPmag'] - df_raw['RPmag']).fillna(0.0).values

    # Validity flags
    out['valid_teff']   = np.ones(n, dtype=np.int8)
    out['valid_logg']   = np.ones(n, dtype=np.int8)
    out['valid_feh']    = df_raw['[Fe/H]'].notna().astype(np.int8).values
    out['valid_absmag'] = df_raw['GMAG'].notna().astype(np.int8).values
    out['valid_bprp']   = (df_raw['BPmag'].notna() & df_raw['RPmag'].notna()).astype(np.int8).values
    # redshift, periodms remain 0 (invalid)

    # Regression targets — Gaia provides Lum-Flame, Mass-Flame, Rad
    out['log_teff']   = np.log10(out['teff'].clip(TEFF_MIN, TEFF_MAX))
    out['log_lum']    = np.where(
        df_raw['Lum-Flame'].notna(),
        np.log10(df_raw['Lum-Flame'].clip(1e-6, 1e14)),
        np.nan
    )
    out['log_mass']   = np.where(
        df_raw['Mass-Flame'].notna(),
        np.log10(df_raw['Mass-Flame'].clip(0.01, 300)),
        np.nan
    )
    out['log_radius'] = np.where(
        df_raw['Rad'].notna(),
        np.log10(df_raw['Rad'].clip(1e-5, 1e6)),
        np.nan
    )

    # ── Label assignment — joint HR criteria ──
    # Default = -1 (unlabelled, will be dropped)
    label = np.full(n, -1, dtype=np.int64)

    teff    = out['teff']
    log_g   = out['log_g']
    log_lum = out['log_lum']
    pwd     = df_raw['PWD'].fillna(0).values

    # White Dwarf (priority — overrides everything)
    is_wd = (pwd > WD_PROB_THRESHOLD)
    label[is_wd] = 2

    # Main Sequence — strict criteria
    log_lum_safe = np.where(np.isnan(log_lum), 0.0, log_lum)
    is_ms = (
        (log_g > LOG_G_MS_MIN) &
        (teff > TEFF_MS_MIN) & (teff < TEFF_MS_MAX) &
        (log_lum_safe < LOG_LUM_MS_MAX) &
        (~is_wd)
    )
    label[is_ms] = 0

    # Red Giant — strict criteria
    is_rg = (
        (log_g < LOG_G_RG_MAX) &
        (teff < TEFF_RG_MAX) &
        (log_lum_safe > LOG_LUM_RG_MIN) &
        (~is_wd) & (~is_ms)
    )
    label[is_rg] = 1

    out['label'] = label

    # Build DataFrame and drop unlabelled rows (subgiants etc.)
    df_out = pd.DataFrame(out)
    df_out = df_out[df_out['label'] != -1].reset_index(drop=True)

    n_ms = (df_out['label'] == 0).sum()
    n_rg = (df_out['label'] == 1).sum()
    n_wd = (df_out['label'] == 2).sum()
    n_dropped = n - len(df_out)
    print(f"  Gaia: {len(df_out):,} kept | MS={n_ms:,} RG={n_rg:,} WD={n_wd:,}")
    print(f"        ({n_dropped:,} ambiguous samples dropped — subgiants etc.)")
    return df_out


# ─────────────────────────────────────────────
# 3. SDSS LOADER (v2)
# Class: Quasar (4)
# log_lum from bolometric correction (Richards 2006)
# log_mass, log_radius, log_teff = NaN (no supervision)
# ─────────────────────────────────────────────
def _load_sdss(path: str) -> pd.DataFrame | None:
    if not path or not os.path.exists(path):
        return None
    print("Loading SDSS...")

    df_raw = pd.read_csv(
        path, skiprows=1, sep=',', header=0,
        names=['objid','ra','dec','u','g','r','i','z',
               'run','rerun','camcol','field','specobjid',
               'class','redshift','plate','mjd','fiberid']
    )
    df_raw = df_raw[df_raw['class'] == 'QSO'].copy().reset_index(drop=True)
    n = len(df_raw)

    out = _empty_row_template(n)

    # Physical features
    out['abs_mag']   = df_raw['r'].clip(-35, 25).values  # apparent r-mag proxy
    out['bp_rp']     = (df_raw['g'] - df_raw['r']).clip(-2, 5).values
    out['redshift']  = df_raw['redshift'].clip(0, 10).values
    # teff, log_g, feh, period_ms remain at default (zero/solar)
    # → validity flags = 0 for those

    # Validity flags
    out['valid_absmag']   = np.ones(n, dtype=np.int8)
    out['valid_bprp']     = np.ones(n, dtype=np.int8)
    out['valid_redshift'] = np.ones(n, dtype=np.int8)
    # All others remain 0

    # Regression — only log_lum from bolometric correction
    # log10(L_bol/L_sun) = (M_i_sun - M_i) / 2.5 + bolometric_correction
    log_L_bol = (QSO_M_I_SUN - df_raw['r'].values) / 2.5 + QSO_BOL_CORR
    out['log_lum'] = np.clip(log_L_bol, LOG_LUM_MIN, LOG_LUM_MAX)
    # log_mass, log_teff, log_radius remain NaN — masked out in loss

    out['label'] = np.full(n, 4, dtype=np.int64)

    df_out = pd.DataFrame(out)
    print(f"  SDSS QSO: {len(df_out):,} (log_lum from bolometric correction)")
    return df_out


# ─────────────────────────────────────────────
# 4. ATNF LOADER (v2)
# Class: Neutron Star (3)
# Supervision: log_mass + log_radius only
# log_teff, log_lum = NaN (cooling model is synthetic)
# ─────────────────────────────────────────────
def _load_atnf(path: str) -> pd.DataFrame | None:
    if not path or not os.path.exists(path):
        return None
    print("Loading ATNF...")

    df_raw = pd.read_csv(
        path, skiprows=4, sep=';',
        names=['idx','P0','P1','DM','DIST','TYPE',
               'R_LUM','AGE','EDOT','BSURF_I','_empty'],
        na_values=['*']
    )
    df_raw['P0'] = pd.to_numeric(df_raw['P0'], errors='coerce')
    df_raw = df_raw.dropna(subset=['P0']).reset_index(drop=True)
    n = len(df_raw)

    out = _empty_row_template(n)

    # Physical features
    out['period_ms'] = (df_raw['P0'] * 1000).clip(0, 1e7).values
    # Set teff/log_g to NS-typical values (filled, not measured)
    np.random.seed(SEED)
    out['teff']  = 10 ** np.random.uniform(5.5, 7.0, n)   # cooling model estimate
    out['log_g'] = np.random.uniform(13.5, 15.0, n)        # NS surface gravity

    # Validity flags
    out['valid_periodms'] = np.ones(n, dtype=np.int8)
    # teff, log_g are estimates from cooling models → valid = 0

    # Regression targets — only log_mass and log_radius supervised
    mass_ns = np.random.normal(1.4, 0.15, n).clip(NS_MASS_MIN, NS_MASS_MAX)
    out['log_mass']   = np.log10(mass_ns)
    out['log_radius'] = np.full(n, np.log10(1.437e-5))  # 10 km in R_sun

    # log_teff, log_lum remain NaN → masked out via reg_mask
    out['label'] = np.full(n, 3, dtype=np.int64)

    df_out = pd.DataFrame(out)
    print(f"  ATNF NS: {len(df_out):,} (supervising log_mass + log_radius only)")
    return df_out


# ─────────────────────────────────────────────
# 5. MONTREAL WD LOADER (v2)
# Class: White Dwarf (2)
# All 4 regression targets supervised (good data)
# ─────────────────────────────────────────────
def _load_montreal_wd(path: str) -> pd.DataFrame | None:
    if not path or not os.path.exists(path):
        return None
    print("Loading Montreal WD...")

    df_raw = pd.read_csv(path)
    df_raw = df_raw.dropna(subset=['spectype'])
    df_raw = df_raw[df_raw['spectype'].str.strip().str.startswith('D')].copy()
    df_raw = df_raw.dropna(subset=['teff', 'logg']).reset_index(drop=True)
    n = len(df_raw)

    out = _empty_row_template(n)

    # Physical features
    out['teff']    = df_raw['teff'].clip(TEFF_MIN, TEFF_MAX).values
    out['log_g']   = df_raw['logg'].clip(LOGG_MIN, LOGG_MAX).values
    out['abs_mag'] = df_raw['Mv'].fillna(0.0).clip(-35, 25).values
    bp_rp_vals     = (df_raw['BP'] - df_raw['RP']).clip(-2, 5)
    out['bp_rp']   = bp_rp_vals.fillna(0.0).values

    # Validity flags
    out['valid_teff']   = np.ones(n, dtype=np.int8)
    out['valid_logg']   = np.ones(n, dtype=np.int8)
    out['valid_absmag'] = df_raw['Mv'].notna().astype(np.int8).values
    out['valid_bprp']   = (df_raw['BP'].notna() & df_raw['RP'].notna()).astype(np.int8).values

    # Regression targets — Mass + Teff direct, Radius from Nauenberg, L from SB
    mass_wd = df_raw['mass'].clip(MASS_WD_MIN, MASS_WD_MAX).values
    out['log_mass']   = np.log10(mass_wd)
    out['log_teff']   = np.log10(out['teff'])
    log_radius_wd     = np.log10((0.0127 * mass_wd ** (-1/3)).clip(1e-3, 0.1))
    out['log_radius'] = log_radius_wd

    # log_lum: use file value if present, else SB law
    if 'logL' in df_raw.columns:
        has_logl = df_raw['logL'].notna() & (df_raw['logL'].abs() < 20)
        out['log_lum'] = np.where(
            has_logl, df_raw['logL'].values,
            2 * log_radius_wd + 4 * (out['log_teff'] - LOG_TEFF_SUN)
        )
    else:
        out['log_lum'] = 2 * log_radius_wd + 4 * (out['log_teff'] - LOG_TEFF_SUN)

    out['label'] = np.full(n, 2, dtype=np.int64)

    df_out = pd.DataFrame(out)
    print(f"  Montreal WD: {len(df_out):,} (D-type, all 4 targets supervised)")
    return df_out


# ─────────────────────────────────────────────
# 6. SYNTHETIC FALLBACK (v2 — only for missing classes)
# Tighter generation, all validity flags = 0 to mark synthetic origin
# ─────────────────────────────────────────────
def _generate_synthetic_class(cls: int, n: int) -> pd.DataFrame:
    """Generate synthetic samples for a single missing class."""
    np.random.seed(SEED + cls)
    out = _empty_row_template(n)

    if cls == 0:    # Main Sequence
        mass    = 10 ** np.random.uniform(np.log10(0.08), np.log10(100), n)
        log_T   = LOG_TEFF_SUN + 0.5*np.log10(mass) + np.random.normal(0, 0.05, n)
        log_L   = 3.5 * np.log10(mass) + np.random.normal(0, 0.15, n)
        log_R   = 0.5 * (log_L - 4 * (log_T - LOG_TEFF_SUN))
        log_g   = 4.44 + 0.5*np.log10(mass) + np.random.normal(0, 0.2, n)
        out['teff']     = 10**log_T
        out['log_g']    = log_g
        out['feh']      = np.random.normal(-0.05, 0.3, n).clip(FEH_MIN, FEH_MAX)
        out['abs_mag']  = 4.83 - 2.5*log_L
        out['bp_rp']    = (1.5 - 0.7*log_T + np.random.normal(0, 0.1, n)).clip(-2, 5)
        out['log_mass'], out['log_lum']    = np.log10(mass), log_L
        out['log_teff'], out['log_radius'] = log_T, log_R

    elif cls == 1:  # Red Giant
        log_T   = np.log10(np.random.uniform(3500, 5500, n))
        log_R   = np.random.uniform(1.0, 2.5, n)
        log_L   = 2*log_R + 4*(log_T - LOG_TEFF_SUN)
        mass    = np.random.uniform(0.8, 8, n)
        out['teff']     = 10**log_T
        out['log_g']    = np.random.uniform(0.5, 3.0, n)
        out['feh']      = np.random.normal(0.0, 0.3, n).clip(FEH_MIN, FEH_MAX)
        out['abs_mag']  = (4.83 - 2.5*log_L).clip(-35, 25)
        out['bp_rp']    = np.random.uniform(1.0, 2.8, n)
        out['log_mass'], out['log_lum']    = np.log10(mass), log_L
        out['log_teff'], out['log_radius'] = log_T, log_R

    elif cls == 2:  # White Dwarf
        mass    = np.random.normal(0.6, 0.12, n).clip(MASS_WD_MIN, MASS_WD_MAX)
        log_T   = np.random.uniform(np.log10(5000), np.log10(8e4), n)
        log_R   = np.log10((0.0127 * mass ** (-1/3)).clip(1e-3, 0.1))
        log_L   = 2*log_R + 4*(log_T - LOG_TEFF_SUN)
        out['teff']     = 10**log_T
        out['log_g']    = np.random.uniform(7.5, 9.5, n)
        out['abs_mag']  = (4.83 - 2.5*log_L).clip(-35, 25)
        out['bp_rp']    = (-0.5 + 2.5*(4.0 - log_T)).clip(-2, 5)
        out['log_mass'], out['log_lum']    = np.log10(mass), log_L
        out['log_teff'], out['log_radius'] = log_T, log_R

    elif cls == 3:  # Neutron Star
        period_ms = 10 ** np.random.uniform(0, 4, n)  # 1ms - 10s
        log_T     = np.random.uniform(5.5, 7.5, n)
        log_R     = np.full(n, np.log10(1.437e-5))
        mass_ns   = np.random.normal(1.4, 0.15, n).clip(NS_MASS_MIN, NS_MASS_MAX)
        out['teff']      = 10**log_T
        out['log_g']     = np.random.uniform(13.5, 15.0, n)
        out['period_ms'] = period_ms
        out['log_mass'], out['log_radius'] = np.log10(mass_ns), log_R
        # log_teff, log_lum left NaN — same supervision policy as ATNF

    elif cls == 4:  # Quasar
        log_L    = np.random.uniform(10, 14, n)
        redshift = 10 ** np.random.uniform(np.log10(0.1), np.log10(7), n)
        out['abs_mag']   = (4.83 - 2.5*log_L).clip(-35, 25)
        out['bp_rp']     = (0.3 + 0.2*redshift + np.random.normal(0, 0.3, n)).clip(-2, 5)
        out['redshift']  = redshift
        out['log_lum']   = log_L
        # log_mass, log_teff, log_radius NaN — same supervision policy as SDSS

    out['label'] = np.full(n, cls, dtype=np.int64)
    # Validity flags all = 0 (synthetic origin)
    return pd.DataFrame(out)


# ─────────────────────────────────────────────
# 7. VALIDATE + CLIP
# ─────────────────────────────────────────────
def _validate(df: pd.DataFrame) -> pd.DataFrame:
    df['teff']      = df['teff'].clip(TEFF_MIN, TEFF_MAX)
    df['log_g']     = df['log_g'].clip(LOGG_MIN, LOGG_MAX)
    df['feh']       = df['feh'].clip(FEH_MIN, FEH_MAX)
    df['abs_mag']   = df['abs_mag'].clip(-35, 25)
    df['bp_rp']     = df['bp_rp'].clip(-2, 5)
    df['redshift']  = df['redshift'].clip(0, 10)
    df['period_ms'] = df['period_ms'].clip(0, 1e7)

    # Clip regression targets where present (NaN preserved)
    df['log_mass']   = df['log_mass'].clip(LOG_MASS_MIN,   LOG_MASS_MAX)
    df['log_lum']    = df['log_lum'].clip(LOG_LUM_MIN,     LOG_LUM_MAX)
    df['log_teff']   = df['log_teff'].clip(LOG_TEFF_MIN,   LOG_TEFF_MAX)
    df['log_radius'] = df['log_radius'].clip(LOG_RADIUS_MIN, LOG_RADIUS_MAX)

    # Enforce Chandrasekhar on WD
    wd_mask = df['label'] == 2
    df.loc[wd_mask, 'log_mass'] = df.loc[wd_mask, 'log_mass'].clip(
        np.log10(MASS_WD_MIN), np.log10(MASS_WD_MAX)
    )
    return df


# ─────────────────────────────────────────────
# 8. CLASS BALANCING
# ─────────────────────────────────────────────
def _balance(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for cls in range(NUM_STELLAR_CLASSES):
        subset = df[df['label'] == cls]
        if len(subset) > MAX_PER_CLASS:
            subset = subset.sample(MAX_PER_CLASS, random_state=SEED)
        parts.append(subset)
    return pd.concat(parts, ignore_index=True)


# ─────────────────────────────────────────────
# 9. BUILD reg_mask
# Per-sample (4,) mask using REG_SUPERVISION_BY_CLASS
# AND data availability (NaN → no supervision)
# ─────────────────────────────────────────────
def _build_reg_mask(df: pd.DataFrame) -> np.ndarray:
    """
    reg_mask[i, j] = 1 iff:
      - REG_SUPERVISION_BY_CLASS[label_i][j] == 1
      - df[REGRESSION_TARGETS[j]][i] is not NaN
    """
    n = len(df)
    mask = np.zeros((n, NUM_REGRESSION), dtype=np.float32)

    for cls in range(NUM_STELLAR_CLASSES):
        cls_mask = (df['label'] == cls).values
        policy   = REG_SUPERVISION_BY_CLASS[cls]    # [1,1,1,1] etc.
        for j, supervised in enumerate(policy):
            if supervised:
                target_col = REGRESSION_TARGETS[j]
                target_present = df[target_col].notna().values
                mask[cls_mask & target_present, j] = 1.0

    return mask


# ─────────────────────────────────────────────
# 10. MASTER LOADER
# ─────────────────────────────────────────────
def load_stellar_data(use_synthetic_fallback: bool = True) -> tuple:
    """
    Returns:
        train_loader, val_loader, test_loader,
        scaler (fitted StandardScaler),
        class_weights (np.ndarray, shape [5])
    """
    paths  = _resolve_paths()
    frames = []

    for name, fn in [("gaia", _load_gaia), ("sdss", _load_sdss),
                      ("atnf", _load_atnf), ("mwdd", _load_montreal_wd)]:
        df = fn(paths[name])
        if df is not None:
            frames.append(df)

    if not frames:
        if not use_synthetic_fallback:
            raise FileNotFoundError("No real data sources found.")
        # Full synthetic fallback (rare)
        print("⚠️  No real data — generating full synthetic dataset")
        frames = [_generate_synthetic_class(c, 5000) for c in range(NUM_STELLAR_CLASSES)]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.concat(frames, ignore_index=True)

        # Synthetic injection — only for fully-absent classes
        present = set(df['label'].unique())
        missing = set(range(NUM_STELLAR_CLASSES)) - present
        if missing:
            mean_class_size = int(df['label'].value_counts().mean())
            n_synth = max(
                SYNTHETIC_MIN_PER_CLASS,
                int(mean_class_size * SYNTHETIC_FALLBACK_FRACTION)
            )
            print(f"⚠️  Missing classes {missing} — injecting {n_synth}/class synthetic")
            for cls in missing:
                df = pd.concat([df, _generate_synthetic_class(cls, n_synth)],
                                ignore_index=True)

    # Standardise schema, validate, balance
    for col in _UNIFIED_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df = df[_UNIFIED_COLS].copy()

    df = _validate(df)
    df = _balance(df)
    df = df.dropna(subset=PHYSICAL_FEATURES + VALIDITY_FLAGS)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # ── Print distribution ───────────────────
    print("\nClass distribution after balancing:")
    for cls, name in enumerate(STELLAR_CLASSES):
        count = (df['label'] == cls).sum()
        print(f"  {name:15s}: {count:>7,}")

    # ── Build reg_mask ───────────────────────
    reg_mask = _build_reg_mask(df)

    # Replace NaN regression targets with 0.0 (masked out anyway in loss)
    for col in REGRESSION_TARGETS:
        df[col] = df[col].fillna(0.0)

    # ── Print supervision coverage ───────────
    print("\nRegression supervision coverage:")
    for j, name in enumerate(REGRESSION_TARGETS):
        coverage = reg_mask[:, j].mean() * 100
        print(f"  {name:12s}: {coverage:.1f}% of samples supervised")

    # ── Extract arrays ───────────────────────
    X       = df[FEATURE_NAMES].values.astype(np.float32)
    y_class = df['label'].values.astype(np.int64)
    y_reg   = df[REGRESSION_TARGETS].values.astype(np.float32)

    # ── Train/Val/Test split ─────────────────
    (X_tr, X_te, yc_tr, yc_te,
     yr_tr, yr_te, m_tr, m_te) = train_test_split(
        X, y_class, y_reg, reg_mask,
        test_size=TEST_SPLIT, random_state=SEED, stratify=y_class
    )
    (X_tr, X_val, yc_tr, yc_val,
     yr_tr, yr_val, m_tr, m_val) = train_test_split(
        X_tr, yc_tr, yr_tr, m_tr,
        test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        random_state=SEED, stratify=yc_tr
    )

    # ── Normalise — physical features only, validity flags untouched ──
    scaler = StandardScaler()
    n_phys = NUM_PHYSICAL

    X_tr_phys, X_tr_flags   = X_tr[:, :n_phys], X_tr[:, n_phys:]
    X_val_phys, X_val_flags = X_val[:, :n_phys], X_val[:, n_phys:]
    X_te_phys, X_te_flags   = X_te[:, :n_phys], X_te[:, n_phys:]

    X_tr_phys  = scaler.fit_transform(X_tr_phys)
    X_val_phys = scaler.transform(X_val_phys)
    X_te_phys  = scaler.transform(X_te_phys)

    X_tr  = np.concatenate([X_tr_phys,  X_tr_flags],  axis=1).astype(np.float32)
    X_val = np.concatenate([X_val_phys, X_val_flags], axis=1).astype(np.float32)
    X_te  = np.concatenate([X_te_phys,  X_te_flags],  axis=1).astype(np.float32)

    # ── Class weights ────────────────────────
    classes = np.unique(yc_tr)
    raw_w   = compute_class_weight("balanced", classes=classes, y=yc_tr)
    class_weights = np.ones(NUM_STELLAR_CLASSES, dtype=np.float32)
    for cls, w in zip(classes, raw_w):
        class_weights[cls] = float(w)

    print(f"\nSplit — Train: {len(X_tr):,} | Val: {len(X_val):,} | Test: {len(X_te):,}")
    print(f"Class weights: {class_weights.round(3)}")

    train_loader = DataLoader(
        StellarDataset(X_tr,  yc_tr,  yr_tr,  m_tr),
        batch_size=ACTUAL_BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        StellarDataset(X_val, yc_val, yr_val, m_val),
        batch_size=ACTUAL_BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        StellarDataset(X_te,  yc_te,  yr_te,  m_te),
        batch_size=ACTUAL_BATCH_SIZE, shuffle=False
    )

    return train_loader, val_loader, test_loader, scaler, class_weights


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Module 3 Dataset Sanity Check (v2)")
    print("=" * 55)

    tr, vl, te, scaler, cw = load_stellar_data()
    X, yc, yr, mask = next(iter(tr))

    print(f"\nBatch shapes:")
    print(f"  X:        {X.shape}")          # (128, 14)
    print(f"  y_class:  {yc.shape}")         # (128,)
    print(f"  y_reg:    {yr.shape}")          # (128, 4)
    print(f"  reg_mask: {mask.shape}")        # (128, 4)

    assert X.shape[1]    == NUM_FEATURES,   f"Expected {NUM_FEATURES}"
    assert yr.shape[1]   == NUM_REGRESSION, f"Expected {NUM_REGRESSION}"
    assert mask.shape[1] == NUM_REGRESSION, "reg_mask wrong shape"

    print(f"\nFeature means (should be ~0 for first 7): {X[:, :7].mean(dim=0).numpy().round(2)}")
    print(f"Validity flag means (binary 0/1):           {X[:, 7:].mean(dim=0).numpy().round(2)}")
    print(f"\nReg mask coverage in batch:")
    for j, name in enumerate(REGRESSION_TARGETS):
        print(f"  {name:12s}: {mask[:, j].mean().item()*100:.1f}%")

    print("\n✅ dataset.py v2 OK")