"""
module3/dataset.py
STELLARIS-DNet — Module 3 Data Pipeline
Sources: Gaia DR3 + SDSS (QSO) + ATNF (NS) + Montreal WD
Falls back to physically-grounded synthetic data if sources unavailable.
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
# Checks Kaggle paths first, falls back to local
# ─────────────────────────────────────────────
def _resolve_paths() -> dict:
    paths = {}
    mapping = {
        "gaia": (KAGGLE_GAIA_PATH, GAIA_PATH),
        "sdss": (KAGGLE_SDSS_PATH, SDSS_PATH),
        "atnf": (KAGGLE_ATNF_PATH, ATNF_PATH),
        "mwdd": (KAGGLE_MWDD_PATH, MWDD_PATH),
    }
    for key, (kaggle_p, local_p) in mapping.items():
        if os.path.exists(kaggle_p):
            paths[key] = kaggle_p
        elif os.path.exists(local_p):
            paths[key] = local_p
        else:
            paths[key] = None
    return paths


# ─────────────────────────────────────────────
# 1. PYTORCH DATASET
# ─────────────────────────────────────────────
class StellarDataset(Dataset):
    """
    Unified dataset for Module 3.
    X:            (N, NUM_FEATURES=8)  — normalized features
    y_class:      (N,)                 — class label [0-4]
    y_reg:        (N, NUM_REGRESSION=4)— log-scale physical params
    is_synthetic: (N,)                 — 1 if synthetic, 0 if real
    """

    def __init__(
        self,
        X:            np.ndarray,
        y_class:      np.ndarray,
        y_reg:        np.ndarray,
        is_synthetic: np.ndarray
    ):
        self.X            = torch.tensor(X,            dtype=torch.float32)
        self.y_class      = torch.tensor(y_class,      dtype=torch.long)
        self.y_reg        = torch.tensor(y_reg,        dtype=torch.float32)
        self.is_synthetic = torch.tensor(is_synthetic, dtype=torch.bool)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y_class[idx],
            self.y_reg[idx],
            self.is_synthetic[idx]
        )


# ─────────────────────────────────────────────
# 2. GAIA DR3 LOADER
# Classes: Main Sequence (0), Red Giant (1), White Dwarf (2)
# ─────────────────────────────────────────────
def _load_gaia(path: str) -> pd.DataFrame | None:
    """
    Loads Gaia DR3 CSV (dataGaia2.csv).
    Key columns used:
        Teff, logg, [Fe/H], GMAG, BPmag, RPmag,
        Rad, Lum-Flame, Mass-Flame, PWD
    Label assignment:
        PWD > WD_PROB_THRESHOLD (0.9) → White Dwarf (2)
        log_g > LOG_G_MS_BOUNDARY (3.5) → Main Sequence (0)
        else → Red Giant (1)
    """
    if not path or not os.path.exists(path):
        return None

    print("Loading Gaia DR3...")
    df = pd.read_csv(path, usecols=[
        'Teff', 'logg', '[Fe/H]', 'GMAG',
        'BPmag', 'RPmag', 'Rad', 'Lum-Flame',
        'Mass-Flame', 'PWD'
    ])

    # Rename to unified names
    df = df.rename(columns={
        'Teff':       'teff',
        'logg':       'log_g',
        '[Fe/H]':     'feh',
        'GMAG':       'abs_mag',
        'Rad':        'radius',
        'Lum-Flame':  'lum',
        'Mass-Flame': 'mass',
    })

    # Compute BP-RP colour index
    df['bp_rp'] = df['BPmag'] - df['RPmag']
    df = df.drop(columns=['BPmag', 'RPmag'])

    # Fill non-physical features
    df['redshift']  = 0.0
    df['period_ms'] = 0.0
    df['domain_id'] = float(DOMAIN_GAIA)
    df['is_synthetic'] = 0

    # Assign labels
    df['label'] = 1  # default Red Giant
    df.loc[df['log_g'] > LOG_G_MS_BOUNDARY, 'label'] = 0  # Main Sequence
    df.loc[df['PWD'] > WD_PROB_THRESHOLD, 'label']   = 2  # White Dwarf
    df = df.drop(columns=['PWD'])

    # Compute log-scale regression targets
    df['log_teff']   = np.log10(df['teff'].clip(TEFF_MIN, TEFF_MAX))
    df['log_lum']    = np.log10(df['lum'].clip(1e-6, 1e14))
    df['log_mass']   = np.log10(df['mass'].clip(0.01, 300))
    df['log_radius'] = np.log10(df['radius'].clip(1e-5, 1e6))

    df = df.drop(columns=['lum', 'mass', 'radius'])

    # Clip features
    df['teff']    = df['teff'].clip(TEFF_MIN, TEFF_MAX)
    df['log_g']   = df['log_g'].clip(LOGG_MIN, LOGG_MAX)
    df['feh']     = df['feh'].clip(FEH_MIN, FEH_MAX)
    df['abs_mag'] = df['abs_mag'].clip(-35, 25)
    df['bp_rp']   = df['bp_rp'].clip(-2, 5)

    df = df.dropna(subset=['teff', 'log_g'])

    n_ms = (df['label'] == 0).sum()
    n_rg = (df['label'] == 1).sum()
    n_wd = (df['label'] == 2).sum()
    print(f"  Gaia: {len(df):,} rows | MS={n_ms:,} RG={n_rg:,} WD={n_wd:,}")
    return df


# ─────────────────────────────────────────────
# 3. SDSS LOADER
# Class: Quasar (4) only
# ─────────────────────────────────────────────
def _load_sdss(path: str) -> pd.DataFrame | None:
    """
    Loads SDSS Skyserver CSV.
    Uses QSO rows only.
    Key columns: class, redshift, g, r
    bp_rp proxy: g - r (similar wavelength range to Gaia BP-RP)
    abs_mag proxy: r (apparent — acceptable for QSO at known redshift)
    """
    if not path or not os.path.exists(path):
        return None

    print("Loading SDSS...")

    # File has a 1-row metadata header — skip it
    df = pd.read_csv(
        path, skiprows=1, sep=',', header=0,
        names=['objid','ra','dec','u','g','r','i','z',
               'run','rerun','camcol','field','specobjid',
               'class','redshift','plate','mjd','fiberid']
    )

    # Keep only QSOs
    df = df[df['class'] == 'QSO'].copy()

    df['teff']      = 50000.0           # accretion disk temperature estimate
    df['log_g']     = 0.0               # not physically meaningful for QSO
    df['feh']       = 0.0
    df['abs_mag']   = df['r'].clip(-35, 25)   # r-band apparent magnitude proxy
    df['bp_rp']     = (df['g'] - df['r']).clip(-2, 5)  # colour proxy
    df['redshift']  = df['redshift'].clip(0, 10)
    df['period_ms'] = 0.0
    df['domain_id'] = float(DOMAIN_SDSS)
    df['is_synthetic'] = 0
    df['label']     = 4   # Quasar

    # Regression targets
    df['log_teff']   = np.log10(df['teff'].clip(TEFF_MIN, TEFF_MAX))
    df['log_lum']    = df['redshift'] * 1.5 + 10.0  # rough proxy from redshift
    df['log_mass']   = np.random.uniform(6, 10, len(df))  # BH mass range
    df['log_radius'] = np.random.uniform(0, 5, len(df))   # accretion disk scale

    df = df[[
        'teff', 'log_g', 'feh', 'abs_mag', 'bp_rp',
        'redshift', 'period_ms', 'domain_id', 'is_synthetic',
        'label', 'log_mass', 'log_lum', 'log_teff', 'log_radius'
    ]]

    print(f"  SDSS QSO: {len(df):,} rows")
    return df


# ─────────────────────────────────────────────
# 4. ATNF LOADER
# Class: Neutron Star (3)
# ─────────────────────────────────────────────
def _load_atnf(path: str) -> pd.DataFrame | None:
    """
    Loads ATNF Pulsar Catalogue.
    Format: semicolon-delimited, 4 header rows, * = NaN
    Columns: P0, P1, DM, DIST, TYPE, R_LUM, AGE, EDOT, BSURF_I
    """
    if not path or not os.path.exists(path):
        return None

    print("Loading ATNF...")

    df = pd.read_csv(
        path, skiprows=4, sep=';',
        names=['idx', 'P0', 'P1', 'DM', 'DIST',
               'TYPE', 'R_LUM', 'AGE', 'EDOT', 'BSURF_I', '_empty'],
        na_values=['*']
    )
    df = df.drop(columns=['idx', '_empty', 'TYPE',
                           'R_LUM', 'AGE', 'EDOT', 'BSURF_I',
                           'DM', 'DIST'], errors='ignore')

    # Convert P0 to float — may be stored as string in some rows
    df['P0'] = pd.to_numeric(df['P0'], errors='coerce')
    df['P1'] = pd.to_numeric(df['P1'], errors='coerce')

    df = df.dropna(subset=['P0'])

    # period_ms — primary NS discriminator
    df['period_ms'] = (df['P0'] * 1000).clip(0, 1e7)

    # Neutron star surface temperature from cooling model
    # log_age (from P and Pdot) → log_T via Yakovlev cooling relation
    # Fallback: uniform log-normal around 10^6 K
    np.random.seed(SEED)
    n = len(df)
    df['teff']    = 10 ** np.random.uniform(5.5, 7.0, n)
    df['log_g']   = np.random.uniform(13.5, 15.0, n)  # extreme NS surface gravity
    df['feh']     = 0.0
    df['abs_mag'] = 0.0   # NS not optically bright
    df['bp_rp']   = 0.0
    df['redshift']= 0.0
    df['domain_id']    = float(DOMAIN_ATNF)
    df['is_synthetic'] = 0
    df['label']        = 3  # Neutron Star

    # Regression targets
    df['log_teff']   = np.log10(df['teff'].clip(1e4, 1e8))
    df['log_mass']   = np.log10(
        np.random.normal(1.4, 0.15, n).clip(NS_MASS_MIN, NS_MASS_MAX)
    )
    # NS radius ≈ 10 km = 1.437e-5 R_sun
    df['log_radius'] = np.full(n, np.log10(1.437e-5))
    # Thermal luminosity from SB law: L = 4π R² σ T⁴
    R_si = 10e3    # 10 km in meters
    T    = df['teff'].values
    L_si = 4 * np.pi * R_si**2 * SIGMA_SB * T**4
    df['log_lum'] = np.log10((L_si / L_SUN).clip(1e-6, 1e14))

    df = df[[
        'teff', 'log_g', 'feh', 'abs_mag', 'bp_rp',
        'redshift', 'period_ms', 'domain_id', 'is_synthetic',
        'label', 'log_mass', 'log_lum', 'log_teff', 'log_radius'
    ]]

    print(f"  ATNF: {len(df):,} neutron stars")
    return df


# ─────────────────────────────────────────────
# 5. MONTREAL WD LOADER
# Class: White Dwarf (2)
# ─────────────────────────────────────────────
def _load_montreal_wd(path: str) -> pd.DataFrame | None:
    """
    Loads Montreal White Dwarf Database export.
    Columns: spectype, teff, logg, mass, logL, Dpc, Mv, BP, RP
    Filters: keep only proper WD spectral types (start with 'D')
    Mass hard-capped at Chandrasekhar limit.
    logL 98% NaN → computed from SB law using teff + WD radius from mass-radius relation.
    """
    if not path or not os.path.exists(path):
        return None

    print("Loading Montreal WD...")

    df = pd.read_csv(path)

    # Keep only proper white dwarfs — spectral types starting with 'D'
    df = df.dropna(subset=['spectype'])
    df = df[df['spectype'].str.strip().str.startswith('D')].copy()

    # Rename
    df = df.rename(columns={'logg': 'log_g', 'mass': 'mass_wd',
                             'Mv': 'abs_mag', 'BP': 'bp_raw', 'RP': 'rp_raw'})

    # Hard clip mass to physical WD range
    df['mass_wd'] = df['mass_wd'].clip(MASS_WD_MIN, MASS_WD_MAX)

    # BP-RP colour
    df['bp_rp'] = (df['bp_raw'] - df['rp_raw']).clip(-2, 5)

    df['feh']       = 0.0   # WD atmospheres are H/He — no metals
    df['redshift']  = 0.0
    df['period_ms'] = 0.0
    df['domain_id']    = float(DOMAIN_MWDD)
    df['is_synthetic'] = 0
    df['label']        = 2  # White Dwarf

    # Clip teff and log_g
    df['teff']    = df['teff'].clip(TEFF_MIN, TEFF_MAX)
    df['log_g']   = df['log_g'].clip(LOGG_MIN, LOGG_MAX)
    df['abs_mag'] = df['abs_mag'].clip(-35, 25)

    # Regression targets
    df['log_teff'] = np.log10(df['teff'].clip(TEFF_MIN, TEFF_MAX))
    df['log_mass'] = np.log10(df['mass_wd'])

    # WD mass-radius relation: R_WD ≈ 0.0127 × (M/M_sun)^(-1/3) R_sun (Nauenberg 1972)
    df['log_radius'] = np.log10(
        (0.0127 * df['mass_wd'] ** (-1/3)).clip(1e-3, 0.1)
    )

    # log_lum: use file value if available, else compute from SB law
    if 'logL' in df.columns:
        has_logl = df['logL'].notna() & (df['logL'].abs() < 20)
        df['log_lum'] = np.where(has_logl, df['logL'], np.nan)
    else:
        df['log_lum'] = np.nan

    # Fill NaN log_lum via Stefan-Boltzmann: log_L = 2*log_R + 4*(log_T - log_T_sun)
    mask_nan = df['log_lum'].isna()
    df.loc[mask_nan, 'log_lum'] = (
        2 * df.loc[mask_nan, 'log_radius']
        + 4 * (df.loc[mask_nan, 'log_teff'] - LOG_TEFF_SUN)
    )

    df = df[[
        'teff', 'log_g', 'feh', 'abs_mag', 'bp_rp',
        'redshift', 'period_ms', 'domain_id', 'is_synthetic',
        'label', 'log_mass', 'log_lum', 'log_teff', 'log_radius'
    ]]

    df = df.dropna(subset=['teff', 'log_g'])
    print(f"  Montreal WD: {len(df):,} white dwarfs (D-type only)")
    return df


# ─────────────────────────────────────────────
# 6. SYNTHETIC FALLBACK
# Physics-grounded — used only when real data unavailable
# ─────────────────────────────────────────────
def _generate_synthetic(n_per_class: int = SYNTHETIC_N) -> pd.DataFrame:
    """
    Generates physically plausible synthetic stellar data.
    All samples get is_synthetic=1 and domain_id=4.
    Relations used:
      MS:  L ∝ M^3.5, L = 4πR²σT⁴
      RG:  T=[3500,5500], R=[10,300] R_sun
      WD:  M < 1.44 M_sun, R ≈ 0.01 R_sun (Nauenberg)
      NS:  M=[1.1,2.5], R≈10km
      QSO: M_BH=[10^6,10^10], L=[10^10,10^14]
    """
    np.random.seed(SEED)
    rows = []

    def _row(teff, log_g, feh, abs_mag, bp_rp, redshift, period_ms,
             label, log_mass, log_lum, log_teff, log_radius):
        return {
            'teff': teff, 'log_g': log_g, 'feh': feh,
            'abs_mag': abs_mag, 'bp_rp': bp_rp,
            'redshift': redshift, 'period_ms': period_ms,
            'domain_id': float(DOMAIN_SYNTHETIC), 'is_synthetic': 1,
            'label': label,
            'log_mass': log_mass, 'log_lum': log_lum,
            'log_teff': log_teff, 'log_radius': log_radius,
        }

    n = n_per_class

    # ── Main Sequence (0) ────────────────────
    mass    = 10 ** np.random.uniform(np.log10(0.08), np.log10(100), n)
    log_T   = LOG_TEFF_SUN + 0.5 * np.log10(mass) + np.random.normal(0, 0.05, n)
    log_L   = 3.5 * np.log10(mass) + np.random.normal(0, 0.15, n)
    log_R   = 0.5 * (log_L - 4 * (log_T - LOG_TEFF_SUN))
    log_g   = 4.44 + 0.5 * np.log10(mass) + np.random.normal(0, 0.2, n)
    feh     = np.random.normal(-0.05, 0.3, n).clip(FEH_MIN, FEH_MAX)
    abs_mag = 4.83 - 2.5 * log_L
    bp_rp   = (1.5 - 0.7 * log_T + np.random.normal(0, 0.1, n)).clip(-2, 5)
    for i in range(n):
        rows.append(_row(
            teff=10**log_T[i], log_g=log_g[i], feh=feh[i],
            abs_mag=abs_mag[i], bp_rp=bp_rp[i],
            redshift=0.0, period_ms=0.0, label=0,
            log_mass=np.log10(mass[i]), log_lum=log_L[i],
            log_teff=log_T[i], log_radius=log_R[i]
        ))

    # ── Red Giant (1) ────────────────────────
    log_T   = np.log10(np.random.uniform(3500, 5500, n))
    log_R   = np.random.uniform(1.0, 2.5, n)
    log_L   = 2 * log_R + 4 * (log_T - LOG_TEFF_SUN)
    mass    = np.random.uniform(0.8, 8, n)
    log_g   = np.random.uniform(0.5, 3.0, n)
    feh     = np.random.normal(0.0, 0.3, n).clip(FEH_MIN, FEH_MAX)
    abs_mag = (4.83 - 2.5 * log_L).clip(-35, 25)
    bp_rp   = np.random.uniform(1.0, 2.8, n)
    for i in range(n):
        rows.append(_row(
            teff=10**log_T[i], log_g=log_g[i], feh=feh[i],
            abs_mag=abs_mag[i], bp_rp=bp_rp[i],
            redshift=0.0, period_ms=0.0, label=1,
            log_mass=np.log10(mass[i]), log_lum=log_L[i],
            log_teff=log_T[i], log_radius=log_R[i]
        ))

    # ── White Dwarf (2) ──────────────────────
    mass    = np.random.normal(0.6, 0.12, n).clip(MASS_WD_MIN, MASS_WD_MAX)
    log_T   = np.random.uniform(np.log10(5000), np.log10(8e4), n)
    log_R   = np.log10((0.0127 * mass ** (-1/3)).clip(1e-3, 0.1))
    log_L   = 2 * log_R + 4 * (log_T - LOG_TEFF_SUN)
    log_g   = np.random.uniform(7.5, 9.5, n)
    abs_mag = (4.83 - 2.5 * log_L).clip(-35, 25)
    bp_rp   = (-0.5 + 2.5 * (4.0 - log_T) + np.random.normal(0, 0.1, n)).clip(-2, 5)
    for i in range(n):
        rows.append(_row(
            teff=10**log_T[i], log_g=log_g[i], feh=0.0,
            abs_mag=abs_mag[i], bp_rp=bp_rp[i],
            redshift=0.0, period_ms=0.0, label=2,
            log_mass=np.log10(mass[i]), log_lum=log_L[i],
            log_teff=log_T[i], log_radius=log_R[i]
        ))

    # ── Neutron Star (3) ─────────────────────
    period_s = 10 ** np.random.uniform(-3, 1, n)
    log_T    = np.random.uniform(5.5, 7.5, n)
    log_R    = np.full(n, np.log10(1.437e-5))
    R_si     = 10e3
    T_arr    = 10 ** log_T
    L_si     = 4 * np.pi * R_si**2 * SIGMA_SB * T_arr**4
    log_L    = np.log10((L_si / L_SUN).clip(1e-6, 1e14))
    mass_ns  = np.random.normal(1.4, 0.15, n).clip(NS_MASS_MIN, NS_MASS_MAX)
    log_g_ns = np.random.uniform(13.5, 15.0, n)
    for i in range(n):
        rows.append(_row(
            teff=T_arr[i], log_g=log_g_ns[i], feh=0.0,
            abs_mag=0.0, bp_rp=0.0,
            redshift=0.0, period_ms=period_s[i]*1000, label=3,
            log_mass=np.log10(mass_ns[i]), log_lum=log_L[i],
            log_teff=log_T[i], log_radius=log_R[i]
        ))

    # ── Quasar (4) ───────────────────────────
    log_mass = np.random.uniform(6, 10, n)
    log_L    = np.random.uniform(10, 14, n)
    log_T    = np.random.uniform(4.0, 5.5, n)
    log_R    = np.random.uniform(0, 6, n)
    redshift = 10 ** np.random.uniform(np.log10(0.1), np.log10(7), n)
    abs_mag  = (4.83 - 2.5 * log_L).clip(-35, 25)
    bp_rp    = (0.3 + 0.2 * redshift + np.random.normal(0, 0.3, n)).clip(-2, 5)
    for i in range(n):
        rows.append(_row(
            teff=10**log_T[i], log_g=0.0, feh=0.0,
            abs_mag=abs_mag[i], bp_rp=bp_rp[i],
            redshift=redshift[i], period_ms=0.0, label=4,
            log_mass=log_mass[i], log_lum=log_L[i],
            log_teff=log_T[i], log_radius=log_R[i]
        ))

    df = pd.DataFrame(rows)
    print(f"⚠️  Synthetic fallback: {len(df):,} samples ({n_per_class}/class)")
    return df


# ─────────────────────────────────────────────
# 7. VALIDATION + CLIPPING
# ─────────────────────────────────────────────
def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clips extremes, enforces Chandrasekhar on WD,
    fills NaN regression targets class-conditionally.
    """
    # Feature bounds
    df['teff']      = df['teff'].clip(TEFF_MIN, TEFF_MAX)
    df['log_g']     = df['log_g'].clip(LOGG_MIN, LOGG_MAX)
    df['feh']       = df['feh'].clip(FEH_MIN, FEH_MAX)
    df['abs_mag']   = df['abs_mag'].clip(-35, 25)
    df['bp_rp']     = df['bp_rp'].clip(-2, 5)
    df['redshift']  = df['redshift'].clip(0, 10)
    df['period_ms'] = df['period_ms'].clip(0, 1e7)

    # Regression target bounds
    df['log_mass']   = df['log_mass'].clip(LOG_MASS_MIN,   LOG_MASS_MAX)
    df['log_lum']    = df['log_lum'].clip(LOG_LUM_MIN,     LOG_LUM_MAX)
    df['log_teff']   = df['log_teff'].clip(LOG_TEFF_MIN,   LOG_TEFF_MAX)
    df['log_radius'] = df['log_radius'].clip(LOG_RADIUS_MIN, LOG_RADIUS_MAX)

    # Enforce Chandrasekhar: WD log_mass < log10(1.43)
    wd_mask = df['label'] == 2
    df.loc[wd_mask, 'log_mass'] = df.loc[wd_mask, 'log_mass'].clip(
        np.log10(MASS_WD_MIN), np.log10(MASS_WD_MAX)
    )

    # Fill NaN regression targets with class-conditional median
    for col in REGRESSION_TARGETS:
        if df[col].isna().any():
            for cls in range(NUM_STELLAR_CLASSES):
                mask = (df['label'] == cls) & df[col].isna()
                if mask.any():
                    median = df.loc[df['label'] == cls, col].median()
                    df.loc[mask, col] = median if not np.isnan(median) else 0.0

    return df


# ─────────────────────────────────────────────
# 8. CLASS BALANCING
# ─────────────────────────────────────────────
def _balance(df: pd.DataFrame) -> pd.DataFrame:
    """Caps each class at MAX_PER_CLASS to prevent Gaia MS domination."""
    parts = []
    for cls in range(NUM_STELLAR_CLASSES):
        subset = df[df['label'] == cls]
        if len(subset) > MAX_PER_CLASS:
            subset = subset.sample(MAX_PER_CLASS, random_state=SEED)
        parts.append(subset)
    return pd.concat(parts, ignore_index=True)


# ─────────────────────────────────────────────
# 9. MASTER LOADER
# ─────────────────────────────────────────────
# Unified column order — must match FEATURE_NAMES + REGRESSION_TARGETS
_ALL_COLS = FEATURE_NAMES + REGRESSION_TARGETS + ['label', 'is_synthetic']


def load_stellar_data(
    use_synthetic_fallback: bool = True
) -> tuple:
    """
    Master loader for Module 3.
    Tries all 4 real sources in order.
    Falls back to synthetic if sources unavailable.

    Returns:
        train_loader, val_loader, test_loader,
        scaler (fitted StandardScaler),
        class_weights (np.ndarray, shape [5])
    """
    paths = _resolve_paths()
    frames = []

    # ── Load real sources ────────────────────
    gaia = _load_gaia(paths["gaia"])
    if gaia is not None:
        frames.append(gaia)

    sdss = _load_sdss(paths["sdss"])
    if sdss is not None:
        frames.append(sdss)

    atnf = _load_atnf(paths["atnf"])
    if atnf is not None:
        frames.append(atnf)

    mwdd = _load_montreal_wd(paths["mwdd"])
    if mwdd is not None:
        frames.append(mwdd)

    # ── Check for missing classes ────────────
    if not frames:
        if not use_synthetic_fallback:
            raise FileNotFoundError(
                "No datasets found.\n"
                "Place CSVs in data/module3/ or upload to Kaggle.\n"
                "Expected: gaia_dr3.csv, sdss_stars.csv, "
                "atnf_catalog.csv, montreal_wd.csv"
            )
        df = _generate_synthetic()
    else:
        df = pd.concat(frames, ignore_index=True)

        # Fill missing classes with synthetic
        present = set(df['label'].unique())
        missing = set(range(NUM_STELLAR_CLASSES)) - present
        if missing:
            print(f"⚠️  Missing classes {missing} — filling with synthetic")
            syn = _generate_synthetic(n_per_class=SYNTHETIC_N)
            extra = [syn[syn['label'] == cls] for cls in missing]
            df = pd.concat([df] + extra, ignore_index=True)

    # ── Standardise columns ──────────────────
    for col in _ALL_COLS:
        if col not in df.columns:
            df[col] = 0.0

    df = df[_ALL_COLS].copy()
    df = _validate(df)
    df = _balance(df)
    df = df.dropna(subset=FEATURE_NAMES + REGRESSION_TARGETS)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # ── Class distribution ───────────────────
    print("\nClass distribution after balancing:")
    for cls, name in enumerate(STELLAR_CLASSES):
        count = (df['label'] == cls).sum()
        syn_c = ((df['label'] == cls) & (df['is_synthetic'] == 1)).sum()
        print(f"  {name:15s}: {count:>6,}  (synthetic: {syn_c:,})")

    # ── Extract arrays ───────────────────────
    X            = df[FEATURE_NAMES].values.astype(np.float32)
    y_class      = df['label'].values.astype(np.int64)
    y_reg        = df[REGRESSION_TARGETS].values.astype(np.float32)
    is_synthetic = df['is_synthetic'].values.astype(np.int8)

    # ── Train / Val / Test split ─────────────
    (X_tr, X_te, yc_tr, yc_te,
     yr_tr, yr_te, syn_tr, syn_te) = train_test_split(
        X, y_class, y_reg, is_synthetic,
        test_size=TEST_SPLIT, random_state=SEED, stratify=y_class
    )
    (X_tr, X_val, yc_tr, yc_val,
     yr_tr, yr_val, syn_tr, syn_val) = train_test_split(
        X_tr, yc_tr, yr_tr, syn_tr,
        test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        random_state=SEED, stratify=yc_tr
    )

    # ── Normalize — fit on train only ────────
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    # ── Class weights ────────────────────────
    classes     = np.unique(yc_tr)
    raw_w       = compute_class_weight("balanced", classes=classes, y=yc_tr)
    class_weights = np.ones(NUM_STELLAR_CLASSES, dtype=np.float32)
    for cls, w in zip(classes, raw_w):
        class_weights[cls] = float(w)

    print(f"\nSplit — Train: {len(X_tr):,} | Val: {len(X_val):,} | Test: {len(X_te):,}")
    print(f"Class weights: {class_weights.round(3)}")

    train_loader = DataLoader(
        StellarDataset(X_tr,  yc_tr,  yr_tr,  syn_tr),
        batch_size=ACTUAL_BATCH_SIZE, shuffle=True,
        drop_last=True, num_workers=0
    )
    val_loader = DataLoader(
        StellarDataset(X_val, yc_val, yr_val, syn_val),
        batch_size=ACTUAL_BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        StellarDataset(X_te,  yc_te,  yr_te,  syn_te),
        batch_size=ACTUAL_BATCH_SIZE, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader, scaler, class_weights


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Module 3 Dataset Sanity Check")
    print("=" * 55)

    tr, vl, te, scaler, cw = load_stellar_data()

    X, yc, yr, syn = next(iter(tr))
    print(f"\nBatch shapes:")
    print(f"  X:            {X.shape}")       # (128, 8)
    print(f"  y_class:      {yc.shape}")      # (128,)
    print(f"  y_reg:        {yr.shape}")       # (128, 4)
    print(f"  is_synthetic: {syn.shape}")      # (128,)

    assert X.shape[1]  == NUM_FEATURES,    f"Expected {NUM_FEATURES} features, got {X.shape[1]}"
    assert yr.shape[1] == NUM_REGRESSION,  f"Expected {NUM_REGRESSION} targets, got {yr.shape[1]}"
    assert yc.max()    <  NUM_STELLAR_CLASSES, "Class index out of range"
    assert not torch.isnan(X).any(),  "NaN in features"
    assert not torch.isnan(yr).any(), "NaN in regression targets"

    print(f"\nFeature means (should be ~0): {X.mean(dim=0).numpy().round(2)}")
    print(f"Feature stds  (should be ~1): {X.std(dim=0).numpy().round(2)}")
    print(f"\nClasses in batch: {yc.unique().tolist()}")
    print(f"Synthetic in batch: {syn.sum().item()} / {len(syn)}")

    print("\n✅ dataset.py OK")