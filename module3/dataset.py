"""
module3/dataset.py
STELLARIS-DNet — Module 3 Data Pipeline (v3)
Sources: Gaia DR3 + SDSS (QSO) + ATNF (NS) + Montreal WD

v3 changes:
  - Validity flags REMOVED (caused domain shortcut)
  - Per-class fills include light noise (no constants)
  - reg_mask retained (per-class regression supervision)
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
def _resolve_paths():
    """Resolve dataset paths (Kaggle vs local auto-detect)"""

    is_kaggle = os.path.exists("/kaggle/input")

    def pick(kaggle_path, local_path):
        if is_kaggle and kaggle_path and os.path.exists(kaggle_path):
            return kaggle_path
        if (not is_kaggle) and local_path and os.path.exists(local_path):
            return local_path
        return None

    print(f"[ENV] Kaggle={is_kaggle}")

    return {
        "gaia": pick(KAGGLE_GAIA_PATH, GAIA_PATH),
        "sdss": pick(KAGGLE_SDSS_PATH, SDSS_PATH),
        "atnf": pick(KAGGLE_ATNF_PATH, ATNF_PATH),
        "mwdd": pick(KAGGLE_MWDD_PATH, MWDD_PATH),
    }
# ─────────────────────────────────────────────
# 1. PYTORCH DATASET
# Returns: (X, y_class, y_reg, reg_mask)
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
# UNIFIED SCHEMA — v3: no validity flags
# ─────────────────────────────────────────────
_UNIFIED_COLS = PHYSICAL_FEATURES + REGRESSION_TARGETS + ['label']


def _empty_row_template(n: int) -> dict:
    """Defaults for missing features. v3: no validity flag columns."""
    return {
        'teff':      np.full(n, 5778.0),
        'log_g':     np.zeros(n),
        'feh':       np.zeros(n),
        'abs_mag':   np.zeros(n),
        'bp_rp':     np.zeros(n),
        'redshift':  np.zeros(n),
        'period_ms': np.zeros(n),
        'log_mass':   np.full(n, np.nan),
        'log_lum':    np.full(n, np.nan),
        'log_teff':   np.full(n, np.nan),
        'log_radius': np.full(n, np.nan),
    }


# ─────────────────────────────────────────────
# 2. GAIA DR3 LOADER (v3 — no validity flags)
# Classes: MS (0), RG (1), WD (2)
# ─────────────────────────────────────────────
def _load_gaia(path: str) -> pd.DataFrame | None:
    if not path or not os.path.exists(path):
        return None
    print("Loading Gaia DR3 (v4 — CMD labelling)...")

    df_raw = pd.read_csv(path, usecols=[
        'Teff', 'logg', '[Fe/H]', 'GMAG',
        'BPmag', 'RPmag', 'Rad', 'Lum-Flame',
        'Mass-Flame', 'PWD'
    ])

    # CMD inputs are mandatory — drop rows missing photometry
    df_raw['bp_rp'] = df_raw['BPmag'] - df_raw['RPmag']
    df_raw = df_raw.dropna(subset=['GMAG', 'bp_rp']).reset_index(drop=True)
    n = len(df_raw)
    out = _empty_row_template(n)

    # Physical features (log_g, teff, feh kept as inputs — model has them but
    # cannot use them to recover the label, since label is now CMD-derived)
    out['teff']    = df_raw['Teff'].fillna(5778.0).values
    out['log_g']   = df_raw['logg'].fillna(4.4).values
    out['feh']     = df_raw['[Fe/H]'].fillna(0.0).values
    out['abs_mag'] = df_raw['GMAG'].values
    out['bp_rp']   = df_raw['bp_rp'].values

    # Regression targets — unchanged logic
    out['log_teff'] = np.log10(np.clip(out['teff'], TEFF_MIN, TEFF_MAX))
    out['log_lum'] = np.where(
        df_raw['Lum-Flame'].notna(),
        np.log10(df_raw['Lum-Flame'].clip(1e-6, 1e14)), np.nan
    )
    out['log_mass'] = np.where(
        df_raw['Mass-Flame'].notna(),
        np.log10(df_raw['Mass-Flame'].clip(0.01, 300)), np.nan
    )
    out['log_radius'] = np.where(
        df_raw['Rad'].notna(),
        np.log10(df_raw['Rad'].clip(1e-5, 1e6)), np.nan
    )

    # ── CMD-based labelling ─────────────────────────────
    bp_rp   = out['bp_rp']
    abs_mag = out['abs_mag']
    pwd     = df_raw['PWD'].fillna(0).values

    # MS ridge: empirical Gaia HRD main-sequence quadratic
    ms_ridge = (
        CMD_MS_RIDGE_C0
        + CMD_MS_RIDGE_C1 * bp_rp
        + CMD_MS_RIDGE_C2 * bp_rp**2
    )
    delta = abs_mag - ms_ridge   # +ve = below ridge (faint), −ve = above (bright)

    in_range = (bp_rp >= CMD_BPRP_MIN) & (bp_rp <= CMD_BPRP_MAX)
    label = np.full(n, -1, dtype=np.int64)

    # Priority 1 — WD via Gaia probability
    is_wd = (pwd > WD_PROB_THRESHOLD)
    label[is_wd] = 2

    # Priority 2 — RG: ≥1.5 mag brighter than MS ridge, redder than 0.5
    is_rg = (
        (delta < CMD_RG_OFFSET) &
        (bp_rp > 0.5) &
        in_range & (~is_wd)
    )
    label[is_rg] = 1

    # Priority 3 — MS: within MS band of ridge
    is_ms = (
        (np.abs(delta) <= CMD_MS_HALFWIDTH) &
        in_range & (~is_wd) & (~is_rg)
    )
    label[is_ms] = 0

    # Subgiants & off-track stars: KEEP, don't drop. Default to MS.
    # These are the samples that v3 dropped (212k) — now we retain them and
    # let the model handle the natural CMD fuzziness.
    n_amb_before = (label == -1).sum()
    is_amb = (label == -1) & in_range
    label[is_amb] = 0

    out['label'] = label
    df_out = pd.DataFrame(out)
    df_out = df_out[df_out['label'] != -1].reset_index(drop=True)

    n_ms = (df_out['label'] == 0).sum()
    n_rg = (df_out['label'] == 1).sum()
    n_wd = (df_out['label'] == 2).sum()
    n_amb_kept = is_amb.sum()
    n_dropped = n - len(df_out)
    print(f"  Gaia: {len(df_out):,} kept | "
          f"MS={n_ms:,} (incl. {n_amb_kept:,} ambiguous) "
          f"RG={n_rg:,} WD={n_wd:,}")
    print(f"        ({n_dropped:,} samples outside CMD bp_rp range)")
    return df_out

# ─────────────────────────────────────────────
# 3. SDSS LOADER (v3)
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

    np.random.seed(SEED + 1)
    out['abs_mag']  = df_raw['r'].clip(-35, 25).values
    out['bp_rp']    = (df_raw['g'] - df_raw['r']).clip(-2, 5).values
    out['redshift'] = df_raw['redshift'].clip(0, 10).values

    # Realistic non-constant fills for placeholder features
    # (avoid constant 50000 K teff that becomes a domain identifier)
    out['teff']  = 10 ** np.random.uniform(4.0, 5.5, n)         # AGN disk range
    out['log_g'] = np.random.normal(0.0, 0.3, n).clip(-1, 1)    # near zero with noise
    out['feh']   = np.random.normal(0.0, 0.2, n).clip(-1, 1)
    out['period_ms'] = 0.0   # genuinely zero — not a pulsar

    # Regression: only log_lum supervised (bolometric correction)
    log_L_bol = (QSO_M_I_SUN - df_raw['r'].values) / 2.5 + QSO_BOL_CORR
    out['log_lum'] = np.clip(log_L_bol, LOG_LUM_MIN, LOG_LUM_MAX)
    out['label']   = np.full(n, 4, dtype=np.int64)

    df_out = pd.DataFrame(out)
    print(f"  SDSS QSO: {len(df_out):,} (log_lum from bolometric correction)")
    return df_out


# ─────────────────────────────────────────────
# 4. ATNF LOADER (v3)
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

    np.random.seed(SEED + 2)
    out['period_ms'] = (df_raw['P0'] * 1000).clip(0, 1e7).values
    # Realistic noise on filled features
    out['teff']    = 10 ** np.random.uniform(5.5, 7.0, n)
    out['log_g']   = np.random.normal(14.5, 0.5, n).clip(13.0, 15.5)
    out['feh']     = np.random.normal(0.0, 0.2, n).clip(-1, 1)
    out['abs_mag'] = np.random.normal(0.0, 1.0, n).clip(-5, 5)
    out['bp_rp']   = np.random.normal(0.0, 0.5, n).clip(-2, 2)

    mass_ns = np.random.normal(1.4, 0.15, n).clip(NS_MASS_MIN, NS_MASS_MAX)
    out['log_mass']   = np.log10(mass_ns)
    out['log_radius'] = np.full(n, np.log10(1.437e-5)) + np.random.normal(0, 0.05, n)
    # log_teff, log_lum remain NaN → masked via reg_mask

    out['label'] = np.full(n, 3, dtype=np.int64)
    df_out = pd.DataFrame(out)
    print(f"  ATNF NS: {len(df_out):,} (log_mass + log_radius supervised)")
    return df_out


# ─────────────────────────────────────────────
# 5. MONTREAL WD LOADER (v3)
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

    np.random.seed(SEED + 3)
    out['teff']    = df_raw['teff'].clip(TEFF_MIN, TEFF_MAX).values
    out['log_g']   = df_raw['logg'].clip(LOGG_MIN, LOGG_MAX).values
    out['abs_mag'] = df_raw['Mv'].fillna(0.0).clip(-35, 25).values
    out['bp_rp']   = (df_raw['BP'] - df_raw['RP']).clip(-2, 5).fillna(0.0).values
    # WD atmospheres are H/He → feh small noise around 0
    out['feh']     = np.random.normal(0.0, 0.2, n).clip(-1, 1)

    mass_wd = df_raw['mass'].clip(MASS_WD_MIN, MASS_WD_MAX).values
    out['log_mass']   = np.log10(mass_wd)
    out['log_teff']   = np.log10(out['teff'])
    log_radius_wd     = np.log10((0.0127 * mass_wd ** (-1/3)).clip(1e-3, 0.1))
    out['log_radius'] = log_radius_wd

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
# 6. SYNTHETIC FALLBACK (v3)
# ─────────────────────────────────────────────
def _generate_synthetic_class(cls: int, n: int) -> pd.DataFrame:
    np.random.seed(SEED + cls + 100)
    out = _empty_row_template(n)

    if cls == 0:
        mass    = 10 ** np.random.uniform(np.log10(0.08), np.log10(100), n)
        log_T   = LOG_TEFF_SUN + 0.5*np.log10(mass) + np.random.normal(0, 0.05, n)
        log_L   = 3.5 * np.log10(mass) + np.random.normal(0, 0.15, n)
        log_R   = 0.5 * (log_L - 4 * (log_T - LOG_TEFF_SUN))
        out['teff']    = 10**log_T
        out['log_g']   = 4.44 + 0.5*np.log10(mass) + np.random.normal(0, 0.2, n)
        out['feh']     = np.random.normal(-0.05, 0.3, n).clip(FEH_MIN, FEH_MAX)
        out['abs_mag'] = 4.83 - 2.5*log_L
        out['bp_rp']   = (1.5 - 0.7*log_T + np.random.normal(0, 0.1, n)).clip(-2, 5)
        out['log_mass'], out['log_lum']    = np.log10(mass), log_L
        out['log_teff'], out['log_radius'] = log_T, log_R

    elif cls == 1:
        log_T   = np.log10(np.random.uniform(3500, 5500, n))
        log_R   = np.random.uniform(1.0, 2.5, n)
        log_L   = 2*log_R + 4*(log_T - LOG_TEFF_SUN)
        mass    = np.random.uniform(0.8, 8, n)
        out['teff']    = 10**log_T
        out['log_g']   = np.random.uniform(0.5, 3.0, n)
        out['feh']     = np.random.normal(0.0, 0.3, n).clip(FEH_MIN, FEH_MAX)
        out['abs_mag'] = (4.83 - 2.5*log_L).clip(-35, 25)
        out['bp_rp']   = np.random.uniform(1.0, 2.8, n)
        out['log_mass'], out['log_lum']    = np.log10(mass), log_L
        out['log_teff'], out['log_radius'] = log_T, log_R

    elif cls == 2:
        mass    = np.random.normal(0.6, 0.12, n).clip(MASS_WD_MIN, MASS_WD_MAX)
        log_T   = np.random.uniform(np.log10(5000), np.log10(8e4), n)
        log_R   = np.log10((0.0127 * mass ** (-1/3)).clip(1e-3, 0.1))
        log_L   = 2*log_R + 4*(log_T - LOG_TEFF_SUN)
        out['teff']    = 10**log_T
        out['log_g']   = np.random.uniform(7.5, 9.5, n)
        out['abs_mag'] = (4.83 - 2.5*log_L).clip(-35, 25)
        out['bp_rp']   = (-0.5 + 2.5*(4.0 - log_T)).clip(-2, 5)
        out['log_mass'], out['log_lum']    = np.log10(mass), log_L
        out['log_teff'], out['log_radius'] = log_T, log_R

    elif cls == 3:
        period_ms = 10 ** np.random.uniform(0, 4, n)
        log_T     = np.random.uniform(5.5, 7.5, n)
        log_R     = np.full(n, np.log10(1.437e-5))
        mass_ns   = np.random.normal(1.4, 0.15, n).clip(NS_MASS_MIN, NS_MASS_MAX)
        out['teff']      = 10**log_T
        out['log_g']     = np.random.uniform(13.5, 15.0, n)
        out['period_ms'] = period_ms
        out['log_mass'], out['log_radius'] = np.log10(mass_ns), log_R

    elif cls == 4:
        log_L    = np.random.uniform(10, 14, n)
        redshift = 10 ** np.random.uniform(np.log10(0.1), np.log10(7), n)
        out['abs_mag']  = (4.83 - 2.5*log_L).clip(-35, 25)
        out['bp_rp']    = (0.3 + 0.2*redshift + np.random.normal(0, 0.3, n)).clip(-2, 5)
        out['redshift'] = redshift
        out['log_lum']  = log_L

    out['label'] = np.full(n, cls, dtype=np.int64)
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

    df['log_mass']   = df['log_mass'].clip(LOG_MASS_MIN,   LOG_MASS_MAX)
    df['log_lum']    = df['log_lum'].clip(LOG_LUM_MIN,     LOG_LUM_MAX)
    df['log_teff']   = df['log_teff'].clip(LOG_TEFF_MIN,   LOG_TEFF_MAX)
    df['log_radius'] = df['log_radius'].clip(LOG_RADIUS_MIN, LOG_RADIUS_MAX)

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
# ─────────────────────────────────────────────
def _build_reg_mask(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    mask = np.zeros((n, NUM_REGRESSION), dtype=np.float32)
    for cls in range(NUM_STELLAR_CLASSES):
        cls_mask = (df['label'] == cls).values
        policy   = REG_SUPERVISION_BY_CLASS[cls]
        for j, supervised in enumerate(policy):
            if supervised:
                target_present = df[REGRESSION_TARGETS[j]].notna().values
                mask[cls_mask & target_present, j] = 1.0
    return mask


# ─────────────────────────────────────────────
# 10. MASTER LOADER (v3)
# ─────────────────────────────────────────────
def load_stellar_data(use_synthetic_fallback: bool = True) -> tuple:
    paths  = _resolve_paths()
     
    # ─────────────────────────────────────────────
    # HARD FAIL: ensure real datasets are present
    # ─────────────────────────────────────────────
    print("\n[DATA SOURCE CHECK — dataset.py]")
    for name, path in paths.items():
        exists = (path is not None) and os.path.exists(path)
        print(f"  {name:5s}: {path} | exists={exists}")

    if not any(path and os.path.exists(path) for path in paths.values()):
        raise RuntimeError(
            "❌ No real datasets found.\n"
            "Refusing to train on synthetic data.\n"
            "Fix your DATA_ROOT / Kaggle paths."
        )

    frames = []

    for name, fn in [("gaia", _load_gaia), ("sdss", _load_sdss),
                      ("atnf", _load_atnf), ("mwdd", _load_montreal_wd)]:
        df = fn(paths[name])
        if df is not None:
            frames.append(df)

    if not frames:
        if not use_synthetic_fallback:
            raise FileNotFoundError("No real data sources found.")
        print("⚠️  No real data — generating full synthetic dataset")
        frames = [_generate_synthetic_class(c, 5000) for c in range(NUM_STELLAR_CLASSES)]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.concat(frames, ignore_index=True)
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

    for col in _UNIFIED_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df = df[_UNIFIED_COLS].copy()
    df = _validate(df)
    df = _balance(df)
    df = df.dropna(subset=PHYSICAL_FEATURES)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print("\nClass distribution after balancing:")
    for cls, name in enumerate(STELLAR_CLASSES):
        count = (df['label'] == cls).sum()
        print(f"  {name:15s}: {count:>7,}")

    reg_mask = _build_reg_mask(df)
    for col in REGRESSION_TARGETS:
        df[col] = df[col].fillna(0.0)

    print("\nRegression supervision coverage:")
    for j, name in enumerate(REGRESSION_TARGETS):
        coverage = reg_mask[:, j].mean() * 100
        print(f"  {name:12s}: {coverage:.1f}% supervised")

    X       = df[FEATURE_NAMES].values.astype(np.float32)
    y_class = df['label'].values.astype(np.int64)
    y_reg   = df[REGRESSION_TARGETS].values.astype(np.float32)

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

    # v3: standardize all 7 features (no validity-flag exclusion logic)
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_te  = scaler.transform(X_te).astype(np.float32)

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
    print("Module 3 Dataset Sanity Check (v3 — no validity flags)")
    print("=" * 55)

    tr, vl, te, scaler, cw = load_stellar_data()
    X, yc, yr, mask = next(iter(tr))

    print(f"\nBatch shapes:")
    print(f"  X:        {X.shape}")           # (128, 7)  ← back to 7
    print(f"  y_class:  {yc.shape}")
    print(f"  y_reg:    {yr.shape}")
    print(f"  reg_mask: {mask.shape}")

    assert X.shape[1]    == NUM_FEATURES,   f"Expected {NUM_FEATURES}"
    assert yr.shape[1]   == NUM_REGRESSION
    assert mask.shape[1] == NUM_REGRESSION

    print(f"\nFeature means (should be ~0): {X.mean(dim=0).numpy().round(2)}")
    print(f"Feature stds  (should be ~1): {X.std(dim=0).numpy().round(2)}")
    print(f"\nReg mask coverage in batch:")
    for j, name in enumerate(REGRESSION_TARGETS):
        print(f"  {name:12s}: {mask[:, j].mean().item()*100:.1f}%")

    print("\n✅ dataset.py v3 OK")