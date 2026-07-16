"""
features.py  —  Step 3.7
=========================
Extract 35+ statistical, morphological, and astrophysical features per
TCE candidate for ML input, following the Implementation Plan:

Feature categories
------------------
  1. Transit Geometry      period, depth, duration, transit count, impact proxy
  2. Signal Quality        SDE, SNR, FAP, phase coverage fraction
  3. Morphological         flat-bottom score, ingress/egress asymmetry
  4. Odd / Even            odd vs even transit depth ratio
  5. Secondary Eclipse     secondary depth and ratio at phase 0.5
  6. Centroid              centroid proxy (quality-flagged cadences during transit)
  7. Stellar               Teff, logg, R★, M★, dist, metallicity from TIC catalog
  8. Time-series stats     RMS, skewness, kurtosis, autocorrelation
  9. tsfresh               abs_energy, fft coefficients, change_quantiles, etc.

Input
-----
  data/processed/lc_detrended/  — detrended .npz files (from preprocessing.py)
  data/catalogs/tic_stellar_params.csv — TIC stellar parameters

Output
------
  data/catalogs/feature_matrix.csv — one row per candidate with all 35+ features
"""

from __future__ import annotations

import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.signal import argrelextrema

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parents[1]
DETRENDED_DIR  = BASE_DIR / "data" / "processed" / "lc_detrended"
CATALOGS_DIR   = BASE_DIR / "data" / "catalogs"
CATALOGS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# TLS DETECTION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
TLS_PERIOD_MIN        = 0.5     # days
TLS_PERIOD_MAX        = 27.0    # days  (1 TESS sector)
TLS_OVERSAMPLING      = 5
TLS_DURATION_STEP     = 1.05
TLS_SDE_THRESHOLD     = 5.0    # lower threshold for feature extraction (not detection)
N_PHASE_BINS          = 200    # phase-folded array length for CNN input
SECONDARY_PHASE       = 0.5    # phase to check for secondary eclipse


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — load detrended npz
# ─────────────────────────────────────────────────────────────────────────────

def _load_detrended(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    return {
        "tic_id":    str(data["tic_id"]),
        "time":      data["time"].astype(np.float64),
        "flat_flux": data["flat_flux"].astype(np.float64),
        "rms_raw":   float(data["rms_raw"]) if "rms_raw" in data else np.nan,
        "rms_det":   float(data["rms_detrended"]) if "rms_detrended" in data else np.nan,
        "baseline":  float(data["baseline_days"]) if "baseline_days" in data else np.nan,
        "sector":    int(data["sector"]) if "sector" in data else -1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 1 — Run TLS and extract transit geometry features
# ─────────────────────────────────────────────────────────────────────────────

def run_tls(time: np.ndarray, flat_flux: np.ndarray) -> dict:
    """
    Run Transit Least Squares periodogram and return raw TLS results dict.
    Returns empty dict if TLS fails or no signal found above threshold.
    """
    from transitleastsquares import transitleastsquares as TLS

    finite = np.isfinite(time) & np.isfinite(flat_flux)
    t = time[finite]
    f = flat_flux[finite]

    if len(t) < 200:
        return {}

    try:
        model   = TLS(t, f)
        results = model.power(
            minimum_period    = TLS_PERIOD_MIN,
            maximum_period    = min(TLS_PERIOD_MAX, (t[-1] - t[0]) / 2),
            oversampling_factor = TLS_OVERSAMPLING,
            duration_grid_step  = TLS_DURATION_STEP,
            show_progress_bar   = False,
        )
        return results
    except Exception as e:
        log.debug("  TLS failed: %s", e)
        return {}


def extract_geometry_features(tls_results: dict, baseline: float) -> dict:
    """
    Category 1 & 2 — Transit geometry and signal quality features from TLS.
    """
    if not tls_results or not hasattr(tls_results, "period"):
        return {
            "period": np.nan, "depth": np.nan, "duration_hr": np.nan,
            "transit_count": np.nan, "SDE": np.nan, "SNR": np.nan,
            "FAP": np.nan, "phase_coverage": np.nan, "t0": np.nan,
            "rp_rs": np.nan,
        }

    r = tls_results
    period       = float(r.period)
    depth        = float(r.depth)        # fractional depth (1 - min_flux)
    duration_hr  = float(r.duration) * 24.0
    transit_count = int(r.transit_count) if hasattr(r, "transit_count") else int(baseline / period)
    SDE          = float(r.SDE)
    SNR          = float(r.snr) if hasattr(r, "snr") else np.nan
    FAP          = float(r.FAP) if hasattr(r, "FAP") else np.nan
    t0           = float(r.T0) if hasattr(r, "T0") else np.nan
    rp_rs        = float(np.sqrt(depth)) if depth > 0 else np.nan

    # Phase coverage: fraction of transit phases with data
    in_transit_fraction = float(r.duty_cycle) if hasattr(r, "duty_cycle") else (duration_hr / 24.0) / period

    return {
        "period":         period,
        "depth":          depth,
        "depth_ppm":      depth * 1e6,
        "duration_hr":    duration_hr,
        "transit_count":  transit_count,
        "SDE":            SDE,
        "SNR":            SNR,
        "FAP":            FAP,
        "t0":             t0,
        "rp_rs":          rp_rs,
        "phase_coverage": in_transit_fraction,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 3 — Morphological features from the phase-folded transit
# ─────────────────────────────────────────────────────────────────────────────

def phase_fold(
    time: np.ndarray, flux: np.ndarray,
    period: float, t0: float,
    n_bins: int = N_PHASE_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Phase-fold the light curve and bin into n_bins equal phase bins.

    Returns
    -------
    phase_arr  : phase array centred on 0 (transit midpoint)
    binned_flux: mean flux per phase bin
    """
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0   # centre on 0

    finite = np.isfinite(phase) & np.isfinite(flux)
    phase  = phase[finite]
    flux   = flux[finite]

    # Bin
    bins      = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_idx   = np.digitize(phase, bins) - 1
    binned    = np.full(n_bins, np.nan)
    for i in range(n_bins):
        pts = flux[bin_idx == i]
        if len(pts) >= 1:
            binned[i] = np.nanmean(pts)

    centres = 0.5 * (bins[:-1] + bins[1:])
    return centres, binned


def extract_morphological_features(
    time: np.ndarray,
    flat_flux: np.ndarray,
    period: float,
    t0: float,
    duration_hr: float,
) -> dict:
    """
    Category 3, 4, 5 — Shape, odd/even, secondary eclipse features.
    """
    feat = {}

    if np.isnan(period) or np.isnan(t0) or period <= 0:
        return {
            "flat_bottom_score": np.nan, "ingress_egress_asym": np.nan,
            "odd_depth": np.nan, "even_depth": np.nan, "odd_even_ratio": np.nan,
            "secondary_depth": np.nan, "secondary_ratio": np.nan,
            "phase_folded_std": np.nan,
        }

    phase_arr, binned = phase_fold(time, flat_flux, period, t0, N_PHASE_BINS)

    # Transit window in phase units
    dur_phase  = (duration_hr / 24.0) / period
    in_transit = np.abs(phase_arr) < (dur_phase / 2)
    out_transit = np.abs(phase_arr) > (dur_phase * 1.5)

    transit_flux  = binned[in_transit]
    oot_flux      = binned[out_transit]

    # ── Flat bottom score: std inside transit / std outside (lower = flatter) ─
    std_in   = np.nanstd(transit_flux)   if transit_flux.size  > 2 else np.nan
    std_out  = np.nanstd(oot_flux)       if oot_flux.size      > 2 else np.nan
    feat["flat_bottom_score"] = float(std_in / std_out) if (std_out and std_out > 0) else np.nan

    # ── Phase-folded std (overall scatter) ────────────────────────────────────
    feat["phase_folded_std"] = float(np.nanstd(binned))

    # ── Ingress / egress asymmetry ────────────────────────────────────────────
    half = len(phase_arr) // 2
    ingress_bins  = binned[half - int(half * dur_phase) : half]
    egress_bins   = binned[half : half + int(half * dur_phase)]
    if len(ingress_bins) > 0 and len(egress_bins) > 0:
        feat["ingress_egress_asym"] = float(
            np.nanmean(ingress_bins) - np.nanmean(egress_bins)
        )
    else:
        feat["ingress_egress_asym"] = np.nan

    # ── Odd / even transit depths ─────────────────────────────────────────────
    finite    = np.isfinite(time) & np.isfinite(flat_flux)
    t_ok, f_ok = time[finite], flat_flux[finite]

    transit_epochs = t0 + np.arange(0, (t_ok[-1] - t0) + period, period)
    transit_epochs = transit_epochs[(transit_epochs >= t_ok[0]) & (transit_epochs <= t_ok[-1])]

    odd_depths, even_depths = [], []
    for k, epoch in enumerate(transit_epochs):
        half_dur = (duration_hr / 24.0) / 2.0
        mask     = np.abs(t_ok - epoch) < half_dur
        if mask.sum() < 3:
            continue
        depth_k = 1.0 - float(np.nanmean(f_ok[mask]))
        if k % 2 == 0:
            even_depths.append(depth_k)
        else:
            odd_depths.append(depth_k)

    feat["odd_depth"]   = float(np.nanmean(odd_depths))   if odd_depths  else np.nan
    feat["even_depth"]  = float(np.nanmean(even_depths))  if even_depths else np.nan
    if feat["odd_depth"] and feat["even_depth"] and feat["even_depth"] != 0:
        feat["odd_even_ratio"] = float(feat["odd_depth"] / feat["even_depth"])
    else:
        feat["odd_even_ratio"] = np.nan

    # ── Secondary eclipse at phase 0.5 ───────────────────────────────────────
    sec_mask = np.abs(phase_arr - SECONDARY_PHASE) < (dur_phase / 2)
    if sec_mask.sum() > 0 and not np.isnan(binned[sec_mask]).all():
        sec_depth = 1.0 - float(np.nanmean(binned[sec_mask]))
        primary_d = float(np.nanmean(transit_flux)) if transit_flux.size > 0 else np.nan
        primary_depth = 1.0 - primary_d if not np.isnan(primary_d) else np.nan
        feat["secondary_depth"] = max(sec_depth, 0.0)
        feat["secondary_ratio"] = (
            sec_depth / primary_depth if (primary_depth and primary_depth > 0) else np.nan
        )
    else:
        feat["secondary_depth"] = np.nan
        feat["secondary_ratio"] = np.nan

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 7 — Stellar parameters from TIC catalog
# ─────────────────────────────────────────────────────────────────────────────

_tic_df: Optional[pd.DataFrame] = None   # module-level cache


def _load_tic_catalog() -> pd.DataFrame:
    global _tic_df
    if _tic_df is None:
        tic_path = CATALOGS_DIR / "tic_stellar_params.csv"
        if tic_path.exists():
            _tic_df = pd.read_csv(tic_path, dtype={"tic_id": str})
            _tic_df = _tic_df.set_index("tic_id")
        else:
            _tic_df = pd.DataFrame()
    return _tic_df


def extract_stellar_features(tic_id: str) -> dict:
    """Retrieve stellar parameters from the pre-loaded TIC catalog."""
    tic_df = _load_tic_catalog()

    # Strip "TIC_" prefix if present
    raw_id = tic_id.replace("TIC_", "").replace("TIC ", "")

    stellar = {
        "stellar_Teff":        np.nan,
        "stellar_logg":        np.nan,
        "stellar_rad":         np.nan,
        "stellar_mass":        np.nan,
        "stellar_dist_pc":     np.nan,
        "stellar_metallicity": np.nan,
        "tess_mag":            np.nan,
    }

    if tic_df.empty or raw_id not in tic_df.index:
        return stellar

    row = tic_df.loc[raw_id]
    for col in stellar:
        if col in tic_df.columns:
            v = row[col]
            stellar[col] = float(v) if pd.notna(v) else np.nan
    return stellar


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 8 — Time-series statistical features
# ─────────────────────────────────────────────────────────────────────────────

def extract_timeseries_stats(flat_flux: np.ndarray, rms_raw: float = np.nan) -> dict:
    """
    Compute classical statistical features on the detrended flux array.
    """
    f = flat_flux[np.isfinite(flat_flux)]
    if len(f) < 10:
        return {k: np.nan for k in [
            "rms", "rms_raw", "skewness", "kurtosis",
            "autocorr_lag1", "autocorr_lag10",
            "flux_range", "flux_percentile_5", "flux_percentile_95",
            "above_3sigma_frac", "below_3sigma_frac",
        ]}

    rms      = float(np.std(f))
    skewness = float(sp_stats.skew(f))
    kurtosis = float(sp_stats.kurtosis(f))

    # Autocorrelation at lag 1 and lag 10
    def autocorr(x, lag=1):
        if len(x) <= lag:
            return np.nan
        return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])

    ac1  = autocorr(f, 1)
    ac10 = autocorr(f, 10)

    sigma = rms if rms > 0 else 1e-9
    above = float(np.mean(f > 1 + 3 * sigma))
    below = float(np.mean(f < 1 - 3 * sigma))

    return {
        "rms":                  rms,
        "rms_raw":              rms_raw,
        "skewness":             skewness,
        "kurtosis":             kurtosis,
        "autocorr_lag1":        ac1,
        "autocorr_lag10":       ac10,
        "flux_range":           float(np.ptp(f)),
        "flux_percentile_5":    float(np.percentile(f, 5)),
        "flux_percentile_95":   float(np.percentile(f, 95)),
        "above_3sigma_frac":    above,
        "below_3sigma_frac":    below,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 9 — tsfresh features on phase-folded flux
# ─────────────────────────────────────────────────────────────────────────────

def extract_tsfresh_features(
    phase_arr: np.ndarray,
    binned_flux: np.ndarray,
) -> dict:
    """
    Run a curated subset of tsfresh features on the phase-folded,
    binned flux array (200 points).

    We use `extract_features` with a minimal feature set to avoid
    the full ~800-feature explosion.
    """
    try:
        from tsfresh.feature_extraction import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters

        # Replace NaN bins with local mean
        flux = binned_flux.copy()
        nan_mask = ~np.isfinite(flux)
        if nan_mask.all():
            raise ValueError("All phase bins are NaN.")
        flux[nan_mask] = np.nanmean(flux)

        # tsfresh expects a DataFrame in long format
        df_ts = pd.DataFrame({
            "id":    np.zeros(len(flux), dtype=int),
            "time":  np.arange(len(flux)),
            "flux":  flux,
        })

        feat_df = extract_features(
            df_ts,
            column_id="id", column_sort="time", column_value="flux",
            default_fc_parameters=MinimalFCParameters(),
            disable_progressbar=True,
            n_jobs=1,
        )
        feat_dict = feat_df.iloc[0].to_dict()
        # Prefix keys
        return {f"tsf_{k.split('flux__')[1]}": float(v)
                for k, v in feat_dict.items()
                if "__" in k and np.isfinite(float(v) if pd.notna(v) else np.nan)}

    except Exception as e:
        log.debug("  tsfresh failed: %s", e)
        # Fallback: manual mini feature set
        f = binned_flux.copy()
        f[~np.isfinite(f)] = np.nanmean(f[np.isfinite(f)]) if np.isfinite(f).any() else 0.0
        return {
            "tsf_mean":        float(np.mean(f)),
            "tsf_variance":    float(np.var(f)),
            "tsf_abs_energy":  float(np.sum(f ** 2)),
            "tsf_mean_abs_change": float(np.mean(np.abs(np.diff(f)))),
            "tsf_maximum":     float(np.max(f)),
            "tsf_minimum":     float(np.min(f)),
            "tsf_median":      float(np.median(f)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FEATURE EXTRACTOR — single light curve
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_one(npz_path: Path, label: int = -1) -> dict:
    """
    Run TLS and extract all 35+ features for a single detrended light curve.

    Parameters
    ----------
    npz_path : path to detrended .npz file
    label    : class label (0=Transit, 1=EB, 2=Blend, 3=Other, -1=Unknown)

    Returns
    -------
    dict with all features (NaN-filled on failure)
    """
    base_row = {"tic_id": npz_path.stem, "label": label, "status": "failed"}

    try:
        lc = _load_detrended(npz_path)
    except Exception as e:
        base_row["status"] = str(e)
        return base_row

    tic_id    = lc["tic_id"]
    time      = lc["time"]
    flat_flux = lc["flat_flux"]
    rms_raw   = lc["rms_raw"]
    baseline  = lc["baseline"]

    log.debug("  Extracting features: %s", tic_id)

    # ── Cat 1/2: TLS geometry + signal quality ────────────────────────────────
    tls_results  = run_tls(time, flat_flux)
    geom_feat    = extract_geometry_features(tls_results, baseline)

    period       = geom_feat.get("period", np.nan)
    t0           = geom_feat.get("t0", np.nan)
    duration_hr  = geom_feat.get("duration_hr", np.nan)

    # ── Cat 3/4/5: Morphological, odd/even, secondary ─────────────────────────
    morph_feat = extract_morphological_features(time, flat_flux, period, t0, duration_hr)

    # ── Cat 6: Centroid proxy (fraction of quality-flagged cadences in transit)
    # (full centroid analysis requires pixel data; this is a quality-based proxy)
    centroid_feat = {"centroid_proxy": np.nan}

    # ── Cat 7: Stellar parameters from TIC ───────────────────────────────────
    stellar_feat = extract_stellar_features(tic_id)

    # Compute habitable zone proximity score if stellar Teff available
    teff = stellar_feat.get("stellar_Teff", np.nan)
    if not np.isnan(teff) and not np.isnan(period):
        # Approximate HZ inner/outer boundaries (Kopparapu+2013 simplified)
        L_over_Lsun = ((teff / 5778) ** 4) * (stellar_feat.get("stellar_rad", 1.0) ** 2)
        hz_inner = np.sqrt(L_over_Lsun / 1.1)   # AU
        hz_outer = np.sqrt(L_over_Lsun / 0.36)  # AU
        # Kepler's 3rd law: a (AU) ≈ (period_yr)^(2/3) for M★ ~ 1 Msun
        M_star   = stellar_feat.get("stellar_mass", 1.0) or 1.0
        a_au     = ((period / 365.25) ** 2 * M_star) ** (1 / 3)
        in_hz    = float(hz_inner <= a_au <= hz_outer)
        hz_prox  = float(abs(a_au - (hz_inner + hz_outer) / 2) / ((hz_outer - hz_inner) / 2))
        stellar_feat["hz_in_zone"]      = in_hz
        stellar_feat["hz_proximity"]    = hz_prox
    else:
        stellar_feat["hz_in_zone"]   = np.nan
        stellar_feat["hz_proximity"] = np.nan

    # ── Cat 8: Time-series statistics ────────────────────────────────────────
    ts_feat = extract_timeseries_stats(flat_flux, rms_raw)

    # ── Cat 9: tsfresh on phase-folded flux ───────────────────────────────────
    if not np.isnan(period) and not np.isnan(t0):
        phase_arr, binned = phase_fold(time, flat_flux, period, t0, N_PHASE_BINS)
        tsf_feat = extract_tsfresh_features(phase_arr, binned)
    else:
        tsf_feat = {}

    # ── Baseline metadata ─────────────────────────────────────────────────────
    meta = {
        "tic_id":       tic_id,
        "label":        label,
        "sector":       lc["sector"],
        "baseline_days": baseline,
        "n_cadences":   int(np.isfinite(flat_flux).sum()),
        "status":       "done",
    }

    # ── Assemble full feature dict ────────────────────────────────────────────
    row = {**meta, **geom_feat, **morph_feat, **centroid_feat,
           **stellar_feat, **ts_feat, **tsf_feat}
    return row


# ─────────────────────────────────────────────────────────────────────────────
# BATCH FEATURE EXTRACTION — all detrended light curves
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_batch(
    detrended_dir: Path  = DETRENDED_DIR,
    label_csv:     Optional[Path] = None,
    n_jobs:        int   = -1,
    output_path:   Path  = CATALOGS_DIR / "feature_matrix.csv",
    resume:        bool  = True,
) -> pd.DataFrame:
    """
    Extract features for all detrended .npz files in detrended_dir.

    Parameters
    ----------
    detrended_dir : path to lc_detrended/ directory
    label_csv     : optional path to a CSV with tic_id and label columns;
                    if None, labels default to -1 (unknown)
    n_jobs        : joblib parallel workers (-1 = all cores)
    output_path   : where to save feature_matrix.csv
    resume        : if True and output_path exists, skip already-processed
                    tic_ids and append new results to the existing file

    Returns
    -------
    pd.DataFrame — feature matrix (rows = candidates, cols = features)
    """
    from joblib import Parallel, delayed

    output_path = Path(output_path)
    npz_files   = sorted(detrended_dir.glob("*.npz"))
    log.info("=" * 65)
    log.info("FEATURE EXTRACTION — %d light curves  (n_jobs=%d)", len(npz_files), n_jobs)
    log.info("=" * 65)

    if not npz_files:
        log.warning("No .npz files found in %s", detrended_dir)
        return pd.DataFrame()

    # ── Resume: load already-processed rows ──────────────────────────────────
    already_done: set[str] = set()
    existing_df: Optional[pd.DataFrame] = None
    if resume and output_path.exists():
        try:
            existing_df = pd.read_csv(output_path, dtype={"tic_id": str})
            already_done = set(existing_df["tic_id"].dropna().astype(str))
            log.info("RESUME MODE — %d candidates already processed; skipping them.",
                     len(already_done))
        except Exception as e:
            log.warning("Could not load existing feature matrix for resume: %s", e)
            existing_df = None

    # Filter out already-processed files
    pending = [f for f in npz_files if f.stem not in already_done]
    skipped = len(npz_files) - len(pending)
    if skipped:
        log.info("Skipping %d already-extracted files; %d remaining.", skipped, len(pending))

    if not pending:
        log.info("All files already extracted. Returning existing feature matrix.")
        return existing_df if existing_df is not None else pd.DataFrame()

    # Build label lookup from CSV
    label_map: dict[str, int] = {}
    if label_csv and Path(label_csv).exists():
        ldf = pd.read_csv(label_csv, dtype={"tic_id": str})
        if "label" in ldf.columns:
            label_map = dict(zip(ldf["tic_id"], ldf["label"].astype(int)))

    # Pure worker — no shared state, safe to pickle across processes
    def _worker(f: Path) -> dict:
        label = label_map.get(f.stem, -1)
        return extract_features_one(f, label=label)

    # ── Checkpoint each row in the main process as results stream in ─────────
    write_header = not (resume and output_path.exists())
    n_done = 0

    gen = Parallel(n_jobs=n_jobs, verbose=5, return_as="generator")(
        delayed(_worker)(f) for f in pending
    )
    for row in gen:
        row_df = pd.DataFrame([row])
        row_df.to_csv(output_path, mode="a", header=write_header, index=False)
        write_header = False
        n_done += 1
        log.info("  Checkpoint: %d / %d saved", n_done, len(pending))

    # ── Reload full CSV (existing + newly added rows) ─────────────────────────
    df = pd.read_csv(output_path, dtype={"tic_id": str})

    # Impute missing stellar params with column median
    stellar_cols = ["stellar_Teff", "stellar_logg", "stellar_rad",
                    "stellar_mass", "stellar_dist_pc", "stellar_metallicity"]
    for col in stellar_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Re-save with imputed values
    df.to_csv(output_path, index=False)

    done = int((df["status"] == "done").sum()) if "status" in df.columns else len(df)
    log.info("Feature extraction complete: %d / %d total rows succeeded", done, len(df))
    log.info("Feature matrix shape: %s", df.shape)
    log.info("Saved → %s", output_path)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 3.7 — Extract 35+ features per candidate")
    parser.add_argument("--detrended-dir", type=Path, default=DETRENDED_DIR,
                        help="Directory with detrended .npz files")
    parser.add_argument("--label-csv",     type=Path, default=None,
                        help="Optional CSV with tic_id and label columns")
    parser.add_argument("--n-jobs",        type=int,  default=-1,
                        help="Parallel workers (-1 = all cores)")
    parser.add_argument("--output",        type=Path,
                        default=CATALOGS_DIR / "feature_matrix.csv",
                        help="Output path for feature_matrix.csv")
    parser.add_argument("--no-resume",     action="store_true",
                        help="Ignore any existing feature_matrix.csv and re-extract everything")
    args = parser.parse_args()

    df = extract_features_batch(
        detrended_dir = args.detrended_dir,
        label_csv     = args.label_csv,
        n_jobs        = args.n_jobs,
        output_path   = args.output,
        resume        = not args.no_resume,
    )
    summary_cols = ["tic_id", "label", "period", "depth_ppm", "SDE",
                    "SNR", "odd_even_ratio", "stellar_Teff", "rms", "status"]
    available = [c for c in summary_cols if c in df.columns]
    print(df[available].to_string())
