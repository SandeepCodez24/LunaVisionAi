"""
preprocessing.py  —  Step 3.6
==============================
Preprocess every raw TESS light curve (.npz from data/processed/lc_raw/)
for ML consumption following the Implementation Plan:

  1. Normalize to unit median flux
  2. Sigma-clip outliers (3σ, 5 iterations)  [astropy.stats.sigma_clip]
  3. Fill small gaps (≤3 cadences) with linear interpolation; flag larger gaps
  4. Detrend with wotan bi-weight filter  (window = 0.75 days)
  5. Quality gate: skip if usable data < 13.5 days or noise > 5× photon noise
  6. Save detrended output to  data/processed/lc_detrended/TIC_XXXXXXX.npz

Runs in parallel using joblib for throughput (~500 LC/min on 8-core CPU).

Usage:
    python preprocessing.py [--input-dir <path>] [--output-dir <path>]
                            [--n-jobs -1] [--window 0.75]
"""

from __future__ import annotations

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

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
BASE_DIR        = Path(__file__).resolve().parents[1]
RAW_LC_DIR      = BASE_DIR / "data" / "processed" / "lc_raw"
DETRENDED_DIR   = BASE_DIR / "data" / "processed" / "lc_detrended"
INJ_LC_DIR      = BASE_DIR / "data" / "processed" / "lc_injected"
CATALOGS_DIR    = BASE_DIR / "data" / "catalogs"

DETRENDED_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# QUALITY GATE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MIN_DAYS          = 13.5    # minimum usable baseline (days)
NOISE_SIGMA_LIMIT = 5.0     # max allowed noise = N × expected photon noise
CADENCE_MIN       = 0.0014  # 2-min cadence in days  ≈ 0.00139
SIGMA_CLIP_SIG    = 3.0
SIGMA_CLIP_ITERS  = 5
GAP_FILL_MAX_CAD  = 3       # interpolate gaps ≤ this many cadences
WOTAN_WINDOW      = 0.75    # bi-weight detrending window (days)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3.6 — Core preprocessing functions
# ─────────────────────────────────────────────────────────────────────────────

def normalize_flux(flux: np.ndarray) -> np.ndarray:
    """Divide all flux values by the median → unit median flux."""
    median = np.nanmedian(flux)
    if median == 0 or np.isnan(median):
        raise ValueError("Median flux is zero or NaN — cannot normalize.")
    return flux / median


def sigma_clip_flux(
    time: np.ndarray,
    flux: np.ndarray,
    sigma: float = SIGMA_CLIP_SIG,
    maxiters: int = SIGMA_CLIP_ITERS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterative sigma-clipping to remove outliers and cosmic rays.

    Returns
    -------
    time_clean, flux_clean, mask_clipped  (True = kept, False = clipped)
    """
    from astropy.stats import sigma_clip as astropy_sigma_clip

    clipped = astropy_sigma_clip(flux, sigma=sigma, maxiters=maxiters, masked=True)
    good    = ~clipped.mask & np.isfinite(flux) & np.isfinite(time)
    return time[good], flux[good], good


def fill_gaps(
    time: np.ndarray,
    flux: np.ndarray,
    max_gap_cadences: int = GAP_FILL_MAX_CAD,
    cadence: float = CADENCE_MIN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect gaps in the time series and fill small gaps (≤ max_gap_cadences)
    with linear interpolation.  Larger gaps are flagged in the gap_mask.

    Returns
    -------
    time_out  : original time array (unchanged)
    flux_out  : flux with NaNs replaced where interpolation was applied
    gap_mask  : boolean array, True = cadence is inside a flagged large gap
    """
    dt         = np.diff(time)
    gap_thresh = cadence * (max_gap_cadences + 1.5)   # ~4.5 cadences
    large_gaps = dt > gap_thresh

    gap_mask = np.zeros(len(time), dtype=bool)
    flux_out  = flux.copy()

    for i, is_large in enumerate(large_gaps):
        gap_start_idx = i + 1
        # Mark large gap cadences
        if is_large:
            gap_mask[gap_start_idx] = True

    # Fill NaN values using linear interpolation on the clean series
    nan_mask = ~np.isfinite(flux_out)
    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) >= 2:
            interp_fn   = interp1d(time[valid_idx], flux_out[valid_idx],
                                   kind="linear", bounds_error=False,
                                   fill_value=np.nan)
            flux_out[nan_mask] = interp_fn(time[nan_mask])

    return time, flux_out, gap_mask


def detrend_flux(
    time: np.ndarray,
    flux: np.ndarray,
    method: str = "biweight",
    window_length: float = WOTAN_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten the light curve using wotan detrending.

    Parameters
    ----------
    time          : time array (BTJD)
    flux          : normalized flux array
    method        : wotan method ('biweight', 'lowess', 'median', etc.)
    window_length : sliding window length in days

    Returns
    -------
    flat_flux : detrended flux (normalized around 1.0)
    trend     : the fitted trend model
    """
    from wotan import flatten

    # wotan requires no NaNs
    finite = np.isfinite(time) & np.isfinite(flux)
    t_in   = time[finite]
    f_in   = flux[finite]

    if len(t_in) < 100:
        raise ValueError(f"Too few finite cadences for detrending ({len(t_in)}).")

    flat_flux, trend = flatten(
        t_in, f_in,
        method=method,
        window_length=window_length,
        return_trend=True,
        robust=True,
    )

    # Map back to full array (fill removed cadences with NaN)
    flat_full  = np.full(len(time), np.nan)
    trend_full = np.full(len(time), np.nan)
    flat_full[finite]  = flat_flux
    trend_full[finite] = trend

    return flat_full, trend_full


def quality_gate(
    time: np.ndarray,
    flat_flux: np.ndarray,
    tic_id: str,
) -> tuple[bool, str]:
    """
    Apply quality gating rules from the PRD.

    Returns
    -------
    (passed: bool, reason: str)
    """
    finite = np.isfinite(time) & np.isfinite(flat_flux)
    t_ok   = time[finite]
    f_ok   = flat_flux[finite]

    # Rule 1: minimum baseline
    baseline = t_ok[-1] - t_ok[0] if len(t_ok) > 1 else 0.0
    if baseline < MIN_DAYS:
        return False, f"baseline {baseline:.1f}d < {MIN_DAYS}d"

    # Rule 2: noise floor check
    rms = float(np.nanstd(f_ok))
    # Expected photon noise: empirical proxy ~ 1/sqrt(N_cadences_per_hour)
    n_per_hour    = 30       # 2-min cadence → 30 cadences/hour
    expected_rms  = 1.0 / np.sqrt(n_per_hour * max(len(f_ok), 1))
    if rms > NOISE_SIGMA_LIMIT * expected_rms:
        return False, f"rms {rms:.4f} > {NOISE_SIGMA_LIMIT}× expected {expected_rms:.4f}"

    return True, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# FULL PREPROCESSING PIPELINE — single light curve
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_one(
    npz_path: Path,
    output_dir: Path,
    window_length: float = WOTAN_WINDOW,
) -> dict:
    """
    Run the full Step 3.6 preprocessing pipeline on a single .npz light curve.

    Returns
    -------
    dict with keys: tic_id, status, reason, n_cadences, baseline_days,
                    rms_raw, rms_detrended, out_path
    """
    tic_id = npz_path.stem          # e.g. "TIC_231663901"
    result = {"tic_id": tic_id, "status": "failed", "reason": "", "out_path": None}

    out_path = output_dir / npz_path.name
    if out_path.exists():
        result.update(status="skipped", reason="already exists", out_path=str(out_path))
        return result

    try:
        # ── Load raw data ────────────────────────────────────────────────────
        data        = np.load(npz_path, allow_pickle=True)
        time        = data["time"].astype(np.float64)
        pdcsap_flux = data["pdcsap_flux"].astype(np.float64)
        quality     = data["quality"].astype(np.int32)
        sector      = int(data["sector"]) if "sector" in data else -1

        if len(time) < 100:
            result["reason"] = "fewer than 100 cadences"
            return result

        # ── Step 1: Normalize ────────────────────────────────────────────────
        flux_norm = normalize_flux(pdcsap_flux)
        rms_raw   = float(np.nanstd(flux_norm))

        # ── Step 2: Sigma-clip ───────────────────────────────────────────────
        time_cl, flux_cl, good_mask = sigma_clip_flux(time, flux_norm)

        if len(time_cl) < 100:
            result["reason"] = f"only {len(time_cl)} cadences after sigma-clip"
            return result

        # ── Step 3: Gap detection and filling ────────────────────────────────
        time_cl, flux_cl, gap_mask = fill_gaps(time_cl, flux_cl)

        # ── Step 4: Wotan bi-weight detrending ───────────────────────────────
        flat_flux, trend = detrend_flux(time_cl, flux_cl, window_length=window_length)

        # ── Step 5: Quality gate ─────────────────────────────────────────────
        passed, reason = quality_gate(time_cl, flat_flux, tic_id)
        if not passed:
            result.update(status="gated", reason=reason)
            return result

        rms_detrended = float(np.nanstd(flat_flux[np.isfinite(flat_flux)]))
        baseline      = float(time_cl[-1] - time_cl[0])
        n_good        = int(np.isfinite(flat_flux).sum())

        # ── Step 6: Save .npz ────────────────────────────────────────────────
        np.savez_compressed(
            out_path,
            tic_id        = tic_id,
            time          = time_cl,
            flat_flux     = flat_flux,
            trend         = trend,
            flux_norm     = flux_cl,
            gap_mask      = gap_mask,
            quality       = quality[:len(time_cl)],   # trimmed to match
            sector        = sector,
            rms_raw       = rms_raw,
            rms_detrended = rms_detrended,
            baseline_days = baseline,
        )

        result.update(
            status        = "done",
            reason        = "ok",
            n_cadences    = n_good,
            baseline_days = round(baseline, 3),
            rms_raw       = round(rms_raw, 6),
            rms_detrended = round(rms_detrended, 6),
            out_path      = str(out_path),
        )

    except Exception as exc:
        result["reason"] = str(exc)
        log.debug("  %s: %s", tic_id, exc)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PREPROCESSING — all light curves in a directory
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_batch(
    input_dir:    Path = RAW_LC_DIR,
    output_dir:   Path = DETRENDED_DIR,
    n_jobs:       int  = -1,
    window_length: float = WOTAN_WINDOW,
    include_injected: bool = True,
) -> pd.DataFrame:
    """
    Preprocess all .npz light curves in input_dir (and optionally lc_injected/).
    Runs in parallel using joblib.

    Parameters
    ----------
    input_dir       : directory containing raw .npz files (lc_raw/)
    output_dir      : where to write detrended .npz files (lc_detrended/)
    n_jobs          : joblib parallel workers (-1 = all cores)
    window_length   : wotan detrending window (days)
    include_injected: also process synthetic injections from lc_injected/

    Returns
    -------
    pd.DataFrame with one row per processed file and quality metrics
    """
    from joblib import Parallel, delayed

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all npz files
    npz_files = sorted(input_dir.glob("*.npz"))
    if include_injected and INJ_LC_DIR.exists():
        npz_files += sorted(INJ_LC_DIR.glob("*.npz"))

    log.info("=" * 65)
    log.info("PREPROCESSING — %d light curves  (n_jobs=%d)", len(npz_files), n_jobs)
    log.info("=" * 65)

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(preprocess_one)(f, output_dir, window_length)
        for f in npz_files
    )

    df = pd.DataFrame(results)

    # Summary statistics
    counts = df["status"].value_counts().to_dict()
    log.info("Results: %s", counts)

    # Save report
    report_path = CATALOGS_DIR / "preprocessing_report.csv"
    df.to_csv(report_path, index=False)
    log.info("Report saved → %s", report_path)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Step 3.6 — Preprocess TESS light curves")
    parser.add_argument("--input-dir",  type=Path, default=RAW_LC_DIR,
                        help="Directory with raw .npz files")
    parser.add_argument("--output-dir", type=Path, default=DETRENDED_DIR,
                        help="Output directory for detrended .npz files")
    parser.add_argument("--n-jobs",     type=int, default=-1,
                        help="Parallel workers (-1 = all cores)")
    parser.add_argument("--window",     type=float, default=WOTAN_WINDOW,
                        help="Wotan detrending window in days")
    parser.add_argument("--no-injected", action="store_true",
                        help="Skip synthetic injected light curves")
    args = parser.parse_args()

    report = preprocess_batch(
        input_dir        = args.input_dir,
        output_dir       = args.output_dir,
        n_jobs           = args.n_jobs,
        window_length    = args.window,
        include_injected = not args.no_injected,
    )
    print(report[["tic_id", "status", "n_cadences", "baseline_days",
                  "rms_raw", "rms_detrended"]].to_string())
