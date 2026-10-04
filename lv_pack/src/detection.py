"""
detection.py — Stage 3: periodicity detection (Transit Least Squares).

Owns the TLS run and the conversion of raw TLS results into transit-geometry /
signal-quality features. features.py imports from here (and re-exports the
names for backward compatibility), so there is exactly one TLS code path.

Public API
----------
run_tls(time, flat_flux)                     -> TLS results object ({} on failure)
extract_geometry_features(results, baseline) -> dict of period/depth/SDE/...
detect_candidate(time, flat_flux, tic_id)    -> candidate dict + pass/fail flag
detect_batch(npz_dir, out_csv)               -> candidates CSV (Stage 3 output)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TLS_PERIOD_MIN      = 0.5      # days
TLS_PERIOD_MAX      = 27.0     # days  (1 TESS sector)
TLS_OVERSAMPLING    = 3        # Standard TLS oversampling (fast & accurate)
TLS_DURATION_STEP   = 1.05
TLS_SDE_THRESHOLD   = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 1 — Run TLS and extract transit geometry features
# ─────────────────────────────────────────────────────────────────────────────

TLS_BIN_MIN       = 10.0    # bin to 10-min for the period search (~5x fewer points)
MAX_SEGMENT_GAP_D = 30.0    # split stitched sectors at gaps longer than this


def select_search_window(time: np.ndarray, flux: np.ndarray, max_gap: float = MAX_SEGMENT_GAP_D,
                          *co_arrays: np.ndarray):
    """
    Keep the longest contiguous segment of a stitched multi-sector light curve.

    Two sectors months/years apart stitch into a baseline of hundreds of days;
    TLS's period grid scales with baseline, so one such target takes >10x
    longer than a normal one (and folding across the gap smears the period).
    Consecutive sectors (gap ~1 day) are left intact.

    Any extra arrays passed in `co_arrays` (e.g. centroid columns) are
    filtered and sliced identically to time/flux, so a caller that needs
    per-cadence data aligned with the trimmed time array doesn't have to
    duplicate this finite-mask/reorder/segment-selection logic itself —
    confirmed necessary in practice: centroid_proxy computed against the
    *untrimmed* mom_centr1/2 arrays crashed with a length mismatch (or,
    silently, could have misaligned points to the wrong phase) whenever
    this trimming actually dropped cadences.

    Note: only time/flux need to be finite for a cadence to be kept —
    `co_arrays` (e.g. centroid columns, which are NaN for every cadence on
    light curves downloaded before that column existed) are along for the
    ride and may still contain NaNs on the cadences that are kept; a
    co_array being all-NaN must never drop otherwise-good flux points.

    Returns
    -------
    (time, flux) normally, or (time, flux, *co_arrays) if any were passed —
    always the same number of arrays as were given.
    """
    finite = np.isfinite(time) & np.isfinite(flux)
    arrays = [time[finite], flux[finite]] + [a[finite] for a in co_arrays]
    t = arrays[0]
    if len(t) < 2:
        return tuple(arrays) if co_arrays else (arrays[0], arrays[1])
    if np.any(np.diff(t) < 0):          # stitched sectors can arrive out of order
        order = np.argsort(t, kind="stable")
        arrays = [a[order] for a in arrays]
        t = arrays[0]
    cuts = np.where(np.diff(t) > max_gap)[0] + 1
    if len(cuts) == 0:
        return tuple(arrays) if co_arrays else (arrays[0], arrays[1])
    edges = np.concatenate(([0], cuts, [len(t)]))
    i = int(np.argmax(np.diff(edges)))
    sl = slice(edges[i], edges[i + 1])
    arrays = [a[sl] for a in arrays]
    return tuple(arrays) if co_arrays else (arrays[0], arrays[1])


def _bin_lightcurve(t: np.ndarray, f: np.ndarray, bin_min: float = TLS_BIN_MIN):
    k = np.floor((t - t[0]) / (bin_min / 1440.0)).astype(np.int64)
    n = np.bincount(k)
    keep = n > 0
    return np.bincount(k, t)[keep] / n[keep], np.bincount(k, f)[keep] / n[keep]


def run_tls(time: np.ndarray, flat_flux: np.ndarray) -> dict:
    """
    Run Transit Least Squares periodogram and return raw TLS results dict.
    Returns empty dict if TLS fails, signal below threshold, or times out.

    TLS is single-threaded per call. The outer joblib/Parallel worker pool
    is already CPU-limited, so we do not need per-TLS threading.
    """
    from transitleastsquares import transitleastsquares as TLS

    finite = np.isfinite(time) & np.isfinite(flat_flux)
    t = time[finite]
    f = flat_flux[finite]

    if len(t) < 200:
        return {}
    t, f = _bin_lightcurve(t, f)      # search on 10-min bins (speed); features keep full res
    if len(t) < 100:
        return {}

    try:
        model   = TLS(t, f)
        results = model.power(
            period_min          = TLS_PERIOD_MIN,
            period_max          = min(TLS_PERIOD_MAX, (t[-1] - t[0]) / 2),
            oversampling_factor = TLS_OVERSAMPLING,
            duration_grid_step  = TLS_DURATION_STEP,
            use_threads         = 2,
            show_progress_bar   = False,
        )
        return results
    except Exception as e:
        log.debug("  TLS failed: %s", e)
        return {}


def extract_geometry_features(tls_results: dict, baseline: float) -> dict:
    """Category 1 & 2 — Transit geometry and signal quality features from TLS."""
    _empty = {
        "period": np.nan, "depth": np.nan, "duration_hr": np.nan,
        "transit_count": np.nan, "SDE": np.nan, "SNR": np.nan,
        "FAP": np.nan, "phase_coverage": np.nan, "t0": np.nan,
        "rp_rs": np.nan, "depth_ppm": np.nan,
    }
    if not tls_results or not hasattr(tls_results, "period"):
        return _empty

    r             = tls_results
    period        = float(r.period)
    if not np.isfinite(period) or period <= 0:      # TLS returned no usable peak
        return _empty
    # NOTE: transitleastsquares' results.depth is the flux LEVEL at the
    # transit bottom (≈1 for a shallow transit), not the fractional dip —
    # confirmed against the library's own source (main.py: `fractional_transit(
    # ..., depth=1-depth, ...)` and `rp_rs_from_depth(depth=1-depth, ...)`
    # both convert it before using it as an actual depth). Using r.depth
    # directly here produced depth_ppm values of ~990,000-999,900 for every
    # candidate regardless of the true transit depth — verified against
    # synthetic injections with known depths of a few hundred to ~17,000 ppm.
    depth         = 1.0 - float(r.depth)
    duration_hr   = float(r.duration) * 24.0
    tc            = float(r.transit_count) if hasattr(r, "transit_count") else np.nan
    transit_count = int(tc) if np.isfinite(tc) else int(baseline / period)
    SDE           = float(r.SDE)
    SNR           = float(r.snr)  if hasattr(r, "snr")  else np.nan
    FAP           = float(r.FAP)  if hasattr(r, "FAP")  else np.nan
    t0            = float(r.T0)   if hasattr(r, "T0")   else np.nan
    # TLS already computes rp_rs correctly from the corrected depth internally
    # (via rp_rs_from_depth) — use it directly rather than re-deriving from
    # our own (previously wrong) depth value.
    if hasattr(r, "rp_rs") and np.isfinite(r.rp_rs):
        rp_rs = float(r.rp_rs)
    else:
        rp_rs = float(np.sqrt(depth)) if depth > 0 else np.nan
    phase_cov     = float(r.duty_cycle) if hasattr(r, "duty_cycle") \
                    else (duration_hr / 24.0) / period

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
        "phase_coverage": phase_cov,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Candidate selection (Implementation_Plan.md §4 Stage 3, steps 4-5, 8)
# ─────────────────────────────────────────────────────────────────────────────

SDE_MIN = 7.0     # candidate threshold (TLS_SDE_THRESHOLD above is the looser feature-extraction gate)
FAP_MAX = 0.01


def detect_candidate(time: np.ndarray, flat_flux: np.ndarray, tic_id: str = "") -> dict:
    """Run TLS on one detrended light curve and decide whether it is a candidate."""
    finite = np.isfinite(time) & np.isfinite(flat_flux)
    baseline = float(time[finite][-1] - time[finite][0]) if finite.sum() > 1 else 0.0
    feats = extract_geometry_features(run_tls(time, flat_flux), baseline)
    sde, fap = feats["SDE"], feats["FAP"]
    feats["is_candidate"] = bool(np.isfinite(sde) and sde >= SDE_MIN
                                 and (not np.isfinite(fap) or fap < FAP_MAX))
    return {"tic_id": tic_id, **feats}


def detect_batch(npz_dir: Path, out_csv: Path) -> pd.DataFrame:
    """Run detection over every detrended .npz in npz_dir and save the candidate table."""
    rows = []
    for f in sorted(Path(npz_dir).glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        rows.append(detect_candidate(d["time"].astype(float), d["flat_flux"].astype(float), f.stem))
    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("Detection: %d targets, %d candidates → %s",
             len(df), int(df["is_candidate"].sum()) if len(df) else 0, out_csv)
    return df
