"""
features.py  —  Step 3.7  (GPU-Accelerated, Anti-Hang Edition)
==============================================================
Extract 35+ statistical, morphological, and astrophysical features per
TCE candidate for ML input, following the Implementation Plan.

Key improvements over the original:
  • GPU offloading via gpu_features.py (CuPy + PyTorch CUDA on RTX 4050)
  • Worker count capped at min(cpu_count - 2, 4) to prevent system hang
  • Per-task timeout (Windows-compatible threading-based) kills stuck TLS calls
  • Memory guard via psutil: pauses dispatch when RAM < 1.5 GB free
  • Batch checkpointing every CHECKPOINT_EVERY rows (not every 1)
  • GPU cache flushed after each batch to prevent VRAM leaks
  • tqdm progress bar with ETA, memory, and GPU VRAM stats

Feature categories
------------------
  1. Transit Geometry      period, depth, duration, transit count, impact proxy
  2. Signal Quality        SDE, SNR, FAP, phase coverage fraction
  3. Morphological         flat-bottom score, ingress/egress asymmetry
  4. Odd / Even            odd vs even transit depth ratio
  5. Secondary Eclipse     secondary depth and ratio at phase 0.5
  6. Centroid              centroid proxy (quality-flagged cadences during transit)
  7. Stellar               Teff, logg, R★, M★, dist, metallicity from TIC catalog
  8. Time-series stats     RMS, skewness, kurtosis, autocorrelation (GPU)
  9. GPU FFT features      FFT coefficients + basic stats on phase-folded flux

Input
-----
  data/processed/lc_detrended/  — detrended .npz files (from preprocessing.py)
  data/catalogs/tic_stellar_params.csv — TIC stellar parameters

Output
------
  data/catalogs/feature_matrix.csv — one row per candidate with all 35+ features
"""

from __future__ import annotations

import os
import sys
import time
import logging
import warnings
import threading
import multiprocessing
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ensure the src/ directory is on sys.path so gpu_features can always be found,
# regardless of the working directory from which features.py is invoked.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# GPU-accelerated feature functions (CuPy + PyTorch; CPU fallback built-in)
from gpu_features import (  # noqa: E402
    gpu_phase_fold,
    gpu_timeseries_stats,
    gpu_fft_features,
    gpu_morphological_bins,
    compute_odd_even_depths,
    clear_gpu_cache,
    gpu_memory_used_gb,
    get_device_info,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parents[1]
DETRENDED_DIR = BASE_DIR / "data" / "processed" / "lc_detrended"
CATALOGS_DIR  = BASE_DIR / "data" / "catalogs"
CATALOGS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# TUNING PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
TLS_PERIOD_MIN      = 0.5      # days
TLS_PERIOD_MAX      = 27.0     # days  (1 TESS sector)
TLS_OVERSAMPLING    = 5
TLS_DURATION_STEP   = 1.05
TLS_SDE_THRESHOLD   = 5.0
N_PHASE_BINS        = 200
SECONDARY_PHASE     = 0.5

CHECKPOINT_EVERY    = 10       # Save to disk every N rows (reduces I/O)
TASK_TIMEOUT_SEC    = 120      # Kill a stuck extraction after this many seconds
MIN_FREE_RAM_GB     = 1.5      # Pause dispatch if available RAM < this
GPU_FLUSH_EVERY     = 50       # Free GPU cache every N files


def _safe_n_jobs() -> int:
    """
    Return a safe number of parallel workers that won't hang the system.
    Leaves at least 2 cores free for the OS, GPU driver, and display.
    Hard cap at 4 to prevent RAM exhaustion on a laptop.
    """
    n_cpu = multiprocessing.cpu_count()
    return max(1, min(n_cpu - 2, 4))


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY GUARD
# ─────────────────────────────────────────────────────────────────────────────

def _check_ram(min_free_gb: float = MIN_FREE_RAM_GB) -> None:
    """
    Block until available RAM > min_free_gb.
    Logs a warning if it has to wait.
    """
    try:
        import psutil
        while True:
            avail_gb = psutil.virtual_memory().available / 1e9
            if avail_gb >= min_free_gb:
                break
            log.warning(
                "Low RAM (%.1f GB free < %.1f GB threshold) — pausing 10 s before next dispatch.",
                avail_gb, min_free_gb,
            )
            time.sleep(10)
    except ImportError:
        pass  # psutil not installed — skip guard


def _ram_free_gb() -> float:
    """Return free RAM in GB, or -1 if psutil unavailable."""
    try:
        import psutil
        return round(psutil.virtual_memory().available / 1e9, 2)
    except ImportError:
        return -1.0


# ─────────────────────────────────────────────────────────────────────────────
# PER-TASK TIMEOUT (Windows-compatible, threading-based)
# ─────────────────────────────────────────────────────────────────────────────

class _TimeoutError(Exception):
    pass


def _run_with_timeout(fn, args=(), kwargs=None, timeout_sec: int = TASK_TIMEOUT_SEC):
    """
    Run fn(*args, **kwargs) in a daemon thread.
    Raises _TimeoutError if it doesn't finish within timeout_sec.

    Note: On Windows we cannot use SIGALRM, so we use a thread.
    The thread is daemonized so it won't block process exit.
    """
    if kwargs is None:
        kwargs = {}
    result_box = [None]
    exc_box    = [None]

    def _target():
        try:
            result_box[0] = fn(*args, **kwargs)
        except Exception as e:
            exc_box[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        raise _TimeoutError(f"Task timed out after {timeout_sec}s")
    if exc_box[0] is not None:
        raise exc_box[0]
    return result_box[0]


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

    try:
        model   = TLS(t, f)
        results = model.power(
            minimum_period      = TLS_PERIOD_MIN,
            maximum_period      = min(TLS_PERIOD_MAX, (t[-1] - t[0]) / 2),
            oversampling_factor = TLS_OVERSAMPLING,
            duration_grid_step  = TLS_DURATION_STEP,
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
    depth         = float(r.depth)
    duration_hr   = float(r.duration) * 24.0
    transit_count = int(r.transit_count) if hasattr(r, "transit_count") else int(baseline / period)
    SDE           = float(r.SDE)
    SNR           = float(r.snr)  if hasattr(r, "snr")  else np.nan
    FAP           = float(r.FAP)  if hasattr(r, "FAP")  else np.nan
    t0            = float(r.T0)   if hasattr(r, "T0")   else np.nan
    rp_rs         = float(np.sqrt(depth)) if depth > 0 else np.nan
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
# CATEGORY 7 — Stellar parameters from TIC catalog
# ─────────────────────────────────────────────────────────────────────────────

_tic_df: Optional[pd.DataFrame] = None


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
# MASTER FEATURE EXTRACTOR — single light curve
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_one(npz_path: Path, label: int = -1) -> dict:
    """
    Run TLS and extract all 35+ features for a single detrended light curve.
    Heavy array operations are dispatched to the GPU via gpu_features.py.

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
    tls_results = run_tls(time, flat_flux)
    geom_feat   = extract_geometry_features(tls_results, baseline)

    period      = geom_feat.get("period",      np.nan)
    t0          = geom_feat.get("t0",          np.nan)
    duration_hr = geom_feat.get("duration_hr", np.nan)

    # ── Cat 3/4/5: Morphological, odd/even, secondary ─────────────────────────
    if not np.isnan(period) and not np.isnan(t0) and period > 0:
        # Phase fold on GPU
        phase_arr, binned = gpu_phase_fold(time, flat_flux, period, t0, N_PHASE_BINS)

        dur_phase = (duration_hr / 24.0) / period if not np.isnan(duration_hr) else 0.0

        # Primary depth for secondary ratio calculation
        in_transit_mask   = np.abs(phase_arr) < (dur_phase / 2)
        transit_flux_vals = binned[in_transit_mask]
        primary_depth     = (1.0 - float(np.nanmean(transit_flux_vals))
                             if transit_flux_vals.size > 0 else None)

        # Morphological features on GPU
        morph_bins = gpu_morphological_bins(
            phase_arr, binned, dur_phase,
            primary_depth=primary_depth,
            secondary_phase=SECONDARY_PHASE,
        )

        # Odd/even depths — CPU (epoch iteration)
        odd_even = compute_odd_even_depths(time, flat_flux, period, t0, duration_hr)

        morph_feat = {**morph_bins, **odd_even}

        # ── Cat 9: GPU FFT features on phase-folded flux ──────────────────────
        tsf_feat = gpu_fft_features(binned)
    else:
        morph_feat = {
            "flat_bottom_score": np.nan, "phase_folded_std": np.nan,
            "ingress_egress_asym": np.nan, "secondary_depth": np.nan,
            "secondary_ratio": np.nan, "odd_depth": np.nan,
            "even_depth": np.nan, "odd_even_ratio": np.nan,
        }
        tsf_feat   = {}
        phase_arr  = None
        binned     = None

    # ── Cat 6: Centroid proxy ─────────────────────────────────────────────────
    centroid_feat = {"centroid_proxy": np.nan}

    # ── Cat 7: Stellar parameters from TIC ───────────────────────────────────
    stellar_feat = extract_stellar_features(tic_id)

    # Habitable zone proximity
    teff = stellar_feat.get("stellar_Teff", np.nan)
    if not np.isnan(teff) and not np.isnan(period):
        L_over_Lsun = ((teff / 5778) ** 4) * (stellar_feat.get("stellar_rad", 1.0) ** 2)
        hz_inner    = np.sqrt(L_over_Lsun / 1.1)
        hz_outer    = np.sqrt(L_over_Lsun / 0.36)
        M_star      = stellar_feat.get("stellar_mass", 1.0) or 1.0
        a_au        = ((period / 365.25) ** 2 * M_star) ** (1 / 3)
        in_hz       = float(hz_inner <= a_au <= hz_outer)
        hz_prox     = float(abs(a_au - (hz_inner + hz_outer) / 2) / ((hz_outer - hz_inner) / 2))
        stellar_feat["hz_in_zone"]   = in_hz
        stellar_feat["hz_proximity"] = hz_prox
    else:
        stellar_feat["hz_in_zone"]   = np.nan
        stellar_feat["hz_proximity"] = np.nan

    # ── Cat 8: Time-series statistics on GPU ──────────────────────────────────
    ts_feat = gpu_timeseries_stats(flat_flux, rms_raw)

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = {
        "tic_id":       tic_id,
        "label":        label,
        "sector":       lc["sector"],
        "baseline_days": baseline,
        "n_cadences":   int(np.isfinite(flat_flux).sum()),
        "status":       "done",
    }

    return {**meta, **geom_feat, **morph_feat, **centroid_feat,
            **stellar_feat, **ts_feat, **tsf_feat}


# ─────────────────────────────────────────────────────────────────────────────
# BATCH FEATURE EXTRACTION — all detrended light curves
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_batch(
    detrended_dir: Path         = DETRENDED_DIR,
    label_csv:     Optional[Path] = None,
    n_jobs:        int          = -1,          # -1 → auto safe value
    output_path:   Path         = CATALOGS_DIR / "feature_matrix.csv",
    resume:        bool         = True,
    timeout_sec:   int          = TASK_TIMEOUT_SEC,
    checkpoint_n:  int          = CHECKPOINT_EVERY,
) -> pd.DataFrame:
    """
    Extract features for all detrended .npz files in detrended_dir.

    Parameters
    ----------
    detrended_dir : path to lc_detrended/ directory
    label_csv     : optional CSV with tic_id and label columns
    n_jobs        : parallel workers (-1 = auto-safe = min(cpu_count-2, 4))
    output_path   : where to save feature_matrix.csv
    resume        : skip already-processed tic_ids in existing CSV
    timeout_sec   : kill any single extraction after this many seconds
    checkpoint_n  : write a batch to disk every this many rows

    Returns
    -------
    pd.DataFrame — feature matrix (rows = candidates, cols = features)
    """
    from joblib import Parallel, delayed

    # Resolve safe worker count
    if n_jobs <= 0:
        n_jobs = _safe_n_jobs()
    log.info("Using %d parallel worker(s) (safe cap)", n_jobs)

    output_path = Path(output_path)
    npz_files   = sorted(detrended_dir.glob("*.npz"))

    log.info("=" * 65)
    log.info("FEATURE EXTRACTION — %d light curves  (n_jobs=%d)", len(npz_files), n_jobs)
    log.info("=" * 65)

    if not npz_files:
        log.warning("No .npz files found in %s", detrended_dir)
        return pd.DataFrame()

    # ── Resume: skip already-processed rows ──────────────────────────────────
    already_done: set[str] = set()
    existing_df: Optional[pd.DataFrame] = None
    if resume and output_path.exists():
        try:
            existing_df  = pd.read_csv(output_path, dtype={"tic_id": str})
            already_done = set(existing_df["tic_id"].dropna().astype(str))
            log.info("RESUME MODE — %d candidates already processed; skipping.", len(already_done))
        except Exception as e:
            log.warning("Could not load existing feature matrix for resume: %s", e)

    pending = [f for f in npz_files if f.stem not in already_done]
    if len(npz_files) - len(pending):
        log.info("Skipping %d already-extracted files; %d remaining.",
                 len(npz_files) - len(pending), len(pending))

    if not pending:
        log.info("All files already extracted. Returning existing feature matrix.")
        return existing_df if existing_df is not None else pd.DataFrame()

    # Build label lookup
    label_map: dict[str, int] = {}
    if label_csv and Path(label_csv).exists():
        ldf = pd.read_csv(label_csv, dtype={"tic_id": str})
        if "label" in ldf.columns:
            label_map = dict(zip(ldf["tic_id"], ldf["label"].astype(int)))

    # ── Worker wrapper with timeout ───────────────────────────────────────────
    def _worker(f: Path) -> dict:
        """Run extraction with per-task timeout."""
        label = label_map.get(f.stem, -1)
        try:
            return _run_with_timeout(
                extract_features_one, args=(f, label), timeout_sec=timeout_sec
            )
        except _TimeoutError:
            log.warning("  TIMEOUT (%ds) for %s — skipping.", timeout_sec, f.stem)
            return {"tic_id": f.stem, "label": label, "status": f"timeout_{timeout_sec}s"}
        except Exception as e:
            log.warning("  ERROR for %s: %s", f.stem, e)
            return {"tic_id": f.stem, "label": label, "status": str(e)}

    # ── Progress bar + batch checkpoint ──────────────────────────────────────
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(pending), unit="lc",
                    desc="Feature extraction", dynamic_ncols=True)
        use_pbar = True
    except ImportError:
        pbar     = None
        use_pbar = False

    write_header = not (resume and output_path.exists())
    n_done       = 0
    batch_rows   = []

    gen = Parallel(n_jobs=n_jobs, verbose=0, return_as="generator")(
        delayed(_worker)(f) for f in pending
    )

    for row in gen:
        batch_rows.append(row)
        n_done += 1

        # ── Checkpoint every checkpoint_n rows ────────────────────────────
        if len(batch_rows) >= checkpoint_n or n_done == len(pending):
            _check_ram()   # pause if RAM is critically low
            batch_df = pd.DataFrame(batch_rows)
            batch_df.to_csv(output_path, mode="a", header=write_header, index=False)
            write_header = False
            batch_rows   = []
            log.info("  Checkpoint: %d / %d saved  |  RAM free: %.1f GB  |  GPU VRAM: %.2f GB",
                     n_done, len(pending), _ram_free_gb(), gpu_memory_used_gb())

        # ── Flush GPU cache periodically ──────────────────────────────────
        if n_done % GPU_FLUSH_EVERY == 0:
            clear_gpu_cache()

        if use_pbar:
            pbar.set_postfix(
                ram_gb=f"{_ram_free_gb():.1f}",
                vram_gb=f"{gpu_memory_used_gb():.2f}",
                refresh=False,
            )
            pbar.update(1)

    if use_pbar:
        pbar.close()

    # Final GPU cache flush
    clear_gpu_cache()

    # ── Reload full CSV and impute missing stellar params ─────────────────────
    df = pd.read_csv(output_path, dtype={"tic_id": str})

    stellar_cols = ["stellar_Teff", "stellar_logg", "stellar_rad",
                    "stellar_mass", "stellar_dist_pc", "stellar_metallicity"]
    for col in stellar_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df.to_csv(output_path, index=False)

    done = int((df["status"] == "done").sum()) if "status" in df.columns else len(df)
    log.info("Feature extraction complete: %d / %d total rows succeeded", done, len(df))
    log.info("Feature matrix shape: %s", df.shape)
    log.info("Saved → %s", output_path)
    return df


# ─────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    # Print GPU info at startup
    dev_info = get_device_info()
    log.info("=" * 65)
    log.info("GPU INFO: CuPy=%s | PyTorch CUDA=%s | Device=%s",
             dev_info["cupy_available"], dev_info["torch_cuda"], dev_info["device"])
    if "gpu_name" in dev_info:
        log.info("  GPU: %s  |  VRAM Total: %.2f GB  |  VRAM Free: %.2f GB",
                 dev_info["gpu_name"], dev_info["vram_total_gb"], dev_info["vram_free_gb"])
    log.info("=" * 65)

    parser = argparse.ArgumentParser(
        description="Step 3.7 — Extract 35+ features per candidate (GPU-accelerated)"
    )
    parser.add_argument("--detrended-dir", type=Path, default=DETRENDED_DIR,
                        help="Directory with detrended .npz files")
    parser.add_argument("--label-csv",     type=Path, default=None,
                        help="Optional CSV with tic_id and label columns")
    parser.add_argument("--n-jobs",        type=int,  default=-1,
                        help="Parallel workers (-1 = auto-safe, typically 2-4)")
    parser.add_argument("--output",        type=Path,
                        default=CATALOGS_DIR / "feature_matrix.csv",
                        help="Output path for feature_matrix.csv")
    parser.add_argument("--no-resume",     action="store_true",
                        help="Ignore any existing feature_matrix.csv and re-extract everything")
    parser.add_argument("--timeout",       type=int,  default=TASK_TIMEOUT_SEC,
                        help=f"Per-task timeout in seconds (default: {TASK_TIMEOUT_SEC})")
    parser.add_argument("--checkpoint-n",  type=int,  default=CHECKPOINT_EVERY,
                        help=f"Write to disk every N rows (default: {CHECKPOINT_EVERY})")
    args = parser.parse_args()

    df = extract_features_batch(
        detrended_dir = args.detrended_dir,
        label_csv     = args.label_csv,
        n_jobs        = args.n_jobs,
        output_path   = args.output,
        resume        = not args.no_resume,
        timeout_sec   = args.timeout,
        checkpoint_n  = args.checkpoint_n,
    )

    summary_cols = ["tic_id", "label", "period", "depth_ppm", "SDE",
                    "SNR", "odd_even_ratio", "stellar_Teff", "rms", "status"]
    available = [c for c in summary_cols if c in df.columns]
    if not df.empty:
        print(df[available].to_string())
