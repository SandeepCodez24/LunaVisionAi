"""
gpu_features.py  —  GPU-Accelerated Feature Computation
=========================================================
Provides drop-in replacements for the heavy numerical operations in
features.py, targeting an NVIDIA RTX 4050 (CUDA 12.x).

Uses CuPy for array operations and PyTorch for GPU FFT.
Falls back silently to CPU (NumPy / SciPy) if CUDA is not available.

Exported functions
------------------
  gpu_phase_fold()         — Phase-fold + bin a light curve on GPU
  gpu_timeseries_stats()   — RMS, skewness, kurtosis, autocorrelation on GPU
  gpu_fft_features()       — FFT-based spectral features via torch.fft on GPU
  gpu_morphological_bins() — Flat-bottom, ingress/egress, secondary via GPU
  get_device_info()        — Returns dict with CUDA availability and device name
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CUDA AVAILABILITY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_CUPY_AVAILABLE  = False
_TORCH_AVAILABLE = False
_CUDA_DEVICE     = "cpu"

# Safety net: suppress CuPy NVRTC guard for CUDA toolkit < 12
# (driver 591 supports CUDA 12 at runtime even if nvcc toolkit is 11.8)
import os as _os
_os.environ.setdefault("CCCL_IGNORE_DEPRECATED_CUDA_BELOW_12", "1")

try:
    import cupy as cp
    if cp.cuda.is_available():
        _CUPY_AVAILABLE = True
        _dev_id = cp.cuda.runtime.getDevice()
        _CUDA_DEVICE = f"cuda:{_dev_id}"
        try:
            _dev_name = cp.cuda.Device(_dev_id).name
        except Exception:
            _dev_name = f"CUDA device {_dev_id}"
        log.info("[GPU] CuPy available — device: %s", _dev_name)
    else:
        log.info("[GPU] CuPy installed but no CUDA device found — using CPU fallback.")
except ImportError:
    log.info("[GPU] CuPy not installed — using CPU fallback (install with: pip install cupy-cuda12x).")

try:
    import torch
    if torch.cuda.is_available():
        _TORCH_AVAILABLE = True
        log.info("[GPU] PyTorch CUDA available — device: %s", torch.cuda.get_device_name(0))
    else:
        log.info("[GPU] PyTorch installed but CUDA not available — FFT will use CPU.")
except ImportError:
    log.info("[GPU] PyTorch not installed — FFT features will use NumPy fallback.")


def get_device_info() -> dict:
    """Return a summary dict of GPU availability."""
    info = {
        "cupy_available":  _CUPY_AVAILABLE,
        "torch_cuda":      _TORCH_AVAILABLE,
        "device":          _CUDA_DEVICE,
    }
    if _TORCH_AVAILABLE:
        import torch
        info["gpu_name"]   = torch.cuda.get_device_name(0)
        info["vram_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
        info["vram_free_gb"]  = round(
            (torch.cuda.get_device_properties(0).total_memory
             - torch.cuda.memory_allocated(0)) / 1e9, 2
        )
    return info


# ─────────────────────────────────────────────────────────────────────────────
# GPU PHASE FOLDING & BINNING
# ─────────────────────────────────────────────────────────────────────────────

def gpu_phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    n_bins: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Phase-fold a light curve and bin it into n_bins equal phase bins.
    Runs on GPU (CuPy) when available, falls back to NumPy.

    Parameters
    ----------
    time    : array of timestamps
    flux    : array of flux values
    period  : orbital period in days
    t0      : transit midpoint time
    n_bins  : number of phase bins

    Returns
    -------
    phase_centres : (n_bins,) array of bin centres in [-0.5, 0.5]
    binned_flux   : (n_bins,) array of mean flux per bin (NaN where empty)
    """
    if _CUPY_AVAILABLE:
        return _gpu_phase_fold_cupy(time, flux, period, t0, n_bins)
    return _cpu_phase_fold(time, flux, period, t0, n_bins)


def _gpu_phase_fold_cupy(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CuPy implementation of phase folding.

    Binning is done with cp.bincount (sum-per-bin / count-per-bin) instead of
    a per-bin Python loop. The old loop launched ~n_bins tiny CUDA kernels
    (a boolean-mask + reduction each) per light curve, so wall-clock time was
    dominated by kernel-launch overhead rather than actual compute — this
    collapses that to a fixed handful of vectorised kernel launches regardless
    of n_bins, which is the standard GPU histogram/binning pattern.
    """
    import cupy as cp

    # Transfer to GPU
    t_gpu = cp.asarray(time, dtype=cp.float64)
    f_gpu = cp.asarray(flux, dtype=cp.float64)

    phase = ((t_gpu - t0) % period) / period
    phase = cp.where(phase > 0.5, phase - 1.0, phase)

    finite = cp.isfinite(phase) & cp.isfinite(f_gpu)
    phase  = phase[finite]
    f_gpu  = f_gpu[finite]

    edges   = cp.linspace(-0.5, 0.5, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    if phase.size == 0:
        return cp.asnumpy(centres), np.full(n_bins, np.nan)

    bin_idx = cp.digitize(phase, edges) - 1
    in_range = (bin_idx >= 0) & (bin_idx < n_bins)
    bin_idx  = bin_idx[in_range]
    f_valid  = f_gpu[in_range]

    sums    = cp.bincount(bin_idx, weights=f_valid, minlength=n_bins)
    counts  = cp.bincount(bin_idx, minlength=n_bins)
    binned  = cp.full(n_bins, cp.nan, dtype=cp.float64)
    nonzero = counts > 0
    binned[nonzero] = sums[nonzero] / counts[nonzero]

    # Transfer back to CPU
    return cp.asnumpy(centres), cp.asnumpy(binned)


def _cpu_phase_fold(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    NumPy fallback for phase folding. Vectorised via np.bincount (sum-per-bin
    / count-per-bin) instead of a per-bin Python loop — same speedup rationale
    as the CuPy path, and it removes a real CPU bottleneck too since this
    function runs once per candidate across every worker process.
    """
    phase = ((time - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)

    finite = np.isfinite(phase) & np.isfinite(flux)
    phase  = phase[finite]
    flux   = flux[finite]

    bins    = np.linspace(-0.5, 0.5, n_bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])

    if phase.size == 0:
        return centres, np.full(n_bins, np.nan)

    bin_idx  = np.digitize(phase, bins) - 1
    in_range = (bin_idx >= 0) & (bin_idx < n_bins)
    bin_idx  = bin_idx[in_range]
    flux_v   = flux[in_range]

    sums    = np.bincount(bin_idx, weights=flux_v, minlength=n_bins)
    counts  = np.bincount(bin_idx, minlength=n_bins)
    binned  = np.full(n_bins, np.nan)
    nonzero = counts > 0
    binned[nonzero] = sums[nonzero] / counts[nonzero]

    return centres, binned


# ─────────────────────────────────────────────────────────────────────────────
# GPU TIME-SERIES STATISTICAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def gpu_timeseries_stats(
    flat_flux: np.ndarray,
    rms_raw: float = np.nan,
) -> dict:
    """
    Compute classical statistical features on the detrended flux array.
    Uses CuPy on GPU when available, otherwise NumPy/SciPy.

    Returns
    -------
    dict with: rms, rms_raw, skewness, kurtosis, autocorr_lag1,
               autocorr_lag10, flux_range, flux_percentile_5,
               flux_percentile_95, above_3sigma_frac, below_3sigma_frac
    """
    nan_result = {k: np.nan for k in [
        "rms", "rms_raw", "skewness", "kurtosis",
        "autocorr_lag1", "autocorr_lag10",
        "flux_range", "flux_percentile_5", "flux_percentile_95",
        "above_3sigma_frac", "below_3sigma_frac",
    ]}

    f_np = flat_flux[np.isfinite(flat_flux)]
    if len(f_np) < 10:
        return nan_result

    if _CUPY_AVAILABLE:
        try:
            return _gpu_timeseries_stats_cupy(f_np, rms_raw)
        except Exception as e:
            log.debug("[GPU] timeseries_stats fallback to CPU: %s", e)

    return _cpu_timeseries_stats(f_np, rms_raw)


def _gpu_timeseries_stats_cupy(f_np: np.ndarray, rms_raw: float) -> dict:
    """CuPy implementation of time-series stats."""
    import cupy as cp

    f = cp.asarray(f_np, dtype=cp.float64)

    rms      = float(cp.std(f).get())
    mean_f   = float(cp.mean(f).get())

    # Skewness and kurtosis via moments
    n        = len(f_np)
    diff     = f - mean_f
    sigma    = float(cp.std(f).get()) or 1e-9

    skewness = float((cp.mean(diff ** 3) / (sigma ** 3)).get())
    kurtosis = float((cp.mean(diff ** 4) / (sigma ** 4)).get()) - 3.0

    # Autocorrelation at lag 1 and lag 10
    def _autocorr_gpu(x: "cp.ndarray", lag: int) -> float:
        if len(x) <= lag:
            return float("nan")
        x1 = x[:-lag] - cp.mean(x[:-lag])
        x2 = x[lag:]  - cp.mean(x[lag:])
        num = float(cp.mean(x1 * x2).get())
        den = float((cp.std(x[:-lag]) * cp.std(x[lag:])).get())
        return num / den if den > 0 else float("nan")

    ac1  = _autocorr_gpu(f, 1)
    ac10 = _autocorr_gpu(f, 10)

    sigma_val = rms if rms > 0 else 1e-9
    above = float(cp.mean(f > 1 + 3 * sigma_val).get())
    below = float(cp.mean(f < 1 - 3 * sigma_val).get())

    p5, p95 = float(cp.percentile(f,  5).get()), float(cp.percentile(f, 95).get())
    f_range = float((cp.max(f) - cp.min(f)).get())

    return {
        "rms":                rms,
        "rms_raw":            rms_raw,
        "skewness":           skewness,
        "kurtosis":           kurtosis,
        "autocorr_lag1":      ac1,
        "autocorr_lag10":     ac10,
        "flux_range":         f_range,
        "flux_percentile_5":  p5,
        "flux_percentile_95": p95,
        "above_3sigma_frac":  above,
        "below_3sigma_frac":  below,
    }


def _cpu_timeseries_stats(f: np.ndarray, rms_raw: float) -> dict:
    """NumPy/SciPy CPU fallback for time-series stats."""
    rms      = float(np.std(f))
    skewness = float(sp_stats.skew(f))
    kurtosis = float(sp_stats.kurtosis(f))

    def autocorr(x, lag=1):
        if len(x) <= lag:
            return np.nan
        return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])

    sigma = rms if rms > 0 else 1e-9
    return {
        "rms":                rms,
        "rms_raw":            rms_raw,
        "skewness":           skewness,
        "kurtosis":           kurtosis,
        "autocorr_lag1":      autocorr(f, 1),
        "autocorr_lag10":     autocorr(f, 10),
        "flux_range":         float(np.ptp(f)),
        "flux_percentile_5":  float(np.percentile(f, 5)),
        "flux_percentile_95": float(np.percentile(f, 95)),
        "above_3sigma_frac":  float(np.mean(f > 1 + 3 * sigma)),
        "below_3sigma_frac":  float(np.mean(f < 1 - 3 * sigma)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GPU FFT FEATURES  (replaces tsfresh MinimalFCParameters on phase-folded flux)
# ─────────────────────────────────────────────────────────────────────────────

_N_FFT_COEFF = 10   # Number of FFT coefficients to extract

def gpu_fft_features(
    binned_flux: np.ndarray,
    n_coeff: int = _N_FFT_COEFF,
) -> dict:
    """
    Extract FFT-based spectral features from the phase-folded, binned flux.
    Runs on GPU (torch.fft) when available, falls back to NumPy FFT.

    This replaces tsfresh's MinimalFCParameters with a faster GPU equivalent.

    Features extracted
    ------------------
    tsf_mean, tsf_variance, tsf_abs_energy, tsf_mean_abs_change,
    tsf_maximum, tsf_minimum, tsf_median,
    tsf_fft_coeff_{0..n_coeff-1}_real,
    tsf_fft_coeff_{0..n_coeff-1}_abs

    Parameters
    ----------
    binned_flux : (N,) phase-folded binned flux array
    n_coeff     : number of FFT coefficients to retain

    Returns
    -------
    dict of feature_name -> float
    """
    flux = binned_flux.copy()
    nan_mask = ~np.isfinite(flux)
    if nan_mask.all():
        return {}
    flux[nan_mask] = np.nanmean(flux[np.isfinite(flux)])

    if _TORCH_AVAILABLE:
        try:
            return _gpu_fft_features_torch(flux, n_coeff)
        except Exception as e:
            log.debug("[GPU] fft_features fallback to CPU: %s", e)

    return _cpu_fft_features(flux, n_coeff)


def _gpu_fft_features_torch(flux: np.ndarray, n_coeff: int) -> dict:
    """PyTorch CUDA FFT implementation."""
    import torch

    t = torch.tensor(flux, dtype=torch.float32, device="cuda")

    # Basic statistics
    mean_v    = float(t.mean().cpu())
    var_v     = float(t.var().cpu())
    abs_en    = float((t ** 2).sum().cpu())
    mac       = float(torch.abs(torch.diff(t)).mean().cpu())
    max_v     = float(t.max().cpu())
    min_v     = float(t.min().cpu())
    median_v  = float(t.median().cpu())

    # FFT on GPU
    fft_out   = torch.fft.rfft(t)
    fft_real  = fft_out.real.cpu().numpy()
    fft_abs   = fft_out.abs().cpu().numpy()

    feat = {
        "tsf_mean":            mean_v,
        "tsf_variance":        var_v,
        "tsf_abs_energy":      abs_en,
        "tsf_mean_abs_change": mac,
        "tsf_maximum":         max_v,
        "tsf_minimum":         min_v,
        "tsf_median":          median_v,
    }

    for i in range(min(n_coeff, len(fft_real))):
        feat[f"tsf_fft_coeff_{i}_real"] = float(fft_real[i])
        feat[f"tsf_fft_coeff_{i}_abs"]  = float(fft_abs[i])

    return feat


def _cpu_fft_features(flux: np.ndarray, n_coeff: int) -> dict:
    """NumPy FFT CPU fallback."""
    fft_out  = np.fft.rfft(flux)
    fft_real = fft_out.real
    fft_abs  = np.abs(fft_out)

    feat = {
        "tsf_mean":            float(np.mean(flux)),
        "tsf_variance":        float(np.var(flux)),
        "tsf_abs_energy":      float(np.sum(flux ** 2)),
        "tsf_mean_abs_change": float(np.mean(np.abs(np.diff(flux)))),
        "tsf_maximum":         float(np.max(flux)),
        "tsf_minimum":         float(np.min(flux)),
        "tsf_median":          float(np.median(flux)),
    }

    for i in range(min(n_coeff, len(fft_real))):
        feat[f"tsf_fft_coeff_{i}_real"] = float(fft_real[i])
        feat[f"tsf_fft_coeff_{i}_abs"]  = float(fft_abs[i])

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# TRANSIT SHAPE FEATURES (flat-bottom vs. V-shape, ingress/egress, secondary)
# ─────────────────────────────────────────────────────────────────────────────

def compute_shape_features(
    time:            np.ndarray,
    flux:            np.ndarray,
    period:          float,
    t0:              float,
    dur_phase:       float,
    secondary_phase: float = 0.5,
    min_pts:         int = 8,
) -> dict:
    """
    Transit-shape features computed directly on phase-folded points, not on
    a fixed 200-bin global fold.

    Why this replaces the old bin-based version: the global fold bins the
    entire [-0.5, 0.5] phase range into N_PHASE_BINS (200) equal-width bins.
    For a typical multi-day period and a few-hour transit, dur_phase is
    ~0.01-0.05, so only 2-10 of those 200 bins fall inside the transit —
    far too few to resolve its *shape*, which is the whole point of these
    features. Verified in practice: flat_bottom_score, ingress_egress_asym
    and secondary_ratio all had ROC-AUC 0.50-0.55 against confirmed TFOP
    dispositions on ~1,700 real targets (indistinguishable from noise)
    before this fix. This version instead masks the raw per-point
    phase-folded data — which pools every observed transit, often hundreds
    to thousands of points — directly by phase, and takes robust (median)
    statistics within each zone.

    Shape diagnostic (flat_bottom_score): let u = |phase| / (dur_phase/2),
    so u=0 is transit center and u=1 is the nominal edge (T14/2). A flat-
    bottomed transit (planet, or a fully-eclipsing EB) stays near full
    depth until close to u=1 (ingress/egress is a small fraction of the
    duration); a V-shaped one (grazing eclipse — the classic EB/blend
    false-positive shape) tapers linearly from center to edge. Comparing
    depth at u in [0, 0.35] ("core") against u in [0.55, 0.85] ("shoulder",
    chosen short of the universal edge taper near u=1) separates these:
    shoulder/core is close to 1 for a flat bottom and well below 1
    (~0.3 for an idealised triangular V-shape) for a graze.

    Returns
    -------
    dict: flat_bottom_score, ingress_egress_asym, secondary_depth,
          secondary_ratio  (each NaN if too few points fall in a zone)
    """
    out = {"flat_bottom_score": np.nan, "ingress_egress_asym": np.nan,
           "secondary_depth": np.nan, "secondary_ratio": np.nan}
    if not (np.isfinite(dur_phase) and dur_phase > 0
            and np.isfinite(period) and period > 0 and np.isfinite(t0)):
        return out

    phase = ((time - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    finite = np.isfinite(phase) & np.isfinite(flux)
    phase, flux = phase[finite], flux[finite]
    if phase.size < min_pts * 4:
        return out

    half = dur_phase / 2.0
    ap   = np.abs(phase)

    baseline_mask = (ap > half * 0.8) & (ap < half * 3.0)
    core_mask     = ap < half * 0.35
    if baseline_mask.sum() < min_pts or core_mask.sum() < min_pts:
        return out
    baseline   = float(np.nanmedian(flux[baseline_mask]))
    depth_core = baseline - float(np.nanmedian(flux[core_mask]))
    if not (np.isfinite(depth_core) and depth_core > 0):
        return out

    shoulder_mask = (ap >= half * 0.55) & (ap < half * 0.85)
    if shoulder_mask.sum() >= min_pts:
        depth_shoulder = baseline - float(np.nanmedian(flux[shoulder_mask]))
        out["flat_bottom_score"] = float(np.clip(depth_shoulder / depth_core, 0.0, 2.0))

    left_mask  = (phase >= -half) & (phase < -half * 0.1)
    right_mask = (phase > half * 0.1) & (phase <= half)
    if left_mask.sum() >= min_pts and right_mask.sum() >= min_pts:
        depth_l = baseline - float(np.nanmedian(flux[left_mask]))
        depth_r = baseline - float(np.nanmedian(flux[right_mask]))
        out["ingress_egress_asym"] = float((depth_l - depth_r) / depth_core)

    # Circular distance from secondary_phase (phase wraps at +/-0.5).
    d = phase - secondary_phase
    d = np.abs(np.mod(d + 0.5, 1.0) - 0.5)
    sec_mask = d < half
    if sec_mask.sum() >= min_pts:
        depth_sec = max(baseline - float(np.nanmedian(flux[sec_mask])), 0.0)
        out["secondary_depth"] = depth_sec
        out["secondary_ratio"] = depth_sec / depth_core

    return out


# ─────────────────────────────────────────────────────────────────────────────
# ODD / EVEN TRANSIT DEPTHS  (CPU — depends on epoch iteration)
# ─────────────────────────────────────────────────────────────────────────────

def compute_odd_even_depths(
    time: np.ndarray,
    flat_flux: np.ndarray,
    period: float,
    t0: float,
    duration_hr: float,
) -> dict:
    """
    Compute median odd and even transit depths.
    This remains CPU-side since it iterates over individual epochs.

    Requires at least MIN_TRANSITS_PER_PARITY valid transits on *each* side
    (odd and even) before returning a ratio — a ratio built from one noisy
    single-transit depth on either side is indistinguishable from real EB
    signal, and was a likely contributor to this feature's ~0.50 AUC
    (essentially random) against confirmed TFOP dispositions. Uses the
    median depth per transit (not the mean) so one bad cadence inside a
    single transit's window doesn't dominate that transit's depth estimate.
    """
    MIN_TRANSITS_PER_PARITY = 2
    MIN_PTS_PER_TRANSIT     = 5

    finite     = np.isfinite(time) & np.isfinite(flat_flux)
    t_ok, f_ok = time[finite], flat_flux[finite]

    transit_epochs = t0 + np.arange(0, (t_ok[-1] - t0) + period, period)
    transit_epochs = transit_epochs[
        (transit_epochs >= t_ok[0]) & (transit_epochs <= t_ok[-1])
    ]

    odd_depths, even_depths = [], []
    half_dur = (duration_hr / 24.0) / 2.0

    for k, epoch in enumerate(transit_epochs):
        mask = np.abs(t_ok - epoch) < half_dur
        if mask.sum() < MIN_PTS_PER_TRANSIT:
            continue
        depth_k = 1.0 - float(np.nanmedian(f_ok[mask]))
        (even_depths if k % 2 == 0 else odd_depths).append(depth_k)

    if len(odd_depths) < MIN_TRANSITS_PER_PARITY or len(even_depths) < MIN_TRANSITS_PER_PARITY:
        return {"odd_depth": np.nan, "even_depth": np.nan, "odd_even_ratio": np.nan}

    odd_d  = float(np.nanmedian(odd_depths))
    even_d = float(np.nanmedian(even_depths))
    ratio  = float(odd_d / even_d) if even_d != 0 else np.nan

    return {"odd_depth": odd_d, "even_depth": even_d, "odd_even_ratio": ratio}


# ─────────────────────────────────────────────────────────────────────────────
# CENTROID OFFSET (background-blend / false-positive check)
# ─────────────────────────────────────────────────────────────────────────────

def compute_centroid_proxy(
    time:      np.ndarray,
    mom_centr1: np.ndarray,
    mom_centr2: np.ndarray,
    period:    float,
    t0:        float,
    dur_phase: float,
    min_pts:   int = 8,
) -> dict:
    """
    Centroid-offset significance: does the star's flux-weighted photocenter
    move during the transit?

    A transit on the target star should not move its own centroid (beyond
    noise). A transit signal actually coming from a fainter background star
    blended into the same aperture — the single most common real false-
    positive mechanism TFOP follow-up rules out with dedicated centroid/
    imaging observations — shifts the *aperture's* photocenter toward that
    background star specifically during the dip. This is the standard
    "difference-image centroid offset" check (as used in the Kepler/TESS
    DV reports and Robovetter), computed here from the MOM_CENTR1/2 columns
    every SPOC light curve FITS already ships — no separate Target Pixel
    File download needed.

    Returns a dimensionless offset-significance: the in-transit vs.
    out-of-transit centroid shift in each axis, divided by that axis's
    out-of-transit scatter (to be comparable across targets of different
    brightness/aperture size), combined in quadrature. ~0 = no detectable
    shift (consistent with an on-target transit); large = the dip's source
    is offset from the target, i.e. likely a blend.

    Returns
    -------
    dict: {"centroid_proxy": float}  (NaN if too few finite centroid points
          or the transit ephemeris is unusable — most light curves
          downloaded before this feature was added have no centroid data
          at all, and will always return NaN here, not an error)
    """
    out = {"centroid_proxy": np.nan}
    if not (np.isfinite(dur_phase) and dur_phase > 0
            and np.isfinite(period) and period > 0 and np.isfinite(t0)):
        return out

    phase = ((time - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    finite = (np.isfinite(phase) & np.isfinite(mom_centr1) & np.isfinite(mom_centr2))
    phase = phase[finite]
    c1, c2 = mom_centr1[finite], mom_centr2[finite]
    if phase.size < min_pts * 4:
        return out

    half = dur_phase / 2.0
    ap   = np.abs(phase)
    in_mask  = ap < half * 0.5
    out_mask = (ap > half * 1.5) & (ap < half * 6.0)
    if in_mask.sum() < min_pts or out_mask.sum() < min_pts:
        return out

    sigma1 = float(np.nanstd(c1[out_mask]))
    sigma2 = float(np.nanstd(c2[out_mask]))
    if not (sigma1 > 0 and sigma2 > 0):
        return out

    shift1 = float(np.nanmedian(c1[in_mask]) - np.nanmedian(c1[out_mask]))
    shift2 = float(np.nanmedian(c2[in_mask]) - np.nanmedian(c2[out_mask]))
    out["centroid_proxy"] = float(np.hypot(shift1 / sigma1, shift2 / sigma2))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# GPU MEMORY MANAGEMENT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def clear_gpu_cache() -> None:
    """Free GPU memory caches after processing a batch."""
    if _CUPY_AVAILABLE:
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    if _TORCH_AVAILABLE:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def gpu_memory_used_gb() -> float:
    """Return currently allocated GPU memory in GB (0.0 if unavailable)."""
    if _TORCH_AVAILABLE:
        try:
            import torch
            return round(torch.cuda.memory_allocated(0) / 1e9, 3)
        except Exception:
            pass
    return 0.0
