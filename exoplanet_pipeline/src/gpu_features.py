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
# GPU MORPHOLOGICAL FEATURES (flat-bottom, ingress/egress, secondary)
# ─────────────────────────────────────────────────────────────────────────────

def gpu_morphological_bins(
    phase_arr:   np.ndarray,
    binned_flux: np.ndarray,
    dur_phase:   float,
    primary_depth: Optional[float] = None,
    secondary_phase: float = 0.5,
) -> dict:
    """
    Compute morphological features from pre-binned phase-folded flux.
    GPU-accelerated via CuPy when available.

    Parameters
    ----------
    phase_arr       : bin centre phases in [-0.5, 0.5]
    binned_flux     : mean flux per bin
    dur_phase       : transit duration in phase units
    primary_depth   : depth of primary transit (for secondary_ratio)
    secondary_phase : phase of secondary eclipse to check (default 0.5)

    Returns
    -------
    dict with: flat_bottom_score, phase_folded_std, ingress_egress_asym,
               secondary_depth, secondary_ratio
    """
    if _CUPY_AVAILABLE:
        try:
            return _gpu_morphological_cupy(
                phase_arr, binned_flux, dur_phase,
                primary_depth, secondary_phase
            )
        except Exception as e:
            log.debug("[GPU] morphological fallback to CPU: %s", e)
    return _cpu_morphological(
        phase_arr, binned_flux, dur_phase,
        primary_depth, secondary_phase
    )


def _gpu_morphological_cupy(
    phase_arr: np.ndarray,
    binned_flux: np.ndarray,
    dur_phase: float,
    primary_depth: Optional[float],
    secondary_phase: float,
) -> dict:
    import cupy as cp

    pa  = cp.asarray(phase_arr,   dtype=cp.float64)
    bf  = cp.asarray(binned_flux, dtype=cp.float64)

    in_transit  = cp.abs(pa) < (dur_phase / 2)
    out_transit = cp.abs(pa) > (dur_phase * 1.5)

    transit_flux = bf[in_transit]
    oot_flux     = bf[out_transit]

    # Flat bottom score
    std_in  = float(cp.nanstd(transit_flux).get())  if transit_flux.size > 2 else np.nan
    std_out = float(cp.nanstd(oot_flux).get())      if oot_flux.size > 2     else np.nan
    fbs = std_in / std_out if (std_out and std_out > 0) else np.nan

    # Phase folded std
    pf_std = float(cp.nanstd(bf).get())

    # Ingress / egress asymmetry (compare left half vs right half of transit window)
    half         = len(phase_arr) // 2
    win          = int(half * dur_phase)
    ingress_bins = bf[half - win : half]
    egress_bins  = bf[half : half + win]
    if ingress_bins.size > 0 and egress_bins.size > 0:
        iea = float((cp.nanmean(ingress_bins) - cp.nanmean(egress_bins)).get())
    else:
        iea = np.nan

    # Secondary eclipse at secondary_phase
    sec_mask = cp.abs(pa - secondary_phase) < (dur_phase / 2)
    if sec_mask.any() and not cp.isnan(bf[sec_mask]).all():
        sec_depth = 1.0 - float(cp.nanmean(bf[sec_mask]).get())
        sec_depth = max(sec_depth, 0.0)
        sec_ratio = (sec_depth / primary_depth
                     if (primary_depth and primary_depth > 0) else np.nan)
    else:
        sec_depth = np.nan
        sec_ratio = np.nan

    return {
        "flat_bottom_score":  fbs,
        "phase_folded_std":   pf_std,
        "ingress_egress_asym": iea,
        "secondary_depth":    sec_depth,
        "secondary_ratio":    sec_ratio,
    }


def _cpu_morphological(
    phase_arr: np.ndarray,
    binned_flux: np.ndarray,
    dur_phase: float,
    primary_depth: Optional[float],
    secondary_phase: float,
) -> dict:
    in_transit  = np.abs(phase_arr) < (dur_phase / 2)
    out_transit = np.abs(phase_arr) > (dur_phase * 1.5)

    transit_flux = binned_flux[in_transit]
    oot_flux     = binned_flux[out_transit]

    std_in  = np.nanstd(transit_flux) if transit_flux.size > 2 else np.nan
    std_out = np.nanstd(oot_flux)     if oot_flux.size > 2     else np.nan
    fbs = std_in / std_out if (std_out and std_out > 0) else np.nan
    pf_std = float(np.nanstd(binned_flux))

    half     = len(phase_arr) // 2
    win      = int(half * dur_phase)
    ing_bins = binned_flux[half - win : half]
    egr_bins = binned_flux[half : half + win]
    iea = float(np.nanmean(ing_bins) - np.nanmean(egr_bins)) \
          if (len(ing_bins) > 0 and len(egr_bins) > 0) else np.nan

    sec_mask = np.abs(phase_arr - secondary_phase) < (dur_phase / 2)
    if sec_mask.sum() > 0 and not np.isnan(binned_flux[sec_mask]).all():
        sec_depth = max(1.0 - float(np.nanmean(binned_flux[sec_mask])), 0.0)
        sec_ratio = (sec_depth / primary_depth
                     if (primary_depth and primary_depth > 0) else np.nan)
    else:
        sec_depth = sec_ratio = np.nan

    return {
        "flat_bottom_score":   fbs,
        "phase_folded_std":    pf_std,
        "ingress_egress_asym": iea,
        "secondary_depth":     sec_depth,
        "secondary_ratio":     sec_ratio,
    }


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
    Compute mean odd and even transit depths.
    This remains CPU-side since it iterates over individual epochs.
    """
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
        if mask.sum() < 3:
            continue
        depth_k = 1.0 - float(np.nanmean(f_ok[mask]))
        (even_depths if k % 2 == 0 else odd_depths).append(depth_k)

    odd_d  = float(np.nanmean(odd_depths))  if odd_depths  else np.nan
    even_d = float(np.nanmean(even_depths)) if even_depths else np.nan
    ratio  = float(odd_d / even_d) if (odd_d and even_d and even_d != 0) else np.nan

    return {"odd_depth": odd_d, "even_depth": even_d, "odd_even_ratio": ratio}


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
