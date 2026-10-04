"""
report.py — Stage 6b: diagnostics, vetting, candidate catalogue and per-target PDF.

Reads what fitting.py wrote (`outputs/fits/<tic>.json`, `outputs/posteriors/<tic>.npz`,
`outputs/fit_results.csv`) plus the detrended light curve, and produces

  outputs/plots/<tic>/*.png      the 7 diagnostic plots
  outputs/reports/<tic>.pdf      3-page summary per candidate
  outputs/candidate_catalog.csv  ranked catalogue (+ .fits)

The seven plots
  1 light curve (raw+trend, detrended)   2 TLS periodogram   3 phase fold + model + residuals
  4 MCMC corner                          5 odd/even          6 secondary eclipse
  7 centroid motion

Vetting (`vet`) turns the odd/even, secondary-eclipse, centroid, size and fit-quality
numbers into explicit flags. A candidate is FALSE-POSITIVE LIKELY if a hard flag
trips, CAUTION if only soft flags do, otherwise PASS. The catalogue is ranked by
    rank_score = p_transit x verdict_weight x min(SNR / 10, 1)
so a confident classifier score cannot lift a candidate that failed vetting.

CLI
---
    python src/report.py --tic TIC_16740101
    python src/report.py --all                # every fit in outputs/fits
    python src/report.py --all --no-pdf       # plots + catalogue only
"""
from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from fitting import BASE_DIR, CATALOGS_DIR, OUT_DIR, LIMB_DARK, load_segment, norm_tic

log = logging.getLogger("report")

NAVY, TEAL, ORANGE, GREY, GOLD, RED = "#1f3a5f", "#2a9d8f", "#e76f51", "#8d99ae", "#e9c46a", "#c0392b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "figure.dpi": 110})

# Vetting thresholds
ODD_EVEN_SIGMA   = 3.0
ODD_EVEN_FRAC    = 0.10     # ...and the depths must differ by >= 10 % (else high-SNR hot Jupiters
                            # fail on differences of a few ppm that are red noise / limb darkening)
SECONDARY_SIGMA  = 3.0
SECONDARY_FRAC   = 0.10     # secondary must be >= 10 % of the primary depth to matter
CENTROID_SIGMA   = 3.0
MAX_PLANET_REARTH = 20.0    # ~1.8 R_Jup: larger objects are stars / brown dwarfs
MIN_SNR          = 7.0
GRAZING_B        = 0.9
VERDICT_WEIGHT   = {"PASS": 1.0, "CAUTION": 0.5, "FALSE-POSITIVE LIKELY": 0.1}


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _med(res: dict, k: str) -> float:
    v = res.get(k)
    return float(v["med"]) if isinstance(v, dict) else np.nan


def _pm(res: dict, k: str, fmt: str = "{:.4g}") -> str:
    v = res.get(k)
    if not isinstance(v, dict):
        return "n/a"
    return f"{fmt.format(v['med'])} +{fmt.format(v['hi'])} / -{fmt.format(v['lo'])}"


def _phase_hr(t, tc, P):
    """Hours from the nearest mid-transit."""
    return (((t - tc + 0.5 * P) % P) - 0.5 * P) * 24.0


def _binned(x, y, edges, min_n=3):
    """Median and standard error of y in each x bin -> (centres, median, err)."""
    idx = np.digitize(x, edges) - 1
    c, m, e = [], [], []
    for i in range(len(edges) - 1):
        sel = idx == i
        if sel.sum() >= min_n:
            yy = y[sel]
            c.append(0.5 * (edges[i] + edges[i + 1])); m.append(np.median(yy))
            e.append(1.253 * np.std(yy) / np.sqrt(sel.sum()))
    return np.array(c), np.array(m), np.array(e)


def _model_curve(res: dict, hours: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
    """batman flux versus hours-from-mid-transit, for median (or given) parameters."""
    import batman
    P = _med(res, "period")
    rp, b, a = (theta[0], theta[1], theta[2]) if theta is not None else (_med(res, "rp_rs"), _med(res, "b"), _med(res, "a_rs"))
    pm = batman.TransitParams()
    pm.t0, pm.per, pm.rp, pm.a = 0.0, P, rp, a
    pm.inc = float(np.degrees(np.arccos(min(b / a, 1.0))))
    pm.ecc, pm.w, pm.u, pm.limb_dark = 0.0, 90.0, list(LIMB_DARK), "quadratic"
    return batman.TransitModel(pm, hours / 24.0).light_curve(pm)


def _load_posterior(out_dir: Path, tic: str) -> Optional[np.ndarray]:
    p = out_dir / "posteriors" / f"{tic}.npz"
    return np.load(p)["samples"].astype(float) if p.exists() else None


# ─────────────────────────────────────────────────────────────────────────────
# diagnostics (numbers behind plots 5-7 and the vetting flags)
# ─────────────────────────────────────────────────────────────────────────────

def compute_diagnostics(seg: dict, res: dict) -> dict:
    """Odd/even depths, secondary-eclipse depth and centroid shift, with uncertainties."""
    t, f = seg["time"], seg["flat"]
    tc, P = _med(res, "tc"), _med(res, "period")
    t14 = _med(res, "t14_hr") / 24.0
    out: dict = {}
    ph = _phase_hr(t, tc, P) / 24.0                                   # days from mid-transit
    n = np.round((t - tc) / P).astype(int)

    # baseline & noise in a window around the transit but outside it
    near = (np.abs(ph) > 0.75 * t14) & (np.abs(ph) < 3.0 * t14)
    base = np.nanmedian(f[near]) if near.sum() > 20 else 1.0
    sig = np.nanstd(f[near]) if near.sum() > 20 else np.nanstd(f)

    def depth(mask):
        k = mask.sum()
        if k < 10:
            return np.nan, np.nan, int(k)
        return float(base - np.mean(f[mask])), float(sig / np.sqrt(k)), int(k)

    core = np.abs(ph) < 0.25 * t14                                    # flat bottom
    d_odd, e_odd, n_odd = depth(core & (n % 2 == 1))
    d_even, e_even, n_even = depth(core & (n % 2 == 0))
    out["odd"], out["even"] = (d_odd, e_odd, n_odd), (d_even, e_even, n_even)
    if np.isfinite(d_odd) and np.isfinite(d_even):
        out["odd_even_sigma"] = float(abs(d_odd - d_even) / np.hypot(e_odd, e_even))
        dm = 0.5 * (d_odd + d_even)
        out["odd_even_frac"] = float(abs(d_odd - d_even) / dm) if dm > 0 else np.nan
    else:
        out["odd_even_sigma"], out["odd_even_frac"] = np.nan, np.nan

    # secondary eclipse at phase 0.5
    ph2 = (((t - tc) / P) % 1.0 - 0.5) * P                              # days from phase 0.5
    near2 = (np.abs(ph2) > 0.75 * t14) & (np.abs(ph2) < 3.0 * t14)
    base2 = np.nanmedian(f[near2]) if near2.sum() > 20 else base
    m2 = np.abs(ph2) < 0.25 * t14
    if m2.sum() >= 10:
        d2, e2 = float(base2 - np.mean(f[m2])), float(sig / np.sqrt(m2.sum()))
    else:
        d2, e2 = np.nan, np.nan
    d1 = _med(res, "depth_ppm") * 1e-6
    out["secondary"] = (d2, e2, int(m2.sum()))
    out["secondary_sigma"] = float(d2 / e2) if np.isfinite(d2) and e2 > 0 else np.nan
    out["secondary_frac"] = float(d2 / d1) if np.isfinite(d2) and d1 > 0 else np.nan

    # centroid shift (pixels): in-transit vs local out-of-transit
    c1, c2 = seg["c1"], seg["c2"]
    ok = np.isfinite(c1) & np.isfinite(c2)
    intr = ok & (np.abs(ph) < 0.4 * t14)
    oot = ok & (np.abs(ph) > 0.75 * t14) & (np.abs(ph) < 3.0 * t14)
    if intr.sum() >= 10 and oot.sum() >= 30:
        dx, dy = np.mean(c1[intr]) - np.mean(c1[oot]), np.mean(c2[intr]) - np.mean(c2[oot])
        ex = np.hypot(np.std(c1[oot]) / np.sqrt(intr.sum()), np.std(c1[oot]) / np.sqrt(oot.sum()))
        ey = np.hypot(np.std(c2[oot]) / np.sqrt(intr.sum()), np.std(c2[oot]) / np.sqrt(oot.sum()))
        out["centroid"] = (float(dx), float(dy), float(ex), float(ey))
        out["centroid_sigma"] = float(np.hypot(dx / ex, dy / ey)) if ex > 0 and ey > 0 else np.nan
    else:
        out["centroid"], out["centroid_sigma"] = None, np.nan
    return out


def vet(res: dict, diag: dict) -> dict:
    """Explicit pass/fail checks -> {'flags': [...], 'hard': [...], 'verdict': str, 'checks': [...]}."""
    checks, hard, soft = [], [], []

    def add(name, status, detail, level=None):
        checks.append((name, status, detail))
        if status == "FAIL":
            (hard if level == "hard" else soft).append(f"{name}: {detail}")

    s = diag.get("odd_even_sigma", np.nan)
    if np.isfinite(s):
        fr = diag.get("odd_even_frac", np.nan)
        bad = s > ODD_EVEN_SIGMA and np.isfinite(fr) and fr > ODD_EVEN_FRAC
        add("Odd/even depth", "FAIL" if bad else "PASS",
            f"{s:.1f} sigma, {100 * fr:.1f}% depth difference" + (" (true period may be 2P: eclipsing binary)" if bad else ""), "hard")
    else:
        add("Odd/even depth", "N/A", "too few in-transit points")

    s, frac = diag.get("secondary_sigma", np.nan), diag.get("secondary_frac", np.nan)
    if np.isfinite(s):
        bad = s > SECONDARY_SIGMA and frac > SECONDARY_FRAC
        add("Secondary eclipse", "FAIL" if bad else "PASS",
            f"{s:.1f} sigma, {100 * frac:.0f}% of primary" + (" (companion is self-luminous)" if bad else ""), "hard")
    else:
        add("Secondary eclipse", "N/A", "phase 0.5 not covered")

    s = diag.get("centroid_sigma", np.nan)
    if np.isfinite(s):
        add("Centroid shift", "FAIL" if s > CENTROID_SIGMA else "PASS",
            f"{s:.1f} sigma" + (" (signal may be on a neighbouring star)" if s > CENTROID_SIGMA else ""), "hard")
    else:
        add("Centroid shift", "N/A", "no centroid columns in this light curve")

    rp = _med(res, "rp_rearth")
    if np.isfinite(rp):
        add("Planet size", "FAIL" if rp > MAX_PLANET_REARTH else "PASS",
            f"{rp:.1f} R_earth" + (" (larger than any planet: stellar companion)" if rp > MAX_PLANET_REARTH else ""), "hard")
    else:
        add("Planet size", "N/A", "no stellar radius in the TIC")

    snr = _med(res, "snr")
    add("Signal-to-noise", "FAIL" if snr < MIN_SNR else "PASS", f"SNR = {snr:.1f}", "soft")
    add("Impact parameter", "FAIL" if _med(res, "b") > GRAZING_B else "PASS",
        f"b = {_med(res, 'b'):.2f}" + (" (grazing: radius poorly constrained)" if _med(res, "b") > GRAZING_B else ""), "soft")
    add("Fit convergence", "PASS" if res.get("method") == "mcmc" and res.get("converged") else "FAIL",
        f"{res.get('method')}, N_eff = {res.get('n_eff', float('nan')):.0f}" if res.get("method") == "mcmc"
        else "least-squares fallback (uncertainties are approximate)", "soft")

    verdict = "FALSE-POSITIVE LIKELY" if hard else ("CAUTION" if soft else "PASS")
    return {"flags": hard + soft, "hard": hard, "soft": soft, "verdict": verdict, "checks": checks}


# ─────────────────────────────────────────────────────────────────────────────
# the seven plots (each draws into axes it is given)
# ─────────────────────────────────────────────────────────────────────────────

def _epochs(seg, res):
    tc, P = _med(res, "tc"), _med(res, "period")
    t = seg["time"]
    n = np.arange(np.ceil((t[0] - tc) / P), np.floor((t[-1] - tc) / P) + 1)
    return tc + n * P


def plot_lightcurve(ax_raw, ax_flat, seg, res):
    t = seg["time"]
    ax_raw.plot(t, seg["norm"], ".", ms=1, color=GREY, rasterized=True)
    ax_raw.plot(t, seg["trend"], color=ORANGE, lw=1.2, label="bi-weight trend")
    ax_raw.set_ylabel("Normalised flux"); ax_raw.legend(loc="lower left", fontsize=7, frameon=False)
    ax_raw.set_title("1 · Light curve: raw and trend")
    ax_flat.plot(t, seg["flat"], ".", ms=1, color=TEAL, rasterized=True)
    for e in _epochs(seg, res):
        ax_flat.axvline(e, color=RED, lw=0.6, alpha=0.5)
    ax_flat.set_ylabel("Detrended flux"); ax_flat.set_xlabel("Time (BTJD, days)")
    ax_flat.set_title("detrended; red lines = predicted transits", fontsize=8, fontweight="normal")


def _periodogram(seg, out_dir: Path, tic: str):
    cache = out_dir / "periodograms" / f"{tic}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["periods"], z["power"]
    import contextlib
    import os
    from detection import run_tls
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):     # TLS prints its banner
        r = run_tls(seg["time"], seg["flat"])
    if not r or not hasattr(r, "periods"):
        return None, None
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, periods=np.asarray(r.periods, float), power=np.asarray(r.power, float))
    return np.asarray(r.periods, float), np.asarray(r.power, float)


def plot_periodogram(ax, seg, res, out_dir, tic):
    ax.set_title("2 · TLS periodogram")
    periods, power = _periodogram(seg, out_dir, tic)
    if periods is None:
        ax.text(0.5, 0.5, "TLS periodogram unavailable", ha="center", transform=ax.transAxes); return
    ax.plot(periods, power, color=NAVY, lw=0.8)
    P = _med(res, "tls_period") if "tls_period" in res else _med(res, "period")
    P = res.get("tls_period", P)
    ax.axvline(P, color=ORANGE, lw=1.2, label=f"P = {P:.4f} d")
    for m, ls in [(0.5, ":"), (2.0, ":")]:
        if periods.min() < P * m < periods.max():
            ax.axvline(P * m, color=GREY, lw=0.8, ls=ls)
    ax.set_xlabel("Period (days)"); ax.set_ylabel("SDE"); ax.legend(fontsize=7, frameon=False)


def plot_fold(ax, ax_res, seg, res, samples):
    t, f = seg["time"], seg["flat"]
    tc, P, t14 = _med(res, "tc"), _med(res, "period"), _med(res, "t14_hr")
    h = _phase_hr(t, tc, P)
    lim = max(3.0 * t14, 2.0)
    sel = np.abs(h) < lim
    edges = np.linspace(-lim, lim, 61)
    ax.plot(h[sel], f[sel], ".", ms=1.5, color=GREY, alpha=0.5, rasterized=True)
    c, m, e = _binned(h[sel], f[sel], edges)
    ax.errorbar(c, m, e, fmt="o", ms=3, color=NAVY, lw=0.8, label="binned")
    grid = np.linspace(-lim, lim, 600)
    ax.plot(grid, _model_curve(res, grid), color=ORANGE, lw=1.6, label="median model")
    if samples is not None and len(samples) > 50:
        idx = np.random.default_rng(0).choice(len(samples), 60, replace=False)
        curves = np.array([_model_curve(res, grid, np.array([samples[i, 2], samples[i, 3], np.exp(samples[i, 4])]))
                           for i in idx])
        ax.fill_between(grid, *np.percentile(curves, [2.5, 97.5], axis=0), color=ORANGE, alpha=0.25, lw=0)
    ax.set_ylabel("Flux"); ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title(f"3 · Phase fold at P = {P:.5f} d with fitted transit model")
    ax.set_xlim(-lim, lim); ax.tick_params(labelbottom=False)
    model_at = np.interp(h[sel], grid, _model_curve(res, grid))
    cr, mr, er = _binned(h[sel], f[sel] - model_at, edges)
    ax_res.errorbar(cr, mr * 1e6, er * 1e6, fmt="o", ms=3, color=NAVY, lw=0.8)
    ax_res.axhline(0, color=ORANGE, lw=1)
    ax_res.set_ylabel("Resid. (ppm)"); ax_res.set_xlabel("Hours from mid-transit"); ax_res.set_xlim(-lim, lim)


def plot_corner_image(res, samples) -> Optional[np.ndarray]:
    """Corner plot rendered to an RGB array (so it can go on a PDF page or be saved as PNG)."""
    if samples is None:
        return None
    import corner
    tc0, P0 = _med(res, "tc"), _med(res, "period")
    data = np.column_stack([(samples[:, 0] - tc0) * 1440.0, (samples[:, 1] - P0) * 86400.0,
                            samples[:, 2], samples[:, 3], np.exp(samples[:, 4])])
    labels = ["T$_c$ − %.4f\n(min)" % tc0, "P − %.5f\n(s)" % P0, "Rp/R*", "b", "a/R*"]
    fig = corner.corner(data, labels=labels, quantiles=[0.1587, 0.5, 0.8413], show_titles=True,
                        title_fmt=".4g", title_kwargs={"fontsize": 7}, label_kwargs={"fontsize": 8},
                        color=NAVY, plot_datapoints=False, fill_contours=True, levels=(0.68, 0.95),
                        hist_kwargs={"color": NAVY})
    for a in fig.axes:
        a.tick_params(labelsize=6)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight"); plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def plot_corner(ax, img):
    ax.set_title("4 · MCMC posterior")
    ax.axis("off")
    if img is None:
        ax.text(0.5, 0.5, "No posterior samples", ha="center", transform=ax.transAxes)
    else:
        ax.imshow(img)


def plot_odd_even(ax, seg, res, diag):
    t, f = seg["time"], seg["flat"]
    tc, P, t14 = _med(res, "tc"), _med(res, "period"), _med(res, "t14_hr")
    h = _phase_hr(t, tc, P)
    n = np.round((t - tc) / P).astype(int)
    lim = max(1.5 * t14, 1.5)
    edges = np.linspace(-lim, lim, 31)
    for name, col, sel_par in [("odd", ORANGE, 1), ("even", TEAL, 0)]:
        s = (np.abs(h) < lim) & (n % 2 == sel_par)
        c, m, e = _binned(h[s], f[s], edges, min_n=2)
        if len(c):
            d, de, k = diag[name]
            lab = f"{name}: {1e6 * d:.0f} ± {1e6 * de:.0f} ppm (N={k})" if np.isfinite(d) else name
            ax.errorbar(c, m, e, fmt="o-", ms=3, lw=0.8, color=col, label=lab)
    s = diag.get("odd_even_sigma", np.nan)
    ax.set_title("5 · Odd vs even transits" + (f"  ({s:.1f}σ)" if np.isfinite(s) else ""))
    ax.set_xlabel("Hours from mid-transit"); ax.set_ylabel("Flux"); ax.legend(fontsize=7, frameon=False, loc="lower right")


def plot_secondary(ax, seg, res, diag):
    t, f = seg["time"], seg["flat"]
    tc, P, t14 = _med(res, "tc"), _med(res, "period"), _med(res, "t14_hr")
    h = ((((t - tc) / P) % 1.0) - 0.5) * P * 24.0                       # hours from phase 0.5
    lim = max(3.0 * t14, 3.0)
    s = np.abs(h) < lim
    ax.set_title("6 · Secondary eclipse (phase 0.5)" + (f"  ({diag['secondary_sigma']:.1f}σ)"
                 if np.isfinite(diag.get("secondary_sigma", np.nan)) else ""))
    if s.sum() < 20:
        ax.text(0.5, 0.5, "phase 0.5 not covered", ha="center", transform=ax.transAxes); return
    ax.plot(h[s], f[s], ".", ms=1.5, color=GREY, alpha=0.5, rasterized=True)
    c, m, e = _binned(h[s], f[s], np.linspace(-lim, lim, 31))
    ax.errorbar(c, m, e, fmt="o-", ms=3, lw=0.8, color=NAVY)
    d, de, _ = diag["secondary"]
    if np.isfinite(d):
        ax.text(0.03, 0.05, f"depth = {1e6 * d:.0f} ± {1e6 * de:.0f} ppm", transform=ax.transAxes, fontsize=7)
    ax.set_xlabel("Hours from phase 0.5"); ax.set_ylabel("Flux")


def plot_centroid(ax, seg, res, diag):
    ax.set_title("7 · Centroid motion")
    c1, c2 = seg["c1"], seg["c2"]
    ok = np.isfinite(c1) & np.isfinite(c2)
    if ok.sum() < 100 or diag.get("centroid") is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "Centroid data not available for this light curve\n"
                          "(downloaded before the mom_centr columns were captured;\n"
                          "re-run acquisition to enable this check)",
                ha="center", va="center", transform=ax.transAxes, fontsize=8, color=GREY)
        return
    t = seg["time"]
    tc, P, t14 = _med(res, "tc"), _med(res, "period"), _med(res, "t14_hr")
    h = _phase_hr(t, tc, P)
    lim = max(3.0 * t14, 3.0)
    s = ok & (np.abs(h) < lim)
    edges = np.linspace(-lim, lim, 31)
    for arr, name, col in [(c1, "column", NAVY), (c2, "row", ORANGE)]:
        c, m, e = _binned(h[s], arr[s] - np.median(arr[s]), edges)
        ax.errorbar(c, m * 1e3, e * 1e3, fmt="o-", ms=3, lw=0.8, color=col, label=name)
    ax.axvspan(-t14 / 2, t14 / 2, color=GOLD, alpha=0.2, lw=0)
    ax.axhline(0, color=GREY, lw=0.6)
    dx, dy, ex, ey = diag["centroid"]
    ax.text(0.03, 0.05, f"shift = ({dx * 1e3:.2f} ± {ex * 1e3:.2f}, {dy * 1e3:.2f} ± {ey * 1e3:.2f}) mpix", transform=ax.transAxes, fontsize=7)
    ax.set_xlabel("Hours from mid-transit"); ax.set_ylabel("Centroid offset (mpix)"); ax.legend(fontsize=7, frameon=False)


# ─────────────────────────────────────────────────────────────────────────────
# PNG export and 3-page PDF
# ─────────────────────────────────────────────────────────────────────────────

def save_pngs(seg, res, samples, diag, out_dir: Path, tic: str, corner_img) -> dict:
    d = out_dir / "plots" / tic
    d.mkdir(parents=True, exist_ok=True)
    paths = {}

    def save(name, fig):
        p = d / f"{name}.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths[name] = str(p)

    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, constrained_layout=True); plot_lightcurve(ax[0], ax[1], seg, res); save("1_lightcurve", fig)
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True); plot_periodogram(ax, seg, res, out_dir, tic); save("2_periodogram", fig)
    fig = plt.figure(figsize=(8, 5.5)); gs = GridSpec(3, 1, hspace=0.08)
    plot_fold(fig.add_subplot(gs[:2]), fig.add_subplot(gs[2]), seg, res, samples); save("3_fold", fig)
    fig, ax = plt.subplots(figsize=(7, 7)); plot_corner(ax, corner_img); save("4_corner", fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True); plot_odd_even(ax, seg, res, diag); save("5_odd_even", fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True); plot_secondary(ax, seg, res, diag); save("6_secondary", fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True); plot_centroid(ax, seg, res, diag); save("7_centroid", fig)
    return paths


def _table(ax, rows, col_w, fs=8, header=None):
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, colWidths=col_w, loc="upper left", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(fs); t.scale(1, 1.35)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0 and header:
            cell.set_facecolor("#dfe7f3"); cell.set_text_props(fontweight="bold")
    return t


def _badge(fig, verdict, x=0.93, y=0.955):
    col = {"PASS": TEAL, "CAUTION": GOLD, "FALSE-POSITIVE LIKELY": RED}[verdict]
    fig.text(x, y, verdict, ha="right", va="center", fontsize=10, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.4", fc=col, ec="none"))


def _header(fig, tic, page, res, cand, verdict):
    fig.text(0.07, 0.955, f"LuNaVision AI · Candidate report · {tic}", fontsize=13, fontweight="bold", color=NAVY, va="center")
    p = cand.get("p_transit")
    sub = f"classifier P(transit) = {p:.2f}" if p is not None and np.isfinite(p) else "classifier score not available"
    fig.text(0.07, 0.928, f"{sub}   ·   fit: {res.get('method')} ({'converged' if res.get('converged') else 'not converged'})",
             fontsize=8, color=GREY, va="center")
    _badge(fig, verdict)
    fig.text(0.5, 0.02, f"Page {page} of 3  ·  candidate for follow-up, not a confirmed planet", ha="center", fontsize=7, color=GREY)


def make_pdf(path: Path, tic, seg, res, samples, diag, verdict_d, cand, corner_img, out_dir):
    A4 = (8.27, 11.69)
    path.parent.mkdir(parents=True, exist_ok=True)
    verdict = verdict_d["verdict"]
    with PdfPages(path) as pdf:
        # ---- page 1: summary, parameters, light curve, periodogram ----------
        fig = plt.figure(figsize=A4); _header(fig, tic, 1, res, cand, verdict)
        gs = GridSpec(5, 2, figure=fig, left=0.09, right=0.95, top=0.90, bottom=0.06, hspace=0.6, wspace=0.3,
                      height_ratios=[1.35, 1.0, 1.0, 1.0, 1.0])
        ax = fig.add_subplot(gs[0, 0])
        _table(ax, [["Period (d)", _pm(res, "period", "{:.6f}")], ["Mid-transit tc (BTJD)", _pm(res, "tc", "{:.5f}")],
                    ["Rp / R*", _pm(res, "rp_rs", "{:.4f}")], ["Impact parameter b", _pm(res, "b", "{:.2f}")],
                    ["a / R*", _pm(res, "a_rs", "{:.2f}")], ["Inclination (deg)", _pm(res, "inc_deg", "{:.2f}")],
                    ["Depth (ppm)", _pm(res, "depth_ppm", "{:.0f}")], ["Duration T14 (h)", _pm(res, "t14_hr", "{:.3f}")]],
               [0.45, 0.55], header=["Fitted parameter", "Median (+1σ / −1σ)"])
        ax2 = fig.add_subplot(gs[0, 1])
        rows = [["Planet radius (R⊕)", _pm(res, "rp_rearth", "{:.2f}")], ["Size class", res.get("size_class", "n/a")],
                ["Semi-major axis (AU)", _pm(res, "a_au", "{:.4f}")], ["Equilibrium temp. (K)", _pm(res, "teq_k", "{:.0f}")],
                ["Habitable-zone score", _pm(res, "hz_score", "{:.2f}") + (" (inside HZ)" if _med(res, "in_hz") > 0.5 else "")],
                ["Transit SNR", _pm(res, "snr", "{:.1f}")], ["Transits in window", f"{res.get('n_in_transit', 'n/a')} pts"],
                ["Photometric noise (ppm)", f"{res.get('sigma_ppm', float('nan')):.0f}"]]
        _table(ax2, rows, [0.5, 0.5], header=["Derived quantity", "Value"])
        plot_lightcurve(fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :]), seg, res)
        plot_periodogram(fig.add_subplot(gs[3, :]), seg, res, out_dir, tic)
        ax = fig.add_subplot(gs[4, :]); ax.axis("off")
        star = f"TIC stellar parameters were {'used' if res.get('stellar_prior') else 'not available'} as a density prior on a/R*."
        ax.text(0, 0.95, "Method. The TLS candidate (period, epoch, duration, radius ratio) seeds a batman quadratic-limb-darkening\n"
                         "transit model fitted to the detrended data within ±3 transit durations of each event. Least-squares starts the\n"
                         f"emcee walkers ({res.get('n_walkers')} walkers, {res.get('n_steps')} steps); intervals are the 16/50/84th percentiles. {star}",
                fontsize=7.5, va="top", color="#333")
        pdf.savefig(fig); plt.close(fig)

        # ---- page 2: fold, odd/even, secondary, vetting ---------------------
        fig = plt.figure(figsize=A4); _header(fig, tic, 2, res, cand, verdict)
        gs = GridSpec(4, 2, figure=fig, left=0.09, right=0.95, top=0.90, bottom=0.06, hspace=0.55, wspace=0.3,
                      height_ratios=[1.6, 0.6, 1.2, 1.3])
        fold = gs[0:2, :].subgridspec(3, 1, hspace=0.08)
        plot_fold(fig.add_subplot(fold[:2]), fig.add_subplot(fold[2]), seg, res, samples)
        plot_odd_even(fig.add_subplot(gs[2, 0]), seg, res, diag)
        plot_secondary(fig.add_subplot(gs[2, 1]), seg, res, diag)
        ax = fig.add_subplot(gs[3, :])
        rows = [[n, s, d] for n, s, d in verdict_d["checks"]]
        t = _table(ax, rows, [0.22, 0.10, 0.68], header=["Vetting check", "Result", "Detail"])
        for (r, c), cell in t.get_celld().items():
            if c == 1 and r > 0:
                cell.set_text_props(color={"PASS": TEAL, "FAIL": RED}.get(rows[r - 1][1], GREY), fontweight="bold")
        pdf.savefig(fig); plt.close(fig)

        # ---- page 3: corner, centroid, interpretation -----------------------
        fig = plt.figure(figsize=A4); _header(fig, tic, 3, res, cand, verdict)
        gs = GridSpec(3, 1, figure=fig, left=0.09, right=0.95, top=0.90, bottom=0.06, hspace=0.35, height_ratios=[2.6, 1.1, 1.3])
        plot_corner(fig.add_subplot(gs[0]), corner_img)
        plot_centroid(fig.add_subplot(gs[1]), seg, res, diag)
        ax = fig.add_subplot(gs[2]); ax.axis("off")
        lines = [f"Verdict: {verdict}.", ""]
        if verdict_d["flags"]:
            lines += ["Flags raised:"] + [f"  • {f}" for f in verdict_d["flags"]]
        else:
            lines += ["No vetting flag was raised; every available check is consistent with a planetary transit."]
        lines += ["", "Interpretation notes:",
                  f"  • Fit quality: reduced chi² = {res.get('chi2_red', float('nan')):.2f}, acceptance = {res.get('acceptance', float('nan')):.2f},"
                  f" autocorrelation time = {res.get('tau_max', float('nan')):.0f} steps.",
                  "  • Limb darkening is fixed at (0.3, 0.1); an eccentric orbit is not modelled.",
                  "  • Vetting checks use the detrended photometry only; a centroid check needs the mom_centr columns.",
                  "  • Candidates should be confirmed with ground-based follow-up before any claim of a planet."]
        ax.text(0, 1, "\n".join(lines), va="top", fontsize=8, color="#222")
        pdf.savefig(fig); plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration: one target, and the catalogue
# ─────────────────────────────────────────────────────────────────────────────

def report_one(tic: str, out_dir: Path = OUT_DIR, pdf: bool = True, cand: Optional[dict] = None) -> dict:
    tic = norm_tic(tic)
    jp = out_dir / "fits" / f"{tic}.json"
    if not jp.exists():
        raise FileNotFoundError(f"{jp} — run fitting.py first")
    res = json.loads(jp.read_text())
    if res.get("status") != "done":
        raise RuntimeError(f"{tic}: fit did not complete ({res.get('status')})")
    seg = load_segment(tic)
    samples = _load_posterior(out_dir, tic)
    diag = compute_diagnostics(seg, res)
    verdict = vet(res, diag)
    corner_img = plot_corner_image(res, samples)
    pngs = save_pngs(seg, res, samples, diag, out_dir, tic, corner_img)
    pdf_path = None
    if pdf:
        pdf_path = make_pdf(out_dir / "reports" / f"{tic}.pdf", tic, seg, res, samples, diag, verdict,
                            cand or {}, corner_img, out_dir)
    return {"tic_id": tic, "res": res, "diag": diag, "vet": verdict, "pngs": pngs,
            "pdf": str(pdf_path) if pdf_path else None}


def _catalog_row(r: dict, p_transit: float) -> dict:
    res, v, dg = r["res"], r["vet"], r["diag"]
    row = {"tic_id": r["tic_id"], "p_transit": p_transit, "verdict": v["verdict"], "flags": "; ".join(v["flags"])}
    for k, nm in [("period", "period_d"), ("tc", "tc_btjd"), ("rp_rs", "rp_rs"), ("b", "b"), ("a_rs", "a_rs"),
                  ("inc_deg", "inc_deg"), ("depth_ppm", "depth_ppm"), ("t14_hr", "t14_hr"), ("rp_rearth", "rp_rearth"),
                  ("a_au", "a_au"), ("teq_k", "teq_k"), ("hz_score", "hz_score"), ("snr", "snr")]:
        s = res.get(k)
        if isinstance(s, dict):
            row[nm], row[nm + "_err_hi"], row[nm + "_err_lo"] = s["med"], s["hi"], s["lo"]
    row.update(in_hz=bool(_med(res, "in_hz") > 0.5) if np.isfinite(_med(res, "in_hz")) else None,
               size_class=res.get("size_class"), odd_even_sigma=dg.get("odd_even_sigma"),
               secondary_sigma=dg.get("secondary_sigma"), centroid_sigma=dg.get("centroid_sigma"),
               fit_method=res.get("method"), converged=res.get("converged"), n_eff=res.get("n_eff"),
               report_pdf=r["pdf"])
    snr = row.get("snr", 0.0) or 0.0
    p = 1.0 if p_transit is None or not np.isfinite(p_transit) else p_transit
    row["rank_score"] = p * VERDICT_WEIGHT[v["verdict"]] * min(snr / 10.0, 1.0)
    return row


def build_catalog(rows: list, out_dir: Path = OUT_DIR) -> pd.DataFrame:
    cat = pd.DataFrame(rows).sort_values("rank_score", ascending=False).reset_index(drop=True)
    cat.insert(0, "rank", np.arange(1, len(cat) + 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    cat.to_csv(out_dir / "candidate_catalog.csv", index=False)
    try:
        from astropy.table import Table
        t = Table.from_pandas(cat.fillna(np.nan).astype({c: str for c in cat.columns if cat[c].dtype == object}))
        t.write(out_dir / "candidate_catalog.fits", overwrite=True)
    except Exception as e:                                         # noqa: BLE001
        log.warning("FITS catalogue not written (%s)", e)
    return cat


def run(out_dir: Path = OUT_DIR, tics: Optional[list] = None, pdf: bool = True, n_jobs: int = 1) -> pd.DataFrame:
    """Report every fitted candidate (or `tics`) and write the ranked catalogue."""
    from joblib import Parallel, delayed
    out_dir = Path(out_dir)
    if tics is None:
        tics = sorted(p.stem for p in (out_dir / "fits").glob("*.json"))
    fr = out_dir / "fit_results.csv"
    prob = {}
    if fr.exists():
        t = pd.read_csv(fr)
        if "p_transit" in t:
            prob = dict(zip(t["tic_id"], t["p_transit"]))
    log.info("Reporting %d candidates", len(tics))

    def one(t):
        try:
            return report_one(t, out_dir, pdf, {"p_transit": prob.get(norm_tic(t), np.nan)})
        except Exception as e:                                     # noqa: BLE001
            log.error("%s: %s", t, e); return None
    results = [r for r in Parallel(n_jobs=n_jobs)(delayed(one)(t) for t in tics) if r]
    if not results:
        raise SystemExit("No candidate could be reported")
    cat = build_catalog([_catalog_row(r, prob.get(r["tic_id"], np.nan)) for r in results], out_dir)
    log.info("Catalogue: %s (%d rows). Verdicts: %s", out_dir / "candidate_catalog.csv", len(cat),
             cat["verdict"].value_counts().to_dict())
    return cat


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Stage 6b — plots, vetting, catalogue and PDF reports")
    ap.add_argument("--tic", action="append", help="report only these TIC ids (repeatable)")
    ap.add_argument("--all", action="store_true", help="report every fit in outputs/fits")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    if not (a.tic or a.all):
        ap.error("give --tic or --all")
    run(a.out, [norm_tic(t) for t in a.tic] if a.tic else None, pdf=not a.no_pdf, n_jobs=a.n_jobs)
