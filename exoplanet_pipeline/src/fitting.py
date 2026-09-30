"""
fitting.py — Stage 6a: orbital-parameter estimation (batman transit model + emcee MCMC).

For each candidate the TLS result (period, epoch, duration, radius ratio) seeds a
physical transit model that is fitted to the detrended light curve. The posterior
gives 16/50/84 % (and 95 %) credible intervals for the transit parameters, and
derived quantities (inclination, T14, planet radius, semi-major axis, equilibrium
temperature, habitable-zone position, SNR).

Design decisions
----------------
* **Same data window as TLS.** The fit uses the longest contiguous segment
  (`detection.select_search_window`), so stitched sectors months apart cannot
  alias the period.
* **Decorrelated epoch.** The fitted epoch `tc` is the transit nearest the middle
  of the data, not TLS's first-transit T0. Fitting (T0, P) directly is strongly
  correlated on long baselines and makes MCMC mix badly.
* **Only data near transit.** Points within +/- `WINDOW_DUR` transit durations of a
  predicted mid-transit are kept; out-of-transit data carry no information about
  the shape and only make each likelihood call slower. If that is still long the
  window is time-binned (with the matching exposure integration in batman).
* **Least-squares first, MCMC second, LSQ fallback.** A multi-start bounded
  least-squares fit seeds the walkers. If the chain does not converge (integrated
  autocorrelation time too long / bad acceptance) or emcee fails, the least-squares
  solution with Jacobian-covariance errors is reported instead and flagged
  `method="lsq"`, `converged=False`.
* **Fixed limb darkening.** Quadratic (u1, u2) = (0.3, 0.1) as in the plan; only the
  radius-ratio/impact-parameter correlation is affected at TESS precision.
* **Optional stellar-density prior.** When TIC mass/radius exist, a Gaussian prior on
  ln(a/Rs) from the stellar density (30 % on density) breaks the b / a degeneracy.

CLI
---
    python src/fitting.py --tic TIC_16740101
    python src/fitting.py --top 20 --min-prob 0.6 --n-jobs 4
    python src/fitting.py --all-candidates --steps 2000 --force
"""
from __future__ import annotations

import argparse
import json
import logging
import time as _time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("fitting")

BASE_DIR     = Path(__file__).resolve().parents[1]
DETRENDED    = BASE_DIR / "data" / "processed" / "lc_detrended"
CATALOGS_DIR = BASE_DIR / "data" / "catalogs"
MODELS_DIR   = BASE_DIR / "models"
OUT_DIR      = BASE_DIR / "outputs"

# ── MCMC settings ────────────────────────────────────────────────────────────
N_WALKERS    = 32
N_STEPS      = 1000        # minimum production length
MAX_STEPS    = 4000        # extend in chunks up to this if not yet converged
CHUNK        = 500
TAU_FACTOR   = 20          # converged when steps > TAU_FACTOR * max autocorr time (b/a_rs are
                           # degenerate, tau ~ 100-200; 20 tau still gives ~500+ effective samples)
BURN_FRAC    = 0.3
THIN_SAVE    = 4000        # max posterior samples stored per target
SEED         = 42

# ── data window ──────────────────────────────────────────────────────────────
WINDOW_DUR   = 3.0         # keep |phase| < WINDOW_DUR * T14 around each transit
MAX_POINTS   = 6000        # bin if more in-window points than this
BIN_MIN      = 6.0         # bin width (minutes) when binning
SUPERSAMPLE  = 3
LIMB_DARK    = (0.3, 0.1)  # quadratic

# ── physical constants ───────────────────────────────────────────────────────
RSUN_IN_REARTH = 109.076
RSUN_IN_AU     = 0.00465047
SUN_TEFF       = 5772.0
G_CGS          = 6.674e-8
RHO_SUN_CGS    = 1.41

PARAMS = ["tc", "period", "rp_rs", "b", "a_rs"]        # reported natural parameters
_SAMPLED = ["tc", "period", "rp_rs", "b", "ln_a", "ln_sigma"]  # ndim = 6


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def norm_tic(t) -> str:
    """'TIC_123', 'TIC 123', 123 or '123.0' -> 'TIC_123'."""
    t = str(t).strip()
    for pre in ("TIC_", "TIC "):
        if t.startswith(pre):
            t = t[len(pre):]
    if t.endswith(".0"):
        t = t[:-2]
    return t if t.startswith("SYN_") else f"TIC_{t}"


def load_segment(tic_id: str, detrended_dir: Path = DETRENDED) -> dict:
    """
    Longest contiguous segment of a detrended light curve with every per-cadence
    array kept index-aligned: time, flat (detrended), norm (normalised, pre-detrend),
    trend, and the centroid columns (all-NaN if the file predates them).
    """
    from detection import select_search_window, MAX_SEGMENT_GAP_D

    path = detrended_dir / f"{norm_tic(tic_id)}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=True)
    n = len(d["time"])
    nan = np.full(n, np.nan)
    get = lambda k: d[k].astype(np.float64) if k in d else nan      # noqa: E731
    t, flat, norm, trend, c1, c2 = select_search_window(
        d["time"].astype(np.float64), d["flat_flux"].astype(np.float64), MAX_SEGMENT_GAP_D,
        get("flux_norm"), get("trend"), get("mom_centr1"), get("mom_centr2"))
    return {"tic_id": norm_tic(tic_id), "time": t, "flat": flat, "norm": norm,
            "trend": trend, "c1": c1, "c2": c2,
            "sector": int(d["sector"]) if "sector" in d else -1}


def load_classifier(path: Path = MODELS_DIR / "classifier.joblib"):
    """
    Load the trained StackedClassifier. The model is pickled from
    `python src/classifier.py`, i.e. under `__main__`, so the class has to be
    registered there before joblib can unpickle it.
    """
    import __main__
    import joblib
    import classifier
    if not hasattr(__main__, "StackedClassifier"):
        __main__.StackedClassifier = classifier.StackedClassifier
    return joblib.load(path)


def select_candidates(features_csv: Path = CATALOGS_DIR / "feature_matrix.csv",
                      model_path: Path = MODELS_DIR / "classifier.joblib",
                      min_prob: float = 0.5, top: Optional[int] = None,
                      include_synthetic: bool = False, use_classifier: bool = True) -> pd.DataFrame:
    """
    Candidates to fit: rows the classifier scores as Transit (class 0) with
    probability >= min_prob, best first. Falls back to the detection cuts
    (SDE >= 7, FAP < 0.01) if the model cannot be loaded.
    """
    df = pd.read_csv(features_csv)
    df["tic_id"] = df["tic_id"].map(norm_tic)
    df = df[df["status"].astype(str).eq("done") & df["period"].notna()]
    if not include_synthetic:
        df = df[~df["tic_id"].str.startswith("SYN_")]

    df = df.copy()
    df["p_transit"] = np.nan
    if use_classifier:
        try:
            model = load_classifier(model_path)
            X = df.reindex(columns=model.feature_names_)
            proba = model.predict_proba(X)
            df["p_transit"] = proba[:, list(model.classes_).index(0)]
            df = df[df["p_transit"] >= min_prob].sort_values("p_transit", ascending=False)
        except Exception as e:                                     # noqa: BLE001
            log.warning("Classifier unavailable (%s); falling back to SDE/FAP cuts.", e)
            use_classifier = False
    if not use_classifier:
        df = df[(df["SDE"] >= 7.0) & ((df["FAP"] < 0.01) | df["FAP"].isna())]
        df = df.sort_values("SDE", ascending=False)
    return df.head(top) if top else df


# ─────────────────────────────────────────────────────────────────────────────
# Transit model
# ─────────────────────────────────────────────────────────────────────────────

class _TransitFit:
    """batman model + priors + likelihood for one candidate's windowed data."""

    def __init__(self, t, f, exp_time, tc0, p0, dur_d, rp0, a_prior, sigma0, span_epochs,
                 limb_dark=LIMB_DARK):
        import batman
        self.t, self.f, self.sigma0 = t, f, sigma0
        self.a_prior = a_prior                                   # (mu_ln_a, sd) or None
        self.pm = batman.TransitParams()
        self.pm.t0, self.pm.per, self.pm.rp = tc0, p0, rp0
        self.pm.a, self.pm.inc, self.pm.ecc, self.pm.w = 10.0, 89.0, 0.0, 90.0
        self.pm.u, self.pm.limb_dark = list(limb_dark), "quadratic"
        ss = SUPERSAMPLE if exp_time > 0 else 1
        self.model = batman.TransitModel(self.pm, t, supersample_factor=ss, exp_time=exp_time)

        dt = max(3.0 * dur_d / max(span_epochs, 1.0), 1e-4 * p0)   # 3 durations of drift over the baseline
        self.lo = np.array([tc0 - max(dur_d, 0.02), p0 - dt, 1e-3, 0.0, np.log(1.5), np.log(sigma0 / 20)])
        self.hi = np.array([tc0 + max(dur_d, 0.02), p0 + dt, 0.5, 1.0, np.log(400.0), np.log(sigma0 * 20)])

    def lc(self, th):
        tc, per, rp, b, ln_a = th[:5]
        a = np.exp(ln_a)
        self.pm.t0, self.pm.per, self.pm.rp, self.pm.a = tc, per, rp, a
        self.pm.inc = np.degrees(np.arccos(min(b / a, 1.0)))
        return self.model.light_curve(self.pm)

    def resid(self, th5):
        return (self.f - self.lc(th5)) / self.sigma0

    def log_prob(self, th):
        if np.any(th < self.lo) or np.any(th > self.hi):
            return -np.inf
        if th[3] / np.exp(th[4]) >= 1.0:                       # b must be < a/Rs
            return -np.inf
        lp = 0.0
        if self.a_prior is not None:
            mu, sd = self.a_prior
            lp += -0.5 * ((th[4] - mu) / sd) ** 2
        sigma = np.exp(th[5])
        r = self.f - self.lc(th)
        return lp - 0.5 * np.sum((r / sigma) ** 2) - len(r) * th[5]


def _window(t, f, tc0, period, dur_d):
    """Points within WINDOW_DUR durations of a predicted mid-transit."""
    ph = ((t - tc0 + 0.5 * period) % period) - 0.5 * period
    keep = np.abs(ph) < max(WINDOW_DUR * dur_d, 0.05)
    return t[keep], f[keep]


def _bin(t, f, width_d):
    k = np.floor((t - t[0]) / width_d).astype(np.int64)
    n = np.bincount(k)
    ok = n > 0
    return np.bincount(k, t)[ok] / n[ok], np.bincount(k, f)[ok] / n[ok], n[ok]


def _density_prior(period, rstar, mstar, sd_ln_rho=0.3):
    """Gaussian prior on ln(a/Rs) from stellar density (a/Rs = (G rho P^2 / 3 pi)^(1/3))."""
    if not (np.isfinite(rstar) and np.isfinite(mstar) and rstar > 0 and mstar > 0):
        return None
    rho = mstar / rstar ** 3 * RHO_SUN_CGS
    a = (G_CGS * rho * (period * 86400.0) ** 2 / (3.0 * np.pi)) ** (1.0 / 3.0)
    return float(np.log(a)), float(sd_ln_rho / 3.0)


# ─────────────────────────────────────────────────────────────────────────────
# Derived quantities
# ─────────────────────────────────────────────────────────────────────────────

def _t14_hours(period, rp, b, a):
    inc = np.arccos(np.clip(b / a, 0, 1))
    arg = np.sqrt(np.clip((1 + rp) ** 2 - b ** 2, 0, None)) / (a * np.sin(inc))
    return period / np.pi * np.arcsin(np.clip(arg, 0, 1)) * 24.0


def derive(samples: dict, star: dict, sigma: float, n_in: float) -> dict:
    """Vectorised derived parameters from posterior samples (dict of arrays)."""
    P, rp, b, a = samples["period"], samples["rp_rs"], samples["b"], samples["a_rs"]
    out = {
        "inc_deg":  np.degrees(np.arccos(np.clip(b / a, 0, 1))),
        "depth_ppm": rp ** 2 * 1e6,
        "t14_hr":   _t14_hours(P, rp, b, a),
        "snr":      rp ** 2 / sigma * np.sqrt(max(n_in, 1.0)),
    }
    R, M, Teff = star.get("rad"), star.get("mass"), star.get("teff")
    if R and np.isfinite(R):
        out["rp_rearth"] = rp * R * RSUN_IN_REARTH
        out["a_au"] = a * R * RSUN_IN_AU
        if Teff and np.isfinite(Teff):
            out["teq_k"] = Teff * np.sqrt(1.0 / (2.0 * a))
            L = R ** 2 * (Teff / SUN_TEFF) ** 4
            inner, outer = np.sqrt(L / 1.1), np.sqrt(L / 0.36)
            mid, half = 0.5 * (inner + outer), 0.5 * (outer - inner)
            out["hz_score"] = np.abs(out["a_au"] - mid) / half      # <1 inside the HZ (as in features.py)
            out["in_hz"] = (out["hz_score"] <= 1.0).astype(float)
    return out


def size_class(rp_rearth: float) -> str:
    if not np.isfinite(rp_rearth):
        return "unknown"
    for lim, name in [(1.25, "Earth-size"), (2.0, "Super-Earth"), (6.0, "Neptune-size"),
                      (15.0, "Jupiter-size")]:
        if rp_rearth < lim:
            return name
    return "too large for a planet (likely stellar companion)"


def _summ(x) -> dict:
    x = np.asarray(x, float)
    lo2, lo16, med, hi84, hi97 = np.nanpercentile(x, [2.5, 15.87, 50, 84.13, 97.5])
    return {"med": float(med), "lo": float(med - lo16), "hi": float(hi84 - med),
            "ci95_lo": float(lo2), "ci95_hi": float(hi97)}


# ─────────────────────────────────────────────────────────────────────────────
# Fit one target
# ─────────────────────────────────────────────────────────────────────────────

def fit_target(tic_id: str, row: Optional[dict] = None, n_walkers: int = N_WALKERS,
               n_steps: int = N_STEPS, max_steps: int = MAX_STEPS, use_stellar_prior: bool = True,
               progress: bool = False, seed: int = SEED) -> tuple[dict, Optional[np.ndarray]]:
    """
    Fit one candidate. Returns (result dict, posterior sample array or None).
    `row` is the candidate's feature-matrix row (period, t0, depth, duration_hr, rp_rs,
    stellar_*); it is looked up in feature_matrix.csv when omitted.
    """
    from scipy.optimize import least_squares
    tic_id = norm_tic(tic_id)
    t_start = _time.time()
    res: dict = {"tic_id": tic_id, "status": "failed", "method": None, "converged": False}
    try:
        if row is None:
            fm = pd.read_csv(CATALOGS_DIR / "feature_matrix.csv")
            fm["tic_id"] = fm["tic_id"].map(norm_tic)
            row = fm[fm.tic_id == tic_id].iloc[0].to_dict()
        P0, T0 = float(row["period"]), float(row["t0"])
        dur_d = float(row["duration_hr"]) / 24.0
        rp0 = float(row["rp_rs"]) if np.isfinite(row.get("rp_rs", np.nan)) else np.sqrt(abs(float(row["depth"])))
        rp0 = float(np.clip(rp0, 0.005, 0.4))
        if not (np.isfinite(P0) and P0 > 0 and np.isfinite(T0) and np.isfinite(dur_d) and dur_d > 0):
            raise ValueError("candidate row has no usable TLS period/epoch/duration")

        seg = load_segment(tic_id)
        t_all, f_all = seg["time"], seg["flat"]
        # epoch nearest the middle of the data
        tmid = 0.5 * (t_all[0] + t_all[-1])
        tc0 = T0 + np.round((tmid - T0) / P0) * P0
        t, f = _window(t_all, f_all, tc0, P0, dur_d)
        if len(t) < 50:
            raise ValueError(f"only {len(t)} points near transit")
        oot_mask = np.abs(((t - tc0 + 0.5 * P0) % P0) - 0.5 * P0) > 0.75 * dur_d
        sigma0 = 1.4826 * np.nanmedian(np.abs(f[oot_mask] - np.nanmedian(f[oot_mask]))) if oot_mask.sum() > 30 \
            else np.nanstd(f)
        sigma0 = float(max(sigma0, 1e-6))

        exp_time = 0.0
        if len(t) > MAX_POINTS:
            width = BIN_MIN / 1440.0
            t, f, cnt = _bin(t, f, width)
            sigma0 = float(sigma0 / np.sqrt(np.median(cnt)))       # binned noise
            exp_time = width
        star = {"rad": row.get("stellar_rad"), "mass": row.get("stellar_mass"),
                "teff": row.get("stellar_Teff")}
        star = {k: (float(v) if v is not None and np.isfinite(v) else np.nan) for k, v in star.items()}
        a_prior = _density_prior(P0, star["rad"], star["mass"]) if use_stellar_prior else None
        span = (t_all[-1] - t_all[0]) / P0

        fit = _TransitFit(t, f, exp_time, tc0, P0, dur_d, rp0, a_prior, sigma0, span)

        # ---- multi-start least squares -------------------------------------
        a_guess = a_prior[0] if a_prior else np.log(max(P0 / (np.pi * dur_d), 2.0))
        best = None
        for b0 in (0.1, 0.5, 0.8):
            x0 = np.array([tc0, P0, rp0, b0, a_guess])
            x0[4] = float(np.clip(x0[4], fit.lo[4] + 1e-3, fit.hi[4] - 1e-3))
            try:
                r = least_squares(fit.resid, x0, bounds=(fit.lo[:5], fit.hi[:5]),
                                  x_scale=[dur_d / 20, P0 * 1e-5, 0.01, 0.1, 0.1], max_nfev=200)
            except Exception:                                      # noqa: BLE001
                continue
            if best is None or r.cost < best.cost:
                best = r
        if best is None:
            raise RuntimeError("least-squares failed for every start")
        th_lsq = best.x
        ln_sig = float(np.log(np.std(fit.f - fit.lc(th_lsq))))
        n_in = float(np.sum(np.abs(((t - th_lsq[0] + 0.5 * th_lsq[1]) % th_lsq[1]) - 0.5 * th_lsq[1])
                            < 0.5 * dur_d))
        res.update(n_points=int(len(t)), n_in_transit=int(n_in), sigma_ppm=float(np.exp(ln_sig) * 1e6),
                   binned=bool(exp_time > 0), stellar_prior=bool(a_prior))

        # ---- MCMC -----------------------------------------------------------
        samples5, method, converged, tau_max, acc, steps, n_eff = None, "lsq", False, np.nan, np.nan, 0, np.nan
        try:
            import emcee
            rng = np.random.default_rng(seed)
            p_best = np.append(th_lsq, ln_sig)
            scale = np.array([dur_d / 200, P0 * 1e-6, 1e-3, 0.02, 0.02, 0.02])
            p0 = p_best + scale * rng.standard_normal((n_walkers, 6))
            p0[:, 3] = np.abs(p0[:, 3])
            for i in range(n_walkers):                             # make sure every start is inside the prior
                tries = 0
                while not np.isfinite(fit.log_prob(p0[i])) and tries < 50:
                    p0[i] = p_best + scale * 0.1 * rng.standard_normal(6); p0[i, 3] = abs(p0[i, 3]); tries += 1
            sampler = emcee.EnsembleSampler(n_walkers, 6, fit.log_prob)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                state = sampler.run_mcmc(p0, n_steps, progress=progress)
                steps = n_steps
                while True:
                    tau = sampler.get_autocorr_time(tol=0)
                    tau_max = float(np.nanmax(tau))
                    if steps > TAU_FACTOR * tau_max or steps >= max_steps:
                        break
                    state = sampler.run_mcmc(state, CHUNK, progress=progress)
                    steps += CHUNK
            acc = float(np.mean(sampler.acceptance_fraction))
            converged = bool(steps > TAU_FACTOR * tau_max and 0.1 < acc < 0.9 and np.isfinite(tau_max))
            n_eff = float(n_walkers * steps * (1 - BURN_FRAC) / tau_max) if np.isfinite(tau_max) else np.nan
            burn = int(max(BURN_FRAC * steps, 2 * tau_max if np.isfinite(tau_max) else 0))
            chain = sampler.get_chain(discard=min(burn, steps // 2), flat=True)
            if converged:
                samples5, method = chain, "mcmc"
            else:
                log.warning("%s: MCMC not converged (steps=%d, tau=%.0f, acc=%.2f) -> LSQ fallback",
                            tic_id, steps, tau_max, acc)
        except Exception as e:                                     # noqa: BLE001
            log.warning("%s: emcee failed (%s) -> LSQ fallback", tic_id, e)

        if samples5 is None:                                       # least-squares + Jacobian covariance
            J = best.jac
            dof = max(len(fit.f) - 5, 1)
            try:
                cov = np.linalg.inv(J.T @ J) * (2 * best.cost / dof)
                cov = 0.5 * (cov + cov.T)
                rng = np.random.default_rng(seed)
                draw = rng.multivariate_normal(th_lsq, cov, size=4000)
                ok = np.all((draw >= fit.lo[:5]) & (draw <= fit.hi[:5]), axis=1)
                draw = draw[ok] if ok.sum() > 200 else np.tile(th_lsq, (200, 1))
            except np.linalg.LinAlgError:
                draw = np.tile(th_lsq, (200, 1))
            samples5 = np.column_stack([draw, np.full(len(draw), ln_sig)])
            method = "lsq"

        if len(samples5) > THIN_SAVE:
            samples5 = samples5[np.linspace(0, len(samples5) - 1, THIN_SAVE).astype(int)]

        nat = {"tc": samples5[:, 0], "period": samples5[:, 1], "rp_rs": samples5[:, 2],
               "b": samples5[:, 3], "a_rs": np.exp(samples5[:, 4])}
        sig = float(np.exp(np.median(samples5[:, 5])))
        der = derive(nat, star, sig, n_in)
        for k, v in {**nat, **der}.items():
            res[k] = _summ(v)
        res.update(method=method, converged=bool(converged) if method == "mcmc" else False,
                   n_steps=int(steps), n_walkers=n_walkers, acceptance=acc, tau_max=tau_max, n_eff=n_eff,
                   size_class=size_class(res.get("rp_rearth", {}).get("med", np.nan)),
                   tls_period=P0, tls_rp_rs=rp0, tls_duration_hr=dur_d * 24.0,
                   chi2_red=float(2 * best.cost / max(len(fit.f) - 5, 1)),
                   status="done", runtime_s=round(_time.time() - t_start, 1))
        return res, samples5
    except Exception as e:                                         # noqa: BLE001
        res.update(status=f"failed: {e}", runtime_s=round(_time.time() - t_start, 1))
        log.error("%s: %s", tic_id, e)
        return res, None


# ─────────────────────────────────────────────────────────────────────────────
# Batch / persistence
# ─────────────────────────────────────────────────────────────────────────────

_CATALOG_KEYS = ["period", "tc", "rp_rs", "b", "a_rs", "inc_deg", "depth_ppm", "t14_hr", "snr",
                 "rp_rearth", "a_au", "teq_k", "hz_score"]


def flatten(res: dict) -> dict:
    """One flat CSV row from a fit result."""
    row = {k: res.get(k) for k in ["tic_id", "status", "method", "converged", "n_steps", "acceptance",
                                   "tau_max", "n_eff", "n_points", "n_in_transit", "sigma_ppm", "chi2_red",
                                   "size_class", "runtime_s"]}
    for k in _CATALOG_KEYS:
        s = res.get(k)
        if isinstance(s, dict):
            row[k] = s["med"]; row[f"{k}_lo"] = s["lo"]; row[f"{k}_hi"] = s["hi"]
    row["in_hz"] = (res.get("in_hz") or {}).get("med")
    return row


def _run_one(tic_id, row, out_dir, kw, force):
    jpath = out_dir / "fits" / f"{tic_id}.json"
    if jpath.exists() and not force:
        return json.loads(jpath.read_text())
    res, samp = fit_target(tic_id, row, **kw)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(res, indent=1, default=float))
    if samp is not None:
        (out_dir / "posteriors").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "posteriors" / f"{tic_id}.npz", samples=samp.astype(np.float32),
                            names=np.array(_SAMPLED))
    return res


def fit_batch(cands: pd.DataFrame, out_dir: Path = OUT_DIR, n_jobs: int = 1, force: bool = False,
              **kw) -> pd.DataFrame:
    """Fit every candidate (resumable: existing fits/<tic>.json are reused unless force)."""
    from joblib import Parallel, delayed
    out_dir = Path(out_dir)
    rows = [(r["tic_id"], r) for r in cands.to_dict("records")]
    log.info("Fitting %d candidates (n_jobs=%d)", len(rows), n_jobs)
    results = Parallel(n_jobs=n_jobs, verbose=5 if n_jobs != 1 else 0)(
        delayed(_run_one)(t, r, out_dir, kw, force) for t, r in rows)
    table = pd.DataFrame([flatten(r) for r in results])
    if "tic_id" in table and "p_transit" in cands:
        table = table.merge(cands[["tic_id", "p_transit"]], on="tic_id", how="left")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "fit_results.csv", index=False)
    ok = (table["status"] == "done").sum()
    log.info("Done: %d/%d fitted, %d converged MCMC. Wrote %s",
             ok, len(table), int((table["method"] == "mcmc").sum()), out_dir / "fit_results.csv")
    return table


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Stage 6a — orbital parameter fitting (batman + emcee)")
    ap.add_argument("--tic", action="append", help="TIC id to fit (repeatable)")
    ap.add_argument("--top", type=int, help="fit the N highest-probability candidates")
    ap.add_argument("--all-candidates", action="store_true", help="fit every candidate above --min-prob")
    ap.add_argument("--min-prob", type=float, default=0.5)
    ap.add_argument("--no-classifier", action="store_true", help="select by SDE/FAP instead of classifier")
    ap.add_argument("--include-synthetic", action="store_true")
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--walkers", type=int, default=N_WALKERS)
    ap.add_argument("--no-stellar-prior", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="refit even if outputs/fits/<tic>.json exists")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args()

    if a.tic:
        fm = pd.read_csv(CATALOGS_DIR / "feature_matrix.csv")
        fm["tic_id"] = fm["tic_id"].map(norm_tic)
        cands = fm[fm.tic_id.isin([norm_tic(t) for t in a.tic])]
        if cands.empty:
            raise SystemExit(f"None of {a.tic} found in feature_matrix.csv")
    elif a.top or a.all_candidates:
        cands = select_candidates(min_prob=a.min_prob, top=a.top, include_synthetic=a.include_synthetic,
                                  use_classifier=not a.no_classifier)
    else:
        ap.error("give --tic, --top N or --all-candidates")
    fit_batch(cands, a.out, n_jobs=a.n_jobs, force=a.force, n_walkers=a.walkers, n_steps=a.steps,
              max_steps=a.max_steps, use_stellar_prior=not a.no_stellar_prior)
