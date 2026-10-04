"""
Single-target pipeline service.

Runs one star through the existing stages and reports progress, so the API (and
later a Temporal activity, per SAAS_ARCHITECTURE_PLAN §7.4) can call it directly:

    locate -> preprocess -> detect_features -> classify -> finalize

Stage code is reused as-is: preprocessing.preprocess_one, features.extract_features_one
(TLS + the 65 model features), fitting.load_classifier.
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

BASE_DIR = _SRC.parent
RAW_DIR = BASE_DIR / "data" / "processed" / "lc_raw"
DETRENDED_DIR = BASE_DIR / "data" / "processed" / "lc_detrended"
MODEL_PATH = BASE_DIR / "models" / "classifier.joblib"
ANALYSES_DIR = BASE_DIR / "outputs" / "analyses"

log = logging.getLogger(__name__)

CLASS_NAMES = {0: "Transit", 1: "Eclipsing binary", 2: "Blend", 3: "Other"}
STAGES = [
    ("acquire", "Acquire light curve"),
    ("preprocess", "Clean & detrend"),
    ("detect_features", "Transit search & features"),
    ("classify", "Classify"),
    ("finalize", "Assemble result"),
]

_model = None
_model_lock = threading.Lock()


def norm_tic(t) -> str:
    t = str(t).strip()
    if t.upper().startswith(("SYN_", "UPL_")):
        return t[:4].upper() + t[4:]
    digits = t.upper().replace("TIC_", "").replace("TIC", "").strip(" _-")
    return f"TIC_{digits}"


def get_model():
    """Load the classifier once (the pickle needs the class registered under __main__)."""
    global _model
    with _model_lock:
        if _model is None:
            from fitting import load_classifier
            _model = load_classifier(MODEL_PATH)
        return _model


def score_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class probabilities for feature rows. Returns columns p_<class>, p_transit, pred."""
    model = get_model()
    X = df.reindex(columns=model.feature_names_)
    proba = model.predict_proba(X)
    out = pd.DataFrame(index=df.index)
    for i, c in enumerate(model.classes_):
        out[f"p_{int(c)}"] = proba[:, i]
    out["p_transit"] = out["p_0"] if "p_0" in out else np.nan
    out["pred"] = [int(model.classes_[k]) for k in np.argmax(proba, axis=1)]
    return out


def lightcurve_payload(tic: str, max_points: int = 900, fold_bins: int = 160) -> dict:
    """
    Downsampled light curve + phase fold for plotting. The fold is built from the
    TLS period/epoch, so pass them in via `ephemeris` after calling this if known.
    """
    from detection import select_search_window, MAX_SEGMENT_GAP_D
    d = np.load(DETRENDED_DIR / f"{norm_tic(tic)}.npz", allow_pickle=True)
    t, f = select_search_window(d["time"].astype(np.float64), d["flat_flux"].astype(np.float64), MAX_SEGMENT_GAP_D)
    seg = {"sector": int(d["sector"]) if "sector" in d else -1}
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    if len(t) == 0:
        raise ValueError("empty light curve")
    edges = np.linspace(t[0], t[-1] + 1e-9, min(max_points, len(t)) + 1)
    idx = np.digitize(t, edges) - 1
    n = len(edges) - 1
    bt, bf = [], []
    for k in range(n):
        m = idx == k
        if m.any():
            bt.append(float(np.mean(t[m])))
            bf.append(float(np.median(f[m])))
    return {"time": bt, "flux": bf, "t_min": float(t[0]), "t_max": float(t[-1]),
            "n_cadences": int(len(t)), "sector": seg["sector"], "_raw": (t, f)}


def fold_payload(t: np.ndarray, f: np.ndarray, period: float, t0: float,
                 dur_hr: Optional[float] = None, bins: int = 60, scatter: int = 900) -> Optional[dict]:
    """Phase fold zoomed on the transit: window = +-max(3.2 x duration, 6 h), binned inside it."""
    if not (np.isfinite(period) and period > 0 and np.isfinite(t0)):
        return None
    ph = ((t - t0 + 0.5 * period) % period) - 0.5 * period          # days from mid-transit
    hrs = ph * 24.0
    dur = dur_hr if (dur_hr and np.isfinite(dur_hr) and dur_hr > 0) else 4.0
    half = float(min(0.5 * period * 24.0, max(dur * 3.2, 6.0)))
    m = np.abs(hrs) <= half
    hrs, ff = hrs[m], f[m]
    if len(hrs) < 10:
        return None
    order = np.argsort(hrs)
    hrs, ff = hrs[order], ff[order]
    edges = np.linspace(-half, half, bins + 1)
    idx = np.clip(np.digitize(hrs, edges) - 1, 0, bins - 1)
    bx, by = [], []
    for k in range(bins):
        sel = idx == k
        if sel.sum() >= 2:
            bx.append(float(np.mean(hrs[sel])))
            by.append(float(np.median(ff[sel])))
    step = max(1, len(hrs) // scatter)
    return {"hours": [float(x) for x in hrs[::step]], "flux": [float(x) for x in ff[::step]],
            "bin_hours": bx, "bin_flux": by, "half_hours": half,
            "period_hours": float(period * 24.0)}


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def clean_row(d: dict) -> dict:
    return {k: _clean(v) for k, v in d.items()}


def fetch_from_mast(tic: str, max_sectors: int = 2) -> dict:
    """Download a SPOC 2-min light curve into lc_raw using the existing acquisition code."""
    try:
        import acquisition
    except Exception as e:                                           # noqa: BLE001
        return {"status": "failed", "reason": f"MAST client unavailable ({e})"}
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    digits = tic.replace("TIC_", "")
    r = acquisition._download_one_light_curve(digits, None, RAW_DIR, max_sectors=max_sectors)
    if r["status"] in ("downloaded", "skipped"):
        return {"status": "ok"}
    reason = str(r.get("reason", "unknown error"))
    if "no SPOC 2-min" in reason:
        reason = (f"{tic} has no SPOC 2-minute light curve on MAST (it may only appear in "
                  "full-frame images, which this pipeline doesn't use).")
    elif "too few good" in reason:
        reason = f"{tic}: {reason}"
    else:
        reason = f"Could not fetch {tic} from MAST: {reason}"
    return {"status": "failed", "reason": reason}


def save_analysis(result: dict) -> None:
    """Persist a finished analysis so the target page can reopen it later."""
    import json
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSES_DIR / f"{result['tic_id']}.json").write_text(json.dumps(result))


def load_analysis(tic: str) -> Optional[dict]:
    import json
    p = ANALYSES_DIR / f"{norm_tic(tic)}.json"
    return json.loads(p.read_text()) if p.exists() else None


def list_analyses(limit: int = 12) -> list:
    import json
    if not ANALYSES_DIR.exists():
        return []
    out = []
    for p in sorted(ANALYSES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            d = json.loads(p.read_text())
            out.append({"tic_id": d["tic_id"], "kind": d.get("kind"), "p_transit": d.get("p_transit"),
                        "pred": d.get("pred"), "when": p.stat().st_mtime})
        except Exception:                                            # noqa: BLE001
            continue
    return out


def run_target(tic_id: str, on_stage: Optional[Callable[[str, str, dict], None]] = None) -> dict:
    """
    Run one star through the pipeline. `on_stage(stage_id, state, info)` is called with
    state in {"start", "done", "skip", "error"}.
    Raises ValueError with a user-readable message when the star cannot be processed.
    A TIC that isn't on disk is fetched live from MAST; UPL_ ids come from ingest.save_upload.
    """
    emit = on_stage or (lambda *_: None)
    tic = norm_tic(tic_id)

    # 1. acquire: already on disk, an uploaded file, or live from MAST
    emit("acquire", "start", {})
    det_path = DETRENDED_DIR / f"{tic}.npz"
    raw_path = RAW_DIR / f"{tic}.npz"
    source = "detrended" if det_path.exists() else "disk" if raw_path.exists() else None
    if source is None:
        if not tic.startswith("TIC_"):
            emit("acquire", "error", {"reason": "not found"})
            raise ValueError(f"{tic} is not on this server. Upload the light curve again.")
        emit("acquire", "start", {"reason": "downloading from MAST…"})
        res = fetch_from_mast(tic)
        if res["status"] != "ok":
            emit("acquire", "error", {"reason": res["reason"]})
            raise ValueError(res["reason"])
        source = "mast"
    if tic.startswith("UPL_"):
        source = "upload"
    emit("acquire", "done", {"source": source})

    # 2. preprocess (only if we only have the raw file)
    if det_path.exists():
        emit("preprocess", "skip", {"reason": "already detrended"})
    else:
        emit("preprocess", "start", {})
        from preprocessing import preprocess_one
        DETRENDED_DIR.mkdir(parents=True, exist_ok=True)
        r = preprocess_one(raw_path, DETRENDED_DIR)
        if not det_path.exists():
            emit("preprocess", "error", {"reason": r.get("reason", "")})
            raise ValueError(f"{tic} failed quality checks: {r.get('reason') or 'unknown'}")
        emit("preprocess", "done", {"rms_detrended": r.get("rms_detrended")})

    # 3. TLS search + feature extraction
    emit("detect_features", "start", {})
    from features import extract_features_one
    row = extract_features_one(det_path)
    if row.get("status") != "done":
        emit("detect_features", "error", {"reason": str(row.get("status"))})
        raise ValueError(f"Feature extraction failed: {row.get('status')}")
    emit("detect_features", "done", {"period": _clean(row.get("period")), "SDE": _clean(row.get("SDE"))})

    # 4. classify
    emit("classify", "start", {})
    scores = score_rows(pd.DataFrame([row])).iloc[0].to_dict()
    emit("classify", "done", {"p_transit": _clean(scores.get("p_transit"))})

    # 5. finalize: plot payloads
    emit("finalize", "start", {})
    lc = lightcurve_payload(tic)
    t_raw, f_raw = lc.pop("_raw")
    fold = fold_payload(t_raw, f_raw, float(row.get("period", np.nan)), float(row.get("t0", np.nan)),
                        dur_hr=float(row.get("duration_hr", np.nan)))
    probs = {CLASS_NAMES[int(k[2:])]: _clean(v) for k, v in scores.items()
             if k.startswith("p_") and k[2:].isdigit()}
    result = {
        "tic_id": tic, "source": source,
        "kind": "upload" if tic.startswith("UPL_") else "synthetic" if tic.startswith("SYN_") else "real",
        "pred": CLASS_NAMES.get(int(scores["pred"]), str(scores["pred"])),
        "p_transit": _clean(scores.get("p_transit")),
        "probabilities": probs,
        "is_candidate": bool((scores.get("p_transit") or 0) >= 0.5),
        "features": clean_row({k: row.get(k) for k in (
            "period", "t0", "depth_ppm", "duration_hr", "SDE", "SNR", "FAP", "transit_count",
            "rp_rs", "odd_even_ratio", "secondary_ratio", "centroid_proxy",
            "stellar_Teff", "stellar_rad", "stellar_mass", "tess_mag", "hz_in_zone")}),
        "lightcurve": lc,
        "fold": fold,
    }
    save_analysis(result)
    emit("finalize", "done", {})
    return result
