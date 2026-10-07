"""
LunaVisionAI API (SAAS_ARCHITECTURE_PLAN Phase 1: pipeline wrapped as async jobs + web app).

Run from exoplanet_pipeline/:
    .venv/bin/uvicorn api.main:app --reload
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
import json
import math
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

import ingest  # noqa: E402
import visualization  # noqa: E402
import pipeline_service as ps  # noqa: E402
from .jobstore import JobStore  # noqa: E402

FRONTEND = BASE_DIR / "frontend"
CATALOGS = BASE_DIR / "data" / "catalogs"
OUTPUTS = ps.OUTPUTS_DIR
LABEL_NAMES = ps.CLASS_NAMES

@asynccontextmanager
async def lifespan(_app):
    """On startup, re-queue jobs a previous run didn't finish; keep the table bounded."""
    store.prune()
    for jid in store.recover():
        _executor.submit(_run_job, jid)
    yield


app = FastAPI(title="LunaVisionAI", version="0.1.0", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────────────────────
# Data access (feature matrix + cached classifier scores)
# ─────────────────────────────────────────────────────────────────────────────
_table_lock = threading.Lock()
_table: Optional[pd.DataFrame] = None
_table_key: tuple = ()
_scoring_error: Optional[str] = None


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def ok(payload, status: int = 200) -> JSONResponse:
    return JSONResponse(_json_safe(payload), status_code=status)


def get_table() -> pd.DataFrame:
    """Feature matrix with p_transit / predicted class, rebuilt when the CSV changes."""
    global _table, _table_key, _scoring_error
    path = CATALOGS / "feature_matrix.csv"
    with _table_lock:
        mt = tuple(p.stat().st_mtime if p.exists() else 0.0 for p in
                   (path, ps.MODEL_PATH, ps.MODELS_DIR / "split.json", ps.MODELS_DIR / "metrics.json"))
        if _table is None or mt != _table_key:
            ps._model = None                                             # pick up a retrained model
            df = pd.read_csv(path)
            df = df[df["status"].astype(str).eq("done")].copy()
            df["tic_id"] = df["tic_id"].map(ps.norm_tic)
            df["kind"] = np.where(df["tic_id"].str.startswith("SYN_"), "synthetic", "real")
            try:
                sc = ps.score_rows(df)
                for c in sc.columns:
                    df[c] = sc[c].values
                _scoring_error = None
            except Exception as e:                                       # noqa: BLE001
                _scoring_error = f"{type(e).__name__}: {e}"
                df["p_transit"] = np.nan
                df["pred"] = np.nan
                df["detected"] = df["period"].notna()
            df["split"] = df["tic_id"].map(ps.split_of)
            _table, _table_key = df.reset_index(drop=True), mt
        return _table


def _row_summary(r: pd.Series) -> dict:
    return {
        "tic_id": r["tic_id"], "kind": r["kind"],
        "label": None if pd.isna(r["label"]) else int(r["label"]),
        "label_name": LABEL_NAMES.get(int(r["label"])) if pd.notna(r["label"]) and r["label"] >= 0 else None,
        "pred": (LABEL_NAMES.get(int(r["pred"])) if pd.notna(r.get("pred")) else
                 "No detection" if not r.get("detected", True) else None),
        "detected": bool(r.get("detected", True)), "split": r.get("split"),
        "p_transit": r.get("p_transit"), "period": r.get("period"), "depth_ppm": r.get("depth_ppm"),
        "duration_hr": r.get("duration_hr"), "SDE": r.get("SDE"), "SNR": r.get("SNR"),
        "sector": None if pd.isna(r.get("sector")) else int(r["sector"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Read endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": ps._model is not None, "scoring_error": _scoring_error}


@app.get("/api/overview")
def overview():
    df = get_table()
    real = df[df["kind"] == "real"]
    metrics = {}
    mp = ps.MODELS_DIR / "metrics.json"
    if mp.exists():
        metrics = json.loads(mp.read_text())
    by_label = {LABEL_NAMES.get(int(k), str(k)): int(v) for k, v in real["label"].value_counts().items()}
    return ok({
        "total": int(len(df)), "real": int((df["kind"] == "real").sum()),
        "synthetic": int((df["kind"] == "synthetic").sum()),
        "candidates": int((real["p_transit"] >= 0.5).sum()) if real["p_transit"].notna().any() else None,
        "by_label": by_label,
        "split": {"status": "valid" if ps.load_split() else "unknown",
                  "held_out": int((df["split"] == "held-out").sum()), "trained_on": int((df["split"] == "trained-on").sum())},
        "no_detection": int((~df["detected"]).sum()) if "detected" in df else None,
        "model": {"test": metrics.get("test"), "top_features": metrics.get("top_features"),
                  "base_learners": metrics.get("base_learners"), "n_features": metrics.get("n_features")},
        "scoring_error": _scoring_error,
    })


@app.get("/api/targets")
def targets(q: str = "", kind: str = "real", label: Optional[int] = None,
            min_p: float = Query(0.0, ge=0, le=1), split: Optional[str] = None,
            sort: str = "p_transit", order: str = "desc",
            limit: int = Query(40, ge=1, le=200), offset: int = Query(0, ge=0)):
    df = get_table()
    if kind in ("real", "synthetic"):
        df = df[df["kind"] == kind]
    if label is not None:
        df = df[df["label"] == label]
    if q.strip():
        needle = re.sub(r"\D", "", q) or q.strip().upper()
        df = df[df["tic_id"].str.contains(needle, case=False, regex=False)]
    if split in ("held-out", "trained-on"):
        df = df[df["split"] == split]
    if min_p > 0:
        df = df[df["p_transit"] >= min_p]
    if sort not in ("p_transit", "SDE", "SNR", "period", "depth_ppm", "duration_hr", "tic_id"):
        raise HTTPException(400, f"cannot sort by {sort}")
    df = df.sort_values(sort, ascending=(order == "asc"), na_position="last")
    page = df.iloc[offset: offset + limit]
    return ok({"total": int(len(df)), "offset": offset,
               "items": [_row_summary(r) for _, r in page.iterrows()]})


@app.get("/api/targets/{tic}")
def target_detail(tic: str):
    df = get_table()
    t = ps.norm_tic(tic)
    hit = df[df["tic_id"] == t]
    if hit.empty:
        saved = ps.load_analysis(t)
        if saved is None:
            raise HTTPException(404, f"{t} is not in the feature table and has no saved analysis")
        return ok({**saved, "label": None, "label_name": None, "sector": saved["lightcurve"].get("sector"),
                   "split": ps.split_of(t), "has_report": (OUTPUTS / "reports" / f"{t}.pdf").exists(),
                   "plots": visualization.available_plots(OUTPUTS, t), "vetting": ps.load_summary(t)})
    r = hit.iloc[0]
    try:
        lc = ps.lightcurve_payload(t)
    except Exception as e:                                               # noqa: BLE001
        raise HTTPException(404, f"light curve unavailable: {e}")
    t_raw, f_raw = lc.pop("_raw")
    fold = ps.fold_payload(t_raw, f_raw, float(r.get("period", np.nan)), float(r.get("t0", np.nan)),
                           dur_hr=float(r.get("duration_hr", np.nan)))
    probs = {LABEL_NAMES[int(c[2:])]: r[c] for c in r.index if c.startswith("p_") and c[2:].isdigit()}
    feat_keys = ("period", "t0", "depth_ppm", "duration_hr", "SDE", "SNR", "FAP", "transit_count",
                 "rp_rs", "odd_even_ratio", "secondary_ratio", "centroid_proxy", "flat_bottom_score",
                 "stellar_Teff", "stellar_rad", "stellar_mass", "stellar_dist_pc", "tess_mag", "hz_in_zone")
    return ok({**_row_summary(r), "probabilities": probs,
               "features": {k: r.get(k) for k in feat_keys},
               "lightcurve": lc, "fold": fold,
               "has_report": (OUTPUTS / "reports" / f"{t}.pdf").exists(),
               "plots": visualization.available_plots(OUTPUTS, t), "vetting": ps.load_summary(t)})


_TIC_RE = r"(TIC_\d{3,12}|SYN_\d+|UPL_[0-9a-f]{8})"


@app.post("/api/targets/{tic}/report", status_code=202)
def create_report(tic: str, force: bool = False):
    """Fit the transit model and build the vetting plots + PDF in the background (about a minute)."""
    t = ps.norm_tic(tic)
    if not re.fullmatch(_TIC_RE, t):
        raise HTTPException(422, "not a valid target id")
    if not force and (OUTPUTS / "reports" / f"{t}.pdf").exists() and ps.load_summary(t):
        return ok({"ready": True, "id": None, "tic_id": t}, 200)
    job = _enqueue(t, "report")
    return ok({"ready": False, "id": job["id"], "tic_id": t}, 202)


@app.get("/api/targets/{tic}/plots/{name}")
def target_plot(tic: str, name: str):
    p = visualization.plot_path(OUTPUTS, ps.norm_tic(tic), name.removesuffix(".png"))
    if p is None:
        raise HTTPException(404, "no such plot")
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": "no-cache"})


@app.get("/api/targets/{tic}/report")
def target_report(tic: str):
    t = ps.norm_tic(tic)
    pdf = OUTPUTS / "reports" / f"{t}.pdf"
    if not pdf.exists():
        raise HTTPException(404, "no report yet: use \"Generate report\" on the target page")
    return FileResponse(pdf, media_type="application/pdf", filename=f"{t}_report.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Async jobs (tenant_id carried from day one, per plan §9 Phase 1)
# ─────────────────────────────────────────────────────────────────────────────
class JobRequest(BaseModel):
    tic_id: str


TENANT = "local"                                       # real tenancy arrives with auth (Phase 2)
store = JobStore(Path(os.environ.get("LUNA_JOBS_DB", ps.BASE_DIR / "outputs" / "jobs.sqlite3")))
_executor = ThreadPoolExecutor(max_workers=1)        # TLS is heavy; one job at a time locally


def _run_job(job_id: str):
    job = store.get(job_id)
    if job is None:
        return

    def on_stage(stage_id: str, state: str, info: dict):
        store.update_stage(job_id, stage_id, state, _json_safe(info))

    store.set_status(job_id, "running")
    try:
        if job["kind"] == "report":
            result = ps.run_fit_report(job["tic_id"], on_stage)
        else:
            result = ps.run_target(job["tic_id"], on_stage)
        store.set_status(job_id, "done", result=_json_safe(result))
    except Exception as e:                                               # noqa: BLE001
        msg = str(e) or type(e).__name__
        store.fail_running_stage(job_id, msg)
        store.set_status(job_id, "failed", error=msg)


def _enqueue(tic: str, kind: str = "analyze") -> dict:
    stages = ps.REPORT_STAGES if kind == "report" else ps.STAGES
    job = store.create(TENANT, tic, [{"id": sid, "label": lbl} for sid, lbl in stages], kind=kind)
    _executor.submit(_run_job, job["id"])
    return job


@app.post("/api/jobs", status_code=202)
def create_job(req: JobRequest):
    tic = ps.norm_tic(req.tic_id)
    if not re.fullmatch(r"(TIC_\d{3,12}|SYN_\d+|UPL_[0-9a-f]{8})", tic):
        raise HTTPException(422, "Enter a TIC ID such as 231663901")
    job = _enqueue(tic)
    return ok({"id": job["id"], "tic_id": tic}, 202)


@app.get("/api/jobs")
def list_jobs(limit: int = Query(20, ge=1, le=100)):
    return ok({"items": store.list(TENANT, limit)})


@app.post("/api/jobs/upload", status_code=202)
async def create_upload_job(file: UploadFile = File(...)):
    """Upload a CSV or TESS-style FITS light curve; it is validated now, analysed in the background."""
    data = await file.read(ingest.MAX_BYTES + 1)
    try:
        uid = await run_in_threadpool(ingest.save_upload, file.filename or "", data, ps.RAW_DIR)
    except ingest.IngestError as e:
        raise HTTPException(422, str(e))
    job = _enqueue(uid)
    return ok({"id": job["id"], "tic_id": uid}, 202)


@app.get("/api/analyses")
def analyses(limit: int = Query(12, ge=1, le=50)):
    return ok({"items": ps.list_analyses(limit)})


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return ok(job)


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    if store.version(job_id) is None:
        raise HTTPException(404, "unknown job")

    async def gen():
        seen = -1
        while True:
            v = store.version(job_id)
            if v is None:                                  # pruned while streaming
                break
            if v != seen:
                seen = v
                snap = store.get(job_id)
                yield f"data: {json.dumps(_json_safe(snap))}\n\n"
                if snap["status"] in ("done", "failed"):
                    break
            await asyncio.sleep(0.25)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─────────────────────────────────────────────────────────────────────────────
# Frontend
# ─────────────────────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=FRONTEND / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><circle cx="16" cy="16" r="12" fill="#d99a35"/>'
           '<circle cx="22" cy="14" r="5" fill="#0a0b14"/></svg>')
    return Response(svg, media_type="image/svg+xml")


@app.get("/target")
def target_page():
    return FileResponse(FRONTEND / "target.html")


@app.get("/log")
def legacy_log():
    return FileResponse(FRONTEND / "transit_log.html")
