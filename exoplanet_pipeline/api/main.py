"""
LunaVisionAI API (SAAS_ARCHITECTURE_PLAN Phase 1: pipeline wrapped as async jobs + web app).

Run from exoplanet_pipeline/:
    .venv/bin/uvicorn api.main:app --reload
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import sys
import threading
import time
import uuid
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
import pipeline_service as ps  # noqa: E402

FRONTEND = BASE_DIR / "frontend"
CATALOGS = BASE_DIR / "data" / "catalogs"
OUTPUTS = BASE_DIR / "outputs"
LABEL_NAMES = ps.CLASS_NAMES

app = FastAPI(title="LunaVisionAI", version="0.1.0")


# ─────────────────────────────────────────────────────────────────────────────
# Data access (feature matrix + cached classifier scores)
# ─────────────────────────────────────────────────────────────────────────────
_table_lock = threading.Lock()
_table: Optional[pd.DataFrame] = None
_table_mtime = 0.0
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
    global _table, _table_mtime, _scoring_error
    path = CATALOGS / "feature_matrix.csv"
    with _table_lock:
        mt = path.stat().st_mtime
        if _table is None or mt != _table_mtime:
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
            _table, _table_mtime = df.reset_index(drop=True), mt
        return _table


def _row_summary(r: pd.Series) -> dict:
    return {
        "tic_id": r["tic_id"], "kind": r["kind"],
        "label": None if pd.isna(r["label"]) else int(r["label"]),
        "label_name": LABEL_NAMES.get(int(r["label"])) if pd.notna(r["label"]) and r["label"] >= 0 else None,
        "pred": None if pd.isna(r.get("pred")) else LABEL_NAMES.get(int(r["pred"])),
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
    mp = BASE_DIR / "models" / "metrics.json"
    if mp.exists():
        metrics = json.loads(mp.read_text())
    by_label = {LABEL_NAMES.get(int(k), str(k)): int(v) for k, v in real["label"].value_counts().items()}
    return ok({
        "total": int(len(df)), "real": int((df["kind"] == "real").sum()),
        "synthetic": int((df["kind"] == "synthetic").sum()),
        "candidates": int((real["p_transit"] >= 0.5).sum()) if real["p_transit"].notna().any() else None,
        "by_label": by_label,
        "model": {"test": metrics.get("test"), "top_features": metrics.get("top_features"),
                  "base_learners": metrics.get("base_learners"), "n_features": metrics.get("n_features")},
        "scoring_error": _scoring_error,
    })


@app.get("/api/targets")
def targets(q: str = "", kind: str = "real", label: Optional[int] = None,
            min_p: float = Query(0.0, ge=0, le=1),
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
                   "pred": saved.get("pred"), "has_report": False})
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
               "has_report": (OUTPUTS / "reports" / f"{t}.pdf").exists()})


@app.get("/api/targets/{tic}/report")
def target_report(tic: str):
    t = ps.norm_tic(tic)
    pdf = OUTPUTS / "reports" / f"{t}.pdf"
    if not pdf.exists():
        raise HTTPException(404, "no report generated for this target (run src/report.py)")
    return FileResponse(pdf, media_type="application/pdf", filename=f"{t}_report.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Async jobs (tenant_id carried from day one, per plan §9 Phase 1)
# ─────────────────────────────────────────────────────────────────────────────
class JobRequest(BaseModel):
    tic_id: str


_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)        # TLS is heavy; one job at a time locally


def _new_job(tic: str) -> dict:
    return {"id": uuid.uuid4().hex[:12], "tenant_id": "local", "tic_id": tic,
            "status": "queued", "created": time.time(), "finished": None,
            "stages": [{"id": sid, "label": lbl, "state": "pending", "info": {}}
                       for sid, lbl in ps.STAGES],
            "result": None, "error": None, "version": 0}


def _touch(job: dict):
    job["version"] += 1


def _run_job(job: dict):
    def on_stage(stage_id: str, state: str, info: dict):
        with _jobs_lock:
            for st in job["stages"]:
                if st["id"] == stage_id:
                    st["state"], st["info"] = state, _json_safe(info)
            _touch(job)

    with _jobs_lock:
        job["status"] = "running"
        _touch(job)
    try:
        result = ps.run_target(job["tic_id"], on_stage)
        with _jobs_lock:
            job.update(status="done", result=_json_safe(result), finished=time.time())
            _touch(job)
    except Exception as e:                                               # noqa: BLE001
        with _jobs_lock:
            job.update(status="failed", error=str(e), finished=time.time())
            _touch(job)


@app.post("/api/jobs", status_code=202)
def create_job(req: JobRequest):
    tic = ps.norm_tic(req.tic_id)
    if not re.fullmatch(r"(TIC_\d{3,12}|SYN_\d+|UPL_[0-9a-f]{8})", tic):
        raise HTTPException(422, "Enter a TIC ID such as 231663901")
    job = _enqueue(tic)
    return ok({"id": job["id"], "tic_id": tic}, 202)


def _enqueue(tic: str) -> dict:
    job = _new_job(tic)
    with _jobs_lock:
        _jobs[job["id"]] = job
        if len(_jobs) > 200:
            for k in sorted(_jobs, key=lambda k: _jobs[k]["created"])[:50]:
                _jobs.pop(k, None)
    _executor.submit(_run_job, job)
    return job


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


def _public(job: dict, with_result: bool = True) -> dict:
    d = {k: v for k, v in job.items() if k != "result"}
    if with_result:
        d["result"] = job["result"]
    return d


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    with _jobs_lock:
        return ok(_public(job))


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")

    async def gen():
        seen = -1
        while True:
            with _jobs_lock:
                v = job["version"]
                snap = _public(job) if v != seen else None
                finished = job["status"] in ("done", "failed")
            if snap is not None:
                seen = v
                yield f"data: {json.dumps(_json_safe(snap))}\n\n"
            if finished and snap is not None:
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
