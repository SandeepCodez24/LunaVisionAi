"""Plot paths, job-store migration, run_pipeline stage logic, candidate rows. Run: .venv/bin/python -m pytest tests -q"""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import pipeline_service as ps  # noqa: E402
import run_pipeline as rp  # noqa: E402
import visualization as viz  # noqa: E402
from api.jobstore import JobStore  # noqa: E402


# ── visualization ────────────────────────────────────────────────────────────
def test_plot_path_only_serves_known_plot_names(tmp_path):
    d = tmp_path / "plots" / "TIC_1"; d.mkdir(parents=True)
    (d / "3_fold.png").write_bytes(b"png"); (d / "secret.png").write_bytes(b"x")
    assert viz.plot_path(tmp_path, "TIC_1", "3_fold") == d / "3_fold.png"
    for bad in ("../../etc/passwd", "secret", "9_fold", "3_fold.png", "", "3_FOLD"):
        assert viz.plot_path(tmp_path, "TIC_1", bad) is None
    assert viz.plot_path(tmp_path, "TIC_1", "4_corner") is None                 # valid name, file missing
    assert viz.available_plots(tmp_path, "TIC_1") == ["3_fold"]


# ── job store ────────────────────────────────────────────────────────────────
def test_jobstore_migrates_a_database_created_before_kind_existed(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""CREATE TABLE jobs (id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, tic_id TEXT NOT NULL,
        status TEXT NOT NULL, created REAL NOT NULL, finished REAL, stages TEXT NOT NULL, result TEXT, error TEXT,
        version INTEGER NOT NULL DEFAULT 0);
        INSERT INTO jobs (id, tenant_id, tic_id, status, created, stages) VALUES ('old1','t','TIC_5','done',1.0,'[]');""")
    con.commit(); con.close()
    s = JobStore(db)
    assert s.get("old1")["kind"] == "analyze"                                    # old rows default to analyze
    j = s.create("t", "TIC_6", [{"id": "fit", "label": "Fit"}], kind="report")
    assert s.get(j["id"])["kind"] == "report" and s.list("t")[0]["kind"] == "report"
    s.close()


def test_a_job_that_dies_mid_stage_shows_that_stage_as_failed(tmp_path):
    s = JobStore(tmp_path / "j.db")
    j = s.create("t", "TIC_1", [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}])
    s.update_stage(j["id"], "a", "done", {}); s.update_stage(j["id"], "b", "start", {})
    s.fail_running_stage(j["id"], "boom")
    st = {x["id"]: x for x in s.get(j["id"])["stages"]}
    assert st["a"]["state"] == "done" and st["b"]["state"] == "error" and st["b"]["info"]["reason"] == "boom"
    s.close()


# ── run_pipeline ─────────────────────────────────────────────────────────────
def test_acquire_is_opt_in_and_stages_run_in_pipeline_order(monkeypatch):
    assert "acquire" not in rp.DEFAULT_STAGES and rp.DEFAULT_STAGES == [s for s in rp.ALL_STAGES if s != "acquire"]
    calls = []
    monkeypatch.setattr(rp, "STAGE_FUNCS", {n: (lambda a, n=n: calls.append(n) or n) for n in rp.ALL_STAGES})
    out = rp.run_stages(["report", "features", "preprocess"], object())        # given out of order
    assert calls == ["preprocess", "features", "report"] and [o[0] for o in out] == calls


def test_a_failing_stage_stops_the_pipeline(monkeypatch):
    calls = []
    def boom(a): raise RuntimeError("no candidates")
    funcs = {n: (lambda a, n=n: calls.append(n) or "ok") for n in rp.ALL_STAGES}; funcs["train"] = boom
    monkeypatch.setattr(rp, "STAGE_FUNCS", funcs)
    with pytest.raises(SystemExit, match="Stopped at stage 'train'"):
        rp.run_stages(["preprocess", "train", "fit"], object())
    assert calls == ["preprocess"]                                               # fit never ran


def test_unknown_stage_is_rejected_with_the_valid_choices(capsys):
    with pytest.raises(SystemExit) as e:
        rp.main(["--stages", "preprocess,nonsense"])
    assert e.value.code == 2 and "nonsense" in capsys.readouterr().err


# ── candidate rows for fitting ───────────────────────────────────────────────
def _analysis(tmp_path, monkeypatch, features, p=0.7):
    monkeypatch.setattr(ps, "ANALYSES_DIR", tmp_path / "analyses")
    monkeypatch.setattr(ps, "FEATURES_CSV", tmp_path / "nope.csv")               # not in the feature matrix
    ps.save_analysis({"tic_id": "UPL_0badc0de", "features": features, "p_transit": p})


def test_uploaded_star_row_comes_from_its_saved_analysis(tmp_path, monkeypatch):
    _analysis(tmp_path, monkeypatch, {"period": 4.0, "t0": 1.0, "duration_hr": 2.6, "depth_ppm": 16000.0, "rp_rs": 0.12})
    row, p = ps.candidate_row("upl_0badc0de")
    assert row["period"] == 4.0 and row["depth"] == pytest.approx(0.016) and p == 0.7


def test_star_without_a_period_cannot_be_fitted(tmp_path, monkeypatch):
    _analysis(tmp_path, monkeypatch, {"period": None, "t0": None})
    with pytest.raises(ValueError, match="No transit period"):
        ps.candidate_row("UPL_0badc0de")


def test_unanalysed_star_is_reported_as_such(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "ANALYSES_DIR", tmp_path / "a"); monkeypatch.setattr(ps, "FEATURES_CSV", tmp_path / "nope.csv")
    with pytest.raises(ValueError, match="not been analysed"):
        ps.candidate_row("TIC_123456")


def test_summary_is_json_safe_and_keeps_the_verdict():
    r = {"tic_id": "TIC_1", "pdf": "x.pdf", "res": {"method": "mcmc", "converged": True, "size_class": "Neptune-size",
         "period": {"med": 4.0, "lo": 3.9, "hi": float("nan")}}, "vet": {"verdict": "PASS", "flags": [], "checks": [("Odd/even depth", "PASS", "ok")]}}
    s = ps.summarize_report(r)
    assert json.loads(json.dumps(s))["verdict"] == "PASS" and s["fit"]["period"]["hi"] is None and s["has_pdf"]
