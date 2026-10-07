"""Held-out split, calibration, and no-detection gating. Run: .venv/bin/python -m pytest tests -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import classifier as C  # noqa: E402
import pipeline_service as ps  # noqa: E402


def _toy(n_real=200, n_syn=40, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.array([str(1000 + i) for i in range(n_real)] + [str(1000 + int(i)) for i in rng.integers(0, n_real, n_syn)])
    is_syn = np.r_[np.zeros(n_real, bool), np.ones(n_syn, bool)]
    y = np.r_[(rng.random(n_real) < 0.65).astype(int) * 3, np.zeros(n_syn, int)]   # labels 0 / 3, like the real data
    y[:n_real] = 3 - y[:n_real]
    return y, groups, is_syn


def test_split_has_no_star_in_both_sets_and_is_reproducible():
    y, groups, is_syn = _toy()
    tr, te = C.group_split(y, groups, is_syn)
    assert not (set(groups[tr]) & set(groups[te]))
    assert not is_syn[te].any()                                       # held-out set is real stars only
    tr2, te2 = C.group_split(y, groups, is_syn)
    assert np.array_equal(tr, tr2) and np.array_equal(te, te2)


def test_manifest_matches_split_and_id_changes_with_split():
    y, groups, is_syn = _toy()
    tr, te = C.group_split(y, groups, is_syn)
    m = C.split_manifest(groups, is_syn, tr, te, len(y))
    assert set(m["test"]).isdisjoint(m["train"]) and all(t.startswith("TIC_") for t in m["test"] + m["train"])
    assert len(m["test"]) == len(set(groups[te]))
    tr3, te3 = C.group_split(y, groups, is_syn, seed=7)
    assert C.split_manifest(groups, is_syn, tr3, te3, len(y))["split_id"] != m["split_id"]


def test_calibration_bins_perfect_and_overconfident():
    rng = np.random.default_rng(1)
    p = rng.random(5000)
    perfect = C.calibration_bins((rng.random(5000) < p).astype(int), p)
    assert perfect["ece"] < 0.04 and sum(b["n"] for b in perfect["bins"]) == 5000
    over = C.calibration_bins(np.zeros(1000), np.full(1000, 0.9))      # claims 90%, never right
    assert over["ece"] == pytest.approx(0.9) and over["bins"][0]["frac_pos"] == 0.0


class _Stub:
    classes_ = np.array([0, 3]); feature_names_ = ["period", "x"]
    def predict_proba(self, X):
        return np.tile([0.93, 0.07], (len(X), 1))                      # would call everything "Transit"


def test_stars_without_a_detection_get_no_score(monkeypatch):
    monkeypatch.setattr(ps, "_model", _Stub())
    out = ps.score_rows(pd.DataFrame({"period": [3.2, np.nan], "x": [1.0, 2.0]}))
    assert out["detected"].tolist() == [True, False]
    assert out.loc[0, "p_transit"] == pytest.approx(0.93)
    assert np.isnan(out.loc[1, "p_transit"]) and np.isnan(out.loc[1, "pred"])


def _write(models, manifest_id, metrics_id):
    models.mkdir(exist_ok=True)
    (models / "split.json").write_text(json.dumps({"split_id": manifest_id, "test": ["TIC_1"], "train": ["TIC_2"]}))
    (models / "metrics.json").write_text(json.dumps({"split_id": metrics_id}))


def test_split_is_only_trusted_when_it_belongs_to_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "MODELS_DIR", tmp_path / "m")
    ps._split_cache.update(key=None, val=None)
    assert ps.split_of("TIC_1") == "unknown"                          # nothing on disk
    _write(tmp_path / "m", "abc", "abc"); ps._split_cache.update(key=None, val=None)
    assert [ps.split_of(t) for t in ("TIC_1", "TIC_2", "TIC_9", "SYN_0001")] == ["held-out", "trained-on", "unseen", "synthetic"]
    _write(tmp_path / "m", "abc", "zzz"); ps._split_cache.update(key=None, val=None)
    assert ps.split_of("TIC_1") == "unknown"                          # split.json from a different training run
    _write(tmp_path / "m", "abc", None); ps._split_cache.update(key=None, val=None)
    assert ps.split_of("TIC_1") == "unknown"                          # old metrics.json with no split_id
