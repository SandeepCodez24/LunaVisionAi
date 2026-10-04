"""Job persistence. Run: .venv/bin/python -m pytest tests -q"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.jobstore import JobStore  # noqa: E402

STAGES = [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}]


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "jobs.db")
    yield s
    s.close()


def test_create_and_get_roundtrip(store):
    j = store.create("t1", "TIC_123", STAGES)
    got = store.get(j["id"])
    assert got["status"] == "queued" and got["tenant_id"] == "t1" and got["result"] is None
    assert [s["state"] for s in got["stages"]] == ["pending", "pending"]


def test_stage_updates_bump_version_and_persist(store):
    j = store.create("t1", "TIC_123", STAGES)
    v0 = store.version(j["id"])
    store.update_stage(j["id"], "a", "done", {"source": "mast"})
    got = store.get(j["id"])
    assert got["stages"][0] == {"id": "a", "label": "A", "state": "done", "info": {"source": "mast"}}
    assert got["stages"][1]["state"] == "pending" and got["version"] > v0


def test_done_and_failed_set_finished_and_keep_result(store):
    ok, bad = store.create("t", "TIC_1", STAGES), store.create("t", "TIC_2", STAGES)
    store.set_status(ok["id"], "done", result={"p_transit": 0.9})
    store.set_status(bad["id"], "failed", error="no data")
    a, b = store.get(ok["id"]), store.get(bad["id"])
    assert a["result"] == {"p_transit": 0.9} and a["finished"] and b["error"] == "no data" and b["result"] is None
    store.set_status(ok["id"], "done")                    # a later status write must not wipe the result
    assert store.get(ok["id"])["result"] == {"p_transit": 0.9}


def test_jobs_survive_reopening_the_database(tmp_path):
    s1 = JobStore(tmp_path / "j.db")
    j = s1.create("t", "TIC_9", STAGES); s1.set_status(j["id"], "done", result={"x": 1}); s1.close()
    s2 = JobStore(tmp_path / "j.db")
    assert s2.get(j["id"])["result"] == {"x": 1}
    s2.close()


def test_recover_requeues_interrupted_jobs_only(tmp_path):
    s = JobStore(tmp_path / "j.db")
    running, queued, done = (s.create("t", f"TIC_{i}", STAGES) for i in range(3))
    s.update_stage(running["id"], "a", "done", {}); s.set_status(running["id"], "running")
    s.set_status(done["id"], "done", result={"ok": True})
    ids = s.recover()
    assert ids == [running["id"], queued["id"]]                       # oldest first, done job untouched
    r = s.get(running["id"])
    assert r["status"] == "queued" and all(x["state"] == "pending" for x in r["stages"])
    assert s.get(done["id"])["status"] == "done"
    s.close()


def test_list_is_tenant_scoped_and_newest_first(store):
    a = store.create("t1", "TIC_1", STAGES); b = store.create("t1", "TIC_2", STAGES); store.create("t2", "TIC_3", STAGES)
    ids = [x["id"] for x in store.list("t1")]
    assert ids == [b["id"], a["id"]]


def test_prune_keeps_newest_and_never_deletes_active(store):
    old_done = store.create("t", "TIC_1", STAGES); store.set_status(old_done["id"], "done")
    active = store.create("t", "TIC_2", STAGES)
    for i in range(3):
        j = store.create("t", f"TIC_{10+i}", STAGES); store.set_status(j["id"], "done")
    store.prune(keep=2)
    assert store.get(old_done["id"]) is None and store.get(active["id"]) is not None


def test_unknown_job(store):
    assert store.get("nope") is None and store.version("nope") is None
    store.update_stage("nope", "a", "done", {})            # must not raise
