"""
SQLite-backed job store (SAAS_ARCHITECTURE_PLAN Phase 1: "Postgres job tracking").

SQLite is the stdlib stand-in for the plan's Postgres: same shape (tenant_id on every
row), no extra service to run. Swapping to Postgres later means reimplementing this
one class. Jobs survive a server restart; jobs that were queued or running when the
server stopped are re-queued by `recover()`.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id        TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    tic_id    TEXT NOT NULL,
    status    TEXT NOT NULL,
    created   REAL NOT NULL,
    finished  REAL,
    stages    TEXT NOT NULL,
    result    TEXT,
    error     TEXT,
    version   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS jobs_tenant_created ON jobs (tenant_id, created DESC);
"""

ACTIVE = ("queued", "running")


class JobStore:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        if "kind" not in {r["name"] for r in self._db.execute("PRAGMA table_info(jobs)")}:
            self._db.execute("ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'analyze'")

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        return {"id": r["id"], "tenant_id": r["tenant_id"], "tic_id": r["tic_id"], "kind": r["kind"], "status": r["status"],
                "created": r["created"], "finished": r["finished"], "stages": json.loads(r["stages"]),
                "result": json.loads(r["result"]) if r["result"] else None,
                "error": r["error"], "version": r["version"]}

    @staticmethod
    def _blank_stages(stages: list) -> list:
        return [{"id": s["id"], "label": s["label"], "state": "pending", "info": {}} for s in stages]

    # ── writes ───────────────────────────────────────────────────────────────
    def create(self, tenant_id: str, tic_id: str, stages: list, kind: str = "analyze") -> dict:
        jid = uuid.uuid4().hex[:12]
        with self._lock:
            self._db.execute(
                "INSERT INTO jobs (id, tenant_id, tic_id, kind, status, created, stages) VALUES (?,?,?,?,?,?,?)",
                (jid, tenant_id, tic_id, kind, "queued", time.time(), json.dumps(self._blank_stages(stages))))
        return self.get(jid)

    def update_stage(self, job_id: str, stage_id: str, state: str, info: dict) -> None:
        with self._lock:
            row = self._db.execute("SELECT stages FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return
            stages = json.loads(row["stages"])
            for s in stages:
                if s["id"] == stage_id:
                    s["state"], s["info"] = state, info
            self._db.execute("UPDATE jobs SET stages=?, version=version+1 WHERE id=?",
                             (json.dumps(stages), job_id))

    def fail_running_stage(self, job_id: str, reason: str) -> None:
        """When a job dies mid-stage, show that stage as failed instead of forever running."""
        with self._lock:
            row = self._db.execute("SELECT stages FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return
            stages = json.loads(row["stages"])
            for st in stages:
                if st["state"] == "start":
                    st["state"], st["info"] = "error", {"reason": reason}
            self._db.execute("UPDATE jobs SET stages=? WHERE id=?", (json.dumps(stages), job_id))

    def set_status(self, job_id: str, status: str, result: Optional[dict] = None,
                   error: Optional[str] = None) -> None:
        finished = time.time() if status in ("done", "failed") else None
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status=?, finished=COALESCE(?, finished), result=COALESCE(?, result), "
                "error=COALESCE(?, error), version=version+1 WHERE id=?",
                (status, finished, json.dumps(result) if result is not None else None, error, job_id))

    # ── reads ────────────────────────────────────────────────────────────────
    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            r = self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row(r) if r else None

    def version(self, job_id: str) -> Optional[int]:
        """Cheap change check for the event stream (avoids re-parsing the result JSON)."""
        with self._lock:
            r = self._db.execute("SELECT version FROM jobs WHERE id=?", (job_id,)).fetchone()
        return r["version"] if r else None

    def list(self, tenant_id: str, limit: int = 20) -> list:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, tic_id, kind, status, created, finished, error FROM jobs "
                "WHERE tenant_id=? ORDER BY created DESC LIMIT ?", (tenant_id, limit)).fetchall()
        return [dict(r) for r in rows]

    # ── lifecycle ────────────────────────────────────────────────────────────
    def recover(self) -> list:
        """Reset jobs interrupted by a restart to a clean 'queued' state; returns their ids, oldest first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT id, stages FROM jobs WHERE status IN ('queued','running') ORDER BY created").fetchall()
            for r in rows:
                self._db.execute("UPDATE jobs SET status='queued', stages=?, error=NULL, version=version+1 WHERE id=?",
                                 (json.dumps(self._blank_stages(json.loads(r["stages"]))), r["id"]))
        return [r["id"] for r in rows]

    def prune(self, keep: int = 500) -> int:
        """Delete the oldest finished jobs beyond `keep`. Active jobs are never removed."""
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM jobs WHERE status NOT IN ('queued','running') AND id NOT IN "
                "(SELECT id FROM jobs ORDER BY created DESC LIMIT ?)", (keep,))
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._db.close()
