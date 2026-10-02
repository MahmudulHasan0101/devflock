"""Durable run state in SQLite (.devflock/state.db under the project dir).

This is what lets a crashed or interrupted run resume: task status,
per-region ownership, per-worker context-usage estimate, and a full event
log survive even if every Claude Code session is gone.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS regions (
    name TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    owner_worker TEXT,
    public_summary TEXT,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    region TEXT NOT NULL,
    description TEXT NOT NULL,
    depends_on TEXT NOT NULL DEFAULT '[]',   -- JSON list of task ids
    acceptance_cmd TEXT,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending|running|verify_failed|done|failed|blocked
    assigned_worker TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    session_id TEXT,
    created_at REAL,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY,
    base_url TEXT,
    gateway_port INTEGER,
    region TEXT,
    session_id TEXT,
    est_context_tokens INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'idle',   -- idle|busy|dead
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    t REAL,
    worker TEXT,
    task TEXT,
    kind TEXT,
    detail TEXT
);

CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    t REAL,
    worker TEXT,
    task TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL
);
"""


class State:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # --- regions -----------------------------------------------------
    def upsert_region(self, name: str, path: str, owner_worker: Optional[str] = None,
                       public_summary: Optional[str] = None):
        self.conn.execute(
            "INSERT INTO regions(name, path, owner_worker, public_summary, updated_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
            "path=excluded.path, "
            "owner_worker=COALESCE(excluded.owner_worker, regions.owner_worker), "
            "public_summary=COALESCE(excluded.public_summary, regions.public_summary), "
            "updated_at=excluded.updated_at",
            (name, path, owner_worker, public_summary, time.time()))
        self.conn.commit()

    def get_region(self, name: str) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM regions WHERE name=?", (name,)).fetchone()

    def all_regions(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM regions").fetchall()

    # --- tasks ---------------------------------------------------------
    def add_task(self, task_id: str, region: str, description: str,
                 depends_on: list[str] | None = None, acceptance_cmd: str | None = None):
        self.conn.execute(
            "INSERT OR IGNORE INTO tasks(id, region, description, depends_on, acceptance_cmd, "
            "status, created_at, updated_at) VALUES (?,?,?,?,?, 'pending', ?, ?)",
            (task_id, region, description, json.dumps(depends_on or []), acceptance_cmd,
             time.time(), time.time()))
        self.conn.commit()

    def set_task_status(self, task_id: str, status: str, **fields):
        sets = ["status=?", "updated_at=?"]
        vals: list[Any] = [status, time.time()]
        for k, v in fields.items():
            sets.append(f"{k}=?")
            vals.append(v)
        vals.append(task_id)
        self.conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id=?", vals)
        self.conn.commit()

    def get_task(self, task_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()

    def ready_tasks(self) -> list[sqlite3.Row]:
        """Pending tasks whose dependencies are all done."""
        done = {r["id"] for r in self.conn.execute("SELECT id FROM tasks WHERE status='done'")}
        out = []
        for row in self.conn.execute("SELECT * FROM tasks WHERE status='pending'"):
            deps = json.loads(row["depends_on"])
            if all(d in done for d in deps):
                out.append(row)
        return out

    def counts_by_status(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in self.conn.execute("SELECT status, COUNT(*) c FROM tasks GROUP BY status"):
            out[row["status"]] = row["c"]
        return out

    def block_orphaned_tasks(self) -> list[str]:
        """Mark pending tasks as 'blocked' when any dependency has failed or
        is itself blocked (transitively). Without this, a single failed task
        leaves its dependents pending forever and the scheduler never exits."""
        newly_blocked: list[str] = []
        while True:
            dead = {r["id"] for r in self.conn.execute(
                "SELECT id FROM tasks WHERE status IN ('failed','blocked')")}
            changed = False
            for row in self.conn.execute("SELECT id, depends_on FROM tasks WHERE status='pending'").fetchall():
                if any(d in dead for d in json.loads(row["depends_on"])):
                    self.set_task_status(row["id"], "blocked")
                    self.log_event("task_blocked", task=row["id"],
                                   detail="a dependency failed or was blocked")
                    newly_blocked.append(row["id"])
                    changed = True
            if not changed:
                return newly_blocked

    # --- workers ---------------------------------------------------------
    def upsert_worker(self, worker_id: str, **fields):
        existing = self.conn.execute("SELECT id FROM workers WHERE id=?", (worker_id,)).fetchone()
        fields["updated_at"] = time.time()
        if existing:
            sets = ", ".join(f"{k}=?" for k in fields)
            self.conn.execute(f"UPDATE workers SET {sets} WHERE id=?",
                               (*fields.values(), worker_id))
        else:
            fields["id"] = worker_id
            cols = ", ".join(fields)
            qs = ", ".join("?" for _ in fields)
            self.conn.execute(f"INSERT INTO workers({cols}) VALUES ({qs})", tuple(fields.values()))
        self.conn.commit()

    def all_workers(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM workers").fetchall()

    # --- events / ledger ---------------------------------------------------------
    def log_event(self, kind: str, worker: str | None = None, task: str | None = None,
                  detail: str = ""):
        self.conn.execute(
            "INSERT INTO events(t, worker, task, kind, detail) VALUES (?,?,?,?,?)",
            (time.time(), worker, task, kind, detail))
        self.conn.commit()

    def log_cost(self, worker: str, task: str | None, input_tokens: int, output_tokens: int,
                 cost_usd: float):
        self.conn.execute(
            "INSERT INTO ledger(t, worker, task, input_tokens, output_tokens, cost_usd) "
            "VALUES (?,?,?,?,?,?)",
            (time.time(), worker, task, input_tokens, output_tokens, cost_usd))
        self.conn.commit()

    def total_cost(self) -> float:
        row = self.conn.execute("SELECT COALESCE(SUM(cost_usd),0) c FROM ledger").fetchone()
        return row["c"]

    def recent_events(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


@contextlib.contextmanager
def open_state(project_dir: str | Path) -> Iterator[State]:
    st = State(Path(project_dir) / ".devflock" / "state.db")
    try:
        yield st
    finally:
        st.close()
