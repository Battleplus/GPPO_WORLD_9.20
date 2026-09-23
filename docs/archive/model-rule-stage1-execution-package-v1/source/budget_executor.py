"""SQLite-backed shared budget reservations for bounded experiment entrypoints.

This is the repository-local copy of the previously validated persistent-budget
schema.  It deliberately counts reservations before side effects; an uncertain
side effect is terminally marked ``unknown`` and is never refunded.
"""

from __future__ import annotations

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import uuid
from typing import Any


class BudgetExhausted(RuntimeError):
    """The global or stable-run budget cannot reserve the requested amount."""


class BudgetStateConflict(RuntimeError):
    """A reservation was already finalized with a different outcome."""


class PersistentBudget:
    """Transactional budget ledger shared across attempts and processes."""

    SCHEMA = "persistent-budget/2.0.0"
    DB_SCHEMA = "persistent-budget-sqlite/1.0.0"

    def __init__(self, path: str | Path, *, limits: dict[str, int], attempt_id: str,
                 run_id: str, run_limits: dict[str, int] | None = None,
                 require_existing: bool = False, lock_timeout: float = 5.0):
        supplied = Path(path)
        self.db_path = supplied if supplied.suffix.lower() == ".sqlite3" else supplied.with_suffix(".sqlite3")
        self.snapshot_path = self.db_path.with_suffix(".json")
        self.path = self.db_path
        self.limits = {str(k): int(v) for k, v in limits.items()}
        self.attempt_id = str(attempt_id)
        self.run_id = str(run_id)
        self.run_limits = {str(k): int(v) for k, v in (run_limits or limits).items()}
        self.lock_timeout = float(lock_timeout)
        if require_existing and not self.db_path.exists():
            raise FileNotFoundError(f"required existing budget database not found: {self.db_path}")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.db_path.exists():
            self._validate_existing_limits()
            self._ensure_run_columns()
        else:
            self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=self.lock_timeout,
                              isolation_level=None, check_same_thread=False)
        con.execute("PRAGMA foreign_keys=ON")
        con.execute(f"PRAGMA busy_timeout={max(1, int(self.lock_timeout * 1000))}")
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=FULL")
        return con

    @classmethod
    def _schema_sql(cls) -> tuple[str, ...]:
        return (
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS stages (stage TEXT PRIMARY KEY, limit_amount INTEGER NOT NULL, reserved INTEGER NOT NULL, verified INTEGER NOT NULL, unknown INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS attempts (attempt_id TEXT NOT NULL, stage TEXT NOT NULL, reserved INTEGER NOT NULL, verified INTEGER NOT NULL, unknown INTEGER NOT NULL, run_id TEXT, PRIMARY KEY (attempt_id, stage))",
            "CREATE TABLE IF NOT EXISTS reservations (reservation_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL, stage TEXT NOT NULL, amount INTEGER NOT NULL, status TEXT NOT NULL CHECK(status IN ('pending','verified','unknown')), reservation_index INTEGER, created_at REAL NOT NULL, finished_at REAL, reason TEXT, run_id TEXT)",
            "CREATE TABLE IF NOT EXISTS history (seq INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT NOT NULL, stage TEXT NOT NULL, amount INTEGER NOT NULL, attempt_id TEXT NOT NULL, reservation_id TEXT, reservation_index INTEGER, pid INTEGER, reason TEXT, created_at REAL NOT NULL, run_id TEXT)",
            "CREATE INDEX IF NOT EXISTS idx_reservations_run_stage ON reservations(run_id, stage)",
        )

    def _initialize_database(self) -> None:
        with closing(self._connect()) as con:
            con.execute("BEGIN IMMEDIATE")
            for sql in self._schema_sql():
                con.execute(sql)
            con.execute("INSERT INTO metadata(key,value) VALUES (?,?)", ("schema", self.DB_SCHEMA))
            con.execute("INSERT INTO metadata(key,value) VALUES (?,?)", ("json_schema", self.SCHEMA))
            con.execute("INSERT INTO metadata(key,value) VALUES (?,?)", ("run_limits", json.dumps(self.run_limits, sort_keys=True)))
            for stage, limit in self.limits.items():
                con.execute("INSERT INTO stages(stage,limit_amount,reserved,verified,unknown) VALUES (?,?,0,0,0)", (stage, limit))
            con.commit()

    def _validate_existing_limits(self) -> None:
        with closing(self._connect()) as con:
            schema = con.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()
            rows = dict(con.execute("SELECT stage,limit_amount FROM stages").fetchall())
            if schema is None or schema[0] != self.DB_SCHEMA:
                raise ValueError(f"unknown persistent budget database schema: {schema}")
            if rows != self.limits:
                raise ValueError(f"budget limits changed: {rows} != {self.limits}")

    def _ensure_run_columns(self) -> None:
        with closing(self._connect()) as con:
            con.execute("BEGIN IMMEDIATE")
            for table in ("attempts", "reservations", "history"):
                cols = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
                if "run_id" not in cols:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN run_id TEXT")
            con.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)", ("run_limits", json.dumps(self.run_limits, sort_keys=True)))
            con.commit()

    def reserve(self, stage: str, amount: int = 1) -> dict[str, Any]:
        stage, amount = str(stage), int(amount)
        if stage not in self.limits or amount < 1:
            raise ValueError(f"invalid budget request: {stage}, {amount}")
        rid, now = uuid.uuid4().hex, time.time()
        with closing(self._connect()) as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT reserved,limit_amount FROM stages WHERE stage=?", (stage,)).fetchone()
            if row is None:
                con.rollback(); raise ValueError(f"unknown budget stage: {stage}")
            reserved, limit = map(int, row)
            if reserved + amount > limit:
                con.rollback(); raise BudgetExhausted(f"global budget exhausted: {reserved}+{amount}>{limit}")
            run_reserved = int(con.execute(
                "SELECT COALESCE(SUM(amount),0) FROM reservations WHERE run_id=? AND stage=?",
                (self.run_id, stage)).fetchone()[0])
            if run_reserved + amount > self.run_limits.get(stage, limit):
                con.rollback(); raise BudgetExhausted(f"run budget exhausted: {run_reserved}+{amount}>{self.run_limits.get(stage)}")
            con.execute("UPDATE stages SET reserved=reserved+? WHERE stage=?", (amount, stage))
            old = con.execute("SELECT reserved,verified,unknown FROM attempts WHERE attempt_id=? AND stage=?", (self.attempt_id, stage)).fetchone() or (0, 0, 0)
            con.execute(
                "INSERT INTO attempts(attempt_id,stage,reserved,verified,unknown,run_id) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(attempt_id,stage) DO UPDATE SET reserved=excluded.reserved,run_id=excluded.run_id",
                (self.attempt_id, stage, int(old[0]) + amount, int(old[1]), int(old[2]), self.run_id))
            con.execute("INSERT INTO reservations(reservation_id,attempt_id,stage,amount,status,created_at,run_id) VALUES (?,?,?,?,?,?,?)",
                         (rid, self.attempt_id, stage, amount, "pending", now, self.run_id))
            cur = con.execute("INSERT INTO history(event,stage,amount,attempt_id,reservation_id,pid,created_at,run_id) VALUES (?,?,?,?,?,?,?,?)",
                              ("reserve", stage, amount, self.attempt_id, rid, os.getpid(), now, self.run_id))
            index = int(cur.lastrowid)
            con.execute("UPDATE reservations SET reservation_index=? WHERE reservation_id=?", (index, rid))
            con.commit()
        return {"reservation_id": rid, "reservation_index": index, "stage": stage, "amount": amount,
                "attempt_id": self.attempt_id, "run_id": self.run_id}

    def _finish(self, token: dict[str, Any], outcome: str, reason: str | None = None) -> None:
        rid = str(token.get("reservation_id", ""))
        if outcome not in ("verified", "unknown") or not rid:
            raise ValueError("invalid reservation finalization")
        now = time.time()
        with closing(self._connect()) as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT attempt_id,stage,amount,status,reservation_index,run_id FROM reservations WHERE reservation_id=?", (rid,)).fetchone()
            if row is None:
                con.rollback(); raise ValueError(f"unknown reservation: {rid}")
            attempt, stage, amount, status, index, run_id = row
            if status == outcome:
                con.commit(); return
            if status != "pending":
                con.rollback(); raise BudgetStateConflict(f"reservation {rid} already {status}")
            con.execute("UPDATE reservations SET status=?,finished_at=?,reason=? WHERE reservation_id=?", (outcome, now, reason, rid))
            con.execute(f"UPDATE stages SET {outcome}={outcome}+? WHERE stage=?", (int(amount), stage))
            col = "verified" if outcome == "verified" else "unknown"
            con.execute(f"UPDATE attempts SET {col}={col}+? WHERE attempt_id=? AND stage=?", (int(amount), attempt, stage))
            con.execute("INSERT INTO history(event,stage,amount,attempt_id,reservation_id,reservation_index,pid,reason,created_at,run_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (outcome, stage, int(amount), attempt, rid, index, os.getpid(), reason, now, run_id))
            con.commit()

    def complete(self, token: dict[str, Any]) -> None:
        self._finish(token, "verified")

    def unknown(self, token: dict[str, Any], reason: str) -> None:
        self._finish(token, "unknown", str(reason))

    def snapshot(self) -> dict[str, Any]:
        with closing(self._connect()) as con:
            con.execute("BEGIN")
            stages = {s: {"limit": int(limit), "reserved": int(r), "verified": int(v), "unknown": int(u), "pending": int(r-v-u)}
                      for s, limit, r, v, u in con.execute("SELECT stage,limit_amount,reserved,verified,unknown FROM stages ORDER BY stage")}
            con.commit()
        return {"schema": self.SCHEMA, "db_schema": self.DB_SCHEMA, "db_path": str(self.db_path),
                "limits": self.limits, "stages": stages, "run_id": self.run_id, "attempt_id": self.attempt_id}

    def export_snapshot(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.snapshot_path
        target.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self.snapshot(), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        with tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False) as f:
            temp = Path(f.name); f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(temp, target)
        return target

    def run_totals(self, run_id: str | None = None) -> dict[str, dict[str, int]]:
        rid = str(run_id or self.run_id)
        with closing(self._connect()) as con:
            con.execute("BEGIN")
            rows = con.execute(
                "SELECT stage,COALESCE(SUM(amount),0),COALESCE(SUM(CASE WHEN status='verified' THEN amount ELSE 0 END),0),COALESCE(SUM(CASE WHEN status='unknown' THEN amount ELSE 0 END),0) FROM reservations WHERE run_id=? GROUP BY stage",
                (rid,)).fetchall()
            con.commit()
        return {s: {"reserved": int(r), "verified": int(v), "unknown": int(u), "pending": int(r-v-u)} for s, r, v, u in rows}

    def integrity_check(self) -> dict[str, Any]:
        with closing(self._connect()) as con:
            result = con.execute("PRAGMA integrity_check").fetchone()[0]
            pending = int(con.execute("SELECT COUNT(*) FROM reservations WHERE status='pending'").fetchone()[0])
        return {"integrity_check": result, "pending_reservations": pending, "db_path": str(self.db_path)}
