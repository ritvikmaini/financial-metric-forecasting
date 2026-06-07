"""SQLite append-only run registry for Financial Metric Forecasting.

Source of truth for 'what's been run with what config against which window'.
Read by the noise-floor (S15) and admission gate (S17) sessions; never mutated.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_NAMESPACE = uuid.UUID("c5d8e2f9-7a14-43b6-8c92-9e58a7d3c1f4")

Mode = Literal["adhoc", "backfill"]

_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS _migrations (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        mode TEXT NOT NULL,
        config_flags_hash TEXT NOT NULL,
        commit_sha TEXT,
        metric TEXT NOT NULL,
        start_year INTEGER,
        end_year INTEGER,
        n_securities INTEGER,
        n_rows_scored INTEGER,
        status TEXT NOT NULL,
        started_at TEXT NOT NULL,
        finished_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS run_config (
        config_flags_hash TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        PRIMARY KEY (config_flags_hash, key)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_runs_mode_metric ON runs(mode, metric)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_runs_config_hash ON runs(config_flags_hash)
    """,
)


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: uuid.UUID
    mode: Mode
    config_flags_hash: str
    commit_sha: str | None
    metric: str
    start_year: int | None
    end_year: int | None
    n_securities: int | None
    n_rows_scored: int | None
    status: str
    started_at: dt.datetime
    finished_at: dt.datetime | None
    config: dict[str, Any] = field(default_factory=dict)


def config_flags_hash(config: dict[str, Any]) -> str:
    """SHA256 over JSON-canonicalized config dict. Sorted keys, ensure_ascii=True."""
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_id_for(
    *,
    mode: Mode,
    config_flags_hash_value: str,
    window: str,
    metric: str,
    started_at_iso_minute: str,
) -> uuid.UUID:
    key = f"{mode}|{config_flags_hash_value}|{window}|{metric}|{started_at_iso_minute}"
    return uuid.uuid5(_NAMESPACE, key)


class Registry:
    """Append-only SQLite registry. Idempotent via INSERT OR IGNORE on runs.run_id."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        for stmt in _SCHEMA:
            self._conn.execute(stmt)
        row = self._conn.execute("SELECT MAX(version) FROM _migrations").fetchone()
        current = row[0] if row and row[0] is not None else 0
        if current > _SCHEMA_VERSION:
            raise RuntimeError(
                f"registry DB at version {current}; code only knows {_SCHEMA_VERSION}. "
                "Upgrade financial-metric-forecasting before reading this DB."
            )
        if current < _SCHEMA_VERSION:
            self._conn.execute(
                "INSERT INTO _migrations(version, applied_at) VALUES (?, ?)",
                (_SCHEMA_VERSION, dt.datetime.now(dt.UTC).isoformat()),
            )
        self._conn.commit()

    def record_run(self, record: RunRecord) -> bool:
        """INSERT OR IGNORE. Returns True if new row inserted, False if already present."""
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO runs "
            "(run_id, mode, config_flags_hash, commit_sha, metric, start_year, end_year, "
            " n_securities, n_rows_scored, status, started_at, finished_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(record.run_id),
                record.mode,
                record.config_flags_hash,
                record.commit_sha,
                record.metric,
                record.start_year,
                record.end_year,
                record.n_securities,
                record.n_rows_scored,
                record.status,
                record.started_at.isoformat(),
                record.finished_at.isoformat() if record.finished_at else None,
            ),
        )
        inserted = cur.rowcount == 1
        if inserted:
            for key, value in record.config.items():
                self._conn.execute(
                    "INSERT OR IGNORE INTO run_config(config_flags_hash, key, value) "
                    "VALUES (?, ?, ?)",
                    (record.config_flags_hash, key, json.dumps(value, default=str)),
                )
        self._conn.commit()
        return inserted

    def get_run(self, run_id: uuid.UUID) -> RunRecord | None:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (str(run_id),)).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_runs(
        self,
        *,
        mode: Mode | None = None,
        metric: str | None = None,
    ) -> list[RunRecord]:
        clauses: list[str] = []
        params: list[str] = []
        if mode is not None:
            clauses.append("mode = ?")
            params.append(mode)
        if metric is not None:
            clauses.append("metric = ?")
            params.append(metric)
        sql = "SELECT * FROM runs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY started_at DESC"
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def verify(self) -> list[tuple[uuid.UUID, str, str]]:
        """Re-hash each run's config and report any drift.

        Returns list of (run_id, stored_hash, recomputed_hash) for mismatches.
        """
        mismatches: list[tuple[uuid.UUID, str, str]] = []
        for row in self._conn.execute("SELECT * FROM runs"):
            config_rows = self._conn.execute(
                "SELECT key, value FROM run_config WHERE config_flags_hash = ?",
                (row["config_flags_hash"],),
            ).fetchall()
            config = {cr["key"]: json.loads(cr["value"]) for cr in config_rows}
            recomputed = config_flags_hash(config)
            if recomputed != row["config_flags_hash"]:
                mismatches.append((uuid.UUID(row["run_id"]), row["config_flags_hash"], recomputed))
        return mismatches

    def _row_to_record(self, row: sqlite3.Row) -> RunRecord:
        config_rows = self._conn.execute(
            "SELECT key, value FROM run_config WHERE config_flags_hash = ?",
            (row["config_flags_hash"],),
        ).fetchall()
        config = {cr["key"]: json.loads(cr["value"]) for cr in config_rows}
        return RunRecord(
            run_id=uuid.UUID(row["run_id"]),
            mode=row["mode"],
            config_flags_hash=row["config_flags_hash"],
            commit_sha=row["commit_sha"],
            metric=row["metric"],
            start_year=row["start_year"],
            end_year=row["end_year"],
            n_securities=row["n_securities"],
            n_rows_scored=row["n_rows_scored"],
            status=row["status"],
            started_at=dt.datetime.fromisoformat(row["started_at"]),
            finished_at=(
                dt.datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None
            ),
            config=config,
        )
