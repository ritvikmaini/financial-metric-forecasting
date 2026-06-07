"""SQLite prediction cache for the S10 backtester (S14).

Key axes (all four required):
  CACHE_VERSION | config_flags_hash | data_fingerprint(security_id) |
  (security_id, as_of_date, metric, model_name)

Versioned manual bump, config hash, per-security data fingerprint, and the
coordinate. Two runs with identical config but a rebuilt fixture must MISS
the cache; that is the correctness gate against stale entries propagating
into S15 noise-floor and S17 admission-gate.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import sqlite3
import uuid
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)

CACHE_VERSION = 1


def data_fingerprint(conn: duckdb.DuckDBPyConnection, security_id: uuid.UUID) -> str:
    """Per-security SHA256 over (row_count, max(accepted_date), max(end_date))
    of income_statement plus (row_count, max(date)) of prices.

    No ingested_at column exists in v1.0 schema; if S2/S3 grow one later,
    extend this and bump CACHE_VERSION.
    """
    sid = str(security_id)
    is_row = conn.execute(
        "SELECT COUNT(*), MAX(accepted_date), MAX(end_date) "
        'FROM "income_statement" WHERE security_id = ?',
        [sid],
    ).fetchone()
    px_row = conn.execute(
        'SELECT COUNT(*), MAX(date) FROM "prices" WHERE security_id = ?',
        [sid],
    ).fetchone()
    is_n, is_max_acc, is_max_end = is_row if is_row else (0, None, None)
    px_n, px_max = px_row if px_row else (0, None)
    canon = (
        f"is_n={is_n}|is_max_accepted={is_max_acc}|is_max_end={is_max_end}|"
        f"px_n={px_n}|px_max_date={px_max}"
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def derive_cache_key(
    *,
    config_flags_hash: str,
    fingerprint: str,
    security_id: uuid.UUID,
    as_of_date: dt.date,
    metric: str,
    model_name: str,
) -> str:
    """SHA256 over the four axes. Stable for identical inputs across processes."""
    canon = (
        f"v={CACHE_VERSION}|cfg={config_flags_hash}|fp={fingerprint}|"
        f"sid={security_id}|asof={as_of_date.isoformat()}|metric={metric}|model={model_name}"
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS predictions (
        cache_key TEXT PRIMARY KEY,
        prediction REAL,
        created_at TEXT NOT NULL
    )
    """,
)


class PredictionCache:
    """SQLite-backed prediction cache. Append-only at the application layer;
    INSERT OR REPLACE handles repeat writes of identical keys.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        for stmt in _SCHEMA:
            self._conn.execute(stmt)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def get(self, cache_key: str) -> float | None:
        row = self._conn.execute(
            "SELECT prediction FROM predictions WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if row is None:
            return None
        return None if row[0] is None else float(row[0])

    def contains(self, cache_key: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM predictions WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        return row is not None

    def put_batch(self, entries: list[tuple[str, float | None]]) -> None:
        """One commit per batch. Each entry: (cache_key, prediction_or_None)."""
        if not entries:
            return
        now_iso = dt.datetime.now(dt.UTC).isoformat()
        self._conn.executemany(
            "INSERT OR REPLACE INTO predictions(cache_key, prediction, created_at) "
            "VALUES (?, ?, ?)",
            [(k, v, now_iso) for k, v in entries],
        )
        self._conn.commit()

    def size(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM predictions").fetchone()
        return int(row[0]) if row else 0
