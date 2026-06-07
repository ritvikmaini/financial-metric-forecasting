"""Shared backtester-test helpers. Reads from the committed fixture DB."""

from __future__ import annotations

import uuid
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini.duckdb"


def fixture_conn() -> duckdb.DuckDBPyConnection:
    if not FIXTURE.exists():
        raise FileNotFoundError(f"fixture DB missing: {FIXTURE}")
    return duckdb.connect(str(FIXTURE), read_only=True)


def aapl_security_id(conn: duckdb.DuckDBPyConnection) -> uuid.UUID:
    row = conn.execute('SELECT security_id FROM "securities" WHERE symbol = ?', ["AAPL"]).fetchone()
    if row is None:
        raise RuntimeError("AAPL not in fixture DB")
    return uuid.UUID(str(row[0]))


def all_fixture_security_ids(conn: duckdb.DuckDBPyConnection) -> list[uuid.UUID]:
    rows = conn.execute('SELECT security_id FROM "securities" ORDER BY symbol').fetchall()
    return [uuid.UUID(str(r[0])) for r in rows]


def two_anchor_ids(conn: duckdb.DuckDBPyConnection) -> list[uuid.UUID]:
    rows = conn.execute(
        'SELECT security_id FROM "securities" WHERE symbol IN (?, ?) ORDER BY symbol',
        ["AAPL", "MSFT"],
    ).fetchall()
    return [uuid.UUID(str(r[0])) for r in rows]
