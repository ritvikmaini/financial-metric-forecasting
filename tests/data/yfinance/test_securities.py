"""securities update tests."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import duckdb
import pytest

from fmf.data.connectors import get_connection
from fmf.data.yfinance._client import YFinanceClient
from fmf.data.yfinance.securities import update_securities_metadata

REPO_ROOT = Path(__file__).parent.parent.parent.parent
SCHEMA_PATH = REPO_ROOT / "fmf" / "data" / "schema.sql"
SAMPLES = REPO_ROOT / "tests" / "fixtures" / "sample_yfinance"


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    c = get_connection(":memory:")
    c.execute(SCHEMA_PATH.read_text())
    return c


@pytest.fixture
def aapl_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
def aapl_security(conn: duckdb.DuckDBPyConnection, aapl_id: uuid.UUID) -> None:
    conn.execute(
        'INSERT INTO "securities" (security_id, symbol, cik) VALUES (?, ?, ?)',
        [str(aapl_id), "AAPL", "0000320193"],
    )


def test_update_does_not_insert_duplicate(
    conn: duckdb.DuckDBPyConnection,
    aapl_id: uuid.UUID,
    aapl_security: None,
) -> None:
    """Pre-existing securities row must be UPDATEd, not duplicated."""
    client = YFinanceClient(base=SAMPLES)
    update_securities_metadata(conn=conn, client=client, ticker="AAPL", cik="0000320193")
    n = conn.execute('SELECT COUNT(*) FROM "securities" WHERE cik = ?', ["0000320193"]).fetchone()
    assert n is not None and n[0] == 1


def test_update_populates_at_least_one_metadata_field(
    conn: duckdb.DuckDBPyConnection,
    aapl_id: uuid.UUID,
    aapl_security: None,
) -> None:
    """AAPL's info should have at least one of sector/industry/country/exchange."""
    client = YFinanceClient(base=SAMPLES)
    update_securities_metadata(conn=conn, client=client, ticker="AAPL", cik="0000320193")
    row = conn.execute(
        'SELECT sector, industry, country, exchange FROM "securities" WHERE cik = ?',
        ["0000320193"],
    ).fetchone()
    assert row is not None
    assert any(v is not None for v in row), f"all four metadata fields are NULL: {row}"


def test_update_tolerates_missing_fields(
    conn: duckdb.DuckDBPyConnection,
    aapl_id: uuid.UUID,
    aapl_security: None,
    tmp_path: Path,
) -> None:
    """If the info dict is missing fields, the update must not raise."""
    sparse = tmp_path / "AAPL_info.json"
    sparse.write_text(json.dumps({"symbol": "AAPL"}))
    client = YFinanceClient(base=tmp_path)
    # Must not raise:
    update_securities_metadata(conn=conn, client=client, ticker="AAPL", cik="0000320193")
    row = conn.execute(
        'SELECT sector, industry, country, exchange FROM "securities" WHERE cik = ?',
        ["0000320193"],
    ).fetchone()
    assert row is not None
    # All four columns NULL because none were in the sparse info.
    assert all(v is None for v in row)


def test_update_tolerates_fetch_failure(
    conn: duckdb.DuckDBPyConnection,
    aapl_id: uuid.UUID,
    aapl_security: None,
    tmp_path: Path,
) -> None:
    """If fetch_info raises (missing fixture file), the update must log
    a warning and continue without raising."""
    # tmp_path doesn't have an info file for AAPL — fetch will raise.
    client = YFinanceClient(base=tmp_path)
    # Must not raise:
    update_securities_metadata(conn=conn, client=client, ticker="AAPL", cik="0000320193")
    # The row exists but metadata stays NULL.
    n = conn.execute('SELECT COUNT(*) FROM "securities" WHERE cik = ?', ["0000320193"]).fetchone()
    assert n is not None and n[0] == 1
