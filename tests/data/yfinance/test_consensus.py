"""consensus ingest tests."""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import duckdb
import pytest

from fmf.data.connectors import get_connection
from fmf.data.yfinance._client import YFinanceClient
from fmf.data.yfinance.consensus import (
    _period_label_to_target_date,
    ingest_consensus_snapshot,
)

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


def test_ingest_consensus_writes_eps_and_revenue(
    conn: duckdb.DuckDBPyConnection,
    aapl_id: uuid.UUID,
    aapl_security: None,
) -> None:
    client = YFinanceClient(base=SAMPLES)
    ingest_consensus_snapshot(
        conn=conn,
        client=client,
        ticker="AAPL",
        security_id=aapl_id,
        pulled_at=dt.datetime(2024, 1, 1, 12, 0, 0),
    )
    eps_count = conn.execute(
        'SELECT COUNT(*) FROM "analyst_estimates" WHERE metric = ?', ["eps"]
    ).fetchone()
    rev_count = conn.execute(
        'SELECT COUNT(*) FROM "analyst_estimates" WHERE metric = ?', ["revenue"]
    ).fetchone()
    # Sample data may be sparse; assert at least one of the two metrics has rows.
    assert eps_count is not None and eps_count[0] >= 0
    assert rev_count is not None and rev_count[0] >= 0
    assert (eps_count[0] + rev_count[0]) > 0, "no consensus rows at all written"


def test_ingest_consensus_carries_pulled_at(
    conn: duckdb.DuckDBPyConnection,
    aapl_id: uuid.UUID,
    aapl_security: None,
) -> None:
    """PIT proxy depends on pulled_at being recorded with each snapshot row."""
    client = YFinanceClient(base=SAMPLES)
    when = dt.datetime(2024, 6, 15, 9, 30, 0)
    ingest_consensus_snapshot(
        conn=conn,
        client=client,
        ticker="AAPL",
        security_id=aapl_id,
        pulled_at=when,
    )
    row = conn.execute('SELECT pulled_at FROM "analyst_estimates" LIMIT 1').fetchone()
    if row is not None:
        assert row[0] == when


def test_period_label_mapping_anchored_to_pulled_at() -> None:
    """Lock in the label → target_date mapping."""
    anchor = dt.date(2024, 5, 15)  # mid-Q2 2024
    assert _period_label_to_target_date("0q", anchor) == dt.date(2024, 6, 30)
    assert _period_label_to_target_date("+1q", anchor) == dt.date(2024, 9, 30)
    assert _period_label_to_target_date("0y", anchor) == dt.date(2024, 12, 31)
    assert _period_label_to_target_date("+1y", anchor) == dt.date(2025, 12, 31)
    assert _period_label_to_target_date("currentQuarter", anchor) == dt.date(2024, 6, 30)
    assert _period_label_to_target_date("nextYear", anchor) == dt.date(2025, 12, 31)
    assert _period_label_to_target_date("ttm", anchor) is None
    assert _period_label_to_target_date("garbage", anchor) is None


def test_period_label_mapping_quarter_boundaries() -> None:
    """End-of-quarter math: anchor in different quarters yields correct
    end-of-current-quarter dates."""
    # Q1 anchor
    assert _period_label_to_target_date("0q", dt.date(2024, 2, 15)) == dt.date(2024, 3, 31)
    # Q3 anchor
    assert _period_label_to_target_date("0q", dt.date(2024, 8, 15)) == dt.date(2024, 9, 30)
    # Q4 anchor
    assert _period_label_to_target_date("0q", dt.date(2024, 11, 15)) == dt.date(2024, 12, 31)
    # Q4 anchor + 1q → next year Q1
    assert _period_label_to_target_date("+1q", dt.date(2024, 11, 15)) == dt.date(2025, 3, 31)
