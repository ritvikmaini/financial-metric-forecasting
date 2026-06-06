"""prices ingest tests.

Key invariants:
- auto_adjust=False semantics: `close` is raw, `adj_close` is back-adjusted.
- On AAPL 2019-06-03 (before the Aug 2020 4:1 split), close should be
  approximately 4× adj_close. If this fails, yfinance's auto_adjust default
  changed silently OR the column-shape pin broke.
- Schema column names use snake_case (lowercase `close`, `adj_close`, etc.).
"""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import duckdb
import pytest

from fmf.data.connectors import get_connection
from fmf.data.yfinance._client import YFinanceClient
from fmf.data.yfinance.prices import ingest_prices

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


def test_ingest_prices_inserts_rows(
    conn: duckdb.DuckDBPyConnection,
    aapl_id: uuid.UUID,
    aapl_security: None,
) -> None:
    client = YFinanceClient(base=SAMPLES)
    n = ingest_prices(
        conn=conn,
        client=client,
        ticker="AAPL",
        security_id=aapl_id,
        start=dt.date(2019, 6, 1),
        end=dt.date(2019, 6, 30),
    )
    assert n > 15  # ~21 trading days in June 2019
    count = conn.execute('SELECT COUNT(*) FROM "prices"').fetchone()
    assert count is not None and count[0] == n


def test_auto_adjust_false_close_differs_from_adj_close(
    conn: duckdb.DuckDBPyConnection,
    aapl_id: uuid.UUID,
    aapl_security: None,
) -> None:
    """AAPL had a 4-for-1 split on Aug 31, 2020. With auto_adjust=False,
    on a date in 2019 raw `close` should be ~4× the back-adjusted `adj_close`.

    If this test fails, yfinance silently changed its auto_adjust default
    (or the prices ingest is using the adjusted series for `close`).
    """
    client = YFinanceClient(base=SAMPLES)
    ingest_prices(
        conn=conn,
        client=client,
        ticker="AAPL",
        security_id=aapl_id,
        start=dt.date(2019, 6, 1),
        end=dt.date(2019, 6, 30),
    )
    row = conn.execute(
        'SELECT "close", "adj_close" FROM "prices" WHERE "date" = ?',
        [dt.date(2019, 6, 3)],
    ).fetchone()
    assert row is not None
    close, adj_close = row
    assert close != adj_close, "close == adj_close — yfinance auto_adjust default changed silently"
    # Pre-2020-split, close ≈ 4× adj_close. Tolerate 10% drift for dividends.
    assert close > adj_close * 3.5, (
        f"close={close} not ~4× adj_close={adj_close}; "
        f"yfinance auto_adjust semantics may have changed"
    )
