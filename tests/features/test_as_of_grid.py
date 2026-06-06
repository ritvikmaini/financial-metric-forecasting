"""As-of grid strategy tests.

Verify each strategy returns the expected sample count and that all
emitted as_of_date values are dt.date (post-Timestamp normalization,
the L-INFRA-011 trap).
"""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import duckdb
import pytest

from fmf.features.as_of_grid import (
    AsOfSample,
    daily_calendar_grid,
    filing_dates_grid,
    fiscal_year_end_grid,
    quarterly_grid,
)

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini.duckdb"


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    if not FIXTURE.exists():
        pytest.skip("fixture not built yet")
    return duckdb.connect(str(FIXTURE), read_only=True)


@pytest.fixture
def aapl(conn: duckdb.DuckDBPyConnection) -> tuple[uuid.UUID, str]:
    row = conn.execute('SELECT security_id FROM "securities" WHERE symbol = ?', ["AAPL"]).fetchone()
    assert row is not None
    return uuid.UUID(str(row[0])), "AAPL"


def test_filing_dates_grid_count_matches_distinct_filings(
    conn: duckdb.DuckDBPyConnection, aapl: tuple[uuid.UUID, str]
) -> None:
    sid, symbol = aapl
    expected = conn.execute(
        "SELECT COUNT(*) FROM ("
        "SELECT DISTINCT fiscal_year, period, accepted_date "
        'FROM "income_statement" WHERE security_id = ?)',
        [str(sid)],
    ).fetchone()[0]
    samples = filing_dates_grid(conn, sid, symbol)
    assert len(samples) == expected
    assert all(isinstance(s, AsOfSample) for s in samples)
    assert all(s.symbol == "AAPL" for s in samples)


def test_fiscal_year_end_grid_one_per_fiscal_year(
    conn: duckdb.DuckDBPyConnection, aapl: tuple[uuid.UUID, str]
) -> None:
    sid, symbol = aapl
    expected = conn.execute(
        'SELECT COUNT(DISTINCT fiscal_year) FROM "income_statement" '
        "WHERE security_id = ? AND period = ?",
        [str(sid), "FY"],
    ).fetchone()[0]
    samples = fiscal_year_end_grid(conn, sid, symbol)
    assert len(samples) == expected
    sources = [s.as_of_source for s in samples]
    assert all(src.startswith("income_statement.FY.") for src in sources)


def test_quarterly_grid_one_per_fy_period(
    conn: duckdb.DuckDBPyConnection, aapl: tuple[uuid.UUID, str]
) -> None:
    sid, symbol = aapl
    expected = conn.execute(
        "SELECT COUNT(*) FROM ("
        'SELECT DISTINCT fiscal_year, period FROM "income_statement" '
        "WHERE security_id = ?)",
        [str(sid)],
    ).fetchone()[0]
    samples = quarterly_grid(conn, sid, symbol)
    assert len(samples) == expected


def test_all_grid_as_of_dates_are_date_type(
    conn: duckdb.DuckDBPyConnection, aapl: tuple[uuid.UUID, str]
) -> None:
    """The L-INFRA-011 trap. DuckDB DATE fetchall may return dt.date or
    pd.Timestamp depending on driver path. All emitted as_of_date values
    must be plain dt.date (not dt.datetime, not pd.Timestamp) so the
    accepted_date <= as_of_date predicate is type-stable downstream."""
    sid, symbol = aapl
    for strategy in (filing_dates_grid, fiscal_year_end_grid, quarterly_grid):
        samples = strategy(conn, sid, symbol)
        assert len(samples) > 0, f"strategy {strategy.__name__} returned empty"
        for s in samples:
            assert type(s.as_of_date) is dt.date, (
                f"{strategy.__name__} emitted non-date: "
                f"{type(s.as_of_date).__name__} {s.as_of_date!r}"
            )


def test_daily_calendar_grid_exact_count(
    conn: duckdb.DuckDBPyConnection, aapl: tuple[uuid.UUID, str]
) -> None:
    sid, symbol = aapl
    samples = daily_calendar_grid(
        conn, sid, symbol, start=dt.date(2020, 1, 1), end=dt.date(2020, 1, 10)
    )
    assert len(samples) == 10
    assert samples[0].as_of_date == dt.date(2020, 1, 1)
    assert samples[-1].as_of_date == dt.date(2020, 1, 10)
    assert samples[5].as_of_source == "calendar.2020-01-06"
