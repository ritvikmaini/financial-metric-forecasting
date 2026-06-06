"""PIT primitives tests.

The cardinal test is the 1-day-shift against the real mini.duckdb:
shift as_of_date back one day before a known AAPL filing's accepted_date
and the visible series must change. If this test ever fails, the PIT
extraction is broken — feature values used for backtesting are
contaminated by look-ahead.
"""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import duckdb
import pytest

from fmf.features.point_in_time import (
    fetch_consensus_pit,
    fetch_pit_series,
    fetch_prices_pit,
)

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini.duckdb"


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    if not FIXTURE.exists():
        pytest.skip("fixture not built yet")
    return duckdb.connect(str(FIXTURE), read_only=True)


@pytest.fixture
def aapl_security_id(conn: duckdb.DuckDBPyConnection) -> uuid.UUID:
    row = conn.execute('SELECT security_id FROM "securities" WHERE symbol = ?', ["AAPL"]).fetchone()
    assert row is not None
    return uuid.UUID(str(row[0]))


# --- Cardinal 1-day-shift PIT test ---


def test_one_day_shift_excludes_filing_on_real_fixture(
    conn: duckdb.DuckDBPyConnection,
    aapl_security_id: uuid.UUID,
) -> None:
    """The FMF-004 cardinal rule, verified against real EDGAR data.

    AAPL's FY2023 10-K was filed 2023-11-03. Extracting features as of
    2023-11-03 must include the FY2023 annual row; as of 2023-11-02
    (one day earlier) must exclude it.

    If this test fails, the PIT primitive is leaking look-ahead.
    """
    # First, look up the actual accepted_date of AAPL FY2023 from the fixture.
    row = conn.execute(
        'SELECT accepted_date FROM "income_statement" '
        "WHERE security_id = ? AND fiscal_year = 2023 AND period = ? "
        "ORDER BY accepted_date ASC LIMIT 1",
        [str(aapl_security_id), "FY"],
    ).fetchone()
    assert row is not None, "AAPL FY2023 missing from fixture"
    fy23_accepted = row[0]
    assert isinstance(fy23_accepted, dt.date), f"got {fy23_accepted!r}"

    # As-of the filing date itself: FY2023 row IS visible.
    series_on = fetch_pit_series(
        conn=conn,
        table="income_statement",
        security_id=aapl_security_id,
        as_of_date=fy23_accepted,
    )
    fy23_on = series_on[series_on["fiscal_year"] == 2023]
    assert len(fy23_on) >= 1, f"FY2023 should be visible at as_of={fy23_accepted}, got: {fy23_on}"

    # As-of one day earlier: FY2023 row is NOT visible.
    one_day_before = fy23_accepted - dt.timedelta(days=1)
    series_before = fetch_pit_series(
        conn=conn,
        table="income_statement",
        security_id=aapl_security_id,
        as_of_date=one_day_before,
    )
    fy23_before = series_before[series_before["fiscal_year"] == 2023]
    fy_before = fy23_before[fy23_before["period"] == "FY"]
    assert len(fy_before) == 0, (
        f"FY2023 leak: 1-day-shift to {one_day_before} returned "
        f"{len(fy_before)} FY rows. Look-ahead bug in PIT extraction."
    )


# --- Multi-period series shape ---


def test_pit_series_returns_one_row_per_period_end(
    conn: duckdb.DuckDBPyConnection,
    aapl_security_id: uuid.UUID,
) -> None:
    """For each distinct (fiscal_year, period, end_date) seen up to as_of,
    the series contains exactly one row — the one with the latest
    accepted_date for that key.
    """
    as_of = dt.date(2024, 6, 1)
    series = fetch_pit_series(
        conn=conn,
        table="income_statement",
        security_id=aapl_security_id,
        as_of_date=as_of,
    )
    # Group by (fiscal_year, period, end_date) and confirm dedup.
    key_cols = ["fiscal_year", "period", "end_date"]
    counts = series.groupby(key_cols).size()
    assert (counts == 1).all(), (
        f"PIT series has duplicate (fy, period, end_date) rows: {counts[counts > 1]}"
    )


def test_pit_series_picks_latest_accepted_date_per_period_end(
    conn: duckdb.DuckDBPyConnection,
    aapl_security_id: uuid.UUID,
) -> None:
    """If a (fy, period, end_date) was restated and both the original and
    restated versions are visible at as_of, only the latest accepted_date
    version appears in the series.
    """
    # Pull all distinct (fy, period, end_date) with their max accepted_date
    # from the table; then pull the PIT series and compare.
    as_of = dt.date(2026, 1, 1)
    latest_in_table = conn.execute(
        "SELECT fiscal_year, period, end_date, MAX(accepted_date) "
        'FROM "income_statement" WHERE security_id = ? AND accepted_date <= ? '
        "GROUP BY fiscal_year, period, end_date",
        [str(aapl_security_id), as_of],
    ).fetchdf()
    series = fetch_pit_series(
        conn=conn,
        table="income_statement",
        security_id=aapl_security_id,
        as_of_date=as_of,
    )
    # Every accepted_date in the series should equal the MAX from the raw table.
    merged = series.merge(
        latest_in_table.rename(columns={"max(accepted_date)": "max_accepted"}),
        on=["fiscal_year", "period", "end_date"],
        how="inner",
    )
    assert (merged["accepted_date"] == merged["max_accepted"]).all(), (
        "PIT series did not pick the latest accepted_date per period-end"
    )


# --- Consensus PIT trap ---


def test_consensus_returns_empty_for_historical_as_of(
    conn: duckdb.DuckDBPyConnection,
    aapl_security_id: uuid.UUID,
) -> None:
    """yfinance consensus is a snapshot pulled near today. For any
    historical as_of (e.g., 2015-06-01), pulled_at > as_of and the
    PIT consensus filter must return EMPTY. This is correct, not a bug:
    using today's snapshot as if it were known in 2015 is look-ahead.
    """
    historical = dt.date(2015, 6, 1)
    consensus = fetch_consensus_pit(
        conn=conn,
        security_id=aapl_security_id,
        as_of_date=historical,
    )
    assert len(consensus) == 0, (
        f"Consensus PIT trap leaked: as_of={historical} returned "
        f"{len(consensus)} rows. The snapshot was pulled in 2026; "
        f"making it visible in 2015 is look-ahead."
    )


def test_consensus_returns_rows_for_post_pull_as_of(
    conn: duckdb.DuckDBPyConnection,
    aapl_security_id: uuid.UUID,
) -> None:
    """For an as_of after the pulled_at, the consensus rows are visible."""
    future_as_of = dt.date(2030, 1, 1)
    consensus = fetch_consensus_pit(
        conn=conn,
        security_id=aapl_security_id,
        as_of_date=future_as_of,
    )
    # AAPL has earnings + revenue estimates in the fixture.
    assert len(consensus) > 0, f"AAPL should have consensus rows visible at as_of={future_as_of}"


# --- Prices PIT ---


def test_prices_pit_excludes_dates_after_as_of(
    conn: duckdb.DuckDBPyConnection,
    aapl_security_id: uuid.UUID,
) -> None:
    """Prices have no accepted_date; PIT filter is date <= as_of_date.

    A 1-day-shift on the price axis would silently leak intraday news
    into a backtest fold — same look-ahead class as the fundamentals
    case, lower amplitude but still real.
    """
    as_of = dt.date(2020, 1, 15)
    prices = fetch_prices_pit(
        conn=conn,
        security_id=aapl_security_id,
        as_of_date=as_of,
    )
    assert len(prices) > 0, f"expected AAPL prices on or before {as_of}"
    # DuckDB / pandas may return either dt.date or pd.Timestamp here;
    # pd.Timestamp is a subclass of dt.date so an isinstance check on
    # dt.date is not sufficient to dispatch on (Timestamp <= dt.date
    # raises TypeError). Normalize unconditionally — matches the sibling
    # one-day-shift test pattern.
    import pandas as pd

    max_date = pd.Timestamp(prices["date"].max()).date()
    assert max_date <= as_of, f"Prices PIT leak: returned date {max_date} > as_of {as_of}"


def test_prices_pit_one_day_shift_excludes_that_day(
    conn: duckdb.DuckDBPyConnection,
    aapl_security_id: uuid.UUID,
) -> None:
    """Cardinal 1-day-shift on the price axis, mirroring the fundamentals
    cardinal test. Pick a known AAPL trading day, then shift as_of back
    one day and confirm that trading day is excluded.
    """
    # AAPL traded on 2020-01-15 (a Wednesday).
    trading_day = dt.date(2020, 1, 15)
    on = fetch_prices_pit(
        conn=conn,
        security_id=aapl_security_id,
        as_of_date=trading_day,
    )
    before = fetch_prices_pit(
        conn=conn,
        security_id=aapl_security_id,
        as_of_date=trading_day - dt.timedelta(days=1),
    )
    # Normalize both date columns to dt.date for the comparison.
    import pandas as pd

    on_dates = {pd.Timestamp(d).date() for d in on["date"].tolist()}
    before_dates = {pd.Timestamp(d).date() for d in before["date"].tolist()}
    assert trading_day in on_dates, f"AAPL did not trade on {trading_day}?"
    assert trading_day not in before_dates, (
        f"Prices PIT leak: {trading_day} visible at as_of={trading_day - dt.timedelta(days=1)}"
    )
