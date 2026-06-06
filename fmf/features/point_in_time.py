"""Point-in-time extraction primitives.

The core invariant: every read filters by accepted_date <= as_of_date
(for fundamentals) or pulled_at <= as_of_date (for consensus). No
exception. This is the project's defining anti-look-ahead rule.

For fundamentals tables (income_statement / balance_sheet / cashflow)
the series has one row per (fiscal_year, period, end_date) — picking
the latest accepted_date per key. S2's collision fix means a single
(fiscal_year, period) can legitimately have multiple end_dates; the
series surfaces all of them with their per-key latest accepted version.
"""

from __future__ import annotations

import datetime as dt
import uuid

import duckdb
import pandas as pd

from fmf.data.sql_safety import validate_table_name

_PIT_TABLES: frozenset[str] = frozenset({"income_statement", "balance_sheet", "cashflow"})


def fetch_pit_series(
    *,
    conn: duckdb.DuckDBPyConnection,
    table: str,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> pd.DataFrame:
    """Return the time series visible at as_of_date for one security.

    Filters to accepted_date <= as_of_date and picks the latest
    accepted_date per (fiscal_year, period, end_date). The result is
    one row per period-end, ordered by end_date ascending.

    Used by:
    - Single-period feature compute functions (most recent FY or quarter).
    - Multi-period derived features (TTM, YoY, growth).
    """
    validate_table_name(table)
    if table not in _PIT_TABLES:
        raise ValueError(
            f"fetch_pit_series requires a fundamentals table with accepted_date "
            f"and end_date columns; {table!r} does not qualify. "
            f"Allowed: {sorted(_PIT_TABLES)}."
        )

    query = f"""
        SELECT *
        FROM "{table}"
        WHERE security_id = ?
          AND accepted_date <= ?
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY fiscal_year, period, end_date
            ORDER BY accepted_date DESC
        ) = 1
        ORDER BY end_date ASC, period ASC
    """
    return conn.execute(query, [str(security_id), as_of_date]).fetchdf()


def fetch_consensus_pit(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> pd.DataFrame:
    """Return analyst_estimates rows for one security visible at as_of_date.

    Filters by pulled_at <= as_of_date. yfinance consensus is a snapshot
    pulled near today; for any historical as_of_date the result is
    empty. This is correct: using today's snapshot as if it were known
    in the past is look-ahead. Callers must handle empty gracefully —
    the benchmark falls back to naive/statistical baselines.
    """
    query = """
        SELECT *
        FROM "analyst_estimates"
        WHERE security_id = ?
          AND pulled_at <= ?
        ORDER BY target_date ASC, pulled_at DESC
    """
    # as_of_date is a date but pulled_at is a timestamp. Compare against
    # the end-of-day timestamp so a same-day pull is included.
    as_of_ts = dt.datetime.combine(as_of_date, dt.time(23, 59, 59))
    return conn.execute(query, [str(security_id), as_of_ts]).fetchdf()


def fetch_prices_pit(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> pd.DataFrame:
    """Return the prices time series visible at as_of_date for one security.

    Prices carry no accepted_date — the row is the public market record
    of that trading day, so the PIT filter is simply date <= as_of_date.
    Returns one row per trading day ordered by date ascending. Used by:
    - Price-based features (close_latest, returns_1m/3m/6m/12m, volatility,
      max_drawdown_1y).
    - Composite ratios that combine prices with fundamentals (pe_ratio_ttm =
      close_latest / eps_diluted_ttm).

    Note: pb / ps_ttm / fcf_yield require a shares-outstanding column that
    S2's concept_map does NOT currently populate. Those features are
    deferred until S2 is extended (see T3 step 0).
    """
    query = """
        SELECT *
        FROM "prices"
        WHERE security_id = ?
          AND "date" <= ?
        ORDER BY "date" ASC
    """
    return conn.execute(query, [str(security_id), as_of_date]).fetchdf()
