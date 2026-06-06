"""Point-in-time extraction primitives.

The core invariant: every read filters by accepted_date <= as_of_date
(for fundamentals) or pulled_at <= as_of_date (for consensus). No
exception. This is the project's defining anti-look-ahead rule.

For fundamentals tables (income_statement / balance_sheet / cashflow)
the series has one row per (fiscal_year, period, end_date). Field-level
PIT assembly (L-INFRA-014): each data column in the synthesized row
takes its LAST_VALUE(IGNORE NULLS) over the group ordered by
accepted_date, so partial re-disclosures (selected-data tables,
multi-year rollforwards) do not null out fields they didn't restate.
S2's collision fix means a single (fiscal_year, period) can legitimately
have multiple end_dates; the series surfaces all of them with their
per-key field-level latest non-null assembly.
"""

from __future__ import annotations

import datetime as dt
import uuid

import duckdb
import pandas as pd

from fmf.data.sql_safety import validate_table_name

_PIT_TABLES: frozenset[str] = frozenset({"income_statement", "balance_sheet", "cashflow"})

_PIT_KEY_COLS: frozenset[str] = frozenset(
    {
        "security_id",
        "fiscal_year",
        "period",
        "filing_date",
        "accepted_date",
        "end_date",
    }
)


def _field_level_pit_select_sql(
    conn: duckdb.DuckDBPyConnection,
    table: str,
) -> str:
    """Build the SELECT clause that assembles one synthesized row per
    (fiscal_year, period, end_date) group using LAST_VALUE(IGNORE NULLS)
    per data column.

    The synthesized row's data columns each take their latest non-null
    value over the group ordered by accepted_date. The synthesized
    row's accepted_date is the MAX in the group (provenance: latest
    restatement that contributed to any field). filing_date is taken
    from the same row as the MAX accepted_date.

    Called by fetch_pit_series and by compute_coverage so the
    coverage metric exactly equals what features deliver.
    """
    cols = [r[0] for r in conn.execute(f'DESCRIBE "{table}"').fetchall()]
    data_cols = [c for c in cols if c not in _PIT_KEY_COLS]
    if not data_cols:
        raise ValueError(f"{table} has no data columns after excluding keys")

    last_value_exprs = ",\n            ".join(
        f'LAST_VALUE("{c}" IGNORE NULLS) OVER ('
        f"PARTITION BY fiscal_year, period, end_date "
        f"ORDER BY accepted_date "
        f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        f') AS "{c}"'
        for c in data_cols
    )

    return f"""
        SELECT
            security_id,
            fiscal_year,
            period,
            filing_date,
            accepted_date,
            end_date,
            {last_value_exprs}
        FROM "{table}"
        WHERE security_id = ?
          AND accepted_date <= ?
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY fiscal_year, period, end_date
            ORDER BY accepted_date DESC
        ) = 1
    """


def fetch_pit_series(
    *,
    conn: duckdb.DuckDBPyConnection,
    table: str,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> pd.DataFrame:
    """Return the time series visible at as_of_date for one security.

    Field-level PIT assembly (L-INFRA-014): for each data field in the
    synthesized series row, the value is LAST_VALUE(field IGNORE NULLS)
    over the (fiscal_year, period, end_date) group ordered by
    accepted_date, restricted to accepted_date <= as_of_date. This is
    the correct semantic for partial re-disclosures (selected-data
    tables, multi-year rollforwards) where a later filing mentions a
    period with fewer fields than the original — omitted fields
    retain their last-known value; restated fields update.

    The synthesized row's accepted_date column is MAX in the group up
    to as_of_date (provenance). Visibility is gated by the
    accepted_date <= as_of_date WHERE clause applied per fact, so the
    row appears as soon as ANY contributing fact is visible — the
    cardinal 1-day-shift test continues to assert visibility via the
    MIN accepted_date pulled from raw data, which is unchanged.

    Used by:
    - Single-period feature compute functions.
    - Multi-period derived features (TTM, YoY, growth).
    - compute_coverage in fmf.features.audit.coverage, via the same
      shared helper so coverage equals what features deliver.
    """
    validate_table_name(table)
    if table not in _PIT_TABLES:
        raise ValueError(
            f"fetch_pit_series requires a fundamentals table with "
            f"accepted_date and end_date columns; {table!r} does not "
            f"qualify. Allowed: {sorted(_PIT_TABLES)}."
        )

    query = _field_level_pit_select_sql(conn, table) + " ORDER BY end_date ASC, period ASC"
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
