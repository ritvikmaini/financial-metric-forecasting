"""As-of sample grids.

A grid strategy decides which (security_id, as_of_date) decision points
the feature matrix represents. Different strategies suit different
modeling tasks:

- filing_dates_grid (default): every fundamentals filing's accepted_date
  per security. Captures every event where new fundamentals became
  visible. ~580 samples for the 9-ticker fixture. The natural grid for
  fundamentals-driven forecasting models.
- fiscal_year_end_grid: one as_of per (security, fiscal_year), at the
  FY 10-K's accepted_date. Coarser — 146 samples for the fixture.
  Matches the T4 audit's sample.
- quarterly_grid: per (security, fiscal_year, period). Like filing_dates
  but deduped to one as_of per filed period.
- daily_calendar_grid: one as_of per calendar day in [start, end].
  Use for backtest grids that decouple from filing cadence (e.g., when
  the target is a forward N-day return).
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import duckdb
import pandas as pd


@dataclass(frozen=True, slots=True)
class AsOfSample:
    security_id: uuid.UUID
    symbol: str
    as_of_date: dt.date
    as_of_source: str  # e.g., "income_statement.FY.2023" or "calendar.2024-01-15"


def _to_date(value: object) -> dt.date:
    """Coerce a DuckDB DATE fetchall result to a stable dt.date.

    DuckDB returns DATE values as either dt.date or pd.Timestamp depending
    on the version / driver path. The L-INFRA-011 trap: comparing
    pd.Timestamp to dt.date silently widens types and breaks the
    accepted_date <= as_of_date predicate. Normalize defensively.
    """
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        result: dt.date = value.date()
        return result
    if isinstance(value, dt.date):
        return value
    raise TypeError(f"cannot coerce {type(value).__name__} to dt.date: {value!r}")


def filing_dates_grid(
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    symbol: str,
) -> list[AsOfSample]:
    """One as_of per distinct (fiscal_year, period, accepted_date) in
    income_statement. The source tag is "income_statement.<period>.<fy>"."""
    rows = conn.execute(
        "SELECT DISTINCT fiscal_year, period, accepted_date "
        'FROM "income_statement" '
        "WHERE security_id = ? "
        "ORDER BY accepted_date ASC",
        [str(security_id)],
    ).fetchall()
    return [
        AsOfSample(
            security_id=security_id,
            symbol=symbol,
            as_of_date=_to_date(accepted),
            as_of_source=f"income_statement.{period}.{fy}",
        )
        for fy, period, accepted in rows
    ]


def fiscal_year_end_grid(
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    symbol: str,
) -> list[AsOfSample]:
    """One as_of per (fiscal_year) at MIN(accepted_date) of the FY filing.

    MIN, not MAX: each subsequent 10-K re-files prior fiscal years as
    comparative columns, so MAX(accepted_date) per fiscal_year always
    collapses to the latest 10-K's date — which means the FY.2023 sample
    would land at 2025-10-31 and see FY2025 data instead of FY2023 data.
    Using MIN ties the as_of to the original 10-K's accepted_date,
    which is the genuine "decision point" where the fiscal year first
    became publicly known. This is what the anchor spot-check expects
    (AAPL FY2023 revenue_ttm = 383B at 2023-11-03).
    """
    rows = conn.execute(
        "SELECT fiscal_year, MIN(accepted_date) AS as_of "
        'FROM "income_statement" '
        "WHERE security_id = ? AND period = ? "
        "GROUP BY fiscal_year "
        "ORDER BY as_of ASC",
        [str(security_id), "FY"],
    ).fetchall()
    return [
        AsOfSample(
            security_id=security_id,
            symbol=symbol,
            as_of_date=_to_date(accepted),
            as_of_source=f"income_statement.FY.{fy}",
        )
        for fy, accepted in rows
    ]


def quarterly_grid(
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    symbol: str,
) -> list[AsOfSample]:
    """One as_of per (fiscal_year, period) at MAX(accepted_date)."""
    rows = conn.execute(
        "SELECT fiscal_year, period, MAX(accepted_date) AS as_of "
        'FROM "income_statement" '
        "WHERE security_id = ? "
        "GROUP BY fiscal_year, period "
        "ORDER BY as_of ASC",
        [str(security_id)],
    ).fetchall()
    return [
        AsOfSample(
            security_id=security_id,
            symbol=symbol,
            as_of_date=_to_date(accepted),
            as_of_source=f"income_statement.{period}.{fy}",
        )
        for fy, period, accepted in rows
    ]


def daily_calendar_grid(
    conn: duckdb.DuckDBPyConnection,  # noqa: ARG001 — kept for strategy uniformity
    security_id: uuid.UUID,
    symbol: str,
    *,
    start: dt.date,
    end: dt.date,
) -> list[AsOfSample]:
    """One as_of per calendar day in [start, end] inclusive."""
    samples: list[AsOfSample] = []
    d = start
    while d <= end:
        samples.append(
            AsOfSample(
                security_id=security_id,
                symbol=symbol,
                as_of_date=d,
                as_of_source=f"calendar.{d.isoformat()}",
            )
        )
        d += dt.timedelta(days=1)
    return samples


GridStrategy = Callable[..., list[AsOfSample]]
