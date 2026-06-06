"""Derived features.

Each derivation calls fetch_pit_series for the components and combines
them. PIT correctness depends on filtering by accepted_date <= as_of
BEFORE selecting "the latest N periods by end_date" — never the reverse.

TTM strategy: prefer the latest visible annual if its end_date is within
~366 days of as_of. Else sum the 4 most recent visible quarters by
end_date AND verify (a) they span ~270 days (i.e., are consecutive) and
(b) the latest of the four is within ~366 days of as_of (recency). Both
guards are load-bearing:
- The span check rejects gaps where a null intervening quarter would
  silently sum a 15-month window (e.g., a forced non-consecutive pick).
- The recency check rejects stale-but-consecutive windows from an
  archival period: JPM has 51 non-null quarterly Revenues for FY2009-
  FY2014 in the fixture (Revenues go null FY2018+); a 2027 as_of
  without the recency guard finds a clean 2013 Q1-Q4 consecutive
  window and silently returns a 13-year-stale TTM.

YoY strategy: period-aligned. Take FY_n vs FY_{n-1} from the two most
recent visible annuals. Falls back to a quarter-windowed YoY (latest 4
vs preceding 4) if no two annuals are visible. The earlier calendar
365-day shift could land just before a prior annual's accepted_date and
silently compare two-year-apart annuals as if they were one year apart.
"""

from __future__ import annotations

import datetime as dt
import math
import uuid

import duckdb
import pandas as pd

from fmf.features.point_in_time import fetch_pit_series

_QUARTERS: frozenset[str] = frozenset({"Q1", "Q2", "Q3", "Q4"})
# Four consecutive quarter end_dates span ~3 quarter-lengths end-to-end
# (~270d). 200-320 brackets normal fiscal-calendar variation; outside
# this window we've reached a non-consecutive quarter.
_TTM_QUARTER_SPAN_MIN_DAYS = 200
_TTM_QUARTER_SPAN_MAX_DAYS = 320
# The latest selected quarter's end_date must be within this many days
# of as_of, mirroring the annual branch's 366d trust window. Without
# this guard, a ticker with a stale-but-consecutive multi-year-old
# quarterly history (e.g., JPM 2009-2014 Revenues; quarterly Revenues
# go null FY2018+) silently returns a 12-year-old TTM at a 2027 as_of.
_TTM_QUARTER_RECENCY_MAX_DAYS = 366


def compute_revenue_ttm(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> float | None:
    """Trailing 12-month revenue.

    Thin wrapper around _ttm_from_series so revenue shares its TTM code
    path with other flow fields (net_income, ebit, gross_profit,
    operating_cash_flow, capital_expenditure). Strategy:
    1. If a visible FY row has end_date within 366 days of as_of, return
       its revenue (= latest annual).
    2. Else sum the 4 most recent visible quarters by end_date AND
       verify they span ~270d (consecutive) AND the latest is within
       ~366d of as_of. If any guard fails, return None.
    3. Else None.
    """
    series = fetch_pit_series(
        conn=conn,
        table="income_statement",
        security_id=security_id,
        as_of_date=as_of_date,
    )
    return _ttm_from_series(series, "revenue", as_of_date)


def _ttm_from_series(
    series: pd.DataFrame,
    field: str,
    as_of_date: dt.date,
) -> float | None:
    """TTM for any flow field.

    Same logic as compute_revenue_ttm: prefer latest visible annual
    within 366d; else 4 consecutive quarters by end_date (200-320d span)
    with the latest within 366d of as_of. Both load-bearing — see
    compute_revenue_ttm docstring for the trap each guard prevents.
    """
    if series.empty or field not in series.columns:
        return None
    fy_rows = series[series["period"] == "FY"]
    if not fy_rows.empty:
        latest_fy = fy_rows.sort_values("end_date").iloc[-1]
        latest_fy_end = _to_date(latest_fy["end_date"])
        if (as_of_date - latest_fy_end).days <= 366:
            v = latest_fy.get(field)
            if v is not None and not _isna(v):
                return float(v)
    q_rows = series[series["period"].isin(_QUARTERS)].dropna(subset=[field])
    if len(q_rows) < 4:
        return None
    last_4 = q_rows.sort_values("end_date", ascending=False).head(4)
    end_dates = sorted(_to_date(d) for d in last_4["end_date"].tolist())
    span_days = (end_dates[-1] - end_dates[0]).days
    if not (_TTM_QUARTER_SPAN_MIN_DAYS <= span_days <= _TTM_QUARTER_SPAN_MAX_DAYS):
        return None
    if (as_of_date - end_dates[-1]).days > _TTM_QUARTER_RECENCY_MAX_DAYS:
        return None
    return float(last_4[field].sum())


def _yoy_growth_from_series(
    series: pd.DataFrame,
    field: str,
) -> float | None:
    """YoY growth for any field. Period-aligned: FY_n vs FY_{n-1}.

    Quarter-windowed fallback omitted — kept in
    compute_revenue_yoy_growth for revenue's specialized case. None of
    the 9 fixture tickers reach the fallback organically.
    """
    if series.empty or field not in series.columns:
        return None
    fy_rows = (
        series[series["period"] == "FY"]
        .dropna(subset=[field])
        .sort_values("end_date", ascending=False)
    )
    if len(fy_rows) >= 2:
        latest = float(fy_rows.iloc[0][field])
        prior = float(fy_rows.iloc[1][field])
        if prior == 0:
            return None
        return (latest - prior) / prior
    return None


def compute_revenue_yoy_growth(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> float | None:
    """YoY revenue growth, period-aligned.

    Strategy:
    1. If two annuals are visible (sorted by end_date desc, both with
       non-null revenue), return (FY_n - FY_{n-1}) / FY_{n-1}.
    2. Else, fall back to a quarter-windowed YoY: the latest 4 visible
       quarters vs the preceding 4. Both windows must independently
       satisfy the consecutive-quarters span check.
    3. Else None.

    Trap avoided: a calendar 365-day shift (TTM at as_of vs TTM at
    as_of - 365d) could land just before the prior annual's
    accepted_date, silently producing a two-year-apart comparison
    labeled as one-year. Period alignment fixes this.
    """
    series = fetch_pit_series(
        conn=conn,
        table="income_statement",
        security_id=security_id,
        as_of_date=as_of_date,
    )
    if series.empty:
        return None

    fy_rows = (
        series[series["period"] == "FY"]
        .dropna(subset=["revenue"])
        .sort_values("end_date", ascending=False)
    )
    if len(fy_rows) >= 2:
        latest = float(fy_rows.iloc[0]["revenue"])
        prior = float(fy_rows.iloc[1]["revenue"])
        if prior == 0:
            return None
        return (latest - prior) / prior

    q_rows = (
        series[series["period"].isin(_QUARTERS)]
        .dropna(subset=["revenue"])
        .sort_values("end_date", ascending=False)
    )
    if len(q_rows) < 8:
        return None
    recent = q_rows.iloc[:4]
    older = q_rows.iloc[4:8]
    # Each window is checked internally for the 200-320d span. We do NOT
    # check that the two windows are adjacent — a null fifth-oldest
    # quarter would slide the "older" window further back and compare
    # 15-month-apart windows. This only fires for a ticker lacking two
    # visible annuals (which would have taken the FY-pair path above);
    # none of the 9 fixture tickers reach this branch organically.
    # Acceptable for v1; tighten if a future ticker exercises it.
    if not (_quarters_consecutive(recent) and _quarters_consecutive(older)):
        return None
    ttm_now = float(recent["revenue"].sum())
    ttm_then = float(older["revenue"].sum())
    if ttm_then == 0:
        return None
    return (ttm_now - ttm_then) / ttm_then


def compute_gross_margin_latest(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> float | None:
    """Gross margin from the latest PIT-visible annual or quarter."""
    series = fetch_pit_series(
        conn=conn,
        table="income_statement",
        security_id=security_id,
        as_of_date=as_of_date,
    )
    if series.empty:
        return None
    candidates = series.dropna(subset=["revenue", "gross_profit"])
    if candidates.empty:
        return None
    latest = candidates.sort_values("end_date").iloc[-1]
    if latest["revenue"] == 0:
        return None
    return float(latest["gross_profit"]) / float(latest["revenue"])


def _to_date(v: object) -> dt.date:
    """Normalize a DuckDB DATE column value to dt.date.

    DuckDB returns dt.date for DATE columns in most versions but
    pd.Timestamp / numpy.datetime64 in some others. Date arithmetic
    against dt.timedelta requires dt.date; normalize at the boundary.

    Ordering matters: check dt.datetime BEFORE dt.date because
    pd.Timestamp is a dt.datetime subclass (and dt.datetime is itself
    a dt.date subclass). A dt.date isinstance check would catch a
    Timestamp and return it unchanged — then date arithmetic against a
    dt.date raises TypeError. See L-INFRA-011.
    """
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    return pd.Timestamp(v).date()  # type: ignore[no-any-return]


def _quarters_consecutive(group: pd.DataFrame) -> bool:
    """Verify four quarter-end dates span the ~270d consecutive window."""
    dates = sorted(_to_date(d) for d in group["end_date"].tolist())
    if len(dates) < 2:
        return False
    span = (dates[-1] - dates[0]).days
    return _TTM_QUARTER_SPAN_MIN_DAYS <= span <= _TTM_QUARTER_SPAN_MAX_DAYS


def _isna(v: object) -> bool:
    """Treat None, NaN, pd.NA as missing."""
    if v is None:
        return True
    return isinstance(v, float) and math.isnan(v)
