"""Analyst consensus ingest from yfinance — SNAPSHOT, NOT HISTORY.

LIMITATION: yfinance exposes only current EPS / revenue estimates per
period (e.g., next quarter, current year). It does NOT expose historical
revisions of those estimates. We store each pull as a row with
`pulled_at = now()`; the PIT proxy is `pulled_at <= as_of_date`, treating
the snapshot as if it were known at the time it was pulled. This is
weaker than IBES-style historical revisions and the limitation flows into
the README's honest-framing note: the benchmark leans on naive and
statistical baselines (random walk, AR(1), seasonal naive, last-year),
with consensus as a caveated secondary reference, not the headline.

Period labels: yfinance's `earnings_estimate` / `revenue_estimate`
indexes are relative labels (`0q`, `+1q`, `0y`, `+1y`) referenced to
fetch time. We anchor on `pulled_at.date()` and map via calendar quarter
/ year ends. The mapping is intentionally calendar-based, not
fiscal-calendar-aware, since yfinance's labels are calendar-quarter-based
even for non-calendar filers (e.g., AAPL with Sep FY-end). target_date
is used only to bin estimates; the snapshot is a caveated reference.
"""

from __future__ import annotations

import datetime as dt
import uuid

import duckdb
import pandas as pd

from fmf.data.connectors import bulk_load
from fmf.data.yfinance._client import YFinanceClient


def _add_months(d: dt.date, months: int) -> dt.date:
    new_month = d.month + months
    new_year = d.year + (new_month - 1) // 12
    new_month = ((new_month - 1) % 12) + 1
    day = min(d.day, 28)
    return dt.date(new_year, new_month, day)


def _end_of_quarter(d: dt.date) -> dt.date:
    q_end_month = ((d.month - 1) // 3 + 1) * 3
    if q_end_month == 12:
        return dt.date(d.year, 12, 31)
    if q_end_month == 3:
        return dt.date(d.year, 3, 31)
    if q_end_month == 6:
        return dt.date(d.year, 6, 30)
    return dt.date(d.year, 9, 30)


def _period_label_to_target_date(label: str, anchor_date: dt.date) -> dt.date | None:
    """Map yfinance period labels to approximate target dates.

    - "0q"  / "currentquarter" → end of the calendar quarter containing anchor_date
    - "+1q" / "nextquarter"    → end of the next calendar quarter
    - "0y"  / "currentyear"    → Dec 31 of anchor_date.year
    - "+1y" / "nextyear"       → Dec 31 of anchor_date.year + 1

    Returns None for unrecognised labels.
    """
    label = label.strip().lower()
    if label in ("0q", "currentquarter"):
        return _end_of_quarter(anchor_date)
    if label in ("+1q", "nextquarter"):
        return _end_of_quarter(_add_months(anchor_date, 3))
    if label in ("0y", "currentyear"):
        return dt.date(anchor_date.year, 12, 31)
    if label in ("+1y", "nextyear"):
        return dt.date(anchor_date.year + 1, 12, 31)
    return None


def _rows_from_estimate_df(
    df: pd.DataFrame,
    *,
    security_id: uuid.UUID,
    metric: str,
    pulled_at: dt.datetime,
) -> pd.DataFrame:
    """Convert a yfinance earnings_estimate or revenue_estimate DataFrame
    to analyst_estimates schema rows.
    """
    if df.empty:
        return pd.DataFrame()
    out_rows: list[dict[str, object]] = []
    anchor_date = pulled_at.date()
    for period_label, row in df.iterrows():
        target_date = _period_label_to_target_date(str(period_label), anchor_date)
        if target_date is None:
            continue
        consensus = row.get("avg")
        n_analysts = row.get("numberOfAnalysts")
        if consensus is None or pd.isna(consensus):
            continue
        out_rows.append(
            {
                "security_id": str(security_id),
                "target_date": target_date,
                "pulled_at": pulled_at,
                "metric": metric,
                "consensus": float(consensus),
                "n_analysts": int(n_analysts) if n_analysts and not pd.isna(n_analysts) else None,
            }
        )
    return pd.DataFrame(out_rows)


def ingest_consensus_snapshot(
    *,
    conn: duckdb.DuckDBPyConnection,
    client: YFinanceClient,
    ticker: str,
    security_id: uuid.UUID,
    pulled_at: dt.datetime,
) -> int:
    """Pull current consensus snapshot for ticker; write rows for eps and revenue.

    SNAPSHOT, NOT HISTORY. See module docstring.
    """
    total = 0
    try:
        eps = client.fetch_earnings_estimate(ticker)
        eps_rows = _rows_from_estimate_df(
            eps, security_id=security_id, metric="eps", pulled_at=pulled_at
        )
        if not eps_rows.empty:
            total += bulk_load(conn=conn, table="analyst_estimates", df=eps_rows)
    except (FileNotFoundError, KeyError, ValueError):
        pass  # tolerate ticker without estimates
    try:
        rev = client.fetch_revenue_estimate(ticker)
        rev_rows = _rows_from_estimate_df(
            rev, security_id=security_id, metric="revenue", pulled_at=pulled_at
        )
        if not rev_rows.empty:
            total += bulk_load(conn=conn, table="analyst_estimates", df=rev_rows)
    except (FileNotFoundError, KeyError, ValueError):
        pass
    return total
