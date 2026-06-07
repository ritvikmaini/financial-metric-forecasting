"""Dechow accruals (modified balance-sheet form).

Modified form well-supported in the literature:
    accruals = (NI - CFO) / average_total_assets

with average_total_assets = (TA_t + TA_{t-1}) / 2. Equivalent quality
signal to the full Dechow form for the columns the v1.0 registry has.
Returns None when any input is missing or when average total assets is
non-positive.
"""

from __future__ import annotations

import datetime as dt
import uuid

import duckdb
import pandas as pd

from fmf.features.point_in_time import fetch_pit_series


def _fy_tail(series: pd.DataFrame, n: int) -> pd.DataFrame:
    if series.empty or "period" not in series.columns:
        return series.iloc[0:0]
    return series[series["period"] == "FY"].sort_values("end_date").tail(n)


def compute_dechow_accruals(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> float | None:
    is_df = fetch_pit_series(
        conn=conn, table="income_statement", security_id=security_id, as_of_date=as_of_date
    )
    bs_df = fetch_pit_series(
        conn=conn, table="balance_sheet", security_id=security_id, as_of_date=as_of_date
    )
    cf_df = fetch_pit_series(
        conn=conn, table="cashflow", security_id=security_id, as_of_date=as_of_date
    )
    fy_is = _fy_tail(is_df, 1)
    fy_bs = _fy_tail(bs_df, 2)
    fy_cf = _fy_tail(cf_df, 1)
    if len(fy_is) < 1 or len(fy_bs) < 2 or len(fy_cf) < 1:
        return None
    ni = fy_is.iloc[-1].get("net_income")
    cfo = fy_cf.iloc[-1].get("operating_cash_flow")
    ta_latest = fy_bs.iloc[-1].get("total_assets")
    ta_prior = fy_bs.iloc[-2].get("total_assets")
    if any(v is None or pd.isna(v) for v in (ni, cfo, ta_latest, ta_prior)):
        return None
    avg_ta = (float(ta_latest) + float(ta_prior)) / 2.0
    if avg_ta <= 0:
        return None
    return (float(ni) - float(cfo)) / avg_ta
