"""Cash Conversion Cycle (CCC) in days.

CCC = DIO + DSO - DPO where
- DIO = inventory / (COGS / 365)
- DSO = receivables / (revenue / 365)
- DPO = payables / (COGS / 365)

COGS is derived as revenue - gross_profit. inventory / receivables /
payables are NOT in the v1.0 EDGAR concept_map; this composite returns
None on the current registry's source tables for nearly every filer.
Tracked under IDEA-S18-003; the implementation is shaped so the
schema-gap fix is the only change needed to light it up.
"""

from __future__ import annotations

import datetime as dt
import uuid

import duckdb
import pandas as pd

from fmf.features.point_in_time import fetch_pit_series


def _latest_fy(series: pd.DataFrame) -> pd.Series | None:
    if series.empty or "period" not in series.columns:
        return None
    fy = series[series["period"] == "FY"].sort_values("end_date")
    if fy.empty:
        return None
    return fy.iloc[-1]


def _get(row: pd.Series, field: str) -> float | None:
    if field not in row.index:
        return None
    v = row.get(field)
    if v is None or pd.isna(v):
        return None
    return float(v)


def compute_ccc_days(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> float | None:
    """Returns CCC in days or None when any input is missing.

    IDEA-S18-003: inventory / receivables / payables tag gap blocks
    most filers in v1.0; the function returns None whenever a required
    field is null or absent.
    """
    is_df = fetch_pit_series(
        conn=conn, table="income_statement", security_id=security_id, as_of_date=as_of_date
    )
    bs_df = fetch_pit_series(
        conn=conn, table="balance_sheet", security_id=security_id, as_of_date=as_of_date
    )
    is_row = _latest_fy(is_df)
    bs_row = _latest_fy(bs_df)
    if is_row is None or bs_row is None:
        return None
    revenue = _get(is_row, "revenue")
    gross_profit = _get(is_row, "gross_profit")
    inventory = _get(bs_row, "inventory")
    receivables = _get(bs_row, "receivables")
    payables = _get(bs_row, "payables")
    if any(v is None for v in (revenue, gross_profit, inventory, receivables, payables)):
        return None
    assert revenue is not None and gross_profit is not None
    assert inventory is not None and receivables is not None and payables is not None
    cogs = revenue - gross_profit
    if cogs <= 0 or revenue <= 0:
        return None
    dio = inventory / (cogs / 365.0)
    dso = receivables / (revenue / 365.0)
    dpo = payables / (cogs / 365.0)
    return dio + dso - dpo
