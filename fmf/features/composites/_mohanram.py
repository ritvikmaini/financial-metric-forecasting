"""Mohanram G-score (partial v1.0 implementation; see IDEA-S18-002).

Mohanram (2005) is an 8-point growth-firm quality score using
industry-relative comparisons (ROA, CFO/A, earnings variability,
sales-growth variability, R&D intensity, capex intensity, advertising
intensity, accrual quality). The full form requires industry medians
and R&D / advertising tags that the v1.0 EDGAR concept_map does not
populate.

v1.0 ships the 4-signal subset that is reliably computable on existing
columns, using absolute-zero thresholds where the industry-median form
would otherwise sit:

- ROA > 0 (+1)
- CFO / total_assets > 0 (+1)
- CFO > NI (accrual quality) (+1)
- ROA_t > ROA_{t-1} (ROA improving) (+1)

The R&D / advertising / capex intensity and variability signals are
tracked under IDEA-S18-002 for the concept-map fix.
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


def compute_mohanram_g_score(
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
    fy_is = _fy_tail(is_df, 2)
    fy_bs = _fy_tail(bs_df, 2)
    fy_cf = _fy_tail(cf_df, 1)
    if len(fy_is) < 2 or len(fy_bs) < 2 or len(fy_cf) < 1:
        return None
    ni_t = fy_is.iloc[-1].get("net_income")
    ni_p = fy_is.iloc[-2].get("net_income")
    ta_t = fy_bs.iloc[-1].get("total_assets")
    ta_p = fy_bs.iloc[-2].get("total_assets")
    cfo_t = fy_cf.iloc[-1].get("operating_cash_flow")
    if any(v is None or pd.isna(v) for v in (ni_t, ni_p, ta_t, ta_p, cfo_t)):
        return None
    if not ta_t or not ta_p:
        return None
    roa_t = float(ni_t) / float(ta_t)
    roa_p = float(ni_p) / float(ta_p)
    score = 0
    score += 1 if roa_t > 0 else 0
    score += 1 if float(cfo_t) / float(ta_t) > 0 else 0
    score += 1 if float(cfo_t) > float(ni_t) else 0
    score += 1 if roa_t > roa_p else 0
    return float(score)
