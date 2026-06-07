"""Piotroski F-score (partial v1.0 implementation; see IDEA-S18-002).

v1.0 ships the four signals reliably computable from EDGAR fundamentals
at the current registry:
- ROA > 0 (+1)
- CFO > 0 (+1)
- CFO > NI accrual-quality signal (+1)
- delta current_ratio > 0 (+1)

The "no new shares issued" / "delta gross margin" / "delta leverage" /
"delta asset turnover" / "delta ROA" signals require fields or YoY
deltas the v1.0 registry does not reliably surface; tracked under
IDEA-S18-002.
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


def compute_piotroski_f_score(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> float | None:
    """4-signal F-score over reliably-computable EDGAR fundamentals.

    Returns None when insufficient history for the YoY delta is visible
    or when any required field is null on the relevant FY row.
    """
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
    if len(fy_is) < 1 or len(fy_bs) < 2 or len(fy_cf) < 1:
        return None
    ni_latest = fy_is.iloc[-1].get("net_income")
    cfo_latest = fy_cf.iloc[-1].get("operating_cash_flow")
    total_assets_latest = fy_bs.iloc[-1].get("total_assets")
    current_assets_latest = fy_bs.iloc[-1].get("current_assets")
    current_liab_latest = fy_bs.iloc[-1].get("current_liabilities")
    current_assets_prior = fy_bs.iloc[-2].get("current_assets")
    current_liab_prior = fy_bs.iloc[-2].get("current_liabilities")
    required = (
        ni_latest,
        cfo_latest,
        total_assets_latest,
        current_assets_latest,
        current_liab_latest,
        current_assets_prior,
        current_liab_prior,
    )
    if any(v is None or pd.isna(v) for v in required):
        return None
    if not total_assets_latest or not current_liab_latest or not current_liab_prior:
        return None
    score = 0
    score += 1 if float(ni_latest) / float(total_assets_latest) > 0 else 0
    score += 1 if float(cfo_latest) > 0 else 0
    score += 1 if float(cfo_latest) > float(ni_latest) else 0
    curr_now = float(current_assets_latest) / float(current_liab_latest)
    curr_prior = float(current_assets_prior) / float(current_liab_prior)
    score += 1 if curr_now > curr_prior else 0
    return float(score)
