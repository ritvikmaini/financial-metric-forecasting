"""Beneish 8-variable M-score.

M = -4.84
    + 0.92 * DSRI    (Days Sales in Receivables Index)
    + 0.528 * GMI    (Gross Margin Index)
    + 0.404 * AQI    (Asset Quality Index)
    + 0.892 * SGI    (Sales Growth Index)
    + 0.115 * DEPI   (Depreciation Index)
    - 0.172 * SGAI   (SGA Index)
    + 4.679 * TATA   (Total Accruals to Total Assets)
    - 0.327 * LVGI   (Leverage Index)

Several required fields (receivables, PPE, depreciation, SGA) are NOT
in the v1.0 EDGAR concept_map. The implementation returns None when
any required field is missing on either the latest or prior FY row.
Tracked under IDEA-S18-003 for the schema-gap fixes.
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


def _get(row: pd.Series, field: str) -> float | None:
    if field not in row.index:
        return None
    v = row.get(field)
    if v is None or pd.isna(v):
        return None
    return float(v)


def compute_beneish_m_score(
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
    fy_cf = _fy_tail(cf_df, 2)
    if len(fy_is) < 2 or len(fy_bs) < 2 or len(fy_cf) < 2:
        return None
    is_t, is_p = fy_is.iloc[-1], fy_is.iloc[-2]
    bs_t, bs_p = fy_bs.iloc[-1], fy_bs.iloc[-2]
    cf_t, cf_p = fy_cf.iloc[-1], fy_cf.iloc[-2]

    rev_t = _get(is_t, "revenue")
    rev_p = _get(is_p, "revenue")
    gp_t = _get(is_t, "gross_profit")
    gp_p = _get(is_p, "gross_profit")
    sga_t = _get(is_t, "sga")
    sga_p = _get(is_p, "sga")

    recv_t = _get(bs_t, "receivables")
    recv_p = _get(bs_p, "receivables")
    ca_t = _get(bs_t, "current_assets")
    ca_p = _get(bs_p, "current_assets")
    ppe_t = _get(bs_t, "ppe")
    ppe_p = _get(bs_p, "ppe")
    ta_t = _get(bs_t, "total_assets")
    ta_p = _get(bs_p, "total_assets")
    ltd_t = _get(bs_t, "long_term_debt")
    ltd_p = _get(bs_p, "long_term_debt")
    cl_t = _get(bs_t, "current_liabilities")
    cl_p = _get(bs_p, "current_liabilities")

    dep_t = _get(cf_t, "depreciation")
    dep_p = _get(cf_p, "depreciation")
    ni_t = _get(is_t, "net_income")
    cfo_t = _get(cf_t, "operating_cash_flow")

    required = (
        rev_t,
        rev_p,
        gp_t,
        gp_p,
        sga_t,
        sga_p,
        recv_t,
        recv_p,
        ca_t,
        ca_p,
        ppe_t,
        ppe_p,
        ta_t,
        ta_p,
        ltd_t,
        ltd_p,
        cl_t,
        cl_p,
        dep_t,
        dep_p,
        ni_t,
        cfo_t,
    )
    if any(v is None for v in required):
        return None
    assert rev_t and rev_p and gp_t and gp_p and ta_t and ta_p
    assert dep_t is not None and dep_p is not None and ppe_t is not None and ppe_p is not None
    assert sga_t is not None and sga_p is not None and recv_t is not None and recv_p is not None
    assert ca_t is not None and ca_p is not None
    assert ltd_t is not None and ltd_p is not None and cl_t is not None and cl_p is not None
    assert ni_t is not None and cfo_t is not None

    gm_t = gp_t / rev_t
    gm_p = gp_p / rev_p
    if gm_t == 0 or rev_p == 0 or ta_p == 0:
        return None

    dsri = (recv_t / rev_t) / (recv_p / rev_p)
    gmi = gm_p / gm_t
    aqi_t = 1.0 - (ca_t + ppe_t) / ta_t
    aqi_p = 1.0 - (ca_p + ppe_p) / ta_p
    if aqi_p == 0:
        return None
    aqi = aqi_t / aqi_p
    sgi = rev_t / rev_p
    depi_p = dep_p / (dep_p + ppe_p)
    depi_t = dep_t / (dep_t + ppe_t)
    if depi_t == 0:
        return None
    depi = depi_p / depi_t
    sgai = (sga_t / rev_t) / (sga_p / rev_p)
    tata = (ni_t - cfo_t) / ta_t
    lvgi = ((ltd_t + cl_t) / ta_t) / ((ltd_p + cl_p) / ta_p)

    return (
        -4.84
        + 0.92 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )
