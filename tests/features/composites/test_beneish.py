"""Beneish M-score hand-calc tests on synthetic in-memory DuckDB rows.

The v1.0 schema does not carry receivables, ppe, depreciation, sga; the
hand-calc test extends the in-memory schema with those columns. IDEA-S18-003
covers the EDGAR concept-map fix to light it up on real data.
"""

from __future__ import annotations

import datetime as dt
import math
import uuid
from pathlib import Path

import duckdb

from fmf.features.composites._beneish import compute_beneish_m_score

SCHEMA = (Path(__file__).parent.parent.parent.parent / "fmf" / "data" / "schema.sql").read_text()


def _mem_extended() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    c.execute(SCHEMA)
    c.execute('ALTER TABLE "income_statement" ADD COLUMN sga DOUBLE')
    c.execute('ALTER TABLE "balance_sheet" ADD COLUMN receivables DOUBLE')
    c.execute('ALTER TABLE "balance_sheet" ADD COLUMN ppe DOUBLE')
    c.execute('ALTER TABLE "cashflow" ADD COLUMN depreciation DOUBLE')
    return c


def _insert_security(conn: duckdb.DuckDBPyConnection, sid: uuid.UUID) -> None:
    conn.execute(
        'INSERT INTO "securities" (security_id, symbol, cik) VALUES (?, ?, ?)',
        [str(sid), "TEST", "0000000001"],
    )


def test_beneish_hand_calc_on_synthetic_rows() -> None:
    """Two FY rows with the Beneish 8-variable inputs.

    Year t-1: rev=1000 gp=400 sga=100 recv=200 ca=300 ppe=500 ta=1000
              ltd=200 cl=100 dep=50 ni=80 cfo=90.
    Year t:   rev=1200 gp=400 sga=140 recv=300 ca=400 ppe=500 ta=1200
              ltd=240 cl=120 dep=60 ni=100 cfo=80.

    DSRI = (300/1200) / (200/1000) = 0.25 / 0.20 = 1.25
    GMI  = (400/1000) / (400/1200) = 0.40 / (1/3) = 1.2
    AQI  = (1 - (400+500)/1200) / (1 - (300+500)/1000)
         = 0.25 / 0.20 = 1.25
    SGI  = 1200 / 1000 = 1.2
    DEPI = (50/(50+500)) / (60/(60+500)) = (50/550) / (60/560)
    SGAI = (140/1200) / (100/1000)
    TATA = (100 - 80) / 1200 = 20/1200
    LVGI = ((240+120)/1200) / ((200+100)/1000) = 0.30 / 0.30 = 1.0
    """
    conn = _mem_extended()
    try:
        sid = uuid.uuid4()
        _insert_security(conn, sid)

        rows_t = (2022, dt.date(2022, 12, 31), dt.date(2023, 2, 1))
        rows_p = (2021, dt.date(2021, 12, 31), dt.date(2022, 2, 1))

        def ins_is(
            fy: int,
            end: dt.date,
            accepted: dt.date,
            rev: float,
            gp: float,
            ni: float,
            sga: float,
        ) -> None:
            conn.execute(
                'INSERT INTO "income_statement" '
                "(security_id, fiscal_year, period, filing_date, accepted_date, end_date, "
                "revenue, gross_profit, net_income, sga) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [str(sid), fy, "FY", accepted, accepted, end, rev, gp, ni, sga],
            )

        def ins_bs(
            fy: int,
            end: dt.date,
            accepted: dt.date,
            ca: float,
            cl: float,
            ta: float,
            ltd: float,
            recv: float,
            ppe: float,
        ) -> None:
            conn.execute(
                'INSERT INTO "balance_sheet" '
                "(security_id, fiscal_year, period, filing_date, accepted_date, end_date, "
                "total_assets, current_assets, current_liabilities, long_term_debt, "
                "receivables, ppe) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [str(sid), fy, "FY", accepted, accepted, end, ta, ca, cl, ltd, recv, ppe],
            )

        def ins_cf(fy: int, end: dt.date, accepted: dt.date, ocf: float, dep: float) -> None:
            conn.execute(
                'INSERT INTO "cashflow" '
                "(security_id, fiscal_year, period, filing_date, accepted_date, end_date, "
                "operating_cash_flow, depreciation) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [str(sid), fy, "FY", accepted, accepted, end, ocf, dep],
            )

        ins_is(rows_p[0], rows_p[1], rows_p[2], 1000.0, 400.0, 80.0, 100.0)
        ins_is(rows_t[0], rows_t[1], rows_t[2], 1200.0, 400.0, 100.0, 140.0)
        ins_bs(rows_p[0], rows_p[1], rows_p[2], 300.0, 100.0, 1000.0, 200.0, 200.0, 500.0)
        ins_bs(rows_t[0], rows_t[1], rows_t[2], 400.0, 120.0, 1200.0, 240.0, 300.0, 500.0)
        ins_cf(rows_p[0], rows_p[1], rows_p[2], 90.0, 50.0)
        ins_cf(rows_t[0], rows_t[1], rows_t[2], 80.0, 60.0)

        m = compute_beneish_m_score(conn=conn, security_id=sid, as_of_date=dt.date(2023, 6, 1))
        assert m is not None

        dsri = (300 / 1200) / (200 / 1000)
        gmi = (400 / 1000) / (400 / 1200)
        aqi = (1 - (400 + 500) / 1200) / (1 - (300 + 500) / 1000)
        sgi = 1200 / 1000
        depi = (50 / (50 + 500)) / (60 / (60 + 500))
        sgai = (140 / 1200) / (100 / 1000)
        tata = (100 - 80) / 1200
        lvgi = ((240 + 120) / 1200) / ((200 + 100) / 1000)
        expected = (
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
        assert math.isclose(m, expected, rel_tol=1e-9)
    finally:
        conn.close()


def test_beneish_none_when_receivables_missing() -> None:
    """Even with all other fields populated, missing receivables -> None."""
    conn = _mem_extended()
    try:
        sid = uuid.uuid4()
        _insert_security(conn, sid)
        for fy, end in ((2021, dt.date(2021, 12, 31)), (2022, dt.date(2022, 12, 31))):
            accepted = end + dt.timedelta(days=45)
            conn.execute(
                'INSERT INTO "income_statement" '
                "(security_id, fiscal_year, period, filing_date, accepted_date, end_date, "
                "revenue, gross_profit, net_income, sga) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [str(sid), fy, "FY", accepted, accepted, end, 1000.0, 400.0, 80.0, 100.0],
            )
            conn.execute(
                'INSERT INTO "balance_sheet" '
                "(security_id, fiscal_year, period, filing_date, accepted_date, end_date, "
                "total_assets, current_assets, current_liabilities, long_term_debt, ppe) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [str(sid), fy, "FY", accepted, accepted, end, 1000.0, 300.0, 100.0, 200.0, 500.0],
            )
            conn.execute(
                'INSERT INTO "cashflow" '
                "(security_id, fiscal_year, period, filing_date, accepted_date, end_date, "
                "operating_cash_flow, depreciation) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [str(sid), fy, "FY", accepted, accepted, end, 90.0, 50.0],
            )

        m = compute_beneish_m_score(conn=conn, security_id=sid, as_of_date=dt.date(2023, 6, 1))
        assert m is None
    finally:
        conn.close()
