"""Mohanram G-score hand-calc tests on synthetic in-memory DuckDB rows."""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import duckdb

from fmf.features.composites._mohanram import compute_mohanram_g_score

SCHEMA = (Path(__file__).parent.parent.parent.parent / "fmf" / "data" / "schema.sql").read_text()


def _mem() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    c.execute(SCHEMA)
    return c


def _insert_security(conn: duckdb.DuckDBPyConnection, sid: uuid.UUID) -> None:
    conn.execute(
        'INSERT INTO "securities" (security_id, symbol, cik) VALUES (?, ?, ?)',
        [str(sid), "TEST", "0000000001"],
    )


def _ins_is(c: duckdb.DuckDBPyConnection, sid: uuid.UUID, fy: int, end: dt.date, ni: float) -> None:
    accepted = end + dt.timedelta(days=45)
    c.execute(
        'INSERT INTO "income_statement" '
        "(security_id, fiscal_year, period, filing_date, accepted_date, end_date, net_income) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [str(sid), fy, "FY", accepted, accepted, end, ni],
    )


def _ins_bs(c: duckdb.DuckDBPyConnection, sid: uuid.UUID, fy: int, end: dt.date, ta: float) -> None:
    accepted = end + dt.timedelta(days=45)
    c.execute(
        'INSERT INTO "balance_sheet" '
        "(security_id, fiscal_year, period, filing_date, accepted_date, end_date, total_assets) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [str(sid), fy, "FY", accepted, accepted, end, ta],
    )


def _ins_cf(
    c: duckdb.DuckDBPyConnection, sid: uuid.UUID, fy: int, end: dt.date, ocf: float
) -> None:
    accepted = end + dt.timedelta(days=45)
    c.execute(
        'INSERT INTO "cashflow" '
        "(security_id, fiscal_year, period, filing_date, accepted_date, end_date, "
        "operating_cash_flow) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [str(sid), fy, "FY", accepted, accepted, end, ocf],
    )


def test_g_score_hand_calc_all_pass() -> None:
    """All four signals fire.

    ROA_t = 120/1000 = 0.12 > 0 (+1).
    CFO/TA_t = 150/1000 = 0.15 > 0 (+1).
    CFO=150 > NI=120 (+1).
    ROA_p = 90/900 = 0.10. ROA_t=0.12 > 0.10 (+1).
    Score = 4.0.
    """
    conn = _mem()
    try:
        sid = uuid.uuid4()
        _insert_security(conn, sid)
        _ins_is(conn, sid, 2021, dt.date(2021, 12, 31), 90.0)
        _ins_is(conn, sid, 2022, dt.date(2022, 12, 31), 120.0)
        _ins_bs(conn, sid, 2021, dt.date(2021, 12, 31), 900.0)
        _ins_bs(conn, sid, 2022, dt.date(2022, 12, 31), 1000.0)
        _ins_cf(conn, sid, 2022, dt.date(2022, 12, 31), 150.0)

        g = compute_mohanram_g_score(conn=conn, security_id=sid, as_of_date=dt.date(2023, 6, 1))
        assert g == 4.0
    finally:
        conn.close()


def test_g_score_hand_calc_all_fail() -> None:
    """All four signals fail.

    ROA_t = -50/1000 = -0.05, not > 0.
    CFO/TA_t = -100/1000, not > 0.
    CFO=-100 not > NI=-50.
    ROA_p = 80/800 = 0.10. ROA_t=-0.05 not > 0.10.
    Score = 0.0.
    """
    conn = _mem()
    try:
        sid = uuid.uuid4()
        _insert_security(conn, sid)
        _ins_is(conn, sid, 2021, dt.date(2021, 12, 31), 80.0)
        _ins_is(conn, sid, 2022, dt.date(2022, 12, 31), -50.0)
        _ins_bs(conn, sid, 2021, dt.date(2021, 12, 31), 800.0)
        _ins_bs(conn, sid, 2022, dt.date(2022, 12, 31), 1000.0)
        _ins_cf(conn, sid, 2022, dt.date(2022, 12, 31), -100.0)

        g = compute_mohanram_g_score(conn=conn, security_id=sid, as_of_date=dt.date(2023, 6, 1))
        assert g == 0.0
    finally:
        conn.close()


def test_g_score_none_when_prior_year_missing() -> None:
    """Only one IS row -> cannot form ROA delta -> None."""
    conn = _mem()
    try:
        sid = uuid.uuid4()
        _insert_security(conn, sid)
        _ins_is(conn, sid, 2022, dt.date(2022, 12, 31), 120.0)
        _ins_bs(conn, sid, 2022, dt.date(2022, 12, 31), 1000.0)
        _ins_cf(conn, sid, 2022, dt.date(2022, 12, 31), 150.0)

        g = compute_mohanram_g_score(conn=conn, security_id=sid, as_of_date=dt.date(2023, 6, 1))
        assert g is None
    finally:
        conn.close()
