"""Dechow accruals hand-calc tests on synthetic in-memory DuckDB rows."""

from __future__ import annotations

import datetime as dt
import math
import uuid
from pathlib import Path

import duckdb

from fmf.features.composites._dechow import compute_dechow_accruals

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


def _insert_is(
    conn: duckdb.DuckDBPyConnection,
    sid: uuid.UUID,
    fy: int,
    end: dt.date,
    net_income: float | None,
) -> None:
    accepted = end + dt.timedelta(days=45)
    conn.execute(
        'INSERT INTO "income_statement" '
        "(security_id, fiscal_year, period, filing_date, accepted_date, end_date, net_income) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [str(sid), fy, "FY", accepted, accepted, end, net_income],
    )


def _insert_bs(
    conn: duckdb.DuckDBPyConnection,
    sid: uuid.UUID,
    fy: int,
    end: dt.date,
    total_assets: float | None,
) -> None:
    accepted = end + dt.timedelta(days=45)
    conn.execute(
        'INSERT INTO "balance_sheet" '
        "(security_id, fiscal_year, period, filing_date, accepted_date, end_date, total_assets) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [str(sid), fy, "FY", accepted, accepted, end, total_assets],
    )


def _insert_cf(
    conn: duckdb.DuckDBPyConnection,
    sid: uuid.UUID,
    fy: int,
    end: dt.date,
    operating_cash_flow: float | None,
) -> None:
    accepted = end + dt.timedelta(days=45)
    conn.execute(
        'INSERT INTO "cashflow" '
        "(security_id, fiscal_year, period, filing_date, accepted_date, end_date, "
        "operating_cash_flow) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [str(sid), fy, "FY", accepted, accepted, end, operating_cash_flow],
    )


def test_dechow_hand_calc_on_synthetic_rows() -> None:
    """NI=100, CFO=120, TA_t=1000, TA_{t-1}=900.

    avg_TA = (1000 + 900) / 2 = 950.
    accruals = (100 - 120) / 950 = -20 / 950 = -0.021052631...
    """
    conn = _mem()
    try:
        sid = uuid.uuid4()
        _insert_security(conn, sid)
        _insert_is(conn, sid, 2022, dt.date(2022, 12, 31), 100.0)
        _insert_bs(conn, sid, 2021, dt.date(2021, 12, 31), 900.0)
        _insert_bs(conn, sid, 2022, dt.date(2022, 12, 31), 1000.0)
        _insert_cf(conn, sid, 2022, dt.date(2022, 12, 31), 120.0)

        accruals = compute_dechow_accruals(
            conn=conn, security_id=sid, as_of_date=dt.date(2023, 6, 1)
        )
        assert accruals is not None
        expected = (100.0 - 120.0) / ((1000.0 + 900.0) / 2.0)
        assert math.isclose(accruals, expected, rel_tol=1e-9)
    finally:
        conn.close()


def test_dechow_none_when_prior_ta_missing() -> None:
    """Only one BS row -> no average_total_assets -> None."""
    conn = _mem()
    try:
        sid = uuid.uuid4()
        _insert_security(conn, sid)
        _insert_is(conn, sid, 2022, dt.date(2022, 12, 31), 100.0)
        _insert_bs(conn, sid, 2022, dt.date(2022, 12, 31), 1000.0)
        _insert_cf(conn, sid, 2022, dt.date(2022, 12, 31), 120.0)

        accruals = compute_dechow_accruals(
            conn=conn, security_id=sid, as_of_date=dt.date(2023, 6, 1)
        )
        assert accruals is None
    finally:
        conn.close()
