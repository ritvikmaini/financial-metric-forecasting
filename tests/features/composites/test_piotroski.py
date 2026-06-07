"""Piotroski F-score hand-calc tests on synthetic in-memory DuckDB rows."""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import duckdb
import pytest

from fmf.features.composites._piotroski import compute_piotroski_f_score

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
    accepted: dt.date,
    end: dt.date,
    net_income: float | None,
) -> None:
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
    accepted: dt.date,
    end: dt.date,
    total_assets: float | None,
    current_assets: float | None,
    current_liabilities: float | None,
) -> None:
    conn.execute(
        'INSERT INTO "balance_sheet" '
        "(security_id, fiscal_year, period, filing_date, accepted_date, end_date, "
        "total_assets, current_assets, current_liabilities) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            str(sid),
            fy,
            "FY",
            accepted,
            accepted,
            end,
            total_assets,
            current_assets,
            current_liabilities,
        ],
    )


def _insert_cf(
    conn: duckdb.DuckDBPyConnection,
    sid: uuid.UUID,
    fy: int,
    accepted: dt.date,
    end: dt.date,
    operating_cash_flow: float | None,
) -> None:
    conn.execute(
        'INSERT INTO "cashflow" '
        "(security_id, fiscal_year, period, filing_date, accepted_date, end_date, "
        "operating_cash_flow) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [str(sid), fy, "FY", accepted, accepted, end, operating_cash_flow],
    )


def test_f_score_hand_calc_on_synthetic_rows() -> None:
    """All four +1 conditions hit. Hand-calc: 4.0.

    NI=100, total_assets=1000 -> ROA=0.10 > 0 (+1).
    CFO=120 > 0 (+1).
    CFO=120 > NI=100 (+1).
    current_ratio_now = 200/100 = 2.0 vs prior = 150/100 = 1.5 (+1).
    """
    conn = _mem()
    try:
        sid = uuid.uuid4()
        _insert_security(conn, sid)
        _insert_is(conn, sid, 2022, dt.date(2023, 2, 1), dt.date(2022, 12, 31), 100.0)
        _insert_bs(
            conn,
            sid,
            2021,
            dt.date(2022, 2, 1),
            dt.date(2021, 12, 31),
            total_assets=900.0,
            current_assets=150.0,
            current_liabilities=100.0,
        )
        _insert_bs(
            conn,
            sid,
            2022,
            dt.date(2023, 2, 1),
            dt.date(2022, 12, 31),
            total_assets=1000.0,
            current_assets=200.0,
            current_liabilities=100.0,
        )
        _insert_cf(conn, sid, 2022, dt.date(2023, 2, 1), dt.date(2022, 12, 31), 120.0)

        score = compute_piotroski_f_score(
            conn=conn, security_id=sid, as_of_date=dt.date(2023, 6, 1)
        )
        assert score == 4.0
    finally:
        conn.close()


def test_f_score_zero_when_all_signals_fail() -> None:
    """All four conditions fail.

    NI=-50, total_assets=1000 -> ROA=-0.05, not > 0.
    CFO=-100, not > 0.
    CFO=-100 not > NI=-50.
    current_now=100/200=0.5 vs prior=150/100=1.5, not > prior.
    """
    conn = _mem()
    try:
        sid = uuid.uuid4()
        _insert_security(conn, sid)
        _insert_is(conn, sid, 2022, dt.date(2023, 2, 1), dt.date(2022, 12, 31), -50.0)
        _insert_bs(
            conn,
            sid,
            2021,
            dt.date(2022, 2, 1),
            dt.date(2021, 12, 31),
            total_assets=900.0,
            current_assets=150.0,
            current_liabilities=100.0,
        )
        _insert_bs(
            conn,
            sid,
            2022,
            dt.date(2023, 2, 1),
            dt.date(2022, 12, 31),
            total_assets=1000.0,
            current_assets=100.0,
            current_liabilities=200.0,
        )
        _insert_cf(conn, sid, 2022, dt.date(2023, 2, 1), dt.date(2022, 12, 31), -100.0)

        score = compute_piotroski_f_score(
            conn=conn, security_id=sid, as_of_date=dt.date(2023, 6, 1)
        )
        assert score == 0.0
    finally:
        conn.close()


def test_f_score_none_when_insufficient_history() -> None:
    """Only one FY balance-sheet row -> cannot form delta current_ratio."""
    conn = _mem()
    try:
        sid = uuid.uuid4()
        _insert_security(conn, sid)
        _insert_is(conn, sid, 2022, dt.date(2023, 2, 1), dt.date(2022, 12, 31), 100.0)
        _insert_bs(
            conn,
            sid,
            2022,
            dt.date(2023, 2, 1),
            dt.date(2022, 12, 31),
            total_assets=1000.0,
            current_assets=200.0,
            current_liabilities=100.0,
        )
        _insert_cf(conn, sid, 2022, dt.date(2023, 2, 1), dt.date(2022, 12, 31), 120.0)

        score = compute_piotroski_f_score(
            conn=conn, security_id=sid, as_of_date=dt.date(2023, 6, 1)
        )
        assert score is None
    finally:
        conn.close()


@pytest.fixture
def fixture_conn() -> duckdb.DuckDBPyConnection:
    fixture = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "mini.duckdb"
    if not fixture.exists():
        pytest.skip("fixture not built")
    return duckdb.connect(str(fixture), read_only=True)


def test_f_score_runs_on_fixture_anchor(fixture_conn: duckdb.DuckDBPyConnection) -> None:
    row = fixture_conn.execute(
        'SELECT security_id FROM "securities" WHERE symbol = ?', ["AAPL"]
    ).fetchone()
    assert row is not None
    sid = uuid.UUID(str(row[0]))
    score = compute_piotroski_f_score(
        conn=fixture_conn, security_id=sid, as_of_date=dt.date(2023, 12, 31)
    )
    assert score is None or (0.0 <= score <= 4.0)
