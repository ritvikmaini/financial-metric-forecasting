"""Per-row target + naive baseline lookup tests.

Mix of fixture-DB anchor checks and a deterministic in-memory hand-built
test that exercises the comparative-trap bug path regardless of how the
F1 grid lands on real filings.
"""

from __future__ import annotations

import datetime as dt
import uuid

import duckdb

from fmf.equity.forecasting.evaluation._target_lookup import (
    last_fy_actual,
    next_fy_target,
)
from tests.equity.forecasting.evaluation._fixture_helpers import (
    aapl_security_id,
    fixture_conn,
)


def test_next_fy_target_returns_next_undisclosed_fy() -> None:
    conn = fixture_conn()
    try:
        aapl = aapl_security_id(conn)
        # AAPL FY2023 10-K accepted around 2023-11-03; FY2024 around 2024-11-01.
        result = next_fy_target(
            conn=conn,
            security_id=aapl,
            as_of_date=dt.date(2024, 5, 15),
            metric="eps_diluted",
        )
        assert result is not None
        assert result.fiscal_year == 2024
        assert result.accepted_date > dt.date(2024, 5, 15)
    finally:
        conn.close()


def test_next_fy_target_strict_greater_excludes_same_day() -> None:
    conn = fixture_conn()
    try:
        aapl = aapl_security_id(conn)
        # Look up AAPL FY2023's true first-visible accepted_date.
        row = conn.execute(
            'SELECT MIN(accepted_date) FROM "income_statement" '
            "WHERE security_id = ? AND period = 'FY' AND fiscal_year = ? "
            "AND eps_diluted IS NOT NULL",
            [str(aapl), 2023],
        ).fetchone()
        assert row is not None and row[0] is not None
        fy2023_first = row[0]
        if isinstance(fy2023_first, dt.datetime):
            fy2023_first = fy2023_first.date()
        # as_of equal to FY2023 first-visible should skip FY2023 -> return FY2024.
        result = next_fy_target(
            conn=conn,
            security_id=aapl,
            as_of_date=fy2023_first,
            metric="eps_diluted",
        )
        assert result is not None
        assert result.fiscal_year == 2024
    finally:
        conn.close()


def test_next_fy_target_returns_none_when_exhausted() -> None:
    conn = fixture_conn()
    try:
        aapl = aapl_security_id(conn)
        result = next_fy_target(
            conn=conn,
            security_id=aapl,
            as_of_date=dt.date(2099, 1, 1),
            metric="eps_diluted",
        )
        assert result is None
    finally:
        conn.close()


def test_last_fy_actual_returns_most_recent_visible_fy() -> None:
    conn = fixture_conn()
    try:
        aapl = aapl_security_id(conn)
        baseline = last_fy_actual(
            conn=conn,
            security_id=aapl,
            as_of_date=dt.date(2024, 5, 15),
            metric="eps_diluted",
        )
        assert baseline is not None
        assert baseline > 0
    finally:
        conn.close()


def test_last_fy_actual_returns_none_before_first_filing() -> None:
    conn = fixture_conn()
    try:
        aapl = aapl_security_id(conn)
        baseline = last_fy_actual(
            conn=conn,
            security_id=aapl,
            as_of_date=dt.date(1990, 1, 1),
            metric="eps_diluted",
        )
        assert baseline is None
    finally:
        conn.close()


def test_next_fy_target_skips_comparative_for_already_disclosed_fy() -> None:
    """Hand-built three-row fixture: FY t original at D1, FY t comparative
    at D2 > D1, FY t+1 original at D3 > D2. For an as_of strictly between
    D1 and D2, FY t was already disclosed at D1 and the next undisclosed
    FY is t+1. The naive accepted_date-ordering implementation would return
    FY t at D2 (the comparative); the CTE-based fix returns FY t+1 at D3.

    Exercises the bug path deterministically regardless of how the F1 grid
    lands on the e2e fixture's real filing dates.
    """
    conn = duckdb.connect(":memory:")
    try:
        conn.execute(
            'CREATE TABLE "income_statement" ('
            "security_id VARCHAR, fiscal_year INTEGER, period VARCHAR, "
            "accepted_date DATE, end_date DATE, eps_diluted DOUBLE)"
        )
        sid = "00000000-0000-0000-0000-000000000001"
        D1 = dt.date(2023, 11, 3)
        D2 = dt.date(2024, 11, 1)
        D3 = dt.date(2024, 11, 1)
        conn.executemany(
            'INSERT INTO "income_statement" VALUES (?, ?, ?, ?, ?, ?)',
            [
                (sid, 2023, "FY", D1, dt.date(2023, 9, 30), 6.13),
                (sid, 2023, "FY", D2, dt.date(2023, 9, 30), 6.13),
                (sid, 2024, "FY", D3, dt.date(2024, 9, 28), 6.72),
            ],
        )
        result = next_fy_target(
            conn=conn,
            security_id=uuid.UUID(sid),
            as_of_date=dt.date(2024, 5, 15),
            metric="eps_diluted",
        )
        assert result is not None
        assert result.fiscal_year == 2024
        assert result.accepted_date == D3
        assert result.value == 6.72
    finally:
        conn.close()
