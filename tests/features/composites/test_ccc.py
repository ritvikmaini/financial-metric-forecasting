"""CCC hand-calc tests on synthetic in-memory DuckDB rows.

The v1.0 schema does not carry inventory / receivables / payables; the
hand-calc test extends the in-memory schema with those columns so the
formula is exercised explicitly. IDEA-S18-003 covers lighting them up
on real EDGAR data.
"""

from __future__ import annotations

import datetime as dt
import math
import uuid
from pathlib import Path

import duckdb

from fmf.features.composites._ccc import compute_ccc_days

SCHEMA = (Path(__file__).parent.parent.parent.parent / "fmf" / "data" / "schema.sql").read_text()


def _mem_with_wc_cols() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    c.execute(SCHEMA)
    c.execute('ALTER TABLE "balance_sheet" ADD COLUMN inventory DOUBLE')
    c.execute('ALTER TABLE "balance_sheet" ADD COLUMN receivables DOUBLE')
    c.execute('ALTER TABLE "balance_sheet" ADD COLUMN payables DOUBLE')
    return c


def _insert_security(conn: duckdb.DuckDBPyConnection, sid: uuid.UUID) -> None:
    conn.execute(
        'INSERT INTO "securities" (security_id, symbol, cik) VALUES (?, ?, ?)',
        [str(sid), "TEST", "0000000001"],
    )


def test_ccc_hand_calc_on_synthetic_rows() -> None:
    """revenue=1000, gross_profit=400 -> COGS=600.

    DIO = 100 / (600/365) = 100*365/600 = 60.83333...
    DSO = 200 / (1000/365) = 200*365/1000 = 73.0
    DPO = 50 / (600/365) = 50*365/600 = 30.41666...
    CCC = DIO + DSO - DPO = 103.41666...
    """
    conn = _mem_with_wc_cols()
    try:
        sid = uuid.uuid4()
        _insert_security(conn, sid)
        conn.execute(
            'INSERT INTO "income_statement" '
            "(security_id, fiscal_year, period, filing_date, accepted_date, end_date, "
            "revenue, gross_profit) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                str(sid),
                2022,
                "FY",
                dt.date(2023, 2, 1),
                dt.date(2023, 2, 1),
                dt.date(2022, 12, 31),
                1000.0,
                400.0,
            ],
        )
        conn.execute(
            'INSERT INTO "balance_sheet" '
            "(security_id, fiscal_year, period, filing_date, accepted_date, end_date, "
            "inventory, receivables, payables) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                str(sid),
                2022,
                "FY",
                dt.date(2023, 2, 1),
                dt.date(2023, 2, 1),
                dt.date(2022, 12, 31),
                100.0,
                200.0,
                50.0,
            ],
        )

        ccc = compute_ccc_days(conn=conn, security_id=sid, as_of_date=dt.date(2023, 6, 1))
        assert ccc is not None
        expected = 100 * 365 / 600 + 200 * 365 / 1000 - 50 * 365 / 600
        assert math.isclose(ccc, expected, rel_tol=1e-9)
    finally:
        conn.close()


def test_ccc_none_when_inventory_missing() -> None:
    """receivables and payables are present but inventory is null -> None."""
    conn = _mem_with_wc_cols()
    try:
        sid = uuid.uuid4()
        _insert_security(conn, sid)
        conn.execute(
            'INSERT INTO "income_statement" '
            "(security_id, fiscal_year, period, filing_date, accepted_date, end_date, "
            "revenue, gross_profit) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                str(sid),
                2022,
                "FY",
                dt.date(2023, 2, 1),
                dt.date(2023, 2, 1),
                dt.date(2022, 12, 31),
                1000.0,
                400.0,
            ],
        )
        conn.execute(
            'INSERT INTO "balance_sheet" '
            "(security_id, fiscal_year, period, filing_date, accepted_date, end_date, "
            "receivables, payables) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                str(sid),
                2022,
                "FY",
                dt.date(2023, 2, 1),
                dt.date(2023, 2, 1),
                dt.date(2022, 12, 31),
                200.0,
                50.0,
            ],
        )

        ccc = compute_ccc_days(conn=conn, security_id=sid, as_of_date=dt.date(2023, 6, 1))
        assert ccc is None
    finally:
        conn.close()
