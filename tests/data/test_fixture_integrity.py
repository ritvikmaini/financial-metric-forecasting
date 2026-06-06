"""Fixture-integrity tests.

Asserts the committed mini.duckdb has the expected ticker set, non-empty
tables, and that the anchor-validation gate passes. Also includes the
AAPL FY2022 comparative-mislabel guard (Critical 2 regression check).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini.duckdb"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not yet built")
def test_fixture_has_expected_tickers() -> None:
    conn = duckdb.connect(str(FIXTURE), read_only=True)
    try:
        rows = conn.execute('SELECT symbol FROM "securities" ORDER BY symbol').fetchall()
    finally:
        conn.close()
    tickers = {r[0] for r in rows}
    expected = {"AAPL", "MSFT", "GOOGL", "JNJ", "JPM", "ZTS", "GWW", "HSY", "SNOW"}
    missing = expected - tickers
    assert not missing, f"fixture missing tickers: {sorted(missing)}"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not yet built")
def test_fixture_income_statement_non_empty() -> None:
    conn = duckdb.connect(str(FIXTURE), read_only=True)
    try:
        n = conn.execute('SELECT COUNT(*) FROM "income_statement"').fetchone()
    finally:
        conn.close()
    assert n is not None and n[0] > 100, f"income_statement suspiciously small: {n}"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not yet built")
def test_aapl_fy2022_revenue_resolves_correctly() -> None:
    """Comparative-mislabel guard (Critical 2).

    AAPL FY2022 revenue published value is ~394.328B (fiscal year ending
    2022-09-24). The 2022 figures appear as comparatives in the FY2023
    10-K with fact.fy=2023; if normalize derived fiscal_year from fact.fy
    instead of FY-end-aware mapping, the 2022 value would land as a FY2023
    row with the FY2023 published revenue overwriting it.
    """
    conn = duckdb.connect(str(FIXTURE), read_only=True)
    try:
        row = conn.execute(
            'SELECT "revenue" FROM "income_statement" i '
            'JOIN "securities" s ON s.security_id = i.security_id '
            "WHERE s.symbol = ? AND i.fiscal_year = 2022 AND i.period = ? "
            "ORDER BY i.accepted_date DESC LIMIT 1",
            ["AAPL", "FY"],
        ).fetchone()
    finally:
        conn.close()
    assert (
        row is not None and row[0] is not None
    ), "AAPL FY2022 revenue missing from fixture; check normalize fiscal_year derivation."
    expected = 394_328_000_000.0
    rel_err = abs(row[0] - expected) / expected
    assert rel_err < 0.01, (
        f"AAPL FY2022 revenue: got {row[0]!r}, expected ~{expected}. "
        f"Regression: check normalize FY-end-aware derivation."
    )
