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
    assert row is not None and row[0] is not None, (
        "AAPL FY2022 revenue missing from fixture; check normalize fiscal_year derivation."
    )
    expected = 394_328_000_000.0
    rel_err = abs(row[0] - expected) / expected
    assert rel_err < 0.01, (
        f"AAPL FY2022 revenue: got {row[0]!r}, expected ~{expected}. "
        f"Regression: check normalize FY-end-aware derivation."
    )


def test_aapl_pre_2014_close_reflects_multi_split_un_apply() -> None:
    """AAPL had two splits in the fixture window: 7:1 on 2014-06-09 and
    4:1 on 2020-08-31. Pre-2014 rows must be un-applied by BOTH (cumulative
    factor 28). The existing 2019-06-03 test only exercises the 4:1; this
    test locks the multi-split compounding by asserting the 2010-12-31
    close lands in the historical few-hundreds range.

    Reference: AAPL's actual unadjusted close on 2010-12-31 was ~$322.56.
    Post-both-splits, back-adjusted is ~$11.52. The un-split transform
    should put close back near the unadjusted value.
    """
    import datetime as dt

    conn = duckdb.connect(str(FIXTURE), read_only=True)
    try:
        row = conn.execute(
            'SELECT close, adj_close FROM "prices" p '
            'JOIN "securities" s ON s.security_id = p.security_id '
            'WHERE s.symbol = ? AND "date" = ?',
            ["AAPL", dt.date(2010, 12, 31)],
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, "AAPL 2010-12-31 missing from fixture"
    close, adj_close = row
    # Reasonable bounds for the actual unadjusted price.
    assert close > 250, (
        f"AAPL 2010-12-31 close={close} too low. Multi-split compounding "
        f"may have failed: 2014 7:1 + 2020 4:1 should give a x28 cumulative "
        f"factor. Post-both-splits back-adjusted is ~$11."
    )
    assert close < 400, (
        f"AAPL 2010-12-31 close={close} too high. Verify against historical "
        f"reference (actual ~$322.56)."
    )
    # Sanity: close > 20× adj_close (since ratio after both splits should
    # be close to 28).
    assert close > adj_close * 20, (
        f"close={close} adj_close={adj_close} ratio={close / adj_close:.2f} "
        f"— expected ratio near 28 for pre-2014 dates"
    )
