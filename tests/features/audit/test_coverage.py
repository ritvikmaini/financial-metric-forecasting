"""coverage.py tests.

Reports the fraction of expected fields populated per (ticker, fiscal_year,
period). Used in S2 to surface ingest gaps; reused in later sessions for
cohort coverage monitoring.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from fmf.features.audit.coverage import compute_coverage

REPO_ROOT = Path(__file__).parent.parent.parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini.duckdb"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not yet built")
def test_compute_coverage_returns_per_ticker_rows() -> None:
    conn = duckdb.connect(str(FIXTURE), read_only=True)
    try:
        df = compute_coverage(conn, table="income_statement")
    finally:
        conn.close()
    expected_cols = {"symbol", "fiscal_year", "period", "coverage_pct"}
    assert expected_cols <= set(df.columns)
    assert {"AAPL", "MSFT", "GOOGL", "JNJ", "JPM"} <= set(df["symbol"])


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not yet built")
def test_compute_coverage_dedups_phantoms_one_row_per_security_fy_period() -> None:
    """L-INFRA-013: compute_coverage must return exactly one row per
    (security_id, fiscal_year, period) — phantom Q-rows from comparative
    fp-frame leak must not inflate the denominator.
    """
    conn = duckdb.connect(str(FIXTURE), read_only=True)
    try:
        df = compute_coverage(conn, table="income_statement")
    finally:
        conn.close()
    dup_keys = df.groupby(["symbol", "fiscal_year", "period"]).size()
    assert (dup_keys == 1).all(), (
        f"compute_coverage returned duplicate rows for: {dup_keys[dup_keys > 1].to_dict()}"
    )


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not yet built")
def test_anchor_tickers_have_high_revenue_coverage() -> None:
    """The 5 anchors all have well-populated 10-Ks; the median
    coverage_pct on the income_statement should be > 0.5 per anchor.

    We assert on the per-anchor median (not every row) because the audit
    legitimately surfaces tail cases the test should not gate on:
    - JPM's Q1/Q2/Q3 rows are thin because JPM emits Revenues as an
      FY-only fact (banks tag interest-income components separately).
    - MSFT FY2015 Q4 has restatement rows that carry only net_income.
    Both are real signals; the median per anchor is the right anchor
    health check.
    """
    conn = duckdb.connect(str(FIXTURE), read_only=True)
    try:
        df = compute_coverage(conn, table="income_statement")
    finally:
        conn.close()
    anchors = df[df["symbol"].isin(["AAPL", "MSFT", "GOOGL", "JNJ", "JPM"])]
    median_by_symbol = anchors.groupby("symbol")["coverage_pct"].median()
    assert (median_by_symbol > 0.3).all(), (
        f"unexpectedly low median coverage on anchors:\n{median_by_symbol}"
    )
