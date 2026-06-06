"""Derived feature tests — TTM, YoY, growth.

Critical: each derivation must assemble components PIT-correctly.
A TTM as of date X sums the 4 most recent QUARTERS by end_date AMONG
ONLY those quarters whose accepted_date <= X. A recent quarter that
hasn't been filed yet must be excluded.
"""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import duckdb
import pytest

from fmf.features.derived import (
    compute_gross_margin_latest,
    compute_revenue_ttm,
    compute_revenue_yoy_growth,
)

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini.duckdb"


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    if not FIXTURE.exists():
        pytest.skip("fixture not built yet")
    return duckdb.connect(str(FIXTURE), read_only=True)


@pytest.fixture
def aapl_security_id(conn: duckdb.DuckDBPyConnection) -> uuid.UUID:
    row = conn.execute('SELECT security_id FROM "securities" WHERE symbol = ?', ["AAPL"]).fetchone()
    assert row is not None
    return uuid.UUID(str(row[0]))


def test_revenue_ttm_at_known_filing_date(
    conn: duckdb.DuckDBPyConnection,
    aapl_security_id: uuid.UUID,
) -> None:
    """At AAPL's FY2023 10-K filing date (2023-11-03), TTM should equal
    the FY2023 annual revenue (~$383.285B) because that's the most
    recent annual figure available — or the sum of last 4 quarters if
    we prefer that. The implementation prefers the latest annual.
    """
    fy23_accepted = conn.execute(
        'SELECT accepted_date FROM "income_statement" '
        "WHERE security_id = ? AND fiscal_year = 2023 AND period = ? "
        "ORDER BY accepted_date ASC LIMIT 1",
        [str(aapl_security_id), "FY"],
    ).fetchone()[0]
    ttm = compute_revenue_ttm(
        conn=conn,
        security_id=aapl_security_id,
        as_of_date=fy23_accepted,
    )
    assert ttm is not None
    # Within 1% of the FY2023 published value.
    assert abs(ttm - 383_285_000_000) / 383_285_000_000 < 0.01, (
        f"TTM revenue at FY2023 filing: got {ttm}, expected ~383.285B"
    )


def test_revenue_ttm_excludes_unfiled_quarter(
    conn: duckdb.DuckDBPyConnection,
    aapl_security_id: uuid.UUID,
) -> None:
    """The L3 trap: as_of between two filings must exclude the unfiled one.

    Pick an as_of_date AFTER the latest annual was filed but BEFORE the
    next quarter's 10-Q was filed. The TTM must NOT include the unfiled
    quarter — even though it's the most recent by end_date, it isn't
    PIT-visible.
    """
    # Find two consecutive AAPL filings.
    rows = conn.execute(
        'SELECT DISTINCT accepted_date FROM "income_statement" '
        "WHERE security_id = ? AND accepted_date > ? "
        "ORDER BY accepted_date ASC LIMIT 2",
        [str(aapl_security_id), dt.date(2023, 1, 1)],
    ).fetchall()
    if len(rows) < 2:
        pytest.skip("need at least 2 AAPL filings post-2023-01-01")
    earlier, later = rows[0][0], rows[1][0]
    between = earlier + dt.timedelta(days=(later - earlier).days // 2)
    # TTM as_of between earlier and later: must not include any row
    # whose accepted_date > between.
    ttm = compute_revenue_ttm(
        conn=conn,
        security_id=aapl_security_id,
        as_of_date=between,
    )
    # Either: TTM is None (insufficient visible quarters) or it's defined
    # but does NOT exceed the latest visible annual + the latest visible
    # quarter (defensive bound).
    if ttm is not None:
        latest_visible_annual = conn.execute(
            'SELECT revenue FROM "income_statement" '
            "WHERE security_id = ? AND accepted_date <= ? AND period = ? "
            "AND revenue IS NOT NULL "
            "ORDER BY end_date DESC LIMIT 1",
            [str(aapl_security_id), between, "FY"],
        ).fetchone()
        if latest_visible_annual is not None:
            # TTM should be plausibly close to the latest annual, not
            # significantly larger (which would indicate leakage from
            # an unfiled quarter).
            assert ttm <= latest_visible_annual[0] * 1.3, (
                f"TTM={ttm} at as_of={between} exceeds 1.3x latest visible "
                f"annual {latest_visible_annual[0]}. Likely PIT leak from "
                f"an unfiled quarter."
            )


def test_revenue_ttm_quarter_path_excludes_unfiled(
    conn: duckdb.DuckDBPyConnection,
    aapl_security_id: uuid.UUID,
) -> None:
    """Force the quarter-summing TTM path on real consecutive quarters.

    The annual branch of compute_revenue_ttm short-circuits when an FY
    end_date is within 366d of as_of, so the existing
    `test_revenue_ttm_excludes_unfiled_quarter` only exercises that
    branch. This test picks an as_of where the latest visible annual is
    OLDER than 366d AND the surrounding quarterly history is dense, so
    the quarter-summing fallback runs and sums real consecutive
    quarters. This is the one L3 face the other three tests don't
    cover: JPM verifies the recency guard returns None on stale
    windows, the synthetic test verifies the span check rejects gapped
    windows, the AAPL annual-path test verifies the within-366d
    branch — only this test confirms the quarter path produces the
    correct TTM from real consecutive quarters when no fresh annual
    exists.

    Target the dense-quarterly region where Q4 is contemporaneously
    derived from the FY filing (FY end_date >= 2022-01-01). Pre-2022
    AAPL fixture data only surfaces Q4 via the NEXT year's comparative
    period (accepted_date = the next year's 10-K), which leaves a Q4
    gap at this test's chosen as_of and trips the span check. The
    skip guards below remain as a backstop for future fixture
    rebuilds that thin that region.
    """
    # Earliest AAPL FY with end_date >= 2022-01-01, jumping 367d past it
    # to land in the next year where that annual is still the latest
    # visible (the next FY hasn't been filed yet) and quarters are dense.
    annuals = conn.execute(
        'SELECT end_date FROM "income_statement" '
        "WHERE security_id = ? AND period = ? AND revenue IS NOT NULL "
        "AND end_date >= ? "
        "ORDER BY end_date ASC",
        [str(aapl_security_id), "FY", dt.date(2022, 1, 1)],
    ).fetchall()
    if len(annuals) < 1:
        pytest.skip("need at least 1 AAPL FY end_date >= 2022-01-01 in fixture")
    # Normalize via pd.Timestamp(_).date() unconditionally — see comment
    # in test_prices_pit_excludes_dates_after_as_of for why a guarded
    # isinstance(dt.date) skip is unsafe (Timestamp is a dt.date subclass).
    import pandas as pd

    target_annual_end = pd.Timestamp(annuals[0][0]).date()
    as_of = target_annual_end + dt.timedelta(days=367)

    # Ground truth: sum the four most recent quarters visible at as_of.
    visible_q = conn.execute(
        'SELECT end_date, revenue FROM "income_statement" '
        "WHERE security_id = ? AND period IN (?, ?, ?, ?) "
        "AND accepted_date <= ? AND revenue IS NOT NULL "
        "ORDER BY end_date DESC LIMIT 4",
        [str(aapl_security_id), "Q1", "Q2", "Q3", "Q4", as_of],
    ).fetchall()
    if len(visible_q) < 4:
        pytest.skip("not enough visible quarters at chosen as_of")
    expected = sum(r[1] for r in visible_q)

    ttm = compute_revenue_ttm(
        conn=conn,
        security_id=aapl_security_id,
        as_of_date=as_of,
    )
    assert ttm is not None, (
        f"quarter-path TTM should be defined at as_of={as_of} with "
        f"{len(visible_q)} visible quarters"
    )
    assert abs(ttm - expected) / expected < 0.001, (
        f"quarter-path TTM={ttm} != independently-summed visible quarters {expected}"
    )

    # Confirm the implementation's selected quarters are consecutive
    # (~270d end_date-to-end_date span). If a null quarter would have
    # forced a non-consecutive selection, the implementation must return
    # None instead of a wrong sum — see the next test.
    dates = sorted(pd.Timestamp(r[0]).date() for r in visible_q)
    span = (dates[-1] - dates[0]).days
    assert 200 <= span <= 320, f"ground-truth 4 quarters span {span}d, outside consecutive window"


def test_revenue_ttm_recency_guard_rejects_stale_quarterly_window(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """When the annual branch is stale AND the latest visible quarter is
    also stale, the recency guard must reject — no fabricated TTM from
    archival quarterly history.

    JPM is the natural ticker: L-INFRA-003 documents that JPM's quarterly
    Revenues column is null FY2018+. BUT the fixture DOES contain 51
    non-null quarterly Revenues rows for JPM FY2009-FY2014. Without a
    recency guard on the latest selected quarter, an as_of in 2027 finds
    a clean consecutive 2013 Q1-Q4 window (275d span, passes the
    consecutiveness check) and silently returns a 13-year-stale TTM.
    The recency guard requires the latest selected quarter's end_date
    be within ~366d of as_of, mirroring the annual branch's tolerance.

    Test compute as_of from the fixture so it stays correct across
    fixture rebuilds.
    """
    row = conn.execute('SELECT security_id FROM "securities" WHERE symbol = ?', ["JPM"]).fetchone()
    if row is None:
        pytest.skip("JPM missing from fixture")
    jpm_sid = uuid.UUID(str(row[0]))

    latest_fy_end_row = conn.execute(
        'SELECT MAX(end_date) FROM "income_statement" '
        "WHERE security_id = ? AND period = ? AND revenue IS NOT NULL",
        [str(jpm_sid), "FY"],
    ).fetchone()
    if latest_fy_end_row is None or latest_fy_end_row[0] is None:
        pytest.skip("no JPM FY annuals in fixture")
    # Normalize unconditionally; see test_prices_pit_excludes_dates_after_as_of.
    import pandas as pd

    latest_fy_end = pd.Timestamp(latest_fy_end_row[0]).date()
    # 400d puts as_of safely past the 366d annual-branch window for
    # the latest visible JPM annual; the fixture has no later JPM
    # annual, so the annual branch is forced to fail.
    as_of = latest_fy_end + dt.timedelta(days=400)

    ttm = compute_revenue_ttm(
        conn=conn,
        security_id=jpm_sid,
        as_of_date=as_of,
    )
    # Annual branch stale (>366d). Quarter branch finds JPM 2013
    # Q1-Q4 (last 4 non-null by end_date, span ~275d, consecutive).
    # Recency guard then rejects because 2013-12-31 is ~13y before
    # as_of. Returns None. No silent fabrication.
    assert ttm is None, (
        f"JPM at as_of={as_of} (latest FY {latest_fy_end} is "
        f"{(as_of - latest_fy_end).days}d stale): expected None, got {ttm}. "
        f"Either the annual branch leaked a stale value or the quarter "
        f"path returned a stale TTM (recency guard failed)."
    )


def test_revenue_ttm_rejects_nonconsecutive_quarter_span() -> None:
    """The span-check rejection branch — quarter-summing path drops a
    selection whose end_dates span >320d (non-consecutive), returning
    None instead of a 15-month-span fabrication.

    No fixture ticker produces a non-consecutive 4-quarter selection
    organically (the 9 tickers have clean consecutive data), so this
    is a constructed in-memory DuckDB test on a hand-built series.
    The real-fixture rule governs PIT EXTRACTION tests, not pure-
    arithmetic guards on the derived layer.
    """
    import duckdb as _duckdb

    schema_sql = (Path(__file__).parent.parent.parent / "fmf" / "data" / "schema.sql").read_text()
    sid = uuid.uuid4()

    mem = _duckdb.connect(":memory:")
    try:
        mem.execute(schema_sql)
        mem.execute(
            'INSERT INTO "securities" (security_id, symbol, cik) VALUES (?, ?, ?)',
            [str(sid), "TEST", "0000000001"],
        )
        # Four quarter rows but DROP the intervening quarter so the
        # 4-row sort_values+head(4) selection spans >320d.
        # Q1 2022 ends 2022-03-31, Q2 2022 ends 2022-06-30, Q3 2022
        # ends 2022-09-30, then we jump to Q3 2023 ending 2023-09-30.
        # The 4 selected by end_date desc are Q3-2023, Q3-2022, Q2-2022,
        # Q1-2022 → earliest 2022-03-31, latest 2023-09-30, span 548d.
        rows = [
            (2022, "Q1", dt.date(2022, 5, 1), dt.date(2022, 5, 1), dt.date(2022, 3, 31), 100.0),
            (2022, "Q2", dt.date(2022, 8, 1), dt.date(2022, 8, 1), dt.date(2022, 6, 30), 110.0),
            (2022, "Q3", dt.date(2022, 11, 1), dt.date(2022, 11, 1), dt.date(2022, 9, 30), 120.0),
            (2023, "Q3", dt.date(2023, 11, 1), dt.date(2023, 11, 1), dt.date(2023, 9, 30), 130.0),
        ]
        for fy, period, filing, accepted, end, rev in rows:
            mem.execute(
                'INSERT INTO "income_statement" '
                "(security_id, fiscal_year, period, filing_date, accepted_date, "
                "end_date, revenue) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [str(sid), fy, period, filing, accepted, end, rev],
            )

        ttm = compute_revenue_ttm(
            conn=mem,
            security_id=sid,
            as_of_date=dt.date(2024, 1, 1),
        )
        assert ttm is None, (
            f"span-check rejection failed: 4 quarters spanning 548d should return None, got {ttm}"
        )
    finally:
        mem.close()


def test_revenue_yoy_growth_uses_pit_components(
    conn: duckdb.DuckDBPyConnection,
    aapl_security_id: uuid.UUID,
) -> None:
    """YoY growth = (TTM_current - TTM_year_ago) / TTM_year_ago.
    Both TTMs must be PIT-correct AT THE SAME as_of_date.
    """
    fy23_accepted = conn.execute(
        'SELECT accepted_date FROM "income_statement" '
        "WHERE security_id = ? AND fiscal_year = 2023 AND period = ? "
        "LIMIT 1",
        [str(aapl_security_id), "FY"],
    ).fetchone()[0]
    yoy = compute_revenue_yoy_growth(
        conn=conn,
        security_id=aapl_security_id,
        as_of_date=fy23_accepted,
    )
    # AAPL FY2023 was ~$383B vs FY2022 ~$394B → YoY -2.8%.
    assert yoy is not None
    assert -0.10 < yoy < 0.05, f"AAPL FY2023 YoY revenue growth: got {yoy:.3f}, expected ~-0.028"


def test_gross_margin_uses_latest_pit_visible_period(
    conn: duckdb.DuckDBPyConnection,
    aapl_security_id: uuid.UUID,
) -> None:
    """Gross margin = gross_profit / revenue at the latest PIT-visible
    period. Must not pick a future period."""
    historical = dt.date(2015, 6, 1)
    gm = compute_gross_margin_latest(
        conn=conn,
        security_id=aapl_security_id,
        as_of_date=historical,
    )
    if gm is not None:
        # AAPL gross margin in mid-2010s was 38-40%.
        assert 0.30 < gm < 0.50, (
            f"AAPL gross margin at as_of={historical}: got {gm}, expected ~0.40"
        )
