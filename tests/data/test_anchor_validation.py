"""Anchor-validation gate test.

Runs the same logic against the committed fixture that build_fixture.py
runs against a fresh ingest. If concept-map resolution drifts, this test
fails before the developer notices in a downstream model.

The committed-fixture test is SKIPPED until the fixture exists (T9 produces it).
The loader / schema tests run unconditionally.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from fmf.data.edgar.validation import (
    load_known_financials,
    validate_anchors,
)

REPO_ROOT = Path(__file__).parent.parent.parent
KNOWN = REPO_ROOT / "tests" / "fixtures" / "known_financials.json"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini.duckdb"


def test_load_known_financials_parses_5_anchors() -> None:
    truth = load_known_financials(KNOWN)
    assert len(truth.anchors) == 5
    tickers = {a.ticker for a in truth.anchors}
    assert tickers == {"AAPL", "MSFT", "GOOGL", "JNJ", "JPM"}


def test_load_known_financials_rejects_null_field_without_skip_reason(
    tmp_path: Path,
) -> None:
    """A null field without a documented _skip_reason is a config bug;
    load must fail loudly so a reviewer always sees why a field was
    excluded from validation.
    """
    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"tolerance":{"revenue_usd_pct":0.005,"net_income_usd_pct":0.005,"eps_diluted_pct":0.005},'
        '"anchors":[{"ticker":"X","cik":"0","fiscal_year":2023,"fp":"FY",'
        '"revenue_usd":null,"net_income_usd":1,"eps_diluted":1}]}'
    )
    with pytest.raises(ValueError, match="revenue_skip_reason"):
        load_known_financials(bad)


def test_load_known_financials_accepts_null_field_with_skip_reason(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good.json"
    good.write_text(
        '{"tolerance":{"revenue_usd_pct":0.005,"net_income_usd_pct":0.005,"eps_diluted_pct":0.005},'
        '"anchors":[{"ticker":"X","cik":"0","fiscal_year":2023,"fp":"FY",'
        '"revenue_usd":null,"revenue_skip_reason":"test reason",'
        '"net_income_usd":1,"eps_diluted":1}]}'
    )
    truth = load_known_financials(good)
    assert truth.anchors[0].revenue_usd is None
    assert truth.anchors[0].net_income_usd == 1.0


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture not built yet (lands in T9)")
def test_committed_fixture_passes_anchor_validation() -> None:
    truth = load_known_financials(KNOWN)
    conn = duckdb.connect(str(FIXTURE), read_only=True)
    try:
        validate_anchors(conn, truth)
    finally:
        conn.close()
