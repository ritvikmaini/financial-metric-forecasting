"""sources.py tests.

Per (ticker, fiscal_year, field), which us-gaap concept resolved the
value? Useful for investigating coverage drops.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fmf.features.audit.sources import audit_field_sources

REPO_ROOT = Path(__file__).parent.parent.parent.parent


@pytest.fixture
def sec_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "Test Test test@example.com")


def test_audit_field_sources_reports_per_ticker(sec_user_agent: None) -> None:
    """The audit re-resolves the field via the concept map against fresh
    EDGAR data (using the file:// sample to avoid live calls) and reports
    which concept matched for each ticker."""
    samples = REPO_ROOT / "tests" / "fixtures" / "sample_filings"
    df = audit_field_sources(
        base_url=f"file://{samples}",
        tickers=[("AAPL", "0000320193")],
        field="revenue",
        fy=2023,
        fp="FY",
        max_rps=1000.0,
    )
    assert set(df.columns) >= {"ticker", "field", "resolved_concept", "value"}
    assert len(df) == 1
