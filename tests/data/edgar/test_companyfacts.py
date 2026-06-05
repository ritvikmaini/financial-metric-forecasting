"""Company-facts loader tests.

Loads SEC company-facts JSON and emits a flat list of Fact tuples.
Each XBRL fact instance is its own Fact, including restatements (same
end date, different filed date). This anchors L2 (keep every version).

Fact.start is critical for the L3 duration filter: it lets normalize
distinguish a discrete Q2 (~91d) from a YTD Q2 (~181d).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from fmf.data.edgar._http import EdgarClient
from fmf.data.edgar.companyfacts import load_facts

REPO_ROOT = Path(__file__).parent.parent.parent.parent
SAMPLES_DIR = REPO_ROOT / "tests" / "fixtures" / "sample_filings"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> EdgarClient:
    monkeypatch.setenv("SEC_USER_AGENT", "Test Test test@example.com")
    return EdgarClient(base_url=f"file://{SAMPLES_DIR}", max_rps=1000.0)


def test_load_facts_returns_a_fact_per_xbrl_entry(client: EdgarClient) -> None:
    facts = load_facts(client, cik="0000320193")
    assert facts, "no facts loaded"
    f = facts[0]
    assert f.concept
    assert isinstance(f.end, dt.date)
    assert isinstance(f.filed, dt.date)
    assert f.unit


def test_load_facts_keeps_restatements_as_separate_entries(client: EdgarClient) -> None:
    """L2 anchor: a concept with two filings for the same period end
    appears as two Fact entries with different `filed` dates.
    """
    facts = load_facts(client, cik="0000320193")
    by_key: dict[tuple[str, dt.date], set[dt.date]] = {}
    for f in facts:
        by_key.setdefault((f.concept, f.end), set()).add(f.filed)
    duplicates = {k: v for k, v in by_key.items() if len(v) > 1}
    assert duplicates, (
        "sample data must include at least one restatement; "
        "rebuild AAPL_companyfacts_min.json per Task 1 step 4 instructions."
    )


def test_load_facts_carries_form_fp_unit(client: EdgarClient) -> None:
    facts = load_facts(client, cik="0000320193")
    f = next(f for f in facts if "Revenue" in f.concept)
    assert f.form in {"10-K", "10-K/A", "10-Q", "10-Q/A"}
    assert f.fp in {"FY", "Q1", "Q2", "Q3", "Q4"}
    assert f.unit == "USD"


def test_load_facts_captures_start_for_duration_concepts(client: EdgarClient) -> None:
    """Flow-statement concepts (revenue, net income, cashflows, EPS) emit
    a `start` date in EDGAR companyfacts; the normalize duration filter
    needs it to distinguish a discrete Q2 (start=Apr 1, end=Jun 30, ~91d)
    from a YTD Q2 (start=Jan 1, end=Jun 30, ~181d). If load_facts drops
    `start`, the disambiguation is impossible and Q4 derivation is corrupted.
    """
    facts = load_facts(client, cik="0000320193")
    flow = next(f for f in facts if "Revenue" in f.concept and f.fp in {"Q2", "Q3"})
    assert flow.start is not None, "flow-concept facts must carry start; load_facts dropped it"
    duration_days = (flow.end - flow.start).days + 1
    assert duration_days > 0


def test_load_facts_leaves_start_none_for_instant_concepts(client: EdgarClient) -> None:
    """Balance-sheet (instant) concepts have no `start`."""
    facts = load_facts(client, cik="0000320193")
    instant_candidates = [f for f in facts if f.concept == "Assets"]
    if not instant_candidates:
        pytest.skip("sample data does not include Assets; skip")
    assert instant_candidates[0].start is None
