"""Submissions parser tests.

The submissions parser reads SEC's per-CIK submissions JSON
(https://data.sec.gov/submissions/CIK{cik}.json) and emits a filtered
list of Filing tuples. Filter: form in {"10-K", "10-K/A", "10-Q", "10-Q/A"}.
The /A suffix is amendments — we keep them because they bring different
filed dates and the keep-every-version rule (L2) depends on it.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from fmf.data.edgar._http import EdgarClient
from fmf.data.edgar.submissions import list_filings

REPO_ROOT = Path(__file__).parent.parent.parent.parent
SAMPLES_DIR = REPO_ROOT / "tests" / "fixtures" / "sample_filings"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> EdgarClient:
    monkeypatch.setenv("SEC_USER_AGENT", "Test Test test@example.com")
    return EdgarClient(base_url=f"file://{SAMPLES_DIR}", max_rps=1000.0)


def test_list_filings_returns_only_10k_and_10q(client: EdgarClient) -> None:
    filings = list_filings(client, cik="0000320193")
    forms = {f.form for f in filings}
    assert forms <= {"10-K", "10-K/A", "10-Q", "10-Q/A"}, f"unexpected forms: {forms}"


def test_list_filings_carries_filing_date_as_date_object(client: EdgarClient) -> None:
    filings = list_filings(client, cik="0000320193")
    assert filings
    assert isinstance(filings[0].filing_date, dt.date)


def test_list_filings_carries_accession(client: EdgarClient) -> None:
    filings = list_filings(client, cik="0000320193")
    assert all(f.accession for f in filings)
