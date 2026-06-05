"""Submissions index parser.

Reads SEC's per-CIK submissions JSON and emits a filtered list of
Filing tuples (form, accession, filing_date) for the forms we care about.
Filter: 10-K, 10-K/A, 10-Q, 10-Q/A. Amendments are kept — they bring
new filed dates and the keep-every-version rule (L2) needs them.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from fmf.data.edgar._http import EdgarClient

_INTERESTING_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A"})


@dataclass(frozen=True, slots=True)
class Filing:
    form: str
    accession: str
    filing_date: dt.date


def list_filings(client: EdgarClient, *, cik: str) -> list[Filing]:
    """Return the filtered list of 10-K and 10-Q filings for cik.

    cik is the 10-digit zero-padded CIK (e.g. "0000320193").
    """
    data = client.get_json(f"/submissions/CIK{cik}.json")
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])

    out: list[Filing] = []
    for form, acc, dstr in zip(forms, accessions, dates, strict=True):
        if form not in _INTERESTING_FORMS:
            continue
        out.append(
            Filing(
                form=form,
                accession=acc,
                filing_date=dt.date.fromisoformat(dstr),
            )
        )
    return out
