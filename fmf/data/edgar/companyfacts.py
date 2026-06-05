"""Company-facts loader.

Reads SEC's per-CIK company-facts JSON and flattens it into a list of
Fact tuples. Each XBRL fact instance — including restatements with the
same (concept, end) but different filed dates — is its own Fact. This
anchors L2 (keep every version).

`start` is present for duration concepts (flow-statement items: revenue,
net income, cashflows, EPS) and absent for instant concepts (balance-sheet
items). The duration filter in normalize._collect_per_period uses
(end - start) to disambiguate discrete-quarter facts from YTD twins.

We only look at the us-gaap taxonomy for now. Custom company taxonomies
(ifrs-full, dei, company-specific extensions) are out of scope; this
covers the fields the anchor-validation gate cares about.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from fmf.data.edgar._http import EdgarClient


@dataclass(frozen=True, slots=True)
class Fact:
    """A single XBRL fact instance.

    `start` is present for duration concepts and absent for instant
    concepts. For Q2/Q3/FY flow concepts, EDGAR emits BOTH the discrete-
    period fact (start=quarter-start) AND the year-to-date cumulative
    fact (start=fiscal-year-start) with the same end, fp, form, and
    filed date. The duration filter uses (end - start) to disambiguate.
    """

    concept: str  # e.g. "Revenues", "NetIncomeLoss"
    end: dt.date  # fiscal-period end
    filed: dt.date  # when this fact was filed; becomes accepted_date
    value: float  # the fact value, in the unit below
    unit: str  # e.g. "USD", "USD/shares", "shares"
    form: str  # "10-K", "10-Q", "10-K/A", "10-Q/A"
    fp: str  # "FY", "Q1", "Q2", "Q3", "Q4", "CY"
    fy: int  # fiscal year (FILING's frame; not the data's FY)
    start: dt.date | None = None  # present for flow concepts only


def load_facts(client: EdgarClient, *, cik: str) -> list[Fact]:
    """Return all us-gaap facts for cik as a flat Fact list."""
    data = client.get_json(f"/api/xbrl/companyfacts/CIK{cik}.json")
    us_gaap = data.get("facts", {}).get("us-gaap", {})
    out: list[Fact] = []
    for concept, payload in us_gaap.items():
        units = payload.get("units", {})
        for unit, entries in units.items():
            for e in entries:
                try:
                    raw_start = e.get("start")
                    start = dt.date.fromisoformat(raw_start) if raw_start else None
                    out.append(
                        Fact(
                            concept=concept,
                            end=dt.date.fromisoformat(e["end"]),
                            filed=dt.date.fromisoformat(e["filed"]),
                            value=float(e["val"]),
                            unit=unit,
                            form=e.get("form", ""),
                            fp=e.get("fp", ""),
                            fy=int(e.get("fy", 0)),
                            start=start,
                        )
                    )
                except (KeyError, ValueError, TypeError):
                    # Skip malformed entries silently; partial-coverage
                    # data is the SEC norm and audit/coverage.py will surface it.
                    continue
    return out
