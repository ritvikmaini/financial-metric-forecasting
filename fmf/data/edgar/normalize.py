"""Normalize XBRL facts to schema rows.

Pipeline:
1. Build the company's fiscal calendar from the input facts (FY-end dates
   per fiscal year), extracted from annual facts (fp=FY, form=10-K[/A]).
   Fallbacks: infer from Q3 ends (Q3-end + ~91 days), else calendar-year FY.
2. Group facts by (fiscal_year, period, end, filed). For FLOW facts
   (start != None) period comes from form+fp via period_from_form_fp and
   fiscal_year is derived FY-end-aware ("next FY-end at or after end").
   For INSTANT facts (balance sheet; start == None) BOTH fiscal_year and
   period are derived from end_date matching the fiscal calendar — this
   keeps a prior-FY-end balance sheet labeled period=FY even when it
   appears as a comparative in a Q1 10-Q (whose `fp` is Q1).
3. For each group, walk concept_map per target field and pick the first
   match (priority + unit gate, L1+L4). For instant buckets, the resolver
   skips the fp filter (the fact's fp may not match the derived period).
4. Emit one row per (fiscal_year, period, filed) — restatements coexist (L2).
5. Derive Q4 per fiscal_year for flow-statement fields. For each FY filing,
   find the latest Q1/Q2/Q3 values whose accepted_date <= the FY filing's
   accepted_date, compute Q4 = FY - (Q1+Q2+Q3), and emit a Q4 row whose
   end_date = the FY filing's end_date (Q4 ends at FY-end).
6. Return NormalizedTables(income_statement, balance_sheet, cashflow).
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Final, cast

import pandas as pd

from fmf.data.edgar.companyfacts import Fact
from fmf.data.edgar.concept_map import CONCEPT_MAP, resolve_field

_INCOME_FIELDS: Final[tuple[str, ...]] = (
    "revenue",
    "gross_profit",
    "ebit",
    "net_income",
    "eps_diluted",
)
_BALANCE_FIELDS: Final[tuple[str, ...]] = (
    "total_assets",
    "total_liabilities",
    "total_equity",
    "cash_and_equivalents",
    "current_assets",
    "current_liabilities",
    "long_term_debt",
)
_CASHFLOW_FIELDS: Final[tuple[str, ...]] = (
    "operating_cash_flow",
    "investing_cash_flow",
    "financing_cash_flow",
    "capital_expenditure",
    "free_cash_flow",
)

# Flow-statement fields (FY = sum of quarters); used for Q4 derivation.
_FLOW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "revenue",
        "gross_profit",
        "ebit",
        "net_income",
        "operating_cash_flow",
        "investing_cash_flow",
        "financing_cash_flow",
        "capital_expenditure",
    }
)

# Duration windows in days, used to disambiguate discrete-quarter facts
# from YTD cumulative twins. EDGAR companyfacts emits BOTH for flow concepts
# in Q2/Q3 with identical (end, fp, form, filed); only the duration tells
# them apart.
_QUARTER_DAYS_MIN = 60
_QUARTER_DAYS_MAX = 100
_ANNUAL_DAYS_MIN = 340
_ANNUAL_DAYS_MAX = 380

# Tolerance for matching instant ends to fiscal-period boundaries.
# Accommodates 52/53-week fiscal calendars and minor calendar drift.
_QUARTER_DAYS_TOLERANCE: Final[int] = 15


@dataclass(frozen=True, slots=True)
class NormalizedTables:
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cashflow: pd.DataFrame


_FORM_FP_TO_PERIOD: Final[dict[tuple[str, str], str]] = {
    ("10-K", "FY"): "FY",
    ("10-K/A", "FY"): "FY",
    ("10-Q", "Q1"): "Q1",
    ("10-Q", "Q2"): "Q2",
    ("10-Q", "Q3"): "Q3",
    ("10-Q/A", "Q1"): "Q1",
    ("10-Q/A", "Q2"): "Q2",
    ("10-Q/A", "Q3"): "Q3",
}


def period_from_form_fp(*, form: str, fp: str) -> str:
    """Derive period (FY/Q1/Q2/Q3) from filing form and fiscal-period code."""
    period = _FORM_FP_TO_PERIOD.get((form, fp))
    if period is None:
        raise ValueError(f"unknown (form, fp) combination: ({form!r}, {fp!r})")
    return period


def _duration_matches_period(fact: Fact, period: str) -> bool:
    """Return True if the fact's duration matches the period.

    - Instant concepts (start is None) pass through; balance-sheet items
      have no duration and the duration filter does not apply.
    - Q1/Q2/Q3: must be the discrete ~91-day quarter, not a YTD cumulative.
    - FY: must be the ~365-day annual figure, not a partial.
    """
    if fact.start is None:
        return True
    days = (fact.end - fact.start).days + 1
    if period == "FY":
        return _ANNUAL_DAYS_MIN <= days <= _ANNUAL_DAYS_MAX
    if period in ("Q1", "Q2", "Q3"):
        return _QUARTER_DAYS_MIN <= days <= _QUARTER_DAYS_MAX
    return True


def _compute_fy_end_dates(facts: list[Fact]) -> dict[int, dt.date]:
    """Compute per-fiscal-year FY-end dates from the company's facts.

    FY-end determination uses ONLY genuine annual flow facts:
    - fp == 'FY' (XBRL fiscal-period code)
    - form in {'10-K', '10-K/A'} (annual reports)
    - start is not None (excludes balance-sheet instants, which have no
      duration to disambiguate and can carry fp=FY at the FY-end too)
    - 340 <= (end - start + 1) <= 380 days (excludes the quarter-
      discrete comparative pieces that non-calendar-FY 10-Ks tag with
      fp=FY because they sit within the fiscal year being reported)

    The duration+start gate is load-bearing — see L-INFRA-012. Without
    it, AAPL, MSFT, JNJ, and SNOW pre-flip-year filings pollute
    max(end) per calendar year with their 90-day quarterly comparatives,
    producing wrong FY-end dates that cascade through _derive_fiscal_year
    and delay Q4 derivation by a year.

    The returned mapping is keyed on `end.year` (calendar year of
    FY-end) so a lookup by fact-end -> fiscal calendar is direct.

    Fallback 1: if no annual flow facts pass the gate, infer from Q3
    flow facts that pass the 60-100d quarter-duration gate (Q3-end + ~91
    days approximates FY-end).
    Fallback 2: empty dict; downstream falls back to end.year (calendar-
    year fiscal years).
    """
    by_year: dict[int, set[dt.date]] = defaultdict(set)
    for f in facts:
        if (
            f.fp == "FY"
            and f.form in ("10-K", "10-K/A")
            and f.start is not None
            and _ANNUAL_DAYS_MIN <= (f.end - f.start).days + 1 <= _ANNUAL_DAYS_MAX
        ):
            by_year[f.end.year].add(f.end)
    if by_year:
        return {y: max(ends) for y, ends in by_year.items()}

    q3_by_year: dict[int, set[dt.date]] = defaultdict(set)
    for f in facts:
        if (
            f.fp == "Q3"
            and f.form in ("10-Q", "10-Q/A")
            and f.start is not None
            and _QUARTER_DAYS_MIN <= (f.end - f.start).days + 1 <= _QUARTER_DAYS_MAX
        ):
            q3_by_year[f.end.year].add(f.end)
    if q3_by_year:
        return {y: max(ends) + dt.timedelta(days=91) for y, ends in q3_by_year.items()}

    return {}


def _derive_fiscal_year(
    end: dt.date,
    fy_end_dates: dict[int, dt.date],
) -> int:
    """Derive the fiscal year for a fact with the given end date.

    A period belongs to the fiscal year whose FY-end is the next FY-end
    at or after `end`. Example: AAPL Q1 FY2010 (end Dec 26, 2009) -> next
    FY-end >= that date = Sep 25, 2010 -> fiscal_year=2010.

    Fallback: if fy_end_dates is empty (no annual or Q3 facts to anchor
    the calendar), return end.year (calendar-year FY).
    """
    if not fy_end_dates:
        return end.year

    candidate = fy_end_dates.get(end.year)
    if candidate is not None and candidate >= end:
        return end.year

    next_year_fy_end = fy_end_dates.get(end.year + 1)
    if next_year_fy_end is not None:
        return end.year + 1

    return end.year


def _classify_instant_end(
    end: dt.date,
    fy_end_dates: dict[int, dt.date],
) -> tuple[int, str] | None:
    """Map an instant fact's end_date to (fiscal_year, period) using the
    fiscal calendar.

    Match `end` against FY-end and approximate Q1/Q2/Q3 ends
    (FY-end - 273/182/91 days) within +/- 15 days. Picks the closest
    match across all candidates. Returns None if nothing matches within
    tolerance (caller skips the fact).
    """
    best: tuple[int, str, int] | None = None  # (fiscal_year, period, abs_dist)

    for fy, fy_end in fy_end_dates.items():
        dist = abs((end - fy_end).days)
        if dist <= _QUARTER_DAYS_TOLERANCE and (best is None or dist < best[2]):
            best = (fy, "FY", dist)

        for q_label, days_back in (("Q3", 91), ("Q2", 182), ("Q1", 273)):
            q_approx = fy_end - dt.timedelta(days=days_back)
            dist = abs((end - q_approx).days)
            if dist <= _QUARTER_DAYS_TOLERANCE and (best is None or dist < best[2]):
                best = (fy, q_label, dist)

    if best is None:
        return None
    return (best[0], best[1])


def _row_template(
    security_id: uuid.UUID,
    fiscal_year: int,
    period: str,
    filing_date: dt.date,
    accepted_date: dt.date,
    end_date: dt.date,
    fields: tuple[str, ...],
) -> dict[str, object]:
    base: dict[str, object] = {
        "security_id": str(security_id),
        "fiscal_year": fiscal_year,
        "period": period,
        "filing_date": filing_date,
        "accepted_date": accepted_date,
        "end_date": end_date,
    }
    for f in fields:
        base[f] = None
    return base


def _collect_per_period(
    facts: list[Fact],
    fields: tuple[str, ...],
    security_id: uuid.UUID,
) -> list[dict[str, object]]:
    """Group facts by (fiscal_year, period, end, filed) and resolve each
    target field using the concept map.

    The company's fiscal calendar (per-year FY-end) is computed first from
    the input facts; this is used to derive fiscal_year FY-end-aware and,
    for instant facts, to derive both fiscal_year and period from end_date.

    The fact's `fy` field is the filing's frame, NOT the data's fiscal
    year, and is ignored.

    For flow facts (start != None), period comes from form+fp via
    period_from_form_fp and the duration filter rejects YTD cumulative
    twins.

    For instant facts (start == None), both fiscal_year and period are
    derived from the end date matching the fiscal calendar. This avoids
    labeling a prior-FY-end balance sheet as period=Q1 just because it
    appears as a comparative in a Q1 10-Q.
    """
    fy_end_dates = _compute_fy_end_dates(facts)

    by_key: dict[tuple[int, str, dt.date, dt.date], list[Fact]] = defaultdict(list)
    for f in facts:
        if f.start is None:
            classification = _classify_instant_end(f.end, fy_end_dates)
            if classification is None:
                continue
            fiscal_year, period = classification
        else:
            try:
                period = period_from_form_fp(form=f.form, fp=f.fp)
            except ValueError:
                continue
            if not _duration_matches_period(f, period):
                continue
            fiscal_year = _derive_fiscal_year(f.end, fy_end_dates)
        by_key[(fiscal_year, period, f.end, f.filed)].append(f)

    rows: list[dict[str, object]] = []
    for (fy, period, end, filed), candidates in by_key.items():
        row = _row_template(
            security_id=security_id,
            fiscal_year=fy,
            period=period,
            filing_date=filed,
            accepted_date=filed,
            end_date=end,
            fields=fields,
        )
        # For instant buckets the derived period may not equal any fact's
        # fp (e.g., a prior-FY-end BS appearing as a Q1 comparative). Skip
        # the fp filter in resolve_field in that case.
        is_instant_bucket = all(f.start is None for f in candidates)
        any_resolved = False
        for field in fields:
            if field not in CONCEPT_MAP or not CONCEPT_MAP[field]:
                continue
            resolved = resolve_field(
                candidates,
                field=field,
                end=end,
                fp=None if is_instant_bucket else period,
            )
            if resolved is not None:
                row[field] = resolved.value
                any_resolved = True
        # Only emit a row if at least one target field resolved. Otherwise
        # we'd emit null-only rows for facts whose concepts belong to other
        # tables (e.g. a Revenues fact creates a bucket in the balance_sheet
        # call but no balance-sheet field resolves to it).
        if any_resolved:
            rows.append(row)
    return rows


def derive_q4_rows(
    fy_and_q_rows: list[dict[str, object]],
    fields: tuple[str, ...],
) -> list[dict[str, object]]:
    """For each FY filing, derive Q4 = FY - (Q1+Q2+Q3) using the latest
    Q1/Q2/Q3 values whose accepted_date <= the FY filing's accepted_date.

    The Q4 row's accepted_date = the FY filing's accepted_date and its
    end_date = the FY filing's end_date (Q4 ends at FY-end).
    Restated FY filings produce additional Q4 rows.
    """
    fy_rows = [r for r in fy_and_q_rows if str(r["period"]) == "FY"]
    quarterly_rows = [r for r in fy_and_q_rows if str(r["period"]) in {"Q1", "Q2", "Q3"}]

    by_fy_period: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for r in quarterly_rows:
        fy = cast(int, r["fiscal_year"])
        period = str(r["period"])
        by_fy_period[(fy, period)].append(r)
    for key in by_fy_period:
        by_fy_period[key].sort(key=lambda r: cast(dt.date, r["accepted_date"]))

    derived: list[dict[str, object]] = []
    for fy_row in fy_rows:
        fy = cast(int, fy_row["fiscal_year"])
        fy_accepted = fy_row["accepted_date"]
        assert isinstance(fy_accepted, dt.date)
        fy_end_date = fy_row["end_date"]
        assert isinstance(fy_end_date, dt.date)

        q_rows: dict[str, dict[str, object] | None] = {"Q1": None, "Q2": None, "Q3": None}
        for q_label in ("Q1", "Q2", "Q3"):
            available = [
                r
                for r in by_fy_period.get((fy, q_label), [])
                if cast(dt.date, r["accepted_date"]) <= fy_accepted
            ]
            if available:
                q_rows[q_label] = available[-1]

        if any(v is None for v in q_rows.values()):
            continue

        q4_row = _row_template(
            security_id=uuid.UUID(str(fy_row["security_id"])),
            fiscal_year=fy,
            period="Q4",
            filing_date=cast(dt.date, fy_row["filing_date"]),
            accepted_date=fy_accepted,
            end_date=fy_end_date,
            fields=fields,
        )
        any_derived = False
        for field in fields:
            if field not in _FLOW_FIELDS:
                continue
            fy_v = fy_row.get(field)
            q1_row = q_rows["Q1"]
            q2_row = q_rows["Q2"]
            q3_row = q_rows["Q3"]
            assert q1_row is not None and q2_row is not None and q3_row is not None
            q1_v = q1_row.get(field)
            q2_v = q2_row.get(field)
            q3_v = q3_row.get(field)
            if any(v is None for v in (fy_v, q1_v, q2_v, q3_v)):
                continue
            q4_row[field] = (
                cast(float, fy_v) - cast(float, q1_v) - cast(float, q2_v) - cast(float, q3_v)
            )
            any_derived = True
        if any_derived:
            derived.append(q4_row)
    return derived


def normalize_to_tables(*, facts: list[Fact], security_id: uuid.UUID) -> NormalizedTables:
    """Flatten facts into the three schema-shaped DataFrames."""
    inc_rows = _collect_per_period(facts, _INCOME_FIELDS, security_id)
    inc_rows.extend(derive_q4_rows(inc_rows, _INCOME_FIELDS))

    bs_rows = _collect_per_period(facts, _BALANCE_FIELDS, security_id)
    # Balance sheet is point-in-time, NOT flow; no Q4 derivation.

    cf_rows = _collect_per_period(facts, _CASHFLOW_FIELDS, security_id)
    cf_rows.extend(derive_q4_rows(cf_rows, _CASHFLOW_FIELDS))

    return NormalizedTables(
        income_statement=pd.DataFrame(inc_rows) if inc_rows else _empty_df(_INCOME_FIELDS),
        balance_sheet=pd.DataFrame(bs_rows) if bs_rows else _empty_df(_BALANCE_FIELDS),
        cashflow=pd.DataFrame(cf_rows) if cf_rows else _empty_df(_CASHFLOW_FIELDS),
    )


def _empty_df(fields: tuple[str, ...]) -> pd.DataFrame:
    cols = ["security_id", "fiscal_year", "period", "filing_date", "accepted_date", "end_date"]
    cols.extend(fields)
    return pd.DataFrame(columns=cols)
