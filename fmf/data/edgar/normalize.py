"""Normalize XBRL facts to schema rows.

Pipeline:
1. Group facts by (fiscal_year, period, end, filed) where fiscal_year is
   derived from `end.year` (US convention: fiscal years are named for
   the calendar year they end in). The fact's `fy` field is the filing's
   frame, NOT the data's fiscal year, so trusting it would mislabel
   comparatives (L4). For period, use form+fp via period_from_form_fp.
2. For each (fiscal_year, period, filed, end) group, walk concept_map per
   target field and pick the first match (priority + unit gate, L1+L4).
3. Emit one row per (fiscal_year, period, filed) — restatements coexist (L2).
4. Derive Q4 per fiscal_year for flow-statement fields. For each FY filing,
   find the latest Q1/Q2/Q3 values whose accepted_date <= the FY filing's
   accepted_date, compute Q4 = FY - (Q1+Q2+Q3), and emit a Q4 row with
   accepted_date = FY filing's accepted_date (Q4 only becomes knowable
   when the 10-K lands). Restated FY filings produce additional Q4 rows
   per restatement. (L3)
5. Return NormalizedTables(income_statement, balance_sheet, cashflow).
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


def _row_template(
    security_id: uuid.UUID,
    fiscal_year: int,
    period: str,
    filing_date: dt.date,
    accepted_date: dt.date,
    fields: tuple[str, ...],
) -> dict[str, object]:
    base: dict[str, object] = {
        "security_id": str(security_id),
        "fiscal_year": fiscal_year,
        "period": period,
        "filing_date": filing_date,
        "accepted_date": accepted_date,
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

    fiscal_year is derived from `f.end.year`. period is derived from form+fp.
    For flow concepts, the duration filter rejects YTD cumulative twins so
    Q2/Q3 carry the discrete-quarter value, not the YTD aggregate.
    """
    by_key: dict[tuple[int, str, dt.date, dt.date], list[Fact]] = defaultdict(list)
    for f in facts:
        try:
            period = period_from_form_fp(form=f.form, fp=f.fp)
        except ValueError:
            continue
        if not _duration_matches_period(f, period):
            continue
        # Critical: fiscal_year comes from end.year, NOT from f.fy.
        fiscal_year = f.end.year
        by_key[(fiscal_year, period, f.end, f.filed)].append(f)

    rows: list[dict[str, object]] = []
    for (fy, period, end, filed), candidates in by_key.items():
        row = _row_template(
            security_id=security_id,
            fiscal_year=fy,
            period=period,
            filing_date=filed,
            accepted_date=filed,
            fields=fields,
        )
        for field in fields:
            if field not in CONCEPT_MAP or not CONCEPT_MAP[field]:
                continue
            resolved = resolve_field(candidates, field=field, end=end, fp=period)
            if resolved is not None:
                row[field] = resolved.value
        rows.append(row)
    return rows


def derive_q4_rows(
    fy_and_q_rows: list[dict[str, object]],
    fields: tuple[str, ...],
) -> list[dict[str, object]]:
    """For each FY filing, derive Q4 = FY - (Q1+Q2+Q3) using the latest
    Q1/Q2/Q3 values whose accepted_date <= the FY filing's accepted_date.

    The Q4 row's accepted_date = the FY filing's accepted_date.
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
    cols = ["security_id", "fiscal_year", "period", "filing_date", "accepted_date"]
    cols.extend(fields)
    return pd.DataFrame(columns=cols)
