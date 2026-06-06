"""Normalize tests — the landmine layer.

Tests cover each landmine:
- L1: concept-map fallback (revenue using Revenues vs contract-revenue)
- L2: keep every filed date as a distinct row (restatements coexist)
- L3: period derivation + PIT-correct Q4 + duration disambiguation
- L4: fiscal-frame matching from end.year (NOT from fact.fy),
      unit scaling, comparative mislabel guard
"""

from __future__ import annotations

import datetime as dt
import uuid

import pandas as pd
import pytest

from fmf.data.edgar.companyfacts import Fact
from fmf.data.edgar.normalize import (
    NormalizedTables,
    normalize_to_tables,
    period_from_form_fp,
)

AAPL_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def _fy_fact(
    concept: str,
    end: dt.date,
    filed: dt.date,
    value: float,
    unit: str = "USD",
    fy_hint: int | None = None,
    start: dt.date | None = None,
) -> Fact:
    if start is None:
        start = end - dt.timedelta(days=364)
    return Fact(
        concept=concept,
        end=end,
        filed=filed,
        value=value,
        unit=unit,
        form="10-K",
        fp="FY",
        fy=fy_hint if fy_hint is not None else end.year,
        start=start,
    )


def _q_fact(
    concept: str,
    end: dt.date,
    filed: dt.date,
    value: float,
    fp: str,
    unit: str = "USD",
    fy_hint: int | None = None,
    start: dt.date | None = None,
) -> Fact:
    if start is None:
        start = end - dt.timedelta(days=90)
    return Fact(
        concept=concept,
        end=end,
        filed=filed,
        value=value,
        unit=unit,
        form="10-Q",
        fp=fp,
        fy=fy_hint if fy_hint is not None else end.year,
        start=start,
    )


# --- period_from_form_fp (L3) ---


@pytest.mark.parametrize(
    "form,fp,expected",
    [
        ("10-K", "FY", "FY"),
        ("10-K/A", "FY", "FY"),
        ("10-Q", "Q1", "Q1"),
        ("10-Q", "Q2", "Q2"),
        ("10-Q", "Q3", "Q3"),
        ("10-Q/A", "Q3", "Q3"),
    ],
)
def test_period_from_form_fp(form: str, fp: str, expected: str) -> None:
    assert period_from_form_fp(form=form, fp=fp) == expected


def test_period_from_form_fp_rejects_unknown_combo() -> None:
    with pytest.raises(ValueError):
        period_from_form_fp(form="10-Q", fp="FY")


# --- normalize_to_tables: L1 concept-map fallback ---


def test_revenue_uses_concept_map_priority() -> None:
    facts = [
        _fy_fact("Revenues", dt.date(2023, 12, 31), dt.date(2024, 2, 1), 100.0),
        _fy_fact("SalesRevenueNet", dt.date(2023, 12, 31), dt.date(2024, 2, 1), 999.0),
    ]
    out = normalize_to_tables(facts=facts, security_id=AAPL_ID)
    inc = out.income_statement
    row = inc[(inc["fiscal_year"] == 2023) & (inc["period"] == "FY")]
    assert len(row) == 1
    assert row.iloc[0]["revenue"] == 100.0


def test_revenue_falls_back_when_top_concept_missing() -> None:
    facts = [
        _fy_fact(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            dt.date(2023, 12, 31),
            dt.date(2024, 2, 1),
            200.0,
        )
    ]
    out = normalize_to_tables(facts=facts, security_id=AAPL_ID)
    inc = out.income_statement
    row = inc[(inc["fiscal_year"] == 2023) & (inc["period"] == "FY")]
    assert row.iloc[0]["revenue"] == 200.0


# --- normalize_to_tables: L2 keep every version ---


def test_restatement_lands_as_separate_row() -> None:
    """Two filings of the same FY2023 NetIncomeLoss, different filed dates."""
    facts = [
        _fy_fact("NetIncomeLoss", dt.date(2023, 12, 31), dt.date(2024, 2, 1), 100.0),
        _fy_fact("NetIncomeLoss", dt.date(2023, 12, 31), dt.date(2024, 8, 15), 105.0),
    ]
    out = normalize_to_tables(facts=facts, security_id=AAPL_ID)
    inc = out.income_statement
    fy2023 = inc[(inc["fiscal_year"] == 2023) & (inc["period"] == "FY")]
    assert len(fy2023) == 2
    accepted = set(fy2023["accepted_date"])
    assert dt.date(2024, 2, 1) in accepted
    assert dt.date(2024, 8, 15) in accepted


# --- normalize_to_tables: L3 Q4 derivation (PIT-correct, FY-anchored) ---


def test_q4_derived_with_realistic_filing_dates() -> None:
    """Real EDGAR data: Q1/Q2/Q3 filed separately throughout the year, FY
    (10-K) filed after FY end. Q4 row's accepted_date = FY filing's date.
    """
    q1 = _q_fact("Revenues", dt.date(2023, 3, 31), dt.date(2023, 5, 1), 80.0, "Q1")
    q2 = _q_fact("Revenues", dt.date(2023, 6, 30), dt.date(2023, 8, 1), 100.0, "Q2")
    q3 = _q_fact("Revenues", dt.date(2023, 9, 30), dt.date(2023, 11, 1), 110.0, "Q3")
    fy = _fy_fact("Revenues", dt.date(2023, 12, 31), dt.date(2024, 2, 1), 400.0)

    out = normalize_to_tables(facts=[q1, q2, q3, fy], security_id=AAPL_ID)
    inc = out.income_statement
    q4 = inc[(inc["fiscal_year"] == 2023) & (inc["period"] == "Q4")]
    assert len(q4) == 1
    assert q4.iloc[0]["revenue"] == pytest.approx(400.0 - 80.0 - 100.0 - 110.0)
    assert q4.iloc[0]["accepted_date"] == dt.date(2024, 2, 1)
    # Q4 ends at FY-end: its end_date inherits from the FY row's end_date.
    assert q4.iloc[0]["end_date"] == dt.date(2023, 12, 31)


def test_q4_uses_only_q_filings_known_at_fy_accepted_date() -> None:
    """PIT correctness: a Q-restatement filed AFTER the FY filing must NOT
    be used by the Q4 derivation for that FY filing.
    """
    q1_original = _q_fact("Revenues", dt.date(2023, 3, 31), dt.date(2023, 5, 1), 80.0, "Q1")
    q1_restated = _q_fact("Revenues", dt.date(2023, 3, 31), dt.date(2024, 8, 15), 82.0, "Q1")
    q2 = _q_fact("Revenues", dt.date(2023, 6, 30), dt.date(2023, 8, 1), 100.0, "Q2")
    q3 = _q_fact("Revenues", dt.date(2023, 9, 30), dt.date(2023, 11, 1), 110.0, "Q3")
    fy = _fy_fact("Revenues", dt.date(2023, 12, 31), dt.date(2024, 2, 1), 400.0)

    out = normalize_to_tables(
        facts=[q1_original, q1_restated, q2, q3, fy],
        security_id=AAPL_ID,
    )
    inc = out.income_statement
    q4 = inc[(inc["fiscal_year"] == 2023) & (inc["period"] == "Q4")]
    assert len(q4) == 1
    assert q4.iloc[0]["revenue"] == pytest.approx(110.0)


def test_q4_derived_per_fy_restatement() -> None:
    """If the FY 10-K is restated, a second Q4 row is derived for the
    restated FY's accepted_date.
    """
    q1 = _q_fact("Revenues", dt.date(2023, 3, 31), dt.date(2023, 5, 1), 80.0, "Q1")
    q2 = _q_fact("Revenues", dt.date(2023, 6, 30), dt.date(2023, 8, 1), 100.0, "Q2")
    q3 = _q_fact("Revenues", dt.date(2023, 9, 30), dt.date(2023, 11, 1), 110.0, "Q3")
    fy_original = _fy_fact("Revenues", dt.date(2023, 12, 31), dt.date(2024, 2, 1), 400.0)
    fy_restated = _fy_fact("Revenues", dt.date(2023, 12, 31), dt.date(2024, 8, 15), 405.0)

    out = normalize_to_tables(
        facts=[q1, q2, q3, fy_original, fy_restated],
        security_id=AAPL_ID,
    )
    inc = out.income_statement
    q4 = inc[(inc["fiscal_year"] == 2023) & (inc["period"] == "Q4")]
    assert len(q4) == 2
    by_accepted = {r["accepted_date"]: r for _, r in q4.iterrows()}
    assert by_accepted[dt.date(2024, 2, 1)]["revenue"] == pytest.approx(110.0)
    assert by_accepted[dt.date(2024, 8, 15)]["revenue"] == pytest.approx(115.0)
    # Both Q4 rows end at FY-end.
    assert by_accepted[dt.date(2024, 2, 1)]["end_date"] == dt.date(2023, 12, 31)
    assert by_accepted[dt.date(2024, 8, 15)]["end_date"] == dt.date(2023, 12, 31)


def test_q4_not_derived_when_any_quarter_missing() -> None:
    fy = _fy_fact("Revenues", dt.date(2023, 12, 31), dt.date(2024, 2, 1), 400.0)
    q1 = _q_fact("Revenues", dt.date(2023, 3, 31), dt.date(2023, 5, 1), 80.0, "Q1")
    out = normalize_to_tables(facts=[fy, q1], security_id=AAPL_ID)
    inc = out.income_statement
    q4 = inc[(inc["fiscal_year"] == 2023) & (inc["period"] == "Q4")]
    assert len(q4) == 0


def test_q2_discrete_picked_over_ytd_twin() -> None:
    """EDGAR companyfacts returns BOTH discrete and YTD facts for flow
    concepts. The duration filter must select the discrete quarter.
    """
    q1 = _q_fact(
        "Revenues",
        end=dt.date(2023, 3, 31),
        filed=dt.date(2023, 5, 1),
        value=80.0,
        fp="Q1",
        start=dt.date(2023, 1, 1),
    )
    q2_discrete = _q_fact(
        "Revenues",
        end=dt.date(2023, 6, 30),
        filed=dt.date(2023, 8, 1),
        value=100.0,
        fp="Q2",
        start=dt.date(2023, 4, 1),
    )
    q2_ytd = _q_fact(
        "Revenues",
        end=dt.date(2023, 6, 30),
        filed=dt.date(2023, 8, 1),
        value=180.0,
        fp="Q2",
        start=dt.date(2023, 1, 1),
    )
    q3_discrete = _q_fact(
        "Revenues",
        end=dt.date(2023, 9, 30),
        filed=dt.date(2023, 11, 1),
        value=110.0,
        fp="Q3",
        start=dt.date(2023, 7, 1),
    )
    q3_ytd = _q_fact(
        "Revenues",
        end=dt.date(2023, 9, 30),
        filed=dt.date(2023, 11, 1),
        value=290.0,
        fp="Q3",
        start=dt.date(2023, 1, 1),
    )
    fy = _fy_fact(
        "Revenues",
        end=dt.date(2023, 12, 31),
        filed=dt.date(2024, 2, 1),
        value=400.0,
        start=dt.date(2023, 1, 1),
    )

    out = normalize_to_tables(
        facts=[q1, q2_discrete, q2_ytd, q3_discrete, q3_ytd, fy],
        security_id=AAPL_ID,
    )
    inc = out.income_statement
    q2 = inc[(inc["fiscal_year"] == 2023) & (inc["period"] == "Q2")]
    assert len(q2) == 1
    assert q2.iloc[0]["revenue"] == pytest.approx(100.0)
    q3 = inc[(inc["fiscal_year"] == 2023) & (inc["period"] == "Q3")]
    assert q3.iloc[0]["revenue"] == pytest.approx(110.0)
    q4 = inc[(inc["fiscal_year"] == 2023) & (inc["period"] == "Q4")]
    assert len(q4) == 1
    assert q4.iloc[0]["revenue"] == pytest.approx(110.0)


# --- normalize_to_tables: L4 fiscal-frame + unit + comparative mislabel ---


def test_eps_diluted_picked_with_correct_unit() -> None:
    facts = [
        _fy_fact(
            "EarningsPerShareDiluted",
            dt.date(2023, 12, 31),
            dt.date(2024, 2, 1),
            value=6.13,
            unit="USD/shares",
        ),
        _fy_fact(
            "EarningsPerShareDiluted",
            dt.date(2023, 12, 31),
            dt.date(2024, 2, 1),
            value=6_130_000_000.0,
            unit="USD",
        ),
    ]
    out = normalize_to_tables(facts=facts, security_id=AAPL_ID)
    inc = out.income_statement
    row = inc[(inc["fiscal_year"] == 2023) & (inc["period"] == "FY")]
    assert row.iloc[0]["eps_diluted"] == pytest.approx(6.13)


def test_fiscal_year_derived_from_end_not_from_fy_hint() -> None:
    """EDGAR's fact.fy field is the filing's frame, not the data's fiscal
    year. A FY2022 comparative in the FY2023 10-K carries fy=2023, fp=FY
    while its end is in 2022. Normalize must label the row by end.year.
    """
    comparative = _fy_fact(
        "Revenues",
        end=dt.date(2022, 12, 31),
        filed=dt.date(2024, 2, 1),
        value=300.0,
        fy_hint=2023,
    )
    out = normalize_to_tables(facts=[comparative], security_id=AAPL_ID)
    inc = out.income_statement
    fy2022 = inc[inc["fiscal_year"] == 2022]
    assert len(fy2022) == 1
    assert fy2022.iloc[0]["revenue"] == pytest.approx(300.0)
    fy2023 = inc[inc["fiscal_year"] == 2023]
    assert len(fy2023) == 0


def test_aapl_fiscal_year_end_is_september_not_december() -> None:
    """AAPL fiscal year ends late September. End=2023-09-30 must be FY2023."""
    aapl_fy23 = _fy_fact(
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        end=dt.date(2023, 9, 30),
        filed=dt.date(2023, 11, 3),
        value=383_285_000_000.0,
    )
    out = normalize_to_tables(facts=[aapl_fy23], security_id=AAPL_ID)
    inc = out.income_statement
    fy2023 = inc[inc["fiscal_year"] == 2023]
    assert len(fy2023) == 1
    assert fy2023.iloc[0]["revenue"] == pytest.approx(383_285_000_000.0)


# --- output shape ---


def test_output_has_three_tables_with_schema_columns() -> None:
    facts = [_fy_fact("Revenues", dt.date(2023, 12, 31), dt.date(2024, 2, 1), 100.0)]
    out = normalize_to_tables(facts=facts, security_id=AAPL_ID)
    assert isinstance(out, NormalizedTables)
    assert isinstance(out.income_statement, pd.DataFrame)
    assert isinstance(out.balance_sheet, pd.DataFrame)
    assert isinstance(out.cashflow, pd.DataFrame)
    for col in (
        "security_id",
        "fiscal_year",
        "period",
        "filing_date",
        "accepted_date",
        "end_date",
    ):
        assert col in out.income_statement.columns


# --- instant-period derivation (Step B) ---


def test_instant_fact_at_fy_end_labeled_fy_period() -> None:
    """A balance-sheet fact at the FY-end date is labeled period=FY,
    even if its fp is something else (e.g., the FILING's fp).

    AAPL's BS at end of FY2009 (Sep 26, 2009) appears as a comparative in
    the Q1 FY2010 10-Q with fp='Q1'. Normalize must label it period='FY',
    not period='Q1'.
    """
    fy_anchor = _fy_fact(
        "Revenues",
        end=dt.date(2009, 9, 26),
        filed=dt.date(2009, 10, 26),
        value=42_905_000_000.0,
    )
    bs_comparative = Fact(
        concept="Assets",
        end=dt.date(2009, 9, 26),
        filed=dt.date(2010, 1, 25),
        value=47_501_000_000.0,
        unit="USD",
        form="10-Q",
        fp="Q1",
        fy=2010,
        start=None,
    )
    out = normalize_to_tables(
        facts=[fy_anchor, bs_comparative],
        security_id=AAPL_ID,
    )
    bs = out.balance_sheet
    bs2009 = bs[bs["fiscal_year"] == 2009]
    assert len(bs2009) == 1
    assert (
        bs2009.iloc[0]["period"] == "FY"
    ), f"Instant BS at FY-end must be labeled period=FY, got {bs2009.iloc[0]['period']}"
    assert bs2009.iloc[0]["end_date"] == dt.date(2009, 9, 26)
    assert bs2009.iloc[0]["total_assets"] == pytest.approx(47_501_000_000.0)


def test_instant_fact_at_q1_fy2010_end_labeled_q1_fy2010() -> None:
    """AAPL's Q1 FY2010 ends Dec 26, 2009. A BS fact at that end_date
    must be labeled (fiscal_year=2010, period=Q1), NOT (fy=2009, period=Q1)
    which is what end.year-based labeling would give.
    """
    fy2010_anchor = _fy_fact(
        "Revenues",
        end=dt.date(2010, 9, 25),
        filed=dt.date(2010, 10, 27),
        value=65_225_000_000.0,
    )
    bs_q1 = Fact(
        concept="Assets",
        end=dt.date(2009, 12, 26),
        filed=dt.date(2010, 1, 25),
        value=53_926_000_000.0,
        unit="USD",
        form="10-Q",
        fp="Q1",
        fy=2010,
        start=None,
    )
    out = normalize_to_tables(
        facts=[fy2010_anchor, bs_q1],
        security_id=AAPL_ID,
    )
    bs = out.balance_sheet
    q1 = bs[(bs["fiscal_year"] == 2010) & (bs["period"] == "Q1")]
    assert len(q1) == 1
    assert q1.iloc[0]["end_date"] == dt.date(2009, 12, 26)
    assert q1.iloc[0]["total_assets"] == pytest.approx(53_926_000_000.0)
