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
    assert bs2009.iloc[0]["period"] == "FY", (
        f"Instant BS at FY-end must be labeled period=FY, got {bs2009.iloc[0]['period']}"
    )
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


# --- L-INFRA-012 regression: non-calendar-FY 10-K emits quarterly pieces
# tagged fp=FY; pre-fix this polluted max(end) over fp=FY facts and
# delayed Q4 derivation by a year. Post-fix the duration+start gate
# restricts FY-end determination to genuine ~365d annual flow facts. ---


@pytest.mark.parametrize(
    "label, fy_end, fy_filed, q_ends, q_filings, q_values, fy_revenue, expected_q4",
    [
        # AAPL FY2015 (late-September FY). Real revenue numbers from the
        # 10-K; Q4 = FY - (Q1+Q2+Q3) = 233.715B - 74.599B - 58.010B - 49.605B
        # = 51.501B. q_filings are realistic ~30d-after-quarter-end 10-Q dates.
        # Pre-fix: fp=FY comparatives pollute max(end), discrete Q1 (end
        # 2014-12-27) gets mislabeled into fiscal_year=2014, derive_q4_rows
        # can't find Q1 in fy=2015 bucket, Q4 missing. Post-fix: comparatives
        # excluded by duration gate, discrete Q1/Q2/Q3 correctly labeled,
        # Q4 emits at FY filing date.
        (
            "AAPL_FY2015",
            dt.date(2015, 9, 26),  # FY end
            dt.date(2015, 10, 28),  # FY filed (10-K)
            [
                dt.date(2014, 12, 27),  # Q1 FY2015 end
                dt.date(2015, 3, 28),  # Q2 FY2015 end
                dt.date(2015, 6, 27),  # Q3 FY2015 end
            ],
            [
                dt.date(2015, 1, 28),  # Q1 10-Q filed
                dt.date(2015, 4, 28),  # Q2 10-Q filed
                dt.date(2015, 7, 22),  # Q3 10-Q filed
            ],
            [
                74_599_000_000.0,  # Q1
                58_010_000_000.0,  # Q2
                49_605_000_000.0,  # Q3
            ],
            233_715_000_000.0,  # FY
            51_501_000_000.0,  # expected Q4
        ),
        # MSFT FY2020 (June-end FY). Synthetic round-number values; the test
        # is about the mechanism, not MSFT's actual revenue.
        (
            "MSFT_FY2020",
            dt.date(2020, 6, 30),  # FY end
            dt.date(2020, 7, 30),  # FY filed
            [
                dt.date(2019, 9, 30),  # Q1 FY2020 end
                dt.date(2019, 12, 31),  # Q2 FY2020 end
                dt.date(2020, 3, 31),  # Q3 FY2020 end
            ],
            [
                dt.date(2019, 10, 23),  # Q1 10-Q filed
                dt.date(2020, 1, 29),  # Q2 10-Q filed
                dt.date(2020, 4, 29),  # Q3 10-Q filed
            ],
            [
                22_000_000_000.0,
                27_000_000_000.0,
                23_000_000_000.0,
            ],
            100_000_000_000.0,
            28_000_000_000.0,
        ),
        # JNJ FY ending early January (calendar-year-shifted; the codebase
        # convention is fiscal_year = end.year, so this FY label is 2021
        # internally even though JNJ markets it as FY2020).
        (
            "JNJ_FY_ending_2021_01_03",
            dt.date(2021, 1, 3),  # FY end (Sunday nearest Dec 31, 2020)
            dt.date(2021, 2, 22),  # FY filed
            [
                dt.date(2020, 3, 29),  # Q1 end
                dt.date(2020, 6, 28),  # Q2 end
                dt.date(2020, 9, 27),  # Q3 end
            ],
            [
                dt.date(2020, 4, 28),  # Q1 10-Q filed
                dt.date(2020, 7, 21),  # Q2 10-Q filed
                dt.date(2020, 10, 20),  # Q3 10-Q filed
            ],
            [
                20_700_000_000.0,
                18_300_000_000.0,
                21_100_000_000.0,
            ],
            82_584_000_000.0,
            22_484_000_000.0,
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_q4_derives_at_fy_filing_for_non_calendar_fy(
    label: str,
    fy_end: dt.date,
    fy_filed: dt.date,
    q_ends: list[dt.date],
    q_filings: list[dt.date],
    q_values: list[float],
    fy_revenue: float,
    expected_q4: float,
) -> None:
    """L-INFRA-012 regression: non-calendar-FY 10-Ks tag the quarterly
    comparative pieces with fp=FY (because they are within the fiscal
    year being reported). Pre-fix, _compute_fy_end_dates picked
    max(end) over ALL fp=FY facts per calendar year, which picked the
    Q1-of-next-FY end (e.g., AAPL Q1 FY2015 end Dec 27, 2014, in the
    calendar-2014 bucket) instead of the real FY-end (Sept 27, 2014).
    The wrong FY-end cascaded through _derive_fiscal_year, mislabeled
    Q1 facts into the prior fiscal_year, and derive_q4_rows could not
    find Q1/Q2/Q3 in the target fy bucket at the FY filing's accepted
    date. Q4 only emerged a year later via the next FY's comparatives.

    The fix in _compute_fy_end_dates restricts FY-end candidates to
    facts with start != None AND duration in [340, 380] days — i.e.,
    genuine annual flow facts only.

    Each parametrization emits THREE sets of facts: (a) the discrete
    Q1/Q2/Q3 from each quarter's own 10-Q (fp=Q1/Q2/Q3, form=10-Q,
    90d) which create the Q1/Q2/Q3 rows; (b) the genuine annual fact
    from the FY's 10-K (fp=FY, form=10-K, 365d) which creates the FY
    row and post-fix anchors fy_end_dates; (c) the FY 10-K's quarterly
    comparative pieces tagged fp=FY (form=10-K, 90d) which are the
    bug-trigger but never become rows themselves (period_from_form_fp
    routes them to FY and the duration filter discards them).

    Scope: AAPL, MSFT, JNJ, SNOW across their year-lagged ranges.
    Calendar-FY tickers (ZTS, GWW, HSY, JPM) are unaffected because
    the annual's December end is naturally the latest in its calendar
    year.
    """
    # Some unique security id per parametrization so the test cases are
    # independent in the output table.
    security_id = uuid.uuid5(uuid.NAMESPACE_DNS, label)

    facts: list[Fact] = []

    # (a) Discrete Q1/Q2/Q3 from each quarter's own 10-Q. These create
    # the Q1/Q2/Q3 rows in the output that derive_q4_rows subtracts from
    # FY. Without these, no Q1/Q2/Q3 rows exist and Q4 cannot derive
    # regardless of the FY-end-determination fix.
    quarter_names = ("Q1", "Q2", "Q3")
    for q_end, q_filed, q_val, q_name in zip(
        q_ends, q_filings, q_values, quarter_names, strict=True
    ):
        facts.append(
            _q_fact(
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                end=q_end,
                filed=q_filed,
                value=q_val,
                fp=q_name,
            )
        )

    # (b) Genuine annual revenue fact (365d duration) from the FY's 10-K.
    facts.append(
        _fy_fact(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            end=fy_end,
            filed=fy_filed,
            value=fy_revenue,
            start=fy_end - dt.timedelta(days=364),
        )
    )

    # (c) Quarter-discrete comparative pieces FROM THE FY 10-K, all tagged
    # fp=FY (the bug-triggering pattern). 90d duration each. These never
    # become rows but pre-fix they pollute max(end) over fp=FY facts and
    # corrupt fy_end_dates. Post-fix they are excluded by the duration
    # gate.
    for q_end, q_val in zip(q_ends, q_values, strict=True):
        facts.append(
            Fact(
                concept="RevenueFromContractWithCustomerExcludingAssessedTax",
                end=q_end,
                filed=fy_filed,
                value=q_val,
                unit="USD",
                form="10-K",
                fp="FY",  # mislabeled by EDGAR's emit format
                fy=fy_end.year,
                start=q_end - dt.timedelta(days=90),
            )
        )

    out = normalize_to_tables(facts=facts, security_id=security_id)
    inc = out.income_statement

    # Target fiscal_year per the codebase convention: fiscal_year = end.year
    # of the FY-end. For JNJ this is 2021 (calendar year of Jan 3 2021),
    # not 2020 (JNJ's marketing label).
    target_fy = fy_end.year

    q4_rows = inc[(inc["fiscal_year"] == target_fy) & (inc["period"] == "Q4")]
    assert len(q4_rows) >= 1, (
        f"{label}: Q4 missing for fiscal_year={target_fy}. "
        f"derive_q4_rows could not assemble Q1+Q2+Q3+FY at FY filing date. "
        f"Either FY-end determination is still wrong (Q1/Q2/Q3 mislabeled "
        f"into the prior fiscal_year), or the discrete 10-Q facts didn't "
        f"land as rows. Inspect inc[['fiscal_year','period','end_date',"
        f"'accepted_date','revenue']] before concluding the fix is wrong."
    )

    q4_accepted = q4_rows.iloc[0]["accepted_date"]
    if not isinstance(q4_accepted, dt.date):
        import pandas as pd

        q4_accepted = pd.Timestamp(q4_accepted).date()
    assert q4_accepted == fy_filed, (
        f"{label}: Q4 accepted_date {q4_accepted} != FY filing date "
        f"{fy_filed}. The year-lag bug is present: Q4 only emerged via "
        f"a later filing instead of contemporaneously with the FY 10-K."
    )

    q4_revenue = q4_rows.iloc[0]["revenue"]
    assert q4_revenue == pytest.approx(expected_q4, rel=0.001), (
        f"{label}: Q4 revenue {q4_revenue} != FY-Q1-Q2-Q3={expected_q4} "
        f"(rel tol 0.1%). derive_q4_rows summed the wrong inputs."
    )


@pytest.mark.parametrize(
    "ticker, fy",
    [
        ("AAPL", 2015),  # Late-September FY; previously year-lagged FY2009-2019.
        ("MSFT", 2020),  # June-end FY; previously year-lagged FY2010-2024.
    ],
    ids=lambda v: str(v),
)
def test_q4_fixture_regression_emits_at_fy_filing(ticker: str, fy: int) -> None:
    """L-INFRA-012 fixture-side regression.

    The synthetic test exercises the no-prior-FY-end-in-map branch
    (the test parametrizations don't include a prior-year genuine
    annual fact). This locks the real-data branch where the prior
    FY-end IS present in the map but earlier than end.year of the
    affected Q1 fact. After the _compute_fy_end_dates fix, an
    affected ticker's Q4 in a previously-year-lagged fiscal_year
    must appear in mini.duckdb with accepted_date equal to that FY's
    10-K filing, not a year later via the next FY's comparative.

    Assertion: MIN(accepted_date) of Q4 row(s) for (ticker, fy) ==
    MIN(accepted_date) of FY row(s) for the same (ticker, fy).
    """
    from pathlib import Path

    import duckdb

    fixture = Path(__file__).parents[2] / "fixtures" / "mini.duckdb"
    if not fixture.exists():
        pytest.skip("fixture not built yet")

    conn = duckdb.connect(str(fixture), read_only=True)
    try:
        sid_row = conn.execute(
            'SELECT security_id FROM "securities" WHERE symbol = ?', [ticker]
        ).fetchone()
        if sid_row is None:
            pytest.skip(f"{ticker} missing from fixture")
        sid = str(sid_row[0])

        fy_filed = conn.execute(
            'SELECT MIN(accepted_date) FROM "income_statement" '
            "WHERE security_id = ? AND period = ? AND fiscal_year = ? "
            "AND revenue IS NOT NULL",
            [sid, "FY", fy],
        ).fetchone()
        assert fy_filed is not None and fy_filed[0] is not None, (
            f"{ticker} FY{fy} missing from fixture"
        )
        fy_filed_date = fy_filed[0]

        q4_filed = conn.execute(
            'SELECT MIN(accepted_date) FROM "income_statement" '
            "WHERE security_id = ? AND period = ? AND fiscal_year = ? "
            "AND revenue IS NOT NULL",
            [sid, "Q4", fy],
        ).fetchone()
        assert q4_filed is not None and q4_filed[0] is not None, (
            f"{ticker} Q4 FY{fy} missing from fixture — "
            f"L-INFRA-012 fix may have regressed or the rebuild "
            f"did not pick up the new normalize logic."
        )
        q4_filed_date = q4_filed[0]

        assert q4_filed_date == fy_filed_date, (
            f"{ticker} Q4 FY{fy} accepted_date {q4_filed_date} != "
            f"FY{fy} 10-K filing date {fy_filed_date}. "
            f"L-INFRA-012 regression: Q4 is year-lagged again."
        )
    finally:
        conn.close()


# --- L-INFRA-013 regression: derive_q4_rows must break accepted_date ties
# by latest end_date, not by stable-sort input order. The Q3 10-Q's
# comparative facts inherit fp=Q3 (the filing's frame) and land in the
# (fy, Q3) bucket alongside the genuine discrete Q3. They share the Q3
# 10-Q's accepted_date. Pre-fix, derive_q4_rows sorts by accepted_date
# alone; ties resolve via stable-sort input order, often selecting a
# phantom null-revenue row as available[-1]. ---


@pytest.mark.parametrize(
    "label, fy_end, fy_filed, q_ends, q_filings, q_values, fy_revenue, expected_q4",
    [
        (
            "AAPL_FY2015",
            dt.date(2015, 9, 26),
            dt.date(2015, 10, 28),
            [
                dt.date(2014, 12, 27),
                dt.date(2015, 3, 28),
                dt.date(2015, 6, 27),
            ],
            [
                dt.date(2015, 1, 28),
                dt.date(2015, 4, 28),
                dt.date(2015, 7, 22),
            ],
            [
                74_599_000_000.0,
                58_010_000_000.0,
                49_605_000_000.0,
            ],
            233_715_000_000.0,
            51_501_000_000.0,
        ),
        (
            "MSFT_FY2020",
            dt.date(2020, 6, 30),
            dt.date(2020, 7, 30),
            [
                dt.date(2019, 9, 30),
                dt.date(2019, 12, 31),
                dt.date(2020, 3, 31),
            ],
            [
                dt.date(2019, 10, 23),
                dt.date(2020, 1, 29),
                dt.date(2020, 4, 29),
            ],
            [
                22_000_000_000.0,
                27_000_000_000.0,
                23_000_000_000.0,
            ],
            100_000_000_000.0,
            28_000_000_000.0,
        ),
        (
            "JNJ_FY_ending_2021_01_03",
            dt.date(2021, 1, 3),
            dt.date(2021, 2, 22),
            [
                dt.date(2020, 3, 29),
                dt.date(2020, 6, 28),
                dt.date(2020, 9, 27),
            ],
            [
                dt.date(2020, 4, 28),
                dt.date(2020, 7, 21),
                dt.date(2020, 10, 20),
            ],
            [
                20_700_000_000.0,
                18_300_000_000.0,
                21_100_000_000.0,
            ],
            82_584_000_000.0,
            22_484_000_000.0,
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_q4_derive_picks_latest_end_among_accepted_date_ties(
    label: str,
    fy_end: dt.date,
    fy_filed: dt.date,
    q_ends: list[dt.date],
    q_filings: list[dt.date],
    q_values: list[float],
    fy_revenue: float,
    expected_q4: float,
) -> None:
    """L-INFRA-013 regression: derive_q4_rows must break accepted_date
    ties by latest end_date, not by stable-sort input order.

    The Q3 10-Q's comparative facts inherit fp=Q3 (the filing's frame)
    and land in the (fy, Q3) bucket alongside the genuine discrete Q3.
    They share the Q3 10-Q's accepted_date. Pre-fix, derive_q4_rows
    sorts by accepted_date alone; ties resolve via stable-sort input
    order, often selecting a phantom null-revenue row as available[-1].
    Q4 then emits with revenue=None (assuming any_derived=True via
    another field that does derive — net_income, in this test setup).

    Fix: sort by (accepted_date, end_date). Latest-end wins among ties.
    Phantom rows are intra-fiscal-year EARLIER periods and always have
    earlier ends than the genuine Q3; the genuine row carries all
    fields populated, so latest-end corrects every derived field at
    once.

    Scenario per parametrization:
    (a) discrete Q1/Q2/Q3 facts from each quarter's own 10-Q seeded
        with BOTH revenue AND net_income, so derive_q4_rows has a
        complete non-null chain on net_income even when the phantom
        wins Q3 for revenue. This triggers any_derived=True and Q4
        emits — with revenue=None because the phantom Q3 row's
        revenue is null. Without the net_income seeding, Q4 wouldn't
        emit pre-fix and the test would fail with "Q4 missing"
        instead of "Q4 revenue is None".
    (b) genuine annual revenue AND net_income from the FY 10-K.
    (c) phantom rows in the (fy, Q3) bucket: NetIncomeLoss at the Q1
        and Q2 ENDS tagged fp=Q3 (the Q3 10-Q's frame). Null revenue,
        non-null net_income. Appended AFTER the discrete Q3 so
        stable-sort places them last, triggering the bug pre-fix.
    """
    security_id = uuid.uuid5(uuid.NAMESPACE_DNS, label)
    facts: list[Fact] = []

    discrete_q_ni = 1_000_000_000.0
    phantom_ni = 2_000_000_000.0
    fy_ni = 10_000_000_000.0

    # (a) Discrete Q1/Q2/Q3 — revenue + net_income.
    quarter_names = ("Q1", "Q2", "Q3")
    for q_end, q_filed, q_val, q_name in zip(
        q_ends, q_filings, q_values, quarter_names, strict=True
    ):
        facts.append(
            _q_fact(
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                end=q_end,
                filed=q_filed,
                value=q_val,
                fp=q_name,
            )
        )
        facts.append(
            _q_fact(
                "NetIncomeLoss",
                end=q_end,
                filed=q_filed,
                value=discrete_q_ni,
                fp=q_name,
            )
        )

    # (b) Genuine annual revenue + net_income.
    facts.append(
        _fy_fact(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            end=fy_end,
            filed=fy_filed,
            value=fy_revenue,
            start=fy_end - dt.timedelta(days=364),
        )
    )
    facts.append(
        _fy_fact(
            "NetIncomeLoss",
            end=fy_end,
            filed=fy_filed,
            value=fy_ni,
            start=fy_end - dt.timedelta(days=364),
        )
    )

    # (c) Phantom rows in the (fy, Q3) bucket from the fp-frame-leak.
    q3_filed = q_filings[2]
    for q_end in q_ends[:2]:
        facts.append(
            _q_fact(
                "NetIncomeLoss",
                end=q_end,
                filed=q3_filed,
                value=phantom_ni,
                fp="Q3",
            )
        )

    out = normalize_to_tables(facts=facts, security_id=security_id)
    inc = out.income_statement
    target_fy = fy_end.year

    # Setup invariant: ≥3 rows in the (fy, Q3) bucket post-normalize.
    q3_rows = inc[(inc["fiscal_year"] == target_fy) & (inc["period"] == "Q3")]
    assert len(q3_rows) >= 3, (
        f"{label}: test setup invariant violated — expected ≥3 Q3 rows "
        f"from the fp-frame-leak, got {len(q3_rows)}."
    )

    # Q4 must derive at FY filing date with the genuine Q3 winning.
    q4_rows = inc[(inc["fiscal_year"] == target_fy) & (inc["period"] == "Q4")]
    assert len(q4_rows) >= 1, f"{label}: Q4 missing for fiscal_year={target_fy}."
    q4_at_fy = q4_rows[q4_rows["accepted_date"] == fy_filed]
    assert len(q4_at_fy) >= 1, (
        f"{label}: no Q4 row at FY filing date {fy_filed}; Q4 is year-lagged."
    )
    q4_revenue = q4_at_fy.iloc[0]["revenue"]
    assert q4_revenue is not None and not pd.isna(q4_revenue), (
        f"{label}: Q4 revenue is None at FY filing date {fy_filed}. "
        f"derive_q4_rows picked a phantom Q3 row with null revenue; "
        f"the (accepted_date, end_date) tie-breaker is not applied."
    )
    assert q4_revenue == pytest.approx(expected_q4, rel=0.001), (
        f"{label}: Q4 revenue {q4_revenue} != FY-Q1-Q2-Q3={expected_q4}."
    )
