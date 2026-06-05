"""Concept map tests.

The concept map resolves a schema field (e.g. 'revenue') to one of
several GAAP concepts in priority order. Companies use different
concepts (JPM uses Revenues, recent tech filers use
RevenueFromContractWithCustomerExcludingAssessedTax, older filers use
SalesRevenueNet). The resolver returns the first concept that has a
matching fact in the input.
"""

from __future__ import annotations

import datetime as dt

from fmf.data.edgar.companyfacts import Fact
from fmf.data.edgar.concept_map import (
    CONCEPT_MAP,
    REQUIRED_FIELDS,
    resolve_field,
)


def _make_fact(concept: str, value: float = 1.0, unit: str = "USD") -> Fact:
    return Fact(
        concept=concept,
        end=dt.date(2023, 12, 31),
        filed=dt.date(2024, 2, 1),
        value=value,
        unit=unit,
        form="10-K",
        fp="FY",
        fy=2023,
    )


def test_revenue_resolves_top_priority_when_present() -> None:
    facts = [
        _make_fact("Revenues", 100.0),
        _make_fact("SalesRevenueNet", 999.0),
    ]
    out = resolve_field(facts, field="revenue", end=dt.date(2023, 12, 31), fp="FY")
    assert out is not None
    assert out.value == 100.0
    assert out.concept == "Revenues"


def test_revenue_falls_back_to_next_concept_when_top_missing() -> None:
    facts = [
        _make_fact("RevenueFromContractWithCustomerExcludingAssessedTax", 200.0),
    ]
    out = resolve_field(facts, field="revenue", end=dt.date(2023, 12, 31), fp="FY")
    assert out is not None
    assert out.value == 200.0


def test_revenue_returns_none_when_nothing_matches() -> None:
    facts = [_make_fact("SomeUnknownConcept", 0.0)]
    out = resolve_field(facts, field="revenue", end=dt.date(2023, 12, 31), fp="FY")
    assert out is None


def test_eps_diluted_resolves_with_correct_unit() -> None:
    facts = [
        _make_fact("EarningsPerShareDiluted", 6.13, unit="USD/shares"),
    ]
    out = resolve_field(facts, field="eps_diluted", end=dt.date(2023, 12, 31), fp="FY")
    assert out is not None
    assert out.value == 6.13


def test_eps_diluted_rejects_wrong_unit() -> None:
    """Wrong unit (USD instead of USD/shares) must NOT resolve. This is
    the unit-scaling defense (L4)."""
    facts = [
        _make_fact("EarningsPerShareDiluted", 6_130_000_000.0, unit="USD"),
    ]
    out = resolve_field(facts, field="eps_diluted", end=dt.date(2023, 12, 31), fp="FY")
    assert out is None


def test_all_required_fields_have_at_least_one_concept() -> None:
    for field in REQUIRED_FIELDS:
        assert CONCEPT_MAP.get(field), f"required field {field!r} has no concept mapping"
        assert isinstance(CONCEPT_MAP[field], list)
        assert len(CONCEPT_MAP[field]) >= 1


def test_concept_map_has_priority_ordering_for_revenue() -> None:
    """Document the priority order so a reviewer can confirm it matches
    the rationale: Revenues > contract-revenue > SalesRevenueNet.
    """
    expected_prefix = [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ]
    assert CONCEPT_MAP["revenue"][: len(expected_prefix)] == expected_prefix
