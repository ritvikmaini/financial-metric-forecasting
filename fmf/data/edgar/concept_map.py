"""GAAP concept map.

Priority-ordered list of us-gaap concept names per schema field. The
resolver walks the list and picks the first concept that has a fact
matching the requested (end, fp, unit). Different companies and eras use
different concepts for the same economic quantity:

- Revenue: post-ASC-606 filers use RevenueFromContractWithCustomerExcludingAssessedTax;
  banks (JPM) use Revenues; older filers use SalesRevenueNet.
- EPS: EarningsPerShareDiluted is the standard; some older filings
  use IncomeLossFromContinuingOperationsPerDilutedShare.
- Unit gate: eps_diluted must be USD/shares, not USD; revenue must be USD,
  not "pure" or "USD/shares". Wrong-unit matches are silently dropped (L4).

REQUIRED_FIELDS is the set of fields that the anchor-validation gate
checks; every entry MUST have at least one concept that resolves on the
anchor tickers.
"""

from __future__ import annotations

import datetime as dt
from typing import Final

from fmf.data.edgar.companyfacts import Fact

# Field name (mirroring schema.sql column) -> priority-ordered list of
# us-gaap concept names.
#
# Revenue priority: Revenues (the umbrella concept; cleanest for most filers)
# > contract-revenue (post-ASC-606 tech filers like AAPL) > legacy SalesRevenueNet /
# SalesRevenueGoodsNet for pre-ASC-606 filings.
#
# NOTE on bank concepts: RevenuesNetOfInterestExpense is intentionally NOT in
# the default list. Bank revenue handling was decided per ticker at T1 Step 1.
# T1 confirmed Case A for JPM: JPM emits Revenues for FY2023 = 158.104B
# directly, so no fallback to RevenuesNetOfInterestExpense is needed.
# - Case A (JPM emits Revenues = 158.104B): leave the list as-is. [CONFIRMED]
# - Case B (JPM emits ONLY RevenuesNetOfInterestExpense): append it here in T5
#   so the resolver falls through to it. [NOT TRIGGERED]
# - Case C (JPM emits Revenues with a different value): do NOT add bank concept
#   here (it would be unreachable for JPM since Revenues wins). Instead drop
#   JPM revenue from the anchor (see known_financials.json revenue_skip_reason).
#   [NOT TRIGGERED]
CONCEPT_MAP: Final[dict[str, list[str]]] = {
    # Income statement
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        # "RevenuesNetOfInterestExpense",  # uncomment ONLY for Case B per T1 Step 1
    ],
    "gross_profit": [
        "GrossProfit",
    ],
    "ebitda": [
        # No standard us-gaap concept; derived in normalize from
        # operating income + D&A. Kept here as documentation.
    ],
    "ebit": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "eps_diluted": [
        "EarningsPerShareDiluted",
        "IncomeLossFromContinuingOperationsPerDilutedShare",
    ],
    # Balance sheet
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "total_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash_and_equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "Cash",
    ],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "long_term_debt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
    ],
    # Cashflow
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
    "investing_cash_flow": [
        "NetCashProvidedByUsedInInvestingActivities",
    ],
    "financing_cash_flow": [
        "NetCashProvidedByUsedInFinancingActivities",
    ],
    "capital_expenditure": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
    ],
    "free_cash_flow": [
        # Derived (operating_cash_flow - capital_expenditure); no GAAP concept.
    ],
}


# Unit gate per field. None means "no unit check" (anything passes).
_UNIT_GATE: Final[dict[str, set[str]]] = {
    "eps_diluted": {"USD/shares"},
    # Default for all other fields is "USD" (set in resolve_field).
}


# The anchor-validation gate checks these.
REQUIRED_FIELDS: Final[frozenset[str]] = frozenset({"revenue", "net_income", "eps_diluted"})


def _allowed_units(field: str) -> set[str]:
    return _UNIT_GATE.get(field, {"USD"})


def resolve_field(
    facts: list[Fact],
    *,
    field: str,
    end: dt.date,
    fp: str | None,
) -> Fact | None:
    """Walk CONCEPT_MAP[field] in priority order; return the first fact
    that matches (end, fp, allowed unit). None if nothing matches.

    fp=None disables the fp filter. Used by the normalize layer for
    instant (balance-sheet) facts where the row's period is derived from
    end_date matching the fiscal calendar rather than from the fact's
    fp (which is the filing's frame, not the data's fiscal period).
    """
    if field not in CONCEPT_MAP:
        raise KeyError(f"unknown field {field!r}; not in CONCEPT_MAP")
    allowed_units = _allowed_units(field)
    by_concept: dict[str, list[Fact]] = {}
    for f in facts:
        if f.end != end:
            continue
        if fp is not None and f.fp != fp:
            continue
        if f.unit not in allowed_units:
            continue
        by_concept.setdefault(f.concept, []).append(f)
    for concept in CONCEPT_MAP[field]:
        if concept in by_concept:
            # If multiple facts match (different filed dates), return the latest.
            return max(by_concept[concept], key=lambda f: f.filed)
    return None
