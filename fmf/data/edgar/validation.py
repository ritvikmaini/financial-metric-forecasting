"""Anchor-validation gate.

Loads tests/fixtures/known_financials.json and asserts that an ingested
DuckDB has the right values for the 5 anchor tickers. Used both by
scripts/build_fixture.py (build-time gate) and by
tests/data/test_anchor_validation.py (committed-fixture regression gate).

If concept-map resolution silently picks a wrong GAAP tag, the resulting
value will be off by orders of magnitude (e.g. $383 vs $383,285,000,000)
and the gate will fail loudly with the offending ticker and field named.

Per-field optionality: an anchor may skip a field (e.g., JPM in Case C
of the T1 bank-concept tree skips revenue). None means "do not validate
this field for this anchor; the JSON must also carry a _skip_reason."
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb


class AnchorValidationError(AssertionError):
    """One or more anchor values failed the validation gate."""


@dataclass(frozen=True, slots=True)
class Anchor:
    ticker: str
    cik: str
    fiscal_year: int
    fp: str
    revenue_usd: float | None
    net_income_usd: float | None
    eps_diluted: float | None


@dataclass(frozen=True, slots=True)
class KnownFinancials:
    anchors: list[Anchor]
    tolerance_revenue_pct: float
    tolerance_net_income_pct: float
    tolerance_eps_pct: float


def _maybe_float(v: object) -> float | None:
    return None if v is None else float(v)  # type: ignore[arg-type]


def load_known_financials(path: Path) -> KnownFinancials:
    raw = json.loads(path.read_text(encoding="utf-8"))
    tol = raw["tolerance"]
    anchors: list[Anchor] = []
    for a in raw["anchors"]:
        if a.get("revenue_usd") is None and "revenue_skip_reason" not in a:
            raise ValueError(f"{a['ticker']}: revenue_usd is null but no revenue_skip_reason given")
        if a.get("net_income_usd") is None and "net_income_skip_reason" not in a:
            raise ValueError(
                f"{a['ticker']}: net_income_usd is null but no net_income_skip_reason given"
            )
        if a.get("eps_diluted") is None and "eps_diluted_skip_reason" not in a:
            raise ValueError(
                f"{a['ticker']}: eps_diluted is null but no eps_diluted_skip_reason given"
            )
        anchors.append(
            Anchor(
                ticker=a["ticker"],
                cik=a["cik"],
                fiscal_year=int(a["fiscal_year"]),
                fp=a["fp"],
                revenue_usd=_maybe_float(a.get("revenue_usd")),
                net_income_usd=_maybe_float(a.get("net_income_usd")),
                eps_diluted=_maybe_float(a.get("eps_diluted")),
            )
        )
    return KnownFinancials(
        anchors=anchors,
        tolerance_revenue_pct=float(tol["revenue_usd_pct"]),
        tolerance_net_income_pct=float(tol["net_income_usd_pct"]),
        tolerance_eps_pct=float(tol["eps_diluted_pct"]),
    )


def _within_tol(actual: float, expected: float, pct: float) -> bool:
    if expected == 0.0:
        return abs(actual) <= pct
    return abs(actual - expected) / abs(expected) <= pct


def validate_anchors(conn: duckdb.DuckDBPyConnection, truth: KnownFinancials) -> None:
    """Raise AnchorValidationError if any anchor's resolved value is off.

    Fields set to None in the anchor (per the load gate's _skip_reason
    requirement) are skipped — these are deliberate skips, not bugs.
    """
    failures: list[str] = []
    for a in truth.anchors:
        row = conn.execute(
            'SELECT "revenue", "net_income", "eps_diluted" '
            'FROM "income_statement" i JOIN "securities" s ON i.security_id = s.security_id '
            "WHERE s.cik = ? AND i.fiscal_year = ? AND i.period = ? "
            "ORDER BY i.accepted_date DESC LIMIT 1",
            [a.cik, a.fiscal_year, a.fp],
        ).fetchone()
        if row is None:
            failures.append(f"{a.ticker}: no row for fy={a.fiscal_year} fp={a.fp}")
            continue
        rev, ni, eps = row
        if a.revenue_usd is not None and (
            rev is None or not _within_tol(float(rev), a.revenue_usd, truth.tolerance_revenue_pct)
        ):
            failures.append(
                f"{a.ticker} fy{a.fiscal_year} revenue: got {rev!r}, expected ~{a.revenue_usd}"
            )
        if a.net_income_usd is not None and (
            ni is None
            or not _within_tol(float(ni), a.net_income_usd, truth.tolerance_net_income_pct)
        ):
            failures.append(
                f"{a.ticker} fy{a.fiscal_year} net_income: got {ni!r}, expected ~{a.net_income_usd}"
            )
        if a.eps_diluted is not None and (
            eps is None or not _within_tol(float(eps), a.eps_diluted, truth.tolerance_eps_pct)
        ):
            failures.append(
                f"{a.ticker} fy{a.fiscal_year} eps_diluted: got {eps!r}, expected ~{a.eps_diluted}"
            )
    if failures:
        msg = "anchor validation failed:\n  - " + "\n  - ".join(failures)
        raise AnchorValidationError(msg)
