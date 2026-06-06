# FMF Learnings Ledger

Empirical findings and ported methodology from the fmf-public project. Each entry captures either:
- A `[methodology, ported]` claim: data-independent methodology, inherited from the prior FMF work on proprietary data, valid as-is here.
- A public-data-reproduced finding: a real number measured on the public-data substrate, with the configuration and commit hash that produced it.

Numbering is per section: `L-LGBM-NNN`, `L-TIREX-NNN`, `L-BLEND-NNN`, `L-FEAT-NNN`, `L-EVAL-NNN`, `L-INFRA-NNN`. IDs are stable as new entries are appended.

All paths are relative to repo root.

---

## 6. Infrastructure

### L-INFRA-001 — Anchor-validation pattern for SEC ingest

**Tag:** `[methodology, ported]`

**Claim:** A small set of anchor tickers (5: AAPL, MSFT, GOOGL, JNJ, JPM) with hand-verified fiscal-year revenue, net income, and diluted EPS, stored in `tests/fixtures/known_financials.json` with explicit `fiscal_year_end` per ticker and the resolving concept named per anchor, makes wrong concept-map resolution a build-time error rather than a downstream silent bug. The gate fires both at fixture-build time (in `scripts/build_fixture.py`) and on the committed fixture (in `tests/data/test_anchor_validation.py`). Required for any SEC-XBRL ingest pipeline; the alternative is silent misalignment between GAAP tags and target fields.

**Mechanism:** Each XBRL fact carries a `unit` (USD, USD/shares, etc.) and a `fp/form/fy` triple. The concept map is a priority-ordered list per field (not 1:1, since `Revenues` vs `RevenueFromContractWithCustomerExcludingAssessedTax` vs `SalesRevenueNet` vary by company and era). Anchor validation compares the resolved value against the published figure with a tight tolerance (0.5%); concept-map mis-resolution typically overshoots by orders of magnitude (1e3 wrong unit; 1e9 different revenue concept on a bank), well outside the band.

**Per-field skip semantics:** An anchor may set a field to `null` plus a documented `_skip_reason` to opt out of validation for that field (e.g., JPM's revenue when the bank-concept fallback is unreachable). The loader hard-fails if a `null` is provided without a `_skip_reason`, surfacing the deliberate skip to any reviewer.

**Source:**
- `fmf/data/edgar/validation.py` (commit `205b705`)
- `tests/fixtures/known_financials.json` (commit `bf5866e`)
- `scripts/build_fixture.py` (commit `0bd66f0`)
- Fixture committed in commit `0b34f43` passes the gate on AAPL/MSFT/GOOGL/JNJ/JPM FY2023 within 0.5% tolerance.

**Date:** 2026-06-06

**Status:** Ported from Bavest FMF methodology; the public-data version anchors on EDGAR's `companyfacts` API and fiscal-frame-aware truth.

### L-INFRA-002 — Fiscal-year derivation must be FY-end-aware, not calendar-year

**Tag:** Public-data finding

**Claim:** For companies whose fiscal year does not align with the calendar year (AAPL's FY ends late September; MSFT's ends June; SNOW's ends January), deriving `fiscal_year = end.year` mislabels early fiscal quarters by one year. AAPL's Q1 FY2010 ends Dec 26, 2009 (calendar 2009) but Apple's own numbering puts it in FY2010. Trusting `end.year` causes (a) downstream join/labeling mismatches with the company's reporting, and (b) a primary-key collision in `income_statement` / `balance_sheet` / `cashflow` when two facts with different end dates collapse onto the same `(security_id, fiscal_year, period, accepted_date)`.

**Mechanism:** Derive fiscal_year from the company's own FY-end calendar (extracted from annual `Revenues` / `RevenueFromContractWithCustomerExcludingAssessedTax` / `SalesRevenueNet` facts with `fp=FY, form=10-K`): a fact belongs to the fiscal year whose FY-end is the next FY-end at or after the fact's end date. Falls back to Q3-end + 91 days if no annual facts; falls back to calendar-year if no fiscal-period facts at all.

**Schema implication:** The PIT-table PKs include `end_date` so that legitimately-distinct period-ends within the same accepted_date (e.g., comparative balance sheet at prior-FY-end vs current-period balance sheet, both filed the same day) can both coexist without colliding.

**Source:**
- Schema amendment: `fmf/data/schema.sql` (commit `dd3f39f`)
- Normalize fix: `fmf/data/edgar/normalize.py::_compute_fy_end_dates`, `_derive_fiscal_year`, `_classify_instant_end` (commit `933dfe9`)
- Live verification: AAPL FY2022 revenue resolves to 394.328B in the committed fixture (commit `0b34f43`); regression-protected by `tests/data/test_fixture_integrity.py::test_aapl_fy2022_revenue_resolves_correctly`.

**Date:** 2026-06-06

**Status:** Reproduced on public data.

### L-INFRA-003 — JPM emits `Revenues` only as an annual fact; quarterly Revenues coverage is 0%

**Tag:** Public-data finding

**Claim:** Bank 10-Q filings tag interest-income components separately rather than emitting a single `Revenues` fact for the quarter. JPM's FY2018-FY2025 `Revenues` coverage is 100% on FY rows and 0% on Q1/Q2/Q3/Q4 rows. The same pattern is expected for other banks. Downstream consumers needing quarterly bank revenue must either derive it from interest-income concepts (`InterestAndDividendIncomeOperating`, etc.) or accept the coverage gap.

**Mechanism:** Surfaced by `fmf/features/audit/coverage.py::compute_coverage` on the committed fixture. Result is stable across the fiscal-year window.

**Source:**
- Coverage scan: `fmf/features/audit/coverage.py` (commit `47b85c7`)
- Spot-check query in T10 confirmed: JPM `(symbol='JPM', period='FY') → Revenues present 100%`, `(symbol='JPM', period in (Q1,Q2,Q3,Q4)) → Revenues present 0%` across FY2018-FY2025.

**Date:** 2026-06-06

**Status:** Reproduced on public data; backlog item B-NEW-001 — derive bank quarterly revenue from interest-income components.

### L-INFRA-004 — EDGAR companyfacts emits both discrete and YTD twins for flow concepts

**Tag:** `[methodology, ported]`

**Claim:** For flow concepts (revenue, net income, EPS, cashflows) in Q2 and Q3, EDGAR emits BOTH the discrete-quarter fact (start = quarter-start) AND the year-to-date cumulative fact (start = fiscal-year-start) under the same (end, fp, form, filed) tuple. Without disambiguation, the resolver arbitrarily picks one and the discrete vs YTD ambiguity corrupts the quarterly series. Q4 derivation specifically: `Q4 = FY - (Q1 + Q2 + Q3)` produces garbage if Q2/Q3 are YTD aggregates.

**Mechanism:** `normalize._duration_matches_period` rejects facts whose `(end - start)` duration is outside the period's expected window (60–100 days for Q1/Q2/Q3, 340–380 days for FY). Discrete quarters pass; YTD aggregates are filtered. Instant concepts (balance sheet, start=None) pass through unchanged. AAPL Q2 FY2023 in the committed sample has both discrete (start=2023-01-01, end=2023-04-01) and YTD (start=2022-09-25, end=2023-04-01) facts — only the discrete survives normalize.

**Source:**
- Duration filter: `fmf/data/edgar/normalize.py::_duration_matches_period` (commit `e025e3c`)
- Test guard: `tests/data/edgar/test_normalize.py::test_q2_discrete_picked_over_ytd_twin` (commit `e025e3c`)

**Date:** 2026-06-06

**Status:** Reproduced on public data; load-bearing for the Q4 PIT-correct derivation.
