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

### L-INFRA-005 — Consensus from yfinance is a snapshot, not history

**Tag:** `[methodology, ported]`

**Claim:** yfinance exposes only current EPS / revenue estimates per period (`earnings_estimate`, `revenue_estimate` indexes are relative labels `0q`, `+1q`, `0y`, `+1y`). Historical revisions are not available. We store each pull as a row with `pulled_at = now()`; the PIT proxy is `pulled_at <= as_of_date`, treating the snapshot as if it were known at the time it was pulled. This is weaker than IBES-style historical consensus and flows into the README's honest-framing note: the benchmark leans on naive and statistical baselines (random walk, AR(1), seasonal naive, last-year), with consensus as a caveated secondary reference.

**Mechanism:** `fmf/data/yfinance/consensus.py::ingest_consensus_snapshot` records `pulled_at` per row. `_period_label_to_target_date` anchors labels to `pulled_at.date()` using calendar quarter / year ends (intentionally not fiscal-calendar-aware; yfinance's labels are calendar-quarter-based even for non-calendar filers). Mapping is regression-tested: `0q → end-of-current-quarter`, `+1q → end-of-next-quarter`, `0y → Dec 31 anchor.year`, `+1y → Dec 31 anchor.year + 1`.

**Source:**
- `fmf/data/yfinance/consensus.py` (commit `6e56b1c`)
- `tests/data/yfinance/test_consensus.py::test_period_label_mapping_anchored_to_pulled_at` (commit `6e56b1c`)
- Sample data: AAPL_earnings_estimate.csv and AAPL_revenue_estimate.csv carry 4 rows each, populating 4 EPS + 4 revenue estimate rows per pull.

**Date:** 2026-06-06

**Status:** Ported limitation; documented in module docstring + will be carried into the README.

### L-INFRA-006 — Yahoo Finance returns split-adjusted OHLC at source; auto_adjust=False only toggles dividends

**Tag:** Public-data finding

**Claim:** **Every yfinance path returns split-adjusted Close on the wire**, regardless of which retrieval function or flags are used. `yf.download(..., auto_adjust=False)`, `yf.download(..., auto_adjust=False, multi_level_index=False)`, `yf.Ticker.history(..., auto_adjust=False, back_adjust=False)`, and the raw chart endpoint all return Close=$43.33 on 2019-06-03 for AAPL — already split-adjusted for the Aug 31, 2020 4:1 split. The `auto_adjust` flag only toggles the dividend layer; split adjustment is baked into Yahoo's stored series. Truly raw historical Close (~$173 on 2019-06-03) only exists by un-applying splits post-hoc.

**Mechanism:** `fmf/data/yfinance/_client.py::fetch_prices` live path uses `yf.Ticker(ticker).history(auto_adjust=False, back_adjust=False)` then iterates the splits index. For each row, multiplies Open/High/Low/Close by `prod(splits with split_date STRICTLY > row_date)` and divides Volume by the same factor. `Adj Close` is left untouched (Yahoo's back-adjusted-for-dividends-and-splits reference). **Critical correctness detail:** date-only comparison when matching splits to row dates. yfinance's split index carries intraday timestamps (09:30 EDT) while the row index is midnight; naive timestamp comparison falsely lumps a split-effective date with its pre-split history and corrupts the un-split factor.

**Verification:** Spot-check from the committed fixture:
- AAPL 2019-06-03: close=173.30, adj_close=41.45, ratio=4.18 (matches public reference for the actual closing price)
- AAPL 2020-08-28 (last day before 4:1 split): close=499.23 (matches public reference)
- AAPL 2020-08-31 (split-effective day): close=129.04 (matches public reference)

**Regression test:** `tests/data/yfinance/test_prices.py::test_auto_adjust_false_close_differs_from_adj_close` asserts `close > adj_close * 3.5` on AAPL 2019-06-03. Will catch silent regressions in either the yfinance behavior or our un-split transform.

**Source:**
- `fmf/data/yfinance/_client.py` un-split implementation (commit `2656fe0`)
- `tests/fixtures/sample_yfinance/AAPL_prices.csv` regenerated with verified raw Close (commit `2656fe0`)
- `tests/data/yfinance/test_prices.py` (commit `36cbdb0`)

**Date:** 2026-06-06

**Status:** Reproduced on public data.

### L-INFRA-007 — Securities metadata is an UPDATE, not an INSERT, in a multi-source ingest

**Tag:** `[methodology, ported]`

**Claim:** When a securities row already exists from an earlier source (S2 EDGAR ingest creates rows by `(security_id, symbol, cik)`), subsequent sources should UPDATE the existing row by primary-key match, not INSERT a new one. This avoids duplicate rows that would corrupt JOINs across data layers and double-count tickers in coverage scans. `fmf/data/yfinance/securities.py::update_securities_metadata` does `UPDATE securities SET ... WHERE cik = ?` and tolerates missing fields in `yf.Ticker.info` (one bad field doesn't fail the whole row; one bad ticker doesn't fail the whole run).

**Mechanism:** Builds the SET clause dynamically from whichever info fields are present (sector, industry, country, exchange). If `fetch_info` raises (missing fixture, network error, ticker delisted), logs a warning and returns without touching the row.

**Verification:** T7 live ingest populated all 9 anchor tickers with sector + country + exchange:
- AAPL/MSFT: Technology / United States / NMS
- GOOGL: Communication Services / United States / NMS
- JNJ/ZTS: Healthcare / United States / NYQ
- JPM: Financial Services / United States / NYQ
- HSY: Consumer Defensive / United States / NYQ
- GWW: Industrials / United States / NYQ
- SNOW: Technology / United States / NYQ

Row count post-augment: 9 securities (matches pre-augment; no duplicates).

**Source:**
- `fmf/data/yfinance/securities.py` (commit `9115b98`)
- `tests/data/yfinance/test_securities.py::test_update_does_not_insert_duplicate` (commit `9115b98`)

**Date:** 2026-06-06

**Status:** Ported pattern; reproduced on public data.
