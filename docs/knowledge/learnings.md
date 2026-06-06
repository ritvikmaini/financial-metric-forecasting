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

### L-INFRA-012 — Non-calendar-FY 10-K emits quarterly comparatives tagged fp=FY; FY-end determination needs a duration+start gate

**Tag:** `[public-data finding]`

**Claim:** Non-calendar-fiscal-year filers (AAPL, MSFT, JNJ, SNOW in our universe) emit each 10-K's quarterly comparative pieces (Q1/Q2/Q3 of the fiscal year being reported, ~90 days each) with `fp='FY'` in the XBRL feed, because those quarters ARE within the fiscal year being reported. A naive FY-end determination that takes `max(end)` over all fp=FY 10-K facts per calendar year picks the latest quarter end_date in that calendar year — for AAPL, the Q1-of-next-fiscal-year end (Dec) instead of the genuine FY-end (Sept). The wrong FY-end cascades through `_derive_fiscal_year`, mislabeling Q1 facts into the prior fiscal_year, and `derive_q4_rows` then can't find Q1/Q2/Q3 in the target FY bucket at the FY filing's accepted_date, so Q4 only emerges via the NEXT FY's comparatives — a year-lagged Q4.

The fix is a duration-and-start gate on which facts contribute to FY-end determination: `start is not None` (excludes balance-sheet instants, which have no duration to disambiguate and can carry fp=FY at the FY-end too) AND `340 <= (end - start + 1) <= 380` (the existing `_ANNUAL_DAYS_MIN/MAX` window, restricting to genuine annual flow facts). After the fix, only the real ~365-day annual fact contributes per calendar year, FY-end dates are correct, Q1/Q2/Q3 are correctly labeled, and Q4 derives contemporaneously with the FY filing.

**Scope:** Non-calendar-FY tickers across their year-lagged ranges. From T3a artifact 2:
- AAPL: year-lagged FY2009-2019, contemporaneous FY2021+.
- MSFT: year-lagged FY2010-2024.
- JNJ: year-lagged FY2010-2023.
- SNOW: year-lagged FY2020-2023.

Calendar-FY tickers (ZTS, GWW, HSY, JPM, GOOGL) are unaffected because the annual's December end is naturally the latest in its calendar year, so even with quarterly comparatives polluting the fp=FY pool, max(end) per calendar year still resolves to the correct annual end.

**Source:** `fmf/data/edgar/normalize.py::_compute_fy_end_dates` (fix commit `f2d1e28`). Tested at `tests/data/edgar/test_normalize.py::test_q4_derives_at_fy_filing_for_non_calendar_fy` parametrizations `AAPL_FY2015`, `MSFT_FY2020`, `JNJ_FY_ending_2021_01_03` (synthetic-regression commit `f466681`) and `::test_q4_fixture_regression_emits_at_fy_filing` parametrizations `AAPL/2015`, `MSFT/2020` (fixture-regression commit `1fd89d7`).

**Diagnostic artifacts:** `reports/aapl_fy2015_q4_diagnosis.txt` (root-cause investigation), `reports/quarterly_period_coverage.txt` (post-fix coverage map across 9 tickers).

### L-INFRA-013 — derive_q4_rows accepted_date ties broken by latest end_date; coverage tool must dedup phantoms

**Tag:** `[public-data finding]`

**Claim:** After the L-INFRA-012 fix, calendar-FY filers (GWW) regressed from contemporaneous to year-lagged Q4 in fiscal_years FY2019-2025. Root cause: a Q3 10-Q's comparative facts inherit `fp='Q3'` (the filing's frame) and land in the `(fy, Q3)` bucket alongside the genuine discrete Q3. They share the Q3 10-Q's `accepted_date`. Pre-fix, `derive_q4_rows` sorts by `accepted_date` alone; ties resolve via stable-sort input order, often selecting a phantom Q3 row with null revenue as `available[-1]`. Q4 then emits with `revenue=None` (when `any_derived=True` via another field that does derive — e.g., net_income), suppressing the Q4 row at the FY filing date. The next FY's comparatives later supply the missing Q4, year-lagged.

The fix sorts by `(accepted_date, end_date)` so the latest-end row wins among ties. Phantom rows in a (fy, Q3) bucket are intra-fiscal-year EARLIER periods (e.g., Q1/Q2 ends tagged fp=Q3 by the comparative-leak); they always have earlier ends than the genuine Q3. The genuine row carries all fields populated, so latest-end corrects every derived field at once.

**Secondary finding — coverage tool was phantom-blind.** `fmf/features/audit/coverage.py::compute_coverage` counted every row including phantoms, inflating denominators and depressing per-ticker coverage_pct. Fix: dedup to one row per `(security_id, fiscal_year, period)` via `QUALIFY ROW_NUMBER() OVER (PARTITION BY security_id, fiscal_year, period ORDER BY end_date DESC, accepted_date DESC) = 1` BEFORE counting non-null per column. Post-fix raw coverage on income_statement (685 deduped rows vs 1587 raw): revenue 92.8% (was 90.4%), gross_profit 60.3% (was 59.1%), ebit 89.6% (was 85.1%), eps_diluted 69.5% (was 64.1%) — phantoms suppressed every non-null %; eps_diluted, ebit, revenue moved up most.

**Scope:** Calendar-FY filers whose Q3 10-Q emits comparative facts at earlier-quarter ends with fp=Q3. From `reports/quarterly_period_coverage.txt` post-rebuild:
- GWW FY2019-2025: Q4 contemporaneous, all at the FY 10-K accepted_date (2020-02-20, 2021-02-24, 2022-02-23, 2023-02-21, 2024-02-22, 2025-02-20, 2026-02-19). Pre-fix these were +363–+370 day lagged.
- MSFT FY2016 also flipped contemporaneous (was year-lagged in the L-INFRA-012 rebuild report).
- AAPL FY2015 remained contemporaneous (already fixed by L-INFRA-012; the synthetic test confirms the tie-breaker doesn't regress non-calendar-FY cases).

**Not a regression — L-INFRA-003 zero quarterly Revenues:** JPM Q4 FY2015-2025 remains null. The post-rebuild probe shows JPM has `Q1-Q3 non-null count = 0` in those fiscal_years — JPM emits Revenues only as FY facts after the period change (banks tag interest-income components separately), so there is nothing to derive Q4 revenue from. This is the L-INFRA-003 condition surfaced via the audit, NOT a fix failure. JPM is deliberately excluded from the fixture-regression test.

**Source:** `fmf/data/edgar/normalize.py::derive_q4_rows` (fix commit `545db81`) and `fmf/features/audit/coverage.py::compute_coverage` (coverage-tool fix commit `39478de`). Tested at:
- `tests/data/edgar/test_normalize.py::test_q4_derive_picks_latest_end_among_accepted_date_ties` parametrizations `MSFT_FY2016`, `GWW_FY2020` (synthetic-regression commit `4e777b3`).
- `tests/data/edgar/test_normalize.py::test_q4_fixture_regression_emits_at_fy_filing` parametrization `GWW/2020` added alongside the existing `AAPL/2015`, `MSFT/2020` from L-INFRA-012 (fixture+rebuild commit `1463970`).
- `tests/features/audit/test_coverage.py::test_compute_coverage_dedups_phantoms_one_row_per_security_fy_period` (asserts one row per (symbol, fiscal_year, period) post-dedup).

**Diagnostic artifacts:** `reports/raw_column_coverage.txt` and `reports/quarterly_period_coverage.txt` re-measured with the phantom-aware coverage tool (re-measure commit `353ae96`).

### L-INFRA-014 — Field-level PIT for partial re-disclosures

**Tag:** `[methodology, ported]` (the field-level PIT pattern) + `[public-data finding]` (the partial-re-disclosure shape)

**Claim:** This is the read-path counterpart to the ingest-side fixes L-INFRA-012 (FY-end determination) and L-INFRA-013 (Q-bucket tie-breaker). The fixture data was correct all along — the original 10-K's full balance sheet (total_assets, total_liabilities, current_assets/liabilities, long_term_debt, etc.) is in the table at its original accepted_date. The bug was in the PIT primitive: `fetch_pit_series`'s row-level dedup (`QUALIFY ROW_NUMBER() OVER (PARTITION BY fy, period, end_date ORDER BY accepted_date DESC) = 1`) selected a later partial re-disclosure of the same period (a 10-K's selected-data table or multi-year rollforward that re-mentions an old fiscal year with only cash + equity), and nulled out fields the original filing carried but the partial re-disclosure didn't touch. A backtest at any as_of after the partial then read, e.g., AAPL FY2015 total_assets as null when the correct point-in-time value is the ~290B from the 2015 10-K.

The correct PIT semantic is field-level: for each field, the latest non-null value among rows with `accepted_date <= as_of` within the same `(fiscal_year, period, end_date)` group. A partial re-disclosure updates only the fields it contains; a field it omits retains its last-known value; a field it genuinely restates to a new non-null value updates. The synthesized row's `accepted_date` column is the MAX in the group up to `as_of` (provenance: latest restatement that contributed to any field). Visibility is gated per fact by the `WHERE accepted_date <= as_of` filter, so the row appears as soon as ANY contributing fact is visible — the cardinal 1-day-shift test still asserts visibility via the MIN accepted_date pulled from raw data, which is unchanged.

`compute_coverage` uses the same field-level path via the shared helper `_field_level_pit_select_sql` so coverage = what features deliver. The L-INFRA-013 coverage tool's "phantom-aware via latest end_date" partition is preserved (genuine quarter wins over earlier-end phantoms); L-INFRA-014 layers the field-level assembly inside it.

**Scope:** Surfaces on every PIT-table field that can be partially re-disclosed. Confirmed empirically on AAPL/MSFT balance-sheet fields where the row-level dedup depressed `total_assets`/`total_liabilities`/`current_assets`/`current_liabilities`/`long_term_debt` to <5% coverage. After field-level assembly these fields rise to plausible levels matching companyfacts emission rates (total_assets 14.4%→93.2%, total_liabilities 3.8%→79.6%, current_assets 3.3%→80.8%, current_liabilities 3.3%→80.8%, long_term_debt 2.9%→73.7%). HSY `eps_diluted` lifts from 3%→7%. The diagnostic that surfaced the bug is in `reports/balance_sheet_coverage_diagnostic.txt`, `reports/balance_sheet_concept_probe.txt`, `reports/instant_period_classification_probe.txt`.

**Anchor values unchanged:** AAPL FY2023 TTM = 383,285,000,000 (exact match, within 0.5% anchor tolerance). AAPL FY2023 YoY = -0.0280 (exact). AAPL 2015 gross margin = 0.4078 (within 0.30–0.50 band). Field-level fix only adds coverage by including previously-thinned-out fields; previously-correct derived values do not shift on this fixture.

**TDD note:** synthetic in-memory test `test_pit_series_partial_redisclosure_retains_omitted_field` failed pre-fix with `revenue=NaN` (expected 100B); confirmed RED. The reframed test `test_pit_series_assembles_each_field_at_latest_non_null` against AAPL `income_statement` passed pre-fix — AAPL's income_statement happens to have no partial-re-disclosure pattern; only `balance_sheet` does. The reframed test still serves as a post-fix invariant guard. The synthetic test carries the mechanism's RED-proof.

**Source:** `fmf/features/point_in_time.py::fetch_pit_series` + `_field_level_pit_select_sql` helper (fix commit `7bbeab4`). `fmf/features/audit/coverage.py::compute_coverage` (commit `35ddd6b`). Tested at `tests/features/test_point_in_time.py::test_pit_series_partial_redisclosure_retains_omitted_field` (constructed in-memory, partial restatement of one field) and `::test_pit_series_assembles_each_field_at_latest_non_null` (re-checked against the real fixture). TDD red commit `660da09`. Re-measure commit `8efde18`.

**Note for T3b:** GOOGL emits `CostOfRevenue` (142 USD facts) + `Revenues`/`RevenueFromContractWithCustomerExcludingAssessedTax` but does NOT emit `GrossProfit`. JPM emits neither `GrossProfit` nor `CostOfRevenue` (bank). Deriving `gross_profit = revenue - CostOfRevenue` in normalize when `GrossProfit` is untagged would make GOOGL computable while JPM remains correctly dropped, and lifts `gross_profit` coverage. Deferred — normalize change with its own surface; surface in T3b registry design.
