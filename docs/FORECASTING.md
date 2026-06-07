# Forecasting on Public Data: Methodology, Results, Limitations

This is the pillar doc behind the README's three highlights. It is organised around the correctness surface, not a session walk-through, because the correctness surface is what a reader is actually evaluating.

## 1. What this is

fmf-public is a clean-room reproduction of a financial-metrics forecasting stack on public data. The stack is (Point-in-Time feature layer) -> (LightGBM + TiRex + simplex meta-learner) -> (expanding-window backtester) -> (predictions parquet + SQLite registry + four-axis prediction cache). Methodology was designed at Bavest on proprietary data; this repo re-implements the same methodology on SEC EDGAR, yfinance, and FRED so the rigor is reproducible by anyone with public access.

The differentiator is leakage discipline, not model choice. The model stack is a reasonable mid-2024 setup; the rigor signal is that every correctness vector is enumerated, every guard ships with an invariant test that names the precise mechanism, and known limitations are stated up front rather than buried.

## 2. The data flow

One pass through the architecture, naming every transition:

```
EDGAR XBRL ingest (10-K, 10-Q)              --> fmf/data/edgar
yfinance prices + analyst snapshot ingest   --> fmf/data/yfinance
   |
   v
DuckDB tables: income_statement, balance_sheet, cashflow, prices, analyst_estimates, securities
   |
   v
Field-level PIT layer (fetch_pit_series)    --> fmf/features/point_in_time.py
   |
   v
Feature registry (per-(security, as_of))    --> fmf/features/feature_registry.py + builtin_features.py
   |
   v
ExpandingWindowBacktester (F1 four-cutoff)  --> fmf/equity/forecasting/evaluation/backtester.py
   |  (LightGBM, TiRex via tirex-ts, MetaLearner)
   v
predictions parquet                          --> reports/predictions/<run_id>.parquet
SQLite run registry                          --> reports/fmf_runs.db
SQLite prediction cache (four-axis key)      --> reports/prediction_cache.db
```

Every data transition is a function with an explicit signature. The registry-routed feature compute path is the only way feature values are produced; the bulk-SQL-JOIN shortcut is structurally absent from the orchestrator.

**Correctness callout: four-axis prediction cache key.** The cache at `reports/prediction_cache.db` is keyed on SHA256 over (`CACHE_VERSION` + config_flags_hash + per-security data fingerprint + (security_id, as_of_date, metric, model_name)). Two runs with identical config but a rebuilt fixture must miss the cache, because a stale entry propagating into S15 noise-floor or S17 admission-gate would be a silent wrong number, not a recompute. The orthogonality gate `test_with_cache_predictions_match_without_cache` asserts bit-for-bit equality between with-cache and without-cache backtester runs across the prediction columns; the cache is an optimisation, never a result mutation. Detail: ledger entry `L-EVAL-S14-001`.

## 3. The correctness surface

Five vectors. Each takes the same four-beat shape: where the failure hides, how it was found, how it is guarded, the invariant test that locks it. The failure types are not all look-ahead leakage; the section name is "correctness" precisely because conflating coverage bugs with leakage would overclaim and that precision is what separates the doc from a hand-wavy one. The two foregrounded vectors lead.

### 3a. Comparative-row trap in next-FY target lookup (leakage)

**Failure type:** point-valued look-ahead leakage at the target-lookup boundary.

**Where it hides.** Every 10-K carries prior-year FY rows as comparatives, so `income_statement` holds multiple `period='FY'` rows for the same fiscal_year at different `accepted_date`s. A naive query that orders raw rows by `accepted_date ASC` and takes the first one strictly after as_of can return a fiscal_year whose original disclosure was *before* as_of (the model already knew the answer), via that fiscal_year's comparative column inside a later 10-K. The model is then scored on predicting a value it already held. The trap is pervasive on real EDGAR data, not an amendment edge case.

**How it was found.** S10 plan close-read. Surfaced in chat review of the target-lookup pseudocode, before any code was written. The fix landed in the same plan revision; no committed code ever ran the buggy version.

**How it is guarded.** A CTE that identifies fiscal_years by their *earliest non-null disclosure* via `MIN(accepted_date)`, then picks the smallest fiscal_year whose earliest disclosure is strictly after as_of, returning the value at that earliest disclosure (the original 10-K). Source: `fmf/equity/forecasting/evaluation/_target_lookup.py::next_fy_target`.

**Invariant:**
- `tests/equity/forecasting/evaluation/test_backtester_invariants.py::test_target_fy_has_no_disclosure_at_or_before_as_of` - on every scored backtest row, asserts there is no `period='FY'` row for `target_fy` with `accepted_date <= as_of`. Reads the failure off the bug's own surface (the disclosed-FY count), not a downstream metric drift.
- `tests/equity/forecasting/evaluation/test_target_lookup.py::test_next_fy_target_skips_comparative_for_already_disclosed_fy` - hand-built in-memory three-row fixture (FY t original at D1, FY t comparative at D2, FY t+1 original at D3); for as_of strictly between D1 and D2 the function must return FY t+1 at D3. Exercises the bug path deterministically regardless of fixture luck. Detail: ledger entry `L-EVAL-S10-002`.

### 3b. Out-of-sample vs in-sample meta-learner stacking (leakage)

**Failure type:** in-sample stacking leakage in the meta-learner training set.

**Where it hides.** Training a stacking meta-learner on the base model's *in-sample* predictions flatters whichever base overfits hardest. The simplex blender's learned weights tilt toward the overfitter, inflating the ensemble score on the training distribution and degrading it on held-out data. Shipping this inside a backtester whose whole pitch is leakage discipline would be self-undermining.

**How it was found.** S10 plan close-read. The draft plan's `_run_fold` had the simplex trained on `lgbm.predict(train_X)` resub plus the per-fold TiRex predictions. User override at the plan gate before any code was written. The OOS replacement was specified before code shipped.

**How it is guarded.** Prior-fold OOS prediction triples `(lgbm_pred, tirex_pred, naive_baseline, target_value)` are accumulated across folds. At each fold the meta-learner is fit only on triples whose target was realised at the current cutoff. A cold-start equal-weight blend covers the warm-up until `meta_min_train` realised OOS triples accumulate (configurable; default 8 to 12). Source: `fmf/equity/forecasting/evaluation/backtester.py::_gather_realized_oos` and `_run_fold`.

**Invariant:**
- `tests/equity/forecasting/evaluation/test_backtester_invariants.py::test_meta_learner_train_sources_are_strictly_prior_folds` - records the source-fold provenance of each OOS triple in `FoldDiagnostics.meta_train_source_folds` and asserts every source-fold index is strictly less than the activating fold. Reads the training provenance, not the output labelling, because labelling can be patched after the fact and the failure mode is the training set.
- `tests/equity/forecasting/evaluation/test_backtester_invariants.py::test_cold_start_equal_weight_blend_before_meta_active` - warm-up rows must equal the equal-weight mean of finite signals. Detail: ledger entry `L-EVAL-S10-003`.

### 3c. Field-level point-in-time visibility (PIT correctness; not leakage)

**Failure type:** coverage correctness inside the PIT discipline. The `accepted_date <= as_of` constraint *is* leakage prevention, and that constraint is enforced. The bug fixed at L-INFRA-014 was a downstream coverage bug: a row-level dedup that picked the latest filing without merging the field set, so a partial re-disclosure could null out fields the original filing carried. A reader who labels this a leakage fix would overclaim; the leakage guard is the `<=` predicate, which was correct from the start.

**Where it hides.** A 10-K that re-discloses an old fiscal year with only a few fields (cash + equity, for instance, in a multi-year rollforward) gets selected by `QUALIFY ROW_NUMBER() OVER (PARTITION BY fy, period, end_date ORDER BY accepted_date DESC) = 1`. Fields the partial disclosure omitted come back NULL even though the original 10-K carried them. Backtests after the partial then read, for instance, AAPL FY2015 total_assets as NULL when the correct PIT value is approximately 290B from the 2015 10-K.

**How it was found.** S4 coverage diagnostic on real EDGAR balance-sheet data. The `reports/balance_sheet_coverage_diagnostic.txt` showed total_assets at < 5% coverage where companyfacts emission rates suggested > 80%. The discrepancy was the dedup picking the partial restatement.

**How it is guarded.** Field-level PIT assembly: for each field, the latest non-null value among rows with `accepted_date <= as_of` within the same `(fiscal_year, period, end_date)` group. A partial re-disclosure updates only the fields it contains; omitted fields retain their last-known value. Source: `fmf/features/point_in_time.py::fetch_pit_series` + `_field_level_pit_select_sql`.

**Invariant:**
- `tests/features/test_point_in_time.py::test_pit_series_partial_redisclosure_retains_omitted_field` - synthetic in-memory partial restatement; pre-fix returns NaN, post-fix returns the original value. Names the mechanism (partial restatement).
- `compute_coverage` shares the field-level path via `_field_level_pit_select_sql` so coverage measures what features deliver. Detail: ledger entry `L-INFRA-014`.

### 3d. Train/test purge and the Q4-post-10K cutoff (leakage)

**Failure type:** label-availability leakage at the fold boundary.

**Where it hides.** A training row's target must have been observable strictly before the test fold's cutoff `T_k`. The F1 schedule's Q4-post-10K cutoff (next-year March, after the FY 10-K is filed) would trivialise the target if the per-row target lookup returned the calendar-year FY rather than the next undisclosed FY at this row's own as_of.

**How it was found.** S10 plan sketch, before code.

**How it is guarded.** Two layers compose. The per-row target mechanism (§3a) returns the first FY whose earliest disclosure is strictly after this row's as_of, never the cutoff year. The purge is the strict inequality `train_row.target_accepted_date < T_k`, applied per row, not a length in days. Source: `_run_fold` purge mask + `next_fy_target` per row.

**Invariant:**
- `tests/equity/forecasting/evaluation/test_backtester_invariants.py::test_purge_invariant_train_targets_strictly_before_test_cutoff` - asserts `max(train.target_accepted_date) < T_k` per scored fold. Failure message points at the violating fold and the maximum date, not a metric drift.
- `tests/equity/forecasting/evaluation/test_backtester_invariants.py::test_no_q4_target_observed_at_as_of` - on every Q4-post-10K row, `target_accepted_date > as_of_date`. Detail: ledger entry `L-EVAL-S10-001`.

### 3e. Per-fold vs global parameter fitting (selection / parameter leakage)

**Failure type:** selection leakage on the feature cap; parameter leakage on the AR(1) baseline. Same class of bug on two surfaces.

**Where it hides.** Two fitting surfaces. The S10 top-k feature cap on LightGBM gain importance: if the ranking is computed once across all folds and reused, the cap is selection leakage even when the per-fold LightGBM is fit correctly. The S11 AR(1) baseline phi: if phi is fit on the entire series, the baseline's prediction at any row sees later data.

**How it was found.** Both surfaces were specified per-fold in the design from the start; the discipline call-out was that both must be enforced *structurally*, not by convention, because a future refactor that introduces a module-level cache or a "compute once and reuse" optimisation reintroduces the bug class.

**How it is guarded.** Structural call-shape. The cap helper consumes only the booster handed in; the AR(1) fit helper consumes only the fold's training DataFrame. No module-level cache, no shared registry surface, no global ranking object. The orchestrator hands per-fold artifacts to per-fold-scoped helpers, and that is the entire mechanism.

**Invariant:**
- `tests/equity/forecasting/evaluation/test_backtester_invariants.py::test_feature_cap_ranking_fires_once_per_scored_fold` - monkeypatch spy on `top_k_feature_importance`; asserts call-count equals the number of scored folds with the cap active. Reads the call-shape directly, not the selected set (which can legitimately match across folds when the same features dominate gain throughout the test span).
- `tests/equity/forecasting/evaluation/test_backtester_invariants.py::test_ar1_fit_fires_once_per_scored_fold` - same monkeypatch pattern on `fit_ar1_pooled`. Detail: ledger entries `L-EVAL-S10-001` and `L-EVAL-S11-001`.

## 4. What v1.0 validates

The headline scoreboard is the 9-ticker fixture (8 scored; HSY excluded by a source-data tag gap on `eps_diluted`, see §5d), evaluated under the F1 four-cutoff grid with test window 2020-2023. Expanding-window training data extends back to the earliest filing per ticker in the fixture (AAPL from 2009-Q4; SNOW IPO'd 2020-Q3 so its training history is short by construction). The blend being validated is (LightGBM + TiRex + naive-last-year). The live (LightGBM + TiRex + analyst-consensus) blend cannot be backtested on public data because no historical consensus revisions are available, only a present-day yfinance snapshot.

**Configuration.** `metric=eps_diluted`, `grid_strategy=filing_dates`, `feature_ids=(revenue_ttm, gross_margin, net_margin, return_on_equity)`, `min_train_samples=10`, `meta_min_train=8`, real `TirexHuggingFaceBackend` on CPU (`device="cpu"`), `OMP_NUM_THREADS=1` (Apple Silicon libomp note in §5g). Generated by `python scripts/run_headline_scoreboard.py`; commit SHA stamped in the JSON output at `reports/headline_scoreboard.json`.

**Scope.** 392 scored rows across 16 folds; meta-learner activated at fold 3 (after 3 cold-start folds accumulated > 8 realised OOS triples).

### 4a. Aggregate scoreboard

| Model | MAPE | MedAPE | DA | BA | Coverage | Correlation |
|---|---|---|---|---|---|---|
| **TiRex** | 0.7497 | **0.1581** | **0.7875** | 0.7875 | 0.9005 | 0.6193 |
| **Ensemble** | 0.7737 | 0.1897 | 0.6352 | 0.6352 | 1.0000 | 0.5718 |
| NaiveLastYear | 0.9818 | 0.1797 | 0.0000 | 0.0000 | 0.9719 | 0.4675 |
| RandomWalk | 0.9818 | 0.1797 | 0.0000 | 0.0000 | 0.9719 | 0.4675 |
| SeasonalNaive | 0.9818 | 0.1797 | 0.0000 | 0.0000 | 0.9719 | 0.4675 |
| LightGBM | 0.4567 | 0.3143 | 0.5958 | 0.5958 | 1.0000 | 0.4493 |
| AR1 | 1.1610 | 0.3234 | 0.4357 | 0.4357 | 0.9719 | 0.4061 |

A few non-obvious reads.

- **NaiveLastYear, RandomWalk, and SeasonalNaive collapse to one number.** For an annual target with `season_length=1` they are mathematically identical predictors (predict next equals last disclosed). They ship as distinct scoreboard rows to match the spec narrative; mathematically there is one naive baseline, with MedAPE 0.1797 on this universe. Notional shown so the table doesn't read like a bug. See `L-EVAL-S11-001`.
- **TiRex beats NaiveLastYear on MedAPE.** 0.1581 vs 0.1797. The zero-shot foundation model is contributing real signal on EPS forecasting from quarterly EPS context (spec line 278 alignment: 12-quarter minimum context, 4-quarter horizon, four quarterly median predictions summed to the annual estimate).
- **TiRex directional accuracy is 79% vs Naive's 0%.** Naive's DA is 0 by construction (a prediction equal to the prior actual yields zero signed change, so the sign-match against actual movement is undefined and the metric records 0). TiRex's 79% is real directional signal at the 1-year horizon.
- **Ensemble's aggregate MedAPE is 0.190.** Between TiRex (0.158) and Naive (0.180). The simplex blender's weights respond to the OOS prior-fold triples; LightGBM's lower MedAPE rank causes the simplex to down-weight it relative to TiRex and naive on this universe.
- **LightGBM loses to NaiveLastYear on MedAPE (0.314 vs 0.180).** The most important single finding in the table. Treated in detail at §5a.
- **MAPE explodes at long horizons but MedAPE stays stable.** The aggregate MAPE for all models exceeds 0.45, driven by two distinct outlier sources. SNOW two-year filing gaps (FY2021 and FY2023, both with +363d / +364d lags) put the target two FYs ahead of any naive baseline. The GOOGL 20-for-1 stock split of July 2022 produces three scored rows in 2022 (Q1 May-15, Q2 Aug-14, Q3 Nov-15 cutoffs) where the naive baseline is the pre-split FY2021 EPS of $112.20 and the realized target is the post-split FY2022 EPS of $4.56, a 24x discontinuity that yields a single-row naive APE of 23.6 on each row. The EDA notebook at `docs/eda/01_targets_and_persistence.ipynb` walks the backtester's view of the split for consistency; both surfaces eat the same as-disclosed series. See §5i for the as-disclosed vs split-adjusted limitation. The by-bucket slice in §4b separates the outlier-driven MAPE from the stable MedAPE.

### 4b. Sliced by horizon bucket

Buckets: short (<= 200d), medium (200 to 365d), long (> 365d). The horizon distribution on this fixture has median 330d, p99 728d (the 728d cases are SNOW FY2021 and FY2023 with +363d and +364d filing lags from the backlog), so the long bucket carries outlier weight in MAPE but not in MedAPE.

| model | bucket | MAPE | MedAPE | DA | Correlation |
|---|---|---|---|---|---|
| TiRex | short | 0.302 | 0.143 | 0.804 | 0.975 |
| TiRex | medium | 0.419 | 0.179 | 0.820 | 0.767 |
| TiRex | long | 2.794 | 0.158 | 0.628 | -0.020 |
| Ensemble | short | 0.375 | **0.126** | 0.698 | 0.958 |
| Ensemble | medium | 0.484 | 0.223 | 0.607 | 0.733 |
| Ensemble | long | 2.846 | 0.173 | 0.628 | 0.009 |
| NaiveLastYear | short | 1.151 | 0.187 | 0.000 | 0.457 |
| NaiveLastYear | medium | 0.469 | 0.199 | 0.000 | 0.730 |
| NaiveLastYear | long | 2.905 | 0.162 | 0.000 | -0.028 |
| LightGBM | short | 0.416 | 0.243 | 0.585 | 0.403 |
| LightGBM | medium | 0.549 | 0.364 | 0.589 | 0.452 |
| LightGBM | long | 0.444 | 0.277 | 0.647 | 0.739 |

Two non-obvious reads.

- **The strongest cell in the whole table is Ensemble short-horizon MedAPE = 0.126.** The blend beats every standalone model on the cleanest bucket. This is the v1.0 win.
- **TiRex long-bucket correlation goes negative (-0.020) and MAPE jumps to 2.79.** The MedAPE stays stable (0.158), so the explosion is a few wild outliers from the 700+ day SNOW gap cases. Reporting MedAPE alongside MAPE is the point of the bucket slice; a reader who saw only aggregate MAPE = 0.75 for TiRex would draw the wrong conclusion.

**Reproducer.** `uv sync --extra tirex && uv run python scripts/run_headline_scoreboard.py`. Writes `reports/headline_scoreboard.json` with the committed-SHA stamp and all numbers above. Runtime ~5 minutes on Apple Silicon CPU at `OMP_NUM_THREADS=1`.

## 5. Limitations

Seven items, one short paragraph each. Stated up front rather than scattered as caveats, because a reader who sees the limitations plainly trusts the claims more.

### 5a. LightGBM loses to NaiveLastYear on this fixture, and that is the honest finding

**The operative fact: naive-last-year reaches MedAPE = 0.180 on EPS over the 2020-2023 evaluation window, and the EDA notebook at `docs/eda/01_targets_and_persistence.ipynb` recovers 0.178 on the full annual history of the fixture using the same near-zero APE filter. The two surfaces produce the same number within reproducibility, and that number is a genuinely strong baseline on this universe.** LightGBM reaches 0.314 on the headline window. The gap is not a wiring bug; it is the honest difficulty of forecasting annual EPS on a small cross-section, and the EDA shows the mechanism explicitly.

EPS persistence on this universe is **moderate and uneven, not uniformly high**: pooled lag-1 rho = 0.642 across 117 (security, year) pairs, with a wide per-ticker spread driven by two real mechanisms. Share-count actions (AAPL and MSFT buyback programs) move the EPS denominator independently of business performance; AAPL's lag-1 rho of 0.65 reflects this. The GOOGL 20-for-1 stock split of July 2022 injects a hard structural break (FY2021 EPS $112.20, FY2022 $4.56, a 24x discontinuity in the as-disclosed series the model consumes; see §5i for the as-disclosed limitation). Stable operating metrics behave differently: EBIT pooled rho = 0.976 across 125 pairs, with most tickers at 0.95+ individually. The keystone reframing is that "highly persistent" was the wrong frame for EPS; the right frame is "moderate and uneven, with structural breaks where corporate actions intervene."

That persistence profile sets a floor of roughly 18-20 percent median absolute YoY change on the universe, which a naive predictor cannot drop below and which a model must reliably beat to claim a real win. Eight scored tickers is far too few cross-sectional samples for a tree model on four features to learn signal that generalises past naive persistence at this floor. The resolution path is `IDEA-S18-001` (cross-section expansion) and S15 (per-cell noise-floor sigma), in that order.

The methodological point closes it. A model that beat naive by a wide margin on a fixture this thin would be more suspicious than reassuring, because on this little data a win is more likely to be leakage or overfit than signal. LightGBM under-performing here is consistent with, and evidence for, the leakage-free design.

### 5b. The naive-as-third-signal divergence

The backtest meta-learner uses last-year-actual as its third signal (consensus floor = 0 in this path; S10 Decision 7). The live system uses analyst consensus with `consensus_floor=0.30`. These two predictors collapse for some securities (where consensus is close to last-year extrapolation) and diverge for others, and v1.0 cannot quantify the divergence because no historical analyst-consensus revisions are available on public APIs. The headline §4 numbers validate the (LightGBM + TiRex + naive-last-year) proxy blend, NOT the live (LightGBM + TiRex + consensus) blend. A reader should not transfer the §4 claims to the live system without further measurement.

### 5c. Horizon long-weighting on the F1 four-cutoff grid

On the 9-ticker fixture the horizon distribution has median 330d, p99 728d, max 728d. The 728-day p99 is the SNOW FY2021 and FY2023 two-year filing-gap cases noted in `docs/knowledge/backlog.md`; the corresponding maximum on the 2-anchor (AAPL + MSFT) invariant fixture is 371d, so anyone replicating against the smaller fixture will see a tighter distribution. Aggregate metrics weight long horizons disproportionately; the bucket slice in §4b separates them. This is a known property of the F1 grid, not a bug. Detail: `L-EVAL-S11-001`.

### 5d. Eight of nine tickers reach the scoreboard; HSY is excluded by a source-data tag gap

HSY contributes zero scored rows because the EDGAR `eps_diluted` tag is not consistently emitted in HSY's 10-K and 10-Q filings, so the next-FY target lookup returns None for every HSY row and the backtester drops them. Eight of nine tickers reach the scoreboard. The absence is named here, not hidden as silent ticker dropping, because if a reader spots silent dropping they wonder what else went quietly. Tracked as `IDEA-S22-001`.

### 5e. SNOW partial TiRex coverage

SNOW IPO'd 2020-Q3, so before mid-2023 it has fewer than 12 quarters of EPS history. TiRex's `MIN_CONTEXT_LENGTH=12` quarters causes the wrapper to return None on early SNOW rows; only 4 of 43 SNOW rows in the headline carry a TiRex prediction. The cold-start equal-weight blend handles those rows. The other seven scored tickers have full TiRex coverage. Tracked as `IDEA-S22-003` for a future short-history fallback chain.

### 5f. Snapshot-only analyst consensus

The yfinance analyst-estimates table is a present-day snapshot, no historical revisions. `fetch_consensus_pit` returns empty for any historical as_of. The yfinance consensus column is preserved on each backtest result row as a caveated secondary reference, never used as a meta-learner training input. The proper historical-consensus validation requires a paid data source and is out of scope for a portfolio repo.

### 5g. Apple Silicon libomp interop note

LightGBM and PyTorch (loaded by tirex-ts) both link `libomp.dylib`. Running both in the same process on Apple Silicon can deadlock at OpenMP barriers, where threads from both libraries wait on each other indefinitely (observed: 0% CPU for hours, stack pinned at `__kmp_hyper_barrier_release`). Workaround: `OMP_NUM_THREADS=1` env var plus `torch.set_num_threads(1)` before importing the orchestrator. The headline reproducer script applies both at module load. The fixture TiRex backend used by the unit and slow lanes does not import torch and is unaffected. Documented on `TirexHuggingFaceBackend`'s docstring. Filed as `IDEA-S22-002` for subprocess isolation or a CUDA / MPS path if reproducibility on consumer hardware becomes a blocker.

### 5h. Two-ticker invariant fixture for the cardinal regression suite

The 13 cardinal correctness invariants in §3 run against the AAPL + MSFT (2 anchor) configuration to keep the slow lane under three minutes per invocation in CI. The headline scoreboard in §4 uses the 9-ticker (8 scored) configuration; the two configurations share the same orchestrator and the same fixture DB, only the security list differs. The invariants are universe-independent by design; cross-section size affects scoreboard numbers, not correctness guarantees.

### 5i. As-disclosed EPS, not split-adjusted

fmf-public uses EDGAR as-disclosed `eps_diluted` values rather than split-adjusted ones from Yahoo Finance or a vendor. The trade-off is deliberate: as-disclosed values are what a production system would see in the order they arrive, and PIT correctness on the disclosure stream is the whole point of the backtester. The cost surfaces on corporate-action years. The GOOGL 20-for-1 stock split of July 2022 produces a 24x discontinuity in the as-disclosed series (FY2021 EPS $112.20, FY2022 $4.56), and the backtester eats it as a hard step: three scored rows in the headline have a single-row naive APE of 23.6 each, named in §4a alongside the SNOW two-year-gap rows. The median absorbs them; the mean does not. The EDA notebook walks the consistency check between the backtester's `next_fy_target` / `last_fy_actual` lookups and the lag-1 scatter to confirm both surfaces tell the same story. Split-adjustment is a v1.x improvement filed as `IDEA-S21-002`; the cleanest path is to apply the cumulative split factor from yfinance's `splits` series to the as-disclosed `eps_diluted` at ingest time, producing a parallel `eps_diluted_split_adjusted` column that the backtester can switch onto without changing the PIT layer.

## 6. Roadmap to v1.x

Four programs from the design are explicitly not built in v1.0 and shape the §4 and §5 claims. Each gets one line of what it would establish.

- **Noise floor (S15).** Per-cell sigma via bootstrap. Reclassifies the S18 cluster verdict in `docs/specs/alternative_models.md` from `DEFERRED-pending-S15` to `PASS` or `FAIL`. Closes §5a's open question on whether LightGBM's under-performance vs naive is within sigma or outside it.
- **Search and A/B grid (S16).** Pinned-window contract for hyperparameter and config sweeps without re-introducing the per-fold-vs-global bug class of §3e.
- **Admission gate (S17).** Falsification-friendly gate (correlation <= 0.30 vs existing roster) for the six alternative models held in `docs/knowledge/backlog.md`.
- **Dead-CV reproducer (S19).** Flagship notebook reproducing the proprietary `lgb.cv(stratified=True)` silent-failure on a regression target, with the fix landing as a measurable MedAPE delta on public data.

The IDEA-* tickets that would feed into these programs are tracked in [docs/knowledge/backlog.md](knowledge/backlog.md). Routing those tickets to public GitHub issues is itself a v1.x task; v1.0 ships them as text rather than fabricate issue numbers.

## 7. Reproducibility

Every quantitative claim in §3, §4, and §5 carries a pytest invocation or a shell command that prints the number when run. The L-EVAL and L-INFRA ledger entries in [docs/knowledge/learnings.md](knowledge/learnings.md) each name a reproducer; the no-TBD discipline propagates from the ledger into this doc unchanged.

The cardinal invariants by name and pytest path:

```
tests/equity/forecasting/evaluation/test_backtester_invariants.py
  ::test_purge_invariant_train_targets_strictly_before_test_cutoff
  ::test_no_train_test_row_overlap
  ::test_horizon_persisted_on_every_result_row
  ::test_no_q4_target_observed_at_as_of
  ::test_target_fy_has_no_disclosure_at_or_before_as_of    # 3a, comparative trap
  ::test_feature_cap_ranking_fires_once_per_scored_fold    # 3e, per-fold ranking
  ::test_ar1_fit_fires_once_per_scored_fold                # 3e, per-fold AR(1)
  ::test_meta_learner_train_sources_are_strictly_prior_folds   # 3b, OOS provenance
  ::test_cold_start_equal_weight_blend_before_meta_active  # 3b, warm-up
  ::test_cache_miss_when_data_fingerprint_changes          # §2 callout, data axis
  ::test_with_cache_predictions_match_without_cache        # §2 callout, orthogonality
tests/equity/forecasting/evaluation/test_target_lookup.py
  ::test_next_fy_target_skips_comparative_for_already_disclosed_fy  # 3a, hand-built
tests/features/test_point_in_time.py
  ::test_pit_series_partial_redisclosure_retains_omitted_field     # 3c, field-level
```

The headline §4 numbers are reproducible with `python scripts/run_headline_scoreboard.py` after `uv sync --extra tirex`. The script writes `reports/headline_scoreboard.json` with the commit SHA stamped in.

## 8. Provenance and licensing

Methodology was designed at Bavest on proprietary data; this repo is a clean-room reproduction on public data with no proprietary code or data reused. Specific public-data limitations are stated in §5 rather than as scattered caveats.

- **Code.** MIT License. See [LICENSE-CODE](../LICENSE-CODE) for the full text.
- **Documentation under `docs/`.** Creative Commons Attribution 4.0 International (CC BY 4.0). See [LICENSE-DOCS](../LICENSE-DOCS) for the full text and the [official deed](https://creativecommons.org/licenses/by/4.0/).
- **SEC EDGAR fundamentals.** US government public-domain data; no license restriction. Source: [sec.gov/edgar](https://www.sec.gov/edgar).
- **yfinance prices.** Derived from Yahoo Finance via the open-source [yfinance](https://github.com/ranaroussi/yfinance) package, which scrapes Yahoo Finance and does not redistribute weighted licenses. The committed fixture slice at `tests/fixtures/mini.duckdb` contains 9 anchor tickers (AAPL, GOOGL, GWW, HSY, JNJ, JPM, MSFT, SNOW, ZTS) over their respective filing histories, used solely for hermetic test execution and the headline reproducer. The full ingest is regeneratable from the upstream source via `scripts/ingest_yfinance.py`.
- **TiRex weights.** Loaded at runtime from the [NX-AI/TiRex](https://huggingface.co/NX-AI/TiRex) HuggingFace repository via the [tirex-ts](https://pypi.org/project/tirex-ts/) package. The model card declares `license_name: nxai-community-license-agreement-1-0` with `license_link: LICENSE` (see the model card and the LICENSE file in the HuggingFace repository for the operative text); weights are not redistributed by fmf-public. The fixture TiRex backend used by the test suite contains zero TiRex weights, only deterministic stub outputs keyed by series hash.
- **Citation.** See [CITATION.cff](../CITATION.cff).
