# Backlog

- SNOW FY2021 (+364d) and FY2023 (+363d) Q4 still year-lagged after L-INFRA-012 + L-INFRA-013 fixes. Not L-INFRA-012 (calendar-shifted but duration gate handles it) or L-INFRA-013 (no Q3-bucket phantom in this case). Suspect a SNOW-specific normalize edge case in the comparative-period emission. Recency guard in `compute_revenue_ttm` returns None for these as_ofs, so no feature corruption; investigate before S5.

## S10 backtester IDEA tickets (filed to S15/S17 tier program)

- `IDEA-S10-001` — Regime-shift regularization embargo as an explicit swept knob. The proprietary `_EMBARGO_QUARTERS = 1` sweep winner (-0.74 pp MedAPE) is a regime-shift regularization effect, not a leakage fix. S10 ships embargo = 0 by construction; the regularization buffer belongs in tier-sweep land.
- `IDEA-S10-002` — Sliding window (`_SLIDING_WINDOW_YEARS > 0`) as a config knob. Expanding-window only in v1.0.
- `IDEA-S10-003` — Nested CV refinement on the meta-learner OOS training set (out-of-time double-CV, walk-forward stratification). v1.0 ships prior-fold OOS triples directly per Decision 11; this is for finer refinements, not the in-sample-versus-OOS fix itself which is the v1.0 default.
- `IDEA-S10-004` — Cache TirexHuggingFaceBackend startup if measured to dominate. v1.0 instantiates TirexForecaster once per backtester run and reuses; per-row predict calls hit the backend's own cache.
- `IDEA-S10-010` — Top-k feature-cap sweep against the scoreboard. Proprietary IDEA-017 noted the cap was never empirically swept; v1.0 ships the mechanism at default k=30 and defers the sweep.
- `IDEA-S10-011` — Naive-baseline floor sweep candidate. v1.0 ships `consensus_floor=0.0` on the naive third signal in the backtest path (the floor's original 0.30 justification was consensus being a strong external prior, not transferable to last-year-actual). Live path keeps 0.30 on consensus.
- `IDEA-S10-012` — Warm-up boundary marker in the S22 scoreboard chart: shade the folds where `ensemble_source == "cold_start_equal_weight"` so readers see the transition explicitly.
- `IDEA-S11-001` - Per-security AR(1) instead of pooled. v1.0 ships pooled OLS phi per fold; per-security would refit phi per ticker per fold. Adds complexity; defer until empirically motivated.
- `IDEA-S11-002` - Per-bucket noise floor measurement in S15. The S11 horizon-bucket scoreboard surfaces a long-weighted distribution; the S15 noise floor should be computed per bucket as well so claimed improvements are gated against the bucket-specific sigma, not the aggregate.
- `IDEA-S12-001` - actuals_backfill loop. Spec line 512 marks it deferred (`B-010`); a portfolio repo has no live truth-arrival reconciliation. If a future version wants to compare past predictions vs realized actuals, the loop reads from `reports/predictions/*.parquet` and joins against the income_statement table by (security_id, target_fy).
- `IDEA-S13-001` - Code-sha variant of config_flags_hash. v1.0 ships dataclass-fields-only so the hash is stable across cosmetic refactors. The noise-floor program (S15) may want a code-sha-included variant when reproducibility-across-commits matters.
- `IDEA-S13-002` - `ab`, `search`, `search_cell` modes. v1.0 ships `adhoc` and `backfill`; the additional modes belong to S16 (`s16/search-ab`).
