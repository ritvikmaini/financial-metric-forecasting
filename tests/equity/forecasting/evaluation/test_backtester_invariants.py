"""S10 backtester correctness invariants — purge, no-overlap, horizon,
Q4-non-trivial, comparative-trap, per-fold cap, OOS meta training,
cold-start.

These are the regression gates. Each one targets a specific class of bug
the close-read review surfaced; their failure modes describe the bug at
its actual surface rather than a downstream symptom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from fmf.equity.forecasting.evaluation._backtester_config import BacktesterConfig
from fmf.equity.forecasting.evaluation.backtester import (
    BacktestResult,
    ExpandingWindowBacktester,
)
from tests.equity.forecasting.evaluation._fixture_helpers import (
    fixture_conn,
    two_anchor_ids,
)


@dataclass
class StubTirexBackend:
    """Deterministic stub: replicate the series' last value across (horizon, 9).
    Lets the orchestrator run end-to-end without HF weights for invariant tests.
    Production / S22 uses TirexHuggingFaceBackend."""

    def forecast(self, series: np.ndarray, horizon: int) -> np.ndarray:
        last = float(series[-1])
        return np.full((horizon, 9), last, dtype=np.float64)


@pytest.fixture(scope="module")
def small_result() -> BacktestResult:
    conn = fixture_conn()
    try:
        cfg = BacktesterConfig(
            metric="eps_diluted",
            start_year=2018,
            end_year=2023,
            grid_strategy="filing_dates",
            feature_ids=("revenue_ttm", "gross_margin", "net_margin", "return_on_equity"),
            min_train_samples=10,
            meta_min_train=8,
            feature_cap_top_k=2,
        )
        bt = ExpandingWindowBacktester(conn, cfg, tirex_backend=StubTirexBackend())
        ids = two_anchor_ids(conn)
        yield bt.run(ids)
    finally:
        conn.close()


def test_purge_invariant_train_targets_strictly_before_test_cutoff(
    small_result: BacktestResult,
) -> None:
    """Cardinal invariant 1: Decisions 3 + 4."""
    for fold_idx, diag in small_result.fold_diagnostics.items():
        if diag.is_seed or diag.train_target_accepted_max is None:
            continue
        assert diag.train_target_accepted_max < diag.cutoff, (
            f"fold {fold_idx} train target max {diag.train_target_accepted_max} "
            f">= cutoff {diag.cutoff}"
        )


def test_no_train_test_row_overlap(small_result: BacktestResult) -> None:
    """Cardinal invariant 2: no (security, as_of) appears in both train and test."""
    for fold_idx, diag in small_result.fold_diagnostics.items():
        if diag.is_seed:
            continue
        intersect = diag.train_keys & diag.test_keys
        assert not intersect, f"fold {fold_idx} overlap: {intersect}"


def test_horizon_persisted_on_every_result_row(small_result: BacktestResult) -> None:
    """Cardinal invariant 3: Decision 2."""
    for row in small_result.rows:
        assert row.horizon_days > 0
        assert row.horizon_days == (row.target_accepted_date - row.as_of_date).days


def test_no_q4_target_observed_at_as_of(small_result: BacktestResult) -> None:
    """Cardinal invariant 4: Decision 5."""
    for row in small_result.rows:
        if "Q4-post-10K" not in row.cutoff_label:
            continue
        assert row.target_accepted_date > row.as_of_date


def test_target_fy_has_no_disclosure_at_or_before_as_of(
    small_result: BacktestResult,
) -> None:
    """Cardinal invariant 8: comparative-trap guard. The bug Invariant 4 misses.
    Fixture precondition (T0 Step 4): >=1 ticker with multiple period='FY' rows."""
    conn = fixture_conn()
    try:
        for row in small_result.rows:
            count = conn.execute(
                'SELECT COUNT(*) FROM "income_statement" '
                "WHERE security_id = ? AND period = 'FY' "
                "AND fiscal_year = ? AND accepted_date <= ?",
                [str(row.security_id), int(row.target_fy), row.as_of_date],
            ).fetchone()[0]
            assert count == 0, (
                f"comparative trap: target_fy={row.target_fy} already disclosed "
                f"for {row.symbol} at or before as_of={row.as_of_date} "
                f"(found {count} period='FY' row(s) <= as_of)"
            )
    finally:
        conn.close()


def test_feature_cap_can_differ_across_folds(small_result: BacktestResult) -> None:
    """Cardinal invariant 5 (structural): the selected feature set is ALLOWED to
    differ across folds. When the spy guard below already pins call-shape, the
    structural check is informational — it skips with a clear message if a
    correct per-fold ranking happens to pick the same top-k across all folds
    (which can occur when 1-2 features dominate gain throughout the test span).
    Use a feature_ids set with enough cross-fold importance variation to make
    this assertion meaningful, or rely on the spy test."""
    fold_features = {
        fi: diag.selected_features
        for fi, diag in small_result.fold_diagnostics.items()
        if diag.selected_features is not None and not diag.is_seed
    }
    if len(fold_features) < 2:
        pytest.skip("need at least two scored folds with cap active")
    sets = [frozenset(s) for s in fold_features.values()]
    if len(set(sets)) < 2:
        pytest.skip(
            f"all {len(sets)} folds picked the same top-k set "
            f"{sorted(next(iter(set(sets))))}. Structural check inconclusive; "
            f"the spy test test_feature_cap_ranking_fires_once_per_scored_fold "
            f"is the load-bearing guard against global ranking."
        )
    assert len(set(sets)) >= 2


def test_feature_cap_ranking_fires_once_per_scored_fold(monkeypatch) -> None:
    """Cardinal invariant 5 (firmed): call-shape guard, doesn't rely on fixture
    importance variation."""
    from fmf.equity.forecasting.evaluation import _feature_cap
    from fmf.equity.forecasting.evaluation import backtester as bt_mod

    calls: list[int] = []
    original = _feature_cap.top_k_feature_importance

    def counted(lgbm, k):
        calls.append(1)
        return original(lgbm, k)

    monkeypatch.setattr(_feature_cap, "top_k_feature_importance", counted)
    monkeypatch.setattr(bt_mod, "top_k_feature_importance", counted)
    conn = fixture_conn()
    try:
        cfg = BacktesterConfig(
            metric="eps_diluted",
            start_year=2018,
            end_year=2022,
            grid_strategy="filing_dates",
            feature_ids=("revenue_ttm", "gross_margin", "net_margin", "return_on_equity"),
            min_train_samples=10,
            meta_min_train=8,
            feature_cap_top_k=2,
        )
        bt = ExpandingWindowBacktester(conn, cfg, tirex_backend=StubTirexBackend())
        result = bt.run(two_anchor_ids(conn))
    finally:
        conn.close()
    scored_with_cap = sum(
        1
        for d in result.fold_diagnostics.values()
        if not d.is_seed and d.train_n >= cfg.min_train_samples and d.selected_features is not None
    )
    assert len(calls) == scored_with_cap, (
        f"feature_cap ranking fired {len(calls)} times across "
        f"{scored_with_cap} scored folds with cap active; expected 1:1."
    )


def test_meta_learner_train_sources_are_strictly_prior_folds(
    small_result: BacktestResult,
) -> None:
    """Cardinal invariant 6: training provenance, not output labeling."""
    activated = [
        (fold_idx, diag)
        for fold_idx, diag in small_result.fold_diagnostics.items()
        if diag.meta_active
    ]
    if not activated:
        pytest.skip("no OOS-learned fold in this run; cold-start only")
    for fold_idx, diag in activated:
        sources = diag.meta_train_source_folds
        assert sources is not None and len(sources) > 0, (
            f"fold {fold_idx} meta_active=True but no source folds recorded"
        )
        assert all(src < fold_idx for src in sources), (
            f"fold {fold_idx} meta-learner trained on non-prior source folds: {sorted(sources)}"
        )


def test_meta_learner_output_labeling_matches_activation(
    small_result: BacktestResult,
) -> None:
    """Companion to Invariant 6: row labeling matches activation bookkeeping."""
    if small_result.meta_learned_from_fold is None:
        pytest.skip("no OOS-learned fold in this run; cold-start only")
    assert small_result.meta_learned_from_fold >= 1
    for row in small_result.rows:
        if row.ensemble_source == "oos_learned":
            assert row.fold_idx >= small_result.meta_learned_from_fold


def test_cold_start_equal_weight_blend_before_meta_active(
    small_result: BacktestResult,
) -> None:
    """Cardinal invariant 7: warm-up reports as the equal-weight blend over
    finite signals."""
    for row in small_result.rows:
        if row.ensemble_source != "cold_start_equal_weight":
            continue
        finite = []
        if row.lgbm_pred is not None and np.isfinite(row.lgbm_pred):
            finite.append(row.lgbm_pred)
        if row.tirex_pred is not None and np.isfinite(row.tirex_pred):
            finite.append(row.tirex_pred)
        if row.naive_baseline is not None and np.isfinite(row.naive_baseline):
            finite.append(row.naive_baseline)
        if not finite:
            assert row.ensemble_pred is None or not np.isfinite(row.ensemble_pred)
            continue
        expected = sum(finite) / len(finite)
        assert abs((row.ensemble_pred or 0.0) - expected) < 1e-9
