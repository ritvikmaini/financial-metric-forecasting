"""End-to-end backtester tests on the committed fixture DB.

Marked `slow` to stay out of the unit suite; CI runs them under the
e2e-backtester job per the spec.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from fmf.equity.forecasting.evaluation._backtester_config import BacktesterConfig
from fmf.equity.forecasting.evaluation.backtester import (
    ExpandingWindowBacktester,
    scoreboard_from_result,
)
from tests.equity.forecasting.evaluation._fixture_helpers import (
    fixture_conn,
    two_anchor_ids,
)


@dataclass
class StubTirexBackend:
    def forecast(self, series: np.ndarray, horizon: int) -> np.ndarray:
        last = float(series[-1])
        return np.full((horizon, 9), last, dtype=np.float64)


@pytest.mark.slow
def test_e2e_fixture_backtest_produces_scoreboard_frame() -> None:
    conn = fixture_conn()
    try:
        cfg = BacktesterConfig(
            metric="eps_diluted",
            start_year=2020,
            end_year=2022,
            grid_strategy="filing_dates",
            feature_ids=("revenue_ttm", "gross_margin", "net_margin"),
            min_train_samples=10,
            meta_min_train=8,
        )
        bt = ExpandingWindowBacktester(conn, cfg, tirex_backend=StubTirexBackend())
        result = bt.run(two_anchor_ids(conn))
    finally:
        conn.close()
    df = result.to_frame()
    assert not df.empty
    expected = {
        "fold_idx",
        "as_of_date",
        "target_fy",
        "horizon_days",
        "lgbm_pred",
        "tirex_pred",
        "ensemble_pred",
        "ensemble_source",
        "target_value",
    }
    assert expected <= set(df.columns)
    scored = df[df["lgbm_pred"].notna()]
    assert len(scored) > 0


@pytest.mark.slow
def test_e2e_deterministic_under_fixed_seed() -> None:
    cfg = BacktesterConfig(
        metric="eps_diluted",
        start_year=2020,
        end_year=2021,
        grid_strategy="filing_dates",
        feature_ids=("revenue_ttm", "gross_margin"),
        min_train_samples=10,
        meta_min_train=8,
        seed=42,
    )
    conn1 = fixture_conn()
    try:
        ids = two_anchor_ids(conn1)
        r1 = (
            ExpandingWindowBacktester(conn1, cfg, tirex_backend=StubTirexBackend())
            .run(ids)
            .to_frame()
        )
    finally:
        conn1.close()
    conn2 = fixture_conn()
    try:
        r2 = (
            ExpandingWindowBacktester(conn2, cfg, tirex_backend=StubTirexBackend())
            .run(ids)
            .to_frame()
        )
    finally:
        conn2.close()
    # Drop UUID-bearing object cols (security_id) for clean equality; compare predictions.
    pd.testing.assert_frame_equal(
        r1[["fold_idx", "as_of_date", "target_fy", "lgbm_pred", "tirex_pred", "ensemble_pred"]],
        r2[["fold_idx", "as_of_date", "target_fy", "lgbm_pred", "tirex_pred", "ensemble_pred"]],
    )


@pytest.mark.slow
def test_scoreboard_helper_produces_seven_metrics_per_model() -> None:
    conn = fixture_conn()
    try:
        cfg = BacktesterConfig(
            metric="eps_diluted",
            start_year=2020,
            end_year=2022,
            grid_strategy="filing_dates",
            feature_ids=("revenue_ttm", "gross_margin"),
            min_train_samples=10,
            meta_min_train=8,
        )
        result = ExpandingWindowBacktester(conn, cfg, tirex_backend=StubTirexBackend()).run(
            two_anchor_ids(conn)
        )
    finally:
        conn.close()
    board = scoreboard_from_result(result)
    assert {"LightGBM", "TiRex", "Ensemble", "NaiveLastYear"} <= set(board.index)
    expected_metrics = {
        "mape",
        "median_ape",
        "accuracy_within_10pct",
        "accuracy_within_25pct",
        "directional_accuracy",
        "beat_miss_accuracy",
        "coverage",
        "correlation",
    }
    assert expected_metrics <= set(board.columns)
