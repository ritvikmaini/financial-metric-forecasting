"""BacktesterConfig validation tests."""

from __future__ import annotations

import dataclasses

import pytest

from fmf.equity.forecasting.evaluation._backtester_config import (
    GRID_STRATEGIES,
    METRICS,
    BacktesterConfig,
)


def test_config_accepts_valid_inputs() -> None:
    cfg = BacktesterConfig(
        metric="eps_diluted", start_year=2018, end_year=2024, feature_ids=("revenue_ttm",)
    )
    assert cfg.start_year == 2018
    assert cfg.seed == 42


def test_config_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="metric must be one of"):
        BacktesterConfig(metric="revenue", start_year=2018, end_year=2024, feature_ids=("a",))


def test_config_rejects_inverted_years() -> None:
    with pytest.raises(ValueError, match="start_year < end_year"):
        BacktesterConfig(metric="eps_diluted", start_year=2024, end_year=2018, feature_ids=("a",))


def test_config_rejects_empty_features() -> None:
    with pytest.raises(ValueError, match="feature_ids must be non-empty"):
        BacktesterConfig(metric="eps_diluted", start_year=2018, end_year=2024, feature_ids=())


def test_config_rejects_min_train_samples_below_three() -> None:
    with pytest.raises(ValueError, match=">= 3"):
        BacktesterConfig(
            metric="eps_diluted",
            start_year=2018,
            end_year=2024,
            feature_ids=("a",),
            min_train_samples=2,
        )


def test_config_rejects_negative_feature_cap() -> None:
    with pytest.raises(ValueError, match="feature_cap_top_k must be >= 0"):
        BacktesterConfig(
            metric="eps_diluted",
            start_year=2018,
            end_year=2024,
            feature_ids=("a",),
            feature_cap_top_k=-1,
        )


def test_config_rejects_meta_min_train_below_three() -> None:
    with pytest.raises(ValueError, match="meta_min_train"):
        BacktesterConfig(
            metric="eps_diluted",
            start_year=2018,
            end_year=2024,
            feature_ids=("a",),
            meta_min_train=2,
        )


def test_config_is_frozen() -> None:
    cfg = BacktesterConfig(metric="eps_diluted", start_year=2018, end_year=2024, feature_ids=("a",))
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 7  # type: ignore[misc]


def test_config_has_no_embargo_field() -> None:
    assert "embargo" not in BacktesterConfig.__dataclass_fields__
    assert "embargo_quarters" not in BacktesterConfig.__dataclass_fields__


def test_metrics_constant_matches_eps_ebitda_ebit() -> None:
    assert set(METRICS) == {"eps_diluted", "ebitda", "ebit"}


def test_grid_strategies_excludes_daily_calendar() -> None:
    assert "daily_calendar" not in GRID_STRATEGIES
