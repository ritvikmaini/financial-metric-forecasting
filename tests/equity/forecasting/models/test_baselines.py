"""Unit tests for S11 baselines (random_walk, seasonal_naive,
last_year_actual_baseline, fit_ar1_pooled, predict_ar1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmf.equity.forecasting.models.baselines import (
    fit_ar1_pooled,
    last_year_actual_baseline,
    predict_ar1,
    random_walk,
    seasonal_naive,
)


def test_random_walk_returns_prev_actual() -> None:
    assert random_walk(3.5) == 3.5


def test_random_walk_returns_none_when_input_none() -> None:
    assert random_walk(None) is None


def test_random_walk_returns_none_for_nan() -> None:
    assert random_walk(float("nan")) is None
    assert random_walk(float("inf")) is None


def test_seasonal_naive_collapses_to_random_walk_at_season_one() -> None:
    assert seasonal_naive(2.5, season_length=1) == 2.5
    assert seasonal_naive(None, season_length=1) is None


def test_seasonal_naive_rejects_non_unit_season_in_v1() -> None:
    with pytest.raises(NotImplementedError):
        seasonal_naive(1.0, season_length=4)


def test_last_year_actual_baseline_matches_random_walk() -> None:
    assert last_year_actual_baseline(7.25) == random_walk(7.25)
    assert last_year_actual_baseline(None) is None


def test_fit_ar1_recovers_phi_on_synthetic_series() -> None:
    rng = np.random.default_rng(0)
    phi_true = 0.85
    n = 200
    y_lag = rng.normal(scale=1.0, size=n)
    y = phi_true * y_lag + 0.05 * rng.normal(size=n)
    df = pd.DataFrame({"y": y, "y_lag": y_lag})
    phi_hat = fit_ar1_pooled(df)
    assert abs(phi_hat - phi_true) < 0.05


def test_fit_ar1_falls_back_to_unit_phi_below_min_pairs() -> None:
    df = pd.DataFrame({"y": [1.0, 2.0], "y_lag": [0.5, 1.5]})
    assert fit_ar1_pooled(df) == 1.0


def test_fit_ar1_falls_back_to_unit_phi_on_zero_lag_norm() -> None:
    df = pd.DataFrame({"y": np.zeros(10), "y_lag": np.zeros(10)})
    assert fit_ar1_pooled(df) == 1.0


def test_fit_ar1_drops_non_finite_rows() -> None:
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, float("inf")],
            "y_lag": [0.5, 1.0, 1.5, float("nan"), 2.5, 3.0, 3.5],
        }
    )
    phi = fit_ar1_pooled(df)
    assert np.isfinite(phi)
    # Only 4 finite pairs remain (rows 0, 1, 4, 5) -> below floor -> 1.0.
    assert phi == 1.0


def test_predict_ar1_returns_none_for_none_input() -> None:
    assert predict_ar1(0.9, None) is None
    assert predict_ar1(0.9, float("nan")) is None


def test_predict_ar1_applies_phi() -> None:
    assert predict_ar1(0.8, 10.0) == pytest.approx(8.0)
    assert predict_ar1(1.0, 5.5) == pytest.approx(5.5)
