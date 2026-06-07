"""Top-k feature-cap tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmf.equity.forecasting.evaluation._feature_cap import top_k_feature_importance
from fmf.equity.forecasting.models.lightgbm_model import LightGBMForecaster


def test_top_k_returns_k_names_in_descending_gain_order() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 6)), columns=[f"f{i}" for i in range(6)])
    y = 3.0 * X["f0"].to_numpy() + 0.1 * rng.normal(size=200)
    lgbm = LightGBMForecaster(seed=42).fit(X, y)
    top3 = top_k_feature_importance(lgbm, k=3)
    assert len(top3) == 3
    assert top3[0] == "f0"
    assert set(top3) <= set(X.columns)


def test_top_k_raises_when_not_fitted() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        top_k_feature_importance(LightGBMForecaster(seed=42), k=3)


def test_top_k_with_k_larger_than_features_returns_all() -> None:
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
    y = rng.normal(size=50)
    lgbm = LightGBMForecaster(seed=42).fit(X, y)
    top10 = top_k_feature_importance(lgbm, k=10)
    assert set(top10) == {"a", "b", "c"}
