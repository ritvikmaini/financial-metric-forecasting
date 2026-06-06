"""LightGBM wrapper tests: API, reproducibility, sample_weight."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fmf.equity.forecasting.models.lightgbm_model import (
    LightGBMForecaster,
    LightGBMHyperparameters,
    _read_seed,
)


@pytest.fixture
def synthetic_xy() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    n, p = 200, 8
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    w_true = rng.normal(size=p)
    y = X.values @ w_true + rng.normal(scale=0.5, size=n)
    return X, y


def test_fit_predict_basic(synthetic_xy: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = synthetic_xy
    model = LightGBMForecaster().fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    # In-sample MAE should beat predicting the mean.
    mae_model = float(np.mean(np.abs(y - preds)))
    mae_mean = float(np.mean(np.abs(y - y.mean())))
    assert mae_model < mae_mean, f"model MAE {mae_model} not better than mean {mae_mean}"


def test_reproducibility_same_seed(synthetic_xy: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = synthetic_xy
    a = LightGBMForecaster(seed=42).fit(X, y).predict(X)
    b = LightGBMForecaster(seed=42).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ(synthetic_xy: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = synthetic_xy
    a = LightGBMForecaster(seed=42).fit(X, y).predict(X)
    b = LightGBMForecaster(seed=7).fit(X, y).predict(X)
    assert not np.allclose(a, b)


def test_sample_weight_changes_predictions(
    synthetic_xy: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = synthetic_xy
    unweighted = LightGBMForecaster().fit(X, y).predict(X)
    weights = np.ones(len(X))
    weights[:50] = 5.0  # upweight first 50 samples
    weighted = LightGBMForecaster().fit(X, y, sample_weight=weights).predict(X)
    assert not np.allclose(unweighted, weighted)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        LightGBMForecaster().predict(pd.DataFrame({"f0": [0.0]}))


def test_predict_missing_features_raises(
    synthetic_xy: tuple[pd.DataFrame, np.ndarray],
) -> None:
    X, y = synthetic_xy
    model = LightGBMForecaster().fit(X, y)
    with pytest.raises(ValueError, match="missing features"):
        model.predict(X.drop(columns=["f0"]))


def test_seed_env_negative_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FMF_LGBM_SEED", "-1")
    with pytest.raises(ValueError, match="must be non-negative"):
        _read_seed()


def test_seed_env_non_integer_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FMF_LGBM_SEED", "abc")
    assert _read_seed() == 42


def test_hyperparameters_default_objective() -> None:
    hp = LightGBMHyperparameters()
    assert hp.objective == "huber"
    assert hp.metric == "mae"


def test_smoke_real_fixture_matrix_slice() -> None:
    """Smoke: read the S5 feature matrix, drop NaN-heavy columns, fit
    a tiny model on a small slice to confirm the wrapper works on
    real-data shape."""
    from pathlib import Path

    matrix_path = Path(__file__).parents[4] / "reports" / "feature_matrix.parquet"
    if not matrix_path.exists():
        pytest.skip("S5 feature matrix not built")
    df = pd.read_parquet(matrix_path)
    # Drop index cols; keep only numeric features with <30% null.
    feature_cols = [
        c for c in df.columns if c not in {"security_id", "symbol", "as_of_date", "as_of_source"}
    ]
    keep = [c for c in feature_cols if df[c].isna().mean() < 0.3]
    X = df[keep].fillna(0.0)
    # Synthetic target: y = revenue_latest (just to exercise fit/predict).
    if "revenue_latest" not in keep:
        pytest.skip("revenue_latest not in available features")
    y = df["revenue_latest"].fillna(0.0).values
    mask = y > 0
    if mask.sum() < 30:
        pytest.skip("not enough samples")
    Xf, yf = X[mask].drop(columns=["revenue_latest"]), y[mask]
    model = LightGBMForecaster().fit(Xf, yf)
    preds = model.predict(Xf)
    assert preds.shape == (len(Xf),)
    assert np.all(np.isfinite(preds))
