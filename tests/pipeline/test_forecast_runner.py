"""Tests for pipeline.forecast_runner."""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fmf.equity.forecasting.models.lightgbm_model import LightGBMForecaster
from fmf.pipeline.forecast_runner import run_forecast


@pytest.fixture
def fitted_model_path(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(80, 3)), columns=["revenue_ttm", "gross_margin", "net_margin"]
    )
    y = 2.0 * X["revenue_ttm"].to_numpy() + rng.normal(scale=0.1, size=80)
    model = LightGBMForecaster(seed=42).fit(X, y)
    out = tmp_path / "lgbm_model"
    model.save_model(out)
    return out


@pytest.fixture
def dataset() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 5
    return pd.DataFrame(
        {
            "security_id": [str(uuid.uuid4()) for _ in range(n)],
            "symbol": [f"SYM{i}" for i in range(n)],
            "as_of_date": [dt.date(2024, 5, 15)] * n,
            "revenue_ttm": rng.normal(size=n),
            "gross_margin": rng.normal(size=n),
            "net_margin": rng.normal(size=n),
        }
    )


def test_round_trip_predictions_match_direct_predict(
    fitted_model_path: Path, dataset: pd.DataFrame, tmp_path: Path
) -> None:
    out_dir = tmp_path / "preds"
    out_path, _ = run_forecast(
        model_path=fitted_model_path,
        dataset=dataset,
        metric="eps_diluted",
        output_dir=out_dir,
    )
    written = pd.read_parquet(out_path)
    direct = LightGBMForecaster.load_model(fitted_model_path).predict(
        dataset[["revenue_ttm", "gross_margin", "net_margin"]]
    )
    np.testing.assert_array_almost_equal(written["prediction"].to_numpy(), direct)


def test_run_id_deterministic_on_same_input(
    fitted_model_path: Path, dataset: pd.DataFrame, tmp_path: Path
) -> None:
    _, run_id_a = run_forecast(
        model_path=fitted_model_path,
        dataset=dataset,
        metric="eps_diluted",
        output_dir=tmp_path / "a",
    )
    _, run_id_b = run_forecast(
        model_path=fitted_model_path,
        dataset=dataset,
        metric="eps_diluted",
        output_dir=tmp_path / "b",
    )
    assert run_id_a == run_id_b


def test_run_id_changes_with_metric(
    fitted_model_path: Path, dataset: pd.DataFrame, tmp_path: Path
) -> None:
    _, run_id_eps = run_forecast(
        model_path=fitted_model_path,
        dataset=dataset,
        metric="eps_diluted",
        output_dir=tmp_path / "a",
    )
    _, run_id_ebit = run_forecast(
        model_path=fitted_model_path,
        dataset=dataset,
        metric="ebit",
        output_dir=tmp_path / "b",
    )
    assert run_id_eps != run_id_ebit


def test_missing_feature_raises(
    fitted_model_path: Path, dataset: pd.DataFrame, tmp_path: Path
) -> None:
    bad = dataset.drop(columns=["gross_margin"])
    with pytest.raises(ValueError, match="missing required features"):
        run_forecast(
            model_path=fitted_model_path,
            dataset=bad,
            metric="eps_diluted",
            output_dir=tmp_path,
        )


def test_parquet_schema(fitted_model_path: Path, dataset: pd.DataFrame, tmp_path: Path) -> None:
    out_path, run_id = run_forecast(
        model_path=fitted_model_path,
        dataset=dataset,
        metric="eps_diluted",
        output_dir=tmp_path,
    )
    written = pd.read_parquet(out_path)
    assert set(written.columns) == {
        "security_id",
        "symbol",
        "as_of_date",
        "metric",
        "prediction",
        "model_name",
        "run_id",
    }
    assert (written["metric"] == "eps_diluted").all()
    assert (written["run_id"] == str(run_id)).all()
    assert (written["model_name"] == "LightGBM").all()


def test_output_path_under_output_dir_and_model_name_wiring(
    fitted_model_path: Path, dataset: pd.DataFrame, tmp_path: Path
) -> None:
    out_dir = tmp_path / "nested" / "preds"
    out_path, run_id = run_forecast(
        model_path=fitted_model_path,
        dataset=dataset,
        metric="eps_diluted",
        output_dir=out_dir,
        model_name="CustomLGBM",
    )
    assert out_path.parent == out_dir
    assert out_path.name == f"{run_id}.parquet"
    written = pd.read_parquet(out_path)
    assert (written["model_name"] == "CustomLGBM").all()
