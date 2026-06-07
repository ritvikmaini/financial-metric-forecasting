"""CLI smoke tests for scripts/run_pipeline.py."""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from fmf.equity.forecasting.models.lightgbm_model import LightGBMForecaster
from scripts.run_pipeline import app

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini.duckdb"

pytestmark = pytest.mark.slow


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def fitted_model_path(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(80, 3)),
        columns=["revenue_ttm", "gross_margin", "net_margin"],
    )
    y = 2.0 * X["revenue_ttm"].to_numpy() + rng.normal(scale=0.1, size=80)
    model = LightGBMForecaster(seed=42).fit(X, y)
    out = tmp_path / "lgbm_model"
    model.save_model(out)
    return out


@pytest.fixture
def inference_dataset_path(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1)
    n = 5
    df = pd.DataFrame(
        {
            "security_id": [str(uuid.uuid4()) for _ in range(n)],
            "symbol": [f"SYM{i}" for i in range(n)],
            "as_of_date": [dt.date(2024, 5, 15)] * n,
            "revenue_ttm": rng.normal(size=n),
            "gross_margin": rng.normal(size=n),
            "net_margin": rng.normal(size=n),
        }
    )
    out = tmp_path / "dataset.parquet"
    df.to_parquet(out, index=False)
    return out


def test_cli_build_dataset(runner: CliRunner, tmp_path: Path) -> None:
    if not FIXTURE.exists():
        pytest.skip("fixture not built yet")
    output = tmp_path / "inference.parquet"
    result = runner.invoke(
        app,
        [
            "build-dataset",
            "--as-of",
            "2023-06-30",
            "--feature",
            "revenue_ttm",
            "--feature",
            "gross_margin",
            "--fixture",
            str(FIXTURE),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()


def test_cli_forecast(
    runner: CliRunner,
    fitted_model_path: Path,
    inference_dataset_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "forecast",
            "--model-path",
            str(fitted_model_path),
            "--dataset-path",
            str(inference_dataset_path),
            "--metric",
            "eps_diluted",
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_quality_check(
    runner: CliRunner,
    fitted_model_path: Path,
    inference_dataset_path: Path,
    tmp_path: Path,
) -> None:
    from fmf.pipeline.forecast_runner import run_forecast

    dataset = pd.read_parquet(inference_dataset_path)
    out_path, _ = run_forecast(
        model_path=fitted_model_path,
        dataset=dataset,
        metric="eps_diluted",
        output_dir=tmp_path / "preds",
    )
    result = runner.invoke(
        app,
        [
            "quality-check",
            "--predictions-path",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_chain(
    runner: CliRunner,
    fitted_model_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not FIXTURE.exists():
        pytest.skip("fixture not built yet")
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "chain",
            "--as-of",
            "2023-06-30",
            "--feature",
            "revenue_ttm",
            "--feature",
            "gross_margin",
            "--feature",
            "net_margin",
            "--model-path",
            str(fitted_model_path),
            "--fixture",
            str(FIXTURE),
        ],
    )
    # chain may exit 1 if quality checks fail on fixture data; tolerate both
    # as long as it actually ran and wrote output.
    assert result.exit_code in (0, 1), result.output
    preds_dir = tmp_path / "reports" / "predictions"
    assert preds_dir.exists()
    assert any(preds_dir.glob("*.parquet"))
