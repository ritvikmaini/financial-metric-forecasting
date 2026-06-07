"""Tests for pipeline.quality_checks."""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from fmf.pipeline.quality_checks import (
    QualityCheckResult,
    any_failed,
    run_quality_checks,
)


def _write_predictions(
    path: Path,
    *,
    n: int = 20,
    nan_fraction: float = 0.0,
    mean: float = 0.0,
    std: float = 1.0,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    preds = rng.normal(loc=mean, scale=std, size=n)
    if nan_fraction > 0.0:
        idx = rng.choice(n, size=int(round(n * nan_fraction)), replace=False)
        preds[idx] = np.nan
    df = pd.DataFrame(
        {
            "security_id": [str(uuid.uuid4()) for _ in range(n)],
            "symbol": [f"S{i}" for i in range(n)],
            "as_of_date": [dt.date(2024, 5, 15)] * n,
            "metric": ["eps_diluted"] * n,
            "prediction": preds,
            "model_name": ["LightGBM"] * n,
            "run_id": ["00000000-0000-0000-0000-000000000000"] * n,
        }
    )
    df.to_parquet(path, index=False)
    return path


def test_nan_bound_passes_under_threshold(tmp_path: Path) -> None:
    p = _write_predictions(tmp_path / "p.parquet", n=20, nan_fraction=0.0)
    results = run_quality_checks(predictions_path=p)
    nan_check = next(r for r in results if r.check_name == "nan_rate")
    assert nan_check.passed
    assert not any_failed(results)


def test_nan_bound_fails_over_threshold(tmp_path: Path) -> None:
    p = _write_predictions(tmp_path / "p.parquet", n=20, nan_fraction=0.5)
    results = run_quality_checks(predictions_path=p, nan_threshold=0.05)
    nan_check = next(r for r in results if r.check_name == "nan_rate")
    assert not nan_check.passed
    assert any_failed(results)


def test_security_count_check_pass_and_fail(tmp_path: Path) -> None:
    p = _write_predictions(tmp_path / "p.parquet", n=20)
    pass_results = run_quality_checks(predictions_path=p, expected_security_count=20)
    pass_check = next(r for r in pass_results if r.check_name == "security_count")
    assert pass_check.passed

    fail_results = run_quality_checks(predictions_path=p, expected_security_count=5)
    fail_check = next(r for r in fail_results if r.check_name == "security_count")
    assert not fail_check.passed


def test_ks_drift_passes_when_distributions_match(tmp_path: Path) -> None:
    curr = _write_predictions(tmp_path / "curr.parquet", n=100, mean=0.0, std=1.0, seed=1)
    base = _write_predictions(tmp_path / "base.parquet", n=100, mean=0.0, std=1.0, seed=2)
    results = run_quality_checks(predictions_path=curr, baseline_path=base)
    drift = next(r for r in results if r.check_name == "distribution_drift")
    assert drift.passed


def test_ks_drift_fails_on_shifted_distribution(tmp_path: Path) -> None:
    curr = _write_predictions(tmp_path / "curr.parquet", n=200, mean=10.0, std=1.0, seed=1)
    base = _write_predictions(tmp_path / "base.parquet", n=200, mean=0.0, std=1.0, seed=2)
    results = run_quality_checks(predictions_path=curr, baseline_path=base)
    drift = next(r for r in results if r.check_name == "distribution_drift")
    assert not drift.passed
    assert any_failed(results)


def test_drift_skipped_without_baseline(tmp_path: Path) -> None:
    p = _write_predictions(tmp_path / "p.parquet")
    results = run_quality_checks(predictions_path=p)
    names = {r.check_name for r in results}
    assert "distribution_drift" not in names


def test_any_failed_true_false_paths() -> None:
    pass_only = [QualityCheckResult("a", True, "ok"), QualityCheckResult("b", True, "ok")]
    assert not any_failed(pass_only)
    mixed = [QualityCheckResult("a", True, "ok"), QualityCheckResult("b", False, "bad")]
    assert any_failed(mixed)
