"""Pipeline-stage 3: invariants on the predictions parquet."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_DEFAULT_NAN_THRESHOLD = 0.05


@dataclass
class QualityCheckResult:
    check_name: str
    passed: bool
    message: str
    details: dict[str, object] = field(default_factory=dict)


def run_quality_checks(
    *,
    predictions_path: str | Path,
    expected_security_count: int | None = None,
    nan_threshold: float = _DEFAULT_NAN_THRESHOLD,
    baseline_path: str | Path | None = None,
) -> list[QualityCheckResult]:
    df = pd.read_parquet(predictions_path)
    results: list[QualityCheckResult] = []
    nan_rate = float(df["prediction"].isna().mean())
    results.append(
        QualityCheckResult(
            check_name="nan_rate",
            passed=nan_rate <= nan_threshold,
            message=f"NaN rate {nan_rate:.3f} (threshold {nan_threshold:.3f})",
            details={"nan_rate": nan_rate, "threshold": nan_threshold},
        )
    )
    if expected_security_count is not None:
        actual = int(df["security_id"].nunique())
        passed = actual == expected_security_count
        results.append(
            QualityCheckResult(
                check_name="security_count",
                passed=passed,
                message=f"got {actual} distinct securities; expected {expected_security_count}",
                details={"actual": actual, "expected": expected_security_count},
            )
        )
    if baseline_path is not None:
        from scipy.stats import ks_2samp

        baseline = pd.read_parquet(baseline_path)
        curr_finite = df["prediction"].dropna().to_numpy(dtype=np.float64)
        base_finite = baseline["prediction"].dropna().to_numpy(dtype=np.float64)
        if len(curr_finite) >= 5 and len(base_finite) >= 5:
            ks_stat, p_value = ks_2samp(curr_finite, base_finite)
            passed = bool(p_value > 0.01)
            results.append(
                QualityCheckResult(
                    check_name="distribution_drift",
                    passed=passed,
                    message=f"KS p-value {p_value:.4f} (threshold 0.01)",
                    details={"ks_stat": float(ks_stat), "p_value": float(p_value)},
                )
            )
    return results


def any_failed(results: list[QualityCheckResult]) -> bool:
    return any(not r.passed for r in results)
