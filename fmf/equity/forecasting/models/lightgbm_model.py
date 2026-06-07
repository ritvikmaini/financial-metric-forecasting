"""LightGBM wrapper for tabular financial-metrics forecasting.

Clean-room implementation: methodology ported from the proprietary FMF
(Huber objective, MAE metric, FMF_LGBM_SEED env discipline) but the code
is re-written and the trained-model numbers are derived later by the
S10 backtester on public data.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_DEFAULT_SEED = 42


def _read_seed() -> int:
    """Read FMF_LGBM_SEED from env. Fail on negative (LightGBM silently
    accepts them with unpredictable behavior). Fall back to 42 on
    non-integer with a warning."""
    raw = os.environ.get("FMF_LGBM_SEED", str(_DEFAULT_SEED))
    try:
        value = int(raw)
    except ValueError:
        log.warning("invalid FMF_LGBM_SEED=%r; falling back to %d", raw, _DEFAULT_SEED)
        return _DEFAULT_SEED
    if value < 0:
        raise ValueError(
            f"FMF_LGBM_SEED={value} must be non-negative; LightGBM "
            f"silently accepts negative seeds with unpredictable behavior."
        )
    return value


@dataclass(frozen=True, slots=True)
class LightGBMHyperparameters:
    """Locked-in defaults per the proprietary spec. Tuning happens in
    a later HP-search task; for v1 we ship the baseline."""

    # Fixed params (match proprietary _FIXED_PARAMS).
    objective: str = "huber"
    metric: str = "mae"
    verbosity: int = -1
    # Search-space defaults (single-point pick from each, NOT optimized).
    num_leaves: int = 15
    learning_rate: float = 0.05
    max_depth: int = 5
    min_child_samples: int = 20
    reg_alpha: float = 0.01
    reg_lambda: float = 0.01
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    min_split_gain: float = 0.0
    path_smooth: float = 0.0
    n_estimators: int = 200

    def as_lgb_params(self, seed: int) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "metric": self.metric,
            "verbosity": self.verbosity,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "min_split_gain": self.min_split_gain,
            "path_smooth": self.path_smooth,
            "n_estimators": self.n_estimators,
            "seed": seed,
            "deterministic": True,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
        }


class LightGBMForecaster:
    """sklearn-style LightGBM wrapper.

    Fit/predict on (X, y). Supports per-sample weights and group_ids
    (group_ids are accepted for protocol compatibility but unused by
    LightGBM itself; the backtester uses them for CV grouping).
    """

    def __init__(
        self,
        hyperparameters: LightGBMHyperparameters | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.hyperparameters = hyperparameters or LightGBMHyperparameters()
        self.seed = seed if seed is not None else _read_seed()
        self._model: lgb.Booster | None = None
        self._feature_names: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        *,
        sample_weight: np.ndarray | None = None,
        group_ids: np.ndarray | None = None,  # noqa: ARG002 (protocol parity)
    ) -> LightGBMForecaster:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be DataFrame, got {type(X).__name__}")
        y_arr = np.asarray(y, dtype=np.float64)
        if len(X) != len(y_arr):
            raise ValueError(f"X has {len(X)} rows but y has {len(y_arr)}")
        self._feature_names = list(X.columns)
        dataset = lgb.Dataset(
            data=X.values,
            label=y_arr,
            weight=sample_weight,
            feature_name=self._feature_names,
            free_raw_data=False,
        )
        params = self.hyperparameters.as_lgb_params(self.seed)
        n_rounds = int(params.pop("n_estimators"))
        self._model = lgb.train(
            params=params,
            train_set=dataset,
            num_boost_round=n_rounds,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model not fitted; call fit() first")
        if self._feature_names is not None:
            missing = set(self._feature_names) - set(X.columns)
            if missing:
                raise ValueError(f"missing features at predict time: {sorted(missing)}")
            X = X[self._feature_names]
        return np.asarray(self._model.predict(X.values), dtype=np.float64)

    def feature_names(self) -> list[str]:
        if self._feature_names is None:
            raise RuntimeError("model not fitted; call fit() first")
        return list(self._feature_names)

    def feature_importance(self, *, importance_type: str = "gain") -> list[tuple[str, float]]:
        if self._model is None:
            raise RuntimeError("model not fitted; call fit() first")
        names = self._model.feature_name()
        importances = self._model.feature_importance(importance_type=importance_type)
        pairs = sorted(zip(names, importances, strict=True), key=lambda p: -float(p[1]))
        return [(name, float(imp)) for name, imp in pairs]
