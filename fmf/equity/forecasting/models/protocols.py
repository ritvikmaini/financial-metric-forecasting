"""Forecasting model protocols. Mirror the proprietary shape so the
S10 backtester can dependency-inject any TabularForecaster."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class TabularForecaster(Protocol):
    """Any model that trains on X/y and predicts on X."""

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        *,
        sample_weight: np.ndarray | None = None,
        group_ids: np.ndarray | None = None,
    ) -> TabularForecaster: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class TirexOutput:
    """TiRex zero-shot forecast output: per-horizon 9-quantile predictions.

    quantiles shape: (horizon, 9). Each row is the 9-quantile vector at
    that horizon step, in the order QUANTILE_LEVELS = (0.1, ..., 0.9).
    """

    quantiles: np.ndarray
    horizon: int


class TimeSeriesForecaster(Protocol):
    """Any univariate time-series forecaster (zero-shot or trained)."""

    def predict(self, series: np.ndarray, *, horizon: int) -> TirexOutput: ...
