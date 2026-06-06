"""Forecasting model protocols. Mirror the proprietary shape so the
S10 backtester can dependency-inject any TabularForecaster."""

from __future__ import annotations

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
