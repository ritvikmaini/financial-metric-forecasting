"""S11 naive + statistical baselines for the FMF backtester.

For the v1.0 annual-FY cadence:
- random_walk, seasonal_naive, last_year_actual_baseline all reduce to "the
  most recent FY actual disclosed at as_of." They ship as distinct functions
  to match the spec §4 scoreboard, with a single shared PIT mechanism.
- AR(1) is the only baseline with a fitted parameter. Per-fold pooled fit
  via OLS on (y_t, y_{t-1}) pairs from the fold's training rows; applied to
  test rows as y_hat = phi * y_{t-1}.

Per-fold PIT discipline mirrors the feature-cap pattern: fit_ar1_pooled
consumes only the fold's training rows. _run_fold preserves the invariant
by calling it immediately after building train_y from fold-train rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MIN_AR1_PAIRS = 5  # OLS floor; below this, phi defaults to 1.0 (random walk).


def random_walk(prev_actual: float | None) -> float | None:
    """y_hat = y_{t-1}."""
    return None if prev_actual is None or not np.isfinite(prev_actual) else float(prev_actual)


def seasonal_naive(prev_actual: float | None, *, season_length: int = 1) -> float | None:
    """For annual data with season_length=1, collapses to random_walk. Kept
    as a distinct function so the scoreboard narrative is unambiguous."""
    if season_length != 1:
        raise NotImplementedError("v1.0 ships annual cadence; season_length must be 1")
    return random_walk(prev_actual)


def last_year_actual_baseline(prev_actual: float | None) -> float | None:
    """Alias for random_walk in the annual cadence. Distinct scoreboard row."""
    return random_walk(prev_actual)


def fit_ar1_pooled(train_pairs: pd.DataFrame) -> float:
    """Pooled-OLS AR(1) on (y, y_lag) pairs from this fold's training rows.

    train_pairs columns required: 'y' (current FY actual), 'y_lag' (the
    naive_baseline at that row's as_of, i.e., last_fy_actual). Drops any
    row where either is non-finite.

    Returns phi via closed-form OLS: phi = sum(y * y_lag) / sum(y_lag^2).
    Below _MIN_AR1_PAIRS finite pairs, returns 1.0 (degenerates to random
    walk).
    """
    if "y" not in train_pairs.columns or "y_lag" not in train_pairs.columns:
        raise ValueError("train_pairs must have columns 'y' and 'y_lag'")
    df = train_pairs.dropna(subset=["y", "y_lag"])
    df = df[np.isfinite(df["y"]) & np.isfinite(df["y_lag"])]
    if len(df) < _MIN_AR1_PAIRS:
        return 1.0
    y = df["y"].to_numpy(dtype=np.float64)
    y_lag = df["y_lag"].to_numpy(dtype=np.float64)
    denom = float(np.sum(y_lag * y_lag))
    if denom <= 0:
        return 1.0
    return float(np.sum(y * y_lag) / denom)


def predict_ar1(phi: float, prev_actual: float | None) -> float | None:
    if prev_actual is None or not np.isfinite(prev_actual):
        return None
    return float(phi * prev_actual)
