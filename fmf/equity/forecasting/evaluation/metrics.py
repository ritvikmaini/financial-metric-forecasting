"""Pure-function forecast evaluation metrics.

Each function returns ``float | None`` (or ``dict | None`` for the
distributional helpers) and is undefined-safe: empty input, all-NaN, or a
zero/near-zero denominator yields ``None``.

APE-family metrics share ``_ape_valid_mask`` to drop near-zero |actuals|
that would dominate the percentage denominator. The filter is a data-driven
5th-percentile floor combined with an absolute ``1e-3`` floor.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

ArrayLike = Sequence[float] | NDArray[np.float64] | NDArray[np.floating]

# Drop the bottom 5% of positive |actuals| from APE metrics.
_NEAR_ZERO_QUANTILE: float = 0.05
# Absolute floor under the quantile (guards against uniformly tiny actuals).
_NEAR_ZERO_ABS_FLOOR: float = 1e-3

# Default TiRex-style quantile grid for CRPS/calibration helpers.
QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _to_1d(x: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64).ravel()


def _ape_valid_mask(
    actuals: NDArray[np.float64], predictions: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Mask of rows usable for APE: finite, finite-pred, |actual| above floor."""
    finite = np.isfinite(actuals) & np.isfinite(predictions)
    abs_actuals = np.abs(actuals)
    positive = finite & (abs_actuals > 0)
    if not positive.any():
        return np.zeros_like(actuals, dtype=bool)
    q = float(np.quantile(abs_actuals[positive], _NEAR_ZERO_QUANTILE))
    threshold = max(_NEAR_ZERO_ABS_FLOOR, q)
    return finite & (abs_actuals >= threshold)


def mape(actuals: ArrayLike, predictions: ArrayLike) -> float | None:
    """Mean Absolute Percentage Error (fraction, not pct). Near-zero filtered."""
    a = _to_1d(actuals)
    p = _to_1d(predictions)
    mask = _ape_valid_mask(a, p)
    if not mask.any():
        return None
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])))


def median_ape(actuals: ArrayLike, predictions: ArrayLike) -> float | None:
    """Median Absolute Percentage Error (fraction). Near-zero filtered."""
    a = _to_1d(actuals)
    p = _to_1d(predictions)
    mask = _ape_valid_mask(a, p)
    if not mask.any():
        return None
    return float(np.median(np.abs((a[mask] - p[mask]) / a[mask])))


def accuracy_within_pct(
    actuals: ArrayLike,
    predictions: ArrayLike,
    threshold_pct: float,
) -> float | None:
    """Fraction of |err/y| at or below ``threshold_pct`` (expressed as percent).

    ``threshold_pct=10`` means within 10%, i.e. |err/y| <= 0.10. Near-zero filtered.
    """
    if threshold_pct < 0:
        raise ValueError("threshold_pct must be non-negative")
    a = _to_1d(actuals)
    p = _to_1d(predictions)
    mask = _ape_valid_mask(a, p)
    if not mask.any():
        return None
    ape = np.abs((a[mask] - p[mask]) / a[mask])
    return float(np.mean(ape <= threshold_pct / 100.0))


def directional_accuracy(
    prev_actuals: ArrayLike,
    actuals: ArrayLike,
    predictions: ArrayLike,
) -> float | None:
    """Fraction of correct up/down direction calls vs ``prev_actuals``.

    Skips rows where ``actual == prev_actual`` (true direction undefined).
    """
    prev = _to_1d(prev_actuals)
    a = _to_1d(actuals)
    p = _to_1d(predictions)
    finite = np.isfinite(prev) & np.isfinite(a) & np.isfinite(p)
    changed = finite & (a != prev)
    if not changed.any():
        return None
    actual_dir = np.sign(a[changed] - prev[changed])
    pred_dir = np.sign(p[changed] - prev[changed])
    return float(np.mean(actual_dir == pred_dir))


def beat_miss_accuracy(
    consensus: ArrayLike,
    actuals: ArrayLike,
    predictions: ArrayLike,
) -> float | None:
    """Fraction of correct beat/miss calls vs analyst ``consensus``.

    Skips rows where ``actual == consensus`` (no ground-truth beat/miss).
    """
    c = _to_1d(consensus)
    a = _to_1d(actuals)
    p = _to_1d(predictions)
    finite = np.isfinite(c) & np.isfinite(a) & np.isfinite(p)
    differing = finite & (a != c)
    if not differing.any():
        return None
    actual_vs = np.sign(a[differing] - c[differing])
    pred_vs = np.sign(p[differing] - c[differing])
    return float(np.mean(actual_vs == pred_vs))


def signed_error_stats(actuals: ArrayLike, predictions: ArrayLike) -> dict[str, float] | None:
    """Signed percentage error distribution: mean/median/skew/over/under.

    Returns fractions (not percent). ``+ve`` SPE means over-prediction.
    Near-zero filtered.
    """
    a_arr = _to_1d(actuals)
    p_arr = _to_1d(predictions)
    mask = _ape_valid_mask(a_arr, p_arr)
    if not mask.any():
        return None
    a = a_arr[mask]
    p = p_arr[mask]
    spe = (p - a) / np.abs(a)
    # Sample skew via Fisher-Pearson (g_1) on >=3 samples; else 0.0.
    if spe.size >= 3:
        mu = float(np.mean(spe))
        sd = float(np.std(spe, ddof=0))
        skew = float(np.mean(((spe - mu) / sd) ** 3)) if sd > 0 else 0.0
    else:
        skew = 0.0
    return {
        "mean_spe": float(np.mean(spe)),
        "median_spe": float(np.median(spe)),
        "std_spe": float(np.std(spe, ddof=0)),
        "skew_spe": skew,
        "pct_over": float(np.mean(spe > 0)),
        "pct_under": float(np.mean(spe < 0)),
    }


def coverage(total_tickers: int, predicted_tickers: int) -> float | None:
    """Prediction coverage: ``predicted / total``. ``None`` when total <= 0."""
    if total_tickers <= 0:
        return None
    return predicted_tickers / total_tickers


def smape(
    actuals: ArrayLike,
    predictions: ArrayLike,
    epsilon: float = 1e-8,
) -> float | None:
    """Symmetric MAPE (fraction in [0, 2]). Skips non-finite rows."""
    a_arr = _to_1d(actuals)
    p_arr = _to_1d(predictions)
    finite = np.isfinite(a_arr) & np.isfinite(p_arr)
    if not finite.any():
        return None
    a = a_arr[finite]
    p = p_arr[finite]
    denom = np.abs(a) + np.abs(p) + epsilon
    return float(np.mean(2.0 * np.abs(a - p) / denom))


def correlation(
    actuals: ArrayLike,
    predictions: ArrayLike,
    method: Literal["pearson", "spearman"] = "pearson",
) -> float | None:
    """Pearson or Spearman correlation. ``None`` when undefined.

    Returns ``None`` if fewer than 2 finite paired rows remain, or if either
    side has zero variance (Pearson) / fewer than 2 unique values (Spearman).
    """
    a = _to_1d(actuals)
    p = _to_1d(predictions)
    finite = np.isfinite(a) & np.isfinite(p)
    if finite.sum() < 2:
        return None
    a = a[finite]
    p = p[finite]
    if method == "spearman":
        a = _rankdata_average(a)
        p = _rankdata_average(p)
    elif method != "pearson":
        raise ValueError(f"unknown method: {method!r}")
    a_var = float(np.var(a))
    p_var = float(np.var(p))
    if a_var == 0.0 or p_var == 0.0:
        return None
    cov = float(np.mean((a - a.mean()) * (p - p.mean())))
    rho = cov / float(np.sqrt(a_var * p_var))
    # Clamp for safety against fp drift.
    return float(np.clip(rho, -1.0, 1.0))


def _rankdata_average(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average ranks with ties (matches scipy.stats.rankdata default)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    # Handle ties: average rank within each tied group.
    sorted_x = x[order]
    # Find runs of equal values.
    i = 0
    n = x.size
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            avg = (i + j + 1) / 2.0  # average of ranks i+1..j
            ranks[order[i:j]] = avg
        i = j
    return ranks


def mase(
    actuals: ArrayLike,
    predictions: ArrayLike,
    naive_predictions: ArrayLike,
) -> float | None:
    """Mean Absolute Scaled Error. ``< 1`` beats the naive baseline.

    Returns ``None`` when the naive baseline error is in the signal's noise
    floor (``naive_mae < max(1e-9, 1e-4 * mean|actual|)``).
    """
    a_arr = _to_1d(actuals)
    p_arr = _to_1d(predictions)
    n_arr = _to_1d(naive_predictions)
    finite = np.isfinite(a_arr) & np.isfinite(p_arr) & np.isfinite(n_arr)
    if not finite.any():
        return None
    a = a_arr[finite]
    p = p_arr[finite]
    n = n_arr[finite]
    naive_mae = float(np.mean(np.abs(a - n)))
    signal_scale = float(np.mean(np.abs(a)))
    if naive_mae < max(1e-9, 1e-4 * signal_scale):
        return None
    return float(np.mean(np.abs(a - p)) / naive_mae)


def picp(
    actuals: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> float | None:
    """Prediction Interval Coverage Probability."""
    a_arr = _to_1d(actuals)
    lo_arr = _to_1d(lower)
    hi_arr = _to_1d(upper)
    finite = np.isfinite(a_arr) & np.isfinite(lo_arr) & np.isfinite(hi_arr)
    if not finite.any():
        return None
    a = a_arr[finite]
    lo = lo_arr[finite]
    hi = hi_arr[finite]
    return float(np.mean((a >= lo) & (a <= hi)))


def crps_quantiles(
    actuals: ArrayLike,
    quantile_preds: ArrayLike,
    tau_levels: ArrayLike,
) -> float | None:
    """CRPS approximated via the pinball-loss average over forecast quantiles.

    For each row i and quantile level ``tau_k`` with forecast ``q_{i,k}``:

        pinball_{i,k} = (tau_k - 1{actual_i < q_{i,k}}) * (q_{i,k} - actual_i)

    The per-row score is averaged over k, and the function returns the mean
    over rows. Assumes monotonically increasing tau (the standard linear
    interpolation identity for CRPS as the integrated pinball loss).
    """
    a = _to_1d(actuals)
    taus = _to_1d(tau_levels)
    q = np.asarray(quantile_preds, dtype=np.float64)
    if q.ndim == 1:
        q = q.reshape(-1, 1) if taus.size == 1 else q.reshape(1, -1)
    if a.size == 0 or q.size == 0 or taus.size == 0:
        return None
    if q.shape != (a.size, taus.size):
        raise ValueError(
            f"quantile_preds shape {q.shape} must equal (len(actuals), len(tau_levels))"
            f" = ({a.size}, {taus.size})"
        )
    row_finite = np.isfinite(a) & np.all(np.isfinite(q), axis=1)
    if not row_finite.any():
        return None
    a_v = a[row_finite][:, None]
    q_v = q[row_finite]
    indicator = (a_v < q_v).astype(np.float64)
    pinball = (taus[None, :] - indicator) * (q_v - a_v)
    per_row = pinball.mean(axis=1)
    return float(np.mean(per_row))


def quantile_calibration(
    actuals: ArrayLike,
    quantile_preds: ArrayLike,
    tau_levels: ArrayLike,
) -> dict[float, float] | None:
    """Empirical coverage P(actual <= q_tau) per requested tau."""
    a = _to_1d(actuals)
    taus = _to_1d(tau_levels)
    q = np.asarray(quantile_preds, dtype=np.float64)
    if q.ndim == 1:
        q = q.reshape(-1, 1) if taus.size == 1 else q.reshape(1, -1)
    if a.size == 0 or q.size == 0 or taus.size == 0:
        return None
    if q.shape != (a.size, taus.size):
        raise ValueError(
            f"quantile_preds shape {q.shape} must equal (len(actuals), len(tau_levels))"
            f" = ({a.size}, {taus.size})"
        )
    row_finite = np.isfinite(a) & np.all(np.isfinite(q), axis=1)
    if not row_finite.any():
        return None
    a_v = a[row_finite][:, None]
    q_v = q[row_finite]
    observed = (a_v <= q_v).mean(axis=0)
    return {float(tau): float(rate) for tau, rate in zip(taus, observed, strict=True)}
