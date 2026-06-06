"""Preprocessing helpers for TiRex zero-shot forecasting.

Clean-room implementation of three methodology pieces ported from the
proprietary FMF:
- MAD-based winsorization (robust outlier capping at k * 1.4826 * MAD).
- Monotonic-quantile enforcement (sort each horizon position's quantile
  triple to fix quantile crossing).
- Linear interpolation over a 9-quantile grid for arbitrary target
  quantile levels.

These are pure-numeric utilities with no model dependency, so they live
in their own module and are unit-tested independently of TiRex itself.
"""

from __future__ import annotations

import numpy as np

QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_QUANTILE_LEVELS_ARR = np.asarray(QUANTILE_LEVELS, dtype=float)
ROBUST_OUTLIER_CAP_DEFAULT = 5.0


def winsorize_mad(series: np.ndarray, cap: float = ROBUST_OUTLIER_CAP_DEFAULT) -> np.ndarray:
    """Clip series values to median +/- cap * 1.4826 * MAD.

    Pass-through if cap <= 0, fewer than 4 finite values, or MAD == 0.
    Preserves NaN positions.
    """
    if cap <= 0:
        return series
    finite = series[~np.isnan(series)]
    if finite.size < 4:
        return series
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if mad <= 0:
        return series
    half_width = cap * 1.4826 * mad
    out = series.copy()
    valid_mask = ~np.isnan(out)
    out[valid_mask] = np.clip(out[valid_mask], median - half_width, median + half_width)
    return out


def enforce_monotonic_quantiles(
    lower: np.ndarray, point: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort each horizon position's (lower, point, upper) triple ascending.

    Removes quantile crossing without changing the marginal values.
    """
    stacked = np.sort(np.stack([np.asarray(x, dtype=float) for x in (lower, point, upper)]), axis=0)
    return stacked[0], stacked[1], stacked[2]


def interp_quantile(q_arr: np.ndarray, target: float) -> np.ndarray:
    """Linear interpolation over the 9-quantile grid along the last axis.

    q_arr shape (..., 9). Returns shape (...,) with values interpolated
    at `target` in [0.1, 0.9]. Clamps outside the grid to nearest endpoint.
    """
    q_arr = np.asarray(q_arr, dtype=float)
    idx = int(
        np.clip(np.searchsorted(_QUANTILE_LEVELS_ARR, target) - 1, 0, len(QUANTILE_LEVELS) - 2)
    )
    lo = _QUANTILE_LEVELS_ARR[idx]
    hi = _QUANTILE_LEVELS_ARR[idx + 1]
    t = float(np.clip((target - lo) / (hi - lo), 0.0, 1.0))
    return (1 - t) * q_arr[..., idx] + t * q_arr[..., idx + 1]
