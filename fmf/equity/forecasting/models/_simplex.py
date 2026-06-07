"""Simplex projection helpers (Duchi et al. 2008, arXiv:0802.0814).

Pure numeric utilities used by SimplexBlender; no model dependency.
Exact O(k log k) projection onto the probability simplex and its
budget / bounded variants.
"""

from __future__ import annotations

import numpy as np


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project onto {w : w >= 0, sum(w) = 1}. O(k log k)."""
    v = np.asarray(v, dtype=np.float64)
    n = v.shape[0]
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    cond = u - cssv / idx > 0
    if not cond.any():
        return np.full(n, 1.0 / n, dtype=np.float64)
    rho = int(np.where(cond)[0][-1])
    theta = cssv[rho] / (rho + 1)
    return np.asarray(np.maximum(v - theta, 0.0), dtype=np.float64)


def project_to_simplex_with_budget(v: np.ndarray, budget: float) -> np.ndarray:
    """Project onto {w : w >= 0, sum(w) = budget}. budget > 0."""
    v = np.asarray(v, dtype=np.float64)
    if budget <= 0:
        return np.zeros_like(v)
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    cond_mask = u * np.arange(1, n + 1) > (cssv - budget)
    rho = int(np.nonzero(cond_mask)[0][-1])
    theta = (cssv[rho] - budget) / (rho + 1.0)
    return np.asarray(np.maximum(v - theta, 0.0), dtype=np.float64)


def project_to_bounded_simplex(v: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Project onto {w : w >= lower, sum(w) = 1}.

    Generalises project_to_simplex to per-dimension lower bounds.
    Falls back to the standard simplex if lower bounds are infeasible
    (sum(lower) > 1).
    """
    v = np.asarray(v, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    n = len(v)
    if n == 0:
        return v
    budget = 1.0 - float(np.sum(lower))
    if budget < -1e-9:
        return project_to_simplex(v)
    shifted = v - lower
    projected = project_to_simplex_with_budget(shifted, budget)
    return np.asarray(projected + lower, dtype=np.float64)
