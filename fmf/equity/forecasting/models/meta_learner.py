"""Simplex-blended meta-learner over (LightGBM, TiRex, consensus).

The blender solves argmin ||Xw - y||^2 s.t. w >= 0, sum(w) = 1 via
projected gradient descent. The bounded-simplex variant enforces a
minimum weight on the consensus signal (default 0.30) so that analyst
consensus always contributes — useful when the trained models drift
during regime change.

Methodology ports from the proprietary FMF; numbers (e.g., the
consensus-floor calibration) are re-derived later by the S10 backtester.
"""

from __future__ import annotations

import logging

import numpy as np

from fmf.equity.forecasting.models._simplex import (
    project_to_bounded_simplex,
    project_to_simplex,
)

log = logging.getLogger(__name__)

SIGNAL_NAMES: tuple[str, str, str] = ("lgbm", "tirex", "consensus")
MIN_SAMPLES = 3
DEFAULT_CONSENSUS_FLOOR = 0.30


class SimplexBlender:
    """Projected-gradient solver for argmin ||Xw - y||^2 s.t. w >= 0,
    sum(w) = 1, optionally w >= lower_bounds.

    sklearn-style: fit / predict / coef_. The convex-combination
    constraint keeps the prediction in the input signals' hull, the
    natural blending invariant.
    """

    MAX_ITER = 500
    TOL = 1e-8

    def __init__(self, lower_bounds: np.ndarray | None = None) -> None:
        self.coef_: np.ndarray | None = None
        self._lower_bounds = lower_bounds

    def _project(self, v: np.ndarray) -> np.ndarray:
        if self._lower_bounds is not None:
            return project_to_bounded_simplex(v, self._lower_bounds)
        return project_to_simplex(v)

    def fit(self, X: np.ndarray, y: np.ndarray) -> SimplexBlender:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        # Initialize uniformly on the simplex; project lifts to bounded
        # simplex if lower bounds are set.
        w = self._project(np.full(n_features, 1.0 / n_features))
        # Lipschitz constant estimate of grad: 2 ||X^T X||_op.
        gram = X.T @ X
        try:
            L = 2.0 * float(np.linalg.norm(gram, ord=2))
        except np.linalg.LinAlgError:
            L = 2.0 * float(np.trace(gram))
        if L <= 0:
            L = 1.0
        step = 1.0 / L

        prev_loss = float("inf")
        for _ in range(self.MAX_ITER):
            grad = 2.0 * (X.T @ (X @ w - y))
            w_new = self._project(w - step * grad)
            loss = float(np.sum((X @ w_new - y) ** 2))
            if abs(prev_loss - loss) < self.TOL:
                w = w_new
                break
            prev_loss = loss
            w = w_new
        self.coef_ = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("blender not fitted; call fit() first")
        return np.asarray(np.asarray(X, dtype=np.float64) @ self.coef_, dtype=np.float64)


class MetaLearner:
    """Three-signal simplex-blended ensemble.

    Inputs: (lgbm_preds, tirex_preds, consensus, actuals). Order is
    FIXED — the blender's coef_ aligns to SIGNAL_NAMES.

    Consensus floor: the consensus signal is forced to carry at least
    `consensus_floor` (default 0.30) of the weight, ensuring analyst
    consensus always contributes to the prediction. Set to 0.0 to
    disable.
    """

    def __init__(self, consensus_floor: float = DEFAULT_CONSENSUS_FLOOR) -> None:
        if not 0.0 <= consensus_floor < 1.0:
            raise ValueError(f"consensus_floor must be in [0, 1); got {consensus_floor}")
        self.consensus_floor = consensus_floor
        lower = np.array([0.0, 0.0, consensus_floor], dtype=np.float64)
        self._blender = SimplexBlender(lower_bounds=lower if consensus_floor > 0 else None)

    def train(
        self,
        lgbm_preds: np.ndarray,
        tirex_preds: np.ndarray,
        consensus: np.ndarray,
        actuals: np.ndarray,
    ) -> MetaLearner:
        lgbm = np.asarray(lgbm_preds, dtype=np.float64)
        tirex = np.asarray(tirex_preds, dtype=np.float64)
        cons = np.asarray(consensus, dtype=np.float64)
        y = np.asarray(actuals, dtype=np.float64)
        if not (len(lgbm) == len(tirex) == len(cons) == len(y)):
            raise ValueError(
                f"signal length mismatch: lgbm={len(lgbm)}, tirex={len(tirex)}, "
                f"consensus={len(cons)}, actuals={len(y)}"
            )
        if len(y) < MIN_SAMPLES:
            raise ValueError(f"need at least {MIN_SAMPLES} samples; got {len(y)}")
        # Drop rows where any input is NaN — the simplex solver assumes
        # finite signals. The proprietary code logs a skew audit; v1
        # logs the count only.
        valid = np.isfinite(lgbm) & np.isfinite(tirex) & np.isfinite(cons) & np.isfinite(y)
        n_dropped = int((~valid).sum())
        if n_dropped > 0:
            log.warning("MetaLearner.train dropped %d rows with NaN", n_dropped)
        X = np.column_stack([lgbm[valid], tirex[valid], cons[valid]])
        self._blender.fit(X, y[valid])
        return self

    def predict(
        self,
        lgbm_pred: np.ndarray,
        tirex_pred: np.ndarray,
        consensus: np.ndarray,
    ) -> np.ndarray:
        X = np.column_stack(
            [
                np.asarray(lgbm_pred, dtype=np.float64),
                np.asarray(tirex_pred, dtype=np.float64),
                np.asarray(consensus, dtype=np.float64),
            ]
        )
        return self._blender.predict(X)

    def get_weights(self) -> dict[str, float]:
        if self._blender.coef_ is None:
            raise RuntimeError("MetaLearner not trained")
        return dict(zip(SIGNAL_NAMES, self._blender.coef_.tolist(), strict=True))
