"""Feature-cap utility: per-fold top-k by LightGBM gain.

The leakage gate is structural: this function consumes only the booster
handed in. The booster handed in must have been fitted on the current
fold's training rows only. The orchestrator's _run_fold preserves that
invariant by calling this immediately after the in-fold LightGBM.fit().
"""

from __future__ import annotations

from fmf.equity.forecasting.models.lightgbm_model import LightGBMForecaster


def top_k_feature_importance(lgbm: LightGBMForecaster, k: int) -> list[str]:
    pairs = lgbm.feature_importance(importance_type="gain")
    return [name for name, _ in pairs[:k]]
