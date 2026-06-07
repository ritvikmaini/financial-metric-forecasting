"""SimplexBlender + MetaLearner tests."""

from __future__ import annotations

import numpy as np
import pytest

from fmf.equity.forecasting.models.meta_learner import (
    DEFAULT_CONSENSUS_FLOOR,
    MIN_SAMPLES,
    SIGNAL_NAMES,
    MetaLearner,
    SimplexBlender,
)


def test_blender_recovers_single_signal() -> None:
    """If X[:, 0] == y exactly, the blender should put all weight on col 0."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=100)
    noise = rng.normal(scale=5.0, size=100)
    X = np.column_stack([y, y + noise, y + 2 * noise])
    blender = SimplexBlender().fit(X, y)
    assert blender.coef_ is not None
    assert blender.coef_[0] > 0.9


def test_blender_weights_sum_to_one() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    y = rng.normal(size=50)
    blender = SimplexBlender().fit(X, y)
    assert blender.coef_ is not None
    assert blender.coef_.sum() == pytest.approx(1.0, abs=1e-6)
    assert (blender.coef_ >= 0).all()


def test_blender_predict_matches_dot_product() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 3))
    y = rng.normal(size=30)
    blender = SimplexBlender().fit(X, y)
    np.testing.assert_array_almost_equal(blender.predict(X), X @ blender.coef_)


def test_meta_learner_respects_consensus_floor() -> None:
    rng = np.random.default_rng(3)
    n = 100
    lgbm = rng.normal(size=n)
    tirex = rng.normal(size=n)
    cons = rng.normal(scale=10.0, size=n)  # very noisy consensus
    y = lgbm + 0.1 * rng.normal(size=n)  # truth follows lgbm
    learner = MetaLearner(consensus_floor=DEFAULT_CONSENSUS_FLOOR).train(lgbm, tirex, cons, y)
    weights = learner.get_weights()
    assert set(weights) == set(SIGNAL_NAMES)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert weights["consensus"] >= DEFAULT_CONSENSUS_FLOOR - 1e-6, (
        f"consensus floor {DEFAULT_CONSENSUS_FLOOR} not enforced; weights={weights}"
    )


def test_meta_learner_zero_floor_lets_consensus_drop_out() -> None:
    rng = np.random.default_rng(4)
    n = 100
    lgbm = rng.normal(size=n)
    tirex = rng.normal(size=n)
    cons = rng.normal(scale=10.0, size=n)
    y = lgbm + 0.1 * rng.normal(size=n)
    learner = MetaLearner(consensus_floor=0.0).train(lgbm, tirex, cons, y)
    weights = learner.get_weights()
    assert weights["consensus"] < 0.1


def test_meta_learner_predict_uses_trained_weights() -> None:
    rng = np.random.default_rng(5)
    n = 100
    lgbm = rng.normal(size=n)
    tirex = rng.normal(size=n)
    cons = rng.normal(size=n)
    y = 0.5 * lgbm + 0.5 * cons + 0.05 * rng.normal(size=n)
    learner = MetaLearner(consensus_floor=0.0).train(lgbm, tirex, cons, y)
    preds = learner.predict(lgbm[:5], tirex[:5], cons[:5])
    w = learner.get_weights()
    expected = w["lgbm"] * lgbm[:5] + w["tirex"] * tirex[:5] + w["consensus"] * cons[:5]
    np.testing.assert_array_almost_equal(preds, expected)


def test_meta_learner_drops_nan_rows() -> None:
    n = 50
    lgbm = np.linspace(0, 1, n)
    tirex = np.linspace(0, 1, n)
    cons = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    # Inject NaN in one row of each input.
    lgbm[5] = np.nan
    tirex[10] = np.nan
    learner = MetaLearner(consensus_floor=0.0).train(lgbm, tirex, cons, y)
    w = learner.get_weights()
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)


def test_meta_learner_rejects_short_training_set() -> None:
    short = np.array([1.0, 2.0])  # 2 < MIN_SAMPLES=3
    with pytest.raises(ValueError, match="at least"):
        MetaLearner().train(short, short, short, short)


def test_meta_learner_rejects_signal_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        MetaLearner().train(np.zeros(10), np.zeros(9), np.zeros(10), np.zeros(10))


def test_meta_learner_rejects_invalid_floor() -> None:
    with pytest.raises(ValueError, match="must be in"):
        MetaLearner(consensus_floor=1.5)
    with pytest.raises(ValueError, match="must be in"):
        MetaLearner(consensus_floor=-0.1)


def test_min_samples_constant_value() -> None:
    assert MIN_SAMPLES == 3


def test_consensus_floor_is_substitution_optimum_not_clamp() -> None:
    # Cardinal regression: when the unfloored optimum wants consensus < 0.30
    # with non-trivial weight on the other signals, the trained weights must
    # be the floored-simplex substitution optimum, not clamp-and-renormalize.
    #
    # Orthonormal one-hot basis so the math reduces to a tractable 2D QP:
    #   lgbm = e1, tirex = e2, consensus = e3, y = 0.6 e1 + 0.4 e2.
    # Unfloored optimum: (0.60, 0.40, 0.00).
    # Constrained optimum at floor 0.30:
    #   minimise (w_lgbm - 0.6)^2 + (w_tirex - 0.4)^2 + w_cons^2
    #   s.t. w_lgbm + w_tirex + w_cons = 1, w_cons >= 0.30, w >= 0
    #   -> w_cons binds at 0.30, Lagrange on the sum gives lambda = 0.15,
    #      so w_lgbm = 0.45, w_tirex = 0.25.
    # Clamp+renormalize would preserve the 0.6/0.4 ratio under sum 0.70
    #   -> w_lgbm = 0.42, w_tirex = 0.28.
    lgbm = np.array([1.0, 0.0, 0.0])
    tirex = np.array([0.0, 1.0, 0.0])
    cons = np.array([0.0, 0.0, 1.0])
    y = np.array([0.60, 0.40, 0.00])
    learner = MetaLearner(consensus_floor=0.30).train(lgbm, tirex, cons, y)
    w = learner.get_weights()
    assert w["consensus"] == pytest.approx(0.30, abs=1e-4)
    assert w["lgbm"] == pytest.approx(0.45, abs=1e-4)
    assert w["tirex"] == pytest.approx(0.25, abs=1e-4)
    # Explicit rejection of the clamp+renormalize outcome.
    assert w["lgbm"] != pytest.approx(0.42, abs=1e-3)
    assert w["tirex"] != pytest.approx(0.28, abs=1e-3)
