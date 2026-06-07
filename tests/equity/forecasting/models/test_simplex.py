"""Simplex projection tests (Duchi et al. 2008)."""

from __future__ import annotations

import numpy as np
import pytest

from fmf.equity.forecasting.models._simplex import (
    project_to_bounded_simplex,
    project_to_simplex,
    project_to_simplex_with_budget,
)


def test_project_already_on_simplex_is_identity() -> None:
    v = np.array([0.3, 0.5, 0.2])
    out = project_to_simplex(v)
    np.testing.assert_array_almost_equal(out, v)


def test_project_sums_to_one() -> None:
    rng = np.random.default_rng(0)
    for _ in range(5):
        v = rng.normal(size=10)
        w = project_to_simplex(v)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()


def test_project_negative_vector() -> None:
    v = np.array([-1.0, -2.0, -3.0])
    w = project_to_simplex(v)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()
    # Largest input gets the mass.
    assert w[0] >= w[1] >= w[2]


def test_project_with_budget() -> None:
    v = np.array([1.0, 0.5, 0.5])
    w = project_to_simplex_with_budget(v, budget=2.0)
    assert w.sum() == pytest.approx(2.0)
    assert (w >= 0).all()


def test_project_bounded_respects_lower() -> None:
    v = np.array([1.0, 0.5, 0.1])
    lower = np.array([0.0, 0.0, 0.4])
    w = project_to_bounded_simplex(v, lower)
    assert w.sum() == pytest.approx(1.0)
    assert w[2] >= 0.4 - 1e-9
    assert (w >= lower - 1e-9).all()


def test_project_bounded_infeasible_falls_back() -> None:
    v = np.array([1.0, 0.5, 0.5])
    lower = np.array([0.5, 0.5, 0.5])  # sum = 1.5 > 1, infeasible
    w = project_to_bounded_simplex(v, lower)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


def test_project_bounded_is_substitution_not_clamp_renormalize() -> None:
    # Regression guard: when the floor binds, the bounded-simplex projection
    # must re-optimise the free coordinates under the binding floor rather
    # than clamp the constrained coordinate and rescale the others.
    #
    # Setup: unfloored optimum v* = (0.60, 0.40, 0.00), lower = (0, 0, 0.30).
    # Substitution projection: shift v* by lower, project (0.60, 0.40, -0.30)
    # onto sum=0.70 simplex with w >= 0, then add lower back.
    # By hand: rho=1, theta=0.15 -> (0.45, 0.25, 0.30).
    # Clamp+renormalize would scale (0.60, 0.40) to sum 0.70 preserving the
    # 0.6/0.4 = 1.5 ratio -> (0.42, 0.28, 0.30). The two outcomes differ on
    # the free coordinates and discriminate the projection routine.
    v = np.array([0.60, 0.40, 0.00])
    lower = np.array([0.0, 0.0, 0.30])
    w = project_to_bounded_simplex(v, lower)
    np.testing.assert_array_almost_equal(w, np.array([0.45, 0.25, 0.30]), decimal=6)
    assert w[0] != pytest.approx(0.42, abs=1e-3)
    assert w[1] != pytest.approx(0.28, abs=1e-3)
