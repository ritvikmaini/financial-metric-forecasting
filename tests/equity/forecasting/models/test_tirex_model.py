"""TiRex wrapper tests using the fixture backend (hermetic)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fmf.equity.forecasting.models._tirex_preprocessing import (
    QUANTILE_LEVELS,
    enforce_monotonic_quantiles,
    interp_quantile,
    winsorize_mad,
)
from fmf.equity.forecasting.models.tirex_model import (
    MIN_CONTEXT_LENGTH,
    FixtureNotFoundError,
    TirexFixtureBackend,
    TirexForecaster,
)

FIXTURE_DIR = Path(__file__).parents[4] / "tests" / "fixtures" / "tirex_outputs"


def test_winsorize_clips_outlier() -> None:
    rng = np.random.default_rng(0)
    s = np.linspace(100.0, 200.0, 24) + rng.normal(scale=3.0, size=24)
    s[3] = 1000.0
    clipped = winsorize_mad(s, cap=5.0)
    # Outlier (1000.0) must be MAD-clipped; threshold loose enough to absorb
    # noise-driven MAD variability while still proving the clip happened.
    assert clipped[3] < 500.0, "outlier not clipped"
    assert clipped[3] < s[3], "outlier value unchanged"


def test_winsorize_pass_through_on_small_series() -> None:
    s = np.array([1.0, 2.0, 3.0])
    out = winsorize_mad(s, cap=5.0)
    np.testing.assert_array_equal(out, s)


def test_winsorize_preserves_nans() -> None:
    s = np.array([1.0, np.nan, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    out = winsorize_mad(s, cap=3.0)
    assert np.isnan(out[1])


def test_enforce_monotonic_sorts_per_position() -> None:
    lower = np.array([10.0, 20.0])
    point = np.array([5.0, 25.0])
    upper = np.array([15.0, 18.0])
    new_l, new_p, new_u = enforce_monotonic_quantiles(lower, point, upper)
    assert (new_l <= new_p).all()
    assert (new_p <= new_u).all()


def test_interp_quantile_at_grid_levels() -> None:
    q = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
    for i, level in enumerate(QUANTILE_LEVELS):
        v = interp_quantile(q, level)
        assert v[0] == pytest.approx(float(i + 1))


def test_interp_quantile_between_levels() -> None:
    q = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
    v = interp_quantile(q, 0.15)
    assert v[0] == pytest.approx(1.5)


def test_fixture_backend_loads_committed_fixture() -> None:
    if not FIXTURE_DIR.exists() or not list(FIXTURE_DIR.glob("*.json")):
        pytest.skip("TiRex fixtures not generated")
    backend = TirexFixtureBackend(fixture_dir=FIXTURE_DIR)
    series = np.linspace(100.0, 220.0, 24)
    q = backend.forecast(series, 4)
    assert q.shape == (4, 9)


def test_forecaster_raises_below_min_context() -> None:
    if not FIXTURE_DIR.exists():
        pytest.skip("TiRex fixtures not generated")
    fc = TirexForecaster(backend=TirexFixtureBackend(fixture_dir=FIXTURE_DIR))
    short = np.arange(MIN_CONTEXT_LENGTH - 1, dtype=np.float64)
    with pytest.raises(ValueError, match="requires at least"):
        fc.predict(short, horizon=4)


def test_forecaster_normal_series_returns_monotonic_quantiles() -> None:
    if not FIXTURE_DIR.exists():
        pytest.skip("TiRex fixtures not generated")
    fc = TirexForecaster(backend=TirexFixtureBackend(fixture_dir=FIXTURE_DIR))
    series = np.linspace(100.0, 220.0, 24)
    out = fc.predict(series, horizon=4)
    assert out.quantiles.shape == (4, 9)
    # Each horizon row must be monotonic across quantiles.
    diffs = np.diff(out.quantiles, axis=1)
    assert (diffs >= 0).all(), "quantiles not monotonic"


def test_fixture_backend_raises_on_unknown_series() -> None:
    if not FIXTURE_DIR.exists():
        pytest.skip("TiRex fixtures not generated")
    backend = TirexFixtureBackend(fixture_dir=FIXTURE_DIR)
    series = np.random.default_rng(99).normal(size=24)  # not in fixtures
    with pytest.raises(FixtureNotFoundError):
        backend.forecast(series, 4)
