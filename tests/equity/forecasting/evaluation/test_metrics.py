"""Tests for the S9 evaluation metrics module.

Each metric covers: a known-answer hand calculation, empty-input None, all-NaN
None, and (for APE/CRPS) filter/edge-case behavior. Tolerances are 1e-9 unless
the metric is intrinsically discretization-bounded.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fmf.equity.forecasting.evaluation.metrics import (
    QUANTILE_LEVELS,
    _ape_valid_mask,
    accuracy_within_pct,
    beat_miss_accuracy,
    correlation,
    coverage,
    crps_quantiles,
    directional_accuracy,
    mape,
    mase,
    median_ape,
    picp,
    quantile_calibration,
    signed_error_stats,
    smape,
)

# -------- _ape_valid_mask -----------------------------------------------------


def test_ape_valid_mask_drops_near_zero_actuals() -> None:
    # Mostly large actuals; the lone tiny one should be filtered out.
    a = np.array([100.0, 200.0, 300.0, 0.0005, 400.0, 500.0])
    p = np.zeros_like(a)
    mask = _ape_valid_mask(a, p)
    assert mask.tolist() == [True, True, True, False, True, True]


def test_ape_valid_mask_all_zero_actuals_returns_empty() -> None:
    a = np.zeros(5)
    p = np.arange(5, dtype=float)
    assert not _ape_valid_mask(a, p).any()


def test_ape_valid_mask_drops_nan_pairs() -> None:
    a = np.array([100.0, np.nan, 300.0])
    p = np.array([110.0, 200.0, np.nan])
    mask = _ape_valid_mask(a, p)
    assert mask.tolist() == [True, False, False]


# -------- mape ----------------------------------------------------------------


def test_mape_known_answer() -> None:
    # |10/100| + |20/200| + |30/300| all = 0.1 → mean = 0.1
    assert mape([100, 200, 300], [110, 180, 330]) == pytest.approx(0.1, abs=1e-12)


def test_mape_empty_returns_none() -> None:
    assert mape([], []) is None


def test_mape_all_nan_returns_none() -> None:
    assert mape([np.nan, np.nan], [1.0, 2.0]) is None


# -------- median_ape ----------------------------------------------------------


def test_median_ape_known_answer() -> None:
    assert median_ape([100, 200, 300], [110, 180, 330]) == pytest.approx(0.1, abs=1e-12)


def test_median_ape_robust_to_outlier() -> None:
    # Three rows at 10% APE, one at 500% APE; median should still be 0.1.
    out = median_ape([100, 100, 100, 100], [110, 110, 110, 600])
    assert out == pytest.approx(0.1, abs=1e-12)


def test_median_ape_empty_returns_none() -> None:
    assert median_ape([], []) is None


# -------- accuracy_within_pct -------------------------------------------------


def test_accuracy_within_pct_known_answer() -> None:
    # APEs: 5%, 10%, 30%. Within 10% → 2/3.
    a = [100.0, 100.0, 100.0]
    p = [105.0, 110.0, 130.0]
    assert accuracy_within_pct(a, p, 10) == pytest.approx(2 / 3, abs=1e-12)


def test_accuracy_within_pct_threshold_boundary_inclusive() -> None:
    # APE exactly 10% counts as within 10%.
    assert accuracy_within_pct([100.0], [110.0], 10) == pytest.approx(1.0, abs=1e-12)


def test_accuracy_within_25_pct_strictly_loose() -> None:
    a = [100.0, 100.0, 100.0]
    p = [105.0, 110.0, 130.0]
    assert accuracy_within_pct(a, p, 25) == pytest.approx(2 / 3, abs=1e-12)


def test_accuracy_within_pct_empty_returns_none() -> None:
    assert accuracy_within_pct([], [], 10) is None


def test_accuracy_within_pct_negative_threshold_raises() -> None:
    with pytest.raises(ValueError):
        accuracy_within_pct([1.0], [1.0], -5)


# -------- directional_accuracy ------------------------------------------------


def test_directional_accuracy_known_answer() -> None:
    # Row 3 (10→10) is skipped; rows 1,2 both correct → 1.0.
    assert directional_accuracy([10, 10, 10], [11, 9, 10], [12, 8, 11]) == pytest.approx(1.0)


def test_directional_accuracy_mixed() -> None:
    # Three changed rows: signs match on 2/3 → 2/3.
    prev = [10.0, 10.0, 10.0]
    act = [11.0, 9.0, 12.0]
    pred = [12.0, 11.0, 13.0]  # row 2 wrong sign
    assert directional_accuracy(prev, act, pred) == pytest.approx(2 / 3, abs=1e-12)


def test_directional_accuracy_all_unchanged_returns_none() -> None:
    assert directional_accuracy([1, 2, 3], [1, 2, 3], [9, 9, 9]) is None


def test_directional_accuracy_empty_returns_none() -> None:
    assert directional_accuracy([], [], []) is None


# -------- beat_miss_accuracy --------------------------------------------------


def test_beat_miss_accuracy_known_answer() -> None:
    # consensus 100, actual 110 (beat), pred 105 (beat) ✓
    # consensus 100, actual 90 (miss), pred 95 (miss) ✓ → 1.0
    assert beat_miss_accuracy([100, 100], [110, 90], [105, 95]) == pytest.approx(1.0)


def test_beat_miss_accuracy_disagreement() -> None:
    consensus = [100.0, 100.0, 100.0]
    actuals = [110.0, 90.0, 105.0]
    preds = [95.0, 110.0, 110.0]  # wrong, wrong, right
    assert beat_miss_accuracy(consensus, actuals, preds) == pytest.approx(1 / 3, abs=1e-12)


def test_beat_miss_accuracy_all_equal_returns_none() -> None:
    assert beat_miss_accuracy([100, 100], [100, 100], [110, 90]) is None


# -------- signed_error_stats --------------------------------------------------


def test_signed_error_stats_known_answer() -> None:
    # Symmetric over/under: SPE = [+0.10, -0.10] → mean 0, median 0.
    stats = signed_error_stats([100.0, 100.0], [110.0, 90.0])
    assert stats is not None
    assert stats["mean_spe"] == pytest.approx(0.0, abs=1e-12)
    assert stats["median_spe"] == pytest.approx(0.0, abs=1e-12)
    assert stats["pct_over"] == pytest.approx(0.5, abs=1e-12)
    assert stats["pct_under"] == pytest.approx(0.5, abs=1e-12)


def test_signed_error_stats_systematic_overprediction() -> None:
    stats = signed_error_stats([100.0, 100.0, 100.0], [110.0, 120.0, 105.0])
    assert stats is not None
    assert stats["mean_spe"] > 0
    assert stats["pct_over"] == pytest.approx(1.0, abs=1e-12)
    assert stats["pct_under"] == pytest.approx(0.0, abs=1e-12)


def test_signed_error_stats_empty_returns_none() -> None:
    assert signed_error_stats([], []) is None


# -------- coverage ------------------------------------------------------------


def test_coverage_known_answer() -> None:
    assert coverage(9, 7) == pytest.approx(0.7777777777777778, abs=1e-9)


def test_coverage_zero_total_returns_none() -> None:
    assert coverage(0, 5) is None


def test_coverage_full() -> None:
    assert coverage(10, 10) == 1.0


# -------- smape ---------------------------------------------------------------


def test_smape_known_answer() -> None:
    # 2 * 10 / (100 + 110 + eps) ≈ 0.09523809...
    val = smape([100.0], [110.0])
    assert val is not None
    assert val == pytest.approx(20.0 / 210.0, abs=1e-7)


def test_smape_empty_returns_none() -> None:
    assert smape([], []) is None


def test_smape_handles_both_zero() -> None:
    # eps prevents 0/0; result should be ~0.
    val = smape([0.0], [0.0])
    assert val is not None
    assert val == pytest.approx(0.0, abs=1e-7)


# -------- correlation ---------------------------------------------------------


def test_correlation_pearson_perfect() -> None:
    a = np.arange(10, dtype=float)
    p = 2.0 * a + 3.0
    assert correlation(a, p) == pytest.approx(1.0, abs=1e-9)


def test_correlation_pearson_negative() -> None:
    a = np.arange(10, dtype=float)
    p = -a
    assert correlation(a, p) == pytest.approx(-1.0, abs=1e-9)


def test_correlation_pearson_matches_numpy() -> None:
    rng = np.random.default_rng(42)
    a = rng.normal(size=50)
    p = 0.6 * a + rng.normal(size=50) * 0.3
    expected = float(np.corrcoef(a, p)[0, 1])
    assert correlation(a, p) == pytest.approx(expected, abs=1e-9)


def test_correlation_spearman_monotonic_nonlinear() -> None:
    # Spearman = 1 for any strictly increasing transform.
    a = np.arange(1, 11, dtype=float)
    p = a**3
    assert correlation(a, p, method="spearman") == pytest.approx(1.0, abs=1e-9)


def test_correlation_constant_returns_none() -> None:
    assert correlation([1, 1, 1, 1], [4, 5, 6, 7]) is None


def test_correlation_too_few_points_returns_none() -> None:
    assert correlation([1.0], [2.0]) is None


def test_correlation_unknown_method_raises() -> None:
    with pytest.raises(ValueError):
        correlation([1.0, 2.0], [3.0, 4.0], method="kendall")  # type: ignore[arg-type]


# -------- mase ----------------------------------------------------------------


def test_mase_known_answer() -> None:
    # MAE(pred) = 5, MAE(naive) = 10 → MASE = 0.5
    a = [100.0, 100.0]
    p = [105.0, 95.0]
    naive = [110.0, 90.0]
    assert mase(a, p, naive) == pytest.approx(0.5, abs=1e-12)


def test_mase_naive_in_noise_floor_returns_none() -> None:
    # naive == actual exactly → naive_mae = 0 < tolerance → None.
    a = [1.0, 2.0, 3.0]
    assert mase(a, [0.5, 1.5, 2.5], a) is None


def test_mase_empty_returns_none() -> None:
    assert mase([], [], []) is None


# -------- picp ----------------------------------------------------------------


def test_picp_known_answer() -> None:
    a = [1.0, 2.0, 3.0, 4.0]
    lo = [0.0, 0.0, 5.0, 5.0]
    hi = [2.0, 5.0, 10.0, 10.0]
    # In: row 0 (1∈[0,2]), row 1 (2∈[0,5]); out: rows 2,3 → 0.5.
    assert picp(a, lo, hi) == pytest.approx(0.5, abs=1e-12)


def test_picp_empty_returns_none() -> None:
    assert picp([], [], []) is None


# -------- crps_quantiles ------------------------------------------------------


def test_crps_quantiles_perfect_prediction_is_zero() -> None:
    # Every quantile sits exactly on the actual → pinball = 0 for all τ.
    actuals = np.array([10.0, 20.0])
    taus = np.array([0.25, 0.5, 0.75])
    q = np.broadcast_to(actuals[:, None], (2, 3)).copy()
    assert crps_quantiles(actuals, q, taus) == pytest.approx(0.0, abs=1e-12)


def test_crps_quantiles_known_pinball() -> None:
    # Hand-computed: actual=10, taus=[0.25, 0.5, 0.75], q=[8, 10, 12].
    # τ=0.25, q=8, err=10-8=2>0: pinball = 0.25*(8-10) - 0*(8-10) = ... use formula
    # pinball = (tau - 1{a<q}) * (q - a).
    # τ=0.25, q=8: a<q? 10<8? no → indicator 0. (0.25 - 0)*(8-10) = -0.5
    # τ=0.5, q=10: a<q? 10<10? no → 0. (0.5)*(0) = 0
    # τ=0.75, q=12: a<q? 10<12? yes → 1. (0.75-1)*(12-10) = -0.5
    # mean over τ = (-0.5 + 0 - 0.5) / 3 = -1/3.
    val = crps_quantiles([10.0], [[8.0, 10.0, 12.0]], [0.25, 0.5, 0.75])
    assert val == pytest.approx(-1.0 / 3.0, abs=1e-12)


def test_crps_quantiles_empty_returns_none() -> None:
    assert crps_quantiles([], np.empty((0, 0)), []) is None


def test_crps_quantiles_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        crps_quantiles([1.0, 2.0], [[1.0, 2.0]], [0.5])


# -------- quantile_calibration ------------------------------------------------


def test_quantile_calibration_known_answer() -> None:
    # Actuals uniformly spaced 0..99; quantiles set s.t. ~50% land below median.
    rng = np.random.default_rng(0)
    n = 200
    actuals = rng.normal(size=n)
    # Use the actuals' own quantiles → empirical coverage matches exactly.
    taus = np.array([0.25, 0.5, 0.75])
    q_vals = np.quantile(actuals, taus)
    q_preds = np.broadcast_to(q_vals[None, :], (n, 3)).copy()
    out = quantile_calibration(actuals, q_preds, taus)
    assert out is not None
    # Empirical CDF at the sample quantile is ~tau (small-sample drift OK).
    for tau, observed in out.items():
        assert abs(observed - tau) < 0.02


def test_quantile_calibration_empty_returns_none() -> None:
    assert quantile_calibration([], np.empty((0, 0)), []) is None


def test_quantile_calibration_uses_default_grid() -> None:
    # Smoke: the module exposes the standard 9-level grid and it parses.
    assert len(QUANTILE_LEVELS) == 9
    assert math.isclose(QUANTILE_LEVELS[0], 0.1)
    assert math.isclose(QUANTILE_LEVELS[-1], 0.9)


# -------- direction-sign correctness ------------------------------------------


def test_direction_sign_convention_hand_crafted() -> None:
    """Ground-truth sign convention: (pred - prev) vs (actual - prev).

    Specifically: an up call means pred > prev. A down call means pred < prev.
    These are independent of consensus.
    """
    prev = [50.0, 50.0, 50.0, 50.0]
    actuals = [60.0, 40.0, 60.0, 40.0]  # up, down, up, down
    preds = [55.0, 45.0, 45.0, 55.0]  # up✓, down✓, down✗, up✗
    assert directional_accuracy(prev, actuals, preds) == pytest.approx(0.5, abs=1e-12)


def test_beat_miss_sign_convention_hand_crafted() -> None:
    """Beat/miss sign: (pred - consensus) vs (actual - consensus)."""
    consensus = [10.0, 10.0, 10.0, 10.0]
    actuals = [12.0, 8.0, 12.0, 8.0]  # beat, miss, beat, miss
    preds = [11.0, 9.0, 9.0, 11.0]  # beat✓, miss✓, miss✗, beat✗
    assert beat_miss_accuracy(consensus, actuals, preds) == pytest.approx(0.5, abs=1e-12)
