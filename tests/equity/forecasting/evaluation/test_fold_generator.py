"""F1 four-cutoff fold generator tests."""

from __future__ import annotations

import datetime as dt

from fmf.equity.forecasting.evaluation._fold_generator import (
    ANNUAL_CUTOFF,
    QUARTERLY_CUTOFFS,
    generate_folds,
)


def test_generates_four_cutoffs_per_year() -> None:
    folds = generate_folds(start_year=2020, end_year=2022)
    assert len(folds) == 12


def test_cutoffs_are_calendar_ordered_strictly() -> None:
    folds = generate_folds(start_year=2020, end_year=2022)
    cutoffs = [f.cutoff for f in folds]
    assert cutoffs == sorted(cutoffs)
    assert len(set(cutoffs)) == len(cutoffs)


def test_q4_cutoff_is_next_calendar_year() -> None:
    folds = generate_folds(start_year=2020, end_year=2020)
    q4 = [f for f in folds if f.cutoff_label.endswith("Q4-post-10K")]
    assert len(q4) == 1
    assert q4[0].cutoff == dt.date(2021, 3, 1)


def test_first_fold_is_seed_train_only() -> None:
    folds = generate_folds(start_year=2020, end_year=2022)
    assert folds[0].is_seed
    assert all(not f.is_seed for f in folds[1:])


def test_window_end_equals_next_cutoff() -> None:
    folds = generate_folds(start_year=2020, end_year=2022)
    for k in range(len(folds) - 1):
        assert folds[k].window_end == folds[k + 1].cutoff


def test_last_fold_window_end_is_after_end_year() -> None:
    folds = generate_folds(start_year=2020, end_year=2022)
    assert folds[-1].window_end > folds[-1].cutoff
    assert folds[-1].window_end.year >= 2023


def test_cutoff_labels_distinguish_q4() -> None:
    folds = generate_folds(start_year=2020, end_year=2020)
    labels = [f.cutoff_label for f in folds]
    assert any("Q4-post-10K" in label for label in labels)
    assert any("Q1" in label for label in labels)


def test_constants_match_sketch() -> None:
    assert QUARTERLY_CUTOFFS == ((5, 15), (8, 14), (11, 15))
    assert ANNUAL_CUTOFF == (3, 1)
