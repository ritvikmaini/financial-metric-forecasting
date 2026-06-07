"""F1 four-cutoff fold generator. See plans/2026-06-07-s10-backtester.md."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

QUARTERLY_CUTOFFS: tuple[tuple[int, int], ...] = ((5, 15), (8, 14), (11, 15))
ANNUAL_CUTOFF: tuple[int, int] = (3, 1)


@dataclass(frozen=True, slots=True)
class FoldSpec:
    fold_idx: int
    cutoff: dt.date
    window_end: dt.date
    cutoff_label: str
    is_seed: bool


def generate_folds(start_year: int, end_year: int) -> list[FoldSpec]:
    """Four-cutoff F1 schedule. Q1/Q2/Q3 anchor on the target year's post-10-Q
    dates; Q4 anchors on the NEXT calendar year's post-10-K date (longest-
    horizon decision point, kept per Decision 5 in the plan).
    """
    cutoffs: list[tuple[dt.date, str]] = []
    for year in range(start_year, end_year + 1):
        for q_idx, (month, day) in enumerate(QUARTERLY_CUTOFFS, start=1):
            cutoffs.append((dt.date(year, month, day), f"{year}-Q{q_idx}"))
        cutoffs.append(
            (dt.date(year + 1, ANNUAL_CUTOFF[0], ANNUAL_CUTOFF[1]), f"{year}-Q4-post-10K")
        )
    cutoffs.sort(key=lambda x: x[0])
    folds: list[FoldSpec] = []
    for idx, (cutoff, label) in enumerate(cutoffs):
        window_end = cutoffs[idx + 1][0] if idx + 1 < len(cutoffs) else dt.date(end_year + 2, 1, 1)
        folds.append(
            FoldSpec(
                fold_idx=idx,
                cutoff=cutoff,
                window_end=window_end,
                cutoff_label=label,
                is_seed=idx == 0,
            )
        )
    return folds
