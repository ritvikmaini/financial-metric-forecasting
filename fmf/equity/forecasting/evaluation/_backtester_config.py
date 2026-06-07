"""BacktesterConfig immutable dataclass."""

from __future__ import annotations

import dataclasses
from typing import Literal

METRICS: tuple[str, ...] = ("eps_diluted", "ebitda", "ebit")
GRID_STRATEGIES: tuple[str, ...] = ("filing_dates", "fiscal_year_end", "quarterly")


@dataclasses.dataclass(frozen=True, slots=True)
class BacktesterConfig:
    """S10 backtester configuration.

    No embargo field by design (Decision 6 of the S10 design):
    expanding-window with the f' < T_k purge has an empty symmetric channel,
    so a leakage embargo would be inert.
    """

    metric: Literal["eps_diluted", "ebitda", "ebit"]
    start_year: int
    end_year: int
    grid_strategy: Literal["filing_dates", "fiscal_year_end", "quarterly"] = "filing_dates"
    feature_ids: tuple[str, ...] = ()
    seed: int = 42
    min_train_samples: int = 30
    feature_cap_top_k: int = 30
    meta_min_train: int = 12
    log_unresolved_targets: bool = True

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}; got {self.metric!r}")
        if self.start_year >= self.end_year:
            raise ValueError(
                f"start_year < end_year required; got {self.start_year}..{self.end_year}"
            )
        if self.grid_strategy not in GRID_STRATEGIES:
            raise ValueError(
                f"grid_strategy must be one of {GRID_STRATEGIES}; got {self.grid_strategy!r}"
            )
        if not self.feature_ids:
            raise ValueError("feature_ids must be non-empty")
        if self.min_train_samples < 3:
            raise ValueError(f"min_train_samples must be >= 3; got {self.min_train_samples}")
        if self.feature_cap_top_k < 0:
            raise ValueError(
                f"feature_cap_top_k must be >= 0 (0 disables); got {self.feature_cap_top_k}"
            )
        if self.meta_min_train < 3:
            raise ValueError(
                f"meta_min_train must be >= 3 (simplex GD floor); got {self.meta_min_train}"
            )
