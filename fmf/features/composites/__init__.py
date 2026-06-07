"""S18 earnings-quality composite features.

Five experimental composites shipped behind a single experimental flag
via the EARNINGS_QUALITY_CLUSTER tuple. The win-gate reproducer in
scripts/run_cluster_win_gate.py runs the backtester with feature_ids =
BASELINE + EARNINGS_QUALITY_CLUSTER versus BASELINE only and records
the delta to docs/specs/alternative_models.md.
"""

from __future__ import annotations

EARNINGS_QUALITY_CLUSTER: tuple[str, ...] = (
    "piotroski_f_score",
    "ccc_days",
    "dechow_accruals",
    "beneish_m_score",
    "mohanram_g_score",
)

__all__ = ["EARNINGS_QUALITY_CLUSTER"]
