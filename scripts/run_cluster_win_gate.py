"""S18 win-gate reproducer: DeltaMedAPE_Ensemble between (cluster ON, cluster OFF).

Runs the backtester twice on AAPL+MSFT 2020-2022 with feature_ids = baseline +
cluster and baseline only. Writes the delta to docs/specs/alternative_models.md.

The verdict is provisional pending S15 noise-floor measurement.
"""

from __future__ import annotations

import datetime as dt
import logging
import subprocess
from pathlib import Path

from fmf.equity.forecasting.evaluation._backtester_config import BacktesterConfig
from fmf.equity.forecasting.evaluation.backtester import (
    ExpandingWindowBacktester,
    scoreboard_from_result,
)
from fmf.features.composites import EARNINGS_QUALITY_CLUSTER

REPO_ROOT = Path(__file__).parent.parent
VERDICT_PATH = REPO_ROOT / "docs" / "specs" / "alternative_models.md"

BASELINE_FEATURES = ("revenue_ttm", "gross_margin", "net_margin", "return_on_equity")

log = logging.getLogger("cluster-win-gate")


def _commit_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _run(cfg: BacktesterConfig) -> float | None:
    from tests.equity.forecasting.evaluation._fixture_helpers import (
        fixture_conn,
        two_anchor_ids,
    )
    from tests.equity.forecasting.evaluation.test_backtester_invariants import (
        StubTirexBackend,
    )

    conn = fixture_conn()
    try:
        ids = two_anchor_ids(conn)
        result = ExpandingWindowBacktester(conn, cfg, tirex_backend=StubTirexBackend()).run(ids)
    finally:
        conn.close()
    board = scoreboard_from_result(result, by_horizon_bucket=False)
    val = board.loc["Ensemble"].get("median_ape")
    if val is None:
        return None
    return float(val)


def main() -> None:
    cfg_off = BacktesterConfig(
        metric="eps_diluted",
        start_year=2020,
        end_year=2022,
        grid_strategy="filing_dates",
        feature_ids=BASELINE_FEATURES,
        min_train_samples=10,
        meta_min_train=8,
    )
    cfg_on = BacktesterConfig(
        metric="eps_diluted",
        start_year=2020,
        end_year=2022,
        grid_strategy="filing_dates",
        feature_ids=BASELINE_FEATURES + EARNINGS_QUALITY_CLUSTER,
        min_train_samples=10,
        meta_min_train=8,
    )
    log.info("running cluster OFF")
    medape_off = _run(cfg_off)
    log.info("running cluster ON")
    medape_on = _run(cfg_on)
    if medape_off is None or medape_on is None:
        delta: float | None = None
    else:
        delta = medape_on - medape_off
    sha = _commit_sha()
    today = dt.date.today().isoformat()

    def fmt(v: float | None) -> str:
        return "N/A" if v is None else f"{v:.6f}"

    row = (
        f"| earnings_quality_cluster | DEFERRED-pending-S15 | "
        f"{fmt(delta)} | {fmt(medape_off)} | {fmt(medape_on)} | "
        f"AAPL,MSFT 2020-2022 | {sha} | {today} |"
    )
    VERDICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not VERDICT_PATH.exists() or VERDICT_PATH.stat().st_size == 0:
        VERDICT_PATH.write_text(
            "# Alternative Models -- Verdicts\n\n"
            "Each row is one cluster experiment. Status is DEFERRED-pending-S15 "
            "until the noise-floor program ships sigma; the v1.0 verdicts are raw "
            "deltas to be classified PASS / FAIL once sigma is known.\n\n"
            "| Cluster | Status | DeltaMedAPE_Ensemble | MedAPE_off | MedAPE_on "
            "| Universe | Commit | Date |\n"
            "|---|---|---|---|---|---|---|---|\n"
        )
    with VERDICT_PATH.open("a") as f:
        f.write(row + "\n")
    log.info("verdict written: %s", row)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
