"""S22 headline reproducer: 9-ticker fixture, 8 scored, 2020-2023, real TiRex.

Runs the backtester end-to-end on the committed fixture with the live HuggingFace
TiRex backend (NX-AI/TiRex via tirex-ts). Prints the aggregate scoreboard and the
by-horizon-bucket slice; writes both to reports/headline_scoreboard.json.

Requires the `tirex` optional extra: `uv sync --extra tirex`.

Apple Silicon caveat: LightGBM and PyTorch both link libomp.dylib; running both
in the same process can deadlock at OpenMP barriers. The script forces single-
thread mode via OMP_NUM_THREADS=1 and torch.set_num_threads(1) before importing
the orchestrator. See fmf.equity.forecasting.models.tirex_model.TirexHuggingFaceBackend
docstring for the longer note.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from fmf.equity.forecasting.evaluation._backtester_config import BacktesterConfig  # noqa: E402
from fmf.equity.forecasting.evaluation.backtester import (  # noqa: E402
    ExpandingWindowBacktester,
    scoreboard_from_result,
)
from fmf.equity.forecasting.models.tirex_model import TirexHuggingFaceBackend  # noqa: E402
from tests.equity.forecasting.evaluation._fixture_helpers import (  # noqa: E402
    all_fixture_security_ids,
    fixture_conn,
)

REPO_ROOT = Path(__file__).parent.parent
OUTPUT = REPO_ROOT / "reports" / "headline_scoreboard.json"

FEATURE_IDS = ("revenue_ttm", "gross_margin", "net_margin", "return_on_equity")


def _commit_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> None:
    cfg = BacktesterConfig(
        metric="eps_diluted",
        start_year=2020,
        end_year=2023,
        grid_strategy="filing_dates",
        feature_ids=FEATURE_IDS,
        min_train_samples=10,
        meta_min_train=8,
    )
    backend = TirexHuggingFaceBackend(device="cpu")
    conn = fixture_conn()
    try:
        ids = all_fixture_security_ids(conn)
        t0 = time.time()
        bt = ExpandingWindowBacktester(conn, cfg, tirex_backend=backend)
        result = bt.run(ids)
        runtime = time.time() - t0
    finally:
        conn.close()

    df = result.to_frame()
    by_ticker = {
        sym: {
            "n_scored": int(len(sub)),
            "tirex_finite": int(sub["tirex_pred"].notna().sum()),
        }
        for sym, sub in df.groupby("symbol")
    }
    h = df["horizon_days"].to_numpy()
    horizon_stats = {
        "n": int(len(h)),
        "min": int(h.min()),
        "median": int(np.median(h)),
        "p75": int(np.percentile(h, 75)),
        "p99": int(np.percentile(h, 99)),
        "max": int(h.max()),
    }

    agg = scoreboard_from_result(result, by_horizon_bucket=False)
    sliced = scoreboard_from_result(result, by_horizon_bucket=True)

    payload = {
        "commit_sha": _commit_sha(),
        "runtime_seconds": round(runtime, 1),
        "config": {
            "metric": cfg.metric,
            "start_year": cfg.start_year,
            "end_year": cfg.end_year,
            "grid_strategy": cfg.grid_strategy,
            "feature_ids": list(cfg.feature_ids),
            "min_train_samples": cfg.min_train_samples,
            "meta_min_train": cfg.meta_min_train,
        },
        "scored_tickers": list(by_ticker.keys()),
        "unresolved_target_count": result.unresolved_target_count,
        "meta_learned_from_fold": result.meta_learned_from_fold,
        "scored_rows": len(result.rows),
        "by_ticker": by_ticker,
        "horizon_days": horizon_stats,
        "scoreboard_aggregate": agg.round(4).to_dict(),
        "scoreboard_by_bucket": {
            f"{idx[0]}__{idx[1]}": row.round(4).to_dict() for idx, row in sliced.iterrows()
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(f"wrote {OUTPUT} ({payload['scored_rows']} rows, runtime {runtime:.1f}s)")


if __name__ == "__main__":
    main()
