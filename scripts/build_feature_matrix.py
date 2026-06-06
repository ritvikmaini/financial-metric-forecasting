"""scripts.build_feature_matrix — CLI for materializing the matrix.

Usage:
    uv run python -m scripts.build_feature_matrix \\
        [--db tests/fixtures/mini.duckdb] \\
        [--grid filing_dates|fiscal_year_end|quarterly] \\
        [--out reports/feature_matrix.parquet]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb

from fmf.features.as_of_grid import (
    filing_dates_grid,
    fiscal_year_end_grid,
    quarterly_grid,
)
from fmf.features.builtin_features import BUILTIN_REGISTRY
from fmf.features.matrix_builder import build_feature_matrix

log = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).parent.parent
DEFAULT_DB = REPO_ROOT / "tests" / "fixtures" / "mini.duckdb"
DEFAULT_OUT = REPO_ROOT / "reports" / "feature_matrix.parquet"


GRID_STRATEGIES = {
    "filing_dates": filing_dates_grid,
    "fiscal_year_end": fiscal_year_end_grid,
    "quarterly": quarterly_grid,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--grid", choices=sorted(GRID_STRATEGIES), default="filing_dates")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    conn = duckdb.connect(str(args.db), read_only=True)
    try:
        df = build_feature_matrix(
            conn=conn,
            registry=BUILTIN_REGISTRY,
            grid_strategy=GRID_STRATEGIES[args.grid],
        )
    finally:
        conn.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".parquet":
        df.to_parquet(args.out, index=False)
    elif args.out.suffix == ".csv":
        df.to_csv(args.out, index=False)
    else:
        raise ValueError(f"unknown output extension: {args.out.suffix}")

    # Best-effort relative path for readability; fall back to absolute
    # when the output lives outside the repo (e.g., pytest tmp_path).
    try:
        display_path: Path = args.out.relative_to(REPO_ROOT)
    except ValueError:
        display_path = args.out
    log.info(
        "wrote feature matrix: %s rows=%d cols=%d",
        display_path,
        len(df),
        len(df.columns),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
