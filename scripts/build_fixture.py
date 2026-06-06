"""build_fixture.py — build tests/fixtures/mini.duckdb from live SEC.

Hard-coded anchor set (5 anchors + 3 mid-caps + 1 short-history = 9 tickers).
Runs ingest_edgar, then the anchor-validation gate, then writes the
fixture. Exits non-zero if any anchor fails validation.

Usage:
    uv run python -m scripts.build_fixture \\
        [--base-url https://data.sec.gov] \\
        [--max-rps 8] \\
        [--out tests/fixtures/mini.duckdb] \\
        [--known-financials tests/fixtures/known_financials.json]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import tempfile
from pathlib import Path

import duckdb

from fmf.data.edgar.validation import (
    AnchorValidationError,
    load_known_financials,
    validate_anchors,
)
from scripts.ingest_edgar import main as run_ingest
from scripts.ingest_yfinance import main as run_ingest_yfinance

log = logging.getLogger(__name__)


FIXTURE_TICKERS: list[tuple[str, str]] = [
    # 5 anchors (validated)
    ("AAPL", "0000320193"),
    ("MSFT", "0000789019"),
    ("GOOGL", "0001652044"),
    ("JNJ", "0000200406"),
    ("JPM", "0000019617"),
    # 3 mid-caps for cross-section coverage
    ("ZTS", "0001555280"),
    ("GWW", "0000277135"),
    ("HSY", "0000047111"),
    # short-history ticker (TiRex fallback chain in S7)
    ("SNOW", "0001640147"),
]


REPO_ROOT = Path(__file__).parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build mini.duckdb fixture")
    parser.add_argument("--base-url", default="https://data.sec.gov")
    parser.add_argument("--max-rps", type=float, default=8.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "mini.duckdb",
    )
    parser.add_argument(
        "--known-financials",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "known_financials.json",
    )
    parser.add_argument(
        "--yfinance-start",
        type=lambda s: dt.date.fromisoformat(s),
        default=dt.date(2010, 1, 1),
        help="yfinance price history start date",
    )
    parser.add_argument(
        "--yfinance-end",
        type=lambda s: dt.date.fromisoformat(s),
        default=dt.date.today(),
        help="yfinance price history end date (default: today)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
        for ticker, cik in FIXTURE_TICKERS:
            tf.write(f"{ticker},{cik}\n")
        tickers_path = Path(tf.name)

    if args.out.exists():
        args.out.unlink()

    rc = run_ingest(
        [
            "--base-url",
            args.base_url,
            "--ticker-file",
            str(tickers_path),
            "--out",
            str(args.out),
            "--max-rps",
            str(args.max_rps),
        ]
    )
    if rc != 0:
        log.error("ingest failed with rc=%s", rc)
        return rc

    truth = load_known_financials(args.known_financials)
    conn = duckdb.connect(str(args.out), read_only=True)
    try:
        try:
            validate_anchors(conn, truth)
            log.info("anchor validation PASSED")
        except AnchorValidationError as e:
            log.error("anchor validation FAILED:\n%s", e)
            return 2
    finally:
        conn.close()

    # yfinance augment: prices + consensus + securities metadata.
    log.info("starting yfinance augment...")
    yfinance_rc = run_ingest_yfinance(
        [
            "--ticker-file",
            str(tickers_path),
            "--db",
            str(args.out),
            "--start",
            args.yfinance_start.isoformat(),
            "--end",
            args.yfinance_end.isoformat(),
        ]
    )
    if yfinance_rc != 0:
        log.error("yfinance ingest failed with rc=%s", yfinance_rc)
        return yfinance_rc

    # Re-run anchor validation as a sanity check that fundamentals were untouched.
    conn = duckdb.connect(str(args.out), read_only=True)
    try:
        validate_anchors(conn, truth)
        log.info("anchor validation PASSED after yfinance augment (fundamentals untouched)")
    except AnchorValidationError as e:
        log.error("anchor validation FAILED after yfinance augment:\n%s", e)
        return 3
    finally:
        conn.close()

    log.info("fixture written: %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
