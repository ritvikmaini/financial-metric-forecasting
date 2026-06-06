"""ingest_yfinance.py — CLI for yfinance ingest.

Usage:
    uv run python -m scripts.ingest_yfinance \\
        [--base-dir tests/fixtures/sample_yfinance]  # for hermetic runs
        --ticker-file path/to/tickers.txt \\
        --db path/to/db.duckdb \\
        --start 2010-01-01 \\
        --end 2024-12-31

Each ticker step (prices, consensus, securities) is wrapped in a try/except
so a single failure does not abort the run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import uuid
from pathlib import Path

from fmf.data.connectors import get_connection
from fmf.data.yfinance._client import YFinanceClient
from fmf.data.yfinance.consensus import ingest_consensus_snapshot
from fmf.data.yfinance.prices import ingest_prices
from fmf.data.yfinance.securities import update_securities_metadata

log = logging.getLogger(__name__)


def _parse_ticker_file(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        ticker, cik = line.split(",", 1)
        out.append((ticker.strip(), cik.strip()))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="yfinance ingest CLI")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="If set, read fixtures from this dir instead of live yfinance.",
    )
    parser.add_argument("--ticker-file", required=True, type=Path)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument(
        "--start",
        required=True,
        type=lambda s: dt.date.fromisoformat(s),
    )
    parser.add_argument(
        "--end",
        required=True,
        type=lambda s: dt.date.fromisoformat(s),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tickers = _parse_ticker_file(args.ticker_file)
    client = YFinanceClient(base=args.base_dir)
    pulled_at = dt.datetime.now()

    conn = get_connection(args.db)
    try:
        for ticker, cik in tickers:
            log.info("yfinance ingest %s (%s)", ticker, cik)
            row = conn.execute(
                'SELECT security_id FROM "securities" WHERE cik = ?', [cik]
            ).fetchone()
            if row is None:
                log.warning("no securities row for %s; skipping", ticker)
                continue
            security_id = uuid.UUID(str(row[0]))

            n_prices = 0
            try:
                n_prices = ingest_prices(
                    conn=conn,
                    client=client,
                    ticker=ticker,
                    security_id=security_id,
                    start=args.start,
                    end=args.end,
                )
            except Exception as e:
                log.warning("prices for %s failed: %s; continuing", ticker, e)

            n_consensus = 0
            try:
                n_consensus = ingest_consensus_snapshot(
                    conn=conn,
                    client=client,
                    ticker=ticker,
                    security_id=security_id,
                    pulled_at=pulled_at,
                )
            except Exception as e:
                log.warning("consensus for %s failed: %s; continuing", ticker, e)

            try:
                update_securities_metadata(
                    conn=conn,
                    client=client,
                    ticker=ticker,
                    cik=cik,
                )
            except Exception as e:
                log.warning("securities update for %s failed: %s; continuing", ticker, e)

            log.info("  prices=%d consensus=%d", n_prices, n_consensus)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
