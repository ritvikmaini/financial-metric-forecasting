"""ingest_edgar.py — CLI for the EDGAR ingest pipeline.

Usage:
    uv run python -m scripts.ingest_edgar \\
        --base-url https://data.sec.gov \\
        --ticker-file path/to/tickers.txt \\
        --out path/to/out.duckdb \\
        [--max-rps 10] [--since 2010-01-01]

The ticker file is one CSV row per ticker: "AAPL,0000320193".
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import uuid
from pathlib import Path

import duckdb

from fmf.data.connectors import bulk_load, get_connection
from fmf.data.edgar._http import EdgarClient
from fmf.data.edgar.companyfacts import load_facts
from fmf.data.edgar.normalize import normalize_to_tables
from fmf.data.edgar.submissions import list_filings

log = logging.getLogger(__name__)


SCHEMA_PATH = Path(__file__).parent.parent / "fmf" / "data" / "schema.sql"


def _ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))


def _ensure_security(conn: duckdb.DuckDBPyConnection, ticker: str, cik: str) -> uuid.UUID:
    """Return existing or newly-created security_id for ticker."""
    row = conn.execute('SELECT security_id FROM "securities" WHERE cik = ?', [cik]).fetchone()
    if row is not None:
        return uuid.UUID(str(row[0]))
    new_id = uuid.uuid4()
    conn.execute(
        'INSERT INTO "securities" (security_id, symbol, cik) VALUES (?, ?, ?)',
        [str(new_id), ticker, cik],
    )
    return new_id


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
    parser = argparse.ArgumentParser(description="EDGAR ingest CLI")
    parser.add_argument("--base-url", default="https://data.sec.gov")
    parser.add_argument("--ticker-file", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-rps", type=float, default=10.0)
    parser.add_argument("--since", type=lambda s: dt.date.fromisoformat(s), default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tickers = _parse_ticker_file(args.ticker_file)
    client = EdgarClient(base_url=args.base_url, max_rps=args.max_rps)

    conn = get_connection(args.out)
    try:
        _ensure_schema(conn)
        for ticker, cik in tickers:
            log.info("ingesting %s (%s)", ticker, cik)
            security_id = _ensure_security(conn, ticker, cik)
            try:
                _ = list_filings(client, cik=cik)
            except Exception as e:
                log.warning("submissions for %s failed: %s; continuing", ticker, e)
            facts = load_facts(client, cik=cik)
            if args.since:
                facts = [f for f in facts if f.filed >= args.since]
            tables = normalize_to_tables(facts=facts, security_id=security_id)
            if not tables.income_statement.empty:
                bulk_load(conn=conn, table="income_statement", df=tables.income_statement)
            if not tables.balance_sheet.empty:
                bulk_load(conn=conn, table="balance_sheet", df=tables.balance_sheet)
            if not tables.cashflow.empty:
                bulk_load(conn=conn, table="cashflow", df=tables.cashflow)
            log.info(
                "  inserted income=%d balance=%d cashflow=%d",
                len(tables.income_statement),
                len(tables.balance_sheet),
                len(tables.cashflow),
            )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
