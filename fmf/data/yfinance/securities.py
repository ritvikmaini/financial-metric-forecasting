"""Securities metadata UPDATE.

S2 inserted securities rows by (security_id, symbol, cik). S3 fills
sector / industry / country / exchange from yf.Ticker.info via UPDATE.

yf.Ticker.info is fragile: fields can be missing or null. We tolerate
absent fields and never fail the whole ingest on one bad lookup —
this is the "one bad ticker doesn't sink the run" guarantee.
"""

from __future__ import annotations

import logging

import duckdb

from fmf.data.yfinance._client import YFinanceClient

log = logging.getLogger(__name__)


_INFO_FIELDS: dict[str, str] = {
    # info_key -> securities column
    "sector": "sector",
    "industry": "industry",
    "country": "country",
    "exchange": "exchange",
}


def update_securities_metadata(
    *,
    conn: duckdb.DuckDBPyConnection,
    client: YFinanceClient,
    ticker: str,
    cik: str,
) -> None:
    """UPDATE securities WHERE cik = ? with sector/industry/country/exchange
    from yfinance. Tolerates missing fields and lookup errors (logs a
    warning instead of raising).
    """
    try:
        info = client.fetch_info(ticker)
    except (FileNotFoundError, KeyError, ValueError) as e:
        log.warning("yfinance info fetch for %s failed: %s; skipping", ticker, e)
        return

    set_parts: list[str] = []
    values: list[object] = []
    for info_key, column in _INFO_FIELDS.items():
        v = info.get(info_key)
        if v is not None:
            set_parts.append(f'"{column}" = ?')
            values.append(v)
    if not set_parts:
        return

    values.append(cik)
    query = f'UPDATE "securities" SET {", ".join(set_parts)} WHERE cik = ?'
    conn.execute(query, values)
