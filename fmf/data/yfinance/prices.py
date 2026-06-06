"""Prices ingest.

Reads OHLCV via YFinanceClient with auto_adjust=False (so raw `close` and
back-adjusted `adj_close` both land in the schema). Writes via bulk_load.

Defensive column-shape pin: if the client somehow returns MultiIndex
columns (older yfinance, fixture-shape drift), we flatten by dropping
the ticker level before column lookups. The client already does this,
but defending in depth is cheap.
"""

from __future__ import annotations

import datetime as dt
import uuid

import duckdb
import pandas as pd

from fmf.data.connectors import bulk_load
from fmf.data.yfinance._client import YFinanceClient


def ingest_prices(
    *,
    conn: duckdb.DuckDBPyConnection,
    client: YFinanceClient,
    ticker: str,
    security_id: uuid.UUID,
    start: dt.date,
    end: dt.date,
) -> int:
    """Fetch prices for ticker and insert into the prices table. Returns
    row count inserted.
    """
    raw = client.fetch_prices(ticker, start=start, end=end)
    if raw.empty:
        return 0
    # Defensive flatten: in case the client didn't catch a MultiIndex
    # (older yfinance, kwarg ignored, or fixture-shape drift), drop the
    # ticker level so raw["Open"] returns a Series, not a frame.
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.copy()
        raw.columns = raw.columns.droplevel(1)
    # Normalize yfinance columns to schema snake_case.
    df = pd.DataFrame(
        {
            "security_id": str(security_id),
            "date": raw.index.date,
            "open": raw["Open"].to_numpy(),
            "high": raw["High"].to_numpy(),
            "low": raw["Low"].to_numpy(),
            "close": raw["Close"].to_numpy(),
            "adj_close": raw["Adj Close"].to_numpy(),
            "volume": raw["Volume"].astype("int64").to_numpy(),
        }
    )
    return bulk_load(conn=conn, table="prices", df=df)
