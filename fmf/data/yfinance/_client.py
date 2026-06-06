"""yfinance client seam.

Abstracts yfinance API calls behind a configurable base:
- base=None -> call live yfinance.
- base=<directory> -> read committed sample CSV/JSON files.

Tests construct with base=tests/fixtures/sample_yfinance so CI never
hits live yfinance. build_fixture.py and the live lane construct with
base=None for real ingest.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd


class YFinanceClient:
    """Wrapper around yfinance with fixture-base support."""

    def __init__(self, *, base: Path | None = None) -> None:
        self.base = base

    def fetch_prices(
        self,
        ticker: str,
        *,
        start: dt.date,
        end: dt.date,
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame with auto_adjust=False semantics.

        Columns are always flat: Open, High, Low, Close, Adj Close, Volume.
        Index: DatetimeIndex of trading days.

        Two yfinance version-rot defenses are pinned here:
        - auto_adjust=False: keeps raw `Close` and back-adjusted `Adj Close`
          as separate columns (default flipped between versions).
        - multi_level_index=False where supported, with a manual flatten
          fallback for older yfinance. Recent yfinance returns a
          (Price, Ticker) MultiIndex; without flattening, raw["Open"]
          becomes a one-column DataFrame and .to_numpy() returns 2D.
        """
        if self.base is not None:
            path = self.base / f"{ticker}_prices.csv"
            if not path.exists():
                raise FileNotFoundError(f"no fixture for {ticker} prices at {path}")
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            mask = (df.index.date >= start) & (df.index.date <= end)
            return df.loc[mask]
        import yfinance as yf  # noqa: PLC0415

        try:
            df = yf.download(
                ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                auto_adjust=False,
                progress=False,
                multi_level_index=False,
            )
        except TypeError:
            df = yf.download(
                ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                auto_adjust=False,
                progress=False,
            )
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.droplevel(1)
        return df

    def fetch_earnings_estimate(self, ticker: str) -> pd.DataFrame:
        """Return current snapshot of EPS estimates per period.

        NOTE: snapshot only. yfinance does not expose historical revisions.
        Caller must record pulled_at on each row.
        """
        if self.base is not None:
            path = self.base / f"{ticker}_earnings_estimate.csv"
            if not path.exists():
                raise FileNotFoundError(f"no fixture for {ticker} earnings at {path}")
            return pd.read_csv(path, index_col=0)
        import yfinance as yf  # noqa: PLC0415

        return yf.Ticker(ticker).earnings_estimate

    def fetch_revenue_estimate(self, ticker: str) -> pd.DataFrame:
        """Return current snapshot of revenue estimates per period."""
        if self.base is not None:
            path = self.base / f"{ticker}_revenue_estimate.csv"
            if not path.exists():
                raise FileNotFoundError(f"no fixture for {ticker} revenue at {path}")
            return pd.read_csv(path, index_col=0)
        import yfinance as yf  # noqa: PLC0415

        return yf.Ticker(ticker).revenue_estimate

    def fetch_info(self, ticker: str) -> dict[str, Any]:
        """Return Ticker.info dict snapshot. Fields may be missing."""
        if self.base is not None:
            path = self.base / f"{ticker}_info.json"
            if not path.exists():
                raise FileNotFoundError(f"no fixture for {ticker} info at {path}")
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            return data
        import yfinance as yf  # noqa: PLC0415

        info: dict[str, Any] = yf.Ticker(ticker).info
        return info
