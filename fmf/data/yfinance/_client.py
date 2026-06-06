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
        """Return OHLCV DataFrame with truly raw `Close` (un-split).

        Columns are always flat: Open, High, Low, Close, Adj Close, Volume.
        Index: DatetimeIndex of trading days.

        Yahoo Finance's API always returns split-adjusted OHLC (Close on
        2019-06-03 reads as ~$43, not the historically observed ~$173).
        Even `auto_adjust=False` only toggles the dividend layer — split
        adjustment is baked into the wire format. To recover truly raw
        Close we multiply OHLC by the cumulative product of all stock
        splits that occurred STRICTLY AFTER each row's calendar date,
        and divide Volume by the same factor. `Adj Close` is left as
        Yahoo returns it (back-adjusted for splits AND dividends), so
        the pair (close, adj_close) preserves the split signal that the
        regression test asserts (close ≈ split_ratio × adj_close).

        We use `Ticker.history(auto_adjust=False)` rather than
        `yf.download(...)` because the `download` path produces a
        single-row-per-day frame that is harder to align with the
        `splits` series (and historically has had silent kwarg drops
        between yfinance versions).
        """
        if self.base is not None:
            path = self.base / f"{ticker}_prices.csv"
            if not path.exists():
                raise FileNotFoundError(f"no fixture for {ticker} prices at {path}")
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            mask = (df.index.date >= start) & (df.index.date <= end)
            return df.loc[mask]
        import yfinance as yf  # noqa: PLC0415

        t = yf.Ticker(ticker)
        df = t.history(
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=False,
            back_adjust=False,
        ).copy()
        if df.empty:
            return df
        splits = t.splits
        if splits is not None and not splits.empty:
            row_dates = pd.Series(df.index.date)
            split_dates = pd.Series(splits.index.date)
            split_values = splits.to_numpy()
            factors = []
            for d in row_dates:
                mask_arr = (split_dates > d).to_numpy()
                f = float(split_values[mask_arr].prod()) if mask_arr.any() else 1.0
                factors.append(f)
            factor_series = pd.Series(factors, index=df.index)
            for col in ("Open", "High", "Low", "Close"):
                df[col] = df[col] * factor_series
            df["Volume"] = (df["Volume"] / factor_series).round().astype("int64")
        # Drop tz info on index so downstream code (and CSV round-trip)
        # sees naive dates matching the fixture-read path.
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
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
