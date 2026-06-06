"""YFinanceClient seam tests.

The client wraps yfinance with a configurable base: live (None) or a
local fixture directory. Tests use the fixture directory so CI never
hits live yfinance.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from fmf.data.yfinance._client import YFinanceClient

REPO_ROOT = Path(__file__).parent.parent.parent.parent
SAMPLES = REPO_ROOT / "tests" / "fixtures" / "sample_yfinance"


def test_client_fetch_prices_from_fixture() -> None:
    c = YFinanceClient(base=SAMPLES)
    df = c.fetch_prices(
        "AAPL",
        start=dt.date(2019, 6, 1),
        end=dt.date(2019, 6, 10),
    )
    assert not df.empty
    assert {"Open", "High", "Low", "Close", "Adj Close", "Volume"} <= set(df.columns)


def test_client_fetch_info_from_fixture() -> None:
    c = YFinanceClient(base=SAMPLES)
    info = c.fetch_info("AAPL")
    assert isinstance(info, dict)
    assert "symbol" in info or "shortName" in info


def test_client_fetch_earnings_estimate_from_fixture() -> None:
    c = YFinanceClient(base=SAMPLES)
    df = c.fetch_earnings_estimate("AAPL")
    assert isinstance(df, pd.DataFrame)


def test_client_fetch_missing_ticker_raises() -> None:
    c = YFinanceClient(base=SAMPLES)
    with pytest.raises(FileNotFoundError):
        c.fetch_prices("ZZZZ", start=dt.date(2023, 1, 1), end=dt.date(2023, 6, 30))


def test_client_fetch_prices_returns_flat_columns_not_multiindex() -> None:
    """The committed CSV fixtures were written from a flattened DataFrame.
    Reading them back must yield a flat column shape; if columns come
    out as MultiIndex, downstream prices.py's `raw["Open"].to_numpy()`
    silently returns a 2D array and the schema insert fails or mis-aligns.
    """
    c = YFinanceClient(base=SAMPLES)
    df = c.fetch_prices(
        "AAPL",
        start=dt.date(2019, 6, 1),
        end=dt.date(2019, 6, 10),
    )
    assert not isinstance(df.columns, pd.MultiIndex), (
        f"fixture CSV produced MultiIndex columns: {df.columns}. "
        "Re-run the T1 bootstrap with the flatten step."
    )
    assert df["Open"].ndim == 1
