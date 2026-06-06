"""Built-in feature registry.

59 features populated as `BUILTIN_REGISTRY`. Each feature wraps a
compute_fn(*, conn, security_id, as_of_date) -> float | None.

Two threshold tiers:
- 34 "final-thresholded" features grounded in measured raw column
  coverage (see reports/raw_column_coverage.txt). Their
  min_coverage_pct values are the user-approved targets.
- 25 "provisional-thresholded" derived features at min_coverage_pct=0.50
  pending the T4 audit's measurement. The audit script writes
  reports/coverage_audit.json with measured values; the user
  collaboratively finalizes these thresholds in a follow-up.

Factory pattern: every "_latest" / "_ttm" / "_yoy" / margin / sector
one-hot / price-derived feature is built by a closure factory so the
phantom-guard (`dropna(subset=[<field>])` before selecting a row) and
the PIT contract (always via fetch_pit_series / fetch_prices_pit) are
preserved uniformly.
"""

from __future__ import annotations

import datetime as dt
import math
import uuid
from collections.abc import Callable

import duckdb
import pandas as pd

from fmf.features.derived import (
    _isna,
    _to_date,
    _ttm_from_series,
    _yoy_growth_from_series,
    compute_gross_margin_latest,
    compute_revenue_ttm,
    compute_revenue_yoy_growth,
)
from fmf.features.feature_registry import Feature, FeatureRegistry
from fmf.features.point_in_time import (
    fetch_pit_series,
    fetch_prices_pit,
)

ComputeFn = Callable[..., "float | None"]

# 11 GICS sectors. Canonical labels match the fixture's
# securities.sector strings (case-insensitive match in compute fn).
_GICS_SECTORS: tuple[tuple[str, str], ...] = (
    ("sector_energy", "Energy"),
    ("sector_materials", "Materials"),
    ("sector_industrials", "Industrials"),
    ("sector_consumer_discretionary", "Consumer Discretionary"),
    ("sector_consumer_defensive", "Consumer Defensive"),
    ("sector_healthcare", "Healthcare"),
    ("sector_financial_services", "Financial Services"),
    ("sector_technology", "Technology"),
    ("sector_communication_services", "Communication Services"),
    ("sector_utilities", "Utilities"),
    ("sector_real_estate", "Real Estate"),
)


def _latest_non_null(series: pd.DataFrame, field: str) -> float | None:
    """Return the latest non-null value of `field` from a PIT series.

    Phantom guard: dropna(subset=[field]) before sorting by end_date so
    a phantom row (later end_date but null field from a partial
    re-disclosure) cannot mask a genuine value.
    """
    if series.empty or field not in series.columns:
        return None
    rows = series.dropna(subset=[field])
    if rows.empty:
        return None
    latest = rows.sort_values("end_date").iloc[-1]
    v = latest[field]
    if v is None or _isna(v):
        return None
    return float(v)


def _make_latest_field_feature(table: str, field: str) -> ComputeFn:
    def compute(
        *,
        conn: duckdb.DuckDBPyConnection,
        security_id: uuid.UUID,
        as_of_date: dt.date,
    ) -> float | None:
        series = fetch_pit_series(
            conn=conn, table=table, security_id=security_id, as_of_date=as_of_date
        )
        return _latest_non_null(series, field)

    return compute


def _make_ttm_feature(table: str, field: str) -> ComputeFn:
    def compute(
        *,
        conn: duckdb.DuckDBPyConnection,
        security_id: uuid.UUID,
        as_of_date: dt.date,
    ) -> float | None:
        series = fetch_pit_series(
            conn=conn, table=table, security_id=security_id, as_of_date=as_of_date
        )
        return _ttm_from_series(series, field, as_of_date)

    return compute


def _make_yoy_feature(table: str, field: str) -> ComputeFn:
    def compute(
        *,
        conn: duckdb.DuckDBPyConnection,
        security_id: uuid.UUID,
        as_of_date: dt.date,
    ) -> float | None:
        series = fetch_pit_series(
            conn=conn, table=table, security_id=security_id, as_of_date=as_of_date
        )
        return _yoy_growth_from_series(series, field)

    return compute


def _make_margin_feature(numerator: str, denominator: str) -> ComputeFn:
    """Per-row latest: pick the latest end_date where BOTH fields are
    non-null (so the ratio is consistent), then divide."""

    def compute(
        *,
        conn: duckdb.DuckDBPyConnection,
        security_id: uuid.UUID,
        as_of_date: dt.date,
    ) -> float | None:
        series = fetch_pit_series(
            conn=conn,
            table="income_statement",
            security_id=security_id,
            as_of_date=as_of_date,
        )
        if series.empty or numerator not in series.columns or denominator not in series.columns:
            return None
        rows = series.dropna(subset=[numerator, denominator])
        if rows.empty:
            return None
        latest = rows.sort_values("end_date").iloc[-1]
        denom = float(latest[denominator])
        if denom == 0:
            return None
        return float(latest[numerator]) / denom

    return compute


def _latest_close(prices: pd.DataFrame) -> float | None:
    if prices.empty or "close" not in prices.columns:
        return None
    rows = prices.dropna(subset=["close"])
    if rows.empty:
        return None
    return float(rows.iloc[-1]["close"])


def _make_close_latest() -> ComputeFn:
    def compute(
        *,
        conn: duckdb.DuckDBPyConnection,
        security_id: uuid.UUID,
        as_of_date: dt.date,
    ) -> float | None:
        prices = fetch_prices_pit(conn=conn, security_id=security_id, as_of_date=as_of_date)
        return _latest_close(prices)

    return compute


def _make_price_return_feature(lookback_days: int) -> ComputeFn:
    """Calendar-day lookback return.

    Picks the latest close as the "now" close and the latest close with
    date <= as_of - lookback_days as the "then" close. Returns None if
    either is missing or the lookback close is zero.
    """

    def compute(
        *,
        conn: duckdb.DuckDBPyConnection,
        security_id: uuid.UUID,
        as_of_date: dt.date,
    ) -> float | None:
        prices = fetch_prices_pit(conn=conn, security_id=security_id, as_of_date=as_of_date)
        if prices.empty or "close" not in prices.columns:
            return None
        rows = prices.dropna(subset=["close"])
        if rows.empty:
            return None
        now_close = float(rows.iloc[-1]["close"])
        cutoff = as_of_date - dt.timedelta(days=lookback_days)
        # Use _to_date for tolerant date comparison.
        prior = rows[rows["date"].apply(lambda d: _to_date(d) <= cutoff)]
        if prior.empty:
            return None
        then_close = float(prior.iloc[-1]["close"])
        if then_close == 0:
            return None
        return (now_close - then_close) / then_close

    return compute


def _make_volatility_feature(trading_days: int) -> ComputeFn:
    """Standard deviation of daily log-style returns over the last N
    trading days. Returns None if fewer than N+1 rows are visible.
    """

    def compute(
        *,
        conn: duckdb.DuckDBPyConnection,
        security_id: uuid.UUID,
        as_of_date: dt.date,
    ) -> float | None:
        prices = fetch_prices_pit(conn=conn, security_id=security_id, as_of_date=as_of_date)
        if prices.empty or "close" not in prices.columns:
            return None
        rows = prices.dropna(subset=["close"]).sort_values("date")
        if len(rows) < trading_days + 1:
            return None
        recent = rows.tail(trading_days + 1)
        closes = recent["close"].astype(float).to_numpy()
        rets = (closes[1:] - closes[:-1]) / closes[:-1]
        # Filter inf from a zero close.
        rets = rets[~pd.isna(rets)]
        if len(rets) < 2:
            return None
        v = float(pd.Series(rets).std(ddof=1))
        if math.isnan(v) or math.isinf(v):
            return None
        return v

    return compute


def _make_max_drawdown_1y() -> ComputeFn:
    """Max drawdown over the trailing 1y (365 calendar days).

    For each trading day d in the window, compute (close_d - rolling_max_to_d)
    / rolling_max_to_d (a non-positive number). Return the minimum.
    """

    def compute(
        *,
        conn: duckdb.DuckDBPyConnection,
        security_id: uuid.UUID,
        as_of_date: dt.date,
    ) -> float | None:
        prices = fetch_prices_pit(conn=conn, security_id=security_id, as_of_date=as_of_date)
        if prices.empty or "close" not in prices.columns:
            return None
        rows = prices.dropna(subset=["close"]).sort_values("date")
        if rows.empty:
            return None
        cutoff = as_of_date - dt.timedelta(days=365)
        in_window = rows[rows["date"].apply(lambda d: _to_date(d) >= cutoff)]
        if in_window.empty:
            return None
        closes = in_window["close"].astype(float)
        running_max = closes.cummax()
        drawdown = (closes - running_max) / running_max
        v = float(drawdown.min())
        if math.isnan(v) or math.isinf(v):
            return None
        return v

    return compute


def _get_sector(conn: duckdb.DuckDBPyConnection, security_id: uuid.UUID) -> str | None:
    """Fetch securities.sector for a security. None if missing/null.

    No PIT filter: sector is a slow-moving classification, not a
    point-in-time fact.
    """
    row = conn.execute(
        'SELECT sector FROM "securities" WHERE security_id = ?',
        [str(security_id)],
    ).fetchone()
    if row is None:
        return None
    v = row[0]
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    return str(v)


def _make_sector_one_hot(canonical_label: str) -> ComputeFn:
    """1.0 if the security's sector matches canonical_label
    (case-insensitive), 0.0 if non-null but different, None if missing."""

    target_lower = canonical_label.lower()

    def compute(
        *,
        conn: duckdb.DuckDBPyConnection,
        security_id: uuid.UUID,
        as_of_date: dt.date,
    ) -> float | None:
        sector = _get_sector(conn, security_id)
        if sector is None:
            return None
        return 1.0 if sector.lower() == target_lower else 0.0

    return compute


# Composite / cross-table features (custom compute functions).


def _compute_total_liabilities_latest(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> float | None:
    """Prefer raw total_liabilities; fallback to total_assets - total_equity
    when raw is null but both fallback fields are non-null on the same row.
    """
    series = fetch_pit_series(
        conn=conn, table="balance_sheet", security_id=security_id, as_of_date=as_of_date
    )
    if series.empty:
        return None
    # Primary path: raw total_liabilities.
    raw_rows = series.dropna(subset=["total_liabilities"])
    if not raw_rows.empty:
        latest = raw_rows.sort_values("end_date").iloc[-1]
        v = latest["total_liabilities"]
        if v is not None and not _isna(v):
            return float(v)
    # Fallback path: same-row Assets - Equity.
    fb_rows = series.dropna(subset=["total_assets", "total_equity"])
    if fb_rows.empty:
        return None
    latest = fb_rows.sort_values("end_date").iloc[-1]
    return float(latest["total_assets"]) - float(latest["total_equity"])


def _compute_equity_to_assets(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> float | None:
    series = fetch_pit_series(
        conn=conn, table="balance_sheet", security_id=security_id, as_of_date=as_of_date
    )
    if series.empty:
        return None
    rows = series.dropna(subset=["total_equity", "total_assets"])
    if rows.empty:
        return None
    latest = rows.sort_values("end_date").iloc[-1]
    denom = float(latest["total_assets"])
    if denom == 0:
        return None
    return float(latest["total_equity"]) / denom


def _compute_liabilities_to_assets(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> float | None:
    """Uses derived total_liabilities (with Assets-Equity fallback)."""
    tl = _compute_total_liabilities_latest(
        conn=conn, security_id=security_id, as_of_date=as_of_date
    )
    if tl is None:
        return None
    ta = _latest_non_null(
        fetch_pit_series(
            conn=conn, table="balance_sheet", security_id=security_id, as_of_date=as_of_date
        ),
        "total_assets",
    )
    if ta is None or ta == 0:
        return None
    return tl / ta


def _compute_current_ratio(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> float | None:
    series = fetch_pit_series(
        conn=conn, table="balance_sheet", security_id=security_id, as_of_date=as_of_date
    )
    if series.empty:
        return None
    rows = series.dropna(subset=["current_assets", "current_liabilities"])
    if rows.empty:
        return None
    latest = rows.sort_values("end_date").iloc[-1]
    denom = float(latest["current_liabilities"])
    if denom == 0:
        return None
    return float(latest["current_assets"]) / denom


def _compute_debt_to_equity(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> float | None:
    series = fetch_pit_series(
        conn=conn, table="balance_sheet", security_id=security_id, as_of_date=as_of_date
    )
    if series.empty:
        return None
    rows = series.dropna(subset=["long_term_debt", "total_equity"])
    if rows.empty:
        return None
    latest = rows.sort_values("end_date").iloc[-1]
    denom = float(latest["total_equity"])
    if denom == 0:
        return None
    return float(latest["long_term_debt"]) / denom


def _compute_debt_to_assets(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> float | None:
    series = fetch_pit_series(
        conn=conn, table="balance_sheet", security_id=security_id, as_of_date=as_of_date
    )
    if series.empty:
        return None
    rows = series.dropna(subset=["long_term_debt", "total_assets"])
    if rows.empty:
        return None
    latest = rows.sort_values("end_date").iloc[-1]
    denom = float(latest["total_assets"])
    if denom == 0:
        return None
    return float(latest["long_term_debt"]) / denom


def _compute_fcf_latest(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> float | None:
    """operating_cash_flow - capital_expenditure on the latest row where
    both are non-null."""
    series = fetch_pit_series(
        conn=conn, table="cashflow", security_id=security_id, as_of_date=as_of_date
    )
    if series.empty:
        return None
    rows = series.dropna(subset=["operating_cash_flow", "capital_expenditure"])
    if rows.empty:
        return None
    latest = rows.sort_values("end_date").iloc[-1]
    return float(latest["operating_cash_flow"]) - float(latest["capital_expenditure"])


def _compute_fcf_ttm(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> float | None:
    """operating_cf_ttm - capex_ttm. Both must be defined."""
    series = fetch_pit_series(
        conn=conn, table="cashflow", security_id=security_id, as_of_date=as_of_date
    )
    if series.empty:
        return None
    ocf = _ttm_from_series(series, "operating_cash_flow", as_of_date)
    capex = _ttm_from_series(series, "capital_expenditure", as_of_date)
    if ocf is None or capex is None:
        return None
    return ocf - capex


def _compute_return_on_assets(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> float | None:
    is_series = fetch_pit_series(
        conn=conn, table="income_statement", security_id=security_id, as_of_date=as_of_date
    )
    bs_series = fetch_pit_series(
        conn=conn, table="balance_sheet", security_id=security_id, as_of_date=as_of_date
    )
    ni = _latest_non_null(is_series, "net_income")
    ta = _latest_non_null(bs_series, "total_assets")
    if ni is None or ta is None or ta == 0:
        return None
    return ni / ta


def _compute_return_on_equity(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> float | None:
    is_series = fetch_pit_series(
        conn=conn, table="income_statement", security_id=security_id, as_of_date=as_of_date
    )
    bs_series = fetch_pit_series(
        conn=conn, table="balance_sheet", security_id=security_id, as_of_date=as_of_date
    )
    ni = _latest_non_null(is_series, "net_income")
    te = _latest_non_null(bs_series, "total_equity")
    if ni is None or te is None or te == 0:
        return None
    return ni / te


def _compute_return_on_invested_capital(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> float | None:
    """net_income / (total_equity + long_term_debt). None if
    long_term_debt is missing (per spec: we cannot meaningfully estimate
    invested capital without the debt side).
    """
    is_series = fetch_pit_series(
        conn=conn, table="income_statement", security_id=security_id, as_of_date=as_of_date
    )
    bs_series = fetch_pit_series(
        conn=conn, table="balance_sheet", security_id=security_id, as_of_date=as_of_date
    )
    ni = _latest_non_null(is_series, "net_income")
    te = _latest_non_null(bs_series, "total_equity")
    ltd = _latest_non_null(bs_series, "long_term_debt")
    if ni is None or te is None or ltd is None:
        return None
    denom = te + ltd
    if denom == 0:
        return None
    return ni / denom


def _compute_pe_ratio_ttm(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> float | None:
    prices = fetch_prices_pit(conn=conn, security_id=security_id, as_of_date=as_of_date)
    close = _latest_close(prices)
    if close is None:
        return None
    is_series = fetch_pit_series(
        conn=conn, table="income_statement", security_id=security_id, as_of_date=as_of_date
    )
    eps_ttm = _ttm_from_series(is_series, "eps_diluted", as_of_date)
    if eps_ttm is None or eps_ttm == 0:
        return None
    return close / eps_ttm


def _build_registry() -> FeatureRegistry:
    reg = FeatureRegistry()

    # === Final-thresholded raw-grounded features (34) ===

    # Income statement latest (5)
    reg.register(
        Feature(
            name="revenue_latest",
            description="Latest non-null revenue from a PIT-visible income statement row.",
            source_tables=("income_statement",),
            compute=_make_latest_field_feature("income_statement", "revenue"),
            required=False,
            experimental=False,
            min_coverage_pct=0.85,
        )
    )
    reg.register(
        Feature(
            name="net_income_latest",
            description="Latest non-null net income from a PIT-visible income statement row.",
            source_tables=("income_statement",),
            compute=_make_latest_field_feature("income_statement", "net_income"),
            required=False,
            experimental=False,
            min_coverage_pct=0.95,
        )
    )
    reg.register(
        Feature(
            name="ebit_latest",
            description="Latest non-null EBIT from a PIT-visible income statement row.",
            source_tables=("income_statement",),
            compute=_make_latest_field_feature("income_statement", "ebit"),
            required=False,
            experimental=False,
            min_coverage_pct=0.85,
        )
    )
    reg.register(
        Feature(
            name="gross_profit_latest",
            description="Latest non-null gross profit from a PIT-visible income statement row.",
            source_tables=("income_statement",),
            compute=_make_latest_field_feature("income_statement", "gross_profit"),
            required=False,
            experimental=True,
            min_coverage_pct=0.55,
        )
    )
    reg.register(
        Feature(
            name="eps_diluted_latest",
            description="Latest non-null diluted EPS from a PIT-visible income statement row.",
            source_tables=("income_statement",),
            compute=_make_latest_field_feature("income_statement", "eps_diluted"),
            required=False,
            experimental=False,
            min_coverage_pct=0.65,
        )
    )

    # Balance sheet latest (6)
    reg.register(
        Feature(
            name="total_assets_latest",
            description="Latest non-null total assets from a PIT-visible balance sheet row.",
            source_tables=("balance_sheet",),
            compute=_make_latest_field_feature("balance_sheet", "total_assets"),
            required=False,
            experimental=False,
            min_coverage_pct=0.85,
        )
    )
    reg.register(
        Feature(
            name="total_equity_latest",
            description="Latest non-null total equity from a PIT-visible balance sheet row.",
            source_tables=("balance_sheet",),
            compute=_make_latest_field_feature("balance_sheet", "total_equity"),
            required=False,
            experimental=False,
            min_coverage_pct=0.90,
        )
    )
    reg.register(
        Feature(
            name="cash_and_equivalents_latest",
            description="Latest non-null cash and equivalents from a PIT-visible balance sheet row.",
            source_tables=("balance_sheet",),
            compute=_make_latest_field_feature("balance_sheet", "cash_and_equivalents"),
            required=False,
            experimental=False,
            min_coverage_pct=0.75,
        )
    )
    reg.register(
        Feature(
            name="current_assets_latest",
            description="Latest non-null current assets from a PIT-visible balance sheet row.",
            source_tables=("balance_sheet",),
            compute=_make_latest_field_feature("balance_sheet", "current_assets"),
            required=False,
            experimental=False,
            min_coverage_pct=0.70,
        )
    )
    reg.register(
        Feature(
            name="current_liabilities_latest",
            description="Latest non-null current liabilities from a PIT-visible balance sheet row.",
            source_tables=("balance_sheet",),
            compute=_make_latest_field_feature("balance_sheet", "current_liabilities"),
            required=False,
            experimental=False,
            min_coverage_pct=0.70,
        )
    )
    reg.register(
        Feature(
            name="long_term_debt_latest",
            description="Latest non-null long-term debt from a PIT-visible balance sheet row.",
            source_tables=("balance_sheet",),
            compute=_make_latest_field_feature("balance_sheet", "long_term_debt"),
            required=False,
            experimental=False,
            min_coverage_pct=0.65,
        )
    )

    # Cashflow latest (4)
    reg.register(
        Feature(
            name="operating_cf_latest",
            description="Latest non-null operating cash flow from a PIT-visible cashflow row.",
            source_tables=("cashflow",),
            compute=_make_latest_field_feature("cashflow", "operating_cash_flow"),
            required=False,
            experimental=False,
            min_coverage_pct=0.80,
        )
    )
    reg.register(
        Feature(
            name="investing_cf_latest",
            description="Latest non-null investing cash flow from a PIT-visible cashflow row.",
            source_tables=("cashflow",),
            compute=_make_latest_field_feature("cashflow", "investing_cash_flow"),
            required=False,
            experimental=False,
            min_coverage_pct=0.80,
        )
    )
    reg.register(
        Feature(
            name="financing_cf_latest",
            description="Latest non-null financing cash flow from a PIT-visible cashflow row.",
            source_tables=("cashflow",),
            compute=_make_latest_field_feature("cashflow", "financing_cash_flow"),
            required=False,
            experimental=False,
            min_coverage_pct=0.80,
        )
    )
    reg.register(
        Feature(
            name="capex_latest",
            description="Latest non-null capital expenditure from a PIT-visible cashflow row.",
            source_tables=("cashflow",),
            compute=_make_latest_field_feature("cashflow", "capital_expenditure"),
            required=False,
            experimental=False,
            min_coverage_pct=0.80,
        )
    )

    # Prices (8)
    reg.register(
        Feature(
            name="close_latest",
            description="Latest PIT-visible close price.",
            source_tables=("prices",),
            compute=_make_close_latest(),
            required=False,
            experimental=False,
            min_coverage_pct=0.95,
        )
    )
    reg.register(
        Feature(
            name="return_1m",
            description="Calendar-month price return: (close_now - close_30d_ago) / close_30d_ago.",
            source_tables=("prices",),
            compute=_make_price_return_feature(30),
            required=False,
            experimental=False,
            min_coverage_pct=0.90,
        )
    )
    reg.register(
        Feature(
            name="return_3m",
            description="3-month price return over a 90-day calendar lookback.",
            source_tables=("prices",),
            compute=_make_price_return_feature(90),
            required=False,
            experimental=False,
            min_coverage_pct=0.85,
        )
    )
    reg.register(
        Feature(
            name="return_6m",
            description="6-month price return over a 180-day calendar lookback.",
            source_tables=("prices",),
            compute=_make_price_return_feature(180),
            required=False,
            experimental=False,
            min_coverage_pct=0.80,
        )
    )
    reg.register(
        Feature(
            name="return_12m",
            description="12-month price return over a 365-day calendar lookback.",
            source_tables=("prices",),
            compute=_make_price_return_feature(365),
            required=False,
            experimental=False,
            min_coverage_pct=0.75,
        )
    )
    reg.register(
        Feature(
            name="volatility_30d",
            description="Stdev (ddof=1) of daily returns over the last 30 trading days.",
            source_tables=("prices",),
            compute=_make_volatility_feature(30),
            required=False,
            experimental=False,
            min_coverage_pct=0.85,
        )
    )
    reg.register(
        Feature(
            name="volatility_90d",
            description="Stdev (ddof=1) of daily returns over the last 90 trading days.",
            source_tables=("prices",),
            compute=_make_volatility_feature(90),
            required=False,
            experimental=False,
            min_coverage_pct=0.80,
        )
    )
    reg.register(
        Feature(
            name="max_drawdown_1y",
            description="Maximum drawdown over the trailing 1y (min of (close - rolling_max) / rolling_max).",
            source_tables=("prices",),
            compute=_make_max_drawdown_1y(),
            required=False,
            experimental=False,
            min_coverage_pct=0.75,
        )
    )

    # Sector one-hots (11)
    for name, label in _GICS_SECTORS:
        reg.register(
            Feature(
                name=name,
                description=f"1.0 if security's GICS sector is {label}, 0.0 otherwise; None if missing.",
                source_tables=("securities",),
                compute=_make_sector_one_hot(label),
                required=False,
                experimental=False,
                min_coverage_pct=0.95,
            )
        )

    # === Provisional-thresholded derived features (25) ===
    prov = 0.50

    # TTMs (4)
    reg.register(
        Feature(
            name="revenue_ttm",
            description="Trailing-12-month revenue (latest annual within 366d, else 4-quarter sum with span+recency guards).",
            source_tables=("income_statement",),
            compute=compute_revenue_ttm,
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )
    reg.register(
        Feature(
            name="net_income_ttm",
            description="Trailing-12-month net income via generic TTM helper.",
            source_tables=("income_statement",),
            compute=_make_ttm_feature("income_statement", "net_income"),
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )
    reg.register(
        Feature(
            name="ebit_ttm",
            description="Trailing-12-month EBIT via generic TTM helper.",
            source_tables=("income_statement",),
            compute=_make_ttm_feature("income_statement", "ebit"),
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )
    reg.register(
        Feature(
            name="gross_profit_ttm",
            description="Trailing-12-month gross profit via generic TTM helper.",
            source_tables=("income_statement",),
            compute=_make_ttm_feature("income_statement", "gross_profit"),
            required=False,
            experimental=True,
            min_coverage_pct=prov,
        )
    )

    # YoYs (4)
    reg.register(
        Feature(
            name="revenue_yoy",
            description="Period-aligned YoY revenue growth (FY_n vs FY_{n-1}; quarter-window fallback).",
            source_tables=("income_statement",),
            compute=compute_revenue_yoy_growth,
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )
    reg.register(
        Feature(
            name="net_income_yoy",
            description="Period-aligned YoY net income growth.",
            source_tables=("income_statement",),
            compute=_make_yoy_feature("income_statement", "net_income"),
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )
    reg.register(
        Feature(
            name="ebit_yoy",
            description="Period-aligned YoY EBIT growth.",
            source_tables=("income_statement",),
            compute=_make_yoy_feature("income_statement", "ebit"),
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )
    reg.register(
        Feature(
            name="eps_yoy",
            description="Period-aligned YoY diluted EPS growth.",
            source_tables=("income_statement",),
            compute=_make_yoy_feature("income_statement", "eps_diluted"),
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )

    # Margins (3)
    reg.register(
        Feature(
            name="gross_margin",
            description="Gross profit / revenue at the latest PIT-visible row with both fields.",
            source_tables=("income_statement",),
            compute=compute_gross_margin_latest,
            required=False,
            experimental=True,
            min_coverage_pct=prov,
        )
    )
    reg.register(
        Feature(
            name="ebit_margin",
            description="EBIT / revenue at the latest PIT-visible row with both fields.",
            source_tables=("income_statement",),
            compute=_make_margin_feature("ebit", "revenue"),
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )
    reg.register(
        Feature(
            name="net_margin",
            description="Net income / revenue at the latest PIT-visible row with both fields.",
            source_tables=("income_statement",),
            compute=_make_margin_feature("net_income", "revenue"),
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )

    # Balance sheet derived (6)
    reg.register(
        Feature(
            name="total_liabilities_latest",
            description="Total liabilities: prefer raw, fallback to total_assets - total_equity on the same row.",
            source_tables=("balance_sheet",),
            compute=_compute_total_liabilities_latest,
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )
    reg.register(
        Feature(
            name="equity_to_assets",
            description="Total equity / total assets at the latest PIT-visible row with both fields.",
            source_tables=("balance_sheet",),
            compute=_compute_equity_to_assets,
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )
    reg.register(
        Feature(
            name="liabilities_to_assets",
            description="Derived total liabilities (with Assets-Equity fallback) / total assets.",
            source_tables=("balance_sheet",),
            compute=_compute_liabilities_to_assets,
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )
    reg.register(
        Feature(
            name="current_ratio",
            description="Current assets / current liabilities at the latest PIT-visible row with both fields.",
            source_tables=("balance_sheet",),
            compute=_compute_current_ratio,
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )
    reg.register(
        Feature(
            name="debt_to_equity",
            description="Long-term debt / total equity at the latest PIT-visible row with both fields.",
            source_tables=("balance_sheet",),
            compute=_compute_debt_to_equity,
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )
    reg.register(
        Feature(
            name="debt_to_assets",
            description="Long-term debt / total assets at the latest PIT-visible row with both fields.",
            source_tables=("balance_sheet",),
            compute=_compute_debt_to_assets,
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )

    # Cashflow derived (4)
    reg.register(
        Feature(
            name="fcf_latest",
            description="Operating cash flow - capex at the latest PIT-visible row with both fields.",
            source_tables=("cashflow",),
            compute=_compute_fcf_latest,
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )
    reg.register(
        Feature(
            name="operating_cf_ttm",
            description="Trailing-12-month operating cash flow via generic TTM helper.",
            source_tables=("cashflow",),
            compute=_make_ttm_feature("cashflow", "operating_cash_flow"),
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )
    reg.register(
        Feature(
            name="capex_ttm",
            description="Trailing-12-month capex via generic TTM helper.",
            source_tables=("cashflow",),
            compute=_make_ttm_feature("cashflow", "capital_expenditure"),
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )
    reg.register(
        Feature(
            name="fcf_ttm",
            description="operating_cf_ttm - capex_ttm; both must be defined.",
            source_tables=("cashflow",),
            compute=_compute_fcf_ttm,
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )

    # Cross-table (4)
    reg.register(
        Feature(
            name="return_on_assets",
            description="Latest net income / latest total assets (independently selected by end_date).",
            source_tables=("income_statement", "balance_sheet"),
            compute=_compute_return_on_assets,
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )
    reg.register(
        Feature(
            name="return_on_equity",
            description="Latest net income / latest total equity (independently selected by end_date).",
            source_tables=("income_statement", "balance_sheet"),
            compute=_compute_return_on_equity,
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )
    reg.register(
        Feature(
            name="return_on_invested_capital",
            description="Net income / (total_equity + long_term_debt); None when long_term_debt is missing.",
            source_tables=("income_statement", "balance_sheet"),
            compute=_compute_return_on_invested_capital,
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )
    reg.register(
        Feature(
            name="pe_ratio_ttm",
            description="Latest close / TTM diluted EPS; None when either component is missing or EPS TTM is 0.",
            source_tables=("prices", "income_statement"),
            compute=_compute_pe_ratio_ttm,
            required=False,
            experimental=False,
            min_coverage_pct=prov,
        )
    )

    return reg


BUILTIN_REGISTRY: FeatureRegistry = _build_registry()
