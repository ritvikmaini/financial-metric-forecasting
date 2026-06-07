"""ExpandingWindowBacktester orchestrator unit tests."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from fmf.equity.forecasting.evaluation._backtester_config import BacktesterConfig
from fmf.equity.forecasting.evaluation.backtester import (
    BacktestResult,
    ExpandingWindowBacktester,
)
from tests.equity.forecasting.evaluation._fixture_helpers import (
    aapl_security_id,
    fixture_conn,
)


def test_materialize_candidates_uses_configured_grid_strategy() -> None:
    conn = fixture_conn()
    try:
        aapl = aapl_security_id(conn)
        cfg = BacktesterConfig(
            metric="eps_diluted",
            start_year=2018,
            end_year=2024,
            grid_strategy="fiscal_year_end",
            feature_ids=("revenue_ttm",),
            min_train_samples=3,
        )
        bt = ExpandingWindowBacktester(conn, cfg)
        rows = bt._materialize_candidates([aapl])
        assert len(rows) >= 6
        assert "security_id" in rows.columns
        assert "as_of_date" in rows.columns
    finally:
        conn.close()


def test_attach_targets_drops_rows_with_no_next_fy() -> None:
    conn = fixture_conn()
    try:
        aapl = aapl_security_id(conn)
        cfg = BacktesterConfig(
            metric="eps_diluted",
            start_year=2018,
            end_year=2024,
            grid_strategy="fiscal_year_end",
            feature_ids=("revenue_ttm",),
            min_train_samples=3,
        )
        bt = ExpandingWindowBacktester(conn, cfg)
        candidates = bt._materialize_candidates([aapl])
        sentinel = pd.DataFrame(
            [
                {
                    "security_id": aapl,
                    "symbol": "AAPL",
                    "as_of_date": dt.date(2099, 1, 1),
                    "as_of_source": "synthetic",
                }
            ]
        )
        candidates = pd.concat([candidates, sentinel], ignore_index=True)
        result = BacktestResult(config=cfg, folds=[])
        attached = bt._attach_targets_and_baseline(candidates, result)
        assert (attached["as_of_date"] != dt.date(2099, 1, 1)).all()
        assert result.unresolved_target_count >= 1
        for col in (
            "target_fy",
            "target_accepted_date",
            "target_value",
            "horizon_days",
            "naive_baseline",
        ):
            assert col in attached.columns
    finally:
        conn.close()


def test_horizon_days_equals_target_accepted_minus_as_of() -> None:
    conn = fixture_conn()
    try:
        aapl = aapl_security_id(conn)
        cfg = BacktesterConfig(
            metric="eps_diluted",
            start_year=2018,
            end_year=2024,
            grid_strategy="fiscal_year_end",
            feature_ids=("revenue_ttm",),
            min_train_samples=3,
        )
        bt = ExpandingWindowBacktester(conn, cfg)
        rows = bt._materialize_candidates([aapl])
        result = BacktestResult(config=cfg, folds=[])
        attached = bt._attach_targets_and_baseline(rows, result)
        for _, r in attached.iterrows():
            assert r["horizon_days"] == (r["target_accepted_date"] - r["as_of_date"]).days
            assert r["horizon_days"] > 0
    finally:
        conn.close()
