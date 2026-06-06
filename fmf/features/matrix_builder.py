"""Feature-matrix builder.

build_feature_matrix iterates an as_of grid and calls the registry's
compute_feature_matrix per (security, as_of). The returned DataFrame
is wide-format: 4 index columns + 1 column per registered feature.

PIT contract: every feature value goes through the registry's compute()
callback, which routes through fetch_pit_series / fetch_consensus_pit /
fetch_prices_pit. NO bulk denormalized JOIN over raw tables. A SQL
shortcut would bypass S4's PIT primitives and re-introduce look-ahead.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterable

import duckdb
import pandas as pd

from fmf.features.as_of_grid import AsOfSample, filing_dates_grid
from fmf.features.feature_registry import (
    FeatureRegistry,
    compute_feature_matrix,
)

log = logging.getLogger(__name__)


GridStrategy = Callable[..., list[AsOfSample]]


def build_feature_matrix(
    *,
    conn: duckdb.DuckDBPyConnection,
    registry: FeatureRegistry,
    grid_strategy: GridStrategy = filing_dates_grid,
    securities: Iterable[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Build the (security x as_of) feature matrix using the registry.

    Args:
        conn: DuckDB connection (read-only is fine).
        registry: feature registry (BUILTIN_REGISTRY in production).
        grid_strategy: function (conn, security_id, symbol) -> list[AsOfSample].
        securities: optional iterable of (security_id_str, symbol). If None,
            fetched from the securities table (all tickers).

    Returns:
        Wide-format DataFrame with index columns (security_id, symbol,
        as_of_date, as_of_source) and one column per registered feature.
        Feature columns may contain NaN where compute returned None.

    PIT contract: every cell is the value of `feature.compute(conn=conn,
    security_id=sample.security_id, as_of_date=sample.as_of_date)`. No
    SQL JOIN is used; the registry's compute functions are the only
    code paths that read fundamentals.
    """
    if securities is None:
        rows = conn.execute(
            'SELECT security_id, symbol FROM "securities" ORDER BY symbol'
        ).fetchall()
        securities = [(str(sid), sym) for sid, sym in rows]

    matrix_rows: list[dict[str, object]] = []
    feature_names = registry.names()

    for sid_str, symbol in securities:
        sid = uuid.UUID(sid_str)
        samples = grid_strategy(conn, sid, symbol)
        log.info("security=%s symbol=%s samples=%d", sid_str[:8], symbol, len(samples))
        for sample in samples:
            values = compute_feature_matrix(
                conn=conn,
                reg=registry,
                security_id=sample.security_id,
                as_of_date=sample.as_of_date,
            )
            row: dict[str, object] = {
                "security_id": str(sample.security_id),
                "symbol": sample.symbol,
                "as_of_date": sample.as_of_date,
                "as_of_source": sample.as_of_source,
            }
            for name in feature_names:
                row[name] = values.get(name)
            matrix_rows.append(row)

    df = pd.DataFrame(matrix_rows)
    # Index columns to the left, features to the right.
    index_cols = ["security_id", "symbol", "as_of_date", "as_of_source"]
    df = df[index_cols + feature_names]
    return df
