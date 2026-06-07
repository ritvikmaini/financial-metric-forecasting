"""Pipeline-stage 1: build inference panel at a single as_of."""

from __future__ import annotations

import datetime as dt
import uuid

import duckdb
import pandas as pd

from fmf.features.builtin_features import BUILTIN_REGISTRY
from fmf.features.feature_registry import FeatureRegistry, compute_feature_matrix


def build_inference_dataset(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_ids: list[uuid.UUID],
    as_of_date: dt.date,
    feature_ids: list[str],
) -> pd.DataFrame:
    """One row per (security, as_of_date); columns are the configured feature_ids.

    Reuses the registry-routed PIT path; never bypasses fetch_pit_series.
    """
    builtin_by_name = {f.name: f for f in BUILTIN_REGISTRY}
    missing = set(feature_ids) - set(builtin_by_name)
    if missing:
        raise ValueError(f"unknown feature_ids: {sorted(missing)}")
    sub = FeatureRegistry()
    for fid in feature_ids:
        sub.register(builtin_by_name[fid])
    if not security_ids:
        return pd.DataFrame(
            columns=["security_id", "symbol", "as_of_date", *feature_ids],
        )
    symbols = _fetch_symbols(conn, security_ids)
    matrix_rows: list[dict[str, object]] = []
    for sid in security_ids:
        values = compute_feature_matrix(
            conn=conn,
            reg=sub,
            security_id=sid,
            as_of_date=as_of_date,
        )
        row: dict[str, object] = {
            "security_id": str(sid),
            "symbol": symbols.get(sid, "UNKNOWN"),
            "as_of_date": as_of_date,
        }
        row.update(values)
        matrix_rows.append(row)
    return pd.DataFrame(matrix_rows)


def _fetch_symbols(
    conn: duckdb.DuckDBPyConnection, security_ids: list[uuid.UUID]
) -> dict[uuid.UUID, str]:
    placeholders = ",".join(["?"] * len(security_ids))
    rows = conn.execute(
        f'SELECT security_id, symbol FROM "securities" WHERE security_id IN ({placeholders})',
        [str(s) for s in security_ids],
    ).fetchall()
    return {uuid.UUID(str(r[0])): r[1] for r in rows}
