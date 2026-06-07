"""Tests for pipeline.dataset_builder."""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd
import pytest

from fmf.pipeline.dataset_builder import build_inference_dataset

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini.duckdb"


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    if not FIXTURE.exists():
        pytest.skip("fixture not built yet")
    return duckdb.connect(str(FIXTURE), read_only=True)


@pytest.fixture
def aapl_id(conn: duckdb.DuckDBPyConnection) -> uuid.UUID:
    row = conn.execute('SELECT security_id FROM "securities" WHERE symbol = ?', ["AAPL"]).fetchone()
    assert row is not None
    return uuid.UUID(str(row[0]))


def test_builder_routes_through_registry_path(
    conn: duckdb.DuckDBPyConnection, aapl_id: uuid.UUID
) -> None:
    """compute_feature_matrix is the registry path; the builder must call it
    rather than synthesizing values through some shortcut."""
    with patch(
        "fmf.pipeline.dataset_builder.compute_feature_matrix",
        return_value={"revenue_ttm": 1.0, "gross_margin": 0.5},
    ) as mocked:
        df = build_inference_dataset(
            conn=conn,
            security_ids=[aapl_id],
            as_of_date=dt.date(2023, 6, 30),
            feature_ids=["revenue_ttm", "gross_margin"],
        )
    assert mocked.call_count == 1
    assert df.loc[0, "revenue_ttm"] == 1.0
    assert df.loc[0, "gross_margin"] == 0.5


def test_unknown_feature_id_raises(conn: duckdb.DuckDBPyConnection, aapl_id: uuid.UUID) -> None:
    with pytest.raises(ValueError, match="unknown feature_ids"):
        build_inference_dataset(
            conn=conn,
            security_ids=[aapl_id],
            as_of_date=dt.date(2023, 6, 30),
            feature_ids=["not_a_real_feature"],
        )


def test_empty_security_list_returns_empty_frame(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    df = build_inference_dataset(
        conn=conn,
        security_ids=[],
        as_of_date=dt.date(2023, 6, 30),
        feature_ids=["revenue_ttm"],
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "revenue_ttm" in df.columns
    assert "security_id" in df.columns
    assert "symbol" in df.columns
    assert "as_of_date" in df.columns


def test_single_security_returns_one_row(
    conn: duckdb.DuckDBPyConnection, aapl_id: uuid.UUID
) -> None:
    df = build_inference_dataset(
        conn=conn,
        security_ids=[aapl_id],
        as_of_date=dt.date(2023, 6, 30),
        feature_ids=["revenue_ttm", "gross_margin"],
    )
    assert len(df) == 1
    assert df.loc[0, "symbol"] == "AAPL"
    assert df.loc[0, "security_id"] == str(aapl_id)
    assert df.loc[0, "as_of_date"] == dt.date(2023, 6, 30)


def test_column_schema_matches_feature_ids(
    conn: duckdb.DuckDBPyConnection, aapl_id: uuid.UUID
) -> None:
    feature_ids = ["revenue_ttm", "gross_margin", "net_margin"]
    df = build_inference_dataset(
        conn=conn,
        security_ids=[aapl_id],
        as_of_date=dt.date(2023, 6, 30),
        feature_ids=feature_ids,
    )
    assert set(df.columns) == {"security_id", "symbol", "as_of_date", *feature_ids}
