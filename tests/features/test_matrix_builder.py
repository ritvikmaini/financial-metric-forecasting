"""Feature matrix builder tests.

Verifies:
- Matrix shape (rows > 500, 63 columns = 4 index + 59 features).
- The matrix routes through the registry's compute (which routes
  through fetch_pit_series), NOT a bulk SQL join.
- Anchor spot-checks carry the S4 PIT correctness through to the matrix
  layer: AAPL FY2023 revenue_ttm ≈ 383.285B; JPM FY2023 current_ratio
  is NaN (Group B sector N/A).
- Subset filtering by `securities` arg works.
- The CLI writes a parquet file.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd
import pytest

from fmf.features.as_of_grid import fiscal_year_end_grid
from fmf.features.builtin_features import BUILTIN_REGISTRY
from fmf.features.matrix_builder import build_feature_matrix

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini.duckdb"


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    if not FIXTURE.exists():
        pytest.skip("fixture not built yet")
    return duckdb.connect(str(FIXTURE), read_only=True)


@pytest.fixture
def aapl_sid(conn: duckdb.DuckDBPyConnection) -> str:
    row = conn.execute('SELECT security_id FROM "securities" WHERE symbol = ?', ["AAPL"]).fetchone()
    assert row is not None
    return str(row[0])


@pytest.fixture
def jpm_sid(conn: duckdb.DuckDBPyConnection) -> str:
    row = conn.execute('SELECT security_id FROM "securities" WHERE symbol = ?', ["JPM"]).fetchone()
    assert row is not None
    return str(row[0])


def test_build_feature_matrix_shape(conn: duckdb.DuckDBPyConnection) -> None:
    df = build_feature_matrix(conn=conn, registry=BUILTIN_REGISTRY)
    # 4 index columns + 59 features = 63 columns.
    assert len(df.columns) == 63, f"expected 63 columns, got {len(df.columns)}"
    assert len(df) > 500, f"expected >500 rows, got {len(df)}"
    # Index column dtype sanity. Modern pandas (>= 2.1) auto-infers string
    # columns to StringDtype; older pandas leaves them as object. Accept
    # either — what matters is the values are strings, not numerics.
    import pandas as pd_

    str_dtypes = (object, pd_.StringDtype())
    assert df["security_id"].dtype in str_dtypes or pd_.api.types.is_string_dtype(df["security_id"])
    assert df["symbol"].dtype in str_dtypes or pd_.api.types.is_string_dtype(df["symbol"])
    assert df["as_of_source"].dtype in str_dtypes or pd_.api.types.is_string_dtype(
        df["as_of_source"]
    )
    # as_of_date is dt.date (object-typed in pandas).
    import datetime as dt

    assert all(isinstance(d, dt.date) for d in df["as_of_date"].head(5))
    # Index columns are first 4.
    assert list(df.columns[:4]) == [
        "security_id",
        "symbol",
        "as_of_date",
        "as_of_source",
    ]
    # All feature names appear.
    for name in BUILTIN_REGISTRY.names():
        assert name in df.columns


def test_matrix_routes_through_registry_not_raw_sql(
    conn: duckdb.DuckDBPyConnection, aapl_sid: str
) -> None:
    """Patch fetch_pit_series to assert the matrix builder actually
    invokes the PIT layer. If a future refactor replaces the per-cell
    compute() call with a SQL JOIN, this test catches it.
    """
    # Patch at every import site the builtin features module uses.
    from fmf.features import builtin_features, derived, point_in_time

    real_fn = point_in_time.fetch_pit_series
    call_count = {"n": 0}

    def counting_fetch(*args: object, **kwargs: object) -> pd.DataFrame:
        call_count["n"] += 1
        return real_fn(*args, **kwargs)  # type: ignore[arg-type]

    with (
        patch.object(point_in_time, "fetch_pit_series", side_effect=counting_fetch),
        patch.object(builtin_features, "fetch_pit_series", side_effect=counting_fetch),
        patch.object(derived, "fetch_pit_series", side_effect=counting_fetch),
    ):
        df = build_feature_matrix(
            conn=conn,
            registry=BUILTIN_REGISTRY,
            grid_strategy=fiscal_year_end_grid,
            securities=[(aapl_sid, "AAPL")],
        )

    assert call_count["n"] > 0, (
        "fetch_pit_series was not invoked — matrix builder may be using a "
        "raw SQL JOIN shortcut that bypasses S4's PIT primitives."
    )
    assert len(df) > 0


def test_matrix_aapl_fy2023_revenue_ttm_anchor(
    conn: duckdb.DuckDBPyConnection, aapl_sid: str
) -> None:
    """At AAPL's FY2023 10-K accepted_date (2023-11-03), revenue_ttm
    should equal the FY2023 annual ≈ $383.285B within 0.5%."""
    fy23_accepted = conn.execute(
        'SELECT accepted_date FROM "income_statement" '
        "WHERE security_id = ? AND fiscal_year = 2023 AND period = ? "
        "ORDER BY accepted_date ASC LIMIT 1",
        [aapl_sid, "FY"],
    ).fetchone()[0]
    df = build_feature_matrix(
        conn=conn,
        registry=BUILTIN_REGISTRY,
        grid_strategy=fiscal_year_end_grid,
        securities=[(aapl_sid, "AAPL")],
    )
    row = df[df["as_of_source"] == "income_statement.FY.2023"]
    assert len(row) == 1, f"expected 1 AAPL FY2023 row, got {len(row)}"
    assert row.iloc[0]["as_of_date"] == fy23_accepted or (
        # Tolerate Timestamp/date coercion via str cmp on isoformat.
        str(row.iloc[0]["as_of_date"]) == str(fy23_accepted)
    )
    revenue_ttm = row.iloc[0]["revenue_ttm"]
    assert revenue_ttm is not None and not math.isnan(revenue_ttm)
    err = abs(revenue_ttm - 383_285_000_000) / 383_285_000_000
    assert err < 0.005, (
        f"AAPL FY2023 revenue_ttm: got {revenue_ttm}, "
        f"expected within 0.5% of 383.285B (err={err:.4f})"
    )


def test_matrix_jpm_fy2023_current_ratio_none(
    conn: duckdb.DuckDBPyConnection, jpm_sid: str
) -> None:
    """JPM is a Group B (bank) sector; current_ratio is not meaningful
    for banks because the balance-sheet partition uses different line
    items. The PIT layer correctly returns None which becomes NaN in
    the DataFrame."""
    df = build_feature_matrix(
        conn=conn,
        registry=BUILTIN_REGISTRY,
        grid_strategy=fiscal_year_end_grid,
        securities=[(jpm_sid, "JPM")],
    )
    row = df[df["as_of_source"] == "income_statement.FY.2023"]
    if len(row) == 0:
        pytest.skip("JPM FY2023 not in fixture")
    val = row.iloc[0]["current_ratio"]
    # None -> NaN in pandas float columns; explicit None stays None in
    # object columns. Either way, must not be a finite number.
    assert val is None or (isinstance(val, float) and math.isnan(val)), (
        f"JPM FY2023 current_ratio: expected NaN/None (Group B sector), got {val!r}"
    )


def test_matrix_subset_of_securities(conn: duckdb.DuckDBPyConnection, aapl_sid: str) -> None:
    df = build_feature_matrix(
        conn=conn,
        registry=BUILTIN_REGISTRY,
        grid_strategy=fiscal_year_end_grid,
        securities=[(aapl_sid, "AAPL")],
    )
    assert (df["symbol"] == "AAPL").all()
    assert (df["security_id"] == aapl_sid).all()


def test_cli_writes_parquet(tmp_path: Path) -> None:
    if not FIXTURE.exists():
        pytest.skip("fixture not built yet")
    out = tmp_path / "matrix.parquet"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.build_feature_matrix",
            "--db",
            str(FIXTURE),
            "--out",
            str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) > 500
    assert len(df.columns) == 63


def test_cli_writes_csv_with_csv_suffix(tmp_path: Path) -> None:
    if not FIXTURE.exists():
        pytest.skip("fixture not built yet")
    out = tmp_path / "matrix.csv"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.build_feature_matrix",
            "--db",
            str(FIXTURE),
            "--grid",
            "fiscal_year_end",
            "--out",
            str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) > 0
    assert len(df.columns) == 63
