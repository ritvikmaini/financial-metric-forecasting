"""Built-in feature registry tests.

Shape, validation, and a handful of correctness sanity checks. The
exhaustive coverage check belongs to scripts/coverage_audit.py.
"""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import duckdb
import pytest

from fmf.features.builtin_features import (
    BUILTIN_REGISTRY,
    _compute_total_liabilities_latest,
)
from fmf.features.feature_registry import validate_registry

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini.duckdb"


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    if not FIXTURE.exists():
        pytest.skip("fixture not built yet")
    return duckdb.connect(str(FIXTURE), read_only=True)


@pytest.fixture
def aapl_security_id(conn: duckdb.DuckDBPyConnection) -> uuid.UUID:
    row = conn.execute('SELECT security_id FROM "securities" WHERE symbol = ?', ["AAPL"]).fetchone()
    assert row is not None
    return uuid.UUID(str(row[0]))


def test_registry_has_64_features() -> None:
    """59 base + 5 S18 earnings-quality cluster composites."""
    assert len(BUILTIN_REGISTRY) == 64


def test_registry_validates() -> None:
    validate_registry(BUILTIN_REGISTRY)


def test_registry_has_eight_experimental_features() -> None:
    """Three legacy coverage-dependent experimentals plus the five
    S18 earnings-quality cluster composites (shipped behind the single
    EARNINGS_QUALITY_CLUSTER flag).
    """
    exp = {f.name for f in BUILTIN_REGISTRY if f.experimental}
    assert exp == {
        "gross_profit_latest",
        "gross_profit_ttm",
        "gross_margin",
        "piotroski_f_score",
        "ccc_days",
        "dechow_accruals",
        "beneish_m_score",
        "mohanram_g_score",
    }


def test_registry_thresholds_all_final() -> None:
    """Post-finalize: 59 base thresholds anchored to measured values
    (raw-coverage for raw-grounded, T4 audit measured-5pp capped at 0.95
    for derived). The five S18 cluster composites carry min_coverage_pct=0.0
    by design (schema-gap fields under IDEA-S18-002 / 003); they are
    excluded from the legacy threshold range check.
    """
    cluster_names = {
        "piotroski_f_score",
        "ccc_days",
        "dechow_accruals",
        "beneish_m_score",
        "mohanram_g_score",
    }
    legacy_floors = [f.min_coverage_pct for f in BUILTIN_REGISTRY if f.name not in cluster_names]
    assert min(legacy_floors) >= 0.55, (
        f"some legacy feature still at a provisional floor: min={min(legacy_floors)}"
    )
    assert max(legacy_floors) <= 0.95, (
        f"some legacy feature exceeds the 0.95 cap: max={max(legacy_floors)}"
    )
    assert all(f.min_coverage_pct != 0.50 for f in BUILTIN_REGISTRY)


def test_every_feature_is_invokable_for_aapl(
    conn: duckdb.DuckDBPyConnection,
    aapl_security_id: uuid.UUID,
) -> None:
    """Every feature compute() must run without raising for a real
    (security, as_of). Value can be None or numeric."""
    as_of = dt.date(2024, 1, 1)
    for f in BUILTIN_REGISTRY:
        try:
            v = f.compute(conn=conn, security_id=aapl_security_id, as_of_date=as_of)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"feature {f.name!r} raised at AAPL/{as_of}: {exc!r}")
        assert v is None or isinstance(v, float | int), (
            f"feature {f.name!r} returned non-numeric {type(v).__name__}"
        )


def test_total_liabilities_latest_falls_back_to_assets_minus_equity(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """GWW at 2009-08-01: raw total_liabilities is null on every PIT-visible
    BS row but raw total_assets and total_equity are non-null on the FY2008
    annual (and the Q2 2009 row). The Assets-Equity fallback fires on the
    latest such row (Q2 2009 end_date 2009-06-30) and returns
    3,405,765,000 - 2,110,678,000 = 1,295,087,000.
    """
    gwwq = conn.execute('SELECT security_id FROM "securities" WHERE symbol = ?', ["GWW"]).fetchone()
    assert gwwq is not None
    gww_sid = uuid.UUID(str(gwwq[0]))
    as_of = dt.date(2009, 8, 1)
    v = _compute_total_liabilities_latest(conn=conn, security_id=gww_sid, as_of_date=as_of)
    assert v is not None, (
        "expected Assets-Equity fallback to compute a non-None value when "
        "raw total_liabilities is null but total_assets and total_equity are non-null"
    )
    expected = 3_405_765_000.0 - 2_110_678_000.0
    assert abs(v - expected) < 1.0, (
        f"GWW total_liabilities_latest fallback: got {v}, expected {expected}"
    )
