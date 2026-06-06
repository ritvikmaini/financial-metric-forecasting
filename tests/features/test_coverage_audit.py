"""Coverage audit script tests."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from scripts.coverage_audit import _measure_feature, _samples, run_audit

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini.duckdb"


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    if not FIXTURE.exists():
        pytest.skip("fixture not built yet")
    return duckdb.connect(str(FIXTURE), read_only=True)


def test_samples_returns_per_security_fy_pairs(conn: duckdb.DuckDBPyConnection) -> None:
    samples = _samples(conn)
    assert len(samples) > 0
    # No duplicate (security_id, fiscal_year).
    keys = [(s, fy) for s, fy, _ in samples]
    assert len(keys) == len(set(keys))


def test_measure_feature_returns_within_bounds(conn: duckdb.DuckDBPyConnection) -> None:
    from fmf.features.builtin_features import BUILTIN_REGISTRY

    samples = _samples(conn)
    # Net income should be near-fully populated; coverage should be high.
    feat = BUILTIN_REGISTRY["net_income_latest"]
    filled, total = _measure_feature(conn, feat, samples)
    assert total == len(samples)
    assert 0 <= filled <= total
    coverage = filled / total
    assert coverage >= 0.95


def test_run_audit_writes_report_and_returns_int(tmp_path: Path) -> None:
    # End-to-end run. Exit code may be 0 or 1 depending on provisional
    # thresholds vs measured; both are acceptable from the audit's POV.
    rc = run_audit()
    assert rc in (0, 1)
    report = REPO_ROOT / "reports" / "coverage_audit.json"
    assert report.exists()
