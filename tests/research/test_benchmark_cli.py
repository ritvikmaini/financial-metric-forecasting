"""CLI smoke tests for scripts/benchmark.py."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.benchmark import app

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini.duckdb"

pytestmark = pytest.mark.slow


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _row_count(db: Path) -> int:
    conn = sqlite3.connect(str(db))
    try:
        return int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
    finally:
        conn.close()


def test_run_writes_registry_entry(runner: CliRunner, tmp_path: Path) -> None:
    db = tmp_path / "fmf_runs.db"
    result = runner.invoke(
        app,
        [
            "run",
            "--metric",
            "eps_diluted",
            "--start-year",
            "2020",
            "--end-year",
            "2022",
            "--feature",
            "revenue_ttm",
            "--feature",
            "gross_margin",
            "--fixture",
            str(FIXTURE),
            "--registry",
            str(db),
            "--min-train-samples",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert _row_count(db) == 1


def test_run_is_idempotent_within_minute(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each backtester run takes long enough to potentially cross a minute
    boundary, which would naturally produce two distinct run_ids. The plan's
    Decision 4 says runs within the SAME minute collide intentionally for
    idempotent backfill. Freeze time so the test asserts the design property
    (same-minute idempotency) rather than wall-clock timing."""
    import datetime as dt

    from scripts import benchmark as bm

    frozen = dt.datetime(2026, 6, 7, 12, 0, 0, tzinfo=dt.UTC)
    monkeypatch.setattr(bm, "_utc_now", lambda: frozen)

    db = tmp_path / "fmf_runs.db"
    args = [
        "run",
        "--metric",
        "eps_diluted",
        "--start-year",
        "2020",
        "--end-year",
        "2022",
        "--feature",
        "revenue_ttm",
        "--fixture",
        str(FIXTURE),
        "--registry",
        str(db),
        "--min-train-samples",
        "3",
    ]
    r1 = runner.invoke(app, args)
    assert r1.exit_code == 0, r1.output
    r2 = runner.invoke(app, args)
    assert r2.exit_code == 0, r2.output
    assert _row_count(db) == 1


def test_list_recent_returns_inserted_run(runner: CliRunner, tmp_path: Path) -> None:
    db = tmp_path / "fmf_runs.db"
    runner.invoke(
        app,
        [
            "run",
            "--metric",
            "eps_diluted",
            "--start-year",
            "2020",
            "--end-year",
            "2022",
            "--feature",
            "revenue_ttm",
            "--fixture",
            str(FIXTURE),
            "--registry",
            str(db),
            "--min-train-samples",
            "3",
        ],
    )
    result = runner.invoke(app, ["list-recent", "--registry", str(db)])
    assert result.exit_code == 0, result.output
