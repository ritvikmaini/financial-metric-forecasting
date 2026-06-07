"""CLI smoke tests for scripts/fmf_runs_cli.py."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fmf.research.fmf_runs import (
    Registry,
    RunRecord,
    config_flags_hash,
    run_id_for,
)
from scripts.fmf_runs_cli import app

pytestmark = pytest.mark.slow


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _seed_one(db: Path) -> RunRecord:
    cfg = {"metric": "eps_diluted", "start_year": 2020, "end_year": 2022}
    h = config_flags_hash(cfg)
    started = dt.datetime(2026, 6, 7, 12, 0, 0, tzinfo=dt.UTC)
    rid = run_id_for(
        mode="adhoc",
        config_flags_hash_value=h,
        window="2020-2022",
        metric="eps_diluted",
        started_at_iso_minute=started.isoformat(),
    )
    record = RunRecord(
        run_id=rid,
        mode="adhoc",
        config_flags_hash=h,
        commit_sha=None,
        metric="eps_diluted",
        start_year=2020,
        end_year=2022,
        n_securities=5,
        n_rows_scored=42,
        status="ok",
        started_at=started,
        finished_at=started + dt.timedelta(seconds=30),
        config=cfg,
    )
    reg = Registry(db)
    try:
        reg.record_run(record)
    finally:
        reg.close()
    return record


def test_list_outputs_inserted_run(runner: CliRunner, tmp_path: Path) -> None:
    db = tmp_path / "r.db"
    record = _seed_one(db)
    result = runner.invoke(app, ["list", "--registry", str(db)])
    assert result.exit_code == 0, result.output
    assert str(record.run_id) in result.output


def test_show_returns_full_record(runner: CliRunner, tmp_path: Path) -> None:
    db = tmp_path / "r.db"
    record = _seed_one(db)
    result = runner.invoke(app, ["show", str(record.run_id), "--registry", str(db)])
    assert result.exit_code == 0, result.output
    assert record.config_flags_hash in result.output
    assert "eps_diluted" in result.output


def test_verify_returns_zero_when_clean(runner: CliRunner, tmp_path: Path) -> None:
    db = tmp_path / "r.db"
    _seed_one(db)
    result = runner.invoke(app, ["verify", "--registry", str(db)])
    assert result.exit_code == 0, result.output
