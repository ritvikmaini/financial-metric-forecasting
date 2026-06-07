"""Unit tests for the SQLite run registry."""

from __future__ import annotations

import datetime as dt
import sqlite3
import uuid
from pathlib import Path

import pytest

from fmf.research.fmf_runs import (
    Registry,
    RunRecord,
    config_flags_hash,
    run_id_for,
)


def _make_record(
    *,
    mode: str = "adhoc",
    metric: str = "eps_diluted",
    cfg: dict | None = None,
    started_at: dt.datetime | None = None,
) -> RunRecord:
    cfg = cfg if cfg is not None else {"metric": metric, "start_year": 2020, "end_year": 2022}
    cfg_hash = config_flags_hash(cfg)
    started = started_at or dt.datetime(2026, 6, 7, 12, 0, 0, tzinfo=dt.UTC)
    rid = run_id_for(
        mode=mode,  # type: ignore[arg-type]
        config_flags_hash_value=cfg_hash,
        window="2020-2022",
        metric=metric,
        started_at_iso_minute=started.replace(second=0, microsecond=0).isoformat(),
    )
    return RunRecord(
        run_id=rid,
        mode=mode,  # type: ignore[arg-type]
        config_flags_hash=cfg_hash,
        commit_sha=None,
        metric=metric,
        start_year=2020,
        end_year=2022,
        n_securities=5,
        n_rows_scored=42,
        status="ok",
        started_at=started,
        finished_at=started + dt.timedelta(seconds=30),
        config=cfg,
    )


def test_init_creates_schema(tmp_path: Path) -> None:
    db = tmp_path / "r.db"
    reg = Registry(db)
    reg.close()
    conn = sqlite3.connect(str(db))
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    conn.close()
    assert {"runs", "run_config", "_migrations"}.issubset(tables)


def test_init_writes_migration_row(tmp_path: Path) -> None:
    db = tmp_path / "r.db"
    reg = Registry(db)
    reg.close()
    conn = sqlite3.connect(str(db))
    rows = conn.execute("SELECT version FROM _migrations").fetchall()
    conn.close()
    assert [r[0] for r in rows] == [1]


def test_init_refuses_newer_schema(tmp_path: Path) -> None:
    db = tmp_path / "r.db"
    reg = Registry(db)
    reg.close()
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE _migrations SET version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="version 99"):
        Registry(db)


def test_config_flags_hash_is_deterministic() -> None:
    h1 = config_flags_hash({"a": 1, "b": "x", "c": [1, 2]})
    h2 = config_flags_hash({"c": [1, 2], "b": "x", "a": 1})
    assert h1 == h2
    assert len(h1) == 64


def test_config_flags_hash_differs_on_value_change() -> None:
    h1 = config_flags_hash({"a": 1})
    h2 = config_flags_hash({"a": 2})
    assert h1 != h2


def test_run_id_for_is_deterministic() -> None:
    args = {
        "mode": "adhoc",
        "config_flags_hash_value": "abc",
        "window": "2020-2022",
        "metric": "eps_diluted",
        "started_at_iso_minute": "2026-06-07T12:00:00+00:00",
    }
    r1 = run_id_for(**args)  # type: ignore[arg-type]
    r2 = run_id_for(**args)  # type: ignore[arg-type]
    assert r1 == r2
    assert isinstance(r1, uuid.UUID)


def test_record_run_inserts_and_returns_true(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "r.db")
    try:
        record = _make_record()
        assert reg.record_run(record) is True
        got = reg.get_run(record.run_id)
        assert got is not None
        assert got.metric == "eps_diluted"
        assert got.config == record.config
    finally:
        reg.close()


def test_record_run_is_idempotent_returns_false(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "r.db")
    try:
        record = _make_record()
        assert reg.record_run(record) is True
        assert reg.record_run(record) is False
        rows = reg._conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        assert rows[0] == 1
    finally:
        reg.close()


def test_list_runs_filters_by_mode_and_metric(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "r.db")
    try:
        t0 = dt.datetime(2026, 6, 7, 12, 0, 0, tzinfo=dt.UTC)
        reg.record_run(
            _make_record(
                mode="adhoc",
                metric="eps_diluted",
                cfg={"k": 1},
                started_at=t0,
            )
        )
        reg.record_run(
            _make_record(
                mode="backfill",
                metric="ebitda",
                cfg={"k": 2},
                started_at=t0 + dt.timedelta(minutes=1),
            )
        )
        reg.record_run(
            _make_record(
                mode="adhoc",
                metric="ebitda",
                cfg={"k": 3},
                started_at=t0 + dt.timedelta(minutes=2),
            )
        )
        all_runs = reg.list_runs()
        assert len(all_runs) == 3
        adhoc = reg.list_runs(mode="adhoc")
        assert len(adhoc) == 2
        eps = reg.list_runs(metric="eps_diluted")
        assert len(eps) == 1
        both = reg.list_runs(mode="adhoc", metric="ebitda")
        assert len(both) == 1
    finally:
        reg.close()


def test_verify_flags_config_drift(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "r.db")
    try:
        record = _make_record(cfg={"a": 1, "b": "x"})
        reg.record_run(record)
        assert reg.verify() == []
        reg._conn.execute(
            "UPDATE run_config SET value = ? WHERE key = ?",
            ('"tampered"', "b"),
        )
        reg._conn.commit()
        mismatches = reg.verify()
        assert len(mismatches) == 1
        rid, stored, recomputed = mismatches[0]
        assert rid == record.run_id
        assert stored != recomputed
    finally:
        reg.close()
