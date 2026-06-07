"""Unit tests for the S14 prediction cache primitives.

Coverage matrix:
- data_fingerprint: determinism + sensitivity (new IS row, new price row, distinct sids)
- derive_cache_key: all four axes (version, config, fingerprint, coordinate)
- PredictionCache: get on missing, put_batch round-trip, idempotent replace
"""

from __future__ import annotations

import datetime as dt
import uuid

import duckdb
import pytest

from fmf.equity.forecasting.evaluation import prediction_cache
from fmf.equity.forecasting.evaluation.prediction_cache import (
    PredictionCache,
    data_fingerprint,
    derive_cache_key,
)


def _conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    c.execute(
        'CREATE TABLE "income_statement" ('
        "security_id UUID, fiscal_year INTEGER, period TEXT, "
        "accepted_date DATE, end_date DATE, eps_diluted DOUBLE)"
    )
    c.execute('CREATE TABLE "prices" (security_id UUID, date DATE, close DOUBLE)')
    return c


def _sid(n: int) -> uuid.UUID:
    return uuid.UUID(f"00000000-0000-0000-0000-{n:012d}")


def _seed(conn: duckdb.DuckDBPyConnection, sid: uuid.UUID) -> None:
    conn.execute(
        'INSERT INTO "income_statement" VALUES (?, ?, ?, ?, ?, ?)',
        [str(sid), 2022, "FY", dt.date(2022, 11, 4), dt.date(2022, 9, 24), 6.11],
    )
    conn.execute(
        'INSERT INTO "income_statement" VALUES (?, ?, ?, ?, ?, ?)',
        [str(sid), 2023, "FY", dt.date(2023, 11, 3), dt.date(2023, 9, 30), 6.13],
    )
    conn.execute(
        'INSERT INTO "prices" VALUES (?, ?, ?)',
        [str(sid), dt.date(2023, 12, 29), 192.53],
    )


def test_data_fingerprint_is_deterministic_on_unchanged_data() -> None:
    conn = _conn()
    sid = _sid(1)
    _seed(conn, sid)
    fp_a = data_fingerprint(conn, sid)
    fp_b = data_fingerprint(conn, sid)
    assert fp_a == fp_b


def test_data_fingerprint_changes_on_new_income_statement_row() -> None:
    conn = _conn()
    sid = _sid(1)
    _seed(conn, sid)
    fp_before = data_fingerprint(conn, sid)
    conn.execute(
        'INSERT INTO "income_statement" VALUES (?, ?, ?, ?, ?, ?)',
        [str(sid), 2024, "FY", dt.date(2024, 11, 1), dt.date(2024, 9, 28), 6.72],
    )
    fp_after = data_fingerprint(conn, sid)
    assert fp_after != fp_before


def test_data_fingerprint_changes_on_new_price_row() -> None:
    conn = _conn()
    sid = _sid(1)
    _seed(conn, sid)
    fp_before = data_fingerprint(conn, sid)
    conn.execute(
        'INSERT INTO "prices" VALUES (?, ?, ?)',
        [str(sid), dt.date(2024, 1, 2), 195.10],
    )
    fp_after = data_fingerprint(conn, sid)
    assert fp_after != fp_before


def test_data_fingerprint_differs_across_securities() -> None:
    conn = _conn()
    sid_a = _sid(1)
    sid_b = _sid(2)
    _seed(conn, sid_a)
    conn.execute(
        'INSERT INTO "income_statement" VALUES (?, ?, ?, ?, ?, ?)',
        [str(sid_b), 2023, "FY", dt.date(2023, 7, 28), dt.date(2023, 6, 30), 9.81],
    )
    conn.execute(
        'INSERT INTO "prices" VALUES (?, ?, ?)',
        [str(sid_b), dt.date(2023, 12, 29), 376.04],
    )
    assert data_fingerprint(conn, sid_a) != data_fingerprint(conn, sid_b)


def _base_key_kwargs() -> dict[str, object]:
    return {
        "config_flags_hash": "cfg-hash",
        "fingerprint": "fp-hash",
        "security_id": _sid(1),
        "as_of_date": dt.date(2024, 5, 15),
        "metric": "eps_diluted",
        "model_name": "LightGBM",
    }


def test_derive_cache_key_is_deterministic() -> None:
    k1 = derive_cache_key(**_base_key_kwargs())  # type: ignore[arg-type]
    k2 = derive_cache_key(**_base_key_kwargs())  # type: ignore[arg-type]
    assert k1 == k2


def test_derive_cache_key_changes_on_version_bump(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _base_key_kwargs()
    k_before = derive_cache_key(**base)  # type: ignore[arg-type]
    monkeypatch.setattr(prediction_cache, "CACHE_VERSION", 999)
    k_after = derive_cache_key(**base)  # type: ignore[arg-type]
    assert k_before != k_after


def test_derive_cache_key_changes_on_fingerprint_change() -> None:
    base = _base_key_kwargs()
    k1 = derive_cache_key(**base)  # type: ignore[arg-type]
    base["fingerprint"] = "fp-hash-mutated"
    k2 = derive_cache_key(**base)  # type: ignore[arg-type]
    assert k1 != k2


def test_derive_cache_key_changes_on_config_change() -> None:
    base = _base_key_kwargs()
    k1 = derive_cache_key(**base)  # type: ignore[arg-type]
    base["config_flags_hash"] = "cfg-hash-other"
    k2 = derive_cache_key(**base)  # type: ignore[arg-type]
    assert k1 != k2


def test_derive_cache_key_changes_on_coordinate_change() -> None:
    base = _base_key_kwargs()
    k0 = derive_cache_key(**base)  # type: ignore[arg-type]
    base_sid = dict(base)
    base_sid["security_id"] = _sid(2)
    base_asof = dict(base)
    base_asof["as_of_date"] = dt.date(2024, 6, 15)
    base_metric = dict(base)
    base_metric["metric"] = "ebitda"
    base_model = dict(base)
    base_model["model_name"] = "TiRex"
    keys = {
        k0,
        derive_cache_key(**base_sid),  # type: ignore[arg-type]
        derive_cache_key(**base_asof),  # type: ignore[arg-type]
        derive_cache_key(**base_metric),  # type: ignore[arg-type]
        derive_cache_key(**base_model),  # type: ignore[arg-type]
    }
    assert len(keys) == 5


def test_cache_get_returns_none_for_missing(tmp_path) -> None:
    cache = PredictionCache(tmp_path / "cache.db")
    try:
        assert cache.get("never-written") is None
        assert cache.contains("never-written") is False
    finally:
        cache.close()


def test_cache_put_batch_then_get_returns_value(tmp_path) -> None:
    cache = PredictionCache(tmp_path / "cache.db")
    try:
        cache.put_batch([("k1", 6.50), ("k2", None), ("k3", -1.25)])
        assert cache.get("k1") == 6.50
        assert cache.get("k2") is None
        assert cache.contains("k2") is True
        assert cache.get("k3") == -1.25
        assert cache.size() == 3
    finally:
        cache.close()


def test_cache_put_batch_is_idempotent_on_duplicate_key(tmp_path) -> None:
    cache = PredictionCache(tmp_path / "cache.db")
    try:
        cache.put_batch([("k1", 1.0)])
        cache.put_batch([("k1", 2.0)])
        assert cache.get("k1") == 2.0
        assert cache.size() == 1
    finally:
        cache.close()
