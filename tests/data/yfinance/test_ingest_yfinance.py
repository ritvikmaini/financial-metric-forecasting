"""ingest_yfinance.py CLI smoke test."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.parent


def test_ingest_yfinance_cli_against_sample_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pre-seed securities row so the UPDATE has something to act on.
    out_db = tmp_path / "out.duckdb"
    samples = REPO_ROOT / "tests" / "fixtures" / "sample_yfinance"
    schema = REPO_ROOT / "fmf" / "data" / "schema.sql"

    conn = duckdb.connect(str(out_db))
    conn.execute(schema.read_text())
    conn.execute(
        'INSERT INTO "securities" (security_id, symbol, cik) VALUES (?, ?, ?)',
        ["00000000-0000-0000-0000-000000000001", "AAPL", "0000320193"],
    )
    conn.close()

    tickers_file = tmp_path / "tickers.txt"
    tickers_file.write_text("AAPL,0000320193\n")

    env = dict(os.environ)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ingest_yfinance",
            "--base-dir",
            str(samples),
            "--ticker-file",
            str(tickers_file),
            "--db",
            str(out_db),
            "--start",
            "2019-06-01",
            "--end",
            "2019-06-30",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    conn = duckdb.connect(str(out_db), read_only=True)
    try:
        n_prices = conn.execute('SELECT COUNT(*) FROM "prices"').fetchone()
        assert n_prices is not None and n_prices[0] > 0
    finally:
        conn.close()
