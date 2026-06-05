"""Smoke tests for the ingest CLIs.

These do not hit live SEC. They invoke the CLIs with a file:// base URL
pointed at tests/fixtures/sample_filings/ and verify the pipeline runs
end-to-end on the sample data.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.parent


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    monkeypatch.setenv("SEC_USER_AGENT", "Test Test test@example.com")
    return dict(os.environ)


def test_ingest_edgar_cli_runs_against_sample_fixture(
    tmp_path: Path,
    env: dict[str, str],
) -> None:
    """ingest_edgar.py --base-url file://<sample-dir> --tickers AAPL ..."""
    tickers_file = tmp_path / "tickers.txt"
    tickers_file.write_text("AAPL,0000320193\n")

    out_db = tmp_path / "out.duckdb"
    samples = REPO_ROOT / "tests" / "fixtures" / "sample_filings"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ingest_edgar",
            "--base-url",
            f"file://{samples}",
            "--ticker-file",
            str(tickers_file),
            "--out",
            str(out_db),
            "--max-rps",
            "1000",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert out_db.exists()

    conn = duckdb.connect(str(out_db), read_only=True)
    try:
        n = conn.execute('SELECT COUNT(*) FROM "income_statement"').fetchone()
        assert n is not None and n[0] > 0
    finally:
        conn.close()
