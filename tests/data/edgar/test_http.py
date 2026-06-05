"""HTTP client tests.

The client must:
- Send a User-Agent header (SEC mandates this).
- Accept a configurable base URL (so tests use file:// or http://localhost
  instead of live data.sec.gov).
- Enforce a token-bucket rate limit (default 10 req/sec).
- Read SEC_USER_AGENT from env.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from fmf.data.edgar._http import (
    EdgarClient,
    MissingUserAgentError,
)


def test_client_reads_user_agent_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "Test User test@example.com")
    c = EdgarClient(base_url="https://data.sec.gov")
    assert "Test User test@example.com" in c.headers["User-Agent"]


def test_client_raises_when_user_agent_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(MissingUserAgentError):
        EdgarClient(base_url="https://data.sec.gov")


def test_client_explicit_user_agent_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "env-value")
    c = EdgarClient(base_url="https://data.sec.gov", user_agent="explicit-value")
    assert "explicit-value" in c.headers["User-Agent"]


def test_client_get_json_from_file_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A file:// base URL reads from disk. Used by CI so we never hit live SEC."""
    monkeypatch.setenv("SEC_USER_AGENT", "Test Test test@example.com")
    sample = tmp_path / "submissions" / "CIK0000320193.json"
    sample.parent.mkdir(parents=True)
    sample.write_text('{"cik": "0000320193", "entityName": "Apple Inc."}')

    c = EdgarClient(base_url=f"file://{tmp_path}")
    data = c.get_json("/submissions/CIK0000320193.json")
    assert data["entityName"] == "Apple Inc."


def test_rate_limiter_enforces_min_interval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """At max_rps=10, two sequential calls take at least ~0.1s."""
    monkeypatch.setenv("SEC_USER_AGENT", "Test Test test@example.com")
    sample = tmp_path / "f.json"
    sample.write_text('{"x": 1}')

    c = EdgarClient(base_url=f"file://{tmp_path}", max_rps=10)
    start = time.monotonic()
    c.get_json("/f.json")
    c.get_json("/f.json")
    elapsed = time.monotonic() - start
    assert elapsed >= 0.08, f"rate limiter did not enforce min interval (elapsed={elapsed:.3f}s)"
