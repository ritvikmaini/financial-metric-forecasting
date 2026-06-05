"""Rate-limited HTTP client for SEC EDGAR.

Wraps httpx with:
- Configurable base_url (default https://data.sec.gov; tests use file://).
- User-Agent header read from SEC_USER_AGENT env (SEC mandates this).
- Token-bucket rate limiter (default 10 req/sec).
- Retries on 5xx and 429 (basic exponential backoff).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx


class MissingUserAgentError(RuntimeError):
    """Raised when SEC_USER_AGENT is neither passed nor set in env."""


class EdgarClient:
    """Rate-limited HTTP client.

    Use:
        c = EdgarClient(base_url="https://data.sec.gov")
        data = c.get_json("/submissions/CIK0000320193.json")
    """

    def __init__(
        self,
        *,
        base_url: str,
        user_agent: str | None = None,
        max_rps: float = 10.0,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        ua = user_agent if user_agent is not None else os.environ.get("SEC_USER_AGENT")
        if not ua:
            raise MissingUserAgentError(
                "SEC_USER_AGENT must be set in the environment (see .env.example) "
                "or passed explicitly. SEC requires a User-Agent identifying the caller."
            )
        self.base_url = base_url.rstrip("/")
        self.headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
        self.max_rps = max_rps
        self.timeout = timeout
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self._last_call_at: float = 0.0

    def _wait_for_slot(self) -> None:
        min_interval = 1.0 / self.max_rps
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_at
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_call_at = time.monotonic()

    def get_json(self, path: str) -> dict[str, Any]:
        """Fetch JSON at base_url + path. Honors rate limit."""
        self._wait_for_slot()
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        parsed = urlparse(url)

        if parsed.scheme == "file":
            local = Path(parsed.path)
            with local.open("r", encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
                return data

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
                    resp = client.get(url)
                    if resp.status_code == 429 or 500 <= resp.status_code < 600:
                        last_exc = httpx.HTTPStatusError(
                            f"status {resp.status_code}",
                            request=resp.request,
                            response=resp,
                        )
                        time.sleep(2**attempt)
                        continue
                    resp.raise_for_status()
                    payload: dict[str, Any] = resp.json()
                    return payload
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exc = e
                time.sleep(2**attempt)
        assert last_exc is not None
        raise last_exc
