"""Shared pytest fixtures.

Real fixtures (fixture_db, mocked_tirex, anchor_tickers) land in later
sessions:
- fixture_db: S2 (`tests/fixtures/mini.duckdb` deliverable)
- mocked_tirex: S7 (`tests/fixtures/tirex_outputs/` deliverable)
- anchor_tickers: S2 (`tests/fixtures/known_financials.json` deliverable)

This file holds shared helpers that don't depend on those fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def repo_root() -> Path:
    """Absolute path to the repo root."""
    return Path(__file__).parent.parent
