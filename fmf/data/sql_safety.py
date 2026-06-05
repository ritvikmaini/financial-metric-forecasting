"""Identifier whitelist for raw-SQL interpolation.

PostgreSQL (and DuckDB) accept value parameters via %s / ? placeholders
but cannot parameterise identifiers (table names, column names). Code
that interpolates an identifier into raw SQL must route it through
validate_identifier first.

ALLOWED_TABLES is the single source of truth that must match
schema.sql; tests/data/test_sql_safety.py::test_allowed_tables_matches_schema
enforces parity.
"""

from __future__ import annotations

import re
from typing import Final

# PostgreSQL identifier length cap is 63 bytes; we follow the same rule.
_IDENT_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")


ALLOWED_TABLES: Final[frozenset[str]] = frozenset(
    {
        "securities",
        "income_statement",
        "balance_sheet",
        "cashflow",
        "analyst_estimates",
        "prices",
    }
)


class InvalidIdentifierError(ValueError):
    """Raised when an identifier fails the whitelist check."""


def validate_identifier(name: str) -> str:
    """Return name if it is a safe SQL identifier; else raise.

    A safe identifier starts with a letter, contains only ASCII letters,
    digits, and underscores, and is at most 63 characters long.
    """
    if not isinstance(name, str) or not _IDENT_RE.fullmatch(name):
        raise InvalidIdentifierError(
            f"unsafe SQL identifier: {name!r}. Identifiers must match [A-Za-z][A-Za-z0-9_]{{0,62}}."
        )
    return name


def validate_table_name(name: str) -> str:
    """Return name if it is a known table; else raise."""
    validate_identifier(name)
    if name not in ALLOWED_TABLES:
        raise InvalidIdentifierError(
            f"unknown table {name!r}. Known tables: {sorted(ALLOWED_TABLES)}."
        )
    return name
