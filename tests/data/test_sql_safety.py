"""Identifier-whitelist tests.

PostgreSQL and DuckDB cannot parameterise table or column names. Any
function that interpolates an identifier into raw SQL must run it
through validate_identifier first.
"""

from __future__ import annotations

import pytest

from fmf.data.sql_safety import (
    ALLOWED_TABLES,
    InvalidIdentifierError,
    validate_identifier,
    validate_table_name,
)


@pytest.mark.parametrize(
    "ident",
    [
        "revenue",
        "ebit_ttm",
        "security_id",
        "accepted_date",
        "Q1_revenue",
        "fy2024",
        "a",
        "x" * 63,
    ],
)
def test_validate_identifier_accepts_safe_names(ident: str) -> None:
    validate_identifier(ident)


@pytest.mark.parametrize(
    "ident,reason",
    [
        ("", "empty"),
        ("1revenue", "starts with digit"),
        ("revenue; DROP TABLE securities", "semicolon injection"),
        ("revenue--comment", "SQL comment marker"),
        ("revenue/*comment*/", "block comment"),
        ("revenue with space", "whitespace"),
        ("revenue'", "single quote"),
        ('revenue"', "double quote"),
        ("revenue`", "backtick"),
        ("revenue\\", "backslash"),
        ("revenue\n", "newline"),
        ("revenue\t", "tab"),
        ("revenue-1", "hyphen"),
        ("revenue.subfield", "dot"),
        ("rev$enue", "dollar sign"),
        ("café", "non-ASCII"),
        ("x" * 64, "too long (>63 chars)"),
    ],
)
def test_validate_identifier_rejects_unsafe_names(ident: str, reason: str) -> None:
    with pytest.raises(InvalidIdentifierError):
        validate_identifier(ident)


def test_validate_table_name_accepts_all_schema_tables() -> None:
    for tbl in ALLOWED_TABLES:
        validate_table_name(tbl)


def test_validate_table_name_rejects_unknown_table() -> None:
    with pytest.raises(InvalidIdentifierError):
        validate_table_name("users")


def test_validate_table_name_rejects_injection_in_table_position() -> None:
    with pytest.raises(InvalidIdentifierError):
        validate_table_name("securities; DROP TABLE prices")


def test_allowed_tables_matches_schema() -> None:
    """ALLOWED_TABLES must list exactly the tables in schema.sql."""
    expected = {
        "securities",
        "income_statement",
        "balance_sheet",
        "cashflow",
        "analyst_estimates",
        "prices",
    }
    assert expected == ALLOWED_TABLES, "ALLOWED_TABLES drifted from schema.sql"
