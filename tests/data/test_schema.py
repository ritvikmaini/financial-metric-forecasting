"""Schema smoke tests.

Loads `fmf/data/schema.sql` into an in-memory DuckDB and verifies the
six tables exist with the expected columns and primary keys.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

SCHEMA_PATH = Path(__file__).parent.parent.parent / "fmf" / "data" / "schema.sql"


EXPECTED_TABLES: dict[str, set[str]] = {
    "securities": {
        "security_id",
        "symbol",
        "cik",
        "sector",
        "industry",
        "country",
        "exchange",
    },
    "income_statement": {
        "security_id",
        "fiscal_year",
        "period",
        "filing_date",
        "accepted_date",
        "revenue",
        "gross_profit",
        "ebitda",
        "ebit",
        "net_income",
        "eps_diluted",
    },
    "balance_sheet": {
        "security_id",
        "fiscal_year",
        "period",
        "filing_date",
        "accepted_date",
        "total_assets",
        "total_liabilities",
        "total_equity",
        "cash_and_equivalents",
        "current_assets",
        "current_liabilities",
        "long_term_debt",
    },
    "cashflow": {
        "security_id",
        "fiscal_year",
        "period",
        "filing_date",
        "accepted_date",
        "operating_cash_flow",
        "investing_cash_flow",
        "financing_cash_flow",
        "capital_expenditure",
        "free_cash_flow",
    },
    "analyst_estimates": {
        "security_id",
        "target_date",
        "pulled_at",
        "metric",
        "consensus",
        "n_analysts",
    },
    "prices": {
        "security_id",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    },
}


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with schema loaded."""
    c = duckdb.connect(":memory:")
    c.execute(SCHEMA_PATH.read_text())
    return c


def test_schema_file_exists() -> None:
    assert SCHEMA_PATH.exists(), f"schema.sql not found at {SCHEMA_PATH}"


def test_all_expected_tables_present(conn: duckdb.DuckDBPyConnection) -> None:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    table_names = {r[0] for r in rows}
    missing = set(EXPECTED_TABLES.keys()) - table_names
    assert not missing, f"missing tables: {sorted(missing)}"


@pytest.mark.parametrize("table_name,expected_cols", list(EXPECTED_TABLES.items()))
def test_table_has_expected_columns(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    expected_cols: set[str],
) -> None:
    rows = conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = ? AND table_schema = 'main'",
        [table_name],
    ).fetchall()
    actual = {r[0] for r in rows}
    missing = expected_cols - actual
    assert not missing, f"table {table_name} missing columns: {sorted(missing)}"


def test_income_statement_primary_key_includes_accepted_date(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    # PIT correctness relies on (security_id, fiscal_year, period, accepted_date)
    # being the primary key so multiple amendments coexist.
    rows = conn.execute(
        """
        SELECT constraint_column_names
        FROM duckdb_constraints()
        WHERE table_name = 'income_statement' AND constraint_type = 'PRIMARY KEY'
        """
    ).fetchall()
    assert rows, "income_statement has no PRIMARY KEY constraint"
    pk_cols = set(rows[0][0])
    assert "accepted_date" in pk_cols, (
        f"income_statement primary key MUST include accepted_date so "
        f"amended filings coexist; PIT correctness depends on this. "
        f"Found PK cols: {pk_cols}"
    )
