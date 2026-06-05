"""DuckDB connector primitives.

Three callables:
- get_connection(path): open a DuckDB connection.
- fetch_latest_as_of(...): point-in-time primitive. Runs the parameterised
  DISTINCT ON pattern with accepted_date <= as_of_date for fundamentals tables.
- bulk_load(...): bulk-insert a pandas DataFrame BY NAME via DuckDB's
  registered-view mechanism. BY NAME matches by column name (not position),
  fills absent columns with NULL, and errors on unknown columns. Returns the
  number of rows inserted.

The PIT invariant (FMF-004 rule): fetching as-of (X - 1 day) MUST exclude a
filing whose accepted_date is X.
"""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from fmf.data.sql_safety import validate_identifier, validate_table_name

# Tables that carry accepted_date; only these are valid PIT-primitive inputs.
_PIT_TABLES: frozenset[str] = frozenset(
    {
        "income_statement",
        "balance_sheet",
        "cashflow",
    }
)


def get_connection(path: str | Path = ":memory:") -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection at the given path."""
    return duckdb.connect(str(path))


def fetch_latest_as_of(
    *,
    conn: duckdb.DuckDBPyConnection,
    table: str,
    security_id: uuid.UUID,
    as_of_date: dt.date,
    columns: list[str],
) -> dict[str, Any] | None:
    """Return the latest row for security_id whose accepted_date <= as_of_date.

    Identifiers (table, each column) are validated against the whitelist
    before interpolation. Returns a dict keyed by column name, or None if
    no row qualifies.
    """
    validate_table_name(table)
    if table not in _PIT_TABLES:
        raise ValueError(
            f"fetch_latest_as_of requires a table with an accepted_date column; "
            f"{table!r} does not. Allowed: {sorted(_PIT_TABLES)}."
        )
    if not columns:
        raise ValueError("columns must be a non-empty list")
    for col in columns:
        validate_identifier(col)

    col_list = ", ".join(columns)
    query = (
        f"SELECT DISTINCT ON (security_id) {col_list}\n"
        f"  FROM {table}\n"
        f" WHERE security_id = ? AND accepted_date <= ?\n"
        f" ORDER BY security_id, accepted_date DESC, filing_date DESC"
    )
    row = conn.execute(query, [str(security_id), as_of_date]).fetchone()
    if row is None:
        return None
    return dict(zip(columns, row, strict=True))


def bulk_load(
    *,
    conn: duckdb.DuckDBPyConnection,
    table: str,
    df: pd.DataFrame,
) -> int:
    """Bulk-insert df into table BY NAME. Returns the row count inserted."""
    validate_table_name(table)
    if df.empty:
        return 0
    conn.register("_bulk_load_df", df)
    try:
        # BY NAME matches by column name (not position), fills absent columns
        # with NULL, and errors on unknown columns. Fundamentals ingest carries
        # partial field sets so positional matching would silently misalign
        # (e.g. revenue landing in gross_profit).
        conn.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _bulk_load_df")
    finally:
        conn.unregister("_bulk_load_df")
    return int(len(df))
