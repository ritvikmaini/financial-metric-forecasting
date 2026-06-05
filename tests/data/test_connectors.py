"""Connector unit tests.

Three primitives:
- get_connection(path): returns a DuckDB connection; path=":memory:" works.
- fetch_latest_as_of(conn, table, security_id, as_of_date, columns): runs
  the parameterised DISTINCT ON pattern. This is the project's PIT
  correctness primitive.
- bulk_load(conn, table, df): bulk-insert a pandas DataFrame BY NAME.

The PIT correctness test is critical: shifting as_of_date back by one
day before a filing's accepted_date MUST return the previous filing.
This is what FMF-004 will rely on in S4.
"""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from fmf.data.connectors import (
    bulk_load,
    fetch_latest_as_of,
    get_connection,
)
from fmf.data.sql_safety import InvalidIdentifierError

SCHEMA_PATH = Path(__file__).parent.parent.parent / "fmf" / "data" / "schema.sql"


@pytest.fixture
def conn() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    c.execute(SCHEMA_PATH.read_text())
    return c


@pytest.fixture
def aapl_id() -> uuid.UUID:
    return uuid.UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
def aapl_filings(conn: duckdb.DuckDBPyConnection, aapl_id: uuid.UUID) -> None:
    """Three AAPL income_statement rows: FY2022, FY2023, an amended FY2023."""
    rows = [
        (
            str(aapl_id),
            2022,
            "FY",
            dt.date(2022, 10, 28),
            dt.date(2022, 10, 28),
            394_328_000_000.0,
            170_782_000_000.0,
            130_541_000_000.0,
            119_437_000_000.0,
            99_803_000_000.0,
            6.11,
        ),
        (
            str(aapl_id),
            2023,
            "FY",
            dt.date(2023, 11, 3),
            dt.date(2023, 11, 3),
            383_285_000_000.0,
            169_148_000_000.0,
            125_820_000_000.0,
            114_301_000_000.0,
            96_995_000_000.0,
            6.13,
        ),
        (
            str(aapl_id),
            2023,
            "FY",
            dt.date(2023, 11, 3),
            dt.date(2024, 1, 15),
            383_285_000_000.0,
            169_148_000_000.0,
            125_820_000_000.0,
            114_301_000_000.0,
            96_995_000_000.0,
            6.14,
        ),
    ]
    conn.executemany(
        "INSERT INTO income_statement VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


# --- get_connection ---


def test_get_connection_in_memory_returns_usable_connection() -> None:
    c = get_connection(":memory:")
    assert c.execute("SELECT 42").fetchone() == (42,)


def test_get_connection_on_path_creates_file(tmp_path: Path) -> None:
    db = tmp_path / "test.duckdb"
    c = get_connection(str(db))
    c.execute("CREATE TABLE t (x INTEGER)")
    c.execute("INSERT INTO t VALUES (1)")
    c.close()
    assert db.exists()


# --- fetch_latest_as_of: the PIT primitive ---


def test_fetch_latest_as_of_returns_latest_by_accepted_date(
    conn: duckdb.DuckDBPyConnection,
    aapl_filings: None,
    aapl_id: uuid.UUID,
) -> None:
    row = fetch_latest_as_of(
        conn=conn,
        table="income_statement",
        security_id=aapl_id,
        as_of_date=dt.date(2024, 6, 1),
        columns=["fiscal_year", "period", "accepted_date", "eps_diluted"],
    )
    assert row is not None
    assert row["fiscal_year"] == 2023
    assert row["accepted_date"] == dt.date(2024, 1, 15)
    assert row["eps_diluted"] == pytest.approx(6.14)


def test_fetch_latest_as_of_one_day_shift_excludes_later_filing(
    conn: duckdb.DuckDBPyConnection,
    aapl_filings: None,
    aapl_id: uuid.UUID,
) -> None:
    """FMF-004 correctness rule: shift cutoff back by one day before a known
    filing's accepted_date and the output MUST change."""
    row = fetch_latest_as_of(
        conn=conn,
        table="income_statement",
        security_id=aapl_id,
        as_of_date=dt.date(2024, 1, 14),
        columns=["fiscal_year", "period", "accepted_date", "eps_diluted"],
    )
    assert row is not None
    assert row["accepted_date"] == dt.date(2023, 11, 3)
    assert row["eps_diluted"] == pytest.approx(6.13)


def test_fetch_latest_as_of_returns_none_when_no_data_visible(
    conn: duckdb.DuckDBPyConnection,
    aapl_filings: None,
    aapl_id: uuid.UUID,
) -> None:
    row = fetch_latest_as_of(
        conn=conn,
        table="income_statement",
        security_id=aapl_id,
        as_of_date=dt.date(2022, 1, 1),
        columns=["fiscal_year"],
    )
    assert row is None


def test_fetch_latest_as_of_rejects_unknown_table(
    conn: duckdb.DuckDBPyConnection,
    aapl_id: uuid.UUID,
) -> None:
    with pytest.raises(InvalidIdentifierError):
        fetch_latest_as_of(
            conn=conn,
            table="users",
            security_id=aapl_id,
            as_of_date=dt.date(2024, 1, 1),
            columns=["x"],
        )


def test_fetch_latest_as_of_rejects_injection_in_column(
    conn: duckdb.DuckDBPyConnection,
    aapl_filings: None,
    aapl_id: uuid.UUID,
) -> None:
    with pytest.raises(InvalidIdentifierError):
        fetch_latest_as_of(
            conn=conn,
            table="income_statement",
            security_id=aapl_id,
            as_of_date=dt.date(2024, 1, 1),
            columns=["eps_diluted; DROP TABLE securities"],
        )


def test_fetch_latest_as_of_securities_uses_no_accepted_date(
    conn: duckdb.DuckDBPyConnection,
    aapl_id: uuid.UUID,
) -> None:
    conn.execute(
        "INSERT INTO securities VALUES (?, 'AAPL', '320193', 'Tech', 'Hardware', 'US', 'NASDAQ')",
        [str(aapl_id)],
    )
    with pytest.raises(ValueError, match="accepted_date"):
        fetch_latest_as_of(
            conn=conn,
            table="securities",
            security_id=aapl_id,
            as_of_date=dt.date(2024, 1, 1),
            columns=["symbol"],
        )


# --- bulk_load ---


def test_bulk_load_inserts_rows(
    conn: duckdb.DuckDBPyConnection,
    aapl_id: uuid.UUID,
) -> None:
    df = pd.DataFrame(
        [
            {
                "security_id": str(aapl_id),
                "date": dt.date(2024, 1, 2),
                "open": 187.15,
                "high": 188.44,
                "low": 183.89,
                "close": 185.64,
                "adj_close": 185.30,
                "volume": 82_488_700,
            },
            {
                "security_id": str(aapl_id),
                "date": dt.date(2024, 1, 3),
                "open": 184.22,
                "high": 185.88,
                "low": 183.43,
                "close": 184.25,
                "adj_close": 183.91,
                "volume": 58_414_500,
            },
        ]
    )
    n = bulk_load(conn=conn, table="prices", df=df)
    assert n == 2

    count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()
    assert count == (2,)


def test_bulk_load_rejects_unknown_table(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    with pytest.raises(InvalidIdentifierError):
        bulk_load(conn=conn, table="users", df=pd.DataFrame())


def test_bulk_load_by_name_accepts_column_subset(
    conn: duckdb.DuckDBPyConnection,
    aapl_id: uuid.UUID,
) -> None:
    """BY NAME insertion: a DataFrame with only some target columns is
    inserted with NULL for the absent ones. Fundamentals ingest in S2 needs
    this because most rows carry partial field sets.
    """
    df = pd.DataFrame(
        [
            {
                "security_id": str(aapl_id),
                "fiscal_year": 2024,
                "period": "FY",
                "filing_date": dt.date(2024, 11, 1),
                "accepted_date": dt.date(2024, 11, 1),
                "revenue": 400_000_000_000.0,
            }
        ]
    )
    n = bulk_load(conn=conn, table="income_statement", df=df)
    assert n == 1

    row = conn.execute(
        "SELECT revenue, ebitda, net_income FROM income_statement WHERE fiscal_year = 2024"
    ).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(400_000_000_000.0)
    assert row[1] is None
    assert row[2] is None


class _ExecCapturingConn:
    """Proxy around a DuckDB connection that records the SQL passed to
    `execute`. Used by the double-quote-identifier tests because the
    DuckDB connection itself is a C-extension whose methods cannot be
    monkeypatched.
    """

    def __init__(
        self,
        inner: duckdb.DuckDBPyConnection,
        captured: dict[str, str],
        filter_pred=None,
    ) -> None:
        self._inner = inner
        self._captured = captured
        self._filter_pred = filter_pred

    def execute(self, query, *args, **kwargs):
        if self._filter_pred is None or self._filter_pred(query):
            self._captured["query"] = query
        return self._inner.execute(query, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_fetch_latest_as_of_emits_double_quoted_identifiers(
    conn: duckdb.DuckDBPyConnection,
    aapl_filings: None,
    aapl_id: uuid.UUID,
) -> None:
    """fetch_latest_as_of must wrap interpolated identifiers in double
    quotes. validate_identifier already gates against quote/injection
    characters, so this is safe for every identifier; it future-proofs
    reserved-word columns (date, open, close) when prices-derived
    features hit the connectors in S5.
    """
    captured: dict[str, str] = {}
    wrapped = _ExecCapturingConn(conn, captured)
    fetch_latest_as_of(
        conn=wrapped,
        table="income_statement",
        security_id=aapl_id,
        as_of_date=dt.date(2024, 6, 1),
        columns=["eps_diluted", "revenue"],
    )
    q = captured["query"]
    assert '"income_statement"' in q, f"table not double-quoted in: {q}"
    assert '"eps_diluted"' in q, f"column not double-quoted in: {q}"
    assert '"revenue"' in q, f"column not double-quoted in: {q}"


def test_bulk_load_emits_double_quoted_table_name(
    conn: duckdb.DuckDBPyConnection,
    aapl_id: uuid.UUID,
) -> None:
    captured: dict[str, str] = {}
    wrapped = _ExecCapturingConn(conn, captured, filter_pred=lambda q: "INSERT" in q)
    df = pd.DataFrame(
        [
            {
                "security_id": str(aapl_id),
                "date": dt.date(2024, 1, 2),
                "open": 187.15,
                "high": 188.44,
                "low": 183.89,
                "close": 185.64,
                "adj_close": 185.30,
                "volume": 82_488_700,
            }
        ]
    )
    bulk_load(conn=wrapped, table="prices", df=df)
    q = captured.get("query", "")
    assert '"prices"' in q, f"table not double-quoted in INSERT: {q}"
