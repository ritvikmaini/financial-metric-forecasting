"""Cohort coverage scan.

For each (security_id, fiscal_year, period), what fraction of the
table's data columns are non-null? Exposed as a DataFrame keyed by
symbol so a developer can quickly spot tickers with thin filings.

This is the audit that surfaces filers with partial XBRL emissions
(e.g., JPM emits Revenues only as an FY fact; the bank's 10-Qs tag
interest-income components separately, so quarterly Revenues is null).

L-INFRA-014: coverage now mirrors fetch_pit_series exactly via the
shared field-level assembly. A partial re-disclosure that omits a
field no longer falsely depresses the metric.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from fmf.data.sql_safety import validate_table_name
from fmf.features.point_in_time import _PIT_KEY_COLS

_PIT_COLUMNS_BY_TABLE: dict[str, tuple[str, ...]] = {
    "income_statement": (
        "revenue",
        "gross_profit",
        "ebitda",
        "ebit",
        "net_income",
        "eps_diluted",
    ),
    "balance_sheet": (
        "total_assets",
        "total_liabilities",
        "total_equity",
        "cash_and_equivalents",
        "current_assets",
        "current_liabilities",
        "long_term_debt",
    ),
    "cashflow": (
        "operating_cash_flow",
        "investing_cash_flow",
        "financing_cash_flow",
        "capital_expenditure",
        "free_cash_flow",
    ),
}


def compute_coverage(conn: duckdb.DuckDBPyConnection, *, table: str) -> pd.DataFrame:
    """Return per-(symbol, fiscal_year, period) coverage_pct.

    coverage_pct is the fraction of non-null data columns per row on
    the field-level PIT synthesized view of `table`.

    Mirrors fetch_pit_series via the same field-level assembly
    (L-INFRA-014): for each data column, the synthesized value is
    LAST_VALUE(IGNORE NULLS) over the (security, fy, period, end_date)
    group ordered by accepted_date. A later filing that drops a field
    no longer falsely nulls it in the coverage metric — coverage now
    equals what features deliver.

    Phantom-aware (L-INFRA-013): after the field-level synthesis, dedup
    to one row per (security, fiscal_year, period) by latest end_date
    so comparative-fp-frame phantom Q-rows (with null revenue and
    earlier end_dates) are excluded. Genuine quarter wins over earlier-
    end phantoms.
    """
    validate_table_name(table)
    cols = _PIT_COLUMNS_BY_TABLE.get(table)
    if cols is None:
        raise ValueError(f"no coverage spec for table {table!r}")

    # Build the all-securities field-level synthesized view inline.
    # Same field assembly as _field_level_pit_select_sql, but partitioned
    # per-security (no security_id WHERE filter, no as_of filter — coverage
    # is a backward-looking probe over the full fixture).
    all_cols = [r[0] for r in conn.execute(f'DESCRIBE "{table}"').fetchall()]
    data_cols = [c for c in all_cols if c not in _PIT_KEY_COLS]
    if not data_cols:
        raise ValueError(f"{table} has no data columns after excluding keys")

    last_value_exprs = ",\n            ".join(
        f'LAST_VALUE("{c}" IGNORE NULLS) OVER ('
        f"PARTITION BY security_id, fiscal_year, period, end_date "
        f"ORDER BY accepted_date "
        f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        f') AS "{c}"'
        for c in data_cols
    )
    synthesized_sql = f"""
        SELECT
            security_id, fiscal_year, period, filing_date, accepted_date, end_date,
            {last_value_exprs}
        FROM "{table}"
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY security_id, fiscal_year, period, end_date
            ORDER BY accepted_date DESC
        ) = 1
    """

    not_null_sum = " + ".join(f'CASE WHEN "{c}" IS NULL THEN 0 ELSE 1 END' for c in cols)
    query = (
        f"WITH synth AS ({synthesized_sql}), "
        f"deduped AS ( "
        f"  SELECT * FROM synth "
        f"  QUALIFY ROW_NUMBER() OVER ( "
        f"    PARTITION BY security_id, fiscal_year, period "
        f"    ORDER BY end_date DESC, accepted_date DESC "
        f"  ) = 1 "
        f") "
        f"SELECT s.symbol, t.fiscal_year, t.period, "
        f"       ({not_null_sum})::DOUBLE / {len(cols)} AS coverage_pct "
        f"  FROM deduped t "
        f'  JOIN "securities" s ON s.security_id = t.security_id '
        f"  ORDER BY s.symbol, t.fiscal_year, t.period"
    )
    return conn.execute(query).fetchdf()
