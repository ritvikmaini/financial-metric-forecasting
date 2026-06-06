"""Cohort coverage scan.

For each (security_id, fiscal_year, period), what fraction of the
table's data columns are non-null? Exposed as a DataFrame keyed by
symbol so a developer can quickly spot tickers with thin filings.

This is the audit that surfaces filers with partial XBRL emissions
(e.g., JPM emits Revenues only as an FY fact; the bank's 10-Qs tag
interest-income components separately, so quarterly Revenues is null).
"""

from __future__ import annotations

import duckdb
import pandas as pd

from fmf.data.sql_safety import validate_table_name

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

    coverage_pct is the fraction of non-null data columns per row.

    Phantom-aware: per (security_id, fiscal_year, period), pick one
    canonical row by latest end_date (ties broken by latest accepted_date)
    before counting non-null columns. Without this dedup, L-INFRA-013
    phantom Q-rows — comparative facts inheriting the filing's fp frame
    and landing in an earlier-period bucket with null revenue — inflate
    the denominator and depress coverage. The latest-end heuristic
    selects the genuine quarter (its end_date equals the quarter-end;
    phantoms carry earlier ends from the comparative quarters). Latest
    accepted_date is the secondary key to surface the most recent
    restatement when multiple genuine rows tie on end_date.
    """
    validate_table_name(table)
    cols = _PIT_COLUMNS_BY_TABLE.get(table)
    if cols is None:
        raise ValueError(f"no coverage spec for table {table!r}")

    not_null_sum = " + ".join(f'CASE WHEN "{c}" IS NULL THEN 0 ELSE 1 END' for c in cols)
    query = (
        f"WITH deduped AS ( "
        f'  SELECT * FROM "{table}" '
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
