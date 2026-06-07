"""Per-row PIT-correct target + naive baseline lookups for the S10 backtester."""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass

import duckdb

_ALLOWED_METRICS = frozenset({"eps_diluted", "ebitda", "ebit"})


def _check_metric(metric: str) -> None:
    if metric not in _ALLOWED_METRICS:
        raise ValueError(f"metric must be one of {sorted(_ALLOWED_METRICS)}; got {metric!r}")


@dataclass(frozen=True, slots=True)
class TargetRecord:
    fiscal_year: int
    accepted_date: dt.date
    value: float


def next_fy_target(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
    metric: str,
) -> TargetRecord | None:
    """Smallest fiscal_year whose earliest non-null period='FY' disclosure
    is strictly after as_of_date. Returns the value at that earliest
    disclosure (the original 10-K), with target_accepted_date set to that
    MIN(accepted_date).

    Avoids the comparative-row trap: every 10-K carries prior-year FY rows
    as comparatives. Ordering raw rows by accepted_date ASC and taking the
    first one > as_of can return a fiscal_year whose ORIGINAL disclosure
    was BEFORE as_of (the model already knew it) but which reappears in a
    later 10-K's comparatives. See L-EVAL-S10-002 for the close-read.
    """
    _check_metric(metric)
    sql = (
        "WITH fy_first_visible AS ("
        "  SELECT fiscal_year, MIN(accepted_date) AS first_accepted "
        '  FROM "income_statement" '
        "  WHERE security_id = ? AND period = 'FY' "
        f'    AND "{metric}" IS NOT NULL '
        "  GROUP BY fiscal_year"
        ") "
        f'SELECT fy.fiscal_year, fy.first_accepted, ist."{metric}" '
        "FROM fy_first_visible fy "
        'JOIN "income_statement" ist '
        "  ON ist.security_id = ? "
        "  AND ist.fiscal_year = fy.fiscal_year "
        "  AND ist.period = 'FY' "
        "  AND ist.accepted_date = fy.first_accepted "
        f'  AND ist."{metric}" IS NOT NULL '
        "WHERE fy.first_accepted > ? "
        "ORDER BY fy.fiscal_year ASC, ist.end_date DESC "
        "LIMIT 1"
    )
    row = conn.execute(sql, [str(security_id), str(security_id), as_of_date]).fetchone()
    if row is None:
        return None
    fy, accepted, value = row
    if isinstance(accepted, dt.datetime):
        accepted = accepted.date()
    return TargetRecord(fiscal_year=int(fy), accepted_date=accepted, value=float(value))


def last_fy_actual(
    *,
    conn: duckdb.DuckDBPyConnection,
    security_id: uuid.UUID,
    as_of_date: dt.date,
    metric: str,
) -> float | None:
    """Naive baseline = the most recent FY actual visible at as_of_date,
    via PIT (accepted_date <= as_of_date). None if no FY has been disclosed
    yet.

    Orders by end_date DESC then accepted_date DESC: the comparative trap
    does not apply here because we want the genuine latest fiscal year,
    however many times it has been restated.
    """
    _check_metric(metric)
    row = conn.execute(
        f'SELECT "{metric}" '
        'FROM "income_statement" '
        "WHERE security_id = ? AND period = 'FY' AND accepted_date <= ? "
        f'AND "{metric}" IS NOT NULL '
        "ORDER BY end_date DESC, accepted_date DESC "
        "LIMIT 1",
        [str(security_id), as_of_date],
    ).fetchone()
    return float(row[0]) if row is not None else None
