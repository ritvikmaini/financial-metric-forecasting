"""T4 coverage audit.

Iterate BUILTIN_REGISTRY. For each feature, compute the value for every
(security, fiscal_year) sample where as_of = MAX(accepted_date) over the
(security, fiscal_year, FY-period) filings. Report per-feature
coverage_pct and writes reports/coverage_audit.json. Exits 1 if any
non-experimental feature is below its declared min_coverage_pct.

The audit is the measurement substrate for collaboratively finalizing
the 25 provisional derived thresholds (set to 0.50 in builtin_features).
"""

from __future__ import annotations

import json
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path

import duckdb

from fmf.features.builtin_features import BUILTIN_REGISTRY
from fmf.features.feature_registry import Feature

REPO_ROOT = Path(__file__).parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mini.duckdb"
REPORT_PATH = REPO_ROOT / "reports" / "coverage_audit.json"


def _samples(conn: duckdb.DuckDBPyConnection) -> list[tuple[uuid.UUID, int, object]]:
    """Per (security, FY) the latest accepted_date over its FY-period rows.

    Use income_statement as the canonical FY filing axis (it has the
    most reliable FY period coverage). For each (security_id,
    fiscal_year), as_of = MAX(accepted_date) where period = 'FY'. This
    gives the audit a hindsight-friendly as_of: the latest restatement
    of that fiscal year's annual filing is fully visible, so the audit
    measures best-case feature coverage rather than the harder T+1
    early-disclosure case.
    """
    rows = conn.execute(
        """
        SELECT security_id, fiscal_year, MAX(accepted_date) AS as_of
        FROM "income_statement"
        WHERE period = 'FY'
        GROUP BY security_id, fiscal_year
        ORDER BY security_id, fiscal_year
        """
    ).fetchall()
    return [(uuid.UUID(str(sid)), int(fy), as_of) for sid, fy, as_of in rows]


def _measure_feature(
    conn: duckdb.DuckDBPyConnection,
    feature: Feature,
    samples: Iterable[tuple[uuid.UUID, int, object]],
) -> tuple[int, int]:
    """Return (filled, total) for a feature across the sample set.

    `filled` counts samples where compute returned a non-None value.
    Raises on first exception from a compute fn — the audit treats that
    as a hard signal that a feature is broken and must not silently pass.
    """
    filled = 0
    total = 0
    for sid, _fy, as_of in samples:
        total += 1
        try:
            v = feature.compute(conn=conn, security_id=sid, as_of_date=as_of)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"feature {feature.name!r} raised at security={sid}, as_of={as_of}: {exc!r}"
            ) from exc
        if v is not None:
            filled += 1
    return filled, total


def run_audit() -> int:
    if not FIXTURE.exists():
        print(f"fixture missing: {FIXTURE}", file=sys.stderr)
        return 2

    conn = duckdb.connect(str(FIXTURE), read_only=True)
    samples = _samples(conn)
    total_samples = len(samples)
    if total_samples == 0:
        print("no FY samples found in income_statement", file=sys.stderr)
        return 2

    per_feature: dict[str, dict[str, object]] = {}
    warnings: list[str] = []
    for f in BUILTIN_REGISTRY:
        filled, total = _measure_feature(conn, f, samples)
        coverage_pct = filled / total if total > 0 else 0.0
        per_feature[f.name] = {
            "coverage_pct": round(coverage_pct, 4),
            "min_required": f.min_coverage_pct,
            "experimental": f.experimental,
            "filled": filled,
            "total_samples": total,
        }
        if not f.experimental and coverage_pct < f.min_coverage_pct:
            warnings.append(
                f"{f.name}: measured {coverage_pct:.3f} < required {f.min_coverage_pct:.3f}"
            )

    report: dict[str, object] = {
        "total_samples": total_samples,
        "feature_count": len(BUILTIN_REGISTRY),
        "warnings": warnings,
        "features": per_feature,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(f"Audit ran across {total_samples} (security, FY) samples.")
    print(f"Wrote report: {REPORT_PATH.relative_to(REPO_ROOT)}")
    if warnings:
        print(f"\n{len(warnings)} non-experimental features below threshold:")
        for w in warnings:
            print(f"  - {w}")
        return 1
    print("\nAll non-experimental features meet their declared thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_audit())
