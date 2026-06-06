"""Per-field source audit.

For each ticker, which us-gaap concept resolved the requested field?
Re-runs the concept-map resolution against fresh facts and surfaces:
- top-priority concept matched? (good)
- fallback concept matched? (worth noting)
- nothing matched? (coverage gap)
"""

from __future__ import annotations

import pandas as pd

from fmf.data.edgar._http import EdgarClient
from fmf.data.edgar.companyfacts import load_facts
from fmf.data.edgar.concept_map import CONCEPT_MAP, resolve_field


def audit_field_sources(
    *,
    base_url: str,
    tickers: list[tuple[str, str]],
    field: str,
    fy: int,
    fp: str,
    max_rps: float = 8.0,
) -> pd.DataFrame:
    """Return per-ticker resolution status for a single field x fy x fp."""
    client = EdgarClient(base_url=base_url, max_rps=max_rps)
    priority = CONCEPT_MAP.get(field, [])
    if not priority:
        raise KeyError(f"field {field!r} has no concept map")

    out: list[dict[str, object]] = []
    for ticker, cik in tickers:
        facts = load_facts(client, cik=cik)
        end_candidates = sorted({f.end for f in facts if f.fy == fy and f.fp == fp})
        if not end_candidates:
            out.append(
                {
                    "ticker": ticker,
                    "field": field,
                    "resolved_concept": None,
                    "fallback_position": None,
                    "value": None,
                }
            )
            continue
        end = end_candidates[-1]
        resolved = resolve_field(facts, field=field, end=end, fp=fp)
        if resolved is None:
            out.append(
                {
                    "ticker": ticker,
                    "field": field,
                    "resolved_concept": None,
                    "fallback_position": None,
                    "value": None,
                }
            )
            continue
        try:
            position = priority.index(resolved.concept)
        except ValueError:
            position = -1
        out.append(
            {
                "ticker": ticker,
                "field": field,
                "resolved_concept": resolved.concept,
                "fallback_position": position,
                "value": resolved.value,
            }
        )
    return pd.DataFrame(out)
