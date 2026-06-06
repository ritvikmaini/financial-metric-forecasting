"""Feature registry.

Declarative records of every feature, each with a compute() function
and metadata (required, experimental, expected min coverage).

The registry powers two downstream artifacts:
- The feature matrix builder (S5) iterates the registry to produce
  one column per feature for every (security_id, as_of_date).
- The coverage audit (T6) runs each feature against the fixture and
  flags features below min_coverage_pct.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import duckdb

# A compute function: (conn, security_id, as_of_date) -> float | None.
ComputeFn = Callable[..., "float | None"]


@dataclass(frozen=True, slots=True)
class Feature:
    name: str
    description: str
    source_tables: tuple[str, ...]
    compute: ComputeFn
    required: bool
    experimental: bool
    min_coverage_pct: float  # 0.0-1.0; coverage audit warns if below


@dataclass
class FeatureRegistry:
    _features: dict[str, Feature] = field(default_factory=dict)

    def register(self, feature: Feature) -> None:
        if feature.name in self._features:
            raise ValueError(f"feature {feature.name!r} already registered")
        self._features[feature.name] = feature

    def __contains__(self, name: object) -> bool:
        return name in self._features

    def __getitem__(self, name: str) -> Feature:
        return self._features[name]

    def __iter__(self) -> Iterator[Feature]:
        return iter(self._features.values())

    def names(self) -> list[str]:
        return list(self._features.keys())

    def __len__(self) -> int:
        return len(self._features)


def validate_registry(reg: FeatureRegistry) -> None:
    """Raise ValueError if any feature is malformed."""
    for f in reg:
        if not f.description.strip():
            raise ValueError(f"feature {f.name!r} missing description")
        if not f.source_tables:
            raise ValueError(f"feature {f.name!r} has no source_tables")
        if not 0.0 <= f.min_coverage_pct <= 1.0:
            raise ValueError(
                f"feature {f.name!r} min_coverage_pct out of range: {f.min_coverage_pct}"
            )


def compute_feature_matrix(
    *,
    conn: duckdb.DuckDBPyConnection,
    reg: FeatureRegistry,
    security_id: uuid.UUID,
    as_of_date: dt.date,
) -> dict[str, float | None]:
    """Compute every registered feature for one (security, as_of_date).
    Returns dict[feature_name -> value or None].
    """
    return {
        f.name: f.compute(conn=conn, security_id=security_id, as_of_date=as_of_date) for f in reg
    }
