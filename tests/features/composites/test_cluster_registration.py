"""S18 cluster registry binding tests."""

from __future__ import annotations

from fmf.features.builtin_features import BUILTIN_REGISTRY
from fmf.features.composites import EARNINGS_QUALITY_CLUSTER


def test_all_five_cluster_features_registered() -> None:
    for name in EARNINGS_QUALITY_CLUSTER:
        assert name in BUILTIN_REGISTRY, f"missing cluster feature in registry: {name!r}"


def test_cluster_features_marked_experimental() -> None:
    for name in EARNINGS_QUALITY_CLUSTER:
        feature = BUILTIN_REGISTRY[name]
        assert feature.experimental is True, (
            f"{name!r} must be experimental=True (S18 single-flag invariant)"
        )


def test_cluster_tuple_is_documented_shape() -> None:
    assert EARNINGS_QUALITY_CLUSTER == (
        "piotroski_f_score",
        "ccc_days",
        "dechow_accruals",
        "beneish_m_score",
        "mohanram_g_score",
    )
