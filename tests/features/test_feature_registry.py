"""Feature registry shape tests."""

from __future__ import annotations

from fmf.features.feature_registry import (
    Feature,
    FeatureRegistry,
    validate_registry,
)


def test_feature_registry_construction() -> None:
    reg = FeatureRegistry()
    reg.register(
        Feature(
            name="test_feature",
            description="Test",
            source_tables=("income_statement",),
            compute=lambda **kwargs: 1.0,
            required=False,
            experimental=False,
            min_coverage_pct=0.5,
        )
    )
    assert "test_feature" in reg
    assert reg["test_feature"].name == "test_feature"


def test_feature_registry_rejects_duplicate_names() -> None:
    import pytest

    reg = FeatureRegistry()
    f = Feature(
        name="dup",
        description="",
        source_tables=(),
        compute=lambda **kwargs: None,
        required=False,
        experimental=False,
        min_coverage_pct=0.0,
    )
    reg.register(f)
    with pytest.raises(ValueError, match="already registered"):
        reg.register(f)


def test_validate_registry_passes_on_well_formed_registry() -> None:
    reg = FeatureRegistry()
    reg.register(
        Feature(
            name="ok",
            description="ok feature",
            source_tables=("income_statement",),
            compute=lambda **kwargs: 1.0,
            required=False,
            experimental=False,
            min_coverage_pct=0.5,
        )
    )
    validate_registry(reg)  # must not raise


def test_validate_registry_rejects_missing_description() -> None:
    import pytest

    reg = FeatureRegistry()
    reg.register(
        Feature(
            name="bad",
            description="",
            source_tables=(),
            compute=lambda **kwargs: None,
            required=False,
            experimental=False,
            min_coverage_pct=0.0,
        )
    )
    with pytest.raises(ValueError, match="description"):
        validate_registry(reg)
