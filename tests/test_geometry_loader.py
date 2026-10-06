"""Tests for finding ORCA geometries, against the installed ORCA."""

from __future__ import annotations

import pytest

from cobra.configuration.configuration import ConfigurationError, GeometryConfig
from cobra.configuration.geometry_loader import (
    PRESETS_MODULE,
    discover_preset_geometries,
    resolve_geometry_class,
)

pytest.importorskip("orca")


def test_presets_are_discovered_from_orcas_exports():
    """Regression: ORCA 2 keeps its presets in subpackages, which a walk over the
    presets folder's own modules did not find.
    """
    presets = dict(discover_preset_geometries())

    assert {"InductorOcta", "TransformerOcta"} <= set(presets)


def test_a_preset_resolves_from_the_presets_module():
    config = GeometryConfig(source="preset", class_name="TransformerOcta", module=PRESETS_MODULE)

    assert resolve_geometry_class(config).__name__ == "TransformerOcta"


def test_a_preset_module_orca_no_longer_has_points_to_the_presets_module():
    config = GeometryConfig(
        source="preset", class_name="TransformerOcta", module="orca.geometry.presets.tf_octa_c_ports"
    )

    with pytest.raises(ConfigurationError, match=f"exports its presets from '{PRESETS_MODULE}'"):
        resolve_geometry_class(config)
