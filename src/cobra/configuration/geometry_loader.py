from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cobra.configuration.configuration import ConfigurationError, GeometryConfig

if TYPE_CHECKING:
    from types import ModuleType

    from orca import BaseGeometry

#: The module that exports ORCA's preset geometries, independent of where ORCA keeps them.
PRESETS_MODULE = "orca.geometry.presets"


def _base_geometry_class() -> type[BaseGeometry]:
    # ORCA is optional: only EM fine-tuning needs it
    try:
        from orca import BaseGeometry
    except ImportError as exc:
        raise ConfigurationError(
            "ORCA must be installed to use EM fine-tuning geometries"
        ) from exc
    return BaseGeometry


def _load_custom_module(file_path: str) -> ModuleType:
    path = Path(file_path).resolve()
    if not path.is_file():
        raise ConfigurationError(f"Geometry file not found: {path}")
    module_name = f"cobra_custom_geometry_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ConfigurationError(f"Unable to load geometry module from {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ConfigurationError(f"Failed to import geometry file '{path}': {exc}") from exc
    return module


def _geometry_classes(module: ModuleType, base_geometry: type) -> list[tuple[str, type]]:
    return [
        (class_name, geometry_class)
        for class_name, geometry_class in inspect.getmembers(module, inspect.isclass)
        if geometry_class is not base_geometry
        and issubclass(geometry_class, base_geometry)
        and geometry_class.__module__ == module.__name__
    ]


def discover_preset_geometries() -> list[tuple[str, type]]:
    """Return the preset geometry classes ORCA exports from ``PRESETS_MODULE``, by class name."""
    base_geometry = _base_geometry_class()
    try:
        presets = importlib.import_module(PRESETS_MODULE)
    except ImportError as exc:
        raise ConfigurationError("Failed to discover ORCA preset geometries") from exc
    discovered = [
        (name, member)
        for name in getattr(presets, "__all__", [])
        if inspect.isclass(member := getattr(presets, name, None))
        and issubclass(member, base_geometry)
    ]
    return sorted(discovered, key=lambda item: item[0].lower())


def discover_custom_geometries(file_path: str) -> list[tuple[str, type]]:
    """Return geometry classes defined directly by a custom Python file."""
    base_geometry = _base_geometry_class()
    module = _load_custom_module(file_path)
    classes = _geometry_classes(module, base_geometry)
    if not classes:
        raise ConfigurationError(
            f"No classes extending BaseGeometry were found in '{Path(file_path).resolve()}'"
        )
    return sorted(classes, key=lambda item: item[0].lower())


def resolve_geometry_class(config: GeometryConfig) -> type:
    """Resolve and validate a configured ORCA geometry class without instantiating it."""
    config.validate()
    base_geometry = _base_geometry_class()
    try:
        if config.source == "preset":
            module = importlib.import_module(config.module or "")
        else:
            module = _load_custom_module(config.file or "")
        geometry_class = getattr(module, config.class_name)
    except (ImportError, AttributeError) as exc:
        location = config.module if config.source == "preset" else config.file
        hint = (
            f"; ORCA exports its presets from '{PRESETS_MODULE}'"
            if config.source == "preset" and config.module != PRESETS_MODULE
            else ""
        )
        raise ConfigurationError(
            f"Geometry class '{config.class_name}' was not found in '{location}'{hint}"
        ) from exc
    if not inspect.isclass(geometry_class) or not issubclass(geometry_class, base_geometry):
        raise ConfigurationError(
            f"Geometry class '{config.class_name}' must extend ORCA BaseGeometry"
        )
    return geometry_class


def create_geometry(config: GeometryConfig) -> Any:
    """Instantiate a configured ORCA geometry class."""
    try:
        return resolve_geometry_class(config)()
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(
            f"Failed to create geometry '{config.class_name}': {exc}"
        ) from exc
