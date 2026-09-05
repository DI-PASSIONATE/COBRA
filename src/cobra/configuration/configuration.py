from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, ClassVar

from cobra.optimizers.base_optimizer import OptimizationType


class ConfigurationError(ValueError):
    """Raised when a COBRA run configuration is invalid."""


def _require_keys(data: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ConfigurationError(f"Unknown {context} field(s): {names}")


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{context} must be an object")
    return dict(value)


def _number(value: Any, context: str, *, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ConfigurationError(f"{context} must be a finite number")


def _relative_path(path: str, destination: Path) -> str:
    absolute = Path(path).expanduser().resolve()
    try:
        return os.path.relpath(absolute, destination.resolve())
    except ValueError:
        return str(absolute)


def _resolved_path(path: str, base_directory: Path) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base_directory / candidate
    return str(candidate.resolve())


@dataclass(slots=True)
class BackendConfig:
    name: str
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: str) -> BackendConfig:
        data = _mapping(data, context)
        _require_keys(data, {"name", "settings"}, context)
        if not isinstance(data.get("name"), str) or not data["name"].strip():
            raise ConfigurationError(f"{context}.name must be a non-empty string")
        settings = data.get("settings", {})
        if not isinstance(settings, dict):
            raise ConfigurationError(f"{context}.settings must be an object")
        return cls(data["name"], settings)


@dataclass(slots=True)
class OptimizationParameterConfig:
    name: str
    type: str
    min_value: float
    max_value: float
    step: float | None = None
    unit: str | None = None
    linked_to: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationParameterConfig:
        data = _mapping(data, "optimization parameter")
        allowed = {"name", "type", "min_value", "max_value", "step", "unit", "linked_to"}
        _require_keys(data, allowed, "optimization parameter")
        try:
            return cls(**data)
        except TypeError as exc:
            raise ConfigurationError(f"Invalid optimization parameter: {exc}") from exc

    def validate(self) -> None:
        if not self.name:
            raise ConfigurationError("Optimization parameter name cannot be empty")
        try:
            OptimizationType(self.type)
        except ValueError as exc:
            raise ConfigurationError(
                f"Unsupported optimization type '{self.type}' for '{self.name}'"
            ) from exc
        if not isinstance(self.min_value, (int, float)) or isinstance(self.min_value, bool):
            raise ConfigurationError(f"Optimization parameter '{self.name}' min_value must be numeric")
        if not isinstance(self.max_value, (int, float)) or isinstance(self.max_value, bool):
            raise ConfigurationError(f"Optimization parameter '{self.name}' max_value must be numeric")
        _number(self.min_value, f"Optimization parameter '{self.name}' min_value")
        _number(self.max_value, f"Optimization parameter '{self.name}' max_value")
        if self.step is not None:
            _number(self.step, f"Optimization parameter '{self.name}' step")
        if self.min_value > self.max_value:
            raise ConfigurationError(
                f"Optimization parameter '{self.name}' has min_value greater than max_value"
            )
        if self.step is not None and self.step <= 0:
            raise ConfigurationError(f"Optimization parameter '{self.name}' step must be positive")


@dataclass(slots=True)
class DesignGoalConfig:
    parameter: str
    frequency_range: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    weight: float = 1.0
    kind: str = "catalogue"
    node: str | None = None
    port: str | None = None
    source_amplitude: float | None = None
    impedance: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DesignGoalConfig:
        data = _mapping(data, "design goal")
        allowed = {
            "parameter", "frequency_range", "min_value", "max_value", "weight",
            "kind", "node", "port", "source_amplitude", "impedance",
        }
        _require_keys(data, allowed, "design goal")
        try:
            return cls(**data)
        except TypeError as exc:
            raise ConfigurationError(f"Invalid design goal: {exc}") from exc

    def validate(self) -> None:
        if not self.parameter:
            raise ConfigurationError("Design goal parameter cannot be empty")
        if self.kind not in {"catalogue", "power_dbm", "gain_db"}:
            raise ConfigurationError(f"Unsupported design goal kind '{self.kind}'")
        if self.min_value is None and self.max_value is None:
            raise ConfigurationError(f"Design goal '{self.parameter}' needs a minimum or maximum")
        _number(self.weight, f"Design goal '{self.parameter}' weight")
        _number(self.min_value, f"Design goal '{self.parameter}' min_value", allow_none=True)
        _number(self.max_value, f"Design goal '{self.parameter}' max_value", allow_none=True)
        if self.weight <= 0:
            raise ConfigurationError(f"Design goal '{self.parameter}' weight must be positive")
        if self.kind in {"power_dbm", "gain_db"} and not self.node:
            raise ConfigurationError(f"Design goal '{self.parameter}' requires an output node")
        if self.kind == "gain_db":
            if not self.port or self.source_amplitude is None:
                raise ConfigurationError(
                    f"Gain goal '{self.parameter}' requires port and source_amplitude"
                )
            _number(
                self.source_amplitude,
                f"Gain goal '{self.parameter}' source_amplitude",
            )
            _number(self.impedance, f"Gain goal '{self.parameter}' impedance", allow_none=True)
            if self.impedance is None or self.impedance <= 0:
                raise ConfigurationError(f"Gain goal '{self.parameter}' requires positive impedance")


@dataclass(slots=True)
class GeometryConfig:
    source: str
    class_name: str
    module: str | None = None
    file: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeometryConfig:
        data = _mapping(data, "geometry")
        _require_keys(data, {"source", "class_name", "module", "file"}, "geometry")
        try:
            return cls(**data)
        except TypeError as exc:
            raise ConfigurationError(f"Invalid geometry: {exc}") from exc

    def validate(self) -> None:
        if self.source not in {"preset", "custom"}:
            raise ConfigurationError(f"Unsupported geometry source '{self.source}'")
        if not self.class_name:
            raise ConfigurationError("Geometry class_name cannot be empty")
        if self.source == "preset" and not self.module:
            raise ConfigurationError(f"Preset geometry '{self.class_name}' requires a module")
        if self.source == "custom" and not self.file:
            raise ConfigurationError(f"Custom geometry '{self.class_name}' requires a file")


@dataclass(slots=True)
class FineTuningConfig:
    enabled: bool = False
    palace_command: str = "palace"
    iterations: int = 3
    optimizer: str = "reuse"
    geometries: dict[str, GeometryConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FineTuningConfig:
        data = _mapping(data, "fine_tuning")
        allowed = {"enabled", "palace_command", "iterations", "optimizer", "geometries"}
        _require_keys(data, allowed, "fine_tuning")
        geometries_data = _mapping(data.get("geometries", {}), "fine_tuning.geometries")
        geometries = {
            name: GeometryConfig.from_dict(value) for name, value in geometries_data.items()
        }
        values = dict(data)
        values["geometries"] = geometries
        try:
            return cls(**values)
        except TypeError as exc:
            raise ConfigurationError(f"Invalid fine_tuning configuration: {exc}") from exc

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("fine_tuning.enabled must be a boolean")
        if not isinstance(self.palace_command, str) or not self.palace_command.strip():
            raise ConfigurationError("fine_tuning.palace_command must be a non-empty string")
        if not isinstance(self.iterations, int) or isinstance(self.iterations, bool):
            raise ConfigurationError("fine_tuning.iterations must be an integer")
        if self.iterations < 1:
            raise ConfigurationError("fine_tuning.iterations must be at least 1")
        if self.optimizer not in {"reuse", "gradient_descent"}:
            raise ConfigurationError(f"Unsupported fine-tuning optimizer '{self.optimizer}'")
        for geometry in self.geometries.values():
            geometry.validate()


@dataclass(slots=True)
class RunConfiguration:
    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1

    netlist: str
    component_models: dict[str, str] = field(default_factory=dict)
    simulation_parameters: dict[str, dict[str, str]] = field(default_factory=dict)
    optimizer: BackendConfig = field(default_factory=lambda: BackendConfig("OptunaOptimizer"))
    simulator: BackendConfig = field(default_factory=lambda: BackendConfig("XyceSimulator"))
    max_iterations: int = 500
    optimization_parameters: list[OptimizationParameterConfig] = field(default_factory=list)
    design_goals: list[DesignGoalConfig] = field(default_factory=list)
    fine_tuning: FineTuningConfig = field(default_factory=FineTuningConfig)
    schema_version: int = CURRENT_SCHEMA_VERSION

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], base_directory: str | Path = "."
    ) -> RunConfiguration:
        allowed = {
            "schema_version", "netlist", "component_models", "simulation_parameters",
            "optimizer", "simulator", "max_iterations", "optimization_parameters",
            "design_goals", "fine_tuning",
        }
        _require_keys(data, allowed, "configuration")
        version = data.get("schema_version")
        if version != cls.CURRENT_SCHEMA_VERSION:
            raise ConfigurationError(
                f"Unsupported schema_version {version!r}; expected {cls.CURRENT_SCHEMA_VERSION}"
            )
        if not isinstance(data.get("netlist"), str) or not data["netlist"].strip():
            raise ConfigurationError("netlist must be a non-empty path")

        base = Path(base_directory).resolve()
        component_models = _mapping(data.get("component_models", {}), "component_models")
        if not all(isinstance(name, str) and isinstance(path, str) for name, path in component_models.items()):
            raise ConfigurationError("component_models must map string names to paths")
        simulation_parameters = _mapping(
            data.get("simulation_parameters", {}), "simulation_parameters"
        )

        fine_tuning = FineTuningConfig.from_dict(data.get("fine_tuning", {}))
        for geometry in fine_tuning.geometries.values():
            if geometry.file:
                geometry.file = _resolved_path(geometry.file, base)

        config = cls(
            schema_version=version,
            netlist=_resolved_path(data["netlist"], base),
            component_models={
                name: _resolved_path(path, base) for name, path in component_models.items()
            },
            simulation_parameters=simulation_parameters,
            optimizer=BackendConfig.from_dict(data.get("optimizer", {"name": "OptunaOptimizer"}), "optimizer"),
            simulator=BackendConfig.from_dict(data.get("simulator", {"name": "XyceSimulator"}), "simulator"),
            max_iterations=data.get("max_iterations", 500),
            optimization_parameters=[
                OptimizationParameterConfig.from_dict(item)
                for item in data.get("optimization_parameters", [])
            ],
            design_goals=[DesignGoalConfig.from_dict(item) for item in data.get("design_goals", [])],
            fine_tuning=fine_tuning,
        )
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> RunConfiguration:
        config_path = Path(path).expanduser().resolve()
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                f"Invalid JSON in '{config_path}': {exc.msg} at line {exc.lineno}"
            ) from exc
        if not isinstance(data, dict):
            raise ConfigurationError("Configuration root must be a JSON object")
        return cls.from_dict(data, config_path.parent)

    def validate(self, check_paths: bool = True) -> None:
        if self.schema_version != self.CURRENT_SCHEMA_VERSION:
            raise ConfigurationError(
                f"Unsupported schema_version {self.schema_version}; expected {self.CURRENT_SCHEMA_VERSION}"
            )
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ConfigurationError("max_iterations must be a positive integer")
        if check_paths and not Path(self.netlist).is_file():
            raise ConfigurationError(f"Netlist file not found: {self.netlist}")
        for component, model in self.component_models.items():
            if not component:
                raise ConfigurationError("Component model names cannot be empty")
            if check_paths and not Path(model).is_file():
                raise ConfigurationError(f"Model file for '{component}' not found: {model}")
        for analysis, parameters in self.simulation_parameters.items():
            if not analysis or not isinstance(parameters, dict):
                raise ConfigurationError("simulation_parameters entries must be named objects")
            if not all(isinstance(name, str) and isinstance(value, str) for name, value in parameters.items()):
                raise ConfigurationError(f"Simulation parameters for '{analysis}' must be strings")
        names = {parameter.name for parameter in self.optimization_parameters}
        if len(names) != len(self.optimization_parameters):
            raise ConfigurationError("Optimization parameter names must be unique")
        for parameter in self.optimization_parameters:
            parameter.validate()
            if parameter.linked_to and parameter.linked_to not in names:
                raise ConfigurationError(
                    f"Optimization parameter '{parameter.name}' links to unknown parameter "
                    f"'{parameter.linked_to}'"
                )
            if parameter.linked_to == parameter.name:
                raise ConfigurationError(f"Optimization parameter '{parameter.name}' cannot link to itself")
        for parameter in self.optimization_parameters:
            seen: set[str] = set()
            current = parameter.name
            while current in names:
                if current in seen:
                    raise ConfigurationError(
                        f"Optimization parameter links contain a cycle involving '{current}'"
                    )
                seen.add(current)
                target = next(item for item in self.optimization_parameters if item.name == current).linked_to
                if not target:
                    break
                current = target
        for goal in self.design_goals:
            goal.validate()
        self.fine_tuning.validate()
        if check_paths:
            for geometry in self.fine_tuning.geometries.values():
                if geometry.file and not Path(geometry.file).is_file():
                    raise ConfigurationError(f"Geometry file not found: {geometry.file}")

    def to_dict(self, destination_directory: str | Path | None = None) -> dict[str, Any]:
        self.validate()
        data = asdict(self)
        data.pop("CURRENT_SCHEMA_VERSION", None)
        if destination_directory is not None:
            destination = Path(destination_directory)
            data["netlist"] = _relative_path(self.netlist, destination)
            data["component_models"] = {
                name: _relative_path(path, destination)
                for name, path in self.component_models.items()
            }
            for geometry in data["fine_tuning"]["geometries"].values():
                if geometry.get("file"):
                    geometry["file"] = _relative_path(geometry["file"], destination)
        return data

    def save(self, path: str | Path) -> Path:
        config_path = Path(path).expanduser().resolve()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(self.to_dict(config_path.parent), indent=2) + "\n",
            encoding="utf-8",
        )
        return config_path