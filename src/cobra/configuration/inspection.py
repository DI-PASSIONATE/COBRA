"""Read-only inspection of COBRA run configurations and Xyce netlists.

This module backs the ``cobra parse`` command.  It never mutates state: it
loads a :class:`~cobra.configuration.configuration.RunConfiguration`, parses
every netlist the configuration references, and cross-checks the two so that
missing surrogate models, unusable optimization variables, or unreachable
design goals surface before a run starts.

The reports are plain dataclasses with a JSON-safe :meth:`to_dict`, so the same
information can be rendered as text for humans or emitted as JSON for agents.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from cobra.configuration.configuration import (
    ConfigurationError,
    RunConfiguration,
)
from cobra.optimizers.base_optimizer import OptimizationType
from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
from cobra.spice_sim.simulation_type import SimulationType

# Elements whose positional value can be tuned via ``BaseNetlistParser.set_value``.
VALUE_ELEMENT_TYPES: frozenset[str] = frozenset({"R", "C", "L", "V", "I"})

# Touchstone suffixes recognised by ``EMSurrogateStage``; anything else is loaded
# as an ONNX model.
_TOUCHSTONE_SUFFIXES: frozenset[str] = frozenset(
    [f".s{index}p" for index in range(1, 10)] + [".snp"]
)
_TOUCHSTONE_SUFFIX_RE = re.compile(r"^\.s(\d+)p$", re.IGNORECASE)

# SPICE magnitude suffixes, used for best-effort frequency comparisons only.
_SPICE_SUFFIXES: dict[str, float] = {
    "t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3,
    "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15,
}
_SPICE_NUMBER_RE = re.compile(r"^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([a-zA-Z]*)$")

# ONNX inputs COBRA supplies itself; they never come from an optimization parameter.
_IMPLICIT_MODEL_INPUTS: frozenset[str] = frozenset({"frequency"})


class Severity(Enum):
    """How badly a finding affects a run."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(slots=True)
class Issue:
    """A single cross-check finding, tied to the field that produced it."""

    severity: Severity
    location: str
    message: str


def _jsonable(value: Any) -> Any:
    """Convert dataclass output into JSON-serialisable values, keeping enum values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _spice_number(token: str) -> float | None:
    """Convert a SPICE literal such as ``130G`` to Hz, or return ``None``."""
    match = _SPICE_NUMBER_RE.match(token.strip())
    if not match:
        return None
    mantissa, suffix = match.group(1), match.group(2).lower()
    if not suffix:
        return float(mantissa)
    for name in ("meg",):
        if suffix.startswith(name):
            return float(mantissa) * _SPICE_SUFFIXES[name]
    multiplier = _SPICE_SUFFIXES.get(suffix[:1])
    return float(mantissa) * multiplier if multiplier is not None else None


# ---------------------------------------------------------------------------
# Netlist report
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ElementReport:
    """One parsed netlist element, reduced to the fields worth reporting."""

    name: str
    etype: str
    line: int
    nodes: list[str] = field(default_factory=list)
    value: str | None = None
    model: str | None = None
    params: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PortReport:
    """A netlist port (``P`` element) and the source it carries, if any."""

    name: str
    nodes: list[str] = field(default_factory=list)
    params: dict[str, str] = field(default_factory=dict)
    z0: float | None = None
    ac_amplitude: float | None = None
    sin_amplitude: float | None = None


@dataclass(slots=True)
class ComponentReport:
    """An ``X`` instance that needs a surrogate model."""

    name: str
    nodes: list[str] = field(default_factory=list)
    model: str = ""
    params: dict[str, str] = field(default_factory=dict)
    touchstone_file: str | None = None


@dataclass(slots=True)
class IncludeReport:
    """An ``.INCLUDE`` directive, whether its target is on disk, and what it defines."""

    file_path: str
    resolved_path: str
    line: int
    exists: bool
    generated_for: str | None = None
    subcircuits: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LibraryReport:
    """A ``.LIB`` directive and whether its target is on disk."""

    file_path: str
    resolved_path: str
    entry: str | None
    line: int
    exists: bool


@dataclass(slots=True)
class NetlistReport:
    """Everything ``cobra parse`` knows about one netlist."""

    path: str
    exists: bool = False
    parsed: bool = False
    lines: int = 0
    simulation_type: str = SimulationType.UNKNOWN.value
    num_ports: int = 0
    ports: list[PortReport] = field(default_factory=list)
    components: list[ComponentReport] = field(default_factory=list)
    inline_subcircuits: list[str] = field(default_factory=list)
    includes: list[IncludeReport] = field(default_factory=list)
    libraries: list[LibraryReport] = field(default_factory=list)
    simulation_directives: list[dict[str, Any]] = field(default_factory=list)
    options_directives: dict[str, dict[str, str]] = field(default_factory=dict)
    print_directives: list[dict[str, Any]] = field(default_factory=list)
    hb_probe_nodes: list[str] = field(default_factory=list)
    available_goal_parameters: list[str] = field(default_factory=list)
    ac_goal_parameters: list[str] = field(default_factory=list)
    hb_goal_parameters: list[str] = field(default_factory=list)
    element_counts: dict[str, int] = field(default_factory=dict)
    elements: list[ElementReport] = field(default_factory=list)
    netlist_variables: dict[str, str] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the report."""
        return _jsonable(asdict(self))


def load_netlist_parser(path: str | Path) -> XyceNetlistParser:
    """Parse *path* with the Xyce parser, raising ``ConfigurationError`` on failure."""
    netlist_path = Path(path).expanduser()
    if not netlist_path.is_file():
        raise ConfigurationError(f"Netlist file not found: {netlist_path}")
    parser = XyceNetlistParser()
    try:
        parser.from_file(netlist_path)
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"Failed to parse netlist '{netlist_path}': {exc}") from exc
    return parser


def build_netlist_report(parser: XyceNetlistParser, path: str | Path) -> NetlistReport:
    """Summarise an already parsed netlist."""
    netlist_path = Path(path).expanduser()
    report = NetlistReport(path=str(netlist_path), exists=True, parsed=True)
    report.lines = len(parser.to_string().splitlines())
    report.simulation_type = parser.simulation_type.value
    report.num_ports = parser.num_ports
    report.inline_subcircuits = sorted(parser.inline_subckt_names)
    report.hb_probe_nodes = list(parser.hb_probe_nodes)
    report.available_goal_parameters = list(parser.available_design_parameters)
    report.ac_goal_parameters = SimulationType.AC.available_parameters(parser.num_ports)
    report.hb_goal_parameters = _hb_goal_parameters(parser)

    for element in parser.list_elements():
        report.elements.append(
            ElementReport(
                name=element.name,
                etype=element.etype,
                line=element.line_index + 1,
                nodes=list(element.nodes),
                value=element.value,
                model=element.model,
                params=dict(element.params),
            )
        )
        report.element_counts[element.etype] = report.element_counts.get(element.etype, 0) + 1
        if element.etype == "P":
            source = parser.port_sources.get(element.name, {})
            report.ports.append(
                PortReport(
                    name=element.name,
                    nodes=list(element.nodes),
                    params=dict(element.params),
                    z0=source.get("z0"),
                    ac_amplitude=source.get("ac_amplitude"),
                    sin_amplitude=source.get("sin_amplitude"),
                )
            )

    report.netlist_variables = _netlist_variables(report.elements)

    for name, component in parser.components.items():
        params = dict(component.params)
        report.components.append(
            ComponentReport(
                name=name,
                nodes=list(component.nodes),
                model=component.model,
                params={key: value for key, value in params.items() if key != "TSTONEFILE"},
                touchstone_file=params.get("TSTONEFILE"),
            )
        )

    base_directory = netlist_path.resolve().parent
    for include in parser.includes:
        candidate = _resolve_reference(include.file_path, base_directory)
        generated_for = candidate.stem if candidate.stem in parser.components else None
        entry = IncludeReport(
            file_path=include.file_path,
            resolved_path=str(candidate),
            line=include.line_index + 1,
            exists=candidate.is_file(),
            generated_for=generated_for,
        )
        if entry.exists:
            entry.subcircuits = _included_subcircuits(candidate, report)
        report.includes.append(entry)
    for library in parser.libraries:
        candidate = _resolve_reference(library.file_path, base_directory)
        report.libraries.append(
            LibraryReport(
                file_path=library.file_path,
                resolved_path=str(candidate),
                entry=library.entry,
                line=library.line_index + 1,
                exists=candidate.is_file(),
            )
        )

    for directive in parser.simulation_directives:
        report.simulation_directives.append(
            {
                "directive": directive.directive,
                "simulation_type": SimulationType.from_directive(directive.directive).value,
                "positional": list(directive.positional),
                "kv_params": dict(directive.kv_params),
                "line": directive.line_index + 1,
            }
        )
    report.options_directives = parser.options_directives
    for directive in parser.print_directives:
        report.print_directives.append(
            {
                "analysis": directive.analysis,
                "signals": list(directive.signals),
                "kv_params": dict(directive.kv_params),
                "line": directive.line_index + 1,
            }
        )

    _check_netlist(report, parser)
    return report


def inspect_netlist(path: str | Path) -> NetlistReport:
    """Parse *path* and return its report, recording load failures as issues."""
    try:
        parser = load_netlist_parser(path)
    except ConfigurationError as exc:
        report = NetlistReport(path=str(Path(path).expanduser()))
        report.exists = Path(path).expanduser().is_file()
        report.issues.append(Issue(Severity.ERROR, "netlist", str(exc)))
        return report
    return build_netlist_report(parser, path)


def _resolve_reference(file_path: str, base_directory: Path) -> Path:
    """Resolve a path referenced by a netlist relative to that netlist."""
    candidate = Path(file_path).expanduser()
    return candidate if candidate.is_absolute() else base_directory / candidate


def _included_subcircuits(path: Path, report: NetlistReport) -> list[str]:
    """Return the ``.SUBCKT`` names an included netlist defines."""
    try:
        return sorted(load_netlist_parser(path).inline_subckt_names)
    except ConfigurationError as exc:
        report.issues.append(Issue(Severity.WARNING, "netlist.include", str(exc)))
        return []


def _hb_goal_parameters(parser: XyceNetlistParser) -> list[str]:
    """List the HB goal parameter names this netlist can support."""
    names = [f"Power_dBm[{node}]" for node in parser.hb_probe_nodes]
    names.extend(
        f"Gain_dB[{port}@{node}]"
        for port in sorted(parser.port_sources)
        for node in parser.hb_probe_nodes
    )
    return names


def _netlist_variables(elements: list[ElementReport]) -> dict[str, str]:
    """Map every tunable netlist variable name to its current value."""
    variables: dict[str, str] = {}
    for element in elements:
        if element.etype in VALUE_ELEMENT_TYPES and element.value is not None:
            variables[element.name] = element.value
        for key, value in element.params.items():
            if key.upper() == "TSTONEFILE":
                continue
            variables[f"{element.name}:{key}"] = value
    return variables


def _check_netlist(report: NetlistReport, parser: XyceNetlistParser) -> None:
    """Record netlist-only findings that block or degrade a run."""
    if parser.simulation_type is SimulationType.UNKNOWN:
        report.issues.append(
            Issue(
                Severity.WARNING,
                "netlist",
                "No .AC/.LIN, .HB, .TRAN, or .DC directive found; COBRA injects the "
                "analysis required by the design goals using default parameters.",
            )
        )
    if not parser.num_ports:
        report.issues.append(
            Issue(
                Severity.WARNING,
                "netlist",
                "No P (port) elements found; S-parameter design goals are unavailable.",
            )
        )
    if parser.simulation_type is SimulationType.HB and not parser.hb_probe_nodes:
        report.issues.append(
            Issue(
                Severity.WARNING,
                "netlist",
                "No HB probe node found; a node needs both V(<node>) and I(V<node>), "
                "which requires a 0 V source named V<node>.",
            )
        )
    included_subcircuits = {
        name: include.resolved_path
        for include in report.includes
        for name in include.subcircuits
    }
    for include in report.includes:
        location = f"netlist.include:{include.line}"
        if not include.exists and not include.generated_for:
            report.issues.append(
                Issue(Severity.ERROR, location, f"Included file not found: {include.resolved_path}")
            )
        elif not include.generated_for and not Path(include.file_path).is_absolute():
            report.issues.append(
                Issue(
                    Severity.WARNING,
                    location,
                    f"'{include.file_path}' is relative; COBRA copies the netlist into "
                    "results/<timestamp>_<name>/ before simulating, so Xyce resolves it from "
                    "there. Use an absolute path.",
                )
            )
    for library in report.libraries:
        location = f"netlist.lib:{library.line}"
        if not library.exists:
            report.issues.append(
                Issue(Severity.ERROR, location, f"Library file not found: {library.resolved_path}")
            )
        elif not Path(library.file_path).is_absolute():
            report.issues.append(
                Issue(
                    Severity.WARNING,
                    location,
                    f"'{library.file_path}' is relative; COBRA copies the netlist into "
                    "results/<timestamp>_<name>/ before simulating, so Xyce resolves it from "
                    "there. Use an absolute path.",
                )
            )
    for component in report.components:
        source = included_subcircuits.get(component.model)
        if source is None:
            continue
        report.issues.append(
            Issue(
                Severity.WARNING,
                f"netlist.component:{component.name}",
                f"'{component.name}' uses subcircuit '{component.model}' defined in "
                f"{source}, but COBRA still treats it as a surrogate: it needs a "
                "component_models entry and its model is replaced by a vector fit.",
            )
        )
    for component in report.components:
        if not component.nodes:
            report.issues.append(
                Issue(
                    Severity.ERROR,
                    f"netlist.component:{component.name}",
                    "Subcircuit instance has no nodes.",
                )
            )


# ---------------------------------------------------------------------------
# Configuration report
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ComponentModelReport:
    """One ``component_models`` entry, checked against netlist and model file."""

    component: str
    path: str
    exists: bool = False
    kind: str = "unknown"
    model_ports: int | None = None
    instance_nodes: int | None = None
    model_inputs: list[str] = field(default_factory=list)
    detail: str | None = None


@dataclass(slots=True)
class OptimizationParameterReport:
    """One optimization parameter and where it resolves to."""

    name: str
    type: str
    min_value: float
    max_value: float
    step: float | None = None
    unit: str | None = None
    linked_to: str | None = None
    target: str | None = None
    current_value: str | None = None
    resolved: bool = False


@dataclass(slots=True)
class DesignGoalReport:
    """One design goal and the analysis it needs."""

    parameter: str
    kind: str
    frequency_range: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    weight: float = 1.0
    node: str | None = None
    port: str | None = None
    simulation_type: str = SimulationType.UNKNOWN.value
    directive_in_netlist: bool = False
    valid: bool = False


@dataclass(slots=True)
class GeometryReport:
    """One fine-tuning geometry entry."""

    component: str
    source: str
    class_name: str
    module: str | None = None
    file: str | None = None
    file_exists: bool | None = None
    resolved: bool = False
    detail: str | None = None


@dataclass(slots=True)
class ConfigurationReport:
    """Everything ``cobra parse`` knows about one JSON configuration."""

    path: str
    valid: bool = False
    schema_version: Any = None
    netlist_path: str | None = None
    optimizer: dict[str, Any] = field(default_factory=dict)
    simulator: dict[str, Any] = field(default_factory=dict)
    max_iterations: int | None = None
    component_models: list[ComponentModelReport] = field(default_factory=list)
    optimization_parameters: list[OptimizationParameterReport] = field(default_factory=list)
    design_goals: list[DesignGoalReport] = field(default_factory=list)
    simulation_parameters: dict[str, dict[str, str]] = field(default_factory=dict)
    fine_tuning: dict[str, Any] = field(default_factory=dict)
    geometries: list[GeometryReport] = field(default_factory=list)
    netlist: NetlistReport | None = None
    issues: list[Issue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the report."""
        return _jsonable(asdict(self))


def inspect_configuration(path: str | Path, *, check_models: bool = True) -> ConfigurationReport:
    """Load *path*, parse its netlist, and cross-check both against each other.

    ``check_models`` also opens every referenced ONNX or Touchstone model so the
    port count and model inputs can be verified.  Set it to ``False`` to keep the
    report to path and mapping checks.
    """
    config_path = Path(path).expanduser().resolve()
    report = ConfigurationReport(path=str(config_path))
    raw = _load_raw_configuration(config_path, report)
    if raw is None:
        return report
    report.schema_version = raw.get("schema_version")

    try:
        configuration = RunConfiguration.load(config_path)
    except ConfigurationError as exc:
        report.issues.append(Issue(Severity.ERROR, "configuration", str(exc)))
        _inspect_raw_netlist(raw, config_path, report)
        return report

    report.valid = True
    report.netlist_path = configuration.netlist
    report.max_iterations = configuration.max_iterations
    report.optimizer = {"name": configuration.optimizer.name, "settings": configuration.optimizer.settings}
    report.simulator = {"name": configuration.simulator.name, "settings": configuration.simulator.settings}
    report.simulation_parameters = configuration.simulation_parameters
    report.fine_tuning = {
        "enabled": configuration.fine_tuning.enabled,
        "palace_command": configuration.fine_tuning.palace_command,
        "iterations": configuration.fine_tuning.iterations,
        "optimizer": configuration.fine_tuning.optimizer,
    }

    _check_backends(configuration, report)

    parser: XyceNetlistParser | None = None
    try:
        parser = load_netlist_parser(configuration.netlist)
    except ConfigurationError as exc:
        report.issues.append(Issue(Severity.ERROR, "netlist", str(exc)))
    if parser is not None:
        report.netlist = build_netlist_report(parser, configuration.netlist)

    _check_component_models(configuration, parser, report, check_models=check_models)
    _check_optimization_parameters(configuration, parser, report)
    _check_design_goals(configuration, parser, report)
    _check_simulation_parameters(configuration, parser, report)
    _check_fine_tuning(configuration, report)
    return report


def _load_raw_configuration(config_path: Path, report: ConfigurationReport) -> dict[str, Any] | None:
    """Read the JSON document, reporting syntax and structure problems."""
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        report.issues.append(
            Issue(Severity.ERROR, "configuration", f"Configuration file not found: {config_path}")
        )
        return None
    except OSError as exc:
        report.issues.append(Issue(Severity.ERROR, "configuration", f"Cannot read configuration: {exc}"))
        return None
    except json.JSONDecodeError as exc:
        report.issues.append(
            Issue(
                Severity.ERROR,
                "configuration",
                f"Invalid JSON in '{config_path}': {exc.msg} at line {exc.lineno}",
            )
        )
        return None
    if not isinstance(data, dict):
        report.issues.append(
            Issue(Severity.ERROR, "configuration", "Configuration root must be a JSON object")
        )
        return None
    return data


def _inspect_raw_netlist(raw: dict[str, Any], config_path: Path, report: ConfigurationReport) -> None:
    """Parse the netlist of an otherwise invalid configuration, best effort."""
    netlist = raw.get("netlist")
    if not isinstance(netlist, str) or not netlist.strip():
        return
    candidate = Path(netlist).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    candidate = candidate.resolve()
    report.netlist_path = str(candidate)
    report.netlist = inspect_netlist(candidate)


def _check_backends(configuration: RunConfiguration, report: ConfigurationReport) -> None:
    """Verify optimizer and simulator names against the run registries."""
    # Imported here: config_runner pulls in the full COBRA pipeline.
    from cobra.configuration.config_runner import OPTIMIZER_REGISTRY, SIMULATOR_REGISTRY

    if configuration.optimizer.name not in OPTIMIZER_REGISTRY:
        report.issues.append(
            Issue(
                Severity.ERROR,
                "optimizer",
                f"Unsupported optimizer '{configuration.optimizer.name}'. Supported: "
                f"{', '.join(OPTIMIZER_REGISTRY)}",
            )
        )
    if configuration.simulator.name not in SIMULATOR_REGISTRY:
        report.issues.append(
            Issue(
                Severity.ERROR,
                "simulator",
                f"Unsupported simulator '{configuration.simulator.name}'. Supported: "
                f"{', '.join(SIMULATOR_REGISTRY)}",
            )
        )
    command = configuration.simulator.settings.get("xyce_command", "Xyce")
    if (
        configuration.simulator.name == "XyceSimulator"
        and isinstance(command, str)
        and shutil.which(command) is None
        and not Path(command).is_file()
    ):
        report.issues.append(
            Issue(
                Severity.WARNING,
                "simulator.settings.xyce_command",
                f"Xyce command '{command}' was not found on PATH; load Xyce (e.g. via Spack) "
                "before running.",
            )
        )


def _model_kind(path: Path) -> str:
    """Classify a model file the same way ``EMSurrogateStage`` does."""
    return "touchstone" if path.suffix.lower() in _TOUCHSTONE_SUFFIXES else "onnx"


def _inspect_touchstone(path: Path) -> tuple[int | None, str | None]:
    """Return the port count of a Touchstone file, or why it could not be read."""
    try:
        import skrf as rf

        return rf.Network(str(path)).number_of_ports, None
    except ImportError as exc:  # pragma: no cover - scikit-rf is a hard dependency
        return None, f"scikit-rf is unavailable ({exc})"
    except Exception as exc:  # noqa: BLE001 - scikit-rf raises unrelated types for bad files
        return None, f"Touchstone file could not be read: {exc}"


def _inspect_onnx(path: Path) -> tuple[int | None, list[str], str | None]:
    """Return (ports, input names, detail) for an ONNX surrogate model."""
    try:
        from onnxruntime import InferenceSession
    except ImportError as exc:
        return None, [], f"onnxruntime is unavailable ({exc})"
    try:
        session = InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001 - onnxruntime errors derive from Exception
        return None, [], f"ONNX model could not be loaded: {exc}"
    inputs = [node.name for node in session.get_inputs()]
    outputs = [node.name for node in session.get_outputs()]
    ports = math.isqrt(len(outputs) // 2)
    # Every port pair contributes an Sij real and imaginary output.
    if 2 * ports * ports != len(outputs):
        return None, inputs, f"Model has {len(outputs)} outputs, which is not 2*N*N S-parameters"
    return ports, inputs, None


def _check_component_models(
    configuration: RunConfiguration,
    parser: XyceNetlistParser | None,
    report: ConfigurationReport,
    *,
    check_models: bool,
) -> None:
    """Cross-check every surrogate mapping against the netlist and the model file."""
    components = parser.components if parser is not None else {}
    for name in sorted(set(components) - set(configuration.component_models)):
        report.issues.append(
            Issue(
                Severity.ERROR,
                f"component_models.{name}",
                f"Netlist component '{name}' has no model mapping; add an .onnx or .sNp file.",
            )
        )
    for component, model in sorted(configuration.component_models.items()):
        model_path = Path(model)
        entry = ComponentModelReport(
            component=component,
            path=model,
            exists=model_path.is_file(),
            kind=_model_kind(model_path),
        )
        location = f"component_models.{component}"
        if parser is not None and component not in components:
            report.issues.append(
                Issue(
                    Severity.ERROR,
                    location,
                    f"Component '{component}' is not a surrogate instance in the netlist. "
                    f"Available: {', '.join(sorted(components)) or 'none'}",
                )
            )
        elif parser is not None:
            entry.instance_nodes = len(components[component].nodes)

        if not entry.exists:
            report.issues.append(Issue(Severity.ERROR, location, f"Model file not found: {model}"))
            report.component_models.append(entry)
            continue

        suffix_match = _TOUCHSTONE_SUFFIX_RE.match(model_path.suffix)
        if suffix_match and entry.kind != "touchstone":
            report.issues.append(
                Issue(
                    Severity.ERROR,
                    location,
                    f"Suffix '{model_path.suffix}' is not recognised as Touchstone (only .s1p-.s9p "
                    "and .snp are); the file would be loaded as an ONNX model.",
                )
            )
        if check_models:
            if entry.kind == "touchstone":
                entry.model_ports, entry.detail = _inspect_touchstone(model_path)
            else:
                entry.model_ports, entry.model_inputs, entry.detail = _inspect_onnx(model_path)
            if entry.detail:
                report.issues.append(Issue(Severity.WARNING, location, entry.detail))
        elif suffix_match:
            entry.model_ports = int(suffix_match.group(1))

        if (
            entry.model_ports is not None
            and entry.instance_nodes is not None
            and entry.model_ports != entry.instance_nodes
        ):
            report.issues.append(
                Issue(
                    Severity.ERROR,
                    location,
                    f"Model has {entry.model_ports} ports but instance '{component}' connects "
                    f"{entry.instance_nodes} nodes; the vector-fitted subcircuit exposes one node "
                    "per port.",
                )
            )
        report.component_models.append(entry)


def _check_optimization_parameters(
    configuration: RunConfiguration,
    parser: XyceNetlistParser | None,
    report: ConfigurationReport,
) -> None:
    """Resolve every optimization parameter against the netlist or a surrogate model."""
    elements = {element.name: element for element in parser.list_elements()} if parser else {}
    variables = _netlist_variables(report.netlist.elements) if report.netlist else {}
    model_inputs = {
        entry.component: entry.model_inputs
        for entry in report.component_models
        if entry.kind == "onnx"
    }
    used_inputs: dict[str, set[str]] = {component: set() for component in model_inputs}

    for parameter in configuration.optimization_parameters:
        entry = OptimizationParameterReport(
            name=parameter.name,
            type=parameter.type,
            min_value=parameter.min_value,
            max_value=parameter.max_value,
            step=parameter.step,
            unit=parameter.unit,
            linked_to=parameter.linked_to,
        )
        location = f"optimization_parameters.{parameter.name}"
        instance, _, key = parameter.name.partition(":")
        if OptimizationType(parameter.type) is OptimizationType.NETLIST_VARIABLE:
            if parser is None:
                entry.target = "unknown (netlist unavailable)"
            elif key:
                element = elements.get(instance)
                if element is None:
                    report.issues.append(
                        Issue(
                            Severity.ERROR,
                            location,
                            f"Netlist has no element '{instance}'; the value would be skipped "
                            "during the run.",
                        )
                    )
                else:
                    entry.resolved = True
                    entry.target = f"{instance} ({element.etype}) parameter '{key}'"
                    entry.current_value = element.params.get(key)
                    if key not in element.params:
                        report.issues.append(
                            Issue(
                                Severity.WARNING,
                                location,
                                f"Element '{instance}' has no '{key}' parameter yet; it would be "
                                "appended to the instance line.",
                            )
                        )
            else:
                element = elements.get(parameter.name)
                if element is None:
                    report.issues.append(
                        Issue(
                            Severity.ERROR,
                            location,
                            f"Netlist has no element '{parameter.name}'; the value would be "
                            "skipped during the run.",
                        )
                    )
                elif element.etype not in VALUE_ELEMENT_TYPES:
                    report.issues.append(
                        Issue(
                            Severity.ERROR,
                            location,
                            f"Element '{parameter.name}' is of type '{element.etype}'; only "
                            f"{', '.join(sorted(VALUE_ELEMENT_TYPES))} elements have a tunable "
                            "positional value. Use '<instance>:<parameter>' instead.",
                        )
                    )
                else:
                    entry.resolved = True
                    entry.target = f"{parameter.name} ({element.etype}) value"
                    entry.current_value = variables.get(parameter.name)
        else:
            targets = [instance] if key else sorted(model_inputs)
            input_name = key or parameter.name
            if not key:
                entry.target = "all ONNX components"
                if not model_inputs:
                    report.issues.append(
                        Issue(
                            Severity.WARNING,
                            location,
                            "No ONNX component is configured, so this model input is unused.",
                        )
                    )
            elif instance not in configuration.component_models:
                report.issues.append(
                    Issue(
                        Severity.ERROR,
                        location,
                        f"'{instance}' is not a configured component; model inputs must be named "
                        "'<component>:<model input>'.",
                    )
                )
            elif instance not in model_inputs:
                report.issues.append(
                    Issue(
                        Severity.WARNING,
                        location,
                        f"Component '{instance}' uses a fixed Touchstone model, so this model "
                        "input has no effect.",
                    )
                )
            else:
                entry.target = f"{instance} model input '{input_name}'"
            for component in targets:
                names = model_inputs.get(component)
                if not names:
                    continue
                used_inputs[component].add(input_name)
                entry.resolved = True
                if input_name not in names:
                    report.issues.append(
                        Issue(
                            Severity.ERROR,
                            location,
                            f"Model for '{component}' has no input '{input_name}'. Available: "
                            f"{', '.join(names) or 'none'}",
                        )
                    )
        report.optimization_parameters.append(entry)

    for component, names in model_inputs.items():
        missing = [
            name
            for name in names
            if name not in used_inputs[component] and name not in _IMPLICIT_MODEL_INPUTS
        ]
        if missing:
            report.issues.append(
                Issue(
                    Severity.ERROR,
                    f"optimization_parameters.{component}",
                    f"Model input(s) {', '.join(missing)} of '{component}' are not covered by an "
                    "optimization parameter; inference needs a value for every model input.",
                )
            )
    if not configuration.optimization_parameters:
        report.issues.append(
            Issue(
                Severity.WARNING,
                "optimization_parameters",
                "No optimization parameters defined; every iteration would simulate the same design.",
            )
        )


def _check_design_goals(
    configuration: RunConfiguration,
    parser: XyceNetlistParser | None,
    report: ConfigurationReport,
) -> None:
    """Rebuild every goal with the run-time validation and record what it needs."""
    # Imported here: both modules pull in the full COBRA pipeline.
    from cobra.configuration.config_runner import build_design_goals
    from cobra.optimizers.design_goal import DesignGoal

    directives = (
        {SimulationType.from_directive(item.directive) for item in parser.simulation_directives}
        if parser is not None
        else set()
    )
    for goal in configuration.design_goals:
        entry = DesignGoalReport(
            parameter=goal.parameter,
            kind=goal.kind,
            frequency_range=goal.frequency_range,
            min_value=goal.min_value,
            max_value=goal.max_value,
            weight=goal.weight,
            node=goal.node,
            port=goal.port,
        )
        location = f"design_goals.{goal.parameter}"
        simulation_type = (
            SimulationType.HB
            if goal.kind in {"power_dbm", "gain_db"}
            else SimulationType.for_parameter(goal.parameter)
        )
        entry.simulation_type = simulation_type.value
        entry.directive_in_netlist = simulation_type in directives
        if parser is not None:
            try:
                # build_design_goals prints a line per goal; the report renders its own.
                with contextlib.redirect_stdout(io.StringIO()):
                    build_design_goals([goal], parser)
            except ConfigurationError as exc:
                report.issues.append(Issue(Severity.ERROR, location, str(exc)))
            else:
                entry.valid = True
        try:
            DesignGoal.str_to_frequency_range(goal.frequency_range)
        except ValueError as exc:
            report.issues.append(Issue(Severity.ERROR, location, str(exc)))
        else:
            _check_goal_frequency(goal.frequency_range, simulation_type, parser, location, report)
        if (
            parser is not None
            and not entry.directive_in_netlist
            and simulation_type is not SimulationType.UNKNOWN
        ):
            report.issues.append(
                Issue(
                    Severity.INFO,
                    location,
                    f"Netlist has no {simulation_type.value} directive; COBRA adds one from "
                    "simulation_parameters and built-in defaults.",
                )
            )
        report.design_goals.append(entry)

    if not configuration.design_goals:
        report.issues.append(
            Issue(Severity.WARNING, "design_goals", "No design goals defined; nothing to optimize for.")
        )


def _check_goal_frequency(
    frequency_range: str | None,
    simulation_type: SimulationType,
    parser: XyceNetlistParser | None,
    location: str,
    report: ConfigurationReport,
) -> None:
    """Warn when a goal frequency falls outside the analysis the netlist requests."""
    from cobra.optimizers.design_goal import DesignGoal

    if parser is None or frequency_range is None:
        return
    low, high = DesignGoal.str_to_frequency_range(frequency_range)
    if low is None or high is None:
        return
    for directive in parser.simulation_directives:
        if SimulationType.from_directive(directive.directive) is not simulation_type:
            continue
        values = [_spice_number(token) for token in directive.positional]
        known = [value for value in values if value is not None and value > 0.0]
        if not known:
            continue
        if simulation_type is SimulationType.AC and len(known) >= 2:
            start, stop = known[-2], known[-1]
        else:
            start = stop = known[0]
        if low < start or high > stop:
            report.issues.append(
                Issue(
                    Severity.WARNING,
                    location,
                    f"Goal frequency {frequency_range} lies outside the netlist "
                    f"{directive.directive} range {start:g}-{stop:g} Hz.",
                )
            )
        return


def _check_simulation_parameters(
    configuration: RunConfiguration,
    parser: XyceNetlistParser | None,
    report: ConfigurationReport,
) -> None:
    """Verify simulation-parameter keys against the netlist and Xyce metadata."""
    # Imported here: the simulator module pulls in scikit-rf and pandas.
    from cobra.spice_sim.xyce_simulator import XyceSimulator

    directives = (
        {SimulationType.from_directive(item.directive): item for item in parser.simulation_directives}
        if parser is not None
        else {}
    )
    for key, values in configuration.simulation_parameters.items():
        location = f"simulation_parameters[{key}]"
        if key.upper().startswith(".OPTIONS:"):
            category = key.split(":", 1)[1].lower()
            if parser is not None and category not in parser.options_directives:
                report.issues.append(
                    Issue(
                        Severity.ERROR,
                        location,
                        f"Netlist has no '.options {category}' line to update. Available: "
                        f"{', '.join(sorted(parser.options_directives)) or 'none'}",
                    )
                )
            continue
        directive = key if key.startswith(".") else f".{key}"
        simulation_type = SimulationType.from_directive(directive)
        if simulation_type is SimulationType.UNKNOWN:
            report.issues.append(Issue(Severity.ERROR, location, f"Unknown simulation directive '{key}'"))
            continue
        target = directives.get(simulation_type)
        if parser is not None and target is None:
            report.issues.append(
                Issue(
                    Severity.INFO,
                    location,
                    f"Netlist has no {directive} directive; these values are used when COBRA "
                    "injects the analysis for a design goal.",
                )
            )
        known = set(XyceSimulator.get_simulation_metadata(simulation_type).positional_param_names)
        if target is not None:
            known |= set(target.kv_params)
        for name in values:
            if name not in known:
                report.issues.append(
                    Issue(
                        Severity.WARNING,
                        location,
                        f"'{name}' is not a positional parameter of {directive} "
                        f"({', '.join(sorted(known)) or 'none'}); it is written as a key=value token.",
                    )
                )


def _check_fine_tuning(configuration: RunConfiguration, report: ConfigurationReport) -> None:
    """Check the Palace command and the ORCA geometry of every ONNX component."""
    fine_tuning = configuration.fine_tuning
    onnx_components = {
        entry.component for entry in report.component_models if entry.kind == "onnx"
    }
    for component, geometry in sorted(fine_tuning.geometries.items()):
        entry = GeometryReport(
            component=component,
            source=geometry.source,
            class_name=geometry.class_name,
            module=geometry.module,
            file=geometry.file,
            file_exists=Path(geometry.file).is_file() if geometry.file else None,
        )
        location = f"fine_tuning.geometries.{component}"
        if component not in {model.component for model in report.component_models}:
            report.issues.append(
                Issue(Severity.ERROR, location, f"Geometry references unknown component '{component}'.")
            )
        if fine_tuning.enabled:
            try:
                # Imported here so ORCA stays optional for non fine-tuning runs.
                from cobra.configuration.geometry_loader import resolve_geometry_class

                resolve_geometry_class(geometry)
            except ConfigurationError as exc:
                entry.detail = str(exc)
                report.issues.append(Issue(Severity.ERROR, location, str(exc)))
            else:
                entry.resolved = True
        report.geometries.append(entry)

    if not fine_tuning.enabled:
        return
    missing = sorted(onnx_components - set(fine_tuning.geometries))
    if missing:
        report.issues.append(
            Issue(
                Severity.ERROR,
                "fine_tuning.geometries",
                f"ONNX component(s) {', '.join(missing)} need an ORCA geometry for EM fine-tuning.",
            )
        )
    command = fine_tuning.palace_command
    if shutil.which(command) is None and not Path(command).is_file():
        report.issues.append(
            Issue(
                Severity.WARNING,
                "fine_tuning.palace_command",
                f"Palace command '{command}' was not found on PATH.",
            )
        )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

#: Long lists are truncated in text output unless ``full`` is requested.
TEXT_ITEM_LIMIT: int = 50

_LABEL_WIDTH = 24


def all_issues(report: ConfigurationReport | NetlistReport) -> list[Issue]:
    """Return the report's own issues plus those of a nested netlist report."""
    issues = list(report.issues)
    if isinstance(report, ConfigurationReport) and report.netlist is not None:
        issues.extend(report.netlist.issues)
    return issues


def count_issues(report: ConfigurationReport | NetlistReport) -> dict[str, int]:
    """Count issues per severity value."""
    counts = {severity.value: 0 for severity in Severity}
    for issue in all_issues(report):
        counts[issue.severity.value] += 1
    return counts


def has_errors(report: ConfigurationReport | NetlistReport) -> bool:
    """Return ``True`` when the report contains at least one error."""
    return count_issues(report)[Severity.ERROR.value] > 0


def _heading(lines: list[str], title: str) -> None:
    lines.append("")
    lines.append(title)
    lines.append("-" * len(title))


def _field(lines: list[str], label: str, value: Any) -> None:
    lines.append(f"  {label:<{_LABEL_WIDTH}}: {value}")


def _join(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def _field_list(lines: list[str], label: str, items: list[str], *, full: bool) -> None:
    """Render a labelled comma-separated list, truncated unless *full* is set."""
    if full or len(items) <= TEXT_ITEM_LIMIT:
        _field(lines, label, _join(items))
        return
    shown = _join(items[:TEXT_ITEM_LIMIT])
    _field(lines, label, f"{shown}, ... and {len(items) - TEXT_ITEM_LIMIT} more (use --full)")


def _entries(lines: list[str], items: list[str], *, full: bool) -> None:
    if not items:
        lines.append("  none")
        return
    shown = items if full or len(items) <= TEXT_ITEM_LIMIT else items[:TEXT_ITEM_LIMIT]
    lines.extend(f"  {item}" for item in shown)
    if len(shown) < len(items):
        lines.append(f"  ... and {len(items) - len(shown)} more (use --full)")


def _render_issues(lines: list[str], issues: list[Issue], *, full: bool) -> None:
    counts: dict[str, int] = {severity.value: 0 for severity in Severity}
    for issue in issues:
        counts[issue.severity.value] += 1
    summary = ", ".join(f"{counts[severity.value]} {severity.value}" for severity in Severity)
    _heading(lines, f"Issues ({summary})")
    ordered = [issue for severity in Severity for issue in issues if issue.severity is severity]
    _entries(
        lines,
        [f"{issue.severity.value.upper():<7} {issue.location}: {issue.message}" for issue in ordered],
        full=full,
    )


def render_netlist_report(report: NetlistReport, *, full: bool = False, issues: bool = True) -> str:
    """Render a netlist report as plain text."""
    lines = [f"Netlist: {report.path}"]
    if not report.parsed:
        lines.append("  not parsed")
        if issues:
            _render_issues(lines, report.issues, full=full)
        return "\n".join(lines)

    _field(lines, "lines", report.lines)
    _field(lines, "primary analysis", report.simulation_type)
    _field(lines, "ports", report.num_ports)
    _field(lines, "surrogate components", len(report.components))
    _field(lines, "hb probe nodes", _join(report.hb_probe_nodes))
    _field(
        lines,
        "elements",
        _join([f"{etype}={count}" for etype, count in sorted(report.element_counts.items())]),
    )

    _heading(lines, "Analysis directives")
    _entries(
        lines,
        [
            f"line {item['line']:>4}  {item['directive']} "
            f"{' '.join(item['positional'])} "
            f"{' '.join(f'{key}={value}' for key, value in item['kv_params'].items())}".rstrip()
            for item in report.simulation_directives
        ],
        full=full,
    )
    _entries(
        lines,
        [
            f"line {item['line']:>4}  .PRINT {item['analysis']} "
            f"{' '.join(f'{key}={value}' for key, value in item['kv_params'].items())} "
            f"{' '.join(item['signals'])}".rstrip()
            for item in report.print_directives
        ],
        full=full,
    )
    _entries(
        lines,
        [
            f"           .options {category} "
            f"{' '.join(f'{key}={value}' for key, value in params.items())}".rstrip()
            for category, params in sorted(report.options_directives.items())
        ],
        full=full,
    )

    _heading(lines, "Ports")
    _entries(
        lines,
        [
            f"{port.name:<12} nodes={_join(port.nodes)}  z0={port.z0}  "
            f"ac={port.ac_amplitude}  sin={port.sin_amplitude}"
            for port in report.ports
        ],
        full=full,
    )

    _heading(lines, "Surrogate components (need a component_models entry)")
    _entries(
        lines,
        [
            f"{component.name:<12} ports={len(component.nodes)}  nodes={_join(component.nodes)}  "
            f"model={component.model}"
            + (f"  TSTONEFILE={component.touchstone_file}" if component.touchstone_file else "")
            for component in report.components
        ],
        full=full,
    )

    _heading(lines, "Included files, libraries, and inline subcircuits")
    _entries(
        lines,
        [
            f"line {include.line:>4}  {include.resolved_path}  "
            + (
                f"(generated for {include.generated_for} during the run)"
                if include.generated_for and not include.exists
                else ("found" if include.exists else "MISSING")
            )
            for include in report.includes
        ],
        full=full,
    )
    _entries(
        lines,
        [
            f"line {library.line:>4}  {library.resolved_path}  entry={library.entry}  "
            + ("found" if library.exists else "MISSING")
            for library in report.libraries
        ],
        full=full,
    )
    _field_list(lines, "inline .SUBCKT", report.inline_subcircuits, full=full)

    _heading(lines, "Design goal parameters")
    _field_list(lines, "available now", report.available_goal_parameters, full=full)
    _field_list(lines, "with .AC analysis", report.ac_goal_parameters, full=full)
    _field_list(lines, "with .HB analysis", report.hb_goal_parameters, full=full)

    _heading(lines, "Netlist variables (type netlist_variable)")
    _entries(
        lines,
        [f"{name:<28} = {value}" for name, value in report.netlist_variables.items()],
        full=full,
    )

    if issues:
        _render_issues(lines, report.issues, full=full)
    return "\n".join(lines)


def render_configuration_report(report: ConfigurationReport, *, full: bool = False) -> str:
    """Render a configuration report as plain text."""
    counts = count_issues(report)
    lines = [f"Configuration: {report.path}"]
    _field(lines, "schema version", report.schema_version)
    _field(lines, "loads", "yes" if report.valid else "no")
    _field(
        lines,
        "issues",
        ", ".join(f"{counts[severity.value]} {severity.value}" for severity in Severity),
    )
    if report.valid:
        _field(lines, "netlist", report.netlist_path)
        _field(lines, "max iterations", report.max_iterations)
        _field(
            lines,
            "optimizer",
            f"{report.optimizer.get('name')}  {report.optimizer.get('settings')}",
        )
        _field(
            lines,
            "simulator",
            f"{report.simulator.get('name')}  {report.simulator.get('settings')}",
        )

        _heading(lines, "Component models")
        _entries(
            lines,
            [
                f"{entry.component:<12} {entry.path}\n"
                f"               kind={entry.kind}  exists={'yes' if entry.exists else 'no'}  "
                f"model_ports={entry.model_ports}  instance_nodes={entry.instance_nodes}"
                + (f"  inputs={_join(entry.model_inputs)}" if entry.model_inputs else "")
                for entry in report.component_models
            ],
            full=full,
        )

        _heading(lines, "Optimization parameters")
        _entries(
            lines,
            [
                f"{entry.name:<28} {entry.type:<17} [{entry.min_value}, {entry.max_value}]"
                f"  step={entry.step}  unit={entry.unit}  linked_to={entry.linked_to}\n"
                f"               -> {entry.target or 'UNRESOLVED'}"
                + (f"  current={entry.current_value}" if entry.current_value is not None else "")
                for entry in report.optimization_parameters
            ],
            full=full,
        )

        _heading(lines, "Design goals")
        _entries(
            lines,
            [
                f"{entry.parameter:<28} kind={entry.kind}  min={entry.min_value}  "
                f"max={entry.max_value}  range={entry.frequency_range}  weight={entry.weight}\n"
                f"               analysis={entry.simulation_type}  "
                f"in_netlist={'yes' if entry.directive_in_netlist else 'no'}  "
                f"valid={'yes' if entry.valid else 'no'}"
                for entry in report.design_goals
            ],
            full=full,
        )

        _heading(lines, "Simulation parameters")
        _entries(
            lines,
            [
                f"{key:<28} {' '.join(f'{name}={value}' for name, value in values.items())}"
                for key, values in report.simulation_parameters.items()
            ],
            full=full,
        )

        _heading(lines, "Fine-tuning")
        _field(lines, "enabled", report.fine_tuning.get("enabled"))
        _field(lines, "palace command", report.fine_tuning.get("palace_command"))
        _field(lines, "iterations", report.fine_tuning.get("iterations"))
        _field(lines, "optimizer", report.fine_tuning.get("optimizer"))
        _entries(
            lines,
            [
                f"{entry.component:<12} source={entry.source}  class={entry.class_name}  "
                f"module={entry.module}  file={entry.file}  resolved={entry.resolved}"
                for entry in report.geometries
            ],
            full=full,
        )

    if report.netlist is not None:
        lines.append("")
        lines.append(render_netlist_report(report.netlist, full=full, issues=False))

    _render_issues(lines, all_issues(report), full=full)
    return "\n".join(lines)


def render_report(report: ConfigurationReport | NetlistReport, *, full: bool = False) -> str:
    """Render either report type as plain text."""
    if isinstance(report, ConfigurationReport):
        return render_configuration_report(report, full=full)
    return render_netlist_report(report, full=full)


def is_configuration_file(path: str | Path) -> bool:
    """Return ``True`` when *path* looks like a COBRA JSON configuration."""
    candidate = Path(path).expanduser()
    if candidate.suffix.lower() == ".json":
        return True
    try:
        with candidate.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(2048).lstrip().startswith("{")
    except OSError:
        return False


def inspect_path(
    path: str | Path, *, kind: str = "auto", check_models: bool = True
) -> ConfigurationReport | NetlistReport:
    """Inspect *path* as a configuration or a netlist.

    ``kind`` is ``"auto"``, ``"config"``, or ``"netlist"``; ``"auto"`` treats
    ``.json`` files and JSON objects as configurations and everything else as a
    netlist.
    """
    if kind not in {"auto", "config", "netlist"}:
        raise ConfigurationError(f"Unsupported parse kind '{kind}'")
    if kind == "config" or (kind == "auto" and is_configuration_file(path)):
        return inspect_configuration(path, check_models=check_models)
    return inspect_netlist(path)
