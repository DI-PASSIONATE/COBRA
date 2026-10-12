"""Build a starter run configuration from a netlist, for ``cobra init``.

Only what the netlist determines is filled in: the netlist path, the analysis
parameters of its simulation directives, and a model for every component whose
model is known.  The optimizer and simulator carry their default settings so the
available options are visible in the file.  Design goals and optimization parameters are design decisions,
so they are left empty for the user to add; ``cobra parse`` reports them as
missing until they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from cobra.configuration.configuration import BackendConfig, ConfigurationError, RunConfiguration
from cobra.configuration.inspection import NetlistReport, build_netlist_report, load_netlist
from cobra.spice_sim.simulation_type import SimulationType

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cobra.spice_sim.netlist_parsers.netlist import Netlist, SimulationDirective

# Positional slots that select *what* is swept rather than how; the GUI hides them too.
_STRUCTURAL_PARAMETERS = {"sweep_type", "src_name"}


@dataclass(slots=True)
class InitialConfiguration:
    """A starter configuration and what the user still has to provide."""

    configuration: RunConfiguration
    unmapped_components: list[str] = field(default_factory=list)
    goal_parameters: list[str] = field(default_factory=list)


def _directive_parameters(directive: SimulationDirective, names: list[str]) -> dict[str, str]:
    """Name the positional tokens of *directive*; the last name absorbs any extra tokens."""
    values: dict[str, str] = {}
    for index, name in enumerate(names):
        if index >= len(directive.positional):
            break
        last = index == len(names) - 1
        values[name] = " ".join(directive.positional[index:]) if last else directive.positional[index]
    return {name: value for name, value in values.items() if name not in _STRUCTURAL_PARAMETERS}


def _simulation_parameters(netlist: Netlist) -> dict[str, dict[str, str]]:
    parameters: dict[str, dict[str, str]] = {}
    for directive in netlist.simulation_directives:
        simulation_type = directive.simulation_type
        # Only analyses: .LIN maps to AC as well but carries no sweep.
        if not directive.is_analysis or simulation_type.value in parameters:
            continue
        names = netlist.parser.analysis_metadata(simulation_type).positional_param_names
        values = _directive_parameters(directive, names)
        if values:
            parameters[simulation_type.value] = values
    return parameters


def _component_models(
    parsed: Netlist, netlist: Path, models: Mapping[str, str]
) -> dict[str, str]:
    unknown = sorted(set(models) - set(parsed.components))
    if unknown:
        available = ", ".join(sorted(parsed.components)) or "none"
        raise ConfigurationError(
            f"Unknown component(s) {', '.join(unknown)} in model mapping; "
            f"components in {netlist.name}: {available}"
        )
    resolved = {name: str(Path(path).expanduser().resolve()) for name, path in models.items()}
    for name, component in parsed.components.items():
        touchstone = component.params.get("TSTONEFILE")
        if name in resolved or not touchstone:
            continue
        candidate = Path(touchstone).expanduser()
        candidate = candidate if candidate.is_absolute() else netlist.parent / candidate
        if candidate.is_file():
            resolved[name] = str(candidate)
    return resolved


def _default_backend(name: str, registry: Mapping[str, type]) -> BackendConfig:
    settings = getattr(registry[name], "_settings", [])
    return BackendConfig(name, {setting.name: setting.default for setting in settings})


def _goal_parameters(report: NetlistReport) -> list[str]:
    """The goal parameter names the netlist's own analysis supports."""
    by_type = {
        SimulationType.HB.value: report.hb_goal_parameters,
        SimulationType.TRAN.value: report.tran_goal_parameters,
    }
    return list(by_type.get(report.simulation_type, report.available_goal_parameters))


def initial_configuration(
    netlist: str | Path,
    models: Mapping[str, str] | None = None,
    simulator: str = "XyceSimulator",
) -> InitialConfiguration:
    """Return a starter configuration for *netlist*, simulated by *simulator*.

    *models* maps component names to ONNX or Touchstone files, relative to the
    current directory.  A component without one keeps the netlist's own
    ``TSTONEFILE`` when that file exists, and is otherwise reported as unmapped.
    """
    # Imported here: the runner pulls in the full COBRA pipeline.
    from cobra.configuration.config_runner import OPTIMIZER_REGISTRY, SIMULATOR_REGISTRY

    if simulator not in SIMULATOR_REGISTRY:
        raise ConfigurationError(
            f"Unsupported simulator '{simulator}'. Supported: {', '.join(SIMULATOR_REGISTRY)}"
        )
    netlist_path = Path(netlist).expanduser().resolve()
    parsed = load_netlist(netlist_path, SIMULATOR_REGISTRY[simulator].netlist_parser)
    parsed.select_surrogates(models or {})
    component_models = _component_models(parsed, netlist_path, models or {})
    defaults = RunConfiguration(netlist=str(netlist_path))
    configuration = RunConfiguration(
        netlist=str(netlist_path),
        component_models=component_models,
        simulation_parameters=_simulation_parameters(parsed),
        optimizer=_default_backend(defaults.optimizer.name, OPTIMIZER_REGISTRY),
        simulator=_default_backend(simulator, SIMULATOR_REGISTRY),
    )
    return InitialConfiguration(
        configuration=configuration,
        unmapped_components=sorted(set(parsed.components) - set(component_models)),
        goal_parameters=_goal_parameters(build_netlist_report(parsed, netlist_path)),
    )
