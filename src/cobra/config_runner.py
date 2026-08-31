from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from cobra.configuration import ConfigurationError, DesignGoalConfig, RunConfiguration
from cobra.cobra import COBRA
from cobra.geometry_loader import create_geometry
from cobra.optimizers.base_optimizer import OptimizationProperty, OptimizationType
from cobra.optimizers.design_goal import DesignGoal
from cobra.optimizers.design_goal_collection import find_parameter, make_gain_db, make_power_dbm
from cobra.optimizers.optuna_optimizer import OptunaOptimizer
from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
from cobra.spice_sim.simulation_type import SimulationType
from cobra.spice_sim.xyce_simulator import XyceSimulator


OPTIMIZER_REGISTRY = {"OptunaOptimizer": OptunaOptimizer}
SIMULATOR_REGISTRY = {"XyceSimulator": XyceSimulator}


def _build_goal(config: DesignGoalConfig, parser: XyceNetlistParser) -> DesignGoal:
    if config.kind == "catalogue":
        parameter = find_parameter(config.parameter)
        if parameter is None:
            raise ConfigurationError(f"Unknown design parameter '{config.parameter}'")
        if parameter.min_ports > parser.num_ports:
            raise ConfigurationError(
                f"Design parameter '{config.parameter}' requires at least "
                f"{parameter.min_ports} ports; netlist has {parser.num_ports}"
            )
    elif config.kind == "power_dbm":
        if config.node not in parser.hb_probe_nodes:
            raise ConfigurationError(
                f"HB goal node '{config.node}' is not available in the netlist"
            )
        parameter = make_power_dbm(config.node or "")
    else:
        source = parser.port_sources.get(config.port or "")
        if config.node not in parser.hb_probe_nodes:
            raise ConfigurationError(
                f"HB goal node '{config.node}' is not available in the netlist"
            )
        if source is None:
            raise ConfigurationError(
                f"HB gain goal port '{config.port}' is not a driven netlist port"
            )
        parameter = make_gain_db(
            config.port or "",
            config.source_amplitude if config.source_amplitude is not None else source["sin_amplitude"],
            config.impedance if config.impedance is not None else source.get("z0", 50.0),
            config.node or "",
        )
    if parameter.name != config.parameter:
        raise ConfigurationError(
            f"Design goal parameter '{config.parameter}' does not match its metadata "
            f"(expected '{parameter.name}')"
        )
    return DesignGoal(
        parameter,
        frequency_range=config.frequency_range,
        min_value=config.min_value,
        max_value=config.max_value,
        weight=config.weight,
    )


def build_design_goals(
    configurations: list[DesignGoalConfig], parser: XyceNetlistParser
) -> list[DesignGoal]:
    """Reconstruct design goals using the same validation as a headless run."""
    return [_build_goal(config, parser) for config in configurations]


def _apply_simulation_parameters(
    parser: XyceNetlistParser, parameters: dict[str, dict[str, str]]
) -> dict[SimulationType, dict[str, str]]:
    existing = {
        SimulationType.from_directive(directive.directive)
        for directive in parser.simulation_directives
    }
    by_type: dict[SimulationType, dict[str, str]] = {}
    for key, values in parameters.items():
        if key.upper().startswith(".OPTIONS:"):
            parser.update_options_directive(key.split(":", 1)[1], values)
            continue
        directive = key if key.startswith(".") else f".{key}"
        simulation_type = SimulationType.from_directive(directive)
        if simulation_type is SimulationType.UNKNOWN:
            raise ConfigurationError(f"Unknown simulation directive '{key}'")
        by_type[simulation_type] = dict(values)
        if simulation_type in existing:
            parser.update_simulation_directive(directive, values)
    return by_type


@dataclass(slots=True)
class ConfiguredRun:
    configuration: RunConfiguration
    cobra: COBRA
    parser: XyceNetlistParser
    design_goals: list[DesignGoal]
    optimization_parameters: list[OptimizationProperty]
    orca_geometries: dict[str, Any]
    simulation_parameters: dict[SimulationType, dict[str, str]]

    def run(self, callback: Callable[[dict], bool | None] | None = None) -> dict:
        return self.cobra.run(
            netlist=self.configuration.netlist,
            design_goals=self.design_goals,
            optimization_parameters=self.optimization_parameters,
            max_iterations=self.configuration.max_iterations,
            orca_geometries=self.orca_geometries,
            callback=callback,
            sim_params_by_type=self.simulation_parameters,
            run_configuration=self.configuration,
        )


def build_configured_run(configuration: RunConfiguration) -> ConfiguredRun:
    """Validate *configuration* and construct all objects needed for one COBRA run."""
    configuration.validate()
    try:
        optimizer_class = OPTIMIZER_REGISTRY[configuration.optimizer.name]
    except KeyError as exc:
        raise ConfigurationError(
            f"Unsupported optimizer '{configuration.optimizer.name}'. Supported: "
            f"{', '.join(OPTIMIZER_REGISTRY)}"
        ) from exc
    try:
        simulator_class = SIMULATOR_REGISTRY[configuration.simulator.name]
    except KeyError as exc:
        raise ConfigurationError(
            f"Unsupported simulator '{configuration.simulator.name}'. Supported: "
            f"{', '.join(SIMULATOR_REGISTRY)}"
        ) from exc

    parser = XyceNetlistParser().from_file(configuration.netlist)
    components = set(parser.components)
    configured_components = set(configuration.component_models)
    if components != configured_components:
        missing = components - configured_components
        unknown = configured_components - components
        details = []
        if missing:
            details.append(f"missing models for {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown components {', '.join(sorted(unknown))}")
        raise ConfigurationError("Component model mapping does not match netlist: " + "; ".join(details))

    simulation_parameters = _apply_simulation_parameters(
        parser, configuration.simulation_parameters
    )
    goals = build_design_goals(configuration.design_goals, parser)
    properties = [
        OptimizationProperty(
            name=item.name,
            type=OptimizationType(item.type),
            min_value=item.min_value,
            max_value=item.max_value,
            step=item.step,
            unit=item.unit,
            linked_to=item.linked_to,
        )
        for item in configuration.optimization_parameters
    ]
    geometries = (
        {name: create_geometry(spec) for name, spec in configuration.fine_tuning.geometries.items()}
        if configuration.fine_tuning.enabled
        else {}
    )
    unknown_geometry_components = set(geometries) - components
    if unknown_geometry_components:
        raise ConfigurationError(
            "Fine-tuning geometries reference unknown components: "
            + ", ".join(sorted(unknown_geometry_components))
        )

    fine_tuning = configuration.fine_tuning
    cobra = COBRA(
        netlist_parser=parser,
        component_onnx_mapping=configuration.component_models,
        optimizer=optimizer_class(**configuration.optimizer.settings),
        circuit_simulator=simulator_class(**configuration.simulator.settings),
        palace_fine_tuning_command=fine_tuning.palace_command if fine_tuning.enabled else None,
        fine_tuning_iterations=fine_tuning.iterations,
        fine_tuning_optimizer=fine_tuning.optimizer if fine_tuning.enabled else "reuse",
    )
    return ConfiguredRun(
        configuration=configuration,
        cobra=cobra,
        parser=parser,
        design_goals=goals,
        optimization_parameters=properties,
        orca_geometries=geometries,
        simulation_parameters=simulation_parameters,
    )


def run_configuration_file(
    path: str | Path, callback: Callable[[dict], bool | None] | None = None
) -> dict:
    return build_configured_run(RunConfiguration.load(path)).run(callback)