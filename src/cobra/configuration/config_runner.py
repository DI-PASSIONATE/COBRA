from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cobra.cobra import COBRA
from cobra.configuration.configuration import (
    AnalysisVariantConfig,
    ConfigurationError,
    DesignGoalConfig,
    RunConfiguration,
)
from cobra.configuration.geometry_loader import create_geometry
from cobra.optimizers.base_optimizer import OptimizationProperty, OptimizationType
from cobra.optimizers.design_goal import DesignGoal, DesignParameter
from cobra.optimizers.design_goal_collection import (
    find_parameter,
    make_expression_parameter,
    make_gain_db,
    make_isolation_db,
    make_power_dbm,
    with_variant,
)
from cobra.optimizers.gradient_descent_optimizer import GradientDescentOptimizer
from cobra.optimizers.optuna_optimizer import OptunaOptimizer
from cobra.spice_sim.simulation_type import AnalysisVariant, SimulationType, base_simulation_type
from cobra.spice_sim.vacask_simulator import VacaskSimulator
from cobra.spice_sim.xyce_simulator import XyceSimulator

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from cobra.optimization_context import OptimizationContext
    from cobra.spice_sim.base_simulator import BaseSimulator
    from cobra.spice_sim.netlist_parsers.netlist import Netlist

OPTIMIZER_REGISTRY = {
    "OptunaOptimizer": OptunaOptimizer,
    "GradientDescentOptimizer": GradientDescentOptimizer,
}
SIMULATOR_REGISTRY: dict[str, type[BaseSimulator]] = {
    "XyceSimulator": XyceSimulator,
    "VacaskSimulator": VacaskSimulator,
}


def build_analysis_variants(
    configurations: Sequence[AnalysisVariantConfig],
    netlist: Netlist,
    simulator: type[BaseSimulator] | None = None,
) -> dict[str, AnalysisVariant]:
    """The run's analysis variants by name, each checked against *netlist* and *simulator*.

    A value the netlist has no element or instance for would be skipped with a
    warning at run time, leaving the variant a copy of the netlist's own analysis.
    """
    variants: dict[str, AnalysisVariant] = {}
    for config in configurations:
        simulation_type = config.simulation_type()
        if simulator is not None and simulation_type not in simulator.supported_simulation_types:
            raise ConfigurationError(
                f"Analysis variant '{config.name}' runs {simulation_type.name}, which "
                f"{simulator.__name__} cannot run"
            )
        for name in config.netlist_values:
            element = name.split(":", 1)[0]
            if not netlist.has_element(element):
                raise ConfigurationError(
                    f"Analysis variant '{config.name}' sets '{name}', but the netlist has no "
                    f"element or instance '{element}'"
                )
        variants[config.name] = AnalysisVariant.create(config.name, simulation_type, config.netlist_values)
    return variants


def _variant_amplitude(variant: AnalysisVariant, port: str) -> float | None:
    """The sine amplitude *variant* gives the source of *port*, if it sets one."""
    value = variant.netlist_value(f"{port}:ampl")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(
            f"Analysis variant '{variant.name}' sets '{port}:ampl' to {value!r}; a gain goal on it "
            "needs the amplitude as a plain number of volts"
        ) from exc


def _build_goal(
    config: DesignGoalConfig, netlist: Netlist, variants: Mapping[str, AnalysisVariant]
) -> DesignGoal:
    variant = variants.get(config.analysis)
    # A goal on a variant is named with the variant's prefix; its parameter is built
    # without it and then moved onto the variant.
    name = config.parameter.removeprefix(f"{variant.name}:") if variant is not None else config.parameter
    if config.kind == "catalogue":
        parameter = find_parameter(name)
        if parameter is None:
            raise ConfigurationError(f"Unknown design parameter '{name}'")
        if parameter.min_ports > netlist.num_ports:
            raise ConfigurationError(
                f"Design parameter '{name}' requires at least "
                f"{parameter.min_ports} ports; netlist has {netlist.num_ports}"
            )
        if variant is not None and variant.simulation_type is not parameter.simulation_type:
            raise ConfigurationError(
                f"Design goal '{config.parameter}' needs a {parameter.simulation_type.name} analysis; "
                f"its variant '{variant.name}' runs {variant.simulation_type.name}"
            )
    elif config.kind in {"power_dbm", "isolation_db"}:
        analysis = variant.simulation_type if variant is not None else config.analysis_type()
        if config.node not in netlist.probe_nodes:
            raise ConfigurationError(
                f"{analysis.value} goal node '{config.node}' is not available in the netlist"
            )
        if config.kind == "isolation_db" and not config.frequency_range:
            raise ConfigurationError(
                f"Isolation goal at '{config.node}' needs a frequency_range naming the "
                "target line, e.g. '35GHz'"
            )
        parameter = (
            make_power_dbm(config.node or "", analysis)
            if config.kind == "power_dbm"
            else make_isolation_db(config.node or "", analysis)
        )
    else:
        analysis = variant.simulation_type if variant is not None else config.analysis_type()
        source = netlist.port_sources.get(config.port or "")
        if config.node not in netlist.probe_nodes:
            raise ConfigurationError(
                f"{analysis.value} gain goal node '{config.node}' is not available in the netlist"
            )
        if source is None:
            raise ConfigurationError(
                f"{analysis.value} gain goal port '{config.port}' is not a driven netlist port"
            )
        amplitude = config.source_amplitude if config.source_amplitude is not None else source["sin_amplitude"]
        driven = _variant_amplitude(variant, config.port or "") if variant is not None else None
        if driven is not None:
            # The gain is referred to the drive the variant sets.
            if config.source_amplitude is not None and config.source_amplitude != driven:
                raise ConfigurationError(
                    f"Gain goal '{config.parameter}' gives source_amplitude {config.source_amplitude}, "
                    f"but its variant '{config.analysis}' drives '{config.port}' with {driven}"
                )
            amplitude = driven
        parameter = make_gain_db(
            config.port or "",
            amplitude,
            config.impedance if config.impedance is not None else source.get("z0", 50.0),
            config.node or "",
            analysis,
        )
    if variant is not None:
        parameter = with_variant(parameter, variant)
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
        scale=config.scale,
    )


def _build_expression_goal(
    config: DesignGoalConfig, parameters: Mapping[str, DesignParameter]
) -> DesignGoal:
    terms = config.terms or {}
    missing = [name for name in terms.values() if name not in parameters]
    if missing:
        raise ConfigurationError(
            f"Expression goal '{config.parameter}' uses {', '.join(missing)}, which no other "
            "design goal reads"
        )
    try:
        parameter = make_expression_parameter(
            config.parameter,
            config.expression or "",
            {variable: parameters[name] for variable, name in terms.items()},
        )
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    if len(parameter.required_analyses) > 1:
        low, high = DesignGoal.str_to_frequency_range(config.frequency_range)
        if low is None or low != high:
            raise ConfigurationError(
                f"Expression goal '{config.parameter}' combines values of several analyses, whose "
                "frequency points need not match; give it a single frequency, e.g. '35GHz'"
            )
    return DesignGoal(
        parameter,
        frequency_range=config.frequency_range,
        min_value=config.min_value,
        max_value=config.max_value,
        weight=config.weight,
        scale=config.scale,
    )


def build_design_goals(
    configurations: list[DesignGoalConfig],
    netlist: Netlist,
    simulator: type[BaseSimulator] | None = None,
    variants: Mapping[str, AnalysisVariant] | None = None,
) -> list[DesignGoal]:
    """Reconstruct design goals using the same validation as a headless run.

    *variants* are the run's analysis variants by name (:func:`build_analysis_variants`).
    An expression goal's terms are the parameters of other goals in *configurations*,
    possibly other expression goals, which are built first.
    With *simulator*, a goal whose analysis that simulator cannot run is rejected.
    """
    variants = variants or {}
    built = {
        index: _build_goal(config, netlist, variants)
        for index, config in enumerate(configurations)
        if config.kind != "expression"
    }
    parameters = {goal.parameter.name: goal.parameter for goal in built.values()}
    pending = {index: config for index, config in enumerate(configurations) if config.kind == "expression"}
    while pending:
        ready = [
            index
            for index, config in pending.items()
            if all(name in parameters or name not in {c.parameter for c in pending.values()}
                   for name in (config.terms or {}).values())
        ]
        if not ready:
            names = ", ".join(config.parameter for config in pending.values())
            raise ConfigurationError(f"Expression goals depend on each other in a cycle: {names}")
        for index in ready:
            goal = _build_expression_goal(pending.pop(index), parameters)
            built[index] = goal
            parameters[goal.parameter.name] = goal.parameter
    goals = [built[index] for index in range(len(configurations))]
    if simulator is not None:
        for goal in goals:
            for analysis in goal.required_analyses:
                simulation_type = base_simulation_type(analysis)
                if simulation_type not in simulator.supported_simulation_types:
                    raise ConfigurationError(
                        f"Design goal '{goal.parameter.name}' needs a {simulation_type.name} analysis, "
                        f"which {simulator.__name__} cannot run"
                    )
    return goals


def _apply_simulation_parameters(
    netlist: Netlist, parameters: dict[str, dict[str, str]]
) -> dict[SimulationType, dict[str, str]]:
    existing = {
        directive.simulation_type
        for directive in netlist.simulation_directives
        if directive.is_analysis
    }
    by_type: dict[SimulationType, dict[str, str]] = {}
    for key, values in parameters.items():
        if key.upper().startswith(".OPTIONS:"):
            netlist.update_options_directive(key.split(":", 1)[1], values)
            continue
        directive = key if key.startswith(".") else f".{key}"
        simulation_type = SimulationType.from_directive(directive)
        if simulation_type is SimulationType.UNKNOWN:
            raise ConfigurationError(f"Unknown simulation directive '{key}'")
        by_type[simulation_type] = dict(values)
        if simulation_type in existing:
            netlist.update_simulation_directive(simulation_type, values)
    return by_type


def _unresolved_includes(netlist: Netlist, configuration: RunConfiguration) -> list[str]:
    """Included files the simulator cannot find, other than the surrogate files COBRA writes."""
    directory = Path(configuration.netlist).resolve().parent
    return [
        included.file_path
        for included in [*netlist.includes, *netlist.libraries]
        if not (candidate := netlist.parser.resolve_include(included.file_path, directory)).is_file()
        and candidate.stem not in configuration.component_models
    ]


@dataclass(slots=True)
class ConfiguredRun:
    configuration: RunConfiguration
    cobra: COBRA
    netlist: Netlist
    design_goals: list[DesignGoal]
    optimization_parameters: list[OptimizationProperty]
    orca_geometries: dict[str, Any]
    simulation_parameters: dict[SimulationType, dict[str, str]]

    def run(self, callback: Callable[[OptimizationContext], bool | None] | None = None) -> OptimizationContext:
        return self.cobra.run(
            netlist=self.configuration.netlist,
            design_goals=self.design_goals,
            optimization_parameters=self.optimization_parameters,
            max_iterations=self.configuration.max_iterations,
            parallel_trials=self.configuration.parallel_trials,
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
    if configuration.parallel_trials > 1 and not optimizer_class.supports_parallel_trials:
        raise ConfigurationError(
            f"{configuration.optimizer.name} needs the result of one trial before it can suggest "
            "the next, so parallel_trials must be 1 for it."
        )
    try:
        simulator_class = SIMULATOR_REGISTRY[configuration.simulator.name]
    except KeyError as exc:
        raise ConfigurationError(
            f"Unsupported simulator '{configuration.simulator.name}'. Supported: "
            f"{', '.join(SIMULATOR_REGISTRY)}"
        ) from exc

    if not configuration.design_goals:
        raise ConfigurationError("No design goals defined; add at least one to design_goals")
    if not configuration.optimization_parameters:
        raise ConfigurationError(
            "No optimization parameters defined; add at least one to optimization_parameters"
        )

    netlist = simulator_class.netlist_parser.parse_file(configuration.netlist)
    netlist.select_surrogates(configuration.component_models)
    components = set(netlist.components)
    configured_components = set(configuration.component_models)
    if components != configured_components:
        missing = components - configured_components
        unknown = configured_components - components
        details = []
        if missing:
            details.append(f"missing models for {', '.join(sorted(missing))}")
            unresolved = _unresolved_includes(netlist, configuration)
            if unresolved:
                details.append(
                    f"these included files were not found, so masters they define look like "
                    f"surrogates: {', '.join(unresolved)} (VACASK searches SIM_INCLUDE_PATH and "
                    "~/.vacaskrc.toml; `cobra parse` lists where it looked)"
                )
        if unknown:
            details.append(f"unknown components {', '.join(sorted(unknown))}")
        raise ConfigurationError("Component model mapping does not match netlist: " + "; ".join(details))

    simulation_parameters = _apply_simulation_parameters(
        netlist, configuration.simulation_parameters
    )
    variants = build_analysis_variants(configuration.analysis_variants, netlist, simulator_class)
    goals = build_design_goals(configuration.design_goals, netlist, simulator_class, variants)
    properties = [
        OptimizationProperty(
            name=item.name,
            type=OptimizationType(item.type),
            min_value=item.min_value,
            max_value=item.max_value,
            step=item.step,
            unit=item.unit,
            linked_to=item.linked_to,
            initial_value=item.initial_value,
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
        netlist=netlist,
        component_onnx_mapping=configuration.component_models,
        optimizer=optimizer_class(**configuration.optimizer.settings),
        circuit_simulator=simulator_class(**configuration.simulator.settings),
        palace_fine_tuning_command=fine_tuning.palace_command if fine_tuning.enabled else None,
        palace_fine_tuning_processes=fine_tuning.palace_processes,
        fine_tuning_iterations=fine_tuning.iterations,
        fine_tuning_optimizer=fine_tuning.optimizer if fine_tuning.enabled else "reuse",
    )
    return ConfiguredRun(
        configuration=configuration,
        cobra=cobra,
        netlist=netlist,
        design_goals=goals,
        optimization_parameters=properties,
        orca_geometries=geometries,
        simulation_parameters=simulation_parameters,
    )


def run_configuration_file(
    path: str | Path, callback: Callable[[OptimizationContext], bool | None] | None = None
) -> OptimizationContext:
    return build_configured_run(RunConfiguration.load(path)).run(callback)
