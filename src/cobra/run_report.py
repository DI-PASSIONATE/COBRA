"""What ``cobra run`` prints to stdout: the run header and the final report.

The counterpart of :mod:`cobra.configuration.inspection` for ``cobra parse``
and :mod:`cobra.diagnostics` for ``cobra doctor`` — each command's reporting
lives beside the data it renders, leaving :mod:`cobra.__main__` to argument
parsing and exit codes.

Formatting primitives (colour, rules, sliders, field alignment) come from
:mod:`cobra.console`; this module only decides what to say about a run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from cobra.console import (
    Palette,
    SliderSpec,
    format_duration,
    format_parameter_line,
    supports_unicode,
    write,
    write_fields,
    write_heading,
    write_rule,
)
from cobra.diagnostics import cobra_version
from cobra.optimizers.base_optimizer import parse_netlist_value

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cobra.configuration.config_runner import ConfiguredRun
    from cobra.optimization_context import OptimizationContext
    from cobra.optimizers.base_optimizer import OptimizationProperty

#: Longest parameter or goal name rendered before it is truncated.
_LABEL_WIDTH = 28

#: Track width of the sliders in the final report.
_SLIDER_WIDTH = 20


def write_run_header(configured: ConfiguredRun, palette: Palette) -> None:
    """Announce what is about to run, before any simulation starts."""
    configuration = configured.configuration
    fine_tuning = configuration.fine_tuning
    write_heading(f"COBRA {cobra_version()}", palette)
    write_rule(palette)
    write_fields(
        [
            ("netlist", Path(configuration.netlist).name),
            ("analysis", configured.parser.simulation_type.value),
            ("optimizer", configuration.optimizer.name),
            ("simulator", configuration.simulator.name),
            ("parameters", str(len(configured.optimization_parameters))),
            ("goals", str(len(configured.design_goals))),
            ("iterations", f"up to {configuration.max_iterations}"),
            (
                "parallel trials",
                str(configuration.parallel_trials) if configuration.parallel_trials > 1 else "",
            ),
            (
                "fine-tuning",
                f"{fine_tuning.palace_command} ({fine_tuning.iterations} iterations)"
                if fine_tuning.enabled
                else "",
            ),
        ],
        palette,
    )
    write()


def write_run_report(context: OptimizationContext, palette: Palette) -> None:
    """The design the run settled on: its parameters, its goals, and a summary."""
    _write_parameters(context, palette)
    _write_goals(context, palette)
    _write_summary(context, palette)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _slider_specs(parameters: Sequence[OptimizationProperty]) -> list[SliderSpec]:
    return [
        SliderSpec(
            name=prop.name,
            minimum=prop.min_value,
            maximum=prop.max_value,
            unit=prop.unit,
            linked_to=prop.linked_to,
        )
        for prop in parameters
    ]


def _parameter_values(context: OptimizationContext) -> dict[str, float | None]:
    """Every optimization parameter's final value, as a number where possible.

    ``None`` for anything unparseable — a netlist expression, say — which the
    report shows as a dash rather than failing over.
    """
    raw_values = {**context.netlist_parameters, **context.model_parameters}
    values: dict[str, float | None] = {}
    for prop in context.optimization_parameters:
        try:
            values[prop.name] = parse_netlist_value(raw_values[prop.name])
        except (KeyError, ValueError):
            values[prop.name] = None
    return values


def _write_parameters(context: OptimizationContext, palette: Palette) -> None:
    specs = _slider_specs(context.optimization_parameters)
    if not specs:
        return

    values = _parameter_values(context)
    label_width = min(max(len(spec.name) for spec in specs), _LABEL_WIDTH)

    write()
    write_heading("Design parameters", palette)
    write_rule(palette)
    for spec in specs:
        line = format_parameter_line(
            spec,
            values.get(spec.name),
            palette=palette,
            width=_SLIDER_WIDTH,
            label_width=label_width,
        )
        write(f"  {line}")


def _goal_target(goal) -> str:
    bounds = []
    if goal.min_value is not None:
        bounds.append(f">= {goal.min_value:g}")
    if goal.max_value is not None:
        bounds.append(f"<= {goal.max_value:g}")
    return " and ".join(bounds) or "no bound"


def _goal_reached(goal, palette: Palette) -> str:
    value = goal.current_value
    if value is None:
        return palette.dim("no result")
    if isinstance(value, (int, float)):
        return f"{float(value):g}"
    # An array of values over the goal's frequency range; report its span.
    low, high = float(value.min()), float(value.max())
    return f"{low:g}" if low == high else f"{low:g} … {high:g}"


def _write_goals(context: OptimizationContext, palette: Palette) -> None:
    goals = context.goals
    if not goals:
        return

    tick, cross = ("✓", "✗") if supports_unicode(sys.stdout) else ("OK", "!!")
    label_width = min(max(len(goal.parameter_name) for goal in goals), _LABEL_WIDTH)

    write()
    write_heading("Design goals", palette)
    write_rule(palette)
    for goal in goals:
        met = goal.current_penalty is not None and goal.current_penalty <= 0.0
        mark = palette.green(tick) if met else palette.red(cross)
        write(
            f"  {mark} {goal.parameter_name[:label_width].ljust(label_width)} "
            f"{palette.dim(goal.frequency_range or 'full range')}  "
            f"target {_goal_target(goal)}  reached {_goal_reached(goal, palette)}"
        )


def _write_summary(context: OptimizationContext, palette: Palette) -> None:
    iteration = context.iteration
    if context.goal_achieved:
        status = palette.green(f"design goals achieved at iteration {iteration}")
    else:
        status = palette.yellow(f"design goals not achieved after {iteration} iterations")

    # Elapsed time, not context.times — those sum the stages over every trial and
    # exceed the run's duration whenever trials were evaluated concurrently.
    wall_time = context.wall_time
    write()
    write_heading("Summary", palette)
    write_rule(palette)
    write_fields(
        [
            ("status", status),
            ("wall time", format_duration(wall_time) if wall_time else ""),
            ("results", context.results_dir),
        ],
        palette,
    )
