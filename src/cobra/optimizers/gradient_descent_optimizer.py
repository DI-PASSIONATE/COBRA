from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from cobra.configuration.setting import CobraSetting
from cobra.optimizers.base_optimizer import (
    BaseOptimizer,
    OptimizationProperty,
    parse_netlist_value,
    resolve_linked_master,
)

if TYPE_CHECKING:
    from cobra.optimization_context import OptimizationContext

#: Decay exponents of the step and probe gains over the steps k (Spall's SPSA values).
_STEP_DECAY = 0.602
_PROBE_DECAY = 0.101
#: Probe distance at exploration_scale 1, as a fraction of each parameter's range.
_PROBE_FRACTION = 0.1


class GradientDescentOptimizer(BaseOptimizer):
    """
    GradientDescentOptimizer - A lightweight local optimizer: simultaneous-perturbation
    (SPSA) gradient descent.

    It works in coordinates normalised to [0, 1] per parameter. Each step probes the
    current point at ± a small distance along a random ±1 direction, one probe per
    trial, estimates the gradient from the two penalties, and moves against it: by
    ``learning_rate`` of the range along its largest component. Both distances shrink
    with every step, so the search settles. The current point is kept unrounded, and
    only the proposals are rounded to each parameter's step, so steps smaller than a
    grid step still add up.

    It starts at the context's values (in EM fine-tuning, the design that was just
    evaluated), else at each parameter's ``initial_value``, else at the middle of its
    range; a start not taken from the context is evaluated first.
    """

    _settings: ClassVar[list[CobraSetting]] = [
        CobraSetting(
            name="multi_objective",
            dtype=bool,
            default=False,
            description=(
                "Enable multi-objective optimisation.\n"
                "When disabled, all goal losses are summed into a single scalar."
            ),
        ),
        CobraSetting(
            name="learning_rate",
            dtype=float,
            default=0.2,
            description=(
                "First step, as a fraction of the parameter range; later steps shrink.\n"
                "Smaller values give finer convergence; larger values explore faster."
            ),
        ),
        CobraSetting(
            name="exploration_scale",
            dtype=float,
            default=0.5,
            description=(
                "Distance of the gradient probes: 1 probes at 10 % of the parameter range,\n"
                "at least one step. Increase to smooth out noise; decrease for fine refinement."
            ),
        ),
        CobraSetting(
            name="random_seed",
            dtype=int,
            default=-1,
            description=(
                "Random seed for reproducibility. Use -1 for a non-deterministic run."
            ),
        ),
    ]

    def __init__(
        self,
        multi_objective: bool = False,
        learning_rate: float = 0.2,
        exploration_scale: float = 0.5,
        random_seed: int | None = None,
    ):
        super().__init__(multi_objective=multi_objective)
        self.learning_rate = learning_rate
        self.exploration_scale = exploration_scale
        # -1 is the setting's "no seed", as in the GUI.
        self._rng = np.random.default_rng(None if random_seed == -1 else random_seed)
        self._master_properties: dict[str, OptimizationProperty] = {}
        self._master_order: list[str] = []
        self._alias_to_master: dict[str, str] = {}
        self._reset()

    def _reset(self) -> None:
        self._point: dict[str, float] = {}
        """Current point, normalised to [0, 1] per master parameter and not rounded."""
        self._steps = 0
        self._phase: str | None = None
        """What the last proposal was: ``"start"``, ``"plus"`` or ``"minus"``."""
        self._proposal: dict[str, float] = {}
        self._probes: dict[str, dict[str, float]] = {}
        self._plus_penalty = float("nan")
        self._best_parameters: dict[str, float] = {}
        self._best_penalty = float("inf")

    @staticmethod
    def _round_to_step(value: float, prop: OptimizationProperty) -> float:
        if prop.step is None or prop.step <= 0:
            return value
        return prop.min_value + round((value - prop.min_value) / prop.step) * prop.step

    @staticmethod
    def _clip(value: float, prop: OptimizationProperty) -> float:
        return float(min(max(value, prop.min_value), prop.max_value))

    @staticmethod
    def _span(prop: OptimizationProperty) -> float:
        return prop.max_value - prop.min_value

    def _normalised(self, value: float, prop: OptimizationProperty) -> float:
        span = self._span(prop)
        return (self._clip(value, prop) - prop.min_value) / span if span > 0 else 0.0

    def _raw(self, normalised: float, prop: OptimizationProperty) -> float:
        """The value on *prop*'s grid nearest the normalised coordinate *normalised*."""
        value = self._clip(prop.min_value + normalised * self._span(prop), prop)
        return self._clip(self._round_to_step(value, prop), prop)

    def _collect_masters(self, params: list[OptimizationProperty]) -> list[OptimizationProperty]:
        by_name = {param.name: param for param in params}
        masters: list[OptimizationProperty] = []
        seen: set[str] = set()
        self._alias_to_master = {}

        for param in params:
            master = resolve_linked_master(param, by_name)
            self._alias_to_master[param.name] = master.name
            if master.name not in seen:
                masters.append(master)
                seen.add(master.name)

        return masters

    def _start(self, context: "OptimizationContext", masters: list[OptimizationProperty]) -> tuple[dict[str, float], bool]:
        """The normalised start point, and whether it was taken from *context*, i.e. already evaluated."""
        values = {**context.model_parameters, **context.netlist_parameters}
        start = {}
        for master in masters:
            if master.name in values:
                value = parse_netlist_value(values[master.name])
            elif master.initial_value is not None:
                value = master.initial_value
            else:
                value = (master.min_value + master.max_value) / 2
            start[master.name] = self._normalised(value, master)
        return start, all(master.name in values for master in masters)

    def _format_values(self, master_values: dict[str, float], params: list[OptimizationProperty], with_unit: bool) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for param in params:
            master_name = self._alias_to_master[param.name]
            master = self._master_properties[master_name]
            value = master_values[master_name]
            if with_unit:
                # A linked parameter inherits the master's unit; see netlist_unit.
                values[param.name] = f"{value}{param.unit or master.unit or ''}"
            else:
                values[param.name] = value
        return values

    def _probe_pair(self) -> dict[str, dict[str, float]]:
        """The two probes around the current point along a random ±1 direction, in raw values."""
        distance = self.exploration_scale * _PROBE_FRACTION / (self._steps + 1) ** _PROBE_DECAY
        plus: dict[str, float] = {}
        minus: dict[str, float] = {}
        for name in self._master_order:
            prop = self._master_properties[name]
            span = self._span(prop)
            # At least one grid step, or rounding puts both probes on the current point.
            offset = max(distance, prop.step / span) if prop.step and span > 0 else distance
            offset *= float(self._rng.choice([-1.0, 1.0]))
            plus[name] = self._raw(self._point[name] + offset, prop)
            minus[name] = self._raw(self._point[name] - offset, prop)
        return {"plus": plus, "minus": minus}

    def _descend(self, minus_penalty: float) -> None:
        """Move the current point against the gradient the two probes estimate."""
        gradient: dict[str, float] = {}
        for name in self._master_order:
            prop = self._master_properties[name]
            # The probes' actual half-distance, after rounding and clipping.
            half = (self._normalised(self._probes["plus"][name], prop) - self._normalised(self._probes["minus"][name], prop)) / 2
            gradient[name] = (self._plus_penalty - minus_penalty) / (2 * half) if half else 0.0
        largest = max((abs(component) for component in gradient.values()), default=0.0)
        if largest > 0:
            step = self.learning_rate / (self._steps + 1) ** _STEP_DECAY
            for name, component in gradient.items():
                self._point[name] = float(np.clip(self._point[name] - step * component / largest, 0.0, 1.0))
        self._steps += 1

    def initialize(self, num_goals: int, parallel_trials: int = 1):  # noqa: ARG002 - part of the BaseOptimizer interface
        if self.multi_objective:
            raise NotImplementedError("GradientDescentOptimizer currently supports single-objective optimization only.")
        self._master_properties = {}
        self._master_order = []
        self._alias_to_master = {}
        self._reset()

    def step(self, context: "OptimizationContext", model_input_ranges: list[OptimizationProperty], netlist_property_ranges: list[OptimizationProperty]) -> None:
        if self.multi_objective:
            raise NotImplementedError("GradientDescentOptimizer currently supports single-objective optimization only.")

        masters = self._collect_masters(model_input_ranges + netlist_property_ranges)
        self._master_properties = {param.name: param for param in masters}
        self._master_order = [param.name for param in masters]

        if not self._point:
            self._point, evaluated = self._start(context, masters)
            if not evaluated:
                self._phase = "start"
                self._proposal = {name: self._raw(self._point[name], self._master_properties[name]) for name in self._master_order}
                self._propose(context, model_input_ranges, netlist_property_ranges)
                return

        if self._phase == "plus":
            self._phase = "minus"
        else:
            self._phase = "plus"
            self._probes = self._probe_pair()
        self._proposal = self._probes[self._phase]
        self._propose(context, model_input_ranges, netlist_property_ranges)

    def _propose(self, context: "OptimizationContext", model_input_ranges: list[OptimizationProperty], netlist_property_ranges: list[OptimizationProperty]) -> None:
        context.model_parameters = self._format_values(self._proposal, model_input_ranges, with_unit=False)
        context.netlist_parameters = self._format_values(self._proposal, netlist_property_ranges, with_unit=True)

    def tell(self, context: "OptimizationContext", penalty: list[float] | float):  # noqa: ARG002 - part of the BaseOptimizer interface
        if self.multi_objective:
            raise NotImplementedError("GradientDescentOptimizer currently supports single-objective optimization only.")

        penalty_value = float(np.sum(penalty)) if isinstance(penalty, list) else float(penalty)
        if np.isfinite(penalty_value) and penalty_value < self._best_penalty:
            self._best_penalty = penalty_value
            self._best_parameters = dict(self._proposal)

        if self._phase == "plus":
            self._plus_penalty = penalty_value
            if not np.isfinite(penalty_value):
                self._phase = None  # no estimate from this pair; the next step draws a new one
        elif self._phase == "minus":
            if np.isfinite(penalty_value):
                self._descend(penalty_value)
            self._phase = None
        else:
            self._phase = None

    def get_moo_results(self) -> Any:
        if self.multi_objective:
            raise NotImplementedError("Multi-objective optimization is not supported by GradientDescentOptimizer.")
        raise ValueError("GradientDescentOptimizer does not produce MOO results.")

    def get_best_parameters(self) -> dict[str, float]:
        if self._best_parameters:
            best = dict(self._best_parameters)
        else:
            best = {name: self._raw(value, self._master_properties[name]) for name, value in self._point.items()}
        for alias_name, master_name in self._alias_to_master.items():
            if master_name in best:
                best[alias_name] = best[master_name]
        return best
