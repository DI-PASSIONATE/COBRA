from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from cobra.optimization_context import OptimizationContext
    from cobra.optimizers.goal_expression import GoalExpression
    from cobra.spice_sim.base_simulator import SimulationResult
    from cobra.spice_sim.simulation_type import Analysis, AnalysisVariant, SimulationType


FAILED_SIMULATION_PENALTY = 1e6
"""Penalty for a goal whose simulation produced no result.
"""


class UnavailableValueError(LookupError):
    """A parameter's value cannot be computed for a trial: an analysis it reads has no
    result, or an expression gives a value that is not finite (e.g. ``ln`` of a negative).
    """


@dataclass(frozen=True)
class GoalInputs:
    """Which inputs a goal on a parameter takes; drives the design-goal dialog.

    Attributes:
    ----------
    max_value:
        Whether an upper bound is meaningful. A margin such as isolation only
        has a lower bound.
    frequency_required:
        Whether the goal needs a frequency; without one, a goal covers the
        whole sweep or spectrum.
    single_frequency:
        Whether the goal targets one frequency point and cannot take a range.
    """

    max_value: bool = True
    frequency_required: bool = False
    single_frequency: bool = False


@dataclass
class DesignParameter:
    """
    A named, self-describing design parameter.

    Attributes:
    ----------
    name:
        Unique identifier used in the GUI, context dict, and log output
        (e.g. ``"S21_dB"``, ``"Lp"``).
    simulation_type:
        Which Xyce analysis type must be run to produce the result.
    formula:
        Callable ``(SimulationResult) -> np.ndarray | float`` that extracts the parameter value(s) from the simulation result.
    loss:
        Callable ``(min_value, max_value, current_value, scale) -> float`` that computes the penalty for the current value relative to the goal range;
        ``scale`` is the goal's unit of deviation, ``None`` for one relative to the bound.
    description:
        Human-readable explanation shown as a tooltip in the GUI.
    inputs:
        Which bounds and frequency inputs a goal on this parameter takes.
    variant:
        The analysis variant the parameter is read from, ``None`` for the netlist's own
        analysis of ``simulation_type``
        (see :func:`~cobra.optimizers.design_goal_collection.with_variant`).
    """

    name: str
    simulation_type: SimulationType
    formula: Callable[[SimulationResult, str | None], np.ndarray | float]
    loss: Callable[[float | None, float | None, np.ndarray | float, float | None], float]
    description: str = ""
    min_ports: int = 1
    """Minimum number of netlist ports required to evaluate this parameter."""
    inputs: GoalInputs = GoalInputs()
    variant: AnalysisVariant | None = None

    @property
    def analysis(self) -> Analysis:
        """The analysis whose result the formula reads."""
        return self.variant if self.variant is not None else self.simulation_type

    @property
    def required_analyses(self) -> tuple[Analysis, ...]:
        """The analyses whose results the parameter reads."""
        return (self.analysis,)

    def values(
        self, sim_results: Mapping[Analysis, SimulationResult], frequency_range: str | None = None
    ) -> np.ndarray | float:
        """The parameter's value(s) in a trial's results, at *frequency_range*.

        Raises:
            UnavailableValueError: The analysis it reads has no result.
        """
        sim_result = sim_results.get(self.analysis)
        if sim_result is None:
            raise UnavailableValueError(f"No {self.analysis.value} result for '{self.name}'")
        return self.formula(sim_result, frequency_range)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, DesignParameter):
            return self.name == other.name
        return NotImplemented



class DesignGoal:
    """
    A single optimisation goal: ``parameter`` must satisfy ``[min_value, max_value]``
    within the given ``frequency_range``.

    ``scale`` is the unit a deviation from a bound is measured in, e.g. ``1.0`` for one dB
    of a dB goal. ``None`` measures it relative to the bound, or in units of 1 for a
    bound of 0 (see :func:`~cobra.optimizers.design_goal_collection.deviation_scale`).
    """

    def __init__(
        self,
        parameter: DesignParameter,
        frequency_range: str | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        weight: float = 1.0,
        scale: float | None = None,
    ):
        if not isinstance(parameter, DesignParameter):
            raise TypeError(
                f"'parameter' must be a DesignParameter instance, got {type(parameter).__name__}. "
                "Use find_parameter() or get_available_parameters() to look up parameters."
            )
        self.parameter = parameter
        self.frequency_range = frequency_range
        self.min_value = min_value
        self.max_value = max_value
        self.weight = weight
        if scale is not None and not scale > 0:
            raise ValueError(f"Design goal '{parameter.name}' scale must be positive, got {scale}")
        self.scale = scale
        self._current_value = None
        self._current_penalty = None

    @property
    def parameter_name(self) -> str:
        return self.parameter.name

    @property
    def current_value(self) -> float | np.ndarray | None:
        return self._current_value

    @current_value.setter
    def current_value(self, value: float | np.ndarray | None):
        self._current_value = value

    @property
    def current_penalty(self) -> float | None:
        return self._current_penalty

    @current_penalty.setter
    def current_penalty(self, value: float | None):
        self._current_penalty = value

    @property
    def required_simulation_type(self) -> SimulationType:
        return self.parameter.simulation_type

    @property
    def required_analyses(self) -> tuple[Analysis, ...]:
        """The analyses whose results the goal reads."""
        return self.parameter.required_analyses

    def penalty(self, sim_result: SimulationResult) -> float:
        """Calculate the penalty for the given simulation result."""
        # Calculates the value of the parameter (e.g. S21_dB) from the simulation result
        return self._score(self.parameter.formula(sim_result, self.frequency_range))

    def evaluate(self, sim_results: Mapping[Analysis, SimulationResult]) -> float:
        """The penalty for a trial's results; :data:`FAILED_SIMULATION_PENALTY` without them."""
        try:
            values = self.parameter.values(sim_results, self.frequency_range)
        except UnavailableValueError:
            return self._fail()
        return self._score(values)

    def _score(self, values: np.ndarray | float) -> float:
        # Store the current value for later reference
        self.current_value = values

        # Calculate the penalty using the parameter's loss function and the goal's min/max values
        penalty_value = self.parameter.loss(self.min_value, self.max_value, values, self.scale) * self.weight
        self.current_penalty = penalty_value
        return penalty_value

    def _fail(self) -> float:
        self.current_value = None
        self.current_penalty = FAILED_SIMULATION_PENALTY
        return FAILED_SIMULATION_PENALTY

    def __str__(self):
        freq_range_str = self.frequency_range or "full range"
        return (
            f"DesignGoal(parameter={self.parameter.name}, "
            f"frequency_range={freq_range_str}, "
            f"min_value={self.min_value}, max_value={self.max_value}, "
            f"weight={self.weight}, scale={self.scale}, "
            f"current_penalty={self.current_penalty}"
        )

    @staticmethod
    def str_to_frequency_range(freq_range_str: str | None) -> tuple[float | None, float | None]:
        """
        Convert a frequency range string like "1-20GHz" into a tuple of floats (1e9, 20e9), and single values like "5GHz" into (5e9, 5e9).

        Returns:
        -------
        Tuple[Optional[float], Optional[float]]
            A tuple of (min_freq, max_freq) in Hz, or a single value if only one frequency is specified. Returns (None, None) if the input is None.
        """
        if freq_range_str is None:
            return (None, None)

        # Match patterns like "1-20GHz" or "5GHz"
        match = re.match(r"^\s*(?P<min>\d*\.?\d+)(?:-(?P<max>\d*\.?\d+))?\s*(?P<unit>[kKmMgG]?[hHz]*)\s*$", freq_range_str)
        if not match:
            raise ValueError(f"Invalid frequency range format: '{freq_range_str}'")

        min_freq = float(match.group("min"))
        max_freq = float(match.group("max")) if match.group("max") else None
        unit = match.group("unit").lower()

        # Convert to Hz based on unit
        multiplier = 1.0
        if unit.startswith("k"):
            multiplier = 1e3
        elif unit.startswith("m"):
            multiplier = 1e6
        elif unit.startswith("g"):
            multiplier = 1e9

        min_freq_hz = min_freq * multiplier
        max_freq_hz = max_freq * multiplier if max_freq is not None else None

        return (min_freq_hz, max_freq_hz) if max_freq_hz is not None else (min_freq_hz, min_freq_hz)


@dataclass(eq=False)
class ExpressionParameter(DesignParameter):
    """A parameter computed from other parameters, e.g. ``a - b`` for gain compression.

    Each variable of ``expression`` stands for one parameter in ``terms``, read at
    the frequency of the goal on this parameter. The terms may come from different
    analyses and may be expressions themselves. Build one with
    :func:`~cobra.optimizers.design_goal_collection.make_expression_parameter`.
    """

    expression: GoalExpression | None = None
    terms: dict[str, DesignParameter] = field(default_factory=dict)

    @property
    def required_analyses(self) -> tuple[Analysis, ...]:
        return tuple(
            dict.fromkeys(analysis for term in self.terms.values() for analysis in term.required_analyses)
        )

    def values(
        self, sim_results: Mapping[Analysis, SimulationResult], frequency_range: str | None = None
    ) -> np.ndarray:
        """The expression over the terms' values at *frequency_range*.

        Raises:
            UnavailableValueError: A term's analysis has no result, or the value is not finite.
        """
        if self.expression is None:
            raise TypeError(f"Expression parameter '{self.name}' has no expression")
        values = {
            variable: np.atleast_1d(np.asarray(term.values(sim_results, frequency_range), dtype=float))
            for variable, term in self.terms.items()
        }
        sizes = {array.size for array in values.values()} - {1}
        if len(sizes) > 1:
            raise ValueError(
                f"Expression '{self.name}': its terms have different numbers of values at "
                f"{frequency_range or 'the whole sweep'}; give the goal a single frequency"
            )
        result = np.atleast_1d(self.expression.evaluate(values))
        if not np.all(np.isfinite(result)):
            raise UnavailableValueError(f"Expression '{self.name}' is not finite: {result}")
        return result


# ---------------------------------------------------------------------------
# DesignGoalChecker — evaluates all goals against simulation results
# ---------------------------------------------------------------------------


class DesignGoalChecker:
    """Evaluates a list of DesignGoals against one or more simulation results."""

    def __init__(self, design_goals: list[DesignGoal]):
        # Group goals by the analysis they read; an expression goal joins its first term's.
        self.design_goals: dict[Analysis, list[DesignGoal]] = {}
        for goal in design_goals:
            self.design_goals.setdefault(goal.required_analyses[0], []).append(goal)

    def required_analyses(self) -> set[Analysis]:
        """Every analysis a goal reads, including those of every term of an expression goal."""
        return {
            analysis
            for goals in self.design_goals.values()
            for goal in goals
            for analysis in goal.required_analyses
        }

    def goals_reading(self, analysis: Analysis) -> list[DesignGoal]:
        """The goals that read *analysis*'s result."""
        return [
            goal
            for goals in self.design_goals.values()
            for goal in goals
            if analysis in goal.required_analyses
        ]

    def check_goals(self, context: OptimizationContext) -> OptimizationContext:
        """
        Evaluate all goals and update *context* with results.

        Reads ``context.simulation_results`` (``dict[Analysis, SimulationResult]``).
        """
        sim_results = context.simulation_results
        penalties = self.loss(sim_results)

        context.goal_achieved = bool(penalties) and all(p <= 0.0 for p in penalties)
        context.goals = [goal for goals in self.design_goals.values() for goal in goals]

        return context

    def loss(self, sim_results: Mapping[Analysis, SimulationResult]) -> list[float]:
        """
        Compute penalties for each goal using the appropriate network.

        A goal whose analysis is missing from *sim_results* — the simulation
        failed — receives :data:`FAILED_SIMULATION_PENALTY` instead of being
        skipped, so the optimizer learns to avoid those parameters.

        Parameters
        ----------
        sim_results:
            ``{analysis: SimulationResult}`` — one entry per analysis the goals read.
        """
        return [goal.evaluate(sim_results) for goals in self.design_goals.values() for goal in goals]

    def __str__(self):
        parts = [
            str(g) for goals in self.design_goals.values() for g in goals
        ]
        return "Design Goals: " + " | ".join(parts)
