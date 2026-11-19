import math
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from cobra.optimization_context import OptimizationContext


class OptimizationType(Enum):
    NETLIST_VARIABLE = "netlist_variable"
    MODEL_INPUT = "model_input"

@dataclass
class OptimizationProperty:
    name: str
    type: OptimizationType
    min_value: float
    max_value: float
    step: float | None = None
    unit: str | None = None
    linked_to: str | None = None
    initial_value: float | None = None
    """Value the optimizer evaluates first; ``None`` leaves it to the optimizer."""

def resolve_linked_master(
    param: OptimizationProperty, by_name: Mapping[str, OptimizationProperty]
) -> OptimizationProperty:
    """Follow *param*'s ``linked_to`` chain to the parameter that owns the value.

    A linked parameter is not sampled: it takes its value, and the range and unit
    that value is expressed in, from its master.
    """
    current = param
    seen = {param.name}
    while current.linked_to:
        target = by_name.get(current.linked_to)
        if target is None:
            raise ValueError(
                f"Parameter '{current.name}' links to unknown parameter '{current.linked_to}'."
            )
        if target.name in seen:
            raise ValueError(f"Circular link detected for parameter '{param.name}'.")
        seen.add(target.name)
        current = target
    return current


def netlist_unit(
    param: OptimizationProperty, by_name: Mapping[str, OptimizationProperty]
) -> str:
    """The unit suffix a netlist value for *param* has to carry.

    A linked parameter usually declares no unit of its own, so it inherits the
    master's. Getting this wrong is not a cosmetic error: writing ``19.0``
    instead of ``19.0F`` puts a value into the netlist that is off by fifteen
    orders of magnitude.
    """
    return param.unit or resolve_linked_master(param, by_name).unit or ""


_LEADING_NUMBER = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_netlist_value(value: Any) -> float:
    """The number in a netlist parameter value, dropping any unit suffix.

    The inverse of the formatting done with :func:`netlist_unit`: ``"42.3F"``
    becomes ``42.3``. Raises ``ValueError`` when *value* holds no leading
    number, which a caller that merely wants to display it should catch.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    match = _LEADING_NUMBER.match(str(value))
    if not match:
        raise ValueError(f"Cannot parse numeric value from '{value}'.")
    return float(match.group(1))


def narrowed_around(
    parameters: list[OptimizationProperty],
    values: Mapping[str, Any],
    fraction: float,
    min_steps: int = 0,
) -> list[OptimizationProperty]:
    """The netlist-variable parameters of *parameters*, with ranges narrowed around *values*.

    Each range becomes the current value ± ``max(fraction * |value|, min_steps * step)``,
    or ± *fraction* of the whole range for a value of 0 without a step, clipped to the
    original range and kept on its grid, and the value becomes the start point. So
    ``fraction=0.3, min_steps=2`` gives ±30 % for a capacitance and ±2 for a
    transistor count ``nx``. Model inputs are left out, and so are netlist variables
    linked to one: the geometry stays as it is. A linked netlist variable keeps
    following its master.
    """
    by_name = {parameter.name: parameter for parameter in parameters}
    narrowed: list[OptimizationProperty] = []
    for parameter in parameters:
        if parameter.type is not OptimizationType.NETLIST_VARIABLE:
            continue
        if resolve_linked_master(parameter, by_name).type is not OptimizationType.NETLIST_VARIABLE:
            continue
        if parameter.linked_to:
            narrowed.append(replace(parameter, initial_value=None))
            continue
        value = parse_netlist_value(values[parameter.name])
        step = parameter.step or 0.0
        half_width = max(fraction * abs(value), min_steps * step) or fraction * (
            parameter.max_value - parameter.min_value
        )
        low = max(parameter.min_value, value - half_width)
        high = min(parameter.max_value, value + half_width)
        if step:
            # On the original grid, which the value lies on, counted from the original minimum.
            low = parameter.min_value + math.ceil((low - parameter.min_value) / step - 1e-9) * step
            high = parameter.min_value + math.floor((high - parameter.min_value) / step + 1e-9) * step
        narrowed.append(
            replace(parameter, min_value=round(low, 12), max_value=round(high, 12), initial_value=value)
        )
    return narrowed


class BaseOptimizer(ABC):
    """
    Abstract base class for optimizers in COBRA.
    Defines the interface that all optimizers must implement.
    """

    supports_parallel_trials: ClassVar[bool] = False
    """Whether :meth:`step` may be called again before the previous :meth:`tell`.

    Optimizers that need the result of one step to compute the next (e.g.
    gradient descent) leave this ``False``, and COBRA refuses to run them with
    more than one trial in flight.
    """

    def __init__(self, multi_objective: bool = False):
        """
        Initialize the optimizer.

        Args:
            multi_objective: A boolean indicating whether the optimizer should handle multiple objectives (design goals) simultaneously. If False, the optimizer will produce a single loss value by aggregating the losses from multiple design goals.
        """
        self.multi_objective = multi_objective

    @abstractmethod
    def initialize(self, num_goals: int, parallel_trials: int = 1):
        """
        Initialize the optimizer with any necessary state or parameters.

        Args:
            num_goals: The number of design goals that the optimizer will be optimizing for.
            parallel_trials: How many trials COBRA will keep in flight at once, so
                the optimizer can adapt its sampling to outstanding trials.
        """

    @abstractmethod
    def step(self, context: "OptimizationContext", model_input_ranges: list[OptimizationProperty], netlist_property_ranges: list[OptimizationProperty]) -> None:
        """
        Optimize the parameters based on the given input parameter range and constraints.

        Args:
            context: A dictionary containing the current design state, including the netlist, design goals, and any other relevant information.
            model_input_ranges: A list of OptimizationProperty objects representing the parameters to be optimized
            netlist_property_ranges: A list of OptimizationProperty objects representing the netlist parameters to be optimized
        """

    @abstractmethod
    def tell(self, context: "OptimizationContext", penalty: list[float] | float):
        """
        Provide feedback to the optimizer about the performance of the given parameters.

        Args:
            context: A dictionary containing the current design state, including the netlist, design goals, and any other relevant information.
            penalty: A list of penalty values corresponding to each design goal, indicating how well the current parameters meet the design goals. The optimizer can use this information to update its internal state and improve future parameter suggestions.
        """

    @abstractmethod
    def get_moo_results(self) -> Any:
        """
        Get the results of the multi-objective optimization process. This method should return a
        list (i.e. a pareto front) of the best trials run.

        Returns:
            A list of the best trials from the multi-objective optimization process, representing the Pareto front of optimal solutions.
        """

    @abstractmethod
    def get_best_parameters(self) -> dict[str, float]:
        """
        Get the best parameters found by the optimizer. This method should return a dictionary of parameter names and their corresponding optimized values.

        Returns:
            A dictionary containing the best parameters found by the optimizer, where the keys are parameter names and the values are the optimized parameter values.
        """

    def reject(self, context: "OptimizationContext", loss: list[float]):
        """
        Tell the optimizer that the parameters in *context* could not be evaluated.

        COBRA calls this instead of :meth:`tell` for a geometry the surrogate cannot
        build, and does not count the trial as an iteration. The default reports
        *loss* (the failed-simulation penalty) like an ordinary result, so an
        optimizer that needs an answer to every step moves on from it.
        """
        self._tell(context, loss)

    def _tell(self, context: "OptimizationContext", loss: list[float]):
        """
        Internal method that converts the list of loss values into a single penalty value if multi_objective is False, and then calls the tell method with the appropriate penalty.

        Args:
            context: A dictionary containing the current design state, including the netlist, design goals, and any other relevant information.
            loss: A list of loss values corresponding to each design goal, indicating how well the current parameters meet the design goals.
        """
        if self.multi_objective:
            return self.tell(context, loss)
        return self.tell(context, aggregate_penalty(loss))


def aggregate_penalty(loss: list[float]) -> float:
    """Combine per-goal losses into the single penalty a single-objective optimizer minimises.

    A result that meets every goal therefore always scores lower than one that does not.
    """
    # All values are above or all below zero -> sum them up for a single loss value
    # If some values are above and some below zero, sum the positive values and disregard the negative values
    if all(value >= 0 for value in loss) or all(value <= 0 for value in loss):
        return sum(loss)
    return sum(value for value in loss if value > 0)
