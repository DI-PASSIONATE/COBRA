from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


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

class BaseOptimizer(ABC):
    """
    Abstract base class for optimizers in COBRA.
    Defines the interface that all optimizers must implement.
    """
    def __init__(self, multi_objective: bool = False):
        """
        Initialize the optimizer.

        Args:
            multi_objective: A boolean indicating whether the optimizer should handle multiple objectives (design goals) simultaneously. If False, the optimizer will produce a single loss value by aggregating the losses from multiple design goals.
        """
        self.multi_objective = multi_objective
        
    @abstractmethod
    def initialize(self, num_goals: int):
        """
        Initialize the optimizer with any necessary state or parameters.

        Args:
            num_goals: The number of design goals that the optimizer will be optimizing for.
        """

    @abstractmethod
    def step(self, context: dict[str, Any], model_input_ranges: list[OptimizationProperty], netlist_property_ranges: list[OptimizationProperty]) -> None:
        """
        Optimize the parameters based on the given input parameter range and constraints.

        Args:
            context: A dictionary containing the current design state, including the netlist, design goals, and any other relevant information.
            model_input_ranges: A list of OptimizationProperty objects representing the parameters to be optimized
            netlist_property_ranges: A list of OptimizationProperty objects representing the netlist parameters to be optimized
        """

    @abstractmethod
    def tell(self, context, penalty: list[float] | float):
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

    def _tell(self, context, loss: list[float]):
        """
        Internal method that converts the list of loss values into a single penalty value if multi_objective is False, and then calls the tell method with the appropriate penalty.
        
        Args:
            context: A dictionary containing the current design state, including the netlist, design goals, and any other relevant information.
            loss: A list of loss values corresponding to each design goal, indicating how well the current parameters meet the design goals.
        """
        if self.multi_objective:
            return self.tell(context, loss)
        
        # All values are above or all below zero -> sum them up for a single loss value
        # If some values are above and some below zero, sum the positive values and disregard the negative values
        if all(l >= 0 for l in loss) or all(l <= 0 for l in loss):
            penalty = sum(loss)
        else:
            penalty = sum(l for l in loss if l > 0)
        return self.tell(context, penalty)