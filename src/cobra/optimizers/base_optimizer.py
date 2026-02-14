from abc import ABC, abstractmethod
from typing import Any, Dict, Callable
import numpy as np

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
        pass

    @abstractmethod
    def step(self, context: Dict[str, Any], input_parameter_range: Dict[str, tuple | list | np.ndarray]) -> Dict:
        """
        Optimize the parameters based on the given input parameter range and constraints.

        Args:
            context: A dictionary containing the current design state, including the netlist, design goals, and any other relevant information.
            input_parameter_range: A dictionary of parameter names and their corresponding ranges.

        Returns:
            A dictionary of optimized parameters that meet the design goals and constraints.
        """

    @abstractmethod
    def tell(self, context, penalty: list[float] | float):
        """
        Provide feedback to the optimizer about the performance of the given parameters.

        Args:
            context: A dictionary containing the current design state, including the netlist, design goals, and any other relevant information.
            penalty: A list of penalty values corresponding to each design goal, indicating how well the current parameters meet the design goals. The optimizer can use this information to update its internal state and improve future parameter suggestions.
        """
        pass

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