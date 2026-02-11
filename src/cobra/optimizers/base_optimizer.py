from abc import ABC, abstractmethod
from typing import Any, Dict, Callable
import numpy as np

class BaseOptimizer(ABC):
    """
    Abstract base class for optimizers in COBRA.
    Defines the interface that all optimizers must implement.
    """

    @abstractmethod
    def step(self, context: Dict[str, Any], input_parameter_range: Dict[str, tuple | list | np.ndarray], constraints: Dict) -> Dict:
        """
        Optimize the parameters based on the given input parameter range and constraints.

        Parameters:
        - input_parameter_range: A dictionary of parameter names and their corresponding ranges.
        - constraints: A dictionary of design goals and constraints that must be satisfied. Not sure yet how to structure this.

        Returns:
        - A dictionary of optimized parameters that meet the design goals and constraints.
        """

    @abstractmethod
    def tell(self, context, loss):
        """
        Provide feedback to the optimizer about the performance of the given parameters.

        Parameters:
        - parameters: A dictionary of parameter names and their corresponding values that were evaluated.
        - metrics: A dictionary of performance metrics resulting from evaluating the given parameters.
        """
        pass