from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """
    Abstract base class for optimizers in COBRA.
    Defines the interface that all optimizers must implement.
    """
    @abstractmethod
    def initialize(self):
        """
        Initialize any necessary state for the optimizer.
        """
        pass

    @abstractmethod
    def step(self, input_parameter_range: dict, constraints: dict) -> dict:
        """
        Optimize the parameters based on the given input parameter range and constraints.

        Parameters:
        - input_parameter_range: A dictionary of parameter names and their corresponding ranges.
        - constraints: A dictionary of design goals and constraints that must be satisfied. Not sure yet how to structure this.

        Returns:
        - A dictionary of optimized parameters that meet the design goals and constraints.
        """

    @abstractmethod
    def tell(self, parameters: dict, metrics: dict):
        """
        Provide feedback to the optimizer about the performance of the given parameters.

        Parameters:
        - parameters: A dictionary of parameter names and their corresponding values that were evaluated.
        - metrics: A dictionary of performance metrics resulting from evaluating the given parameters.
        """
        pass