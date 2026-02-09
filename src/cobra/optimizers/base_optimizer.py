from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """
    Abstract base class for optimizers in COBRA.
    Defines the interface that all optimizers must implement.
    """

    @abstractmethod
    def step(self, last_parameters: dict, last_metrics: dict) -> dict:
        """
        Perform an optimization step.

        Parameters:
        - last_parameters: A dictionary of the parameters used in the last evaluation.
        - last_metrics: A dictionary of the metrics obtained from the last evaluation.

        Returns:
        - A dictionary of new parameters to evaluate next.
        """
        pass
