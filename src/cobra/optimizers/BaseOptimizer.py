from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    """
    Abstract base class for optimizers in COBRA.
    Defines the interface that all optimizers must implement.
    """

    @abstractmethod
    def optimize(self, design, constraints):
        """
        Optimize the given design under the specified constraints.

        Parameters:
        - design: The circuit design to be optimized.
        - constraints: The constraints to be satisfied during optimization.

        Returns:
        - An optimized design that meets the constraints.
        """
        pass