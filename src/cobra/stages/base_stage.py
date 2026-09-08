from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cobra.optimization_context import OptimizationContext


class COBRABaseStage(ABC):
    """
    Abstract base class for all stages in the COBRA optimization flow.
    Each stage must implement the `run` method, which takes the current state of the design
    and returns an updated state after processing.
    """

    @abstractmethod
    def run(self, context: "OptimizationContext") -> "OptimizationContext":
        """
        Process the given design state and return an updated state.

        Parameters:
            context: The current state of the design.

        Returns:
            The same context, updated with the fields this stage owns.
        """
