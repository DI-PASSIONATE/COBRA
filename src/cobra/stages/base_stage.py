from abc import ABC, abstractmethod


class COBRABaseStage(ABC):
    """
    Abstract base class for all stages in the COBRA optimization flow.
    Each stage must implement the `run` method, which takes the current state of the design
    and returns an updated state after processing.
    """

    @abstractmethod
    def run(self, context):
        """
        Process the given design state and return an updated state.

        Parameters:
            context (dict): A dictionary representing the current state of the design.
        Returns:
            dict: An updated state of the design after processing.
        """
