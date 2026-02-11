from abc import ABC, abstractmethod

class BaseSimulator(ABC):
    @abstractmethod
    def run_simulation(self, netlist_name, output_name) -> str:
        """
        Run the circuit simulation using the provided netlist and output name.
        Args:
            netlist_name (str): The name of the netlist file to be simulated.
            output_name (str): The name of the output file where simulation results will be stored.
        Returns:
            str: The path to the output file containing the simulation results (e.g., a Touchstone file).
        """
        pass