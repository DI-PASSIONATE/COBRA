from abc import ABC, abstractmethod
import math
import skrf as rf

class BaseSimulator(ABC):
    @abstractmethod
    def run_simulation(self, netlist_name: str) -> rf.Network:
        """
        Run the circuit simulation using the provided netlist and output name.
        Args:
            netlist_name (str): The name of the netlist file to be simulated.
            output_name (str): The name of the output file where simulation results will be stored.
        Returns:
            str: The path to the output file containing the simulation results (e.g., a Touchstone file).
        """
        pass

    @abstractmethod
    def preprocess_ntwk(self, ntwk) -> str:
        """
        Preprocess the network by performing some operations (e.g., vector fitting) 
        that the simulator requires before running the simulation. 
        Args:
            ntwk: The network object containing the S-parameters and frequency information.
        Returns:
            A file path to the preprocessed network data (e.g., a SPICE subcircuit file) that can be included in the netlist for circuit simulation.
        """
        pass

    def equivalent_RCL(self, Z, f):
        # returns RLC values for a specific impedance
        # returns large inductor instead of negative cap values, and cap becomes zero
        w = 2 * math.pi * f
    
        # Parallel form
        Y = 1 / Z
        G = Y.real
        B = Y.imag
    
        Rpar = round(1 / G,2) if G != 0 else None
    
        if B > 0:
            Cpar_fF = round((B/w)*1e15,2)
            Cpar = Cpar_fF/1e15
            Lpar = 1e3 # verryy large to have no impact for Rf-freq
        elif B < 0:
            Lpar_pH = round(-1 / (w * B)*1e12,2)
            Lpar = Lpar_pH/1e12
            Cpar = 0 # keine Cap da
        else:
            Lpar = 1e3 # verryy large to have no impact for Rf-freq
            Cpar = 0 # keine Cap da
    
        return (Rpar, Cpar, Lpar) 