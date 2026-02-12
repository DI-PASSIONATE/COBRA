from cobra.spice_sim.base_simulator import BaseSimulator
import subprocess
import PySpice
from typing import Dict


class NgspiceSimulator(BaseSimulator):
    """
    NgspiceSimulator - An implementation of the BaseSimulator using Ngspice for circuit simulation.
    """

    def __init__(self):
        super().__init__()
    
    def run_simulation(self, netlist_name: str, output_name: str) -> str:
        return output_name

