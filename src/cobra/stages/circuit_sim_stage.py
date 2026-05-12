from typing import Dict, List
from cobra.spice_sim.base_simulator import BaseSimulator
from cobra.spice_sim.xyce_simulator import XyceSimulator
from cobra.stages.base_stage import COBRABaseStage
import skrf as rf
import matplotlib.pyplot as plt

class CircuitSimulationStage(COBRABaseStage):
    """
    Circuit Simulation Stage - This stage performs the circuit simulation using the provided simulator (e.g. Ngspice, Xyce, etc.).
    It takes the current design state, runs the circuit simulation, and updates the design state with the new simulation results.
    """

    def __init__(self, simulator: BaseSimulator = XyceSimulator("Xyce")):
        self.simulator = simulator

    def run(self, context: Dict) -> Dict:
        ntwk: List[rf.Network] = context["predicted_networks"]
        # Preprocess (e.g. vector fitting)
        #preprocessed_file = self.simulator.preprocess_ntwk(ntwk)
        for n in ntwk:
            self.simulator.preprocess_ntwk(n)

        # Output is a single network - even if we have multiple surrogate models
        new_ntwk = self.simulator.run_simulation(netlist_name=context["netlist"])
        context["simulated_network"] = new_ntwk

        return context
