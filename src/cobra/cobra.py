from typing import List, Dict, Optional
from cobra.optimizers import OptunaOptimizer
from cobra.optimizers.base_optimizer import BaseOptimizer
from cobra.spice_sim.base_simulator import BaseSimulator
from cobra.spice_sim.xyce_simulator import XyceSimulator
from cobra.stages import (
    CircuitSimulationStage,
    EMSurrogateStage,
    OptimizerStage,
    EMFineTuningStage,
    COBRABaseStage,
)
from cobra.optimizers.check_design_goals import check_design_goals
import tqdm, time


class COBRA:
    """
    COBRA - A Circuit-Level Open-Source Based RFIC AI-Assisted Optimizer
    """

    def __init__(
        self,
        em_surrogate_model: str,
        optimizer: BaseOptimizer = OptunaOptimizer(),
        circuit_simulator: BaseSimulator = XyceSimulator(),
        palace_fine_tuning_command: Optional[str] = None,
    ):
        self.optimizer_stage = OptimizerStage(optimizer)
        self.em_surrogate_stage = EMSurrogateStage(em_surrogate_model)
        self.circuit_simulation_stage = CircuitSimulationStage(circuit_simulator)
        self.em_fine_tuning_stage = EMFineTuningStage(palace_fine_tuning_command) if palace_fine_tuning_command else None

    def run(self, netlist: str, design_goals: dict, parameter_range: dict, max_iterations: int = 500) -> dict:
        """
        Predict the next set of parameters based on the given netlist, design goals, and current parameters.

        Parameters:
        - netlist: A string representation of the circuit netlist.
        - design_goals: A dictionary of design goals and constraints.
        - parameter_range: A dictionary of input parameters and their ranges.

        Returns:
        - The optimized parameters that meet the design goals.
        """
        context = {
            "netlist": netlist,
            "design_goals": design_goals,
            "parameter_range": parameter_range,
            "output": None,
            "goal_achieved": False,
        }

        # Perform optimizer step
        for iteration in tqdm.tqdm(range(max_iterations), desc="COBRA Optimization Progress"):
            context["iteration"] = iteration + 1
            # Generate new parameters using the optimizer stage
            context = self.optimizer_stage.run(context)

            # Perform EM simulations / s parameter prediction using surrogate model from ORCA
            context = self.em_surrogate_stage.run(context)

            # Perform circuit-level simulation
            context = self.circuit_simulation_stage.run(context)

            # Tell the optimizer about the performance of the current parameters
            self.optimizer_stage.tell(context)

            # Check design goals
            context = check_design_goals(context)

            # If design goals are achieved, break the loop
            if context["goal_achieved"]:
                print(f"Design goals achieved at iteration {iteration + 1}.")
                break

            time.sleep(0.1)  # Simulate time taken for each iteration
        
        if self.em_fine_tuning_stage is None:
            if context["iteration"] == max_iterations:
                print("Maximum iterations reached without achieving design goals. Returning best parameters found.")
            return context
        else:
            return self.fine_tuning(context)
        

    def fine_tuning(self, context: Dict) -> Dict:
        """ Perform EM fine-tuning using the EM fine-tuning stage. """
        if self.em_fine_tuning_stage is None:
            raise ValueError("EM fine-tuning stage is not defined. Cannot perform fine-tuning.")
        
        context = self.em_fine_tuning_stage.run(context)

        if context["goal_achieved"]:
            print("Design goals ensured via EM fine-tuning. Returning optimized parameters.")
            return context

        #### Optional EM fine-tuning stage
        for iteration in tqdm.tqdm(range(3), desc="COBRA EM Fine-Tuning Progress"):

            # Perform EM fine-tuning using the fine-tuning stage
            context = self.em_fine_tuning_stage.run(context)

            # Check design goals again after fine-tuning
            context = check_design_goals(context)

            # If design goals are achieved, break the loop
            if context["goal_achieved"]:
                print(f"Design goals achieved after EM fine-tuning at iteration {context['iteration']}.")
                break

        if not context["goal_achieved"]:
            print("EM fine-tuning completed without achieving design goals. Returning best parameters found.")

        return context