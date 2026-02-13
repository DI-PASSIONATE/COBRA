from typing import List, Dict, Optional
from cobra.optimizers import OptunaOptimizer
from cobra.optimizers.base_optimizer import BaseOptimizer
from cobra.optimizers.design_goal import DesignGoal, plot_rfic_transformer_metrics
from cobra.spice_sim.base_simulator import BaseSimulator
from cobra.spice_sim.xyce_simulator import XyceSimulator
from cobra.stages import (
    CircuitSimulationStage,
    EMSurrogateStage,
    OptimizerStage,
    EMFineTuningStage,
    COBRABaseStage,
)
import numpy as np
import matplotlib.pyplot as plt
from cobra.optimizers.design_goal import DesignGoalChecker
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

    def run(self, netlist: str, design_goals: list[DesignGoal], frequency_range: str, parameter_range: dict, max_iterations: int = 500) -> dict:
        """
        Predict the next set of parameters based on the given netlist, design goals, and current parameters.

        Parameters:
        - netlist: A string representation of the circuit netlist.
        - design_goals: A list of DesignGoal objects representing the design goals and constraints.
        - frequency_range: A string representing the frequency range of interest. Example: "110-130ghz" for 110 GHz to 130 GHz.
        - parameter_range: A dictionary of input parameters and their ranges.

        Returns:
        - The optimized parameters that meet the design goals.
        """
        design_goal_checker = DesignGoalChecker(design_goals, frequency_range=frequency_range)

        context = {
            "netlist": netlist,
            "design_goal_checker": design_goal_checker,
            "parameter_range": parameter_range,
            "output": None,
            "goal_achieved": False,
            "times": {
                "optimizer": 0.0,
                "em_surrogate": 0.0,
                "circuit_simulation": 0.0,
                "design_goal_checking": 0.0,
                "em_fine_tuning": 0.0,
                "total_time": 0.0,
            }
        }


        # Perform optimizer step
        for iteration in tqdm.tqdm(range(max_iterations), desc="COBRA Optimization Progress"):
            context["iteration"] = iteration + 1
            # Generate new parameters using the optimizer stage
            t1 = time.time()
            context = self.optimizer_stage.run(context)
            # Perform EM simulations / s parameter prediction using surrogate model from ORCA
            t2 = time.time()
            context = self.em_surrogate_stage.run(context)

            # Perform circuit-level simulation
            t3 = time.time()
            context = self.circuit_simulation_stage.run(context)

            # Tell the optimizer about the performance of the current parameters
            t4 = time.time()
            self.optimizer_stage.tell(context)

            # Check design goals
            t5 = time.time()
            context = design_goal_checker.check_goals(context)

            t6 = time.time()

            # Log times for each stage
            context["times"]["optimizer"] += t2 - t1
            context["times"]["em_surrogate"] += t3 - t2
            context["times"]["circuit_simulation"] += t4 - t3
            context["times"]["design_goal_checking"] += t5 - t4
            context["times"]["em_fine_tuning"] += t6 - t5
            context["times"]["total_time"] += t6 - t1

            # If design goals are achieved, break the loop
            if context["goal_achieved"]:
                print(f"Design goals achieved at iteration {iteration + 1}.")
                ntwk = context["network"]
                ntwk.plot_s_db()
                plot_rfic_transformer_metrics(ntwk)
                return context["parameters"]
        
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
            context = context["design_goal_checker"].check(context)

            # If design goals are achieved, break the loop
            if context["goal_achieved"]:
                print(f"Design goals achieved after EM fine-tuning at iteration {context['iteration']}.")
                break

        if not context["goal_achieved"]:
            print("EM fine-tuning completed without achieving design goals. Returning best parameters found.")

        return context
    
    def print_time(self, context: Dict):
        """
        Prints the percentage of time spent in each stage of the optimization process.
        """
        total_time = context["times"]["total_time"]
        if total_time == 0:
            return
        
        print(f"Time spent in Optimizer: {context['times']['optimizer'] / total_time * 100:.2f}%")
        print(f"Time spent in EM Surrogate: {context['times']['em_surrogate'] / total_time * 100:.2f}%")
        print(f"Time spent in Circuit Simulation: {context['times']['circuit_simulation'] / total_time * 100:.2f}%")
        print(f"Time spent in Design Goal Checking: {context['times']['design_goal_checking'] / total_time * 100:.2f}%")
        if self.em_fine_tuning_stage is not None:
            print(f"Time spent in EM Fine-Tuning: {context['times']['em_fine_tuning'] / total_time * 100:.2f}%")