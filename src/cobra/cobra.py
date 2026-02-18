from typing import List, Dict, Optional
import json
from cobra.optimizers import OptunaOptimizer
from cobra.optimizers.base_optimizer import BaseOptimizer, OptimizationProperty, OptimizationType
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

    def run(self, netlist: str, design_goals: list[DesignGoal], frequency_range: str, optimization_parameters: list[OptimizationProperty], max_iterations: int = 500, orca_geometry=None) -> dict:
        """
        Predict the next set of parameters based on the given netlist, design goals, and current parameters.

        Parameters:
        - netlist: A path to the netlist
        - design_goals: A list of DesignGoal objects representing the design goals and constraints.
        - frequency_range: A string representing the frequency range of interest. Example: "110-130ghz" for 110 GHz to 130 GHz.
        - optimization_parameters: A list of OptimizationProperty objects representing the parameters to be optimized, their types, and their ranges.
        - max_iterations: The maximum number of optimization iterations to perform.

        Returns:
        - The optimized parameters that meet the design goals.
        """
        design_goal_checker = DesignGoalChecker(design_goals, frequency_range=frequency_range)
        self.optimizer_stage.optimizer.initialize(len(design_goals))
        netlist_parser = self.circuit_simulation_stage.simulator.netlist_parser.from_file(netlist)

        context = {
            "netlist": netlist,
            "design_goal_checker": design_goal_checker,
            "optimization_parameters": optimization_parameters,
            "output": None,
            "goal_achieved": False,
            "iterations": [],
            "orca_geometry": orca_geometry,
            "times": {
                "optimizer": 0.0,
                "em_surrogate": 0.0,
                "circuit_simulation": 0.0,
                "design_goal_checking": 0.0,
                "em_fine_tuning": 0.0,
                "total_time": 0.0,
            }
        }

        # Backup netlist before optimization
        with open(netlist, "r") as f:
            original_netlist_content = f.read()
        with open(f"{netlist}.backup", "w") as f:
            f.write(original_netlist_content)

        # Perform optimizer step
        for iteration in tqdm.tqdm(range(max_iterations), desc="COBRA Optimization Progress"):
            context["iteration"] = iteration + 1
            # Generate new parameters using the optimizer stage
            t1 = time.time()
            context = self.optimizer_stage.run(context)

            # Update netlist with possibly new netlist parameters from the optimizer
            params = context["netlist_parameters"]
            netlist_parser.update_parameters(params)
            netlist_parser.save(netlist)  # Save the updated netlist back to disk for the circuit simulator to use

            # Perform EM simulations / s parameter prediction using surrogate model from ORCA
            t2 = time.time()
            context = self.em_surrogate_stage.run(context)

            # Perform circuit-level simulation
            t3 = time.time()
            context = self.circuit_simulation_stage.run(context)

            # Check design goals
            t4 = time.time()
            context = design_goal_checker.check_goals(context)

            t5 = time.time()

            # Log times for each stage
            context["times"]["optimizer"] += t2 - t1
            context["times"]["em_surrogate"] += t3 - t2
            context["times"]["circuit_simulation"] += t4 - t3
            context["times"]["design_goal_checking"] += t5 - t4
            context["times"]["total_time"] += t5 - t1

            # If design goals are achieved, break the loop
            if context["goal_achieved"]:
                break

            self.optimizer_stage.tell(context)
                
        
        ntwk = context["predicted_network"]
        ntwk.write_touchstone(f"surrogate_s_params.s{ntwk.nports}p")
        
        if not context["goal_achieved"]:
            print("Maximum iterations reached without achieving design goals.")
        else:
            print(f"Design goals achieved at iteration {context['iteration']}.")

        if self.em_fine_tuning_stage is not None:
            context = self.fine_tuning(context)

        # Save final context to a JSON file for analysis            
        with open("cobra_optimization_context.json", "w") as f:
            # Save context as txt file
            json.dump(context, f, indent=4, default=str)

        # Plot final results
        print("Plotting final results...")
        ntwk = context["predicted_network"]
        ntwk2 = context["simulated_network"]
        
        plot_rfic_transformer_metrics(ntwk)
        ntwk2.name = "Simulated Network"
        ntwk2.plot_s_db()
        plt.show()
        return context
        

    def fine_tuning(self, context: Dict) -> Dict:
        """ Perform EM fine-tuning using the EM fine-tuning stage. """
        if self.em_fine_tuning_stage is None:
            raise ValueError("EM fine-tuning stage is not defined. Cannot perform fine-tuning.")

        design_goal_checker: DesignGoalChecker = context["design_goal_checker"]

        for iteration in tqdm.tqdm(range(3), desc="COBRA EM Fine-Tuning Progress"):
            context["iteration"] = iteration + 1
            # Perform EM simulation INSTEAD OF surrogate model prediction
            context = self.em_fine_tuning_stage.run(context, orca_geometry=context.get("orca_geometry", None))

            # Perform circuit-level simulation
            context = self.circuit_simulation_stage.run(context)

            # Check design goals
            context = design_goal_checker.check_goals(context)

            # If design goals are achieved, break the loop and don't tell the optimizer about the results → fine-tune
            if context["goal_achieved"]:
                print(f"Design goals achieved after EM fine-tuning at iteration {iteration}.")
                break
            else:
                print(f"Design goals not achieved after EM fine-tuning at iteration {iteration}. Continuing optimization...")

            self.optimizer_stage.tell(context)
            context = self.optimizer_stage.run(context)

        if not context["goal_achieved"]:
            print("EM fine-tuning completed without achieving design goals. Returning best parameters found.")
        else:
            print(f"Design goals achieved and geometry verified with EM simulation. Returning optimized parameters.")

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