import json
import shutil
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import skrf as rf
import tqdm

from cobra.optimizers import OptunaOptimizer
from cobra.optimizers.base_optimizer import (
    BaseOptimizer,
    OptimizationProperty,
    OptimizationType,
)
from cobra.optimizers.design_goal import DesignGoal, DesignGoalChecker
from cobra.setting import CobraSetting
from cobra.spice_sim.base_simulator import BaseSimulator
from cobra.spice_sim.netlist_parsers.netlist_parser import BaseNetlistParser
from cobra.spice_sim.xyce_simulator import XyceSimulator
from cobra.stages import (
    CircuitSimulationStage,
    EMFineTuningStage,
    EMSurrogateStage,
    OptimizerStage,
)

if TYPE_CHECKING:
    from cobra.configuration import RunConfiguration


def _sanitize_for_json(obj):
    """Recursively convert a context dict to a JSON-safe structure.

    Enum keys/values become their ``.value``; anything else that is not a
    native JSON type is left for ``json.dump``'s ``default=str`` fallback.
    """
    if isinstance(obj, dict):
        return {
            (k.value if isinstance(k, Enum) else k): _sanitize_for_json(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    return obj


class COBRA:
    """
    COBRA - A Circuit-Level Open-Source Based RFIC AI-Assisted Optimizer
    
    Can be initialized with:
    Pass a NetlistParser and component_onnx_mapping dict
       >>> parser = XyceNetlistParser().from_file("netlist.cir")
       >>> cobra = COBRA(netlist_parser=parser, 
       ...               component_onnx_mapping={"X1": "model.onnx"})
    """

    # Settings for run() and __init__ parameters exposed in the GUI.
    _settings = [
        CobraSetting(
            name="max_iterations",
            dtype=int,
            default=500,
            description=(
                "Maximum number of surrogate-model optimisation iterations.\n"
                "The loop exits early once all design goals are satisfied."
            ),
        ),
        CobraSetting(
            name="fine_tuning_iterations",
            dtype=int,
            default=3,
            description=(
                "Number of EM fine-tuning iterations performed with Palace after the\n"
                "surrogate optimisation loop. Only used when fine-tuning is enabled."
            ),
        ),
        CobraSetting(
            name="palace_fine_tuning_command",
            dtype=str,
            default="palace",
            description=(
                "Shell command used to invoke Palace for EM fine-tuning.\n"
                "Can be a plain command name (if on PATH) or an absolute executable path."
            ),
        ),
    ]

    def __init__(
        self,
        netlist_parser: BaseNetlistParser,
        component_onnx_mapping: dict[str, str],
        optimizer: BaseOptimizer = OptunaOptimizer(),
        circuit_simulator: BaseSimulator = XyceSimulator(),
        palace_fine_tuning_command: str | None = None,
        fine_tuning_iterations: int = 3,
        fine_tuning_optimizer: BaseOptimizer | str | None = "reuse",
    ):
        # Validate initialization arguments
        if not isinstance(netlist_parser, BaseNetlistParser):
            raise TypeError("netlist_parser must be an instance of BaseNetlistParser")
        
        components = netlist_parser.components
        if component_onnx_mapping is None:
            component_onnx_mapping = {}

        # Validate that all components that have models are present in the netlist
        missing_components = set(component_onnx_mapping.keys()) - set(components.keys())
        if missing_components:
            raise ValueError(
                f"component_onnx_mapping references unknown components: {missing_components}. "
                f"Components found in netlist: {set(components.keys())}"
            )

        self.netlist_parser = netlist_parser
        self.component_onnx_mapping = component_onnx_mapping

        # Only create a surrogate stage when there are components with models
        if components and component_onnx_mapping:
            self.em_surrogate_stage = EMSurrogateStage(
                em_surrogate_model=[component_onnx_mapping[comp] for comp in components if comp in component_onnx_mapping],
                component_names=[comp for comp in components if comp in component_onnx_mapping]
            )
        else:
            self.em_surrogate_stage = None
        
        self.optimizer_stage = OptimizerStage(optimizer)
        self.circuit_simulation_stage = CircuitSimulationStage(circuit_simulator)
        self.em_fine_tuning_stage = EMFineTuningStage(palace_fine_tuning_command) if palace_fine_tuning_command else None
        self.fine_tuning_iterations = fine_tuning_iterations
        self.fine_tuning_optimizer = fine_tuning_optimizer

    def _build_fine_tuning_optimizer_stage(self) -> OptimizerStage:
        fine_tuning_optimizer = self.fine_tuning_optimizer

        if fine_tuning_optimizer is None:
            return self.optimizer_stage

        if isinstance(fine_tuning_optimizer, BaseOptimizer):
            return OptimizerStage(fine_tuning_optimizer)

        if isinstance(fine_tuning_optimizer, str):
            normalized_mode = fine_tuning_optimizer.replace("-", "_").strip().lower()
            if normalized_mode in {"reuse", "same", "continue", "surrogate", "surrogate_optimizer"}:
                return self.optimizer_stage
            if normalized_mode in {"gradient_descent", "gradientdescent", "gd"}:
                from cobra.optimizers import GradientDescentOptimizer

                return OptimizerStage(GradientDescentOptimizer())

        raise ValueError(
            "Unsupported fine-tuning optimizer. Choose 'reuse' or 'gradient_descent', or pass a BaseOptimizer instance."
        )

    def run(self, netlist: str, design_goals: list[DesignGoal], optimization_parameters: list[OptimizationProperty], max_iterations: int = 500, orca_geometries: dict | None = None, callback=None, results_name: str | None = None, sim_params_by_type: dict | None = None, run_configuration: Optional["RunConfiguration"] = None) -> dict:
        """
        Run the optimization workflow.

        Parameters:
        - netlist: A path to the netlist file. If a netlist_parser was provided in __init__, 
                   it should correspond to this file.
        - design_goals: A list of DesignGoal objects representing the design goals and constraints.
        - optimization_parameters: A list of OptimizationProperty objects representing the parameters 
                                   to be optimized, their types, and their ranges.
        - max_iterations: The maximum number of optimization iterations to perform.
        - orca_geometries: Optional dict mapping component names to ORCA geometry objects for EM
                           fine-tuning. Only required for ONNX-based (non-Touchstone) components.
        - callback: An optional callback function that takes the current context as an argument. 
                    If the callback returns False, the optimization is stopped.
        - results_name: Optional name for the results folder. If not provided, derives from netlist filename.

        Returns:
        - The optimized parameters that meet the design goals.
        """
        # Create results folder with timestamp and name
        if results_name is None:
            results_name = Path(netlist).stem
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        results_dir = Path("results") / f"{timestamp}_{results_name}"
        results_dir.mkdir(parents=True, exist_ok=True)

        if run_configuration is not None:
            run_configuration.save(results_dir / "cobra_config.json")
        
        # Copy original netlist to results directory
        original_netlist_path = Path(netlist)
        netlist_in_results = results_dir / original_netlist_path.name
        shutil.copy(netlist, netlist_in_results)
        
        # Update netlist path to point to the results directory for all operations
        netlist = str(netlist_in_results)
        
        netlist_parser = self.netlist_parser
        
        # Replace the component model names in the netlist to match the vector fitted subcircuits
        for comp_name in self.component_onnx_mapping.keys():
            try:
                netlist_parser.set_model(comp_name, f"{comp_name}_subct")
            except Exception as e:
                print(f"Warning: Could not set subcircuit model for {comp_name}: {e}")
        netlist_parser.save(netlist)
        
        design_goal_checker = DesignGoalChecker(design_goals)
        self.optimizer_stage.optimizer.initialize(len(design_goals))

        context = {
            "netlist": netlist,
            "native_sim_type": netlist_parser.simulation_type,
            "design_goal_checker": design_goal_checker,
            "optimization_parameters": optimization_parameters,
            "goal_achieved": False,
            "max_iterations": max_iterations,
            "iterations": [],
            "orca_geometries": orca_geometries or {},
            "results_dir": str(results_dir),
            "sim_params_by_type": sim_params_by_type or {},
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
        pbar = tqdm.tqdm(total=context["max_iterations"], desc="COBRA Optimization Progress")
        
        iteration = 0
        while iteration < context["max_iterations"]:
            iteration += 1
            context["iteration"] = iteration
            # Generate new parameters using the optimizer stage
            t1 = time.time()
            context = self.optimizer_stage.run(context)

            # Update netlist with possibly new netlist parameters from the optimizer
            params = context["netlist_parameters"]
            netlist_parser.update_parameters(params)
            netlist_parser.save(netlist)  # Save the updated netlist back to disk for the circuit simulator to use

            # Perform EM simulations / s parameter prediction using surrogate model from ORCA
            t2 = time.time()
            if self.em_surrogate_stage is not None:
                context = self.em_surrogate_stage.run(context)
            else:
                context.setdefault("predicted_networks", [])

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

            # Callback
            if callback:
                should_continue = callback(context)
                if should_continue is False:
                    print("Optimization stopped by callback.")
                    break
            
            pbar.update(1)
            if pbar.total != context["max_iterations"]:
                pbar.total = context["max_iterations"]
                pbar.refresh()

            # Tells the optimizer about the current state and saves it to the context for logging
            self.optimizer_stage.tell(context)

            # If design goals are achieved, break the loop
            if context["goal_achieved"]:
                break

            
        pbar.close()
        
        # If goals not achieved, try to retrieve best parameters from optimizer and use those for final context
        if not context["goal_achieved"]:
            context = self.re_run_best_parameters(netlist, optimization_parameters, design_goal_checker, netlist_parser, context)
        else:
            print(f"Design goals achieved at iteration {context['iteration']}.")
        
        # Save the surrogate model's predicted S-parameters to the results directory for the user
        ntwks: list[rf.Network] = context.get("predicted_networks", [])
        for i, ntwk in enumerate(ntwks):
            name_suffix = f"_{ntwk.name}" if ntwk.name else f"_{i+1}"
            surrogate_file = results_dir / f"surrogate_s_params{name_suffix}.s{ntwk.nports}p"
            ntwk.write_touchstone(str(surrogate_file))

        if self.em_fine_tuning_stage is not None:
            context["fine_tuning_active"] = True
            context["fine_tuning_iteration"] = 0
            context["fine_tuning_total"] = self.fine_tuning_iterations
            if context.get("goal_achieved"):
                context["fine_tuning_start_iteration"] = context.get("iteration", 0)

            if callback:
                should_continue = callback(context)
                if should_continue is False:
                    print("Fine Tuning stopped by callback.")
                    return context

            context = self.fine_tuning(context, callback)

        # Save final context to a JSON file for analysis
        context_file = results_dir / "cobra_optimization_context.json"
        with open(context_file, "w") as f:
            json.dump(_sanitize_for_json(context), f, indent=4, default=str)

        print(f"\nAll results saved to: {results_dir}")

        return context

    def re_run_best_parameters(self, netlist, optimization_parameters, design_goal_checker, netlist_parser, context):
        print("Maximum iterations reached without achieving design goals.")
            
            # If not MOO, retrieve best parameters and update context to reflect them
        if not self.optimizer_stage.optimizer.multi_objective:
            best_params_flat = self.optimizer_stage.optimizer.get_best_parameters()

            # Split best_params_flat into model parameters and netlist parameters based on optimization_parameters list
            model_params = {}
            netlist_params = {}
                
            for prop in optimization_parameters:
                if prop.name in best_params_flat:
                    val = best_params_flat[prop.name]
                    if prop.type == OptimizationType.NETLIST_VARIABLE:
                        unit = prop.unit if prop.unit else ""
                        netlist_params[prop.name] = f"{val}{unit}"
                    elif prop.type == OptimizationType.MODEL_INPUT:
                        model_params[prop.name] = val
                
            context["netlist_parameters"] = netlist_params
            context["model_parameters"] = model_params
                
            # Update netlist
            netlist_parser.update_parameters(netlist_params)
            netlist_parser.save(netlist)
                
            # Rerun simulation to update context with best result
            print("Re-simulating with best parameters...")
            if self.em_surrogate_stage is not None:
                context = self.em_surrogate_stage.run(context)
            context = self.circuit_simulation_stage.run(context)
            context = design_goal_checker.check_goals(context)
        
        return context



    def fine_tuning(self, context: dict, callback=None) -> dict:
        """ Perform EM fine-tuning using the EM fine-tuning stage. """
        if self.em_fine_tuning_stage is None:
            raise ValueError("EM fine-tuning stage is not defined. Cannot perform fine-tuning.")

        orca_geometries = context.get("orca_geometries") or {}

        # Validate that every ONNX-based component has a geometry
        if self.em_surrogate_stage is not None:
            onnx_components = [
                comp for comp, is_ts in zip(
                    self.em_surrogate_stage.component_names, self.em_surrogate_stage.is_touchstone
                )
                if not is_ts
            ]
            missing = [c for c in onnx_components if c not in orca_geometries]
            if missing:
                raise ValueError(
                    f"No ORCA geometry provided for component(s): {missing}. "
                    "Cannot perform EM fine-tuning without geometry information."
                )

        design_goal_checker: DesignGoalChecker = context["design_goal_checker"]
        fine_tuning_optimizer_stage = self._build_fine_tuning_optimizer_stage()

        if fine_tuning_optimizer_stage is not self.optimizer_stage:
            fine_tuning_optimizer_stage.optimizer.initialize(len(design_goal_checker.design_goals))

        for iteration in tqdm.tqdm(range(self.fine_tuning_iterations), desc="COBRA EM Fine-Tuning Progress"):
            context["iteration"] = iteration + 1
            context["fine_tuning_active"] = True
            context["fine_tuning_iteration"] = iteration + 1
            context["fine_tuning_total"] = self.fine_tuning_iterations

            # Build a name→network map from the previous iteration for .snp components
            prior_networks_by_comp = {
                ntwk.name: ntwk
                for ntwk in context.get("predicted_networks", [])
                if ntwk.name
            }

            # Run Palace for every ONNX component; carry forward .snp networks unchanged
            assembled_networks = []
            for comp_name, is_ts in zip(
                self.em_surrogate_stage.component_names, self.em_surrogate_stage.is_touchstone
            ):
                if is_ts:
                    ntwk = prior_networks_by_comp.get(comp_name)
                    if ntwk is not None:
                        assembled_networks.append(ntwk)
                else:
                    context = self.em_fine_tuning_stage.run(
                        context,
                        orca_geometry=orca_geometries[comp_name],
                        comp_name=comp_name,
                    )
                    assembled_networks.append(context["predicted_networks"][0])

            context["predicted_networks"] = assembled_networks

            # Perform circuit-level simulation
            context = self.circuit_simulation_stage.run(context)

            # Check design goals
            context = design_goal_checker.check_goals(context)

            if callback:
                should_continue = callback(context)
                if should_continue is False:
                    print("Fine Tuning stopped by callback.")
                    break

            # If design goals are achieved, break the loop
            if context["goal_achieved"]:
                print(f"Design goals achieved after EM fine-tuning at iteration {iteration}.")
                break
            else:
                print(f"Design goals not achieved after EM fine-tuning at iteration {iteration}. Continuing optimization...")

            fine_tuning_optimizer_stage.tell(context)
            context = fine_tuning_optimizer_stage.run(context)

        if not context["goal_achieved"]:
            print("EM fine-tuning completed without achieving design goals. Returning best parameters found.")
        else:
            print("Design goals achieved and geometry verified with EM simulation. Returning optimized parameters.")

        return context
    
    def print_time(self, context: dict):
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