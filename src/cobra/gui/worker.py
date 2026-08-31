from typing import Any, List, Optional
import time
from PySide6.QtCore import QThread, Signal

from cobra.cobra import COBRA
from cobra.optimizers.base_optimizer import OptimizationProperty
from cobra.optimizers.design_goal import DesignGoal, DesignGoalChecker

class OptimizationWorker(QThread):
    progress = Signal(dict)
    finished = Signal()
    error = Signal(str)
    ask_continue = Signal(int)

    def __init__(self, cobra_instance: COBRA, netlist: str, design_goals: List[DesignGoal], 
                 optimization_parameters: List[OptimizationProperty], 
                 max_iterations: int, orca_geometries: Optional[Any] = None,
                 sim_params_by_type: Optional[Any] = None):
        super().__init__()
        self.cobra = cobra_instance
        self.netlist = netlist
        self.design_goals = design_goals
        self.optimization_parameters = optimization_parameters
        self.max_iterations = max_iterations
        self.orca_geometries = orca_geometries
        self.sim_params_by_type = sim_params_by_type or {}
        self.stop_requested = False
        self.paused = False

    def run(self):
        try:
            # Helper to calculate loss and notify text of "prev_network" tracking
            # The context in cobra.run is persistent, so we can store prev_network directly in it if we want,
            # but ideally we handle the state tracking in the callback.
            # However, the context passed to the callback IS the mutable dictionary from cobra.run.
            
            # We'll attach a small state container to the worker to track prev_network
            self.prev_network = None
            self.start_time = time.time()

            def optimization_callback(context):
                while self.paused:
                    if self.stop_requested:
                        return False
                    time.sleep(0.1)

                if self.stop_requested:
                    return False
                
                # Calculate elapsed time
                context["elapsed_time"] = time.time() - self.start_time
                
                # Handle prev_network logic for plotting
                context["prev_network"] = self.prev_network
                
                # Emit progress
                self.progress.emit(context)
                
                # Check if we reached max iterations and ask to continue
                if context["iteration"] >= context["max_iterations"] and not context["goal_achieved"]:
                    self.ask_continue.emit(context["max_iterations"])
                    context["max_iterations"] = self.max_iterations

                # Update prev_network for next iteration
                sim_results = context.get("simulation_results") or {}
                self.prev_network = next((r.network for r in sim_results.values() if r.network is not None), None)
                
                return True

            # Call COBRA run directly
            self.cobra.run(
                netlist=self.netlist,
                design_goals=self.design_goals,
                optimization_parameters=self.optimization_parameters,
                max_iterations=self.max_iterations,
                orca_geometries=self.orca_geometries,
                callback=optimization_callback,
                sim_params_by_type=self.sim_params_by_type,
            )
            
            self.finished.emit()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def stop(self):
        self.stop_requested = True

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False
