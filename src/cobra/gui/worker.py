from typing import List, Optional
import time
from PySide6.QtCore import QThread, Signal

from cobra.cobra import COBRA
from cobra.optimizers.base_optimizer import OptimizationProperty
from cobra.optimizers.design_goal import DesignGoal, DesignGoalChecker

class OptimizationWorker(QThread):
    progress = Signal(dict)
    finished = Signal()
    error = Signal(str)

    def __init__(self, cobra_instance: COBRA, netlist: str, design_goals: List[DesignGoal], 
                 frequency_range: str, optimization_parameters: List[OptimizationProperty], 
                 max_iterations: int, orca_geometry: Optional[str] = None):
        super().__init__()
        self.cobra = cobra_instance
        self.netlist = netlist
        self.design_goals = design_goals
        self.frequency_range = frequency_range
        self.optimization_parameters = optimization_parameters
        self.max_iterations = max_iterations
        self.orca_geometry = orca_geometry
        self.stop_requested = False

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
                if self.stop_requested:
                    return False
                
                # Calculate elapsed time
                context["elapsed_time"] = time.time() - self.start_time

                # Calculate loss for display (similar to how it was done before)
                design_goal_checker = context["design_goal_checker"]
                current_losses = design_goal_checker.loss(context["simulated_network"])
                context["current_losses"] = current_losses
                
                # Handle prev_network logic for plotting
                context["prev_network"] = self.prev_network
                
                # Emit progress
                self.progress.emit(context)
                
                # Update prev_network for next iteration
                self.prev_network = context["simulated_network"]
                
                return True

            # Call COBRA run directly
            self.cobra.run(
                netlist=self.netlist,
                design_goals=self.design_goals,
                frequency_range=self.frequency_range,
                optimization_parameters=self.optimization_parameters,
                max_iterations=self.max_iterations,
                orca_geometry=self.orca_geometry,
                callback=optimization_callback
            )
            
            self.finished.emit()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def stop(self):
        self.stop_requested = True
