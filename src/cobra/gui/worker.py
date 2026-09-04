import threading
from typing import Any

from PySide6.QtCore import QThread, Signal

from cobra.cobra import COBRA
from cobra.configuration import RunConfiguration
from cobra.optimizers.base_optimizer import OptimizationProperty
from cobra.optimizers.design_goal import DesignGoal


class OptimizationWorker(QThread):
    progress = Signal(dict)
    finished = Signal()
    error = Signal(str)
    ask_continue = Signal(int)

    def __init__(self, cobra_instance: COBRA, netlist: str, design_goals: list[DesignGoal], 
                 optimization_parameters: list[OptimizationProperty], 
                 max_iterations: int, orca_geometries: Any | None = None,
                 sim_params_by_type: Any | None = None,
                 run_configuration: RunConfiguration | None = None):
        super().__init__()
        self.cobra = cobra_instance
        self.netlist = netlist
        self.design_goals = design_goals
        self.optimization_parameters = optimization_parameters
        self.max_iterations = max_iterations
        self.orca_geometries = orca_geometries
        self.sim_params_by_type = sim_params_by_type or {}
        self.run_configuration = run_configuration
        self.stop_requested = False
        self.paused = False
        self.resume_event = threading.Event()
        self.resume_event.set()
        self.prev_network = None

    def run(self):
        try:
            def optimization_callback(context):
                if self.stop_requested:
                    return False

                if self.paused:
                    self.resume_event.wait()
                    if self.stop_requested:
                        return False
                
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
                run_configuration=self.run_configuration,
            )
            
            self.finished.emit()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def stop(self):
        self.stop_requested = True
        self.resume_event.set()

    def pause(self):
        self.paused = True
        self.resume_event.clear()

    def resume(self):
        self.paused = False
        self.resume_event.set()
