import os
from typing import TYPE_CHECKING, Any

from cobra.configuration.configuration import (
    DEFAULT_PALACE_PROCESSES,
    ConfigurationError,
)
from cobra.spice_sim.base_simulator import SimulatorError
from cobra.stages.base_stage import COBRABaseStage

if TYPE_CHECKING:
    from cobra.optimization_context import OptimizationContext


class EMFineTuningStage(COBRABaseStage):
    """
    EM Fine-Tuning Stage - This stage performs real EM simulations using the Palace EM simulator to fine-tune the design parameters.
    This is to ensure that the surrogate model's predictions are accurate and to refine the design based on real EM results.
    """

    def __init__(self, palace_executable, num_processes: int = DEFAULT_PALACE_PROCESSES):
        if isinstance(num_processes, bool) or not isinstance(num_processes, int):
            raise ConfigurationError("num_processes must be an integer")
        if num_processes < 1:
            raise ConfigurationError(f"num_processes must be at least 1, got {num_processes}")
        self.palace_executable = palace_executable
        self.num_processes = num_processes


    def run(self, context: "OptimizationContext", orca_geometry=None, comp_name: str | None = None) -> "OptimizationContext":
        """
        Simulates the geometry with its current parameters in Palace through ORCA.
        If comp_name is provided, only parameters for that component are forwarded.
        """
        # ORCA is optional: only EM fine-tuning needs it
        from orca import BaseGeometry, SimulationError, simulate_geometry

        if not isinstance(orca_geometry, BaseGeometry):
            raise TypeError("orca_geometry must be an instance of BaseGeometry")

        base_dir = os.path.abspath(context.results_dir)
        fine_tuning_run = context.fine_tuning_iteration
        name_suffix = f"_{comp_name}" if comp_name else ""
        name = f"cobra_result_ft_{fine_tuning_run}_{context.iteration}{name_suffix}"

        # Filter parameters for this specific component if comp_name is given
        all_parameters = context.model_parameters
        if comp_name:
            prefix = f"{comp_name}:"
            parameters: dict[str, Any] = {}
            for k, v in all_parameters.items():
                if k.startswith(prefix):
                    parameters[k[len(prefix):]] = v  # strip component prefix
                elif ":" not in k:
                    parameters[k] = v  # shared / unscoped parameter
        else:
            parameters = all_parameters

        try:
            ntwk = simulate_geometry(
                orca_geometry,
                parameters,
                base_dir,
                name,
                palace_executable=self.palace_executable,
                num_processes=self.num_processes,
            )
        except SimulationError as exc:
            raise SimulatorError(f"EM fine-tuning of {comp_name or name} failed: {exc}") from exc
        if comp_name:
            ntwk.name = comp_name
        context.predicted_networks = [ntwk]
        return context
