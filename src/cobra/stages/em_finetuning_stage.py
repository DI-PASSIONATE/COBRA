import importlib
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any, cast

import skrf as rf

from cobra.configuration.configuration import (
    DEFAULT_PALACE_PROCESSES,
    ConfigurationError,
)
from cobra.spice_sim.base_simulator import SimulatorError
from cobra.stages.base_stage import COBRABaseStage

if TYPE_CHECKING:
    from cobra.optimization_context import OptimizationContext

logger = logging.getLogger(__name__)

#: Ask ORCA to write every Touchstone variant it can after a Palace run.
TOUCHSTONE_TYPE = "all"

#: The variants to read back, most corrected first. ORCA only adds the DC-extrapolated
#: ones when the sweep starts at or below 1 GHz and has more than 20 points.
RESULT_PREFERENCE = ("dc_deembedded", "deembedded", "dc", "normal")


def _mesh_gds_and_run_palace(
    *,
    name: str,
    parameters: dict[str, Any],
    base_dir: str,
    gds_output_path: str,
    stackup_xml: str,
    simconfig_filename: str,
    palace_executable: str,
    num_processes: int,
) -> bool:
    """Run gmsh-dependent model creation and Palace simulation in a child process.

    Returns whether Palace succeeded; ORCA logs the reason when it did not.
    """
    create_palace_model_from_gds = importlib.import_module(
        "orca.simulation.gds_converter"
    ).create_palace_model_from_gds
    run_palace = importlib.import_module("orca.simulation.simulate").run_palace
    LocalLauncher = importlib.import_module("orca.simulation.launchers").LocalLauncher

    _, _, config_name, sim_path, data_dir = create_palace_model_from_gds(
        geometry_name=name,
        params=parameters,
        output_dir=base_dir,
        gds_filename=gds_output_path,
        stackup_xml=stackup_xml,
        simconfig_filename=simconfig_filename,
        show_mesh_results=False,
    )
    launcher = LocalLauncher()
    return run_palace(
        sim_path=sim_path,
        data_dir=data_dir,
        result_dir=base_dir,
        cmd=launcher.command(launcher.slots[0], palace_executable, num_processes, config_name),
        touchstone_type=TOUCHSTONE_TYPE,
    )


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
        Creates a GDS file based on the current parameters, meshes it.
        If comp_name is provided, only parameters for that component are forwarded.
        """
        BaseGeometry = importlib.import_module("orca.geometry.base_geometry").BaseGeometry
        if not isinstance(orca_geometry, BaseGeometry):
            raise TypeError("orca_geometry must be an instance of BaseGeometry")
        geometry = cast("Any", orca_geometry)

        base_dir = os.path.abspath(context.results_dir)
        fine_tuning_run = context.fine_tuning_iteration
        name_suffix = f"_{comp_name}" if comp_name else ""
        name = f"cobra_result_ft_{fine_tuning_run}_{context.iteration}{name_suffix}"
        gds_output_path = os.path.join(base_dir, f"{name}.gds")

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

        geometry.create_gds_file(name=name, output_path=gds_output_path, params=parameters)

        # !!!
        # gmsh.initialize must run in a process-main thread, not in the PySide worker thread (QThread)
        # so we use a single-worker ProcessPoolExecutor to prevent crashes due to gmsh
        spawn_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=spawn_ctx) as executor:
            future = executor.submit(
                _mesh_gds_and_run_palace,
                name=name,
                parameters=parameters,
                base_dir=base_dir,
                gds_output_path=gds_output_path,
                stackup_xml=geometry.stackup_xml,
                simconfig_filename=geometry.simconfig_filename,
                palace_executable=self.palace_executable,
                num_processes=self.num_processes,
            )
            succeeded = future.result()
        if not succeeded:
            raise SimulatorError(
                f"The Palace simulation for {comp_name or name} failed in {base_dir}; "
                "see ORCA's log output above for the reason."
            )

        ntwk = rf.Network(self._result_file(base_dir, name, geometry.n_ports, comp_name or name))
        if comp_name:
            ntwk.name = comp_name
        context.predicted_networks = [ntwk]
        return context

    @staticmethod
    def _result_file(base_dir: str, name: str, n_ports: int, label: str) -> str:
        """The most corrected Touchstone file ORCA wrote for *name*."""
        touchstone_filename = importlib.import_module(
            "orca.simulation.combine_snp_results"
        ).touchstone_filename
        candidates = [
            os.path.join(base_dir, touchstone_filename(name, n_ports, variant))
            for variant in RESULT_PREFERENCE
        ]
        for variant, path in zip(RESULT_PREFERENCE, candidates, strict=True):
            if os.path.isfile(path):
                if variant != RESULT_PREFERENCE[0]:
                    logger.info(
                        "Using the %s Palace result for %s: ORCA only extrapolates to DC when "
                        "the sweep starts at or below 1 GHz with more than 20 points",
                        variant,
                        label,
                    )
                return path
        raise SimulatorError(
            f"Palace finished for {label}, but ORCA wrote no Touchstone file; looked for "
            + ", ".join(os.path.basename(path) for path in candidates)
            + f" in {base_dir}."
        )
