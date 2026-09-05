import importlib
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, cast

import skrf as rf

from cobra.stages.base_stage import COBRABaseStage


def _mesh_gds_and_run_palace(
    *,
    name: str,
    parameters: dict[str, Any],
    base_dir: str,
    gds_output_path: str,
    stackup_xml: str,
    simconfig_filename: str,
    palace_executable: str,
) -> None:
    """Run gmsh-dependent model creation and Palace simulation in a child process."""
    from ihp import PDK

    create_palace_model_from_gds = importlib.import_module(
        "orca.simulation.gds_converter"
    ).create_palace_model_from_gds
    run_palace = importlib.import_module("orca.simulation.simulate").run_palace

    PDK.activate()
    create_palace_model_from_gds(
        geometry_name=name,
        params=parameters,
        output_dir=base_dir,
        gds_filename=gds_output_path,
        stackup_xml=stackup_xml,
        simconfig_filename=simconfig_filename,
        show_mesh_results=False,
    )
    sim_path = os.path.join(base_dir, "palace_sims", f"{name}_data")
    run_palace(
        sim_path=sim_path,
        data_dir=os.path.join(sim_path, "output", name),
        result_dir=os.path.join(base_dir),
        config_name=os.path.join(sim_path, "config.json"),
        palace_executable=palace_executable,
        num_processes=16,
        touchstone_type="all",
    )

class EMFineTuningStage(COBRABaseStage):
    """
    EM Fine-Tuning Stage - This stage performs real EM simulations using the Palace EM simulator to fine-tune the design parameters.
    This is to ensure that the surrogate model's predictions are accurate and to refine the design based on real EM results.
    """

    def __init__(self, palace_executable):
        self.palace_executable = palace_executable

        
    def run(self, context: dict, orca_geometry=None, comp_name: str | None = None) -> dict:
        """
        Creates a GDS file based on the current parameters, meshes it.
        If comp_name is provided, only parameters for that component are forwarded.
        """
        from ihp import PDK
        BaseGeometry = importlib.import_module("orca.geometry.base_geometry").BaseGeometry
        
        PDK.activate()
        if not isinstance(orca_geometry, BaseGeometry):
            raise TypeError("orca_geometry must be an instance of BaseGeometry")
        geometry = cast(Any, orca_geometry)
        
        base_dir = os.path.abspath(context.get("results_dir", os.path.join(os.getcwd(), "results")))
        fine_tuning_run = context.get("fine_tuning_iteration", 0)
        name_suffix = f"_{comp_name}" if comp_name else ""
        name = f"cobra_result_ft_{fine_tuning_run}_{context['iteration']}{name_suffix}"
        gds_output_path = os.path.join(base_dir, f"{name}.gds")
        
        # Filter parameters for this specific component if comp_name is given
        all_parameters = context["model_parameters"]
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
            )
            future.result()

        ntwk = rf.Network(os.path.join(base_dir, f"{name}_dc_deembedded.s6p"))
        if comp_name:
            ntwk.name = comp_name
        context["predicted_networks"] = [ntwk]
        return context
