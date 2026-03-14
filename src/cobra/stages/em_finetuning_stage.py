from cobra.stages.base_stage import COBRABaseStage
from typing import Any, Dict, cast
import importlib
import os
import uuid
import skrf as rf

class EMFineTuningStage(COBRABaseStage):
    """
    EM Fine-Tuning Stage - This stage performs real EM simulations using the Palace EM simulator to fine-tune the design parameters.
    This is to ensure that the surrogate model's predictions are accurate and to refine the design based on real EM results.
    """

    def __init__(self, palace_executable):
        self.palace_executable = palace_executable

        
    def run(self, context: Dict, orca_geometry=None) -> Dict:
        """
        Creates a GDS file based on the current parameters, meshes it 
        """
        from ihp import PDK
        BaseGeometry = importlib.import_module("orca.geometry.base_geometry").BaseGeometry
        create_palace_model_from_gds = importlib.import_module(
            "orca.simulation.gds_converter"
        ).create_palace_model_from_gds
        run_palace = importlib.import_module("orca.simulation.simulate").run_palace
        
        PDK.activate()
        if not isinstance(orca_geometry, BaseGeometry):
            raise ValueError("orca_geometry must be an instance of BaseGeometry")
        geometry = cast(Any, orca_geometry)
        
        base_dir = os.path.join(os.getcwd(), "output")
        fine_tuning_run = context.get("fine_tuning_run", 0) + 1
        context["fine_tuning_run"] = fine_tuning_run
        unique_suffix = uuid.uuid4().hex[:8]
        name = f"cobra_result_ft_{fine_tuning_run}_{context['iteration']}_{unique_suffix}"
        gds_output_path = os.path.join(base_dir, f"{name}.gds")
        
        parameters = context["model_parameters"]
        geometry.create_gds_file(name=name, output_path=gds_output_path, params=parameters)
        create_palace_model_from_gds(
            geometry_name=name,
            params=parameters,
            output_dir=base_dir,
            gds_filename=gds_output_path,
            stackup_xml=geometry.stackup_xml,
            simconfig_filename=geometry.simconfig_filename,
            show_mesh_results=False,
        )
        run_palace(
            sim_path=os.path.join(base_dir, "palace_sims", f"{name}_data"),
            data_dir=os.path.join("output", name),
            result_dir=os.path.join(base_dir),
            config_name=os.path.join(base_dir, "palace_sims", f"{name}_data", "config.json"),
            palace_executable=self.palace_executable,
            cpu_cores=16,
            touchstone_type="all",
        )
        ntwk = rf.Network(os.path.join(base_dir, f"{name}_dc_deembedded.s6p"))
        context["predicted_network"] = ntwk
        return context
