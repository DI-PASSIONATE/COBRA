"""Example COBRA workflow using mixed component sources and linked netlist values."""
import os

from cobra import (
    COBRA,
    DesignGoal,
    DesignParameter,
    OptimizationProperty,
    OptimizationType,
    OptunaOptimizer,
    XyceSimulator,
)
from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
from orca.geometry.presets.tf_octa_c_ports import TransformerOcta


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NETLIST_PATH = os.path.join(BASE_DIR, "netlist_multiple_SPFiles.cir")
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "tf_octa_c_ports.onnx")
FIXED_TOUCHSTONE_PATH = os.path.join(BASE_DIR, "XYLIN_Trafo_output_predicted.s6p")


design_goals = [
    DesignGoal(DesignParameter.S11_dB, max_value=-9, frequency_range="125-135ghz"),
    DesignGoal(DesignParameter.S21_dB, min_value=-3, max_value=0, frequency_range="125-135ghz"),
]

parser = XyceNetlistParser().from_file(NETLIST_PATH)

cobra = COBRA(
    netlist_parser=parser,
    component_onnx_mapping={
        "X1": str(ONNX_MODEL_PATH),
        "X2": str(FIXED_TOUCHSTONE_PATH),
    },
    optimizer=OptunaOptimizer(
        multi_objective=False,
        sampler="tpe",
        pruner="median",
    ),
    circuit_simulator=XyceSimulator(),
)

parameters = [
    OptimizationProperty(name="X1:bottom_winding_diameter", type=OptimizationType.MODEL_INPUT, min_value=20.0, max_value=100.0, step=0.1),
    OptimizationProperty(name="X1:top_winding_diameter", type=OptimizationType.MODEL_INPUT, min_value=20.0, max_value=100.0, step=0.1),
    OptimizationProperty(name="X1:center_displacement", type=OptimizationType.MODEL_INPUT, min_value=0.0, max_value=20.0, step=0.1),
    OptimizationProperty(name="X1:bottom_linewidth", type=OptimizationType.MODEL_INPUT, min_value=2.0, max_value=8.0, step=0.1),
    OptimizationProperty(name="X1:top_linewidth", type=OptimizationType.MODEL_INPUT, min_value=2.0, max_value=8.0, step=0.1),
    # Directly optimize parsed component values in the netlist text.
    OptimizationProperty(name="Cshunt_p", type=OptimizationType.NETLIST_VARIABLE, unit="F", min_value=0.0, max_value=20.0, step=1.0),
    OptimizationProperty(name="Cshunt_n", type=OptimizationType.NETLIST_VARIABLE, linked_to="Cshunt_p", unit="F", min_value=0.0, max_value=0.0, step=0.0),
]

context = cobra.run(
    netlist=str(NETLIST_PATH),
    design_goals=design_goals,
    optimization_parameters=parameters,
    orca_geometry=TransformerOcta(),
    max_iterations=200,
    results_name="main_mixed_sources_example",
)

print(context)