# Example usage
from cobra import COBRA, DesignGoal, DesignParameter, OptimizationProperty, OptimizationType, XyceSimulator, OptunaOptimizer
from orca.geometry.presets.tf_octa_c_ports import TransformerOcta

design_goals = [
    DesignGoal(DesignParameter.S11_dB, max_value=-8, frequency_range="125-135ghz"),
    DesignGoal(DesignParameter.S21_dB, min_value=-3, max_value=0, frequency_range="125-135ghz"),
]

cobra = COBRA(
    optimizer=OptunaOptimizer(
        multi_objective=False,
        sampler="tpe",  # Options: "tpe", "random", "auto", "simulated_annealing"
        pruner="median",  # Options: None, "median", "successive_halving", "hyperband"
    ),
    circuit_simulator=XyceSimulator(),
    em_surrogate_model="/home/david/Documents/git/COBRA/tf_octa_c_ports.onnx",
    palace_fine_tuning_command="apptainer exec ~/Documents/git/palace/palace.sif palace"
)

netlist = "netlist_xyce_vector.cir"

parameters = {
    "input_winding_diameter": (20.0, 100.0),
    "output_winding_diameter": (20.0, 100.0),
    "center_displacement": (0.0, 20.0),
    "bottom_linewidth": (2.0, 8.0),
    "upper_linewidth": (2.0, 8.0),
}

parameters = [
    OptimizationProperty(name="input_winding_diameter", type=OptimizationType.MODEL_INPUT, min_value=20.0, max_value=100.0, step=0.1),
    OptimizationProperty(name="output_winding_diameter", type=OptimizationType.MODEL_INPUT, min_value=20.0, max_value=100.0, step=0.1),
    OptimizationProperty(name="center_displacement", type=OptimizationType.MODEL_INPUT, min_value=0.0, max_value=20.0, step=0.1),
    OptimizationProperty(name="bottom_linewidth", type=OptimizationType.MODEL_INPUT, min_value=2.0, max_value=8.0, step=0.1),
    OptimizationProperty(name="upper_linewidth", type=OptimizationType.MODEL_INPUT, min_value=2.0, max_value=8.0, step=0.1),
    # Optimize C3 from 0 to 20 fF with a step of 1 fF
    OptimizationProperty(name="C3", type=OptimizationType.NETLIST_VARIABLE, unit="F", min_value=0.0, max_value=20.0, step=1.0),
    # C4 is linked to C3 and will always use the same value
    OptimizationProperty(name="C4", type=OptimizationType.NETLIST_VARIABLE, min_value=0.0, max_value=0.0, step=0.0, linked_to="C3"),
]

context = cobra.run(
    netlist, 
    design_goals,
    parameters, 
    orca_geometry=TransformerOcta()
)

print(context)