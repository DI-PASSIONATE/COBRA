# Example usage
from cobra import COBRA, DesignGoal, DesignParameter, OptimizationProperty, OptimizationType
from cobra.optimizers.optuna_optimizer import OptunaOptimizer
from orca.geometry.presets.tf_octa_c_ports import TransformerOcta

frequency_range = "125-135ghz"  # Define the frequency range of interest for the design goals

design_goals = [
    DesignGoal(DesignParameter.S11, min_value=90, max_value=-11), # 
    DesignGoal(DesignParameter.S21, min_value=-3, max_value=0), #
]

cobra = COBRA(
    optimizer=OptunaOptimizer(multi_objective=False),
    em_surrogate_model="/home/david/Documents/git/COBRA/tf_octa_c_ports.onnx",
    #palace_fine_tuning_command="apptainer exec ~/Documents/git/palace/palace.sif palace"
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
    OptimizationProperty(name="C1", type=OptimizationType.NETLIST_VARIABLE, min_value=1.0, max_value=100.0, step=0.1),
    OptimizationProperty(name="C2", type=OptimizationType.NETLIST_VARIABLE, min_value=1.0, max_value=100.0, step=0.1),
]

context = cobra.run(
    netlist, 
    design_goals, 
    frequency_range, 
    parameters, 
    orca_geometry=TransformerOcta()
)
print("Optimized Parameters:", context["parameters"])