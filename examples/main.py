# Example usage
from cobra import COBRA, DesignGoal, DesignParameter, DesignGoalChecker

frequency_range = "140-150ghz"  # Define the frequency range of interest for the design goals

design_goals = [
    DesignGoal(DesignParameter.S11, min_value=-90, max_value=-10), # 
    DesignGoal(DesignParameter.S21, min_value=-5, max_value=0),
]

cobra = COBRA(
    em_surrogate_model="/home/david/Documents/git/COBRA/tf_octa_c_ports.onnx",
)

netlist = "netlist_xyce_vector.cir"

parameters = {
    "input_winding_diameter": (20.0, 100.0),
    "output_winding_diameter": (20.0, 100.0),
    "center_displacement": (0.0, 20.0),
    "bottom_linewidth": (2.0, 8.0),
    "upper_linewidth": (2.0, 8.0),
}
optimized_parameters = cobra.run(netlist, design_goals, frequency_range, parameters)
print("Optimized Parameters:", optimized_parameters)