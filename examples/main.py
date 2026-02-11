# Example usage
from cobra.cobra import COBRA
from cobra.stages import *


cobra = COBRA(
    em_surrogate_stage=EMSurrogateStage("/home/david/Documents/git/COBRA/tf_octa_c_ports.onnx"),
)
netlist = "netlist.cir"
design_goals = {}
parameters = {
    "input_winding_diameter": [20, 100],
    "output_winding_diameter": [20, 100],
    "center_displacement": [0, 20],
    "bottom_linewidth": [2, 8],
    "upper_linewidth": [2, 8],
}
optimized_parameters = cobra.run(netlist, design_goals, parameters)
print("Optimized Parameters:", optimized_parameters)