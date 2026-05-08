from typing import Dict
from cobra.optimizers.base_optimizer import BaseOptimizer, OptimizationType
from cobra.optimizers.design_goal import DesignGoalChecker
from cobra.stages.base_stage import COBRABaseStage
import numpy as np

class OptimizerStage(COBRABaseStage):
    """
    Optimizer Stage - This stage performs the optimization step using the provided optimizer.
    It takes the current design state, runs the optimizer, and updates the design state with the new parameters.
    """

    def __init__(self, optimizer: BaseOptimizer):
        self.optimizer = optimizer

    def run(self, context: Dict) -> Dict:
        optimization_parameters = context["optimization_parameters"]
        model_input_parameters = [p for p in optimization_parameters if p.type == OptimizationType.MODEL_INPUT]
        netlist_variable_parameters = [p for p in optimization_parameters if p.type == OptimizationType.NETLIST_VARIABLE]

        self.optimizer.step(context, model_input_parameters, netlist_variable_parameters)
        return context
    
    def tell(self, context):
        loss_values = context["penalties"]
        status = "finetuning" if context.get("fine_tuning_active") else "optimization"
        context["iterations"].append({
            "iteration": context.get("iteration"),
            "status": status,
            "model_parameters": context["model_parameters"],
            "netlist_parameters": context["netlist_parameters"],
            "loss": loss_values,
        })
        # Use _tell to possibly convert the list of loss values into a single penalty value if multi_objective is False
        self.optimizer._tell(context, loss_values)

