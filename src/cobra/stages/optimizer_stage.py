from typing import Dict
from cobra.optimizers.base_optimizer import BaseOptimizer
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
        parameter_range = context["parameter_range"]
        self.optimizer.step(context, parameter_range)
        return context
    
    def tell(self, context):
        design_goal_checker: DesignGoalChecker = context["design_goal_checker"]
        ntwk = context["simulated_network"]
        loss = design_goal_checker.loss(ntwk)
        self.optimizer.tell(context, loss)

