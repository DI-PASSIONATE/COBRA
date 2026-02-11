from typing import Dict
from cobra.optimizers.base_optimizer import BaseOptimizer
from cobra.stages.base_stage import COBRABaseStage

class OptimizerStage(COBRABaseStage):
    """
    Optimizer Stage - This stage performs the optimization step using the provided optimizer.
    It takes the current design state, runs the optimizer, and updates the design state with the new parameters.
    """

    def __init__(self, optimizer: BaseOptimizer):
        self.optimizer = optimizer

    def run(self, context: Dict) -> Dict:
        parameter_range = context["parameters"]
        constraints = context["design_goals"]
        self.optimizer.step(parameter_range, constraints)
        return context
