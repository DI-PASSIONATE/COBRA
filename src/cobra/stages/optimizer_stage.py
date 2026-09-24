

from typing import TYPE_CHECKING

from cobra.optimizers.base_optimizer import BaseOptimizer, OptimizationType
from cobra.stages.base_stage import COBRABaseStage

if TYPE_CHECKING:
    from cobra.optimization_context import OptimizationContext


class OptimizerStage(COBRABaseStage):
    """
    Optimizer Stage - This stage performs the optimization step using the provided optimizer.
    It takes the current design state, runs the optimizer, and updates the design state with the new parameters.
    """

    def __init__(self, optimizer: BaseOptimizer):
        self.optimizer = optimizer

    def run(self, context: "OptimizationContext") -> "OptimizationContext":
        optimization_parameters = context.optimization_parameters
        model_input_parameters = [p for p in optimization_parameters if p.type == OptimizationType.MODEL_INPUT]
        netlist_variable_parameters = [p for p in optimization_parameters if p.type == OptimizationType.NETLIST_VARIABLE]

        self.optimizer.step(context, model_input_parameters, netlist_variable_parameters)
        return context

    @staticmethod
    def losses(context: "OptimizationContext") -> list[float]:
        """The per-goal losses of the evaluated iteration in *context*."""
        return [goal.current_penalty if goal.current_penalty is not None else 0.0 for goal in context.goals]

    def record(self, context: "OptimizationContext"):
        """Log the evaluated iteration in *context* without telling the optimizer about it."""
        status = "finetuning" if context.fine_tuning_active else "optimization"
        context.iterations.append({
            "iteration": context.iteration,
            "status": status,
            "model_parameters": context.model_parameters,
            "netlist_parameters": context.netlist_parameters,
            "losses": self.losses(context)
        })

    def tell(self, context: "OptimizationContext"):
        self.record(context)
        # Use _tell to possibly convert the list of loss values into a single penalty value if multi_objective is False
        self.optimizer._tell(context, self.losses(context))  # noqa: SLF001 - documented entry point for stages

