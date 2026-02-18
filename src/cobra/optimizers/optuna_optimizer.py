from typing import Any, Dict
from cobra.optimizers import BaseOptimizer
from typing import Callable, Dict
import numpy as np
import optuna

from cobra.optimizers.base_optimizer import OptimizationProperty

class OptunaOptimizer(BaseOptimizer):
    """
    OptunaOptimizer - An implementation of the BaseOptimizer using Optuna for optimization.
    """
    def initialize(self, num_goals: int):
        if self.multi_objective:
            directions = ["minimize"] * num_goals
        else:
            directions = ["minimize"]
        self.study = optuna.create_study(directions=directions)

    def tell(self, context, penalty: list[float] | float):
        trial = context["trial"]
        self.study.tell(trial, penalty)

    def step(self, context: Dict[str, Any], model_input_ranges: list[OptimizationProperty], netlist_property_ranges: list[OptimizationProperty]) -> None:
        trial = self.study.ask()
        context["trial"] = trial

        # Suggest parameters for the current trial
        model_parameters = {}
        for param in model_input_ranges:
            model_parameters[param.name] = trial.suggest_float(param.name, low=param.min_value, high=param.max_value, step=param.step)

        netlist_parameters = {}
        for param in netlist_property_ranges:
            netlist_parameters[param.name] = trial.suggest_float(param.name, low=param.min_value, high=param.max_value, step=param.step)

        # Update context
        context["model_parameters"] = model_parameters
        context["netlist_parameters"] = netlist_parameters