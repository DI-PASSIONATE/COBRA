from typing import Dict
from cobra.optimizers import BaseOptimizer
from typing import Callable, Dict
import numpy as np
import optuna

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

    def step(self, context, input_parameter_range: Dict[str, tuple | list | np.ndarray]) -> Dict:
        trial = self.study.ask()
        context["trial"] = trial

        # Suggest parameters for the current trial
        suggested_parameters = {}
        for param_name, param_range in input_parameter_range.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                suggested_parameters[param_name] = trial.suggest_float(param_name, low=param_range[0], high=param_range[1], step=0.1)
            elif isinstance(param_range, list):
                suggested_parameters[param_name] = trial.suggest_categorical(param_name, param_range)
            elif isinstance(param_range, np.ndarray):
                suggested_parameters[param_name] = trial.suggest_float(param_name, low=np.min(param_range), high=np.max(param_range), step=0.1)
            else:
                raise ValueError(f"Unsupported parameter range type {type(param_range)} for {param_name}")

        context["parameters"] = suggested_parameters
        #print(f"Optuna suggested parameters for trial {trial.number}: {suggested_parameters}")
        return suggested_parameters