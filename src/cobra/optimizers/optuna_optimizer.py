import importlib
import logging
from typing import TYPE_CHECKING, Any, ClassVar

import optuna

from cobra.configuration.setting import CobraSetting
from cobra.optimizers.base_optimizer import (
    BaseOptimizer,
    OptimizationProperty,
    netlist_unit,
    resolve_linked_master,
)

if TYPE_CHECKING:
    from cobra.optimization_context import OptimizationContext

logger = logging.getLogger(__name__)


class OptunaOptimizer(BaseOptimizer):
    """
    OptunaOptimizer - An implementation of the BaseOptimizer using Optuna for optimization.
    """

    # Optuna's ask-and-tell interface allows any number of outstanding trials.
    supports_parallel_trials: ClassVar[bool] = True

    _settings: ClassVar[list[CobraSetting]] = [
        CobraSetting(
            name="multi_objective",
            dtype=bool,
            default=False,
            description=(
                "Enable multi-objective optimisation (Pareto front).\n"
                "When disabled, per-goal losses are aggregated into a single scalar."
            ),
        ),
        CobraSetting(
            name="sampler",
            dtype=str,
            default="tpe",
            description=(
                "Optuna sampling algorithm used to suggest trial parameters.\n"
                "TPE (Tree-structured Parzen Estimator) is a good general-purpose choice.\n"
                "SimulatedAnnealing requires the optunahub package."
            ),
            choices=[
                ("TPESampler (default)", "tpe"),
                ("RandomSampler", "random"),
                ("SimulatedAnnealingSampler (optunahub)", "simulated_annealing"),
            ],
        ),
        CobraSetting(
            name="pruner",
            dtype=str,
            default=None,
            description=(
                "Optuna pruner that terminates unpromising trials early.\n"
                "Pruning can reduce runtime but may miss good regions of the search space."
            ),
            choices=[
                ("None", None),
                ("MedianPruner", "median"),
                ("SuccessiveHalvingPruner", "successive_halving"),
                ("HyperbandPruner", "hyperband"),
            ],
        ),
    ]

    def __init__(
        self,
        multi_objective: bool = False,
        sampler: str | optuna.samplers.BaseSampler | None = "tpe",
        pruner: str | optuna.pruners.BasePruner | None = None,
        sampler_kwargs: dict[str, Any] | None = None,
        pruner_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(multi_objective=multi_objective)
        self.sampler = sampler
        self.pruner = pruner
        self.sampler_kwargs = sampler_kwargs or {}
        self.pruner_kwargs = pruner_kwargs or {}
        self.study: optuna.study.Study | None = None
        self.parallel_trials = 1
        self._param_to_trial_name: dict[str, str] = {}

    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.replace("_", "").replace("-", "").lower()

    def _get_study(self) -> optuna.study.Study:
        if self.study is None:
            raise RuntimeError("The Optuna study has not been initialized yet. Call initialize() first.")
        return self.study

    def _load_optunahub_sampler(self, package_name: str, class_name: str) -> optuna.samplers.BaseSampler:
        try:
            optunahub = importlib.import_module("optunahub")
        except ImportError as exc:
            raise ImportError(
                "optunahub is required to use the requested sampler. Install it with `pip install optunahub`."
            ) from exc

        try:
            module = optunahub.load_module(package_name)
            sampler_cls = getattr(module, class_name)
            return sampler_cls(**self.sampler_kwargs)
        except Exception as exc:
            if class_name == "AutoSampler":
                raise RuntimeError(
                    "Failed to initialize AutoSampler. Install its optional dependencies, e.g. `pip install optunahub cmaes scipy torch`."
                ) from exc
            raise

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        if isinstance(self.sampler, optuna.samplers.BaseSampler):
            return self.sampler

        if self.sampler is None:
            sampler_name = "tpe"
        elif isinstance(self.sampler, str):
            sampler_name = self._normalize_name(self.sampler)
        else:
            raise TypeError("sampler must be a string, an Optuna sampler instance, or None.")

        if sampler_name in {"tpe", "tpesampler"}:
            # With several trials in flight, TPE would keep suggesting the same
            # point until the first result arrives; the constant liar penalises
            # running trials so concurrent asks explore different regions.
            kwargs = dict(self.sampler_kwargs)
            if self.parallel_trials > 1:
                kwargs.setdefault("constant_liar", True)
            return optuna.samplers.TPESampler(**kwargs)
        if sampler_name in {"random", "randomsampler"}:
            return optuna.samplers.RandomSampler(**self.sampler_kwargs)
        if sampler_name in {"simulatedannealing", "simulatedannealingsampler"}:
            return self._load_optunahub_sampler("samplers/simulated_annealing", "SimulatedAnnealingSampler")
        if sampler_name in {"auto", "autosampler"}:
            return self._load_optunahub_sampler("samplers/auto_sampler", "AutoSampler")

        raise ValueError(
            "Unsupported sampler. Choose one of: AutoSampler, RandomSampler, TPESampler, SimulatedAnnealingSampler."
        )

    def _create_pruner(self) -> optuna.pruners.BasePruner | None:
        if isinstance(self.pruner, optuna.pruners.BasePruner):
            return self.pruner

        if self.pruner is None:
            return None
        if not isinstance(self.pruner, str):
            raise TypeError("pruner must be a string, an Optuna pruner instance, or None.")

        pruner_name = self._normalize_name(self.pruner)

        if pruner_name in {"median", "medianpruner"}:
            return optuna.pruners.MedianPruner(**self.pruner_kwargs)
        if pruner_name in {
            "successivehalving",
            "successivehalvingpruner",
            "sucessivehalving",
            "sucessivehalvingpruner",
        }:
            return optuna.pruners.SuccessiveHalvingPruner(**self.pruner_kwargs)
        if pruner_name in {"hyperband", "hyperbandpruner"}:
            return optuna.pruners.HyperbandPruner(**self.pruner_kwargs)

        raise ValueError(
            "Unsupported pruner. Choose one of: MedianPruner, SuccessiveHalvingPruner, HyperbandPruner."
        )

    def initialize(self, num_goals: int, parallel_trials: int = 1):
        self.parallel_trials = parallel_trials
        # Optuna installs its own handler and announces every trial; COBRA already
        # reports progress, so let its chatter follow COBRA's own verbosity.
        optuna.logging.set_verbosity(
            optuna.logging.INFO if logger.isEnabledFor(logging.DEBUG) else optuna.logging.WARNING
        )
        directions = ["minimize"] * num_goals if self.multi_objective else ["minimize"]
        self.study = optuna.create_study(
            directions=directions,
            sampler=self._create_sampler(),
            pruner=self._create_pruner(),
        )

    def tell(self, context: "OptimizationContext", penalty: list[float] | float):
        trial = context.trial
        self._get_study().tell(trial, penalty)

    def step(self, context: "OptimizationContext", model_input_ranges: list[OptimizationProperty], netlist_property_ranges: list[OptimizationProperty]) -> None:
        trial = self._get_study().ask()
        context.trial = trial
        self._param_to_trial_name = {}

        def _suggest(params: list[OptimizationProperty], with_unit: bool) -> dict[str, Any]:
            by_name = {p.name: p for p in params}
            sampled: dict[str, Any] = {}
            values: dict[str, Any] = {}

            for param in params:
                master = resolve_linked_master(param, by_name)
                trial_name = master.name

                if trial_name not in sampled:
                    sampled[trial_name] = trial.suggest_float(
                        trial_name,
                        low=master.min_value,
                        high=master.max_value,
                        step=master.step,
                    )
                value = sampled[trial_name]

                self._param_to_trial_name[param.name] = trial_name
                if with_unit:
                    values[param.name] = f"{value}{netlist_unit(param, by_name)}"
                else:
                    values[param.name] = value
            return values

        # Suggest parameters for the current trial (linked params share a sampled value)
        model_parameters = _suggest(model_input_ranges, with_unit=False)
        netlist_parameters = _suggest(netlist_property_ranges, with_unit=True)

        # Update context
        context.model_parameters = model_parameters
        context.netlist_parameters = netlist_parameters

    def get_best_parameters(self) -> dict[str, Any]:
        if self.multi_objective:
            raise ValueError("get_best_parameters is not available for multi-objective optimization. Use get_moo_results instead.")
        best = dict(self._get_study().best_params)
        for param_name, trial_name in self._param_to_trial_name.items():
            if trial_name in best:
                best[param_name] = best[trial_name]
        return best

    def get_moo_results(self) -> Any:
        if self.multi_objective:
            return self._get_study().best_trials
        raise ValueError("Multi-objective optimization is not enabled for this optimizer.")

