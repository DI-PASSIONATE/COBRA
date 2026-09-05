import re
from typing import Any

import numpy as np

from cobra.configuration.setting import CobraSetting
from cobra.optimizers.base_optimizer import BaseOptimizer, OptimizationProperty


class GradientDescentOptimizer(BaseOptimizer):
    """
    GradientDescentOptimizer - A lightweight local optimizer that uses paired
    finite-difference probes to approximate gradients and refine the current
    solution.
    """

    _settings = [
        CobraSetting(
            name="multi_objective",
            dtype=bool,
            default=False,
            description=(
                "Enable multi-objective optimisation.\n"
                "When disabled, all goal losses are summed into a single scalar."
            ),
        ),
        CobraSetting(
            name="learning_rate",
            dtype=float,
            default=0.2,
            description=(
                "Gradient-descent step size (fraction of parameter range).\n"
                "Smaller values give finer convergence; larger values explore faster."
            ),
        ),
        CobraSetting(
            name="exploration_scale",
            dtype=float,
            default=0.5,
            description=(
                "Scale of finite-difference probe offsets relative to the parameter range.\n"
                "Increase to explore wider areas; decrease for fine local refinement."
            ),
        ),
        CobraSetting(
            name="random_seed",
            dtype=int,
            default=-1,
            description=(
                "Random seed for reproducibility. Use -1 for a non-deterministic run."
            ),
        ),
    ]

    def __init__(
        self,
        multi_objective: bool = False,
        learning_rate: float = 0.2,
        exploration_scale: float = 0.5,
        random_seed: int | None = None,
    ):
        super().__init__(multi_objective=multi_objective)
        self.learning_rate = learning_rate
        self.exploration_scale = exploration_scale
        self._rng = np.random.default_rng(random_seed)
        self._master_properties: dict[str, OptimizationProperty] = {}
        self._master_order: list[str] = []
        self._alias_to_master: dict[str, str] = {}
        self._current_point: dict[str, float] = {}
        self._best_parameters: dict[str, float] = {}
        self._best_penalty = float("inf")
        self._pending_probe: dict[str, Any] | None = None

    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.replace("_", "").replace("-", "").lower()

    @staticmethod
    def _coerce_numeric(value: Any) -> float:
        if isinstance(value, (int, float, np.floating)):
            return float(value)
        match = re.match(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", str(value))
        if not match:
            raise ValueError(f"Cannot parse numeric value from '{value}'.")
        return float(match.group(1))

    @staticmethod
    def _round_to_step(value: float, prop: OptimizationProperty) -> float:
        if prop.step is None or prop.step <= 0:
            return value
        return prop.min_value + round((value - prop.min_value) / prop.step) * prop.step

    @staticmethod
    def _clip(value: float, prop: OptimizationProperty) -> float:
        return float(min(max(value, prop.min_value), prop.max_value))

    def _resolve_master(self, prop: OptimizationProperty, by_name: dict[str, OptimizationProperty]) -> OptimizationProperty:
        current = prop
        seen = {prop.name}
        while current.linked_to:
            target_name = current.linked_to
            target = by_name.get(target_name)
            if target is None:
                raise ValueError(f"Parameter '{current.name}' links to unknown parameter '{target_name}'.")
            if target.name in seen:
                raise ValueError(f"Circular link detected for parameter '{prop.name}'.")
            seen.add(target.name)
            current = target
        return current

    def _collect_masters(self, params: list[OptimizationProperty]) -> list[OptimizationProperty]:
        by_name = {param.name: param for param in params}
        masters: list[OptimizationProperty] = []
        seen: set[str] = set()
        self._alias_to_master = {}

        for param in params:
            master = self._resolve_master(param, by_name)
            self._alias_to_master[param.name] = master.name
            if master.name not in seen:
                masters.append(master)
                seen.add(master.name)

        return masters

    def _seed_from_context(self, context: dict[str, Any], masters: list[OptimizationProperty]) -> dict[str, float]:
        seed: dict[str, float] = {}
        model_parameters = context.get("model_parameters", {}) or {}
        netlist_parameters = context.get("netlist_parameters", {}) or {}

        for master in masters:
            if master.name in model_parameters:
                seed[master.name] = self._clip(float(model_parameters[master.name]), master)
                continue

            if master.name in netlist_parameters:
                seed[master.name] = self._clip(self._coerce_numeric(netlist_parameters[master.name]), master)
                continue

            seed[master.name] = self._clip((master.min_value + master.max_value) / 2.0, master)

        return seed

    def _format_values(self, master_values: dict[str, float], params: list[OptimizationProperty], with_unit: bool) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for param in params:
            master_name = self._alias_to_master[param.name]
            master = self._master_properties[master_name]
            value = master_values[master_name]
            if with_unit:
                unit = param.unit if param.unit else (master.unit if master.unit else "")
                values[param.name] = f"{value}{unit}"
            else:
                values[param.name] = value
        return values

    def _make_direction(self) -> dict[str, float]:
        return {name: float(self._rng.choice([-1.0, 1.0])) for name in self._master_order}

    def _step_sizes(self) -> dict[str, float]:
        step_sizes: dict[str, float] = {}
        for name in self._master_order:
            prop = self._master_properties[name]
            if prop.step is not None and prop.step > 0:
                step_sizes[name] = prop.step
            else:
                span = prop.max_value - prop.min_value
                step_sizes[name] = max(span * 0.05, 1e-6)
        return step_sizes

    def _propose(self, base_point: dict[str, float], direction: dict[str, float], sign: float) -> dict[str, float]:
        step_sizes = self._step_sizes()
        candidate: dict[str, float] = {}
        for name in self._master_order:
            prop = self._master_properties[name]
            raw_value = base_point[name] + sign * self.exploration_scale * step_sizes[name] * direction[name]
            clipped_value = self._clip(raw_value, prop)
            candidate[name] = self._round_to_step(clipped_value, prop)
            candidate[name] = self._clip(candidate[name], prop)
        return candidate

    def initialize(self, num_goals: int):
        if self.multi_objective:
            raise NotImplementedError("GradientDescentOptimizer currently supports single-objective optimization only.")
        self._master_properties = {}
        self._master_order = []
        self._alias_to_master = {}
        self._current_point = {}
        self._best_parameters = {}
        self._best_penalty = float("inf")
        self._pending_probe = None

    def step(self, context: dict[str, Any], model_input_ranges: list[OptimizationProperty], netlist_property_ranges: list[OptimizationProperty]) -> None:
        if self.multi_objective:
            raise NotImplementedError("GradientDescentOptimizer currently supports single-objective optimization only.")

        params = model_input_ranges + netlist_property_ranges
        masters = self._collect_masters(params)
        self._master_properties = {param.name: param for param in masters}
        self._master_order = [param.name for param in masters]

        if not self._current_point:
            self._current_point = self._seed_from_context(context, masters)

        if self._pending_probe is None:
            direction = self._make_direction()
            candidate = self._propose(self._current_point, direction, sign=1.0)
            self._pending_probe = {
                "base": dict(self._current_point),
                "direction": direction,
                "plus_candidate": candidate,
                "plus_penalty": None,
                "side": "plus",
            }
        elif self._pending_probe["side"] == "plus":
            candidate = self._propose(self._pending_probe["base"], self._pending_probe["direction"], sign=-1.0)
            self._pending_probe["minus_candidate"] = candidate
            self._pending_probe["side"] = "minus"
        else:
            # If the probe state is stale, restart from the latest point.
            self._pending_probe = None
            return self.step(context, model_input_ranges, netlist_property_ranges)

        current_candidate = self._pending_probe["plus_candidate"] if self._pending_probe["side"] == "plus" else self._pending_probe["minus_candidate"]
        context["model_parameters"] = self._format_values(current_candidate, model_input_ranges, with_unit=False)
        context["netlist_parameters"] = self._format_values(current_candidate, netlist_property_ranges, with_unit=True)

    def tell(self, context, penalty: list[float] | float):
        if self.multi_objective:
            raise NotImplementedError("GradientDescentOptimizer currently supports single-objective optimization only.")

        if isinstance(penalty, list):
            penalty_value = float(np.sum(penalty))
        else:
            penalty_value = float(penalty)

        if not np.isfinite(penalty_value):
            return

        if self._pending_probe is None:
            current_point = self._extract_point(context)
            if current_point is not None and penalty_value < self._best_penalty:
                self._best_penalty = penalty_value
                self._best_parameters = current_point
            return

        candidate = self._pending_probe["plus_candidate"] if self._pending_probe["side"] == "plus" else self._pending_probe.get("minus_candidate")
        if candidate is not None and penalty_value < self._best_penalty:
            self._best_penalty = penalty_value
            self._best_parameters = dict(candidate)

        if self._pending_probe["side"] == "plus":
            self._pending_probe["plus_penalty"] = penalty_value
            return

        plus_penalty = self._pending_probe.get("plus_penalty")
        if plus_penalty is None or not np.isfinite(plus_penalty):
            self._pending_probe = None
            return

        direction = self._pending_probe["direction"]
        base_point = self._pending_probe["base"]
        step_sizes = self._step_sizes()

        updated_point: dict[str, float] = {}
        for name in self._master_order:
            prop = self._master_properties[name]
            denominator = 2.0 * max(step_sizes[name], 1e-9)
            gradient = ((penalty_value - plus_penalty) / denominator) * direction[name]
            next_value = base_point[name] - self.learning_rate * gradient
            next_value = self._round_to_step(self._clip(next_value, prop), prop)
            updated_point[name] = self._clip(next_value, prop)

        self._current_point = updated_point
        self._pending_probe = None

    def _extract_point(self, context: dict[str, Any]) -> dict[str, float] | None:
        if not self._master_order:
            return None

        point: dict[str, float] = {}
        model_parameters = context.get("model_parameters", {}) or {}
        netlist_parameters = context.get("netlist_parameters", {}) or {}

        for name in self._master_order:
            if name in model_parameters:
                point[name] = float(model_parameters[name])
                continue
            if name in netlist_parameters:
                point[name] = self._coerce_numeric(netlist_parameters[name])
                continue
            return None

        return point

    def get_moo_results(self) -> Any:
        if self.multi_objective:
            raise NotImplementedError("Multi-objective optimization is not supported by GradientDescentOptimizer.")
        raise ValueError("GradientDescentOptimizer does not produce MOO results.")

    def get_best_parameters(self) -> dict[str, float]:
        if self._best_parameters:
            best = dict(self._best_parameters)
        else:
            best = dict(self._current_point)
        for alias_name, master_name in self._alias_to_master.items():
            if master_name in best:
                best[alias_name] = best[master_name]
        return best