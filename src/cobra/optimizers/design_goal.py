from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union
import re

import numpy as np
import skrf as rf
import numpy as np

from cobra.spice_sim.base_simulator import SimulationResult
from cobra.spice_sim.simulation_type import SimulationType

# ---------------------------------------------------------------------------
# DesignParameter — an object-oriented descriptor for a single measurable goal
# ---------------------------------------------------------------------------


@dataclass
class DesignParameter:
    """
    A named, self-describing design parameter.

    Attributes
    ----------
    name:
        Unique identifier used in the GUI, context dict, and log output
        (e.g. ``"S21_dB"``, ``"Lp"``).
    simulation_type:
        Which Xyce analysis type must be run to produce the result.
    formula:
        Callable ``(SimulationResult) -> np.ndarray | float`` that extracts the parameter value(s) from the simulation result.
    loss:
        Callable ``(min_value, max_value, current_value) -> float`` that computes the penalty for the current value relative to the goal range.
    description:
        Human-readable explanation shown as a tooltip in the GUI.
    """

    name: str
    simulation_type: SimulationType
    formula: Callable[[SimulationResult], Union[np.ndarray, float]]
    loss: Callable[[Optional[float], Optional[float], Union[np.ndarray, float]], float]
    description: str = ""
    min_ports: int = 1
    """Minimum number of netlist ports required to evaluate this parameter."""

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, DesignParameter):
            return self.name == other.name
        return NotImplemented



class DesignGoal:
    """
    A single optimisation goal: ``parameter`` must satisfy ``[min_value, max_value]``
    within the given ``frequency_range``.
    """

    def __init__(
        self,
        parameter: DesignParameter,
        frequency_range: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        weight: float = 1.0,
    ):
        if not isinstance(parameter, DesignParameter):
            raise TypeError(
                f"'parameter' must be a DesignParameter instance, got {type(parameter).__name__}. "
                "Use find_parameter() or get_available_parameters() to look up parameters."
            )
        self.parameter = parameter
        self.frequency_range = frequency_range
        self.min_value = min_value
        self.max_value = max_value
        self.weight = weight
        self._eps = 1e-9
        self._current_value = None
        self._current_penalty = None

    @property
    def parameter_name(self) -> str:
        return self.parameter.name
    
    @property
    def current_value(self) -> Optional[float | np.ndarray]:
        return self._current_value
    
    @current_value.setter
    def current_value(self, value: Optional[float | np.ndarray]):
        self._current_value = value

    @property
    def current_penalty(self) -> Optional[float]:
        return self._current_penalty

    @current_penalty.setter
    def current_penalty(self, value: Optional[float]):
        self._current_penalty = value

    @property
    def required_simulation_type(self) -> SimulationType:
        return self.parameter.simulation_type

    def penalty(self, sim_result: SimulationResult) -> float:
        """Calculate the penalty for the given simulation result."""
        # Calculates the value of the parameter (e.g. S21_dB) from the simulation result
        values = self.parameter.formula(sim_result)

        # Store the current value for later reference
        self.current_value = values  

        # Calculate the penalty using the parameter's loss function and the goal's min/max values
        penalty_value = self.parameter.loss(self.min_value, self.max_value, values) * self.weight
        self.current_penalty = penalty_value
        return penalty_value
    
    def __str__(self):
        freq_range_str = self.frequency_range if self.frequency_range else "full range"
        return (
            f"DesignGoal(parameter={self.parameter.name}, "
            f"frequency_range={freq_range_str}, "
            f"min_value={self.min_value}, max_value={self.max_value}, "
            f"weight={self.weight}, "
            f"current_penalty={self.current_penalty}"
        )


# ---------------------------------------------------------------------------
# DesignGoalChecker — evaluates all goals against simulation results
# ---------------------------------------------------------------------------


class DesignGoalChecker:
    """Evaluates a list of DesignGoals against one or more simulation results."""

    def __init__(self, design_goals: list[DesignGoal]):
        # Group goals by simulation type for efficient evaluation
        self.design_goals: Dict[SimulationType, list[DesignGoal]] = {}
        for goal in design_goals:
            st = goal.required_simulation_type
            self.design_goals.setdefault(st, []).append(goal)

    def check_goals(self, context: dict) -> dict:
        """
        Evaluate all goals and update *context* with results.

        Reads ``context["simulation_results"]`` (``dict[SimulationType, SimulationResult]``).
        """
        sim_results: dict = context.get("simulation_results") or {}
        penalties = self.loss(sim_results)

        context["goal_achieved"] = all(p <= 0.0 for p in penalties)
        context["goals"] = [goal for goals in self.design_goals.values() for goal in goals]
        
        return context

    def loss(self, sim_results: dict[SimulationType, SimulationResult]) -> list[float]:
        """
        Compute penalties for each goal using the appropriate network.

        Parameters
        ----------
        sim_results:
            ``{SimulationType: SimulationResult}`` — one entry per required analysis type.
        """
        penalties = []

        # Iterate over all results and check all goals that require that simulation type
        for sim_type, sim_result in sim_results.items():
            goals_for_type = self.design_goals.get(sim_type, [])
            if not goals_for_type:
                continue  # No goals for this simulation type

            for goal in goals_for_type:
                ntwk_backup = sim_result.network
                if sim_type is SimulationType.AC and goal.frequency_range:
                    # Slice network to the goal's frequency range if specified
                    ntwk_backup = sim_result.network.copy()
                    if goal.frequency_range:
                        sim_result.network = ntwk_backup[goal.frequency_range]
                    
                # TODO: remove this, will be unnecessary since we store current results in the goal object
                values = goal.parameter.formula(sim_result)
                penalties.append(goal.penalty(sim_result))
                

                sim_result.network = ntwk_backup

        return penalties

    def __str__(self):
        parts = [
            str(g) for goals in self.design_goals.values() for g in goals
        ]
        return "Design Goals: " + " | ".join(parts)
