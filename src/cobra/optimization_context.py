"""The state that flows through every stage of an optimization run.

One :class:`OptimizationContext` is created by :meth:`cobra.cobra.COBRA.run`
and handed to each stage in turn.  A stage reads the fields it needs, writes
the fields it owns, and returns the same object.

Field ownership is what keeps the pipeline honest, so it is recorded per
group below: everything a stage does not own it should treat as read-only.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cobra.spice_sim.simulation_type import SimulationType

if TYPE_CHECKING:
    import skrf as rf

    from cobra.optimizers.base_optimizer import OptimizationProperty
    from cobra.optimizers.design_goal import DesignGoal, DesignGoalChecker
    from cobra.spice_sim.base_simulator import SimulationResult

#: Wall-clock seconds accumulated per stage over the whole run.
STAGE_TIME_KEYS = (
    "optimizer",
    "em_surrogate",
    "circuit_simulation",
    "design_goal_checking",
    "em_fine_tuning",
    "total_time",
)


def _zeroed_stage_times() -> dict[str, float]:
    return dict.fromkeys(STAGE_TIME_KEYS, 0.0)


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert *obj* to a JSON-safe structure.

    Enum keys and values become their ``.value``; anything else that is not a
    native JSON type is left for ``json.dump``'s ``default=str`` fallback.
    Needed because ``simulation_results`` and ``sim_params_by_type`` are keyed
    by :class:`SimulationType`, and JSON object keys must be strings.
    """
    if isinstance(obj, dict):
        return {
            (k.value if isinstance(k, Enum) else k): sanitize_for_json(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    return obj


@dataclass(kw_only=True)
class OptimizationContext:
    """Mutable state shared by the stages of a single optimization run."""

    # --- Fixed for the whole run, written by COBRA.run ---------------------
    netlist: str
    """Path to the working copy of the netlist inside ``results_dir``."""
    design_goal_checker: DesignGoalChecker
    """Evaluates the goals; also tells the simulation stage which types to run."""
    native_sim_type: SimulationType = SimulationType.UNKNOWN
    optimization_parameters: list[OptimizationProperty] = field(default_factory=list)
    max_iterations: int = 0
    """Raised mid-run by the GUI when the user chooses to keep going."""
    results_dir: str = "."
    orca_geometries: dict[str, Any] = field(default_factory=dict)
    sim_params_by_type: dict[SimulationType, dict[str, str]] = field(default_factory=dict)

    # --- Current iteration, written by OptimizerStage ----------------------
    iteration: int = 0
    model_parameters: dict[str, Any] = field(default_factory=dict)
    netlist_parameters: dict[str, str] = field(default_factory=dict)
    trial: Any = None
    """Optimizer-owned handle for the trial in flight (an Optuna ``Trial``).

    Opaque to every other stage, and left ``None`` by optimizers that do not
    need one.
    """

    # --- Written by EMSurrogateStage / EMFineTuningStage -------------------
    predicted_networks: list[rf.Network] = field(default_factory=list)

    # --- Written by CircuitSimulationStage ---------------------------------
    simulation_results: dict[SimulationType, SimulationResult] = field(default_factory=dict)
    """One entry per simulation type that succeeded.

    A type that failed is absent, which
    :meth:`~cobra.optimizers.design_goal.DesignGoalChecker.check_goals` turns
    into ``FAILED_SIMULATION_PENALTY`` rather than a satisfied goal.
    """

    # --- Written by DesignGoalChecker.check_goals --------------------------
    goal_achieved: bool = False
    goals: list[DesignGoal] = field(default_factory=list)

    # --- EM fine-tuning phase ----------------------------------------------
    fine_tuning_active: bool = False
    fine_tuning_iteration: int = 0
    fine_tuning_total: int = 0
    fine_tuning_start_iteration: int | None = None
    """Iteration at which the goals were met, when fine-tuning follows a success."""

    # --- Run log ------------------------------------------------------------
    iterations: list[dict[str, Any]] = field(default_factory=list)
    """One record per optimizer step, appended by :meth:`OptimizerStage.tell`."""
    times: dict[str, float] = field(default_factory=_zeroed_stage_times)
    """Seconds spent per stage, summed over every trial.

    With trials running concurrently these add up to more than the run took:
    they measure work done, not time elapsed. For elapsed time use
    :attr:`wall_time`.
    """
    wall_time: float = 0.0
    """Seconds of real elapsed time for the whole run, set by :meth:`COBRA.run`."""

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of this context."""
        return sanitize_for_json(vars(self))

    # --- Parallel trial evaluation ------------------------------------------

    def for_trial(self, trial_dir: Path) -> OptimizationContext:
        """Return a context for one trial, evaluated inside *trial_dir*.

        Trials may run concurrently, so everything a trial writes has to be its
        own: its working directory, its netlist copy, and its design goals —
        :meth:`~cobra.optimizers.design_goal.DesignGoal.penalty` stores the
        value and penalty on the goal object itself.  The run-wide fields are
        shared by reference and must be treated as read-only by the trial.
        """
        return replace(
            self,
            results_dir=str(trial_dir),
            netlist=str(trial_dir / Path(self.netlist).name),
            design_goal_checker=deepcopy(self.design_goal_checker),
            model_parameters={},
            netlist_parameters={},
            trial=None,
            predicted_networks=[],
            simulation_results={},
            goals=[],
            goal_achieved=False,
            iterations=[],
            times=_zeroed_stage_times(),
            wall_time=0.0,
        )

    def absorb_trial(self, trial: OptimizationContext, include_times: bool = True) -> None:
        """Merge a finished trial's results into this run-wide context.

        The stage times are summed rather than replaced, so with more than one
        trial in flight they add up to more than the run's wall-clock time.
        Pass ``include_times=False`` to re-apply a trial whose time was already
        counted, which happens when a successful trial has to be reinstated
        after the trials that were still in flight alongside it.
        """
        self.iteration = trial.iteration
        self.trial = trial.trial
        self.model_parameters = trial.model_parameters
        self.netlist_parameters = trial.netlist_parameters
        self.predicted_networks = trial.predicted_networks
        self.simulation_results = trial.simulation_results
        self.goals = trial.goals
        self.goal_achieved = trial.goal_achieved
        if include_times:
            for key, seconds in trial.times.items():
                self.times[key] = self.times.get(key, 0.0) + seconds
