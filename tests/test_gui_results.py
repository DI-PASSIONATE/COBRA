"""The main window's goal display during and after a run, headless (Qt's offscreen platform)."""

from __future__ import annotations

import numpy as np

from cobra.optimizers.design_goal import DesignGoal, DesignParameter
from cobra.optimizers.design_goal_collection import calculate_array_penalty
from cobra.spice_sim.simulation_type import SimulationType
from tests.conftest import make_context


def _goal(value: np.ndarray | None, penalty: float) -> DesignGoal:
    parameter = DesignParameter(
        name="S21_dB",
        simulation_type=SimulationType.AC,
        formula=lambda _result, _freq=None: np.array([0.0]),
        loss=calculate_array_penalty,
    )
    goal = DesignGoal(parameter, min_value=6.5)
    goal.current_value = value
    goal.current_penalty = penalty
    return goal


def test_the_final_result_replaces_a_failed_last_trial(window):
    """A run that misses its goals ends on the best parameters, not on its last trial."""
    window.loss_history = {0: []}
    failed_last_trial = make_context(iteration=600, goals=[_goal(None, 1e6)])
    best = make_context(iteration=600, goals=[_goal(np.array([5.8, 6.2]), 0.67)])

    window.on_progress(failed_last_trial)
    assert window.goal_table.item(0, 2).text().startswith("N/A")

    window.on_result(best)
    assert window.goal_table.item(0, 2).text().startswith("[5.8000, 6.2000] (penalty 0.67, violated)")
    assert window.loss_history == {0: [1e6]}  # the final result is not another iteration


def test_the_worker_hands_over_the_final_context_before_it_finishes(window):
    from types import SimpleNamespace

    from cobra.gui.worker import OptimizationWorker

    final = make_context(iteration=3)
    configured_run = SimpleNamespace(
        configuration=SimpleNamespace(max_iterations=3), run=lambda _callback: final
    )
    worker = OptimizationWorker(configured_run)  # ty: ignore[invalid-argument-type]
    events = []
    worker.result.connect(lambda context: events.append(("result", context)))
    worker.finished.connect(lambda: events.append(("finished", None)))

    worker.run()  # in this thread: the signals reach the lambdas directly
    assert events == [("result", final), ("finished", None)]


def test_a_burst_of_trials_redraws_once_and_records_every_loss(window):
    """Fast parallel trials must not redraw per trial, but none may drop out of the loss history."""
    window.loss_history = {0: []}
    trials = [make_context(iteration=i, goals=[_goal(np.array([float(i)]), float(i))]) for i in (1, 2, 3)]

    for trial in trials:
        window.on_progress(trial)
    assert window.goal_table.item(0, 2).text().startswith("[1.0000, 1.0000]")  # the first shows at once
    assert window.loss_history == {0: [1.0, 2.0, 3.0]}

    window._run_display_timer.timeout.emit()  # the interval elapses
    assert window.goal_table.item(0, 2).text().startswith("[3.0000, 3.0000]")  # the newest, not the second


def test_a_pending_trial_does_not_overwrite_the_final_result(window):
    window.loss_history = {0: []}
    window.on_progress(make_context(iteration=1, goals=[_goal(np.array([1.0]), 1.0)]))
    window.on_progress(make_context(iteration=2, goals=[_goal(None, 1e6)]))  # throttled

    window.on_result(make_context(iteration=2, goals=[_goal(np.array([6.8]), 0.0)]))
    window._run_display_timer.timeout.emit()
    assert window.goal_table.item(0, 2).text().startswith("[6.8000, 6.8000] (penalty 0.00, met)")
