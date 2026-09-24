"""Tests for what ``cobra run`` reports when a run finishes.

The report is the only place a user reads the design COBRA settled on, so the
values it prints have to survive the awkward cases: a parameter that follows
another, a value that is not a number, and a goal whose simulation produced
nothing.
"""

from __future__ import annotations

import numpy as np

from cobra.console import Palette
from cobra.optimizers.base_optimizer import OptimizationProperty, OptimizationType
from cobra.optimizers.design_goal import DesignGoal, DesignParameter
from cobra.run_report import write_run_report
from cobra.spice_sim.simulation_type import SimulationType
from tests.conftest import make_context

PLAIN = Palette(enabled=False)


def _parameter(name: str, unit: str | None = None, linked_to: str | None = None):
    return OptimizationProperty(
        name=name,
        type=OptimizationType.NETLIST_VARIABLE,
        min_value=0.0 if linked_to else 10.0,
        max_value=0.0 if linked_to else 100.0,
        unit=unit,
        linked_to=linked_to,
    )


def _goal(*, value=None, penalty=None, max_value=-10.0) -> DesignGoal:
    goal = DesignGoal(
        DesignParameter(
            name="S11_dB",
            simulation_type=SimulationType.AC,
            formula=lambda _result, _freq=None: 0.0,
            loss=lambda *_args: 0.0,
        ),
        frequency_range="1-10GHz",
        max_value=max_value,
    )
    goal.current_value = value
    goal.current_penalty = penalty
    return goal


def _report(capsys, **overrides) -> str:
    fields = {"results_dir": "results/demo", "wall_time": 12.0}
    fields.update(overrides)
    write_run_report(make_context(**fields), PLAIN)
    return capsys.readouterr().out


# ---------------------------------------------------------------------------
# Design parameters and goals
# ---------------------------------------------------------------------------


def test_parameters_are_reported_even_when_linked_or_not_numeric(capsys):
    out = _report(
        capsys,
        optimization_parameters=[
            _parameter("R1", unit="F"),
            _parameter("C4", linked_to="C3"),
            _parameter("R2"),
        ],
        # C4 is written with its master's unit, which must not defeat the parsing.
        netlist_parameters={"R1": "55.0F", "C4": "19.0F", "R2": "{expr}"},
    )

    assert "55F" in out
    assert "follows C3" in out
    assert "—" in out  # the expression is shown as missing rather than failing


def test_a_met_goal_is_ticked_and_reports_what_it_reached(capsys):
    out = _report(capsys, goals=[_goal(value=np.array([-20.0, -12.0, -15.0]), penalty=-1.0)])

    assert "✓" in out
    assert "target <= -10" in out
    assert "reached -20 … -12" in out


def test_a_goal_whose_simulation_failed_is_crossed_and_says_so(capsys):
    out = _report(capsys, goals=[_goal(value=None, penalty=1e6)])

    assert "✗" in out
    assert "no result" in out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_the_summary_reports_elapsed_time_not_accumulated_stage_time(capsys):
    """``times`` sums the stages over every trial and may exceed the run."""
    out = _report(capsys, wall_time=90.0, times={"total_time": 300.0}, iteration=12)

    assert "1m 30s" in out
    assert "5m" not in out


def test_the_summary_states_whether_the_goals_were_met(capsys):
    assert "achieved at iteration 7" in _report(capsys, goal_achieved=True, iteration=7)
    assert "not achieved after 7" in _report(capsys, goal_achieved=False, iteration=7)


def test_the_summary_names_the_fine_tuning_phase(capsys):
    """After fine-tuning, ``iteration`` is the fine-tuning iteration that was returned,
    not a count of the surrogate iterations. Stopping before fine-tuning keeps the
    surrogate wording.
    """
    fine_tuned = {"fine_tuning_active": True, "fine_tuning_iteration": 3, "iteration": 2}

    missed = _report(capsys, goal_achieved=False, **fine_tuned)
    assert "not achieved after 3 EM fine-tuning iterations (best: iteration 2)" in missed

    stopped = _report(
        capsys, goal_achieved=False, fine_tuning_active=True, fine_tuning_iteration=0, iteration=30
    )
    assert "not achieved after 30 iterations" in stopped
