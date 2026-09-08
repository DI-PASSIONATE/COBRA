"""Tests for what ``cobra run`` reports when a run finishes.

The report is the only place a user reads the design COBRA settled on, so the
values it prints have to survive the awkward cases: a parameter that follows
another, a value that is not a number, and a goal whose simulation produced
nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

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
# Design parameters
# ---------------------------------------------------------------------------


def test_the_final_value_is_reported_on_its_range(capsys):
    out = _report(
        capsys,
        optimization_parameters=[_parameter("R1", unit="F")],
        netlist_parameters={"R1": "55.0F"},
    )

    assert "Design parameters" in out
    assert "R1" in out
    assert "55F" in out


def test_a_linked_parameter_reports_its_master_rather_than_a_range(capsys):
    out = _report(
        capsys,
        optimization_parameters=[_parameter("C4", linked_to="C3")],
        # Written with the master's unit, which must not defeat the parsing.
        netlist_parameters={"C4": "19.0F"},
    )

    assert "follows C3" in out
    assert "19" in out


def test_a_value_that_is_not_a_number_is_shown_as_missing(capsys):
    out = _report(
        capsys,
        optimization_parameters=[_parameter("R1")],
        netlist_parameters={"R1": "{expr}"},
    )

    assert "Design parameters" in out
    assert "—" in out


def test_a_parameter_with_no_value_at_all_does_not_break_the_report(capsys):
    out = _report(capsys, optimization_parameters=[_parameter("R1")], netlist_parameters={})

    assert "R1" in out


def test_the_section_is_skipped_when_there_are_no_parameters(capsys):
    assert "Design parameters" not in _report(capsys)


# ---------------------------------------------------------------------------
# Design goals
# ---------------------------------------------------------------------------


def test_a_met_goal_is_ticked_and_reports_what_it_reached(capsys):
    out = _report(capsys, goals=[_goal(value=-14.5, penalty=-4.5)])

    assert "Design goals" in out
    assert "✓" in out
    assert "target <= -10" in out
    assert "reached -14.5" in out


def test_an_unmet_goal_is_crossed(capsys):
    out = _report(capsys, goals=[_goal(value=-2.0, penalty=8.0)])

    assert "✗" in out


def test_an_array_of_values_is_reported_as_a_span(capsys):
    out = _report(capsys, goals=[_goal(value=np.array([-20.0, -12.0, -15.0]), penalty=-1.0)])

    assert "reached -20 … -12" in out


def test_a_goal_whose_simulation_failed_says_so(capsys):
    out = _report(capsys, goals=[_goal(value=None, penalty=1e6)])

    assert "no result" in out
    assert "✗" in out


@pytest.mark.parametrize(
    ("minimum", "maximum", "expected"),
    [
        (None, -10.0, "target <= -10"),
        (-3.0, None, "target >= -3"),
        (-3.0, 3.0, "target >= -3 and <= 3"),
        (None, None, "target no bound"),
    ],
)
def test_the_target_reads_as_the_bounds_that_were_set(capsys, minimum, maximum, expected):
    goal = _goal(value=0.0, penalty=0.0, max_value=maximum)
    goal.min_value = minimum

    assert expected in _report(capsys, goals=[goal])


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
