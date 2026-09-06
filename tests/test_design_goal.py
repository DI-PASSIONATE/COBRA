"""Tests for design-goal parsing, penalty math and goal grouping."""

from __future__ import annotations

import numpy as np
import pytest

from cobra.optimizers.design_goal import DesignGoal, DesignGoalChecker, DesignParameter
from cobra.optimizers.design_goal_collection import calculate_array_penalty
from cobra.spice_sim.base_simulator import SimulationResult
from cobra.spice_sim.simulation_type import SimulationType

# ---------------------------------------------------------------------------
# Frequency range parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (None, (None, None)),
        ("5", (5.0, 5.0)),
        ("0.5", (0.5, 0.5)),
        ("1-2", (1.0, 2.0)),
        ("5GHz", (5e9, 5e9)),
        ("130-140ghz", (130e9, 140e9)),
        ("1-20MHz", (1e6, 20e6)),
        ("2.5k", (2.5e3, 2.5e3)),
        ("  5GHz  ", (5e9, 5e9)),
    ],
)
def test_str_to_frequency_range(text, expected):
    assert DesignGoal.str_to_frequency_range(text) == expected


def test_single_frequency_collapses_to_an_equal_pair():
    """A lone frequency becomes (f, f) so downstream slicing snaps to one bin."""
    assert DesignGoal.str_to_frequency_range("5GHz") == (5e9, 5e9)


@pytest.mark.parametrize("text", ["", "abc", "GHz", "1--2", "1-2-3"])
def test_malformed_frequency_range_raises(text):
    with pytest.raises(ValueError, match="Invalid frequency range format"):
        DesignGoal.str_to_frequency_range(text)


def test_fully_uppercase_hz_is_not_accepted():
    """Current limitation: the unit pattern accepts 'Hz'/'hz' but not 'HZ'.

    Pinned so the restriction is visible; see the backlog item on unifying the
    three frequency parsers.
    """
    with pytest.raises(ValueError, match="Invalid frequency range format"):
        DesignGoal.str_to_frequency_range("5GHZ")


# ---------------------------------------------------------------------------
# Penalty math
# ---------------------------------------------------------------------------


def test_penalty_is_zero_inside_a_two_sided_band():
    assert calculate_array_penalty(-20.0, -10.0, np.array([-15.0, -12.0])) == 0.0


def test_penalty_is_positive_below_a_two_sided_band():
    values = np.array([-30.0])
    expected = ((-20.0 - -30.0) / abs(-20.0)) ** 2
    assert calculate_array_penalty(-20.0, -10.0, values) == pytest.approx(expected)


def test_penalty_is_positive_above_a_two_sided_band():
    values = np.array([-5.0])
    expected = ((-5.0 - -10.0) / abs(-10.0)) ** 2
    assert calculate_array_penalty(-20.0, -10.0, values) == pytest.approx(expected)


def test_two_sided_penalty_sums_over_all_violating_points():
    values = np.array([-30.0, -15.0, -5.0])
    below = ((-20.0 - -30.0) / 20.0) ** 2
    above = ((-5.0 - -10.0) / 10.0) ** 2
    assert calculate_array_penalty(-20.0, -10.0, values) == pytest.approx(below + above)


def test_min_only_penalty_is_positive_when_violated():
    assert calculate_array_penalty(0.0, None, np.array([-1.0])) > 0


def test_min_only_penalty_is_negative_when_satisfied():
    """A single-sided goal rewards margin, so satisfied values score below zero."""
    assert calculate_array_penalty(10.0, None, np.array([20.0])) < 0


def test_max_only_penalty_is_positive_when_violated():
    assert calculate_array_penalty(None, 10.0, np.array([20.0])) > 0


def test_max_only_penalty_is_negative_when_satisfied():
    assert calculate_array_penalty(None, 10.0, np.array([5.0])) < 0


def test_single_violation_dominates_a_single_sided_goal():
    """One violating point makes the whole goal positive, ignoring satisfied ones."""
    assert calculate_array_penalty(None, 10.0, np.array([1.0, 2.0, 50.0])) > 0


def test_scalar_values_are_accepted():
    assert calculate_array_penalty(None, 10.0, 20.0) == pytest.approx(
        calculate_array_penalty(None, 10.0, np.array([20.0]))
    )


def test_penalty_requires_at_least_one_bound():
    with pytest.raises(ValueError, match="At least one of min_value or max_value"):
        calculate_array_penalty(None, None, np.array([1.0]))


def test_zero_bound_does_not_divide_by_zero():
    """The eps floor keeps a bound of exactly 0 finite."""
    assert np.isfinite(calculate_array_penalty(0.0, None, np.array([-1.0])))


# ---------------------------------------------------------------------------
# DesignParameter identity
# ---------------------------------------------------------------------------


def _parameter(name: str, sim_type: SimulationType = SimulationType.AC) -> DesignParameter:
    return DesignParameter(
        name=name,
        simulation_type=sim_type,
        formula=lambda _result, _freq=None: np.array([0.0]),
        loss=calculate_array_penalty,
    )


def test_design_parameters_compare_by_name_only():
    """DesignGoalChecker groups goals in dicts, so identity must follow the name."""
    assert _parameter("S21_dB") == _parameter("S21_dB")
    assert hash(_parameter("S21_dB")) == hash(_parameter("S21_dB"))
    assert _parameter("S21_dB") != _parameter("S11_dB")


def test_design_parameter_comparison_with_other_types_is_not_implemented():
    assert _parameter("S21_dB").__eq__("S21_dB") is NotImplemented


# ---------------------------------------------------------------------------
# DesignGoal
# ---------------------------------------------------------------------------


def test_design_goal_rejects_a_bare_string_parameter():
    with pytest.raises(TypeError, match="must be a DesignParameter"):
        DesignGoal("S21_dB", max_value=1.0)  # ty: ignore[invalid-argument-type]


def test_penalty_applies_the_weight_and_caches_the_value():
    parameter = DesignParameter(
        name="fake",
        simulation_type=SimulationType.AC,
        formula=lambda _result, _freq=None: np.array([20.0]),
        loss=calculate_array_penalty,
    )
    goal = DesignGoal(parameter, max_value=10.0, weight=3.0)

    unweighted = calculate_array_penalty(None, 10.0, np.array([20.0]))
    assert goal.penalty(SimulationResult()) == pytest.approx(unweighted * 3.0)
    assert goal.current_value == pytest.approx(np.array([20.0]))
    assert goal.current_penalty == pytest.approx(unweighted * 3.0)


def test_required_simulation_type_comes_from_the_parameter():
    goal = DesignGoal(_parameter("x", SimulationType.HB), max_value=1.0)
    assert goal.required_simulation_type is SimulationType.HB


# ---------------------------------------------------------------------------
# DesignGoalChecker
# ---------------------------------------------------------------------------


def _goal(value: float, *, max_value: float, sim_type: SimulationType) -> DesignGoal:
    parameter = DesignParameter(
        name=f"p{value}",
        simulation_type=sim_type,
        formula=lambda _result, _freq=None, _v=value: np.array([_v]),
        loss=calculate_array_penalty,
    )
    return DesignGoal(parameter, max_value=max_value)


def test_checker_groups_goals_by_simulation_type():
    checker = DesignGoalChecker(
        [
            _goal(1.0, max_value=10.0, sim_type=SimulationType.AC),
            _goal(2.0, max_value=10.0, sim_type=SimulationType.HB),
            _goal(3.0, max_value=10.0, sim_type=SimulationType.AC),
        ]
    )
    assert set(checker.design_goals) == {SimulationType.AC, SimulationType.HB}
    assert len(checker.design_goals[SimulationType.AC]) == 2


def test_check_goals_marks_success_when_every_penalty_is_non_positive():
    checker = DesignGoalChecker([_goal(5.0, max_value=10.0, sim_type=SimulationType.AC)])
    context = checker.check_goals({"simulation_results": {SimulationType.AC: SimulationResult()}})

    assert context["goal_achieved"] is True
    assert len(context["goals"]) == 1


def test_check_goals_marks_failure_when_any_penalty_is_positive():
    checker = DesignGoalChecker(
        [
            _goal(5.0, max_value=10.0, sim_type=SimulationType.AC),
            _goal(50.0, max_value=10.0, sim_type=SimulationType.AC),
        ]
    )
    context = checker.check_goals({"simulation_results": {SimulationType.AC: SimulationResult()}})

    assert context["goal_achieved"] is False


def test_goals_without_matching_results_are_skipped():
    """Only simulation types actually present in the results are evaluated."""
    checker = DesignGoalChecker([_goal(50.0, max_value=10.0, sim_type=SimulationType.HB)])
    assert checker.loss({SimulationType.AC: SimulationResult()}) == []


def test_check_goals_with_no_results_reports_success_vacuously():
    """``all([])`` is True — an empty result set currently counts as achieved.

    Pinned as current behaviour; it is the reason a silently failed simulation
    can look like a satisfied run (see the backlog item on Xyce error handling).
    """
    checker = DesignGoalChecker([_goal(50.0, max_value=10.0, sim_type=SimulationType.AC)])
    assert checker.check_goals({"simulation_results": {}})["goal_achieved"] is True
