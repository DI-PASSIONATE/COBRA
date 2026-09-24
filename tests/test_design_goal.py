"""Tests for design-goal parsing, penalty math and goal checking."""

from __future__ import annotations

import numpy as np
import pytest

from cobra.optimizers.design_goal import DesignGoal, DesignGoalChecker, DesignParameter
from cobra.optimizers.design_goal_collection import calculate_array_penalty
from cobra.spice_sim.base_simulator import SimulationResult
from cobra.spice_sim.simulation_type import SimulationType
from tests.conftest import make_context

# ---------------------------------------------------------------------------
# Frequency range parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (None, (None, None)),
        # A lone frequency becomes (f, f) so downstream slicing snaps to one bin.
        ("5GHz", (5e9, 5e9)),
        ("130-140ghz", (130e9, 140e9)),
        ("1-20MHz", (1e6, 20e6)),
    ],
)
def test_str_to_frequency_range(text, expected):
    assert DesignGoal.str_to_frequency_range(text) == expected


def test_malformed_frequency_range_raises():
    for text in ("abc", "1-2-3"):
        with pytest.raises(ValueError, match="Invalid frequency range format"):
            DesignGoal.str_to_frequency_range(text)


# ---------------------------------------------------------------------------
# Penalty math
# ---------------------------------------------------------------------------


def test_two_sided_penalty_is_zero_inside_and_sums_violations_outside():
    assert calculate_array_penalty(-20.0, -10.0, np.array([-15.0, -12.0])) == 0.0

    values = np.array([-30.0, -15.0, -5.0])
    below = ((-20.0 - -30.0) / 20.0) ** 2
    above = ((-5.0 - -10.0) / 10.0) ** 2
    assert calculate_array_penalty(-20.0, -10.0, values) == pytest.approx(below + above)


def test_single_sided_penalty_rewards_margin_but_any_violation_dominates():
    assert calculate_array_penalty(None, 10.0, np.array([5.0])) < 0
    assert calculate_array_penalty(10.0, None, np.array([20.0])) < 0
    # One violating point makes the whole goal positive, ignoring satisfied ones.
    assert calculate_array_penalty(None, 10.0, np.array([1.0, 2.0, 50.0])) > 0


def test_penalty_requires_at_least_one_bound():
    with pytest.raises(ValueError, match="At least one of min_value or max_value"):
        calculate_array_penalty(None, None, np.array([1.0]))


# ---------------------------------------------------------------------------
# DesignParameter and DesignGoal
# ---------------------------------------------------------------------------


def _parameter(name: str, value: float = 0.0) -> DesignParameter:
    return DesignParameter(
        name=name,
        simulation_type=SimulationType.AC,
        formula=lambda _result, _freq=None, _v=value: np.array([_v]),
        loss=calculate_array_penalty,
    )


def test_design_parameters_compare_by_name_only():
    """DesignGoalChecker groups goals in dicts, so identity must follow the name."""
    assert _parameter("S21_dB") == _parameter("S21_dB", value=1.0)
    assert hash(_parameter("S21_dB")) == hash(_parameter("S21_dB"))
    assert _parameter("S21_dB") != _parameter("S11_dB")


def test_penalty_applies_the_weight_and_caches_the_value():
    goal = DesignGoal(_parameter("fake", value=20.0), max_value=10.0, weight=3.0)

    unweighted = calculate_array_penalty(None, 10.0, np.array([20.0]))
    assert goal.penalty(SimulationResult()) == pytest.approx(unweighted * 3.0)
    assert goal.current_value == pytest.approx(np.array([20.0]))
    assert goal.current_penalty == pytest.approx(unweighted * 3.0)


def test_check_goals_succeeds_only_when_every_goal_is_met():
    results = {SimulationType.AC: SimulationResult()}
    met = DesignGoal(_parameter("met", value=5.0), max_value=10.0)
    missed = DesignGoal(_parameter("missed", value=50.0), max_value=10.0)

    assert DesignGoalChecker([met]).check_goals(make_context(simulation_results=results)).goal_achieved
    context = DesignGoalChecker([met, missed]).check_goals(make_context(simulation_results=results))
    assert context.goal_achieved is False
    assert len(context.goals) == 2
