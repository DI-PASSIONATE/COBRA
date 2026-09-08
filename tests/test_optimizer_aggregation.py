"""Tests for :meth:`BaseOptimizer._tell` — how per-goal losses become one penalty.

This aggregation rule is undocumented outside the source comment but every
optimizer depends on it, so it is pinned here before anyone changes it.
"""

from __future__ import annotations

from typing import Any

import pytest

from cobra.optimizers.base_optimizer import BaseOptimizer, OptimizationProperty, OptimizationType
from tests.conftest import make_context


class RecordingOptimizer(BaseOptimizer):
    """Minimal concrete optimizer that just records what ``tell`` receives."""

    def __init__(self, multi_objective: bool = False):
        super().__init__(multi_objective)
        self.told: list[Any] = []

    def initialize(self, num_goals: int):
        self.num_goals = num_goals

    def step(self, context, model_input_ranges, netlist_property_ranges):  # pragma: no cover
        raise NotImplementedError

    def tell(self, context, penalty):
        self.told.append(penalty)
        return penalty

    def get_moo_results(self):  # pragma: no cover
        return []

    def get_best_parameters(self):  # pragma: no cover
        return {}


@pytest.fixture
def optimizer() -> RecordingOptimizer:
    return RecordingOptimizer()


def test_all_positive_losses_are_summed(optimizer):
    assert optimizer._tell(make_context(), [1.0, 2.0, 3.0]) == pytest.approx(6.0)


def test_all_negative_losses_are_summed(optimizer):
    """With every goal satisfied, total margin is the score to keep improving."""
    assert optimizer._tell(make_context(), [-1.0, -2.0]) == pytest.approx(-3.0)


def test_mixed_signs_discard_the_negative_margin(optimizer):
    """Once any goal is violated, satisfied-goal margin stops counting."""
    assert optimizer._tell(make_context(), [-5.0, 2.0, -1.0, 3.0]) == pytest.approx(5.0)


def test_zero_counts_as_non_negative(optimizer):
    """A list of zeros and positives takes the all-non-negative branch."""
    assert optimizer._tell(make_context(), [0.0, 4.0]) == pytest.approx(4.0)


def test_zeros_and_negatives_are_summed(optimizer):
    assert optimizer._tell(make_context(), [0.0, -4.0]) == pytest.approx(-4.0)


def test_empty_loss_list_sums_to_zero(optimizer):
    assert optimizer._tell(make_context(), []) == pytest.approx(0.0)


def test_multi_objective_forwards_the_raw_list():
    optimizer = RecordingOptimizer(multi_objective=True)
    losses = [-5.0, 2.0]

    assert optimizer._tell(make_context(), losses) == losses
    assert optimizer.told == [losses]


def test_single_objective_forwards_a_scalar(optimizer):
    optimizer._tell(make_context(), [-5.0, 2.0])
    assert optimizer.told == [pytest.approx(2.0)]


# ---------------------------------------------------------------------------
# OptimizationProperty / OptimizationType
# ---------------------------------------------------------------------------


def test_optimization_type_values_match_the_config_schema():
    """These strings appear verbatim in run-configuration JSON."""
    assert OptimizationType.NETLIST_VARIABLE.value == "netlist_variable"
    assert OptimizationType.MODEL_INPUT.value == "model_input"


def test_optimization_property_defaults():
    prop = OptimizationProperty(
        name="R1", type=OptimizationType.NETLIST_VARIABLE, min_value=1.0, max_value=2.0
    )
    assert prop.step is None
    assert prop.unit is None
    assert prop.linked_to is None
