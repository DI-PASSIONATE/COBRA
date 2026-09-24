"""Tests for how per-goal losses become the penalty an optimizer minimises.

Every optimizer, and the choice of the best fine-tuning iteration, depends on
this rule, so it is pinned here before anyone changes it.
"""

from __future__ import annotations

from typing import Any

import pytest

from cobra.optimizers.base_optimizer import BaseOptimizer, aggregate_penalty
from tests.conftest import make_context


@pytest.mark.parametrize(
    ("losses", "expected"),
    [
        ([1.0, 2.0, 3.0], 6.0),  # all violated: summed
        ([-1.0, -2.0], -3.0),  # all met: total margin is the score to keep improving
        ([-5.0, 2.0, -1.0, 3.0], 5.0),  # once any goal is violated, margin stops counting
    ],
    ids=["all-violated", "all-met", "mixed"],
)
def test_aggregate_penalty(losses, expected):
    assert aggregate_penalty(losses) == pytest.approx(expected)


class _RecordingOptimizer(BaseOptimizer):
    """Minimal concrete optimizer that just records what ``tell`` receives."""

    def __init__(self, multi_objective: bool = False):
        super().__init__(multi_objective)
        self.told: list[Any] = []

    def initialize(self, num_goals: int, parallel_trials: int = 1):
        pass

    def step(self, context, model_input_ranges, netlist_property_ranges):  # pragma: no cover
        raise NotImplementedError

    def tell(self, context, penalty):
        self.told.append(penalty)

    def get_moo_results(self):  # pragma: no cover
        return []

    def get_best_parameters(self):  # pragma: no cover
        return {}


def test_single_objective_gets_a_scalar_and_multi_objective_the_raw_list():
    single, multi = _RecordingOptimizer(), _RecordingOptimizer(multi_objective=True)

    single._tell(make_context(), [-5.0, 2.0])
    multi._tell(make_context(), [-5.0, 2.0])

    assert single.told == [pytest.approx(2.0)]
    assert multi.told == [[-5.0, 2.0]]
