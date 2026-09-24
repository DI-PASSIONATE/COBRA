"""Tests for :class:`GradientDescentOptimizer` convergence."""

from __future__ import annotations

import pytest

from cobra.optimizers import GradientDescentOptimizer
from cobra.optimizers.base_optimizer import OptimizationProperty, OptimizationType
from tests.conftest import make_context


@pytest.mark.parametrize("start", [7.0, 1.0])
def test_descends_to_the_minimum_of_a_quadratic(start):
    """Regression: the finite-difference gradient had the wrong sign, so the
    base point walked away from the minimum (7 → 9.5 on ``(x−3)²``).
    """
    x_range = OptimizationProperty("x", OptimizationType.MODEL_INPUT, 0.0, 10.0)
    optimizer = GradientDescentOptimizer(random_seed=0)
    optimizer.initialize(num_goals=1)
    context = make_context(model_parameters={"x": start})

    for _ in range(40):
        optimizer.step(context, [x_range], [])
        optimizer.tell(context, (context.model_parameters["x"] - 3.0) ** 2)

    assert optimizer._current_point["x"] == pytest.approx(3.0, abs=0.5)
