"""Tests for :class:`GradientDescentOptimizer` convergence."""

from __future__ import annotations

import pytest

from cobra.optimizers import GradientDescentOptimizer
from cobra.optimizers.base_optimizer import OptimizationProperty, OptimizationType
from tests.conftest import make_context


def _run(optimizer, ranges, penalty, context, trials):
    """Alternate step and tell *trials* times; return the points proposed."""
    proposed = []
    for _ in range(trials):
        optimizer.step(context, ranges, [])
        point = dict(context.model_parameters)
        proposed.append(point)
        optimizer.tell(context, penalty(point))
    return proposed


def _current(optimizer, prop: OptimizationProperty) -> float:
    return prop.min_value + optimizer._point[prop.name] * (prop.max_value - prop.min_value)


@pytest.mark.parametrize("start", [7.0, 1.0])
def test_descends_to_the_minimum_of_a_quadratic(start):
    """Regression: the finite-difference gradient had the wrong sign, so the
    base point walked away from the minimum (7 → 9.5 on ``(x−3)²``).
    """
    x_range = OptimizationProperty("x", OptimizationType.MODEL_INPUT, 0.0, 10.0)
    optimizer = GradientDescentOptimizer(random_seed=0)
    optimizer.initialize(num_goals=1)
    context = make_context(model_parameters={"x": start})

    _run(optimizer, [x_range], lambda point: (point["x"] - 3.0) ** 2, context, 40)

    assert _current(optimizer, x_range) == pytest.approx(3.0, abs=0.5)


def test_steps_below_the_grid_still_add_up():
    """Regression: every update was rounded to the parameter step, so an update
    smaller than one step was lost and the point never moved (1 fF steps, C5 stuck).
    """
    c_range = OptimizationProperty("c", OptimizationType.MODEL_INPUT, 300.0, 400.0, step=1.0)
    optimizer = GradientDescentOptimizer(learning_rate=0.005, random_seed=1)
    optimizer.initialize(num_goals=1)
    context = make_context(model_parameters={"c": 330.0})

    _run(optimizer, [c_range], lambda point: (point["c"] - 377.0) ** 2 / 1e4, context, 60)

    assert _current(optimizer, c_range) > 332.0


def test_starts_at_the_initial_value_and_evaluates_it_first():
    c_range = OptimizationProperty("c", OptimizationType.MODEL_INPUT, 300.0, 400.0, step=1.0, initial_value=377.0)
    optimizer = GradientDescentOptimizer(random_seed=0)
    optimizer.initialize(num_goals=1)

    proposed = _run(optimizer, [c_range], lambda point: abs(point["c"] - 377.0), make_context(), 5)

    assert proposed[0] == {"c": 377.0}
    assert optimizer.get_best_parameters() == {"c": 377.0}


def test_a_start_from_the_context_is_not_evaluated_again():
    """In EM fine-tuning the context holds the design that was just simulated."""
    x_range = OptimizationProperty("x", OptimizationType.MODEL_INPUT, 0.0, 10.0)
    optimizer = GradientDescentOptimizer(random_seed=0)
    optimizer.initialize(num_goals=1)

    proposed = _run(optimizer, [x_range], lambda point: point["x"], make_context(model_parameters={"x": 5.0}), 2)

    assert sorted(point["x"] for point in proposed) == pytest.approx([4.5, 5.5])


def test_without_an_initial_value_it_starts_in_the_middle():
    x_range = OptimizationProperty("x", OptimizationType.MODEL_INPUT, 0.0, 10.0)
    optimizer = GradientDescentOptimizer(random_seed=0)
    optimizer.initialize(num_goals=1)

    assert _run(optimizer, [x_range], lambda point: point["x"], make_context(), 1) == [{"x": 5.0}]
