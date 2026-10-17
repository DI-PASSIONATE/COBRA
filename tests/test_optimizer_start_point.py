"""Tests for the ``initial_value`` of optimization parameters: the point an optimizer evaluates first."""

from __future__ import annotations

from cobra.optimizers import OptunaOptimizer
from cobra.optimizers.base_optimizer import OptimizationProperty, OptimizationType
from tests.conftest import make_context


def _netlist_property(name: str, **kwargs) -> OptimizationProperty:
    return OptimizationProperty(name, OptimizationType.NETLIST_VARIABLE, 300.0, 400.0, step=1.0, unit="f", **kwargs)


def _first_points(params: list[OptimizationProperty], trials: int = 2) -> list[dict]:
    optimizer = OptunaOptimizer(random_seed=0)
    optimizer.initialize(num_goals=1)
    points = []
    for _ in range(trials):
        context = make_context()
        optimizer.step(context, [], params)
        points.append(context.netlist_parameters)
        optimizer.tell(context, 1.0)
    return points


def test_optuna_evaluates_the_start_point_first():
    params = [_netlist_property("C5", initial_value=377.0), _netlist_property("Cout", initial_value=78.0 + 300)]

    first, second = _first_points(params)

    assert first == {"C5": "377.0f", "Cout": "378.0f"}
    assert second != first


def test_optuna_samples_the_parameters_without_a_start_value():
    params = [_netlist_property("C5", initial_value=377.0), _netlist_property("Cout")]

    first, _ = _first_points(params)

    assert first["C5"] == "377.0f"
    assert 300.0 <= float(first["Cout"][:-1]) <= 400.0


def test_a_linked_parameter_takes_its_masters_start_value():
    params = [_netlist_property("Cp", initial_value=350.0), _netlist_property("Cn", linked_to="Cp")]

    first, _ = _first_points(params)

    assert first == {"Cp": "350.0f", "Cn": "350.0f"}


def test_a_reinitialised_study_queues_the_start_point_again():
    optimizer = OptunaOptimizer(random_seed=0)
    params = [_netlist_property("C5", initial_value=377.0)]
    for _ in range(2):
        optimizer.initialize(num_goals=1)
        context = make_context()
        optimizer.step(context, [], params)
        assert context.netlist_parameters == {"C5": "377.0f"}
        optimizer.tell(context, 1.0)
