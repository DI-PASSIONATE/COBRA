"""Tests for :class:`OptimizationContext`.

The stage-by-stage behaviour is covered by the stage tests; what is pinned here
is the JSON view, because ``cobra_optimization_context.json`` is the run's only
durable record and nothing else would notice if it stopped being serialisable.
"""

from __future__ import annotations

import json

import pytest

from cobra.optimization_context import OptimizationContext
from cobra.spice_sim.base_simulator import SimulationResult
from cobra.spice_sim.simulation_type import SimulationType
from tests.conftest import make_context


def test_enum_keys_become_strings_so_the_context_is_json_serialisable():
    context = make_context(
        native_sim_type=SimulationType.AC,
        simulation_results={SimulationType.AC: SimulationResult()},
        sim_params_by_type={SimulationType.HB: {"numfreq": "5"}},
    )

    payload = context.to_json_dict()

    assert list(payload["simulation_results"]) == [SimulationType.AC.value]
    assert payload["sim_params_by_type"] == {SimulationType.HB.value: {"numfreq": "5"}}
    assert payload["native_sim_type"] == SimulationType.AC.value
    # default=str is what COBRA.run passes for the objects json cannot reach.
    json.dumps(payload, default=str)


def test_the_fields_a_run_cannot_do_without_have_no_default():
    with pytest.raises(TypeError):
        OptimizationContext()  # ty: ignore[missing-argument]


def test_stage_times_start_at_zero():
    assert set(make_context().times) == {
        "optimizer",
        "em_surrogate",
        "circuit_simulation",
        "design_goal_checking",
        "em_fine_tuning",
        "total_time",
    }
    assert all(value == 0.0 for value in make_context().times.values())


def test_each_context_gets_its_own_mutable_fields():
    first, second = make_context(), make_context()
    first.iterations.append({"iteration": 1})
    first.times["optimizer"] += 1.0

    assert second.iterations == []
    assert second.times["optimizer"] == 0.0
