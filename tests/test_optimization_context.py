"""Tests for :class:`OptimizationContext`.

The stage-by-stage behaviour is covered by the stage tests; what is pinned here
is the JSON view, because ``cobra_optimization_context.json`` is the run's only
durable record and nothing else would notice if it stopped being serialisable.
"""

from __future__ import annotations

import json

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
