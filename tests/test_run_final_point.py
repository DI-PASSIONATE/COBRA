"""Tests for the point :meth:`COBRA.run` ends at when no trial meets the goals, and for
what the callback sees of the trials still in flight when the run stops.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from cobra.cobra import COBRA
from cobra.optimizers import OptunaOptimizer
from cobra.optimizers.base_optimizer import (
    OptimizationProperty,
    OptimizationType,
    parse_netlist_value,
)
from cobra.optimizers.design_goal import DesignGoal, DesignParameter
from cobra.optimizers.design_goal_collection import calculate_array_penalty
from cobra.spice_sim.base_simulator import SimulationResult
from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
from cobra.spice_sim.simulation_type import SimulationType
from cobra.spice_sim.xyce_simulator import XyceSimulator
from tests.conftest import netlist_path


class _NetlistEchoSimulator(XyceSimulator):
    """A simulator whose result is the netlist it was given, so a goal can read C1 back."""

    def __init__(self):
        self.runs = 0
        self._lock = threading.Lock()

    def preprocess_ntwk(self, ntwk, name: str = "cobra_output") -> str:
        return name

    def run_simulation(self, netlist_path: str, netlist) -> SimulationResult | None:
        with self._lock:
            self.runs += 1
        return SimulationResult(output_files=[netlist_path])


def _c1(result: SimulationResult, _freq=None) -> np.ndarray:
    line = next(line for line in Path(result.output_files[0]).read_text().splitlines() if line.startswith("C1 "))
    return np.array([parse_netlist_value(line.split()[3])])


def _run(tmp_path, monkeypatch, *, max_iterations=6, parallel_trials=1, callback=None):
    monkeypatch.chdir(tmp_path)
    netlist = tmp_path / "circuit.cir"
    netlist.write_text(netlist_path("minimal_ac").read_text(encoding="utf-8"), encoding="utf-8")
    simulator = _NetlistEchoSimulator()
    cobra = COBRA(
        netlist=XyceNetlistParser().parse_file(netlist_path("minimal_ac")),
        component_onnx_mapping={},
        optimizer=OptunaOptimizer(random_seed=3),
        circuit_simulator=simulator,
    )
    goal = DesignGoal(
        DesignParameter("C1", SimulationType.AC, _c1, calculate_array_penalty),
        max_value=1.0,  # C1 ranges over [2, 9]: never met, the smaller the closer
    )
    c1 = OptimizationProperty("C1", OptimizationType.NETLIST_VARIABLE, 2.0, 9.0, step=1.0, unit="p")
    context = cobra.run(
        netlist=str(netlist),
        design_goals=[goal],
        optimization_parameters=[c1],
        max_iterations=max_iterations,
        parallel_trials=parallel_trials,
        callback=callback,
    )
    return context, simulator


def test_without_a_hit_the_run_ends_at_its_closest_trial_without_simulating_again(tmp_path, monkeypatch):
    seen = []
    context, simulator = _run(tmp_path, monkeypatch, callback=lambda ctx: seen.append(ctx.netlist_parameters["C1"]))

    closest = min(seen, key=parse_netlist_value)
    assert not context.goal_achieved
    assert context.netlist_parameters["C1"] == closest
    assert f"C1 n1 0 {closest}" in Path(context.netlist).read_text(encoding="utf-8")
    assert simulator.runs == 6


def test_the_closest_trials_directory_is_kept(tmp_path, monkeypatch):
    context, _ = _run(tmp_path, monkeypatch)

    trial_dirs = list((Path(context.results_dir) / "trials").iterdir())
    assert any(f"{context.netlist_parameters['C1']}" in (path / "circuit.cir").read_text() for path in trial_dirs)
    assert len(trial_dirs) <= 2  # the closest one and the last one


def test_trials_drained_after_the_callback_stops_the_run_are_reported_to_it(tmp_path, monkeypatch):
    calls = []

    def stop_at_once(ctx):
        calls.append(ctx.iteration)
        return False

    context, simulator = _run(tmp_path, monkeypatch, parallel_trials=3, callback=stop_at_once)

    assert simulator.runs == 3
    assert calls == [1, 2, 3]
    assert not context.goal_achieved
