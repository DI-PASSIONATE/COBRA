"""Tests for how a failed simulation reaches the optimizer.

A simulation that Xyce could not complete (e.g. because the optimizer picked
parameters that describe an invalid geometry) must become a high loss so the
optimizer avoids those parameters, while a technical failure (Xyce missing)
must abort the run.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cobra.optimizers.design_goal import (
    FAILED_SIMULATION_PENALTY,
    DesignGoal,
    DesignGoalChecker,
    DesignParameter,
)
from cobra.optimizers.design_goal_collection import calculate_array_penalty
from cobra.spice_sim.base_simulator import (
    BaseSimulator,
    SimulationResult,
    SimulatorError,
)
from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
from cobra.spice_sim.simulation_type import SimulationType
from cobra.spice_sim.xyce_simulator import XyceSimulator
from cobra.stages.circuit_sim_stage import CircuitSimulationStage
from tests.conftest import netlist_path


def _goal(value: float, *, max_value: float, sim_type: SimulationType) -> DesignGoal:
    parameter = DesignParameter(
        name=f"p{value}",
        simulation_type=sim_type,
        formula=lambda _result, _freq=None, _v=value: np.array([_v]),
        loss=calculate_array_penalty,
    )
    return DesignGoal(parameter, max_value=max_value)


# ---------------------------------------------------------------------------
# DesignGoalChecker — a missing result is a failure, not a satisfied goal
# ---------------------------------------------------------------------------


def test_missing_result_penalises_the_goal_instead_of_skipping_it():
    checker = DesignGoalChecker([_goal(50.0, max_value=10.0, sim_type=SimulationType.HB)])
    assert checker.loss({SimulationType.AC: SimulationResult()}) == [FAILED_SIMULATION_PENALTY]


def test_check_goals_with_no_results_is_a_failure():
    checker = DesignGoalChecker([_goal(50.0, max_value=10.0, sim_type=SimulationType.AC)])
    context = checker.check_goals({"simulation_results": {}})

    assert context["goal_achieved"] is False
    assert context["goals"][0].current_penalty == FAILED_SIMULATION_PENALTY


def test_failure_penalty_ignores_the_goal_weight():
    """A zero-weighted goal must not turn a failed simulation into a success."""
    goal = _goal(50.0, max_value=10.0, sim_type=SimulationType.AC)
    goal.weight = 0.0
    context = DesignGoalChecker([goal]).check_goals({"simulation_results": {}})

    assert context["goal_achieved"] is False


def test_failed_goal_drops_the_value_from_the_previous_iteration():
    goal = _goal(5.0, max_value=10.0, sim_type=SimulationType.AC)
    checker = DesignGoalChecker([goal])

    checker.check_goals({"simulation_results": {SimulationType.AC: SimulationResult()}})
    assert goal.current_value is not None

    checker.check_goals({"simulation_results": {}})
    assert goal.current_value is None


def test_a_partial_failure_only_penalises_the_missing_type():
    ac_goal = _goal(5.0, max_value=10.0, sim_type=SimulationType.AC)
    hb_goal = _goal(5.0, max_value=10.0, sim_type=SimulationType.HB)
    checker = DesignGoalChecker([ac_goal, hb_goal])

    context = checker.check_goals(
        {"simulation_results": {SimulationType.AC: SimulationResult()}}
    )

    assert ac_goal.current_penalty is not None
    assert ac_goal.current_penalty < 0.0
    assert hb_goal.current_penalty == FAILED_SIMULATION_PENALTY
    assert context["goal_achieved"] is False


def test_all_goals_met_still_reports_success():
    checker = DesignGoalChecker([_goal(5.0, max_value=10.0, sim_type=SimulationType.AC)])
    context = checker.check_goals(
        {"simulation_results": {SimulationType.AC: SimulationResult()}}
    )

    assert context["goal_achieved"] is True


# ---------------------------------------------------------------------------
# CircuitSimulationStage — a failed run must not leave stale results behind
# ---------------------------------------------------------------------------


class _StubSimulator(BaseSimulator):
    """Simulator that returns a queued result (or ``None``) per invocation."""

    netlist_parser = XyceNetlistParser()

    def __init__(self, results: list[SimulationResult | None]):
        self.results = list(results)

    def preprocess_ntwk(self, ntwk, name: str) -> str:
        return name

    def run_simulation(self, netlist_name: str) -> SimulationResult | None:
        return self.results.pop(0)


def _stage_context(tmp_path: Path) -> dict:
    netlist = tmp_path / "circuit.cir"
    netlist.write_text(netlist_path("minimal_ac").read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "netlist": str(netlist),
        "results_dir": str(tmp_path),
        "native_sim_type": SimulationType.AC,
        "predicted_networks": [],
        "design_goal_checker": DesignGoalChecker(
            [_goal(5.0, max_value=10.0, sim_type=SimulationType.AC)]
        ),
    }


def test_failed_run_clears_the_previous_iterations_result(tmp_path, caplog):
    stale = SimulationResult(output_files=["from_the_previous_iteration.s2p"])
    context = _stage_context(tmp_path)
    context["simulation_results"] = {SimulationType.AC: stale}
    stage = CircuitSimulationStage(_StubSimulator([None]))

    with caplog.at_level(logging.WARNING):
        context = stage.run(context)

    assert context["simulation_results"] == {}
    assert "AC" in caplog.text


def test_successful_run_stores_the_result(tmp_path):
    result = SimulationResult(output_files=["circuit.s2p"])
    stage = CircuitSimulationStage(_StubSimulator([result]))

    context = stage.run(_stage_context(tmp_path))

    assert context["simulation_results"] == {SimulationType.AC: result}


# ---------------------------------------------------------------------------
# XyceSimulator — simulation failure vs. technical failure
# ---------------------------------------------------------------------------


def test_missing_xyce_executable_aborts_the_run(tmp_path):
    netlist = tmp_path / "circuit.cir"
    netlist.write_text(netlist_path("minimal_ac").read_text(encoding="utf-8"), encoding="utf-8")
    simulator = XyceSimulator(xyce_command="definitely-not-installed-xyce")

    with pytest.raises(SimulatorError, match="definitely-not-installed-xyce"):
        simulator.run_simulation(str(netlist))


def test_non_zero_exit_code_is_a_simulation_failure(tmp_path, monkeypatch, caplog):
    netlist = tmp_path / "circuit.cir"
    netlist.write_text(netlist_path("minimal_ac").read_text(encoding="utf-8"), encoding="utf-8")

    def _fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["Xyce"], returncode=1, stdout="", stderr="Newton solver failed to converge"
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    simulator = XyceSimulator()

    with caplog.at_level(logging.WARNING):
        assert simulator.run_simulation(str(netlist)) is None

    assert "converge" in caplog.text


def test_missing_output_is_a_simulation_failure(tmp_path, monkeypatch, caplog):
    netlist = tmp_path / "circuit.cir"
    netlist.write_text(netlist_path("minimal_ac").read_text(encoding="utf-8"), encoding="utf-8")

    def _fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=["Xyce"], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with caplog.at_level(logging.WARNING):
        assert XyceSimulator().run_simulation(str(netlist)) is None

    assert "no output files" in caplog.text
