"""Tests for EM fine-tuning: the loop in :meth:`COBRA.fine_tuning` and the
Palace integration in :class:`EMFineTuningStage`.

Palace is not needed: the loop tests replace the fine-tuning stage with one
that returns a fixed network, and the stage tests replace ORCA's
``simulate_geometry`` with a stub that keeps its real signature.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import create_autospec

import numpy as np
import optuna
import pytest
import skrf as rf

from cobra.cobra import COBRA
from cobra.optimizers.base_optimizer import OptimizationProperty, OptimizationType
from cobra.optimizers.design_goal import (
    FAILED_SIMULATION_PENALTY,
    DesignGoal,
    DesignGoalChecker,
    DesignParameter,
)
from cobra.optimizers.design_goal_collection import calculate_array_penalty
from cobra.optimizers.optuna_optimizer import OptunaOptimizer
from cobra.spice_sim.base_simulator import BaseSimulator, SimulationResult, SimulatorError
from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
from cobra.spice_sim.simulation_type import SimulationType
from cobra.stages.em_finetuning_stage import EMFineTuningStage
from cobra.stages.surrogate_metadata import FeasibilityConstraints
from tests.conftest import MINIMAL_S2P, make_context, netlist_path

if TYPE_CHECKING:
    from collections.abc import Callable

    from cobra.optimization_context import OptimizationContext

FINE_TUNING_ITERATIONS = 3


def _r1(netlist_text: str) -> float:
    """The value of ``R1`` in *netlist_text*."""
    line = next(line for line in netlist_text.splitlines() if line.startswith("R1 "))
    return float(line.split()[3])


class _RecordingSimulator(BaseSimulator):
    """Returns the simulated netlist as the result and remembers what it simulated."""

    netlist_parser = XyceNetlistParser()

    def __init__(self):
        self.simulated: list[tuple[str, str]] = []

    def preprocess_ntwk(self, ntwk, name: str) -> str:
        return name

    def run_simulation(self, netlist_name: str) -> SimulationResult | None:
        self.simulated.append((netlist_name, Path(netlist_name).read_text(encoding="utf-8")))
        return SimulationResult(output_files=[netlist_name])


class _FakePalaceStage(EMFineTuningStage):
    """Stands in for Palace: returns the model's network and records what it was given."""

    def __init__(self, model: Path):
        super().__init__("palace", 1)
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def run(self, context, orca_geometry=None, comp_name=None):
        self.calls.append(
            {
                "iteration": context.fine_tuning_iteration,
                "results_dir": context.results_dir,
                "model_parameters": dict(context.model_parameters),
                "netlist_parameters": dict(context.netlist_parameters),
            }
        )
        ntwk = rf.Network(str(self.model))
        ntwk.name = comp_name
        context.predicted_networks = [ntwk]
        return context


def _r1_goal() -> DesignGoal:
    """A goal that is never met and gets worse the larger R1 is."""
    return DesignGoal(
        DesignParameter(
            name="R1_value",
            simulation_type=SimulationType.AC,
            formula=lambda result, _freq=None: np.array(
                [_r1(Path(result.output_files[0]).read_text(encoding="utf-8"))]
            ),
            loss=calculate_array_penalty,
        ),
        max_value=1.0,  # R1 ranges over 10-100, so this is never satisfied
    )


def _run_with_fine_tuning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fine_tuning_optimizer: str = "reuse",
    callback: Callable[[OptimizationContext], bool | None] | None = None,
) -> tuple[COBRA, OptimizationContext, _RecordingSimulator, _FakePalaceStage]:
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "model.s2p"
    model.write_text(MINIMAL_S2P, encoding="utf-8")
    netlist = tmp_path / "circuit.cir"
    netlist.write_text(netlist_path("minimal_ac").read_text(encoding="utf-8"), encoding="utf-8")

    simulator = _RecordingSimulator()
    cobra = COBRA(
        netlist_parser=XyceNetlistParser().from_file(netlist),
        component_onnx_mapping={"X1": str(model)},
        optimizer=OptunaOptimizer(sampler=optuna.samplers.RandomSampler(seed=1)),
        circuit_simulator=simulator,
        palace_fine_tuning_command="palace",
        palace_fine_tuning_processes=1,
        fine_tuning_iterations=FINE_TUNING_ITERATIONS,
        fine_tuning_optimizer=fine_tuning_optimizer,
    )
    # Treat X1 as an ONNX component, so fine-tuning simulates it with "Palace".
    assert cobra.em_surrogate_stage is not None
    cobra.em_surrogate_stage.is_touchstone = [False]
    palace = _FakePalaceStage(model)
    cobra.em_fine_tuning_stage = palace

    context = cobra.run(
        netlist=str(netlist),
        design_goals=[_r1_goal()],
        optimization_parameters=[
            OptimizationProperty("R1", OptimizationType.NETLIST_VARIABLE, 10.0, 100.0, step=1.0),
            OptimizationProperty("X1:width", OptimizationType.MODEL_INPUT, 1.0, 5.0),
        ],
        max_iterations=3,
        orca_geometries={"X1": object()},
        callback=callback,
    )
    return cobra, context, simulator, palace


def _fine_tuning_simulations(simulator: _RecordingSimulator) -> list[str]:
    """The netlists simulated during fine-tuning, in order."""
    return [text for path, text in simulator.simulated if "fine_tuning" in Path(path).parts]


# ---------------------------------------------------------------------------
# COBRA.fine_tuning — the loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fine_tuning_optimizer", ["reuse", "gradient_descent"])
def test_fine_tuning_runs_every_iteration_when_the_goals_are_never_met(
    tmp_path, monkeypatch, fine_tuning_optimizer
):
    """Regression: with "reuse", the second iteration told Optuna about a trial it
    had already been told about, which Optuna rejects.
    """
    cobra, context, _, palace = _run_with_fine_tuning(
        tmp_path, monkeypatch, fine_tuning_optimizer
    )

    assert [call["iteration"] for call in palace.calls] == [1, 2, 3]
    assert context.fine_tuning_iteration == FINE_TUNING_ITERATIONS
    assert not context.goal_achieved

    optimizer = cobra.optimizer_stage.optimizer
    assert isinstance(optimizer, OptunaOptimizer)
    assert optimizer.study is not None
    states = {trial.state for trial in optimizer.study.trials}
    assert states == {optuna.trial.TrialState.COMPLETE}


def test_fine_tuning_simulates_the_netlist_values_it_suggested(tmp_path, monkeypatch):
    """Regression: fine-tuning kept simulating the netlist of the surrogate loop,
    so only the geometry changed between iterations.
    """
    _, _, simulator, palace = _run_with_fine_tuning(tmp_path, monkeypatch)

    simulated = _fine_tuning_simulations(simulator)
    assert len(simulated) == len(palace.calls) == FINE_TUNING_ITERATIONS
    for text, call in zip(simulated, palace.calls, strict=True):
        assert _r1(text) == float(call["netlist_parameters"]["R1"])


def test_fine_tuning_returns_the_best_iteration_it_evaluated(tmp_path, monkeypatch):
    """Regression: the optimizer stepped after the last iteration, so the returned
    parameters had never been simulated.
    """
    _, context, simulator, palace = _run_with_fine_tuning(tmp_path, monkeypatch)

    evaluated = [_r1(text) for text in _fine_tuning_simulations(simulator)]
    best = int(np.argmin(evaluated))

    assert float(context.netlist_parameters["R1"]) == evaluated[best]
    assert context.model_parameters == palace.calls[best]["model_parameters"]
    # The run's own netlist holds the returned parameters too.
    final_netlist = Path(context.results_dir) / "circuit.cir"
    assert _r1(final_netlist.read_text(encoding="utf-8")) == evaluated[best]


def test_stopping_before_fine_tuning_still_saves_the_context(tmp_path, monkeypatch):
    """Regression: stopping at the fine-tuning prompt returned before the context
    JSON was written.
    """

    def _stop_before_fine_tuning(context: OptimizationContext) -> bool:
        return not (context.fine_tuning_active and context.fine_tuning_iteration == 0)

    _, context, _, palace = _run_with_fine_tuning(
        tmp_path, monkeypatch, callback=_stop_before_fine_tuning
    )

    assert palace.calls == []
    saved = Path(context.results_dir) / "cobra_optimization_context.json"
    assert json.loads(saved.read_text(encoding="utf-8"))["netlist"] == context.netlist


# ---------------------------------------------------------------------------
# EMFineTuningStage — the Palace integration through ORCA
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_simulate_geometry(monkeypatch):
    """Replace ``orca.simulate_geometry`` with a stub that keeps ORCA's real signature."""
    orca = pytest.importorskip("orca")
    network = rf.Network(frequency=rf.Frequency.from_f(np.array([1e9, 2e9]), unit="Hz"), s=np.zeros((2, 3, 3)))
    stub = create_autospec(orca.simulate_geometry, return_value=network)
    monkeypatch.setattr(orca, "simulate_geometry", stub)
    return stub


def _run_stage(geometry: Any = None) -> OptimizationContext:
    from orca.geometry.presets import InductorOcta

    context = make_context(
        results_dir="results",
        model_parameters={"X1:turns": 2, "X1:width": 6.0, "X2:width": 3.0, "space": 3.0},
        fine_tuning_iteration=1,
        iteration=2,
    )
    return EMFineTuningStage("palace", 4).run(
        context, orca_geometry=geometry or InductorOcta(), comp_name="X1"
    )


def test_stage_simulates_the_components_parameters_through_orca(fake_simulate_geometry):
    context = _run_stage()

    fake_simulate_geometry.assert_called_once()
    geometry, parameters, output_dir, name = fake_simulate_geometry.call_args.args
    assert type(geometry).__name__ == "InductorOcta"
    # X2's parameters are left out, the X1: prefix is stripped, unscoped ones are shared
    assert parameters == {"turns": 2, "width": 6.0, "space": 3.0}
    assert output_dir == str(Path("results").resolve())
    assert name == "cobra_result_ft_1_2_X1"
    assert fake_simulate_geometry.call_args.kwargs == {"palace_executable": "palace", "num_processes": 4}
    [network] = context.predicted_networks
    assert network.name == "X1"


def test_stage_reports_orcas_simulation_error_for_the_component(fake_simulate_geometry):
    from orca import SimulationError

    fake_simulate_geometry.side_effect = SimulationError("Palace failed")

    with pytest.raises(SimulatorError, match=r"X1.*Palace failed"):
        _run_stage()


def test_stage_rejects_a_geometry_that_is_not_an_orca_geometry(fake_simulate_geometry):
    with pytest.raises(TypeError, match="BaseGeometry"):
        _run_stage(geometry=object())
    fake_simulate_geometry.assert_not_called()


# ---------------------------------------------------------------------------
# Infeasible geometries — skipped, and penalised like a failed simulation
# ---------------------------------------------------------------------------


def _cobra_with_constraint(tmp_path: Path) -> tuple[COBRA, _RecordingSimulator, _FakePalaceStage, list[str]]:
    """A COBRA whose X1 model only accepts ``width <= 2``, and the netlist template."""
    model = tmp_path / "model.s2p"
    model.write_text(MINIMAL_S2P, encoding="utf-8")
    netlist = tmp_path / "circuit.cir"
    netlist.write_text(netlist_path("minimal_ac").read_text(encoding="utf-8"), encoding="utf-8")
    simulator = _RecordingSimulator()
    parser = XyceNetlistParser().from_file(netlist)
    cobra = COBRA(
        netlist_parser=parser,
        component_onnx_mapping={"X1": str(model)},
        circuit_simulator=simulator,
    )
    assert cobra.em_surrogate_stage is not None
    cobra.em_surrogate_stage.constraints = [FeasibilityConstraints(["width <= 2"], ["width"])]
    palace = _FakePalaceStage(model)
    return cobra, simulator, palace, parser.lines


def _trial(tmp_path: Path, width: float) -> OptimizationContext:
    return make_context(
        netlist=str(tmp_path / "trial" / "circuit.cir"),
        results_dir=str(tmp_path / "trial"),
        design_goal_checker=DesignGoalChecker([_r1_goal()]),
        model_parameters={"X1:width": width},
    )


@pytest.mark.parametrize(("width", "simulated"), [(1.5, True), (3.0, False)])
def test_infeasible_trial_is_penalised_without_simulating(tmp_path, width, simulated):
    cobra, simulator, _, template = _cobra_with_constraint(tmp_path)
    trial = _trial(tmp_path, width)

    cobra._evaluate_trial(trial, template)

    assert bool(simulator.simulated) is simulated
    penalties = [goal.current_penalty for goal in trial.goals]
    assert (penalties == [FAILED_SIMULATION_PENALTY]) is not simulated


def test_infeasible_fine_tuning_trial_never_reaches_palace(tmp_path):
    cobra, simulator, palace, template = _cobra_with_constraint(tmp_path)
    assert cobra.em_surrogate_stage is not None
    trial = _trial(tmp_path, 3.0)

    cobra._evaluate_fine_tuning_trial(trial, template, palace, cobra.em_surrogate_stage, {})

    assert palace.calls == []
    assert simulator.simulated == []
    assert [goal.current_penalty for goal in trial.goals] == [FAILED_SIMULATION_PENALTY]
