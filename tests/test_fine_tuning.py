"""Tests for EM fine-tuning: the loop in :meth:`COBRA.fine_tuning` and the
Palace integration in :class:`EMFineTuningStage`.

Neither Palace nor ORCA is needed: the loop tests replace the fine-tuning
stage with one that returns a fixed network, and the stage tests install
stand-ins for the ORCA modules the stage imports.
"""

from __future__ import annotations

import json
import sys
import types
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
import pytest
import skrf as rf

from cobra.cobra import COBRA
from cobra.optimizers.base_optimizer import OptimizationProperty, OptimizationType
from cobra.optimizers.design_goal import DesignGoal, DesignParameter
from cobra.optimizers.design_goal_collection import calculate_array_penalty
from cobra.optimizers.optuna_optimizer import OptunaOptimizer
from cobra.spice_sim.base_simulator import BaseSimulator, SimulationResult, SimulatorError
from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
from cobra.spice_sim.simulation_type import SimulationType
from cobra.stages import em_finetuning_stage
from cobra.stages.em_finetuning_stage import EMFineTuningStage
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


def _module(monkeypatch: pytest.MonkeyPatch, name: str, **attributes: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


class _FakeLauncher:
    """Mirrors ``orca.simulation.launchers.LocalLauncher`` for an unpinned machine."""

    slots: tuple[str, ...] = ("slot0",)

    def command(self, slot: str, palace_executable: str, num_processes: int, config_name: str) -> str:
        assert slot in self.slots
        return f"{palace_executable} -np {num_processes} {config_name}"


#: Mirrors ``orca.simulation.combine_snp_results.TOUCHSTONE_SUFFIXES``.
_TOUCHSTONE_SUFFIXES = {
    "normal": "",
    "dc": "_dc",
    "deembedded": "_deembedded",
    "dc_deembedded": "_dc_deembedded",
    "all": "_dc_deembedded",
}


def _fake_touchstone_filename(base_name: str, n_ports: int, touchstone_type: str) -> str:
    """Mirrors ``orca.simulation.combine_snp_results.touchstone_filename``."""
    return f"{base_name}{_TOUCHSTONE_SUFFIXES[touchstone_type]}.s{n_ports}p"


class _FakeBaseGeometry:
    pass


class _TwoPortGeometry(_FakeBaseGeometry):
    n_ports = 2
    stackup_xml = "stackup.xml"
    simconfig_filename = "simconfig.json"

    def __init__(self):
        self.created: list[dict[str, Any]] = []

    def create_gds_file(self, name: str, output_path: str, params: dict[str, Any]) -> str:
        self.created.append({"name": name, "output_path": output_path, "params": params})
        return output_path


class _InlineExecutor:
    """A ``ProcessPoolExecutor`` stand-in that runs the job in this process."""

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def submit(self, fn, /, *args, **kwargs) -> Future:
        future: Future = Future()
        future.set_result(fn(*args, **kwargs))
        return future


@pytest.fixture
def fake_orca(monkeypatch):
    """Install stand-ins for the ORCA modules the stage imports.

    ``ihp`` is made unimportable: ORCA no longer depends on the IHP gdsfactory PDK,
    so the stage must not need it either.
    """
    monkeypatch.setitem(sys.modules, "ihp", None)
    _module(monkeypatch, "orca")
    _module(monkeypatch, "orca.geometry")
    _module(monkeypatch, "orca.geometry.base_geometry", BaseGeometry=_FakeBaseGeometry)
    _module(monkeypatch, "orca.simulation")
    _module(
        monkeypatch,
        "orca.simulation.combine_snp_results",
        touchstone_filename=_fake_touchstone_filename,
    )
    _module(monkeypatch, "orca.simulation.launchers", LocalLauncher=_FakeLauncher)
    calls: dict[str, Any] = {}

    def _create_palace_model_from_gds(**kwargs):
        calls["create_palace_model_from_gds"] = kwargs
        name = kwargs["geometry_name"]
        return name, kwargs["params"], "config.json", f"/sims/{name}", f"output/{name}"

    def _run_palace(**kwargs):
        calls["run_palace"] = kwargs
        return calls.get("palace_succeeds", True)

    _module(
        monkeypatch,
        "orca.simulation.gds_converter",
        create_palace_model_from_gds=_create_palace_model_from_gds,
    )
    _module(monkeypatch, "orca.simulation.simulate", run_palace=_run_palace)
    return calls


def test_palace_is_run_with_orcas_current_signature(fake_orca, tmp_path):
    """Regression: COBRA passed config_name/palace_executable/num_processes, which
    ORCA's run_palace no longer accepts.
    """
    succeeded = em_finetuning_stage._mesh_gds_and_run_palace(
        name="ft",
        parameters={"width": 2.0},
        base_dir=str(tmp_path),
        gds_output_path=str(tmp_path / "ft.gds"),
        stackup_xml="stackup.xml",
        simconfig_filename="simconfig.json",
        palace_executable="palace",
        num_processes=4,
    )

    assert succeeded is True
    assert fake_orca["run_palace"] == {
        "sim_path": "/sims/ft",
        "data_dir": "output/ft",
        "result_dir": str(tmp_path),
        "cmd": "palace -np 4 config.json",
        "touchstone_type": "all",
    }


#: What ORCA writes for a sweep from 1 GHz up with more than 20 points.
ALL_VARIANTS = ("", "_dc", "_deembedded", "_dc_deembedded")
#: What ORCA writes otherwise: it skips the DC extrapolation.
NO_DC_VARIANTS = ("", "_deembedded")


def _run_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    palace_succeeds: bool = True,
    written: tuple[str, ...] = ALL_VARIANTS,
) -> OptimizationContext:
    """Run the stage with a fake Palace that writes the Touchstone variants in *written*.

    Each file is tagged with its variant in a comment, so a test can tell which one
    the stage loaded.
    """
    monkeypatch.setattr(em_finetuning_stage, "ProcessPoolExecutor", _InlineExecutor)

    def _fake_palace(**kwargs) -> bool:
        if palace_succeeds:
            for suffix in written:
                path = Path(kwargs["base_dir"]) / f"{kwargs['name']}{suffix}.s2p"
                path.write_text(f"! variant={suffix or 'normal'}\n{MINIMAL_S2P}", encoding="utf-8")
        return palace_succeeds

    monkeypatch.setattr(em_finetuning_stage, "_mesh_gds_and_run_palace", _fake_palace)
    context = make_context(
        results_dir=str(tmp_path),
        model_parameters={"X1:width": 2.0, "X2:width": 3.0},
        fine_tuning_iteration=1,
        iteration=1,
    )
    return EMFineTuningStage("palace", 1).run(
        context, orca_geometry=_TwoPortGeometry(), comp_name="X1"
    )


def test_stage_loads_the_most_corrected_result_for_the_geometrys_port_count(
    fake_orca, monkeypatch, tmp_path
):
    """Regression: the stage always read a ``_dc_deembedded.s6p`` file, so only 6-port
    geometries worked, and a narrow sweep (e.g. 120-150 GHz), for which ORCA skips
    the DC extrapolation, failed although Palace had succeeded.
    """
    (tmp_path / "full").mkdir()
    (tmp_path / "narrow").mkdir()
    full = _run_stage(monkeypatch, tmp_path / "full", written=ALL_VARIANTS)
    narrow = _run_stage(monkeypatch, tmp_path / "narrow", written=NO_DC_VARIANTS)

    [network] = full.predicted_networks
    assert network.nports == 2
    assert network.name == "X1"
    assert "variant=_dc_deembedded" in network.comments
    assert "variant=_deembedded" in narrow.predicted_networks[0].comments


def test_stage_raises_when_palace_fails_or_writes_no_result(fake_orca, monkeypatch, tmp_path):
    with pytest.raises(SimulatorError, match="X1"):
        _run_stage(monkeypatch, tmp_path, palace_succeeds=False)
    with pytest.raises(SimulatorError, match="X1"):
        _run_stage(monkeypatch, tmp_path, written=())
