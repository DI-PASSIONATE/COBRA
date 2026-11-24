"""Tests for the reuse of surrogate fits."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import skrf as rf

from cobra.spice_sim.base_simulator import SimulationResult
from cobra.spice_sim.vector_fit import VectorFitError
from cobra.spice_sim.xyce_simulator import XyceSimulator
from cobra.stages.circuit_sim_stage import CircuitSimulationStage
from cobra.stages.fit_cache import FitCache
from tests.conftest import skrf_medium
from tests.test_simulation_failure import _stage_context


def _network(resistance: float = 50.0, z0: float = 50.0) -> rf.Network:
    media = skrf_medium(rf.Frequency(1, 20, 11, "GHz"))
    network = media.resistor(resistance)
    network.z0 = z0
    network.name = "X1"
    return network


class _Fit:
    """Writes a file naming the network, and counts its calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, ntwk: rf.Network, name: str) -> str:
        self.calls += 1
        path = Path(name + ".inc")
        path.write_text(f"subckt {Path(name).name} {ntwk.s[0, 0, 0]}", encoding="utf-8")
        return str(path)


def _prefix(tmp_path: Path, trial: str, component: str = "X1") -> str:
    (tmp_path / trial).mkdir(exist_ok=True)
    return str(tmp_path / trial / component)


def test_the_same_network_is_fitted_once(tmp_path):
    cache, fit = FitCache(), _Fit()

    first = cache.get(_network(), _prefix(tmp_path, "a"), fit)
    second = cache.get(_network(), _prefix(tmp_path, "b"), fit)

    assert fit.calls == 1
    assert (cache.fitted, cache.reused) == (1, 1)
    assert second == str(tmp_path / "b" / "X1.inc")
    assert Path(second).read_text(encoding="utf-8") == Path(first).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("network", "component"),
    [(_network(60.0), "X1"), (_network(z0=75.0), "X1"), (_network(), "X2")],
)
def test_another_network_reference_impedance_or_name_is_fitted_again(tmp_path, network, component):
    cache, fit = FitCache(), _Fit()

    cache.get(_network(), _prefix(tmp_path, "a"), fit)
    cache.get(network, _prefix(tmp_path, "b", component), fit)

    assert (fit.calls, cache.reused) == (2, 0)


def test_other_frequencies_are_fitted_again(tmp_path):
    cache, fit = FitCache(), _Fit()
    moved = _network()
    moved.frequency = rf.Frequency(2, 40, 11, "GHz")

    cache.get(_network(), _prefix(tmp_path, "a"), fit)
    cache.get(moved, _prefix(tmp_path, "b"), fit)

    assert fit.calls == 2


def test_a_failed_fit_is_not_kept(tmp_path):
    cache, fit = FitCache(), _Fit()

    def failing(_ntwk: rf.Network, _name: str) -> str:
        raise VectorFitError("no fit")

    with pytest.raises(VectorFitError, match="no fit"):
        cache.get(_network(), _prefix(tmp_path, "a"), failing)
    cache.get(_network(), _prefix(tmp_path, "b"), fit)

    assert fit.calls == 1


def test_the_least_recently_used_fit_is_dropped(tmp_path):
    cache, fit = FitCache(max_fits=2), _Fit()

    for trial, resistance in enumerate((10.0, 20.0, 10.0, 30.0, 10.0, 20.0)):
        cache.get(_network(resistance), _prefix(tmp_path, f"t{trial}"), fit)

    assert (fit.calls, cache.reused) == (4, 2)


def test_concurrent_trials_share_one_fit(tmp_path):
    cache, started, release = FitCache(), threading.Event(), threading.Event()
    fit = _Fit()

    def slow(ntwk: rf.Network, name: str) -> str:
        started.set()
        assert release.wait(10)
        return fit(ntwk, name)

    with ThreadPoolExecutor(max_workers=4) as pool:
        fits = [pool.submit(cache.get, _network(), _prefix(tmp_path, "t0"), slow)]
        assert started.wait(10)
        fits += [pool.submit(cache.get, _network(), _prefix(tmp_path, f"t{k}"), slow) for k in range(1, 4)]
        release.set()
        paths = [f.result(10) for f in fits]

    assert fit.calls == 1
    assert all(Path(path).is_file() for path in paths)


def test_a_failed_fit_releases_the_trials_waiting_for_it(tmp_path):
    cache, started, release = FitCache(), threading.Event(), threading.Event()

    def failing(_ntwk: rf.Network, _name: str) -> str:
        started.set()
        assert release.wait(10)
        raise VectorFitError("no fit")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cache.get, _network(), _prefix(tmp_path, "a"), failing)
        assert started.wait(10)
        second = pool.submit(cache.get, _network(), _prefix(tmp_path, "b"), failing)
        release.set()
        for trial in (first, second):
            with pytest.raises(VectorFitError, match="no fit"):
                trial.result(10)


class _CountingSimulator(XyceSimulator):
    def __init__(self):
        self.fit = _Fit()

    def preprocess_ntwk(self, ntwk, name: str = "cobra_output") -> str:
        return self.fit(ntwk, name)

    def run_simulation(self, netlist_path, netlist) -> SimulationResult | None:
        return SimulationResult(output_files=[netlist_path])


class _UncachedSimulator(_CountingSimulator):
    reuses_fits = False


@pytest.mark.parametrize(("simulator", "fits"), [(_CountingSimulator(), 1), (_UncachedSimulator(), 3)])
def test_the_stage_reuses_fits_only_if_the_simulator_allows_it(tmp_path, simulator, fits):
    stage = CircuitSimulationStage(simulator)
    for _ in range(3):
        context = _stage_context(tmp_path)
        context.predicted_networks = [_network()]
        stage.run(context)

    assert simulator.fit.calls == fits


def test_a_reused_xyce_fit_equals_a_new_one(tmp_path):
    media = skrf_medium(rf.Frequency(1, 20, 41, "GHz"))
    network = media.line(30, "deg") ** media.shunt_capacitor(0.2e-12)
    network.name = "X1"
    stage = CircuitSimulationStage(XyceSimulator("Xyce"))

    stage._preprocess(network, _prefix(tmp_path, "a"))
    stage._preprocess(network.copy(), _prefix(tmp_path, "b"))
    XyceSimulator("Xyce").preprocess_ntwk(network, _prefix(tmp_path, "c"))

    texts = [(tmp_path / trial / "X1.sp").read_text(encoding="utf-8") for trial in "abc"]
    assert stage.fit_cache.reused == 1
    assert texts[0] == texts[1] == texts[2]
