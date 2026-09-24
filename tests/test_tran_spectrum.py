"""Tests for the transient (time-domain) spectrum path.

Covers :mod:`cobra.spice_sim.tran_spectrum`, the transient variants of the
spectrum goals, how the Xyce simulator turns a ``.PRINT tran format=csv`` table
into a spectrum, and how configuration and inspection treat ``TRAN`` goals.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cobra.configuration.config_runner import build_design_goals
from cobra.configuration.configuration import ConfigurationError, DesignGoalConfig
from cobra.configuration.inspection import (
    all_issues,
    inspect_path,
)
from cobra.optimizers.design_goal import DesignGoal
from cobra.optimizers.design_goal_collection import (
    make_power_dbm,
)
from cobra.spice_sim import hb_spectrum
from cobra.spice_sim.base_simulator import SimulationResult
from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
from cobra.spice_sim.simulation_type import SimulationType
from cobra.spice_sim.tran_spectrum import (
    to_frequency_domain,
)
from cobra.spice_sim.xyce_simulator import XyceSimulator
from cobra.stages.circuit_sim_stage import CircuitSimulationStage
from tests.conftest import make_config_data, netlist_path

F0 = 130e9
PERIODS = 26
WINDOW = PERIODS / F0


def _transient_table(
    v_amplitude: float = 1.0,
    i_amplitude: float = 0.02,
    dc: float = 0.0,
    samples: int = 2000,
    jitter: float = 0.0,
    start: float = 0.0,
) -> pd.DataFrame:
    """A ``.PRINT tran`` table of a sine at ``F0`` over ``PERIODS`` whole periods.

    *samples* points span the window inclusive of both ends, as Xyce prints it.
    *jitter* perturbs the interior instants so the grid is non-uniform.
    """
    rng = np.random.default_rng(0)
    times = start + np.linspace(0.0, WINDOW, samples)
    if jitter:
        step = WINDOW / (samples - 1)
        times[1:-1] += rng.uniform(-jitter, jitter, samples - 2) * step
        times.sort()
    phase = 2 * np.pi * F0 * times
    return pd.DataFrame(
        {
            "TIME": times,
            "V(OUT)": dc + v_amplitude * np.sin(phase),
            "I(VOUT)": i_amplitude * np.sin(phase),
        }
    )


def _bin(spectrum: pd.DataFrame, frequency: float) -> int:
    return int(np.argmin(np.abs(spectrum["FREQ"].to_numpy() - frequency)))


# ---------------------------------------------------------------------------
# to_frequency_domain
# ---------------------------------------------------------------------------


def _phasor(spectrum: pd.DataFrame, frequency: float) -> complex:
    row = spectrum.iloc[_bin(spectrum, frequency)]
    return complex(row["Re(V(OUT))"], row["Im(V(OUT))"])


def test_phasors_hold_half_the_amplitude_and_the_dc_bin_the_mean():
    spectrum = to_frequency_domain(_transient_table(v_amplitude=0.8, dc=1.5))

    # Same layout as Xyce's HB table, so the HB goals read it unchanged.
    assert hb_spectrum.probe_nodes(spectrum) == ["OUT"]
    # The resolution is the reciprocal of the printed window.
    assert spectrum["FREQ"].iloc[1] == pytest.approx(1.0 / WINDOW)
    assert abs(_phasor(spectrum, F0)) == pytest.approx(0.4, rel=1e-6)
    assert _phasor(spectrum, 0.0).real == pytest.approx(1.5, rel=1e-6)
    # A whole number of periods leaves the neighbouring bins empty.
    assert abs(_phasor(spectrum, F0 + 1.0 / WINDOW)) < 1e-6


def test_non_uniform_samples_and_a_late_window_start_are_handled():
    """Xyce's adaptive time step prints a non-uniform grid, and ``.TRAN ... <start_time>``
    drops the start-up transient; only the window length matters.
    """
    jittered = to_frequency_domain(_transient_table(jitter=0.3))
    late = to_frequency_domain(_transient_table(start=100e-12))

    assert abs(_phasor(jittered, F0)) == pytest.approx(0.5, rel=1e-3)
    assert abs(_phasor(late, F0)) == pytest.approx(0.5, rel=1e-6)


def test_unusable_tables_are_errors():
    with pytest.raises(KeyError, match="TIME"):
        to_frequency_domain(pd.DataFrame({"FREQ": [0.0, 1.0]}))
    with pytest.raises(ValueError, match="distinct time points"):
        to_frequency_domain(_transient_table(samples=3))


# ---------------------------------------------------------------------------
# Transient goals
# ---------------------------------------------------------------------------


def _tran_result() -> SimulationResult:
    return SimulationResult(
        dataframes={"run.cir.TRAN.FD.csv": to_frequency_domain(_transient_table())}
    )


def test_transient_power_goal_uses_the_hb_power_formula():
    """V = 1 V, I = 20 mA peak → S = V_rms·I_rms = 10 mW = 10 dBm."""
    goal = DesignGoal(make_power_dbm("OUT", SimulationType.TRAN), "130GHz", min_value=5.0)

    assert goal.penalty(_tran_result()) < 0.0
    assert np.asarray(goal.current_value)[0] == pytest.approx(10.0, abs=1e-3)


def test_transient_goal_configuration():
    data = {
        "parameter": "TRAN:Power_dBm[OUT]",
        "frequency_range": "130GHz",
        "max_value": 10.0,
        "kind": "power_dbm",
        "node": "OUT",
        "analysis": "tran",  # case-insensitive
    }
    parser = XyceNetlistParser().from_file(netlist_path("tran_single_tone"))

    [goal] = build_design_goals([DesignGoalConfig.from_dict(data)], parser)
    assert goal.required_simulation_type is SimulationType.TRAN

    with pytest.raises(ConfigurationError, match="analysis"):
        DesignGoalConfig.from_dict({**data, "analysis": "dc"}).validate()
    with pytest.raises(ConfigurationError, match="expected 'TRAN:Power_dBm\\[OUT\\]'"):
        build_design_goals([DesignGoalConfig.from_dict({**data, "parameter": "Power_dBm[OUT]"})], parser)


# ---------------------------------------------------------------------------
# Netlist preparation and simulator output
# ---------------------------------------------------------------------------


def test_injected_tran_directive_prints_the_probes(tmp_path: Path):
    prepared = CircuitSimulationStage._prepare_netlist_for_type(
        str(netlist_path("hb_two_tone")),
        SimulationType.TRAN,
        {"step": "1p", "stop_time": "1n", "start_time": "0", "max_step": "1p"},
        str(tmp_path),
        simulator=XyceSimulator(),
    )
    text = Path(prepared).read_text(encoding="utf-8")

    assert ".TRAN 1p 1n 0 1p" in text
    assert ".PRINT TRAN format=csv V(OUT) I(VOUT)" in text
    assert ".HB" not in text


def test_simulator_derives_the_spectrum_from_the_transient_csv(tmp_path: Path, monkeypatch):
    netlist = tmp_path / "circuit.cir"
    shutil.copy(netlist_path("tran_single_tone"), netlist)

    def _fake_run(*_args, **_kwargs):
        _transient_table().to_csv(tmp_path / "circuit.cir.csv", index=False)
        return subprocess.CompletedProcess(args=["Xyce"], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    result = XyceSimulator().run_simulation(str(netlist))

    assert result is not None
    fd_path = str(tmp_path / "circuit.cir.TRAN.FD.csv")
    assert fd_path in result.output_files
    spectrum = hb_spectrum.find_dataframe(result.dataframes, "OUT", "power")
    assert spectrum is not None
    _, p_dbm = hb_spectrum.spectrum(spectrum, "OUT", "power", (F0, F0))
    assert p_dbm[0] == pytest.approx(10.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def _frequency_issues(config_dir: Path, frequency_range: str) -> list[str]:
    shutil.copy(netlist_path("tran_single_tone"), config_dir / "amp.cir")
    data = make_config_data(
        netlist="amp.cir",
        simulation_parameters={},
        design_goals=[
            {
                "parameter": "TRAN:Power_dBm[OUT]",
                "frequency_range": frequency_range,
                "max_value": 10.0,
                "kind": "power_dbm",
                "node": "OUT",
                "analysis": "TRAN",
            },
        ],
    )
    path = config_dir / "tran.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return [
        issue.message
        for issue in all_issues(inspect_path(path, check_models=False))
        if "Goal frequency" in issue.message
    ]


def test_transient_goal_off_the_fft_grid_is_flagged(config_dir: Path):
    # .tran 1p 300p 100p → 200 ps window → 5 GHz bins; 130 GHz is the 26th.
    assert _frequency_issues(config_dir, "130GHz") == []
    [message] = _frequency_issues(config_dir, "132GHz")
    assert "FFT grid" in message
