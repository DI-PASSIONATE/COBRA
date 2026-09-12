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
    ConfigurationReport,
    all_issues,
    inspect_netlist,
    inspect_path,
)
from cobra.optimizers.design_goal import DesignGoal
from cobra.optimizers.design_goal_collection import (
    large_signal_name,
    make_gain_db,
    make_isolation_db,
    make_power_dbm,
)
from cobra.spice_sim import hb_spectrum
from cobra.spice_sim.base_simulator import SimulationResult
from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
from cobra.spice_sim.simulation_type import SimulationType
from cobra.spice_sim.tran_spectrum import (
    is_time_domain,
    on_fft_grid,
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


def test_is_time_domain_looks_for_a_time_column():
    assert is_time_domain(_transient_table())
    assert not is_time_domain(pd.DataFrame({"FREQ": [0.0], "Re(V(OUT))": [1.0]}))


def test_spectrum_layout_matches_the_hb_table():
    spectrum = to_frequency_domain(_transient_table())

    assert list(spectrum.columns) == ["FREQ", "Re(V(OUT))", "Im(V(OUT))", "Re(I(VOUT))", "Im(I(VOUT))"]
    assert hb_spectrum.probe_nodes(spectrum) == ["OUT"]
    assert spectrum["FREQ"].iloc[0] == 0.0
    # The resolution is the reciprocal of the printed window.
    assert spectrum["FREQ"].iloc[1] == pytest.approx(1.0 / WINDOW)


def test_phasors_hold_half_the_amplitude_and_the_dc_bin_the_mean():
    spectrum = to_frequency_domain(_transient_table(v_amplitude=0.8, dc=1.5))
    phasor = spectrum["Re(V(OUT))"] + 1j * spectrum["Im(V(OUT))"]

    line = _bin(spectrum, F0)
    assert spectrum["FREQ"].iloc[line] == pytest.approx(F0)
    assert abs(phasor.iloc[line]) == pytest.approx(0.4, rel=1e-6)
    assert phasor.iloc[0].real == pytest.approx(1.5, rel=1e-6)
    # A whole number of periods leaves the neighbouring bins empty.
    assert abs(phasor.iloc[line - 1]) < 1e-6
    assert abs(phasor.iloc[line + 1]) < 1e-6


def test_non_uniform_samples_are_resampled():
    spectrum = to_frequency_domain(_transient_table(jitter=0.3))
    phasor = spectrum["Re(V(OUT))"] + 1j * spectrum["Im(V(OUT))"]

    assert abs(phasor.iloc[_bin(spectrum, F0)]) == pytest.approx(0.5, rel=1e-3)


def test_duplicate_time_points_are_tolerated():
    table = _transient_table()
    doubled = pd.concat([table.iloc[:1], table], ignore_index=True)

    spectrum = to_frequency_domain(doubled)
    assert len(spectrum) == len(to_frequency_domain(table))


def test_window_may_start_after_zero():
    """``.TRAN ... <start_time>`` drops the start-up transient; only the window length matters."""
    spectrum = to_frequency_domain(_transient_table(start=100e-12))
    phasor = spectrum["Re(V(OUT))"] + 1j * spectrum["Im(V(OUT))"]

    assert spectrum["FREQ"].iloc[1] == pytest.approx(1.0 / WINDOW)
    assert abs(phasor.iloc[_bin(spectrum, F0)]) == pytest.approx(0.5, rel=1e-6)


def test_missing_time_column_is_an_error():
    with pytest.raises(KeyError, match="TIME"):
        to_frequency_domain(pd.DataFrame({"FREQ": [0.0, 1.0]}))


def test_too_few_samples_is_an_error():
    with pytest.raises(ValueError, match="distinct time points"):
        to_frequency_domain(_transient_table(samples=3))


def test_power_from_a_transient_spectrum_uses_the_hb_formula():
    """V = 1 V, I = 20 mA peak → S = V_rms·I_rms = 10 mW = 10 dBm."""
    spectrum = to_frequency_domain(_transient_table())
    freqs, p_dbm = hb_spectrum.spectrum(spectrum, "OUT", "power", (F0, F0))

    assert freqs[0] == pytest.approx(F0)
    assert p_dbm[0] == pytest.approx(10.0, abs=1e-3)


# ---------------------------------------------------------------------------
# on_fft_grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("window", "low", "high", "expected"),
    [
        (1e-9, 35e9, 35e9, True),      # 35 GHz is the 35th bin of a 1 ns window
        (1e-9, 35.5e9, 35.5e9, False),
        (1e-9, 34.6e9, 35.4e9, True),  # the band holds the 35 GHz bin
        (1e-9, 35.2e9, 35.8e9, False),
        (200e-12, 130e9, 130e9, True),
        (200e-12, 132e9, 132e9, False),
        (0.0, 132e9, 132e9, True),     # unusable window: stay quiet
    ],
)
def test_on_fft_grid(window, low, high, expected):
    assert on_fft_grid(window, low, high) is expected


# ---------------------------------------------------------------------------
# Transient goal parameters
# ---------------------------------------------------------------------------


def _tran_result() -> SimulationResult:
    return SimulationResult(
        dataframes={"run.cir.TRAN.FD.csv": to_frequency_domain(_transient_table())}
    )


def test_large_signal_names_prefix_only_the_transient_variant():
    assert large_signal_name("Power_dBm[Out]", SimulationType.HB) == "Power_dBm[Out]"
    assert large_signal_name("Power_dBm[Out]", SimulationType.TRAN) == "TRAN:Power_dBm[Out]"
    with pytest.raises(ValueError, match="HB or TRAN"):
        large_signal_name("Power_dBm[Out]", SimulationType.AC)


def test_transient_parameters_require_the_tran_analysis():
    power = make_power_dbm("OUT", SimulationType.TRAN)
    gain = make_gain_db("P1", 0.2, 50.0, "OUT", SimulationType.TRAN)
    isolation = make_isolation_db("OUT", SimulationType.TRAN)

    assert (power.name, gain.name, isolation.name) == (
        "TRAN:Power_dBm[OUT]",
        "TRAN:Gain_dB[P1@OUT]",
        "TRAN:Isolation_dB[OUT]",
    )
    assert {p.simulation_type for p in (power, gain, isolation)} == {SimulationType.TRAN}
    # The HB variant is unchanged.
    assert make_power_dbm("OUT").name == "Power_dBm[OUT]"
    assert make_power_dbm("OUT").simulation_type is SimulationType.HB


def test_transient_power_goal_reads_the_spectrum_table():
    goal = DesignGoal(make_power_dbm("OUT", SimulationType.TRAN), "130GHz", min_value=5.0)

    assert goal.penalty(_tran_result()) < 0.0
    assert np.asarray(goal.current_value)[0] == pytest.approx(10.0, abs=1e-3)


def test_transient_isolation_is_measured_against_the_other_bins():
    value = make_isolation_db("OUT", SimulationType.TRAN).formula(_tran_result(), "130GHz")

    # A pure sine leaves every other bin at the numerical floor.
    assert value > 100.0


def test_missing_transient_probe_names_the_analysis():
    with pytest.raises(KeyError, match=r"No \.TRAN result.*'\.PRINT tran'"):
        make_power_dbm("OUT", SimulationType.TRAN).formula(SimulationResult(), "130GHz")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _tran_goal_config(**overrides) -> DesignGoalConfig:
    data = {
        "parameter": "TRAN:Power_dBm[OUT]",
        "frequency_range": "130GHz",
        "max_value": 10.0,
        "kind": "power_dbm",
        "node": "OUT",
        "analysis": "TRAN",
    }
    data.update(overrides)
    return DesignGoalConfig.from_dict(data)


def test_analysis_defaults_to_hb():
    assert DesignGoalConfig(parameter="S11_dB", max_value=-10.0).analysis == "HB"
    assert DesignGoalConfig(parameter="S11_dB", max_value=-10.0).analysis_type() is SimulationType.HB


def test_analysis_must_be_hb_or_tran():
    with pytest.raises(ConfigurationError, match="analysis"):
        _tran_goal_config(analysis="dc").validate()


def test_analysis_is_case_insensitive():
    goal = _tran_goal_config(analysis="tran")
    goal.validate()
    assert goal.analysis_type() is SimulationType.TRAN


def test_build_transient_goal_from_configuration():
    parser = XyceNetlistParser().from_file(netlist_path("tran_single_tone"))

    [goal] = build_design_goals([_tran_goal_config()], parser)

    assert goal.parameter_name == "TRAN:Power_dBm[OUT]"
    assert goal.required_simulation_type is SimulationType.TRAN


def test_parameter_name_must_match_the_analysis():
    parser = XyceNetlistParser().from_file(netlist_path("tran_single_tone"))

    with pytest.raises(ConfigurationError, match="expected 'TRAN:Power_dBm\\[OUT\\]'"):
        build_design_goals([_tran_goal_config(parameter="Power_dBm[OUT]")], parser)


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
    assert Path(fd_path).is_file()
    spectrum = hb_spectrum.find_dataframe(result.dataframes, "OUT", "power")
    assert spectrum is not None
    _, p_dbm = hb_spectrum.spectrum(spectrum, "OUT", "power", (F0, F0))
    assert p_dbm[0] == pytest.approx(10.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def _tran_config(config_dir: Path, frequency_range: str, **overrides) -> Path:
    shutil.copy(netlist_path("tran_single_tone"), config_dir / "amp.cir")
    data = make_config_data(
        netlist="amp.cir",
        simulation_parameters={},
        design_goals=[
            {
                "parameter": "TRAN:Power_dBm[OUT]",
                "frequency_range": frequency_range,
                "min_value": None,
                "max_value": 10.0,
                "weight": 1.0,
                "kind": "power_dbm",
                "node": "OUT",
                "analysis": "TRAN",
            },
        ],
    )
    data.update(overrides)
    path = config_dir / "tran.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _frequency_issues(path: Path) -> list[str]:
    return [
        issue.message
        for issue in all_issues(inspect_path(path, check_models=False))
        if "Goal frequency" in issue.message
    ]


def test_netlist_report_lists_transient_goal_parameters():
    report = inspect_netlist(netlist_path("tran_single_tone"))

    assert report.probe_nodes == ["OUT"]
    assert "TRAN:Power_dBm[OUT]" in report.tran_goal_parameters
    assert "TRAN:Gain_dB[P1@OUT]" in report.tran_goal_parameters
    assert "Power_dBm[OUT]" in report.hb_goal_parameters


def test_transient_goal_on_an_fft_bin_is_accepted(config_dir: Path):
    # .tran 1p 300p 100p → 200 ps window → 5 GHz bins; 130 GHz is the 26th.
    report = inspect_path(_tran_config(config_dir, "130GHz"), check_models=False)

    assert isinstance(report, ConfigurationReport)
    assert [goal.simulation_type for goal in report.design_goals] == [".TRAN"]
    assert _frequency_issues(_tran_config(config_dir, "130GHz")) == []


def test_transient_goal_off_the_fft_grid_is_flagged(config_dir: Path):
    messages = _frequency_issues(_tran_config(config_dir, "132GHz"))

    assert len(messages) == 1
    assert "FFT grid" in messages[0]


def test_configured_window_changes_the_fft_grid(config_dir: Path):
    # start_time=0 makes the window 300 ps → 3.33 GHz bins: 130 GHz stays on, 135 GHz falls off.
    parameters = {".TRAN": {"start_time": "0"}}
    assert _frequency_issues(_tran_config(config_dir, "135GHz")) == []
    assert _frequency_issues(_tran_config(config_dir, "135GHz", simulation_parameters=parameters))


def test_raw_print_format_is_flagged(tmp_path: Path):
    netlist = tmp_path / "raw.cir"
    netlist.write_text(
        netlist_path("tran_single_tone")
        .read_text(encoding="utf-8")
        .replace("format=csv", "format=raw file=spice4qucs.tran.plot"),
        encoding="utf-8",
    )

    messages = [issue.message for issue in inspect_netlist(netlist).issues]

    assert any("format=raw" in message for message in messages)
