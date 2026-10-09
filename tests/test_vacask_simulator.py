"""Tests for :class:`~cobra.spice_sim.vacask_simulator.VacaskSimulator` that need no VACASK binary."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cobra.configuration.configuration import ConfigurationError
from cobra.spice_sim.base_simulator import SimulatorError
from cobra.spice_sim.raw_file import RawPlot
from cobra.spice_sim.simulation_type import SimulationType
from cobra.spice_sim.vacask_simulator import (
    T0,
    VacaskSimulator,
    hb_dataframe,
    noise_factor,
    tran_dataframe,
)
from tests.conftest import NETLIST_DIR

FIXTURE = NETLIST_DIR / "vacask_trafo.sim"


@pytest.fixture
def simulator():
    return VacaskSimulator()


@pytest.fixture
def netlist(simulator):
    return simulator.netlist_parser.parse_file(FIXTURE)


def _analysis_lines(netlist) -> list[str]:
    return [line.strip() for line in netlist.to_string().splitlines() if "analysis " in line]


# ---------------------------------------------------------------------------
# prepare_netlist
# ---------------------------------------------------------------------------


def test_prepare_keeps_only_the_requested_analysis(simulator, netlist):
    prepared = simulator.prepare_netlist(netlist, SimulationType.HB, {})
    assert prepared.simulation_type is SimulationType.HB
    assert [d.name for d in prepared.simulation_directives] == ["hb1"]
    assert len(prepared.to_string().splitlines()) == len(netlist.to_string().splitlines())
    assert _analysis_lines(prepared) == [
        "//cobra-off   analysis op1 op",
        '//cobra-off   analysis sp1 acsp ports=["vp1", "rp1",',
        "analysis hb1 hb freq=[95G, 10G] nharm=4",
    ]
    # The switched-off acsp still declares the ports.
    assert prepared.ports == {"vp1": 1, "vp2": 2}
    assert prepared.port_sources == netlist.port_sources
    # What the includes define survives the rewrite.
    assert prepared.external_masters == netlist.external_masters


def test_prepare_injects_a_missing_analysis(simulator, netlist):
    prepared = simulator.prepare_netlist(netlist, SimulationType.TRAN, {"stop": "50n"})
    assert [d.name for d in prepared.simulation_directives] == ["cobra_tran"]
    assert "analysis cobra_tran tran step=1p stop=50n start=0 maxstep=1p" in _analysis_lines(prepared)


def test_prepare_injects_hb_tones_as_a_vector(simulator):
    netlist = simulator.netlist_parser.parse("t\nr1 (a 0) r r=1\ncontrol\nendc\n")
    prepared = simulator.prepare_netlist(netlist, SimulationType.HB, {"freq": "95G 10G", "nharm": "5"})
    assert _analysis_lines(prepared) == ["analysis cobra_hb hb freq=[95G, 10G] nharm=5"]


def test_prepare_adds_a_control_block_when_there_is_none(simulator):
    netlist = simulator.netlist_parser.parse("t\nr1 (a 0) r r=1")
    prepared = simulator.prepare_netlist(netlist, SimulationType.HB, {})
    assert prepared.to_string().endswith("control\n  analysis cobra_hb hb freq=[1G]\nendc\n")
    assert prepared.simulation_type is SimulationType.HB


def test_prepare_returns_the_netlist_when_it_runs_only_that_analysis(simulator):
    netlist = simulator.netlist_parser.parse("t\ncontrol\n  analysis h hb freq=[1G]\nendc\n")
    assert simulator.prepare_netlist(netlist, SimulationType.HB, {}) is netlist


def test_prepare_ac_needs_declared_ports(simulator):
    netlist = simulator.netlist_parser.parse("t\ncontrol\n  analysis h hb freq=[1G]\nendc\n")
    with pytest.raises(ConfigurationError, match="acsp"):
        simulator.prepare_netlist(netlist, SimulationType.AC, {})


# ---------------------------------------------------------------------------
# Result tables
# ---------------------------------------------------------------------------


def test_hb_table_halves_every_harmonic_but_dc():
    plot = RawPlot(
        "Harmonic Balance Analysis",
        {
            "frequency": np.array([0.0, 1e9, 2e9], dtype=complex),
            "out": np.array([1.0, 2.0 - 4.0j, 1.0j]),
            "vout:flow(br)": np.array([0.5, 0.2, 0.0], dtype=complex),
        },
    )
    frame = hb_dataframe(plot)
    assert list(frame.columns) == ["FREQ", "Re(V(out))", "Im(V(out))", "Re(I(vout))", "Im(I(vout))"]
    np.testing.assert_allclose(frame["Re(V(out))"], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(frame["Im(V(out))"], [0.0, -2.0, 0.5])
    np.testing.assert_allclose(frame["Re(I(vout))"], [0.5, 0.1, 0.0])


def test_tran_table_names_signals_like_xyce():
    plot = RawPlot(
        "Transient Analysis",
        {"time": np.array([0.0, 1.0]), "out": np.array([0.0, 1.0]), "vout:flow(br)": np.array([1.0, 2.0])},
    )
    assert list(tran_dataframe(plot).columns) == ["TIME", "V(out)", "I(vout)"]


def test_network_is_referenced_to_each_port_resistor(simulator, netlist):
    s21 = np.array([0.5, 0.25], dtype=complex)
    plot = RawPlot(
        "AC S-parameter Analysis",
        {
            "s(1,1)": np.zeros(2, complex), "s(1,2)": s21, "s(2,1)": s21, "s(2,2)": np.zeros(2, complex),
            "frequency": np.array([1e9, 2e9], dtype=complex),
        },
    )
    directive = next(d for d in netlist.simulation_directives if d.name == "sp1")
    network = simulator._network(plot, directive, netlist)
    np.testing.assert_allclose(network.z0[0], [925.0, 100.0])
    np.testing.assert_allclose(network.s[:, 1, 0], s21)
    np.testing.assert_allclose(network.f, [1e9, 2e9])


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def test_missing_executable_is_a_simulator_error(tmp_path, netlist):
    path = tmp_path / "deck.sim"
    netlist.save(path)
    with pytest.raises(SimulatorError, match="vacask_command"):
        VacaskSimulator("/definitely/not/vacask").run_simulation(str(path), netlist)


def test_failed_run_returns_none(tmp_path, netlist):
    path = tmp_path / "deck.sim"
    netlist.save(path)
    assert VacaskSimulator("false").run_simulation(str(path), netlist) is None


@pytest.mark.parametrize("threads", [-1, 1.5, True])
def test_threads_must_be_a_non_negative_integer(threads):
    with pytest.raises(ConfigurationError, match="vacask_threads"):
        VacaskSimulator(vacask_threads=threads)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

_NOISE_DECK = """Pad between two ports
vp1 (a1 0) vsource dc=0 mag=1
rp1 (a1 in) r r=50
r1 (in out) r r=10
vp2 (a2 0) vsource dc=0
rp2 (a2 out) r r=50
control
  analysis sp1 acsp ports=["vp1", "rp1", "vp2", "rp2"] from=1G to=2G mode="lin" points=1
endc
"""


def test_noise_factor_refers_added_noise_to_t0():
    # At T = T0 the factor is the total over the source contribution, terminations excluded.
    np.testing.assert_allclose(noise_factor(np.array([4.0]), np.array([1.0]), 1.0, T0), [3.0])
    # A hotter simulation counts the added noise for more.
    np.testing.assert_allclose(noise_factor(np.array([2.0]), np.array([1.0]), 0.0, 2 * T0), [3.0])


def test_noise_injection_defaults_its_input_to_port_one(simulator):
    netlist = simulator.netlist_parser.parse(_NOISE_DECK)
    prepared = simulator.prepare_netlist(netlist, SimulationType.NOISE, {"out": "out"})
    assert _analysis_lines(prepared)[0] == (
        'analysis cobra_noise noise mode="lin" points=500 from=1G to=10G out="out" in="vp1"'
    )
    assert prepared.ports == {"vp1": 1, "vp2": 2}


def test_noise_injection_needs_an_output(simulator):
    netlist = simulator.netlist_parser.parse(_NOISE_DECK)
    with pytest.raises(ConfigurationError, match="out"):
        simulator.prepare_netlist(netlist, SimulationType.NOISE, {})


def test_hbnoise_gets_an_image_sideband_twin(simulator):
    netlist = simulator.netlist_parser.parse(_NOISE_DECK)
    prepared = simulator.prepare_netlist(
        netlist, SimulationType.HBNOISE, {"out": "out", "freq": "95G", "from": "1G", "to": "2G"}
    )
    sig, img = _analysis_lines(prepared)[:2]
    assert sig.startswith('analysis cobra_hbnoise hbnoise mode="lin" points=500 from=1G to=2G freq=[95G]')
    assert "inspur=[1] outspur=[0]" in sig
    assert img.startswith("analysis cobra_hbnoise_img hbnoise")
    assert "inspur=[-1] outspur=[0]" in img


def test_existing_hbnoise_analysis_gets_its_twin_once(simulator):
    deck = _NOISE_DECK.replace(
        "endc", '  analysis hn hbnoise freq=[95G] in="vp1" out="out" inspur=[1, 0] from=1G to=2G\nendc'
    )
    netlist = simulator.netlist_parser.parse(deck)
    prepared = simulator.prepare_netlist(netlist, SimulationType.HBNOISE, {})
    assert [d.name for d in prepared.simulation_directives] == ["hn", "hn_img"]
    assert "inspur=[-1, 0]" in _analysis_lines(prepared)[-1]
    again = simulator.prepare_netlist(prepared, SimulationType.HBNOISE, {})
    assert [d.name for d in again.simulation_directives] == ["hn", "hn_img"]


def _noise_plot(onoise, source, termination, gain, frequencies=(1e9,)) -> RawPlot:
    count = len(frequencies)
    return RawPlot(
        "Noise",
        {
            "frequency": np.array(frequencies, dtype=complex),
            "onoise": np.full(count, onoise, dtype=complex),
            "gain": np.full(count, gain, dtype=complex),
            "n(rp1)": np.full(count, source, dtype=complex),
            "n(rp2)": np.full(count, termination, dtype=complex),
        },
    )


def test_noise_table_leaves_out_the_terminations(simulator):
    netlist = simulator.netlist_parser.parse(_NOISE_DECK.replace("endc", "  options temp=16.85\nendc"))
    prepared = simulator.prepare_netlist(netlist, SimulationType.NOISE, {"out": "out"})
    directive = prepared.simulation_directives[-1]
    frame = simulator._noise_frame(_noise_plot(5.0, 1.0, 1.0, 0.25), directive, prepared, None)
    np.testing.assert_allclose(frame["NF"], [10 * np.log10(4.0)])
    assert list(frame.columns) == ["FREQ", "ONOISE", "GAIN", "NF"]


def test_hbnoise_table_gives_ssb_from_the_image_gain(simulator):
    netlist = simulator.netlist_parser.parse(_NOISE_DECK.replace("endc", "  options temp=16.85\nendc"))
    prepared = simulator.prepare_netlist(netlist, SimulationType.HBNOISE, {"out": "out"})
    directive = next(d for d in prepared.simulation_directives if d.name == "cobra_hbnoise")
    signal = _noise_plot(3.0, 1.0, 0.0, 0.5)
    image = _noise_plot(3.0, 1.0, 0.0, 0.5)
    frame = simulator._noise_frame(signal, directive, prepared, image)
    np.testing.assert_allclose(frame["NF_DSB"], [10 * np.log10(3.0)])
    np.testing.assert_allclose(frame["NF_SSB"], [10 * np.log10(6.0)])
    without_image = simulator._noise_frame(signal, directive, prepared, None)
    assert "NF_SSB" not in without_image.columns


def test_noise_figure_goal_selects_the_frequency(simulator):
    from cobra.optimizers.design_goal_collection import find_parameter
    from cobra.spice_sim.base_simulator import SimulationResult

    netlist = simulator.netlist_parser.parse(_NOISE_DECK)
    prepared = simulator.prepare_netlist(netlist, SimulationType.NOISE, {"out": "out"})
    directive = prepared.simulation_directives[-1]
    plot = _noise_plot(2.0, 1.0, 0.0, 1.0, frequencies=(1e9, 2e9, 3e9))
    plot.variables["onoise"] = np.array([2.0, 3.0, 4.0], dtype=complex)
    frame = simulator._noise_frame(plot, directive, prepared, None)
    parameter = find_parameter("NF")
    assert parameter is not None
    assert parameter.simulation_type is SimulationType.NOISE
    result = SimulationResult(dataframes={"noise.csv": frame})
    assert len(parameter.formula(result, None)) == 3
    assert len(parameter.formula(result, "1-2.5GHz")) == 2
    np.testing.assert_allclose(parameter.formula(result, "2.1GHz"), frame["NF"].to_numpy()[[1]])


def test_ports_become_probe_points_named_after_their_source():
    from cobra.spice_sim import hb_spectrum
    from cobra.spice_sim.vacask_simulator import add_port_signals

    plot = RawPlot(
        "Harmonic Balance Analysis",
        {"frequency": np.array([0.0, 1e9], dtype=complex), "Vout:flow(br)": np.array([0.0, 2e-3 + 0j])},
    )
    frame = add_port_signals(hb_dataframe(plot), {"Vout": 50.0})
    # 1 mA phasor (2 mA peak halved) into 50 Ohm: P = 2 R |I|^2 = 0.1 mW = -10 dBm.
    _, power = hb_spectrum.spectrum(frame, "Vout", "power", frequency_range=(1e9, 1e9))
    np.testing.assert_allclose(power, [-10.0])
    tran = add_port_signals(
        tran_dataframe(RawPlot("t", {"time": np.array([0.0, 1.0]), "Vout:flow(br)": np.array([1.0, 2.0])})),
        {"Vout": 50.0},
    )
    np.testing.assert_allclose(tran["V(Vout)"], [50.0, 100.0])
    np.testing.assert_allclose(tran["I(VVout)"], [1.0, 2.0])


def test_parallel_surrogate_fits_leave_the_console_alone(tmp_path):
    """snp2le silences stdout/stderr while it fits; concurrent trials must not lose them."""
    import sys
    from concurrent.futures import ThreadPoolExecutor

    import skrf as rf

    from cobra.spice_sim.vector_fit import vector_fit_vacask

    frequency = rf.Frequency(1, 20, 41, "GHz")
    media = rf.media.DefinedGammaZ0(frequency, z0=50)
    network = media.line(30, "deg") ** media.shunt_capacitor(0.2e-12)
    stdout, stderr = sys.stdout, sys.stderr
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda k: vector_fit_vacask(network, str(tmp_path / f"X{k}")), range(16)))
    assert sys.stdout is stdout
    assert sys.stderr is stderr


def test_a_fit_that_does_not_converge_raises_instead_of_looping(tmp_path, monkeypatch):
    """scikit-rf's auto_fit can cycle forever at the order cap; the fit gives up with an error."""
    import sys

    import skrf as rf
    from skrf.vectorFitting import VectorFitting

    from cobra.spice_sim.vector_fit import VectorFitError, vector_fit_vacask

    frequency = rf.Frequency(1, 20, 41, "GHz")
    media = rf.media.DefinedGammaZ0(frequency, z0=50)
    network = media.line(30, "deg") ** media.shunt_capacitor(0.2e-12)
    relocate = VectorFitting._pole_relocation
    # The package re-exports the function vector_fit under the module's name.
    monkeypatch.setattr(sys.modules[vector_fit_vacask.__module__], "_MAX_POLE_RELOCATIONS", 2)

    with pytest.raises(VectorFitError, match="did not converge in 2 pole relocations"):
        vector_fit_vacask(network, str(tmp_path / "X1"))
    assert VectorFitting._pole_relocation is relocate


# ---------------------------------------------------------------------------
# Noise sweeps follow the goals
# ---------------------------------------------------------------------------


def test_injected_noise_sweep_covers_the_goal_band(simulator):
    netlist = simulator.netlist_parser.parse(_NOISE_DECK)
    prepared = simulator.prepare_netlist(netlist, SimulationType.NOISE, {"out": "out"}, (120e9, 140e9))
    assert "from=1.2e+11 to=1.4e+11" in _analysis_lines(prepared)[0]
    single = simulator.prepare_netlist(netlist, SimulationType.NOISE, {"out": "out"}, (130e9, 130e9))
    assert "points=1 from=1.3e+11 to=1.3e+11" in _analysis_lines(single)[0]


def test_configured_noise_sweep_wins_over_the_goal_band(simulator):
    netlist = simulator.netlist_parser.parse(_NOISE_DECK)
    prepared = simulator.prepare_netlist(
        netlist, SimulationType.HBNOISE, {"out": "out", "from": "1G", "to": "2G"}, (120e9, 140e9)
    )
    assert "from=1G to=2G" in _analysis_lines(prepared)[0]


def test_goal_band_spans_the_goals_with_a_frequency():
    from cobra.optimizers.design_goal import DesignGoal
    from cobra.optimizers.design_goal_collection import find_parameter
    from cobra.stages.circuit_sim_stage import goal_band

    parameter = find_parameter("NF")
    assert parameter is not None
    goals = [
        DesignGoal(parameter, frequency_range="120-130GHz", max_value=5.0),
        DesignGoal(parameter, frequency_range="140GHz", max_value=6.0),
        DesignGoal(parameter, max_value=7.0),
    ]
    assert goal_band(goals) == (120e9, 140e9)
    assert goal_band(goals[2:]) is None


def test_parse_warns_about_a_noise_goal_outside_its_sweep(tmp_path):
    from cobra.configuration.inspection import all_issues, inspect_path
    from tests.conftest import make_config_data

    (tmp_path / "pad.sim").write_text(_NOISE_DECK)
    data = make_config_data(
        netlist="pad.sim",
        component_models={},
        simulator={"name": "VacaskSimulator", "settings": {}},
        simulation_parameters={"NOISE": {"out": "out", "from": "1G", "to": "10G"}},
        optimization_parameters=[
            {"name": "r1", "type": "netlist_variable", "min_value": 1.0, "max_value": 20.0,
             "step": 1.0, "unit": None, "linked_to": None},
        ],
        design_goals=[
            {"parameter": "NF", "frequency_range": "130GHz", "min_value": None, "max_value": 5.0,
             "weight": 1.0, "kind": "catalogue"},
        ],
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps(data))
    messages = [issue.message for issue in all_issues(inspect_path(config, check_models=False))]
    assert any("outside the noise sweep 1e+09-1e+10 Hz" in message for message in messages), messages


def test_a_missing_pdk_include_is_named_in_the_mapping_error(tmp_path, monkeypatch):
    """PDK devices from an include VACASK cannot find look like surrogates; say which file is missing."""
    from cobra.configuration.config_runner import build_configured_run
    from cobra.configuration.configuration import RunConfiguration
    from tests.conftest import MINIMAL_S2P, make_config_data

    monkeypatch.delenv("SIM_INCLUDE_PATH", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.vacaskrc.toml
    (tmp_path / "model.s2p").write_text(MINIMAL_S2P, encoding="utf-8")
    (tmp_path / "deck.sim").write_text(
        "deck\nground 0\nmodel r resistor\nmodel vsource vsource\n"
        'include "pdk.lib"\ninclude "X1.inc"\n'
        "X1 (a b) s_equivalent\nxq1 (b a 0 0) npn13G2 nx=1.0\nR1 (a 0) r r=50\n"
        "vp1 (p1 0) vsource dc=0 mag=1\nrp1 (p1 a) r r=50\nvp2 (p2 0) vsource dc=0\nrp2 (p2 b) r r=50\n"
        'control\n  analysis sp1 acsp ports=["vp1", "rp1", "vp2", "rp2"] from=1G to=10G mode="lin" points=9\nendc\n',
        encoding="utf-8",
    )
    data = make_config_data(netlist="deck.sim", simulator={"name": "VacaskSimulator"}, simulation_parameters={})
    config = RunConfiguration.from_dict(data, tmp_path)

    with pytest.raises(ConfigurationError, match=r"missing models for xq1; .*not found.*: pdk\.lib \("):
        build_configured_run(config)
