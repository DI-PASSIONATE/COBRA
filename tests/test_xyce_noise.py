"""Small-signal noise (``.NOISE``) with the Xyce backend.

The integration test runs when an ``Xyce`` executable is found: on PATH, or
named by the ``COBRA_XYCE`` environment variable.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from cobra.configuration.configuration import ConfigurationError
from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
from cobra.spice_sim.simulation_type import SimulationType
from cobra.spice_sim.vector_fit import noiseless_resistors
from cobra.spice_sim.xyce_simulator import XyceSimulator

XYCE = shutil.which(os.environ.get("COBRA_XYCE", "Xyce"))

# A matched 6 dB T pad between two 50 Ohm P ports, at T0 = 290 K.
_PAD = """* 6 dB T pad
P1 in 0 port=1 z0=50 AC 1
R1 in mid 16.6139
R3 mid 0 66.9310
R2 mid out 16.6139
P2 out 0 port=2 z0=50
.OPTIONS DEVICE TEMP=16.85
.AC LIN 2 1G 2G
.LIN format=touchstone sparcalc=1
.END
"""


def _lines(netlist) -> list[str]:
    return [line.strip() for line in netlist.to_string().splitlines()]


def test_noise_directive_is_an_analysis():
    netlist = XyceNetlistParser().parse("* t\nP1 a 0 port=1 z0=50\n.NOISE V(out) P1 LIN 10 120G 140G\n.END\n")
    assert netlist.simulation_type is SimulationType.NOISE
    [directive] = netlist.simulation_directives
    assert directive.positional == ["V(out)", "P1", "LIN", "10", "120G", "140G"]


def test_injected_noise_refers_to_port_one_and_sweeps_the_goal_band():
    simulator = XyceSimulator()
    netlist = simulator.netlist_parser.parse(_PAD)
    prepared = simulator.prepare_netlist(netlist, SimulationType.NOISE, {"out": "out"}, (130e9, 130e9))
    lines = _lines(prepared)
    assert ".NOISE V(out) P1 LIN 1 1.3e+11 1.3e+11" in lines
    assert ".PRINT NOISE format=csv ONOISE INOISE" in lines
    assert ".OPTIONS DEVICE TEMP=16.85" in lines  # applies to every analysis
    assert ".AC LIN 2 1G 2G" not in lines


def test_configured_noise_sweep_wins_over_the_goal_band():
    simulator = XyceSimulator()
    netlist = simulator.netlist_parser.parse(_PAD)
    prepared = simulator.prepare_netlist(
        netlist,
        SimulationType.NOISE,
        {"out": "V(out)", "in": "P2", "start_freq": "1G", "stop_freq": "2G"},
        (130e9, 130e9),
    )
    assert ".NOISE V(out) P2 LIN 21 1G 2G" in _lines(prepared)


def test_injected_noise_needs_an_output():
    simulator = XyceSimulator()
    with pytest.raises(ConfigurationError, match="output node"):
        simulator.prepare_netlist(simulator.netlist_parser.parse(_PAD), SimulationType.NOISE, {})


def test_own_noise_analysis_gets_the_print_it_needs():
    simulator = XyceSimulator()
    netlist = simulator.netlist_parser.parse(_PAD.replace(".AC LIN 2 1G 2G", ".NOISE V(out) P1 LIN 2 1G 2G"))
    prepared = simulator.prepare_netlist(netlist, SimulationType.NOISE, {})
    assert ".PRINT NOISE format=csv ONOISE INOISE" in _lines(prepared)
    assert simulator.prepare_netlist(prepared, SimulationType.NOISE, {}) is prepared


def test_surrogate_resistors_become_noiseless_conductances():
    text = "V1 p1 s1 0\nR1 s1 0 50.0\nRp1_a1 0 x1_a1 2.0\nGd1_1 0 s1 p1 0 -4.8e-05\n"
    assert noiseless_resistors(text) == (
        "V1 p1 s1 0\nGR1 s1 0 s1 0 0.02\nGRp1_a1 0 x1_a1 0 x1_a1 0.5\nGd1_1 0 s1 p1 0 -4.8e-05\n"
    )


def test_noise_run_drives_only_its_input_with_unit_magnitude():
    simulator = XyceSimulator()
    netlist = simulator.netlist_parser.parse(
        _PAD.replace("P1 in 0 port=1 z0=50 AC 1", "P1 in 0 port=1 z0=50")
        .replace("P2 out 0 port=2 z0=50", "P2 out 0 port=2 z0=50 AC 0.5 SIN 0 0.5 1G")
    )
    lines = _lines(simulator.prepare_netlist(netlist, SimulationType.NOISE, {"out": "out"}))
    assert "P1 in 0 port=1 z0=50 AC 1" in lines
    assert "P2 out 0 port=2 z0=50 AC 0 SIN 0 0.5 1G" in lines


@pytest.mark.skipif(XYCE is None, reason="Xyce executable not found")
@pytest.mark.parametrize(
    "drive",
    [("AC 1", ""), ("AC 2", " AC 0.5")],
    ids=["unit-input", "other-magnitudes"],
)
def test_noise_figure_of_a_matched_pad_is_its_loss(tmp_path, drive):
    """Xyce's INOISE gain sees every AC source, so COBRA drives the input alone with AC 1."""
    simulator = XyceSimulator(XYCE or "Xyce")
    input_ac, output_ac = drive
    netlist = simulator.netlist_parser.parse(
        _PAD.replace("z0=50 AC 1", f"z0=50 {input_ac}").replace("port=2 z0=50", f"port=2 z0=50{output_ac}")
    )
    prepared = simulator.prepare_netlist(netlist, SimulationType.NOISE, {"out": "out"}, (1e9, 2e9))
    path = tmp_path / "pad_noise.cir"
    prepared.save(path)
    result = simulator.run_simulation(str(path), prepared)
    assert result is not None
    [frame] = result.dataframes.values()
    np.testing.assert_allclose(frame["NF"], 6.0, atol=1e-3)


def test_parse_warns_about_a_noise_goal_outside_the_xyce_sweep(tmp_path):
    import json

    from cobra.configuration.inspection import all_issues, inspect_path
    from tests.conftest import make_config_data

    (tmp_path / "pad.cir").write_text(_PAD.replace(".AC LIN 2 1G 2G", ".NOISE V(out) P1 LIN 2 1G 2G"))
    data = make_config_data(
        netlist="pad.cir",
        component_models={},
        simulation_parameters={},
        design_goals=[
            {"parameter": "NF", "frequency_range": "130GHz", "min_value": None, "max_value": 5.0,
             "weight": 1.0, "kind": "catalogue"},
        ],
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps(data))
    messages = [issue.message for issue in all_issues(inspect_path(config, check_models=False))]
    assert any("outside the noise sweep 1e+09-2e+09 Hz" in message for message in messages), messages
