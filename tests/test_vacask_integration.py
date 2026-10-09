"""End-to-end checks of the VACASK backend against circuits with known answers.

Skipped unless a ``vacask`` executable is found: on PATH, or named by the
``COBRA_VACASK`` environment variable.
"""

from __future__ import annotations

import math
import os
import shutil

import numpy as np
import pytest
import skrf as rf

from cobra.spice_sim import hb_spectrum
from cobra.spice_sim.simulation_type import SimulationType
from cobra.spice_sim.vacask_simulator import VacaskSimulator
from cobra.spice_sim.vector_fit import vector_fit_vacask

VACASK = shutil.which(os.environ.get("COBRA_VACASK", "vacask"))
pytestmark = pytest.mark.skipif(VACASK is None, reason="vacask executable not found")

# A 0.2 V sine source with 50 Ohm into a 50 Ohm load: -10 dBm at the load, S11 = 0.
_MATCHED_PAD = """Matched source into a matched load
ground 0
load "resistor.osdi"
model r resistor
model vsource vsource
vp1 (a 0) vsource dc=0 mag=1 type="sine" ampl=0.2 freq=1G
rp1 (a out) r r=50
vout (out x) vsource dc=0
rload (x 0) r r=50
control
  analysis sp1 acsp ports=["vp1", "rp1"] from=1G to=2G mode="lin" points=1
  analysis hb1 hb freq=[1G] nharm=3
  analysis tr1 tran step=10p stop=20n start=10n maxstep=10p
endc
"""


def _run(tmp_path, text: str, sim_type: SimulationType):
    simulator = VacaskSimulator(VACASK or "vacask")
    deck = tmp_path / "deck.sim"
    deck.write_text(text)
    netlist = simulator.netlist_parser.parse_file(deck)
    prepared = simulator.prepare_netlist(netlist, sim_type, {})
    path = tmp_path / f"deck_{sim_type.name.lower()}.sim"
    prepared.save(path)
    result = simulator.run_simulation(str(path), prepared)
    assert result is not None
    return result


@pytest.mark.parametrize("sim_type", [SimulationType.HB, SimulationType.TRAN])
def test_large_signal_power_matches_the_analytic_value(tmp_path, sim_type):
    """Pins down the phasor convention: VACASK peak amplitudes are halved like Xyce's."""
    result = _run(tmp_path, _MATCHED_PAD, sim_type)
    frame = hb_spectrum.find_dataframe(result.dataframes, "out", "power")
    assert frame is not None
    _, power = hb_spectrum.spectrum(frame, "out", "power", frequency_range=(0.95e9, 1.05e9))
    expected = 10 * math.log10((0.1**2 / (2 * 50)) / 1e-3)
    np.testing.assert_allclose(power, [expected], atol=0.01)


def test_acsp_matches_the_analytic_two_port(tmp_path):
    """An RC network with a VCCS and unequal port impedances (VACASK's own acsp test)."""
    deck = """RC two-port
ground 0
load "resistor.osdi"
load "capacitor.osdi"
model r resistor
model c capacitor
model vsrc vsource
model vccs vccs
r1 (1 0) r r=1k
r2 (2 0) r r=2k
c1 (1 2) c c=1u
g1 (2 0 1 0) vccs gain=1e-3
vp1 (a1 0) vsrc dc=0
rp1 (a1 1) r r=50
vp2 (a2 0) vsrc dc=0
rp2 (a2 2) r r=75
control
  analysis sp1 acsp ports=["vp1", "rp1", "vp2", "rp2"] from=1 to=100M mode="dec" points=5
endc
"""
    network = _run(tmp_path, deck, SimulationType.AC).network
    assert network is not None
    jw = 2j * np.pi * network.f
    y = np.array([[1 / 1e3 + jw * 1e-6, -jw * 1e-6], [-jw * 1e-6 + 1e-3, 1 / 2e3 + jw * 1e-6]])
    expected = rf.network.y2s(np.transpose(y, (2, 0, 1)), z0=np.array([50.0, 75.0]))
    np.testing.assert_allclose(network.z0[0], [50.0, 75.0])
    np.testing.assert_allclose(network.s, expected, atol=1e-6)
    # The Touchstone copy keeps both reference impedances and leaves the netlist alone.
    written = rf.Network(str(tmp_path / "deck_ac.sim.s2p"))
    np.testing.assert_allclose(written.z0[0], [50.0, 75.0])
    np.testing.assert_allclose(written.s, network.s, atol=1e-9)
    assert (tmp_path / "deck_ac.sim").read_text().startswith("RC two-port")


def test_snp2le_subcircuit_reproduces_its_model(tmp_path):
    """The include COBRA writes runs in a netlist grounded at 0 and matches snp2le's own model."""
    from snp2le.core.engine import convert
    from snp2le.core.state import ConverterState

    frequency = rf.Frequency(1, 20, 101, "GHz")
    media = rf.media.DefinedGammaZ0(frequency, z0=50)
    network = media.line(30, "deg") ** media.shunt_capacitor(0.2e-12) ** media.resistor(5)
    vector_fit_vacask(network, str(tmp_path / "X1"))
    deck = """snp2le round trip
ground 0
load "resistor.osdi"
model r resistor
model vsource vsource
include "X1.inc"
X1 (n1 n2) X1_subct
vp1 (a1 0) vsource dc=0
rp1 (a1 n1) r r=50
vp2 (a2 0) vsource dc=0
rp2 (a2 n2) r r=50
control
  analysis sp1 acsp ports=["vp1", "rp1", "vp2", "rp2"] from=1G to=20G mode="lin" points=100
endc
"""
    simulated = _run(tmp_path, deck, SimulationType.AC).network
    assert simulated is not None
    model = convert(ConverterState(mode="universal", enforce_passivity=False, max_order=12), network)
    np.testing.assert_allclose(simulated.s, model.model_s, atol=1e-6)


# A matched 6 dB T pad between two 50 Ohm ports, simulated at T0 = 290 K.
_PAD_6DB = """6 dB T pad between two 50 Ohm ports
ground 0
load "resistor.osdi"
model r resistor
model vsource vsource
vp1 (a1 0) vsource dc=0 mag=1
rp1 (a1 in) r r=50
r1 (in mid) r r=16.6139
r3 (mid 0) r r=66.9310
r2 (mid out) r r=16.6139
vp2 (a2 0) vsource dc=0
rp2 (a2 out) r r=50
control
  options temp=16.85
  analysis sp1 acsp ports=["vp1", "rp1", "vp2", "rp2"] from=1G to=2G mode="lin" points=1
endc
"""


def _noise_frame(tmp_path, sim_type: SimulationType, params: dict[str, str]):
    simulator = VacaskSimulator(VACASK or "vacask")
    deck = tmp_path / "pad.sim"
    deck.write_text(_PAD_6DB)
    netlist = simulator.netlist_parser.parse_file(deck)
    prepared = simulator.prepare_netlist(netlist, sim_type, params)
    path = tmp_path / f"pad_{sim_type.name.lower()}.sim"
    prepared.save(path)
    result = simulator.run_simulation(str(path), prepared)
    assert result is not None
    [frame] = result.dataframes.values()
    return frame


def test_noise_figure_of_a_matched_pad_is_its_loss(tmp_path):
    frame = _noise_frame(tmp_path, SimulationType.NOISE, {"out": "out", "from": "1G", "to": "2G", "points": "1"})
    np.testing.assert_allclose(frame["NF"], 6.0, atol=1e-3)


def _supports_hbnoise(tmp_path) -> bool:
    deck = tmp_path / "probe.sim"
    deck.write_text('probe\nground 0\nmodel vsource vsource\nv1 (a 0) vsource dc=0\ncontrol\n'
                    '  analysis h hbnoise freq=[1G] in="v1" out="a" from=1M to=2M\nendc\n')
    import subprocess

    proc = subprocess.run([VACASK or "vacask", "-sp", "-qp", deck.name], cwd=tmp_path, capture_output=True, text=True, check=False)
    return "not found" not in proc.stderr + proc.stdout


def test_hbnoise_dsb_figure_of_a_linear_pad_is_its_loss(tmp_path):
    if not _supports_hbnoise(tmp_path):
        pytest.skip("this VACASK build has no hbnoise analysis (it is on VACASK main only)")
    # A linear pad converts nothing: with the input at the output sideband the
    # DSB figure is the small-signal one.
    frame = _noise_frame(
        tmp_path,
        SimulationType.HBNOISE,
        {"out": "out", "freq": "10G", "inspur": "0", "from": "1G", "to": "2G", "points": "1"},
    )
    np.testing.assert_allclose(frame["NF_DSB"], 6.0, atol=1e-2)


# An xschem-style netlist: no acsp line, no probe sources, and the block to
# replace is a subcircuit used inside the hierarchy.
_XSCHEM_PAD = """// sch_path: pad_hb.sch
I1 ( RF_in RF_out ) Chain
Rin ( RF_in net7 ) resistor r=50
Vin ( net7 GND ) vsource type="sine" sinedc=0 ampl=0.2 freq=1G
Rout ( RF_out net8 ) resistor r=50
Vout ( net8 GND ) vsource dc=0
ground GND
load "resistor.osdi"
model resistor resistor
model vsource vsource
control
  analysis hb1 hb freq=[1G] nharm=3
endc
subckt Chain ( in out )
x1 ( in out ) Pad_le
ends
subckt Pad_le ( a b )
r1 ( a b ) resistor r=1
ends
"""


def test_subcircuit_surrogate_and_port_gain(tmp_path):
    """A 3 dB attenuator replaces Pad_le deep in the hierarchy; the port-based gain is -3 dB."""
    frequency = rf.Frequency(0.5, 2, 31, "GHz")
    pad = rf.media.DefinedGammaZ0(frequency, z0=50).attenuator(-3, d_b=True, name="Pad_le")
    simulator = VacaskSimulator(VACASK or "vacask")
    deck = tmp_path / "pad.sim"
    deck.write_text(_XSCHEM_PAD)
    netlist = simulator.netlist_parser.parse_file(deck)
    netlist.select_surrogates(["Pad_le"])
    netlist.use_surrogate("Pad_le", "Pad_le_subct")
    simulator.preprocess_ntwk(pad, name=str(tmp_path / "Pad_le"))
    prepared = simulator.prepare_netlist(netlist, SimulationType.HB, {})
    path = tmp_path / "pad_hb.sim"
    prepared.save(path)
    result = simulator.run_simulation(str(path), prepared)
    assert result is not None

    from cobra.optimizers.design_goal_collection import make_gain_db

    source = prepared.port_sources["Vin"]
    gain = make_gain_db("Vin", source["sin_amplitude"], source["z0"], "Vout", SimulationType.HB)
    np.testing.assert_allclose(gain.formula(result, "1GHz"), [-3.0], atol=0.05)


# An ideal multiplying mixer (Verilog-A, compiled by VACASK). The multiplier is
# noiseless, so with nothing else the noise is all the source's: F_DSB = 1 and,
# with equal upper and lower sideband gains, F_SSB = 2. A noisy 50 Ohm shunt at
# the RF node adds as much noise as the 50 Ohm source: F_DSB = 2, F_SSB = 4.
_MIXER = """Ideal multiplying mixer
ground 0
load "resistor.osdi"
load "mult.va"
model r resistor
model vsource vsource
model mult mult gm=10m
vlo (lo 0) vsource type="sine" ampl=1 freq=1G
x1 (lo rf out) mult
vp1 (a1 0) vsource dc=0 mag=1
rp1 (a1 rf) r r=50
{shunt}
vp2 (a2 0) vsource dc=0
rp2 (a2 out) r r=50
control
  options temp=16.85
  analysis sp1 acsp ports=["vp1", "rp1", "vp2", "rp2"] from=1M to=2M mode="lin" points=1
endc

embed "mult.va" <<<FILE
`include "disciplines.vams"
module mult(lo, rf, out);
  inout lo, rf, out;
  electrical lo, rf, out;
  parameter real gm = 1e-2;
  analog I(out) <+ -gm * V(lo) * V(rf);
endmodule
>>>FILE
"""


@pytest.mark.parametrize(
    ("shunt", "nf_dsb", "nf_ssb"),
    [("", 0.0, 10 * np.log10(2)), ("rsh (rf 0) r r=50", 10 * np.log10(2), 10 * np.log10(4))],
    ids=["noiseless-mixer", "with-shunt"],
)
def test_hbnoise_mixer_noise_figures_match_the_analytic_values(tmp_path, shunt, nf_dsb, nf_ssb):
    if not _supports_hbnoise(tmp_path):
        pytest.skip("this VACASK build has no hbnoise analysis (it is on VACASK main only)")
    simulator = VacaskSimulator(VACASK or "vacask")
    deck = tmp_path / "mixer.sim"
    deck.write_text(_MIXER.replace("{shunt}", shunt))
    netlist = simulator.netlist_parser.parse_file(deck)
    prepared = simulator.prepare_netlist(
        netlist,
        SimulationType.HBNOISE,
        {"out": "out", "freq": "1G", "nharm": "5", "from": "50M", "to": "100M", "points": "1"},
    )
    path = tmp_path / "mixer_hbnoise.sim"
    prepared.save(path)
    result = simulator.run_simulation(str(path), prepared)
    assert result is not None
    [frame] = result.dataframes.values()
    # The default spurs: signal in the upper sideband of the LO, image in the lower one.
    np.testing.assert_allclose(frame["GAIN_SIG"], frame["GAIN_IMG"], rtol=1e-6)
    np.testing.assert_allclose(frame["NF_DSB"], nf_dsb, atol=1e-3)
    np.testing.assert_allclose(frame["NF_SSB"], nf_ssb, atol=1e-3)
