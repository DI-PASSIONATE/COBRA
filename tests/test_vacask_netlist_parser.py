"""Tests for the :class:`~cobra.spice_sim.netlist_parsers.vacask_netlist_parser.VacaskNetlistParser` dialect."""

from __future__ import annotations

import pytest

from cobra.spice_sim.netlist_parsers.statement import StatementKind
from cobra.spice_sim.netlist_parsers.vacask_netlist_parser import VacaskNetlistParser, vacask_float
from tests.conftest import EXAMPLE_NETLIST_DIR


def _parse(text: str):
    return VacaskNetlistParser().parse(text)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path", sorted((EXAMPLE_NETLIST_DIR / "VACASK").glob("*.sim")), ids=lambda path: path.name
)
def test_example_netlists_round_trip_and_declare_two_ports(path):
    netlist = VacaskNetlistParser().parse_file(path)
    assert netlist.to_string() == path.read_text(encoding="utf-8")
    assert netlist.ports == {"vp1": 1, "vp2": 2}


def test_title_line_is_never_code():
    statements = VacaskNetlistParser().split("ground 0\nr1 (a 0) r r=1\n")
    assert statements[0].kind is StatementKind.TITLE
    assert statements[1].keyword == "r1"


def test_brackets_backslash_and_block_comments_join_lines():
    netlist = _parse(
        "t\n"
        "r1 (a\n"
        "    b) r r=1\n"
        "c1 (a b) c \\\n"
        "   c=2f\n"
        "/* a\n"
        "   b */\n"
        'embed "x.py" <<<FILE\n'
        "r9 (a b) r r=9\n"
        ">>>FILE\n"
    )
    assert [e.name for e in netlist.list_elements()] == ["r1", "c1"]
    assert netlist.get_element("r1").nodes == ["a", "b"]
    assert netlist.get_element("c1").params == {"c": "2f"}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def test_includes_and_libraries():
    netlist = _parse('t\ninclude "a.inc"\ninclude "pdk.lib" section=tt lang=ngspice\n')
    assert [i.file_path for i in netlist.includes] == ["a.inc"]
    assert [(lib.file_path, lib.entry) for lib in netlist.libraries] == [("pdk.lib", "tt")]


def test_probe_nodes_fall_back_to_probe_sources():
    netlist = _parse("t\nvOut (Out 0) vsource dc=0\nv1 (a 0) vsource dc=1\n")
    assert netlist.probe_nodes == ["Out"]


# ---------------------------------------------------------------------------
# Edits
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Included files
# ---------------------------------------------------------------------------


def test_masters_defined_by_includes_are_not_components(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "pdk.lib").write_text(".subckt npn13G2 c b e\n.ends\n.model rres r\n")
    (tmp_path / "native.inc").write_text('subckt amp (a b)\nends\ninclude "inner.inc"\n')
    (tmp_path / "inner.inc").write_text("model nmos psp103\n")
    deck = tmp_path / "deck.sim"
    deck.write_text(
        "t\n"
        'include "native.inc"\n'
        'include "models/pdk.lib" lang=ngspice section=tt\n'
        'include "missing.inc"\n'
        "q1 (c b e) npn13G2\n"
        "x1 (a b) amp\n"
        "m1 (d g s b) nmos\n"
        "r1 (a b) m_rres r=1\n"
        "s1 (a b) surrogate\n"
    )
    netlist = VacaskNetlistParser().parse_file(deck)
    assert list(netlist.components) == ["s1"]
    # Edits keep what the includes defined.
    netlist.set_param("s1", "k", "1")
    assert list(netlist.components) == ["s1"]


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "value"),
    [("50", 50.0), ("1k", 1e3), ("130G", 130e9), ("2m", 2e-3), ("2M", 2e6), ("1.5meg", 1.5e6),
     ("10fF", 10e-15), ("-1e-3", -1e-3)],
)
def test_vacask_float(text, value):
    assert vacask_float(text) == pytest.approx(value)


def test_vacask_float_rejects_expressions():
    with pytest.raises(ValueError, match="number"):
        vacask_float("2*r0")


# ---------------------------------------------------------------------------
# xschem-style netlists: detected ports and subcircuit surrogates
# ---------------------------------------------------------------------------

# The layout xschem's spectre netlister writes: instances first, then the user
# code with the ground and includes, the control block and the subcircuits.
_XSCHEM = """// sch_path: /designs/lna_hb.sch
//subckt lna_hb ( )
I1 ( RF_in RF_out Vdd net3 ) Amplifier
Rin ( RF_in net7 ) resistor r=50 $mfactor=1
Vin ( net7 GND ) vsource type="sine" sinedc=0 ampl=63.2456m freq=130G
Rout ( RF_out net8 ) resistor r=50 $mfactor=1
Vout ( net8 GND ) vsource dc=0
V0 ( Vdd GND ) vsource dc=2.5
Rbias ( Vdd net9 ) resistor r=1k
V2 ( net3 GND ) vsource dc=1.8
ground GND
model resistor resistor
model vsource vsource
control
  analysis hb1 hb freq=[130G] nharm=3
endc
subckt Amplifier ( in out vdd bias )
x1 ( in mid ) Match_le
x2 ( mid out ) Match_le
Rb ( bias vdd ) resistor r=1k
ends
subckt Match_le ( a b )
C1 ( a b ) capacitor c=10f
ends
"""


def test_ports_are_detected_without_acsp():
    netlist = _parse(_XSCHEM)
    assert netlist.ports == {"Vin": 1, "Vout": 2}  # not the supply V0 or the bias V2
    assert netlist.port_sources == {
        "Vin": {"sin_amplitude": 0.0632456, "sin_frequency": 130e9, "z0": 50.0},
    }
    assert netlist.probe_nodes == ["Vin", "Vout"]


def test_subcircuit_named_in_the_mapping_becomes_a_surrogate():
    netlist = _parse(_XSCHEM)
    assert netlist.masters == {"Amplifier": ("in", "out", "vdd", "bias"), "Match_le": ("a", "b")}
    netlist.select_surrogates(["Match_le", "unknown"])
    assert netlist.components["Match_le"].nodes == ["a", "b"]

    netlist.use_surrogate("Match_le", "Match_le_subct")
    text = netlist.to_string()
    assert text.splitlines()[1] == 'include "Match_le.inc"'
    assert "x1 ( in mid ) Match_le_subct" in text
    assert "x2 ( mid out ) Match_le_subct" in text
    assert "subckt Match_le ( a b )" in text  # the original definition stays, unused


def test_included_subcircuits_can_be_surrogates(tmp_path):
    (tmp_path / "blocks.inc").write_text("subckt Balun_le (p1 p2 p3)\nends\n")
    deck = tmp_path / "deck.sim"
    deck.write_text('t\ninclude "blocks.inc"\nT0 (a b c) Balun_le\n')
    netlist = VacaskNetlistParser().parse_file(deck)
    assert netlist.components == {}
    netlist.select_surrogates(["Balun_le"])
    assert netlist.components["Balun_le"].nodes == ["p1", "p2", "p3"]
