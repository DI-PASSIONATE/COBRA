"""Tests for :class:`~cobra.spice_sim.netlist_parsers.netlist.Netlist` and the
:class:`~cobra.spice_sim.netlist_parsers.xyce_netlist_parser.XyceNetlistParser` dialect.
"""

from __future__ import annotations

import logging

import pytest

from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
from cobra.spice_sim.simulation_type import SimulationType
from tests.conftest import EXAMPLE_NETLIST_DIR, NETLIST_DIR, netlist_path

EXAMPLE_NETLISTS = sorted(EXAMPLE_NETLIST_DIR.rglob("*.cir"))


def _parse(text: str):
    return XyceNetlistParser().parse(text)


# ---------------------------------------------------------------------------
# Round-trip fidelity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path", sorted(NETLIST_DIR.glob("*.cir")) + EXAMPLE_NETLISTS, ids=lambda path: path.name
)
def test_split_is_byte_identical(path):
    """Splitting into statements must keep every byte, so untouched lines render as they were."""
    text = path.read_text(encoding="utf-8")
    statements = XyceNetlistParser().split(text)
    assert "".join(statement.text for statement in statements) == text


@pytest.mark.parametrize(
    "name", ["minimal_ac", "hb_two_tone", "subckt_and_includes", "simulatable_ac"]
)
def test_roundtrip_is_byte_identical(netlist_factory, name):
    """Parsing then serialising an untouched netlist must not alter a single byte."""
    netlist = netlist_factory(name)
    assert netlist.to_string() == netlist_path(name).read_text(encoding="utf-8")


def test_edit_touches_only_the_edited_line(netlist_factory):
    netlist = netlist_factory("minimal_ac")
    before = netlist.to_string().splitlines()
    netlist.set_value("R1", "75")
    after = netlist.to_string().splitlines()

    assert len(before) == len(after)
    changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert len(changed) == 1
    assert after[changed[0]].strip() == "R1 in n1 75"


def test_inline_comment_spacing_and_missing_final_newline_survive_an_edit():
    netlist = _parse("* header\nR1  a  b  50 ; keep me\nR2 b 0 50")

    netlist.set_value("R1", "75")
    netlist.set_value("R2", "75")

    assert netlist.to_string() == "* header\nR1  a  b  75 ; keep me\nR2 b 0 75"


def test_the_first_line_is_the_title():
    """Like Xyce, the first line is a title even when it looks like an element."""
    netlist = _parse("R0 a b 1\nR1 a b 50\n")
    assert [element.name for element in netlist.list_elements()] == ["R1"]


def test_an_included_file_has_no_title():
    netlist = XyceNetlistParser().parse(".SUBCKT XFMR 1 2\nR1 1 2 1\n.ENDS\n", has_title=False)
    assert netlist.inline_subckt_names == {"XFMR"}


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def test_minimal_ac_structure(minimal_ac):
    assert minimal_ac.simulation_type is SimulationType.AC
    assert minimal_ac.num_ports == 2
    assert minimal_ac.components["X1"].model == "surrogate_model"
    assert minimal_ac.components["X1"].nodes == ["in", "out"]
    # XF1 references a .SUBCKT defined in this file, so it needs no surrogate.
    assert set(minimal_ac.components) == {"X1"}


def test_devices_inside_subcircuits_are_not_top_level(netlist_factory):
    """.SUBCKT bodies must be skipped, including nested ones."""
    netlist = netlist_factory("subckt_and_includes")
    names = {e.name for e in netlist.list_elements()}

    assert {"Q1", "D1", "M2", "XB1", "X2"} <= names
    assert "MB1" not in names  # inside .SUBCKT buffer
    assert "RIN" not in names  # inside the nested .SUBCKT inner
    assert "XI9" not in names  # inside .SUBCKT buffer, after the nested .ENDS
    assert netlist.inline_subckt_names == {"buffer", "inner"}


def test_an_instance_may_come_before_its_subcircuit():
    """Regression: an X line before its .SUBCKT was taken for a surrogate component."""
    netlist = _parse("* order\nX1 a b mysub\n.SUBCKT MySub p q\nR1 p q 1\n.ENDS\n.END\n")
    assert netlist.components == {}


def test_device_and_model_lines(netlist_factory):
    netlist = netlist_factory("subckt_and_includes")

    m2 = netlist.get_element("M2")
    assert m2.nodes == ["d", "g", "s", "bulk"]
    assert m2.model == "nch"
    assert m2.params == {"W": "2u", "L": "0.1u"}

    model = netlist.get_element("nch")
    assert model.etype == "MODEL"
    assert model.params == {"LEVEL": "1", "VTO": "0.7"}


def test_includes_and_libraries(netlist_factory):
    netlist = netlist_factory("subckt_and_includes")

    assert [i.file_path for i in netlist.includes] == ["extra_models.sp"]
    assert [(lib.file_path, lib.entry) for lib in netlist.libraries] == [
        ("cornerHBT.lib", "hbt_typ")
    ]


def test_simulation_and_print_directives(minimal_ac):
    ac, lin = minimal_ac.simulation_directives
    assert ac.positional == ["LIN", "101", "1G", "10G"]
    assert lin.kv_params == {"format": "touchstone", "sparcalc": "1"}
    # .LIN post-processes .AC, so it is not an analysis of its own.
    assert (ac.is_analysis, lin.is_analysis) == (True, False)
    assert ac.simulation_type is lin.simulation_type is SimulationType.AC

    (printed,) = minimal_ac.print_directives
    assert printed.signals == ["v(out)"]


def test_port_sources(netlist_factory, minimal_ac):
    """``SIN <offset> <amplitude> <freq>``: amplitude and frequency are the 2nd and 3rd
    arguments. A port without a source is left out.
    """
    assert set(minimal_ac.port_sources) == {"P1"}
    netlist = netlist_factory("hb_two_tone")
    assert netlist.port_sources["P1"] == {"z0": 50.0, "sin_amplitude": 0.2, "sin_frequency": 95e9}


@pytest.mark.parametrize("source", ["SIN(0 0.2 95G)", "SIN (0 0.2 95G)"])
def test_port_sources_in_bracketed_form(source):
    netlist = _parse(f"* ports\nP1 rfin 0 port=1 z0=50 {source}\n")
    assert netlist.port_sources["P1"] == {"z0": 50.0, "sin_amplitude": 0.2, "sin_frequency": 95e9}


def test_probe_nodes_need_matching_voltage_and_current(netlist_factory):
    assert netlist_factory("hb_two_tone").probe_nodes == ["OUT"]
    assert netlist_factory("tran_single_tone").probe_nodes == ["OUT"]


def test_parameters_with_spaces_and_expressions():
    netlist = _parse("* tokens\nX1 a b sub w = 2u l={L * 2}\nR1 a b {R * 2}\n")

    assert netlist.get_element("X1").params == {"w": "2u", "l": "{L * 2}"}
    assert netlist.get_element("X1").model == "sub"
    assert netlist.get_element("R1").value == "{R * 2}"


def test_parameter_values_include_touching_tokens():
    netlist = _parse("* values\nX1 a b sub off=-{dv} w=1u\n")
    netlist.set_param("X1", "off", "0.5")

    assert netlist.get_element("X1").params == {"off": "0.5", "w": "1u"}
    assert "X1 a b sub off=0.5 w=1u" in netlist.to_string()


def test_a_comment_only_line_does_not_take_a_continuation():
    netlist = _parse("* comments\nR1 a b\n; note\n+ 1k\n")
    assert netlist.get_element("R1").value == "1k"


def test_parameters_on_continuation_lines():
    netlist = _parse("* continued\nX1 a b sub\n* note\n+ w=1u ; inline\n+ l=2u\n")

    assert netlist.get_element("X1").params == {"w": "1u", "l": "2u"}


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------


def test_mutators_reject_unsupported_elements(netlist_factory):
    netlist = netlist_factory("minimal_ac")
    with pytest.raises(ValueError, match="set_value is for R/C/L"):
        netlist.set_value("X1", "5")
    with pytest.raises(ValueError, match="set_model not supported"):
        netlist.set_model("R1", "whatever")
    with pytest.raises(ValueError, match="too short"):
        _parse("* short\nR1 a b\n.END\n").set_value("R1", "50")


def test_a_value_given_as_a_parameter_is_not_overwritten():
    netlist = _parse("* keyword value\nR1 a b R=1k\n")

    assert netlist.get_element("R1").params == {"R": "1k"}
    with pytest.raises(ValueError, match="key=value"):
        netlist.set_value("R1", "2k")
    netlist.update_parameters({"R1:R": "2k"})
    assert "R1 a b R=2k" in netlist.to_string()


def test_an_instance_wins_over_a_model_with_the_same_folded_name():
    netlist = _parse("* names\nQ1 c b e q1 area=1\n.MODEL q1 NPN(BF=100)\n")
    netlist.set_param("Q1", "area", "2")

    assert netlist.get_element("Q1").etype == "Q"
    assert netlist.get_element("Q1").params == {"area": "2"}
    assert ".MODEL q1 NPN(BF=100)" in netlist.to_string()


def test_set_model_keeps_key_value_params_in_place(netlist_factory):
    """The subcircuit name is the last token *before* the first key=value."""
    netlist = netlist_factory("subckt_and_includes")
    netlist.set_model("X2", "replacement_sub")

    element = netlist.get_element("X2")
    assert element.params == {"param1": "5"}
    line = netlist.to_string().splitlines()[element.line_index]
    assert line.strip() == "X2 p1 p2 replacement_sub param1=5"


def test_set_param_updates_case_insensitively_and_appends_missing_keys(netlist_factory):
    netlist = netlist_factory("subckt_and_includes")
    netlist.set_param("M2", "w", "9u")
    netlist.set_param("M2", "AD", "1p")

    # The netlist keeps its own spelling of an existing key.
    assert netlist.get_element("M2").params == {"W": "9u", "L": "0.1u", "AD": "1p"}


def test_set_param_on_a_continuation_line_does_not_duplicate_it():
    netlist = _parse("* continued\nX1 a b sub\n+ w=1u ; width\n.END\n")
    netlist.set_param("X1", "w", "3u")

    assert netlist.to_string() == "* continued\nX1 a b sub\n+ w=3u ; width\n.END\n"


def test_set_value_across_continuation_lines_keeps_the_line_count():
    netlist = _parse("* source\nV1 a 0 DC 1\n+ AC 1\nR1 a 0 50\n")
    netlist.set_value("V1", "2")

    assert netlist.get_element("V1").value == "2"
    assert netlist.get_element("R1").line_index == 3
    assert len(netlist.to_string().splitlines()) == 4


def test_set_param_on_a_bracketed_model():
    netlist = _parse("* model\n.MODEL n NMOS (LEVEL=1\n+ VTO=0.7)\n")
    netlist.set_param("n", "vto", "0.5")
    netlist.set_param("n", "KP", "1e-4")

    assert netlist.get_element("n").params == {"LEVEL": "1", "VTO": "0.5", "KP": "1e-4"}
    assert netlist.to_string() == "* model\n.MODEL n NMOS (LEVEL=1\n+ VTO=0.5 KP=1e-4)\n"


def test_update_parameters_handles_both_conventions(netlist_factory):
    netlist = netlist_factory("minimal_ac")
    netlist.update_parameters({"R1": 82.0, "X1:width": 12.5})

    assert netlist.get_element("R1").value == "82.0"
    assert netlist.get_element("X1").params["width"] == "12.5"


def test_names_match_case_insensitively(netlist_factory):
    """SPICE names are case-insensitive, so a parameter named ``c1`` updates ``C1``."""
    netlist = netlist_factory("minimal_ac")
    netlist.update_parameters({"c1": "2p", "x1:WIDTH": "3"})
    netlist.update_parameters({"X1:width": "4"})

    assert netlist.get_element("c1").name == "C1"
    assert netlist.get_element("C1").value == "2p"
    assert netlist.get_element("X1").params == {"WIDTH": "4"}


def test_update_parameters_warns_but_continues_on_unknown_names(netlist_factory, caplog):
    netlist = netlist_factory("minimal_ac")
    with caplog.at_level(logging.WARNING, logger="cobra"):
        netlist.update_parameters({"DOES_NOT_EXIST": 1.0, "NOPE:key": 2.0, "R1": 33.0})

    assert "DOES_NOT_EXIST" in caplog.text
    assert "NOPE" in caplog.text
    assert netlist.get_element("R1").value == "33.0"


def test_a_copy_is_independent_of_its_template(minimal_ac):
    trial = minimal_ac.copy()
    trial.update_parameters({"R1": "1k"})

    assert minimal_ac.get_element("R1").value == "50"
    assert trial.get_element("R1").value == "1k"
    assert "R1 in n1 50" in minimal_ac.to_string()


def test_rendering_a_trial_never_reparses(minimal_ac, monkeypatch):
    """A trial is a copy of the template plus edits: no statement is split again."""
    calls = []
    original = XyceNetlistParser.split
    monkeypatch.setattr(
        XyceNetlistParser, "split", lambda self, *a, **k: calls.append(1) or original(self, *a, **k)
    )

    trial = minimal_ac.copy()
    trial.update_parameters({"R1": "1k", "C1": "2p", "X1:width": "3"})
    trial.to_string()

    assert calls == []


# ---------------------------------------------------------------------------
# Directive editing
# ---------------------------------------------------------------------------


def test_update_simulation_directive_positional_params(netlist_factory):
    netlist = netlist_factory("minimal_ac")
    netlist.update_simulation_directive(SimulationType.AC, {"start_freq": "2G", "stop_freq": "20G"})

    ac = next(d for d in netlist.simulation_directives if d.directive == ".AC")
    assert ac.positional == ["LIN", "101", "2G", "20G"]


def test_update_simulation_directive_expands_multi_token_values(netlist_factory):
    """HB frequencies arrive as one space-separated string and fill several slots."""
    netlist = netlist_factory("hb_two_tone")
    netlist.update_simulation_directive(SimulationType.HB, {"frequencies": "80E9 5E9"})

    hb = next(d for d in netlist.simulation_directives if d.directive == ".HB")
    assert hb.positional == ["80E9", "5E9"]


def test_update_simulation_directive_matches_keys_case_insensitively():
    netlist = _parse("* keys\n.AC LIN 10 1G 2G SWEEPOPT=1\n")
    netlist.update_simulation_directive(SimulationType.AC, {"sweepopt": "2"})

    (ac,) = netlist.simulation_directives
    assert ac.kv_params == {"SWEEPOPT": "2"}


def test_update_simulation_directive_needs_the_analysis(netlist_factory):
    with pytest.raises(KeyError, match="TRAN"):
        netlist_factory("minimal_ac").update_simulation_directive(SimulationType.TRAN, {})


def test_update_options_directive_keeps_other_options(netlist_factory):
    netlist = netlist_factory("hb_two_tone")
    netlist.update_options_directive("hbint", {"NUMFREQ": "9"})

    assert netlist.options_directives["hbint"] == {"numfreq": "9", "startupperiods": "2"}


# ---------------------------------------------------------------------------
# Qucs-S TSTONEFILE normalisation
# ---------------------------------------------------------------------------


def test_tstonefile_block_is_rewritten_to_a_subcircuit_call(netlist_factory):
    netlist = netlist_factory("tstonefile_qucs")

    component = netlist.components["XTrafo1"]
    assert component.model == "XTrafo1_subct"
    # Qucs-S emits "<signal> 0" per port; the literal ground nodes are dropped.
    assert component.nodes == ["n1", "n2", "n3"]
    # The original Touchstone path is carried through for the GUI, and survives edits.
    netlist.set_model("XTrafo1", "XTrafo1_subct")
    assert netlist.components["XTrafo1"].params["TSTONEFILE"] == "trafo.s6p"
    assert [i.file_path for i in netlist.includes] == ["XTrafo1.sp"]

    # Normalising again changes nothing.
    once = netlist.to_string()
    assert _parse(once).to_string() == once


def test_tstonefile_model_names_match_case_insensitively_and_lose_their_quotes():
    netlist = _parse(
        '* qucs\n.MODEL M1 LIN TSTONEFILE="tr.s2p"\nYLIN T1 a 0 b 0 m1\n.END\n'
    )
    assert netlist.components["XT1"].params["TSTONEFILE"] == "tr.s2p"


# ---------------------------------------------------------------------------
# Smoke test over the committed example netlists
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not EXAMPLE_NETLISTS, reason="examples/netlists is unavailable")
def test_example_netlists_parse():
    """Real-world Qucs-S/Xyce syntax must keep parsing, even though the example
    assets themselves (Touchstone, ONNX) are not in the repository.
    """
    parser = XyceNetlistParser()
    unparsed = [path.name for path in EXAMPLE_NETLISTS if not parser.parse_file(path).list_elements()]
    assert unparsed == []
