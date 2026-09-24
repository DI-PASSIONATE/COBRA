"""Tests for :class:`~cobra.spice_sim.netlist_parsers.xyce_netlist_parser.XyceNetlistParser`."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
from cobra.spice_sim.simulation_type import SimulationType
from tests.conftest import EXAMPLE_NETLIST_DIR, netlist_path

# ---------------------------------------------------------------------------
# Round-trip fidelity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["minimal_ac", "hb_two_tone", "subckt_and_includes", "simulatable_ac"]
)
def test_roundtrip_is_byte_identical(parser_factory, name):
    """Parsing then serialising an untouched netlist must not alter a single byte."""
    parser = parser_factory(name)
    assert parser.to_string() == netlist_path(name).read_text(encoding="utf-8")


def test_edit_touches_only_the_edited_line(editable_netlist):
    parser = editable_netlist("minimal_ac")
    before = parser.lines
    parser.set_value("R1", "75")
    after = parser.lines

    assert len(before) == len(after)
    changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert len(changed) == 1
    assert after[changed[0]].strip() == "R1 in n1 75"


def test_inline_comment_and_missing_final_newline_survive_an_edit(tmp_path: Path):
    source = tmp_path / "commented.cir"
    source.write_text("* header\nR1 a b 50 ; keep me\nR2 b 0 50", encoding="utf-8")
    parser = XyceNetlistParser().from_file(source)

    parser.set_value("R1", "75")
    parser.set_value("R2", "75")

    assert parser.to_string() == "* header\nR1 a b 75 ; keep me\nR2 b 0 75"


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


def test_devices_inside_subcircuits_are_not_top_level(parser_factory):
    """.SUBCKT bodies must be skipped, including nested ones."""
    parser = parser_factory("subckt_and_includes")
    names = {e.name for e in parser.list_elements()}

    assert {"Q1", "D1", "M2", "XB1", "X2"} <= names
    assert "MB1" not in names  # inside .SUBCKT buffer
    assert "RIN" not in names  # inside the nested .SUBCKT inner
    assert "XI9" not in names  # inside .SUBCKT buffer, after the nested .ENDS
    assert parser.inline_subckt_names == {"buffer", "inner"}


def test_device_and_model_lines(parser_factory):
    parser = parser_factory("subckt_and_includes")

    m2 = parser.get_element("M2")
    assert m2.nodes == ["d", "g", "s", "bulk"]
    assert m2.model == "nch"
    assert m2.params == {"W": "2u", "L": "0.1u"}

    model = parser.get_element("nch")
    assert model.etype == "MODEL"
    assert model.params == {"LEVEL": "1", "VTO": "0.7"}


def test_includes_and_libraries(parser_factory):
    parser = parser_factory("subckt_and_includes")

    assert [i.file_path for i in parser.includes] == ["extra_models.sp"]
    assert [(lib.file_path, lib.entry) for lib in parser.libraries] == [
        ("cornerHBT.lib", "hbt_typ")
    ]


def test_simulation_and_print_directives(minimal_ac):
    ac, lin = minimal_ac.simulation_directives
    assert ac.positional == ["LIN", "101", "1G", "10G"]
    assert lin.kv_params == {"format": "touchstone", "sparcalc": "1"}
    # .LIN post-processes .AC, so it must not override the detected type.
    assert minimal_ac.simulation_type is SimulationType.AC

    (printed,) = minimal_ac.print_directives
    assert printed.signals == ["v(out)"]


def test_port_sources(parser_factory, minimal_ac):
    """``SIN <offset> <amplitude> <freq>``: amplitude and frequency are the 2nd and 3rd
    arguments. A port without a source is left out.
    """
    assert set(minimal_ac.port_sources) == {"P1"}
    parser = parser_factory("hb_two_tone")
    assert parser.port_sources["P1"] == {"z0": 50.0, "sin_amplitude": 0.2, "sin_frequency": 95e9}


def test_probe_nodes_need_matching_voltage_and_current(parser_factory):
    assert parser_factory("hb_two_tone").probe_nodes == ["OUT"]
    assert parser_factory("tran_single_tone").probe_nodes == ["OUT"]


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------


def test_mutators_reject_unsupported_elements(editable_netlist, tmp_path: Path):
    parser = editable_netlist("minimal_ac")
    with pytest.raises(ValueError, match="set_value is for R/C/L"):
        parser.set_value("X1", "5")
    with pytest.raises(ValueError, match="set_model not supported"):
        parser.set_model("R1", "whatever")

    source = tmp_path / "short.cir"
    source.write_text("R1 a b\n.END\n", encoding="utf-8")
    with pytest.raises(ValueError, match="too short"):
        XyceNetlistParser().from_file(source).set_value("R1", "50")


def test_set_model_keeps_key_value_params_in_place(editable_netlist):
    """The subcircuit name is the last token *before* the first key=value."""
    parser = editable_netlist("subckt_and_includes")
    parser.set_model("X2", "replacement_sub")

    element = parser.get_element("X2")
    assert element.params == {"param1": "5"}
    assert parser.lines[element.line_index].strip() == "X2 p1 p2 replacement_sub param1=5"


def test_set_param_updates_case_insensitively_and_appends_missing_keys(editable_netlist):
    parser = editable_netlist("subckt_and_includes")
    parser.set_param("M2", "w", "9u")
    parser.set_param("M2", "AD", "1p")

    assert parser.get_element("M2").params == {"w": "9u", "L": "0.1u", "AD": "1p"}


def test_update_parameters_handles_both_conventions(editable_netlist):
    parser = editable_netlist("minimal_ac")
    parser.update_parameters({"R1": 82.0, "X1:width": 12.5})

    assert parser.get_element("R1").value == "82.0"
    assert parser.get_element("X1").params["width"] == "12.5"


def test_update_parameters_warns_but_continues_on_unknown_names(editable_netlist, caplog):
    parser = editable_netlist("minimal_ac")
    with caplog.at_level(logging.WARNING, logger="cobra"):
        parser.update_parameters({"DOES_NOT_EXIST": 1.0, "NOPE:key": 2.0, "R1": 33.0})

    assert "DOES_NOT_EXIST" in caplog.text
    assert "NOPE" in caplog.text
    assert parser.get_element("R1").value == "33.0"


# ---------------------------------------------------------------------------
# Directive editing
# ---------------------------------------------------------------------------


def test_update_simulation_directive_positional_params(editable_netlist):
    parser = editable_netlist("minimal_ac")
    parser.update_simulation_directive(".AC", {"start_freq": "2G", "stop_freq": "20G"})

    ac = next(d for d in parser.simulation_directives if d.directive == ".AC")
    assert ac.positional == ["LIN", "101", "2G", "20G"]


def test_update_simulation_directive_expands_multi_token_values(editable_netlist):
    """HB frequencies arrive as one space-separated string and fill several slots."""
    parser = editable_netlist("hb_two_tone")
    parser.update_simulation_directive(".HB", {"frequencies": "80E9 5E9"})

    hb = next(d for d in parser.simulation_directives if d.directive == ".HB")
    assert hb.positional == ["80E9", "5E9"]


def test_update_options_directive_keeps_other_options(editable_netlist):
    parser = editable_netlist("hb_two_tone")
    parser.update_options_directive("hbint", {"numfreq": "9"})

    assert parser.options_directives["hbint"] == {"numfreq": "9", "startupperiods": "2"}


# ---------------------------------------------------------------------------
# Qucs-S TSTONEFILE normalisation
# ---------------------------------------------------------------------------


def test_tstonefile_block_is_rewritten_to_a_subcircuit_call(parser_factory):
    parser = parser_factory("tstonefile_qucs")

    component = parser.components["XTrafo1"]
    assert component.model == "XTrafo1_subct"
    # Qucs-S emits "<signal> 0" per port; the literal ground nodes are dropped.
    assert component.nodes == ["n1", "n2", "n3"]
    # The original Touchstone path is carried through for the GUI.
    assert component.params["TSTONEFILE"] == "trafo.s6p"
    assert [i.file_path for i in parser.includes] == ["XTrafo1.sp"]

    # Normalising again changes nothing.
    once = parser.to_string()
    parser.parse_netlist()
    assert parser.to_string() == once


# ---------------------------------------------------------------------------
# Smoke test over the committed example netlists
# ---------------------------------------------------------------------------

EXAMPLE_NETLISTS = sorted(EXAMPLE_NETLIST_DIR.rglob("*.cir"))


@pytest.mark.skipif(not EXAMPLE_NETLISTS, reason="examples/netlists is unavailable")
def test_example_netlists_parse():
    """Real-world Qucs-S/Xyce syntax must keep parsing, even though the example
    assets themselves (Touchstone, ONNX) are not in the repository.
    """
    unparsed = [
        path.name for path in EXAMPLE_NETLISTS if not XyceNetlistParser().from_file(path).list_elements()
    ]
    assert unparsed == []
