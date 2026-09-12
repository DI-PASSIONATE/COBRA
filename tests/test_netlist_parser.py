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


def test_save_roundtrip(editable_netlist, tmp_path: Path):
    parser = editable_netlist("minimal_ac")
    out = tmp_path / "saved.cir"
    parser.save(out)
    assert out.read_text(encoding="utf-8") == netlist_path("minimal_ac").read_text(encoding="utf-8")


def test_edit_touches_only_the_edited_line(editable_netlist):
    parser = editable_netlist("minimal_ac")
    before = parser.lines
    parser.set_value("R1", "75")
    after = parser.lines

    assert len(before) == len(after)
    changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert len(changed) == 1
    assert after[changed[0]].strip() == "R1 in n1 75"


def test_lines_property_returns_a_copy(minimal_ac):
    lines = minimal_ac.lines
    lines[0] = "* mutated\n"
    assert minimal_ac.lines[0] != "* mutated\n"


def test_inline_comment_and_newline_survive_an_edit(tmp_path: Path):
    source = tmp_path / "commented.cir"
    source.write_text("* header\nR1 a b 50 ; keep me\n.END\n", encoding="utf-8")
    parser = XyceNetlistParser().from_file(source)

    parser.set_value("R1", "75")

    assert parser.lines[1] == "R1 a b 75 ; keep me\n"


def test_final_line_without_newline_stays_without_newline(tmp_path: Path):
    source = tmp_path / "no_trailing_newline.cir"
    source.write_text("* header\nR1 a b 50", encoding="utf-8")
    parser = XyceNetlistParser().from_file(source)

    parser.set_value("R1", "75")

    assert parser.to_string() == "* header\nR1 a b 75"


# ---------------------------------------------------------------------------
# Reader properties
# ---------------------------------------------------------------------------


def test_minimal_ac_structure(minimal_ac):
    assert minimal_ac.simulation_type is SimulationType.AC
    assert minimal_ac.num_ports == 2
    assert set(minimal_ac.components) == {"X1"}
    assert minimal_ac.components["X1"].model == "surrogate_model"
    assert minimal_ac.components["X1"].nodes == ["in", "out"]


def test_inline_subcircuit_instances_are_not_components(minimal_ac):
    """XF1 references a .SUBCKT defined in this file, so it needs no surrogate."""
    assert minimal_ac.inline_subckt_names == {"lowpass"}
    assert "XF1" in minimal_ac._elements
    assert "XF1" not in minimal_ac.components


def test_devices_inside_subcircuits_are_not_top_level(parser_factory):
    """.SUBCKT bodies must be skipped, including nested ones."""
    parser = parser_factory("subckt_and_includes")
    names = {e.name for e in parser.list_elements()}

    assert {"Q1", "D1", "M2", "XB1", "X2"} <= names
    assert "MB1" not in names  # inside .SUBCKT buffer
    assert "RIN" not in names  # inside the nested .SUBCKT inner
    assert "XI9" not in names  # inside .SUBCKT buffer, after the nested .ENDS
    assert parser.inline_subckt_names == {"buffer", "inner"}


def test_element_positional_layouts(parser_factory):
    parser = parser_factory("subckt_and_includes")

    q1 = parser.get_element("Q1")
    assert q1.nodes == ["c", "b", "e"]
    assert q1.model == "npn_model"

    d1 = parser.get_element("D1")
    assert d1.nodes == ["anode", "cathode"]
    assert d1.model == "diode_model"

    m2 = parser.get_element("M2")
    assert m2.nodes == ["d", "g", "s", "bulk"]
    assert m2.model == "nch"
    assert m2.params == {"W": "2u", "L": "0.1u"}


def test_model_directive_is_parsed_as_an_element(parser_factory):
    parser = parser_factory("subckt_and_includes")
    model = parser.get_element("nch")

    assert model.etype == "MODEL"
    assert model.model == "NMOS"
    assert model.params == {"LEVEL": "1", "VTO": "0.7"}


def test_includes_and_libraries(parser_factory):
    parser = parser_factory("subckt_and_includes")

    assert [i.file_path for i in parser.includes] == ["extra_models.sp"]
    assert [(lib.file_path, lib.entry) for lib in parser.libraries] == [
        ("cornerHBT.lib", "hbt_typ")
    ]


def test_simulation_and_print_directives(minimal_ac):
    ac, lin = minimal_ac.simulation_directives
    assert ac.directive == ".AC"
    assert ac.positional == ["LIN", "101", "1G", "10G"]
    assert lin.directive == ".LIN"
    assert lin.kv_params == {"format": "touchstone", "sparcalc": "1"}

    (printed,) = minimal_ac.print_directives
    assert printed.analysis == "ac"
    assert printed.signals == ["v(out)"]
    assert printed.kv_params == {"format": "csv"}


def test_first_simulation_directive_wins(minimal_ac):
    """.LIN post-processes .AC, so it must not override the detected type."""
    assert {d.directive for d in minimal_ac.simulation_directives} == {".AC", ".LIN"}
    assert minimal_ac.simulation_type is SimulationType.AC


def test_port_sources_only_include_ports_with_a_source(minimal_ac):
    """P2 carries no AC/SIN declaration, so it is omitted entirely."""
    assert set(minimal_ac.port_sources) == {"P1"}
    assert minimal_ac.port_sources["P1"] == {"z0": 50.0, "ac_amplitude": 1.0}


def test_sin_source_amplitude_and_frequency_follow_the_offset(parser_factory):
    """``SIN <offset> <amplitude> <freq>`` — amplitude and frequency are the 2nd and 3rd arguments."""
    parser = parser_factory("hb_two_tone")
    assert parser.port_sources["P1"] == {"z0": 50.0, "sin_amplitude": 0.2, "sin_frequency": 95e9}
    assert parser.port_sources["P2"] == {"z0": 50.0, "sin_amplitude": 0.5, "sin_frequency": 10e9}


def test_probe_nodes_requires_matching_voltage_and_current(parser_factory):
    parser = parser_factory("hb_two_tone")
    assert parser.probe_nodes == ["OUT"]


def test_probe_nodes_are_read_from_a_print_tran_line(parser_factory):
    parser = parser_factory("tran_single_tone")
    assert parser.probe_nodes == ["OUT"]


def test_options_directives_strip_the_internal_line_index(parser_factory):
    parser = parser_factory("hb_two_tone")
    assert parser.options_directives == {
        "hbint": {"numfreq": "5", "startupperiods": "2"}
    }


def test_list_elements_filters_by_type_and_keeps_line_order(minimal_ac):
    names = [e.name for e in minimal_ac.list_elements()]
    assert names == ["P1", "P2", "R1", "C1", "L1", "V1", "X1", "XF1"]
    assert [e.name for e in minimal_ac.list_elements(["p"])] == ["P1", "P2"]


def test_get_element_raises_for_unknown_name(minimal_ac):
    with pytest.raises(KeyError, match="NOPE"):
        minimal_ac.get_element("NOPE")


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [("R1", "R1 in n1 75"), ("C1", "C1 n1 0 75"), ("L1", "L1 n1 out 75")],
)
def test_set_value_on_rcl_replaces_the_fourth_token(editable_netlist, name, expected):
    parser = editable_netlist("minimal_ac")
    parser.set_value(name, "75")
    assert parser.get_element(name).value == "75"
    assert parser.lines[parser.get_element(name).line_index].strip() == expected


def test_set_value_on_voltage_source_truncates_the_waveform(editable_netlist):
    """V/I keep only three positional tokens, so any old waveform is dropped."""
    parser = editable_netlist("minimal_ac")
    parser.set_value("V1", "2.5")
    assert parser.lines[parser.get_element("V1").line_index].strip() == "V1 vdd 0 2.5"


def test_set_value_rejects_unsupported_element_types(editable_netlist):
    parser = editable_netlist("minimal_ac")
    with pytest.raises(ValueError, match="set_value is for R/C/L"):
        parser.set_value("X1", "5")


def test_set_value_rejects_a_short_line(tmp_path: Path):
    source = tmp_path / "short.cir"
    source.write_text("R1 a b\n.END\n", encoding="utf-8")
    parser = XyceNetlistParser().from_file(source)
    with pytest.raises(ValueError, match="too short"):
        parser.set_value("R1", "50")


def test_set_model_on_subcircuit_instance(editable_netlist):
    parser = editable_netlist("minimal_ac")
    parser.set_model("X1", "new_surrogate")
    assert parser.components["X1"].model == "new_surrogate"


def test_set_model_keeps_key_value_params_in_place(editable_netlist):
    """The subcircuit name is the last token *before* the first key=value."""
    parser = editable_netlist("subckt_and_includes")
    parser.set_model("X2", "replacement_sub")

    element = parser.get_element("X2")
    assert element.model == "replacement_sub"
    assert element.params == {"param1": "5"}
    assert parser.lines[element.line_index].strip() == "X2 p1 p2 replacement_sub param1=5"


@pytest.mark.parametrize("name", ["Q1", "D1", "M2"])
def test_set_model_on_devices(editable_netlist, name):
    parser = editable_netlist("subckt_and_includes")
    parser.set_model(name, "swapped")
    assert parser.get_element(name).model == "swapped"


def test_set_model_rejects_unsupported_type(editable_netlist):
    parser = editable_netlist("minimal_ac")
    with pytest.raises(ValueError, match="set_model not supported"):
        parser.set_model("R1", "whatever")


def test_set_param_updates_an_existing_key_case_insensitively(editable_netlist):
    parser = editable_netlist("subckt_and_includes")
    parser.set_param("M2", "w", "9u")

    element = parser.get_element("M2")
    assert element.params == {"w": "9u", "L": "0.1u"}


def test_set_param_appends_a_missing_key(editable_netlist):
    parser = editable_netlist("subckt_and_includes")
    parser.set_param("M2", "AD", "1p")
    assert parser.get_element("M2").params["AD"] == "1p"


def test_set_param_on_model_line_skips_the_positional_tokens(editable_netlist):
    """.MODEL has two positional tokens, so parameters start at index 3."""
    parser = editable_netlist("subckt_and_includes")
    parser.set_param("nch", "VTO", "0.5")

    element = parser.get_element("nch")
    assert element.model == "NMOS"
    assert element.params == {"LEVEL": "1", "VTO": "0.5"}


def test_update_parameters_dispatches_on_the_colon_convention(editable_netlist):
    parser = editable_netlist("subckt_and_includes")
    parser.update_parameters({"M2:W": 4.0})
    assert parser.get_element("M2").params["W"] == "4.0"


def test_update_parameters_handles_both_conventions_at_once(editable_netlist):
    parser = editable_netlist("minimal_ac")
    parser.update_parameters({"R1": 82.0, "X1:width": 12.5})

    assert parser.get_element("R1").value == "82.0"
    assert parser.get_element("X1").params["width"] == "12.5"


def test_update_parameters_warns_but_continues_on_unknown_names(editable_netlist, caplog):
    parser = editable_netlist("minimal_ac")
    with caplog.at_level(logging.WARNING, logger="cobra"):
        parser.update_parameters({"DOES_NOT_EXIST": 1.0, "NOPE:key": 2.0, "R1": 33.0})

    warnings = caplog.text
    assert "DOES_NOT_EXIST" in warnings
    assert "NOPE" in warnings
    assert parser.get_element("R1").value == "33.0"


# ---------------------------------------------------------------------------
# Directive editing
# ---------------------------------------------------------------------------


def test_update_simulation_directive_positional_params(editable_netlist):
    parser = editable_netlist("minimal_ac")
    parser.update_simulation_directive(".AC", {"start_freq": "2G", "stop_freq": "20G"})

    ac = next(d for d in parser.simulation_directives if d.directive == ".AC")
    assert ac.positional == ["LIN", "101", "2G", "20G"]


def test_update_simulation_directive_key_value_params(editable_netlist):
    parser = editable_netlist("minimal_ac")
    parser.update_simulation_directive(".LIN", {"format": "csv"})

    lin = next(d for d in parser.simulation_directives if d.directive == ".LIN")
    assert lin.kv_params["format"] == "csv"


def test_update_simulation_directive_expands_multi_token_values(editable_netlist):
    """HB frequencies arrive as one space-separated string and fill several slots."""
    parser = editable_netlist("hb_two_tone")
    parser.update_simulation_directive(".HB", {"frequencies": "80E9 5E9"})

    hb = next(d for d in parser.simulation_directives if d.directive == ".HB")
    assert hb.positional == ["80E9", "5E9"]


def test_update_simulation_directive_raises_for_missing_directive(editable_netlist):
    parser = editable_netlist("minimal_ac")
    with pytest.raises(KeyError, match=r"\.TRAN"):
        parser.update_simulation_directive(".TRAN", {"initial_step": "1n"})


def test_update_options_directive(editable_netlist):
    parser = editable_netlist("hb_two_tone")
    parser.update_options_directive("hbint", {"numfreq": "9"})

    assert parser.options_directives["hbint"]["numfreq"] == "9"
    assert parser.options_directives["hbint"]["startupperiods"] == "2"


def test_update_options_directive_raises_for_missing_category(editable_netlist):
    parser = editable_netlist("hb_two_tone")
    with pytest.raises(KeyError, match="nosuch"):
        parser.update_options_directive("nosuch", {"a": "1"})


# ---------------------------------------------------------------------------
# Qucs-S TSTONEFILE normalisation
# ---------------------------------------------------------------------------


def test_tstonefile_block_is_rewritten_to_a_subcircuit_call(parser_factory):
    parser = parser_factory("tstonefile_qucs")

    assert "XTrafo1" in parser.components
    component = parser.components["XTrafo1"]
    assert component.model == "XTrafo1_subct"
    # Qucs-S emits "<signal> 0" per port; the literal ground nodes are dropped.
    assert component.nodes == ["n1", "n2", "n3"]
    # The original Touchstone path is carried through for the GUI.
    assert component.params["TSTONEFILE"] == "trafo.s6p"


def test_tstonefile_model_line_becomes_an_include(parser_factory):
    parser = parser_factory("tstonefile_qucs")
    assert [i.file_path for i in parser.includes] == ["XTrafo1.sp"]


def test_normalisation_is_idempotent(parser_factory):
    parser = parser_factory("tstonefile_qucs")
    once = parser.to_string()
    parser.parse_netlist()
    assert parser.to_string() == once


# ---------------------------------------------------------------------------
# Smoke test over the committed example netlists
# ---------------------------------------------------------------------------

EXAMPLE_NETLISTS = sorted(EXAMPLE_NETLIST_DIR.rglob("*.cir"))


@pytest.mark.skipif(not EXAMPLE_NETLISTS, reason="examples/netlists is unavailable")
@pytest.mark.parametrize("path", EXAMPLE_NETLISTS, ids=lambda p: p.stem)
def test_example_netlists_parse(path: Path):
    """Real-world Qucs-S/Xyce syntax must keep parsing, even though the example
    assets themselves (Touchstone, ONNX) are not in the repository.
    """
    parser = XyceNetlistParser().from_file(path)
    assert parser.list_elements(), f"no elements parsed from {path.name}"
