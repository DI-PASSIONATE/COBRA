"""Tests for :class:`~cobra.spice_sim.simulation_type.SimulationType`."""

from __future__ import annotations

import pytest

from cobra.spice_sim.simulation_type import SimulationType, SimulationTypeMetadata


@pytest.mark.parametrize(
    ("directive", "expected"),
    [
        (".AC", SimulationType.AC),
        (".HB", SimulationType.HB),
        (".TRAN", SimulationType.TRAN),
        (".DC", SimulationType.DC),
    ],
)
def test_from_directive_maps_each_analysis(directive, expected):
    assert SimulationType.from_directive(directive) is expected


def test_lin_is_treated_as_ac():
    """.LIN post-processes .AC for Touchstone output; it is not its own analysis."""
    assert SimulationType.from_directive(".LIN") is SimulationType.AC


@pytest.mark.parametrize("directive", [".ac", ".Ac", "  .TRAN  ", ".hb"])
def test_from_directive_is_case_and_whitespace_insensitive(directive):
    assert SimulationType.from_directive(directive) is not SimulationType.UNKNOWN


@pytest.mark.parametrize("directive", [".OP", "AC", "", "nonsense"])
def test_unrecognised_directives_map_to_unknown(directive):
    assert SimulationType.from_directive(directive) is SimulationType.UNKNOWN


def test_display_name_is_the_enum_value():
    assert SimulationType.AC.display_name == ".AC"
    assert SimulationType.UNKNOWN.display_name == "unknown"


def test_available_parameters_are_restricted_to_the_simulation_type():
    names = SimulationType.AC.available_parameters(num_ports=2)

    assert "S21_dB" in names
    assert all(not n.startswith("Power_dBm") for n in names)


def test_available_parameters_respect_the_port_count():
    two_port = set(SimulationType.AC.available_parameters(num_ports=2))
    four_port = set(SimulationType.AC.available_parameters(num_ports=4))

    assert "S43_dB" not in two_port
    assert "S43_dB" in four_port
    assert two_port < four_port


def test_for_parameter_resolves_through_the_catalogue():
    assert SimulationType.for_parameter("S21_dB") is SimulationType.AC


def test_for_parameter_returns_unknown_for_a_missing_name():
    assert SimulationType.for_parameter("not_a_parameter") is SimulationType.UNKNOWN


def test_static_catalogue_holds_only_ac_parameters():
    """Every catalogued parameter is AC-typed.

    HB goals (``Power_dBm[...]``, ``Gain_dB[...]``) are built on demand by
    ``make_power_dbm``/``make_gain_db`` because they depend on a node name, so
    they never appear in the static catalogue.
    """
    everything = SimulationType.all_available_parameters(num_ports=2)
    assert everything == SimulationType.AC.available_parameters(num_ports=2)
    assert SimulationType.HB.available_parameters(num_ports=2) == []


def test_metadata_defaults_are_empty():
    metadata = SimulationTypeMetadata()

    assert metadata.positional_param_names == []
    assert metadata.positional_param_descriptions == {}
    assert metadata.positional_param_defaults == {}
    assert metadata.options_category is None


def test_metadata_instances_do_not_share_mutable_defaults():
    first, second = SimulationTypeMetadata(), SimulationTypeMetadata()
    first.positional_param_names.append("points")
    assert second.positional_param_names == []
