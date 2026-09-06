"""Tests for the design-parameter catalogue and its factory functions."""

from __future__ import annotations

import pytest

from cobra.optimizers.design_goal_collection import (
    ALL_PARAMETERS,
    MAX_PORTS,
    find_parameter,
    get_available_parameters,
    make_gain_db,
    make_power_dbm,
    make_s_param_db,
    make_s_param_linear,
)
from cobra.spice_sim.simulation_type import SimulationType

# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["S11_dB", "S21_dB", "S11", "S21"])
def test_find_parameter_resolves_catalogue_entries(name):
    parameter = find_parameter(name)
    assert parameter is not None
    assert parameter.name == name


def test_find_parameter_returns_none_for_unknown_names():
    assert find_parameter("S99_dB") is None
    assert find_parameter("") is None


def test_catalogue_is_keyed_by_name():
    assert all(name == parameter.name for name, parameter in ALL_PARAMETERS.items())


# ---------------------------------------------------------------------------
# Port-count filtering
# ---------------------------------------------------------------------------


def test_s_parameters_exist_for_every_port_pair_up_to_max_ports():
    for i in (1, MAX_PORTS):
        for j in (1, MAX_PORTS):
            assert find_parameter(f"S{i}{j}_dB") is not None
            assert find_parameter(f"S{i}{j}") is not None


def test_beyond_max_ports_is_not_catalogued():
    assert find_parameter(f"S{MAX_PORTS + 1}{MAX_PORTS + 1}_dB") is None


def test_min_ports_is_the_larger_index():
    assert make_s_param_db(4, 2).min_ports == 4
    assert make_s_param_db(2, 7).min_ports == 7


def test_get_available_parameters_filters_on_port_count():
    names = {p.name for p in get_available_parameters(num_ports=2)}

    assert "S21_dB" in names
    assert "S31_dB" not in names


@pytest.mark.parametrize("num_ports", [0, -1])
def test_get_available_parameters_is_empty_without_ports(num_ports):
    assert get_available_parameters(num_ports) == []


def test_get_available_parameters_grows_monotonically_with_ports():
    previous: set[str] = set()
    for ports in range(1, MAX_PORTS + 1):
        current = {p.name for p in get_available_parameters(ports)}
        assert previous <= current
        previous = current


def test_get_available_parameters_can_filter_by_simulation_type():
    everything = get_available_parameters(2)
    ac_only = get_available_parameters(2, simulation_type=SimulationType.AC)
    hb_only = get_available_parameters(2, simulation_type=SimulationType.HB)

    assert ac_only == everything
    assert hb_only == []


# ---------------------------------------------------------------------------
# Factories — the names round-trip through configs and the GUI
# ---------------------------------------------------------------------------


def test_s_param_names_and_types():
    db = make_s_param_db(2, 1)
    linear = make_s_param_linear(2, 1)

    assert db.name == "S21_dB"
    assert linear.name == "S21"
    assert db.simulation_type is SimulationType.AC
    assert linear.simulation_type is SimulationType.AC


def test_power_dbm_name_format():
    """``config_runner._build_goal`` and the GUI both parse this exact shape."""
    parameter = make_power_dbm("OUT")

    assert parameter.name == "Power_dBm[OUT]"
    assert parameter.simulation_type is SimulationType.HB


def test_gain_db_name_format():
    """The GUI recovers port and node by regexing this name, so it must not drift."""
    parameter = make_gain_db("P2", sin_amplitude=0.2, z0=50.0, node="OUT")

    assert parameter.name == "Gain_dB[P2@OUT]"
    assert parameter.simulation_type is SimulationType.HB


def test_gain_db_records_the_input_power_in_its_description():
    parameter = make_gain_db("P2", sin_amplitude=0.2, z0=50.0, node="OUT")
    assert "P_in=" in parameter.description


def test_factories_produce_parameters_usable_as_dict_keys():
    """Equality is by name, so two independently built parameters collide."""
    assert make_power_dbm("OUT") == make_power_dbm("OUT")
    assert len({make_power_dbm("OUT"), make_power_dbm("OUT")}) == 1
    assert len({make_power_dbm("OUT"), make_power_dbm("IN")}) == 2
