"""Tests for the design-parameter catalogue and its factory functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cobra.optimizers.design_goal import DesignGoal
from cobra.optimizers.design_goal_collection import (
    ALL_PARAMETERS,
    MAX_PORTS,
    find_parameter,
    get_available_parameters,
    make_gain_db,
    make_isolation_db,
    make_power_dbm,
    make_s_param_db,
    make_s_param_linear,
)
from cobra.spice_sim.base_simulator import SimulationResult
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


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

# Every bin carries I = 1 A, so P_w = 2*V and the isolation between two bins
# reduces to 10*log10(V_target / V_spur). DC is deliberately the strongest line.
_ISO_FREQS = [0.0, 10e9, 35e9, 95e9, 130e9]
_ISO_VOLTS = [100.0, 0.01, 1.0, 0.001, 0.0001]


def _hb_result(freqs=None, volts=None) -> SimulationResult:
    """An HB result with a ``V(OUT)``/``I(VOUT)`` probe pair."""
    freqs = _ISO_FREQS if freqs is None else freqs
    volts = _ISO_VOLTS if volts is None else volts
    frame = pd.DataFrame(
        {
            "FREQ": freqs,
            "Re(V(OUT))": volts,
            "Im(V(OUT))": np.zeros(len(freqs)),
            "Re(I(VOUT))": np.ones(len(freqs)),
            "Im(I(VOUT))": np.zeros(len(freqs)),
        }
    )
    return SimulationResult(dataframes={"run.HB.FD.csv": frame})


def test_isolation_db_name_format():
    """``config_runner._build_goal`` and the GUI both parse this exact shape."""
    parameter = make_isolation_db("OUT")

    assert parameter.name == "Isolation_dB[OUT]"
    assert parameter.simulation_type is SimulationType.HB


def test_isolation_is_the_margin_to_the_strongest_spur():
    # Target 1.0 V against the strongest spur 0.01 V at 10 GHz -> 20 dB.
    value = make_isolation_db("OUT").formula(_hb_result(), "35GHz")

    assert value == pytest.approx(20.0)


def test_isolation_is_negative_when_a_spur_dominates():
    # Aiming at the 0.01 V line makes the 1.0 V line at 35 GHz the spur.
    value = make_isolation_db("OUT").formula(_hb_result(), "10GHz")

    assert value == pytest.approx(-20.0)


def test_isolation_excludes_dc():
    """The 100 V DC line would swamp every spur search and is not a mixing product."""
    value = make_isolation_db("OUT").formula(_hb_result(), "35GHz")

    # Including DC would give 10*log10(1.0/100) = -20 dB instead.
    assert value == pytest.approx(20.0)


def test_isolation_tracks_the_target_level():
    """The margin is relative: lifting target and spurs together leaves it unchanged."""
    lifted = [volt * 10.0 for volt in _ISO_VOLTS]

    assert make_isolation_db("OUT").formula(_hb_result(volts=lifted), "35GHz") == pytest.approx(
        make_isolation_db("OUT").formula(_hb_result(), "35GHz")
    )


def test_isolation_without_a_target_frequency_is_rejected():
    with pytest.raises(ValueError, match="needs a target frequency"):
        make_isolation_db("OUT").formula(_hb_result(), None)


def test_isolation_without_a_probe_pair_is_rejected():
    empty = SimulationResult(dataframes={"run.HB.FD.csv": pd.DataFrame({"FREQ": _ISO_FREQS})})

    with pytest.raises(KeyError, match=r"I\(VOUT\)"):
        make_isolation_db("OUT").formula(empty, "35GHz")


def test_isolation_needs_a_line_outside_the_target():
    single = _hb_result(freqs=[0.0, 35e9], volts=[100.0, 1.0])

    with pytest.raises(ValueError, match="no line outside"):
        make_isolation_db("OUT").formula(single, "35GHz")


@pytest.mark.parametrize(
    ("required", "satisfied"),
    [(15.0, True), (20.0, True), (30.0, False)],
)
def test_isolation_goal_penalty_sign(required, satisfied):
    """A satisfied goal reports a negative penalty, a violated one a positive."""
    goal = DesignGoal(make_isolation_db("OUT"), frequency_range="35GHz", min_value=required)

    penalty = goal.penalty(_hb_result())

    assert (penalty <= 0.0) is satisfied

