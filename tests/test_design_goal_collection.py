"""Tests for the design-parameter catalogue and its factory functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cobra.optimizers.design_goal import GoalInputs
from cobra.optimizers.design_goal_collection import (
    MAX_PORTS,
    find_parameter,
    get_available_parameters,
    make_gain_db,
    make_isolation_db,
    make_power_dbm,
)
from cobra.spice_sim.base_simulator import SimulationResult


def test_available_parameters_follow_the_port_count():
    names = {p.name for p in get_available_parameters(num_ports=2)}

    assert {"S21_dB", "S21", "K"} <= names
    assert "S31_dB" not in names
    assert get_available_parameters(0) == []
    assert find_parameter(f"S{MAX_PORTS + 1}{MAX_PORTS + 1}_dB") is None


def test_large_signal_parameter_names():
    """``config_runner._build_goal`` and the GUI both parse these exact shapes."""
    assert make_power_dbm("OUT").name == "Power_dBm[OUT]"
    assert make_gain_db("P2", sin_amplitude=0.2, z0=50.0, node="OUT").name == "Gain_dB[P2@OUT]"
    isolation = make_isolation_db("OUT")
    assert isolation.name == "Isolation_dB[OUT]"
    # The goal dialog hides the max value and only offers a single frequency.
    assert isolation.inputs == GoalInputs(
        max_value=False, frequency_required=True, single_frequency=True
    )


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


def test_isolation_is_the_margin_to_the_strongest_spur_excluding_dc():
    # Target 1.0 V against the strongest spur 0.01 V at 10 GHz -> 20 dB. Including
    # the 100 V DC line would give -20 dB instead.
    assert make_isolation_db("OUT").formula(_hb_result(), "35GHz") == pytest.approx(20.0)


def test_isolation_errors_explain_what_is_missing():
    with pytest.raises(ValueError, match="needs a target frequency"):
        make_isolation_db("OUT").formula(_hb_result(), None)
    with pytest.raises(ValueError, match="no line outside"):
        make_isolation_db("OUT").formula(_hb_result(freqs=[0.0, 35e9], volts=[100.0, 1.0]), "35GHz")
