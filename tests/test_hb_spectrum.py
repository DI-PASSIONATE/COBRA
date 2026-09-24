"""Tests for harmonic-balance spectrum extraction in :mod:`cobra.spice_sim.hb_spectrum`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cobra.spice_sim.hb_spectrum import (
    available_power_dbm,
    classify_bins,
    covers_frequency,
    probe_nodes,
    spectrum,
    spice_float,
)

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("95E9", 95e9),
        ("130G", 130e9),
        ("7F", 7e-15),
        ("1MEG", 1e6),  # SPICE: MEG is mega ...
        ("1M", 1e-3),  # ... and M is milli
        ("130GHz", 130e9),  # trailing unit letters are ignored
    ],
)
def test_spice_float(token, expected):
    assert spice_float(token) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# The HB mixing grid
# ---------------------------------------------------------------------------


def test_covers_frequency_accepts_mixing_products_and_rejects_lines_off_the_grid():
    # The mixer testbench: .HB 95E9 10E9 with numfreq=4,40 resolves the 130 GHz RF
    # drive and the 35 GHz IF even though neither is a fundamental.
    assert covers_frequency([95e9, 10e9], [4, 40], 130e9, 130e9)
    assert covers_frequency([95e9, 10e9], [4, 40], 35e9, 35e9)
    assert not covers_frequency([95e9, 10e9], [4, 40], 37e9, 37e9)


def test_covers_frequency_spreads_a_single_order_over_every_tone():
    # numfreq=5 bounds both tones, so f1-6f2 = 35 GHz is out of reach but
    # f1-f2 = 85 GHz is not.
    assert not covers_frequency([95e9, 10e9], [5], 35e9, 35e9)
    assert covers_frequency([95e9, 10e9], [5], 85e9, 85e9)


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------


def _hb_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "FREQ": [0.0, 10e9, 20e9],
            "Re(V(OUT))": [0.0, 0.5, 0.05],
            "Im(V(OUT))": [0.0, 0.0, 0.0],
            "Re(I(VOUT))": [0.0, 0.01, 0.001],
            "Im(I(VOUT))": [0.0, 0.0, 0.0],
        }
    )


def test_available_power_dbm_matches_the_closed_form():
    # P = A^2 / (8 z0) = 1 / 400 = 2.5 mW -> 10*log10(2.5) dBm
    assert available_power_dbm(1.0, 50.0) == pytest.approx(10 * np.log10(2.5))
    assert np.isfinite(available_power_dbm(0.0, 50.0))


def test_a_probe_needs_both_voltage_and_current():
    assert probe_nodes(_hb_frame()) == ["OUT"]
    assert probe_nodes(pd.DataFrame({"FREQ": [1.0], "Re(V(IN))": [0.0]})) == []


def test_spectrum_power_uses_the_two_sided_convention():
    """Xyce bins hold amplitude/2, so S = 2 * |V * I|."""
    df = pd.DataFrame({"FREQ": [10e9], "Re(V(OUT))": [0.5], "Re(I(VOUT))": [0.01]})
    _, power = spectrum(df, "OUT", "power")
    _, gain = spectrum(df, "OUT", "gain", pin_dbm=3.0)

    assert power[0] == pytest.approx(10 * np.log10(2 * 0.5 * 0.01 / 1e-3))
    assert gain[0] == pytest.approx(power[0] - 3.0)


def test_frequency_range_filters_bins_and_a_single_frequency_snaps_to_the_nearest():
    assert spectrum(_hb_frame(), "OUT", frequency_range=(5e9, 25e9))[0].tolist() == [10e9, 20e9]
    assert spectrum(_hb_frame(), "OUT", frequency_range=(11e9, 11e9))[0].tolist() == [10e9]


def test_spectrum_errors_say_what_is_missing():
    with pytest.raises(ValueError, match="No HB frequency bins left"):
        spectrum(_hb_frame(), "OUT", frequency_range=(500e9, 600e9))
    with pytest.raises(KeyError, match="Available probe nodes"):
        spectrum(_hb_frame(), "NOPE")


# ---------------------------------------------------------------------------
# Bin classification
# ---------------------------------------------------------------------------


def test_single_tone_bins_are_harmonics():
    labels = classify_bins(np.array([0.0, 10e9, 20e9, 3.3e9]), [10e9])
    assert labels == ["DC", "H1", "H2", ""]


def test_two_tone_bins_are_mixing_products():
    freqs = np.array([95e9, 10e9, 105e9, 180e9])
    labels = classify_bins(freqs, [95e9, 10e9])
    assert labels == ["f1", "f2", "f1+f2", "2f1-f2"]
