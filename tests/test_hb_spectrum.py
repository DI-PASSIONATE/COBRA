"""Tests for harmonic-balance spectrum extraction in :mod:`cobra.spice_sim.hb_spectrum`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cobra.spice_sim.hb_spectrum import (
    QUANTITY_META,
    available_power_dbm,
    classify_bins,
    covers_frequency,
    find_dataframe,
    has_probe,
    is_fundamental,
    parse_fundamentals,
    parse_harmonic_orders,
    probe_nodes,
    spectrum,
    spice_float,
)

# ---------------------------------------------------------------------------
# SPICE number parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("130", 130.0),
        ("1.5", 1.5),
        ("95E9", 95e9),
        ("-2.5e-3", -2.5e-3),
        ("130G", 130e9),
        ("1T", 1e12),
        ("1.5K", 1.5e3),
        ("2.5N", 2.5e-9),
        ("4U", 4e-6),
        ("3P", 3e-12),
        ("7F", 7e-15),
        ("1MIL", 25.4e-6),
        ("  130G  ", 130e9),
        ("130g", 130e9),
    ],
)
def test_spice_float(token, expected):
    assert spice_float(token) == pytest.approx(expected)


def test_meg_is_matched_before_m():
    """SPICE semantics: M is milli, MEG is mega."""
    assert spice_float("1MEG") == pytest.approx(1e6)
    assert spice_float("1M") == pytest.approx(1e-3)


def test_trailing_unit_letters_are_ignored():
    assert spice_float("130GHz") == pytest.approx(130e9)


@pytest.mark.parametrize("token", ["", "   ", "abc", "G"])
def test_spice_float_rejects_malformed_tokens(token):
    with pytest.raises(ValueError, match=r"[Ee]mpty|Cannot interpret|could not convert"):
        spice_float(token)


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("95E9 10E9", [95e9, 10e9]),
        ("95G,10G", [95e9, 10e9]),
        ("  130G  ", [130e9]),
        ("", []),
        (None, []),
    ],
)
def test_parse_fundamentals(text, expected):
    assert parse_fundamentals(text) == pytest.approx(expected)


def test_parse_fundamentals_skips_unparsable_tokens():
    assert parse_fundamentals("95E9 junk 10E9") == pytest.approx([95e9, 10e9])


# ---------------------------------------------------------------------------
# Harmonic orders and the mixing grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4,40", [4, 40]),
        ("5", [5]),
        ("3 7", [3, 7]),
        ("4, junk, 40", [4, 40]),
        ("0,4", [4]),
        ("", []),
        (None, []),
    ],
)
def test_parse_harmonic_orders(text, expected):
    assert parse_harmonic_orders(text) == expected


@pytest.mark.parametrize("frequency", [95e9, 10e9, 35e9, 130e9, 225e9])
def test_covers_frequency_accepts_multi_tone_mixing_products(frequency):
    # The mixer testbench: .HB 95E9 10E9 with numfreq=4,40 resolves the 130 GHz RF
    # drive and the 35 GHz IF even though neither is a fundamental.
    assert covers_frequency([95e9, 10e9], [4, 40], frequency, frequency)


def test_covers_frequency_rejects_a_line_off_the_grid():
    assert not covers_frequency([95e9, 10e9], [4, 40], 37e9, 37e9)


def test_covers_frequency_spreads_a_single_order_over_every_tone():
    # numfreq=5 bounds both tones, so f1-6f2 = 35 GHz is out of reach but
    # f1-f2 = 85 GHz is not.
    assert not covers_frequency([95e9, 10e9], [5], 35e9, 35e9)
    assert covers_frequency([95e9, 10e9], [5], 85e9, 85e9)


def test_covers_frequency_accepts_a_range_holding_a_line():
    assert covers_frequency([95e9, 10e9], [4, 40], 30e9, 40e9)
    assert not covers_frequency([95e9, 10e9], [4, 40], 36e9, 39e9)


def test_covers_frequency_ignores_negative_lines():
    # 6f2-f1 is -35 GHz; only the non-negative half of the grid is solved for.
    assert not covers_frequency([95e9, 10e9], [4, 40], -35e9, -35e9)


@pytest.mark.parametrize(("tones", "orders"), [([], [4, 40]), ([95e9, 10e9], [])])
def test_covers_frequency_is_permissive_when_the_grid_is_unknown(tones, orders):
    assert covers_frequency(tones, orders, 37e9, 37e9)


# ---------------------------------------------------------------------------
# Available power
# ---------------------------------------------------------------------------


def test_available_power_dbm_matches_the_closed_form():
    # P = A^2 / (8 z0) = 1 / 400 = 2.5 mW -> 10*log10(2.5) dBm
    assert available_power_dbm(1.0, 50.0) == pytest.approx(10 * np.log10(2.5))


def test_available_power_of_one_milliwatt_is_zero_dbm():
    amplitude = np.sqrt(8 * 50.0 * 1e-3)
    assert available_power_dbm(amplitude, 50.0) == pytest.approx(0.0)


def test_available_power_is_floored_for_zero_amplitude():
    assert np.isfinite(available_power_dbm(0.0, 50.0))


# ---------------------------------------------------------------------------
# Probe discovery
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


def test_probe_nodes_requires_both_voltage_and_current():
    assert probe_nodes(_hb_frame()) == ["OUT"]


def test_probe_nodes_ignores_a_voltage_without_its_current():
    df = pd.DataFrame({"FREQ": [1.0], "Re(V(IN))": [0.0]})
    assert probe_nodes(df) == []


def test_has_probe_per_quantity():
    df = _hb_frame()

    assert has_probe(df, "OUT", "power")
    assert has_probe(df, "out", "power")  # case-insensitive
    assert has_probe(df, "OUT", "voltage")
    assert has_probe(df, "OUT", "current")
    assert not has_probe(df, "MISSING", "power")


def test_has_probe_requires_a_frequency_column():
    df = _hb_frame().drop(columns=["FREQ"])
    assert not has_probe(df, "OUT", "power")


def test_find_dataframe_picks_the_first_usable_frame():
    good = _hb_frame()
    bad = pd.DataFrame({"FREQ": [1.0]})

    assert find_dataframe({"a": bad, "b": good}, "OUT") is good
    assert find_dataframe({"a": bad}, "OUT") is None


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------


def test_spectrum_drops_negative_frequency_bins():
    df = pd.DataFrame(
        {
            "FREQ": [-10e9, 0.0, 10e9],
            "Re(V(OUT))": [0.5, 0.0, 0.5],
            "Re(I(VOUT))": [0.01, 0.0, 0.01],
        }
    )
    freqs, values = spectrum(df, "OUT")

    assert freqs.tolist() == [0.0, 10e9]
    assert len(values) == 2


def test_spectrum_power_uses_the_two_sided_convention():
    """Xyce bins hold amplitude/2, so S = 2 * |V * I|."""
    df = pd.DataFrame({"FREQ": [10e9], "Re(V(OUT))": [0.5], "Re(I(VOUT))": [0.01]})
    _, values = spectrum(df, "OUT", "power")

    expected = 10 * np.log10(2 * 0.5 * 0.01 / 1e-3)
    assert values[0] == pytest.approx(expected)


def test_gain_is_power_referred_to_pin():
    df = pd.DataFrame({"FREQ": [10e9], "Re(V(OUT))": [0.5], "Re(I(VOUT))": [0.01]})
    _, power = spectrum(df, "OUT", "power")
    _, gain = spectrum(df, "OUT", "gain", pin_dbm=3.0)

    assert gain[0] == pytest.approx(power[0] - 3.0)


def test_voltage_and_current_quantities():
    df = pd.DataFrame({"FREQ": [10e9], "Re(V(OUT))": [0.5], "Re(I(VOUT))": [0.01]})

    _, voltage = spectrum(df, "OUT", "voltage")
    _, current = spectrum(df, "OUT", "current")

    assert voltage[0] == pytest.approx(20 * np.log10(0.5))
    assert current[0] == pytest.approx(20 * np.log10(0.01 / 1e-3))


def test_missing_imaginary_column_is_treated_as_zero():
    with_imag = pd.DataFrame(
        {"FREQ": [10e9], "Re(V(OUT))": [0.5], "Im(V(OUT))": [0.0], "Re(I(VOUT))": [0.01]}
    )
    without_imag = pd.DataFrame({"FREQ": [10e9], "Re(V(OUT))": [0.5], "Re(I(VOUT))": [0.01]})

    assert spectrum(with_imag, "OUT")[1] == pytest.approx(spectrum(without_imag, "OUT")[1])


def test_frequency_range_filters_bins():
    freqs, _ = spectrum(_hb_frame(), "OUT", frequency_range=(5e9, 25e9))
    assert freqs.tolist() == [10e9, 20e9]


def test_equal_bounds_snap_to_the_nearest_bin():
    freqs, values = spectrum(_hb_frame(), "OUT", frequency_range=(11e9, 11e9))

    assert freqs.tolist() == [10e9]
    assert len(values) == 1


def test_empty_frequency_selection_raises():
    with pytest.raises(ValueError, match="No HB frequency bins left"):
        spectrum(_hb_frame(), "OUT", frequency_range=(500e9, 600e9))


def test_unknown_quantity_raises():
    with pytest.raises(ValueError, match="quantity must be one of"):
        spectrum(_hb_frame(), "OUT", "impedance")


def test_missing_frequency_column_raises():
    with pytest.raises(KeyError, match="No 'FREQ' column"):
        spectrum(pd.DataFrame({"Re(V(OUT))": [0.0]}), "OUT")


def test_missing_signal_reports_the_available_probes():
    with pytest.raises(KeyError, match="Available probe nodes"):
        spectrum(_hb_frame(), "NOPE")


def test_quantity_metadata_covers_every_supported_quantity():
    assert set(QUANTITY_META) == {"power", "gain", "voltage", "current"}


# ---------------------------------------------------------------------------
# Bin classification
# ---------------------------------------------------------------------------


def test_single_tone_bins_are_harmonics():
    labels = classify_bins(np.array([0.0, 10e9, 20e9, 30e9]), [10e9])
    assert labels == ["DC", "H1", "H2", "H3"]


def test_two_tone_bins_are_mixing_products():
    freqs = np.array([95e9, 10e9, 105e9, 180e9])
    labels = classify_bins(freqs, [95e9, 10e9])
    assert labels == ["f1", "f2", "f1+f2", "2f1-f2"]


def test_unmatched_bins_get_an_empty_label():
    labels = classify_bins(np.array([3.3e9]), [10e9])
    assert labels == [""]


def test_no_fundamentals_yields_empty_labels():
    assert classify_bins(np.array([1e9, 2e9]), []) == ["", ""]


def test_non_positive_fundamentals_are_ignored():
    assert classify_bins(np.array([1e9]), [0.0, -5.0]) == [""]


@pytest.mark.parametrize(("label", "expected"), [("H1", True), ("f1", True), ("f2", True)])
def test_is_fundamental_accepts_first_order_labels(label, expected):
    assert is_fundamental(label) is expected


@pytest.mark.parametrize("label", ["H2", "f1+f2", "2f1-f2", "DC", ""])
def test_is_fundamental_rejects_everything_else(label):
    assert is_fundamental(label) is False
