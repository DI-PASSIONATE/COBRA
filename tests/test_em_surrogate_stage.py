"""Tests for :meth:`EMSurrogateStage.s_param_dict_to_network`: turning a surrogate's
named ``S<i><j>_real``/``_imag`` outputs into a network, for both output layouts
ORCA exports (the full matrix and the upper triangle only), and for the inference
frequency grid.
"""

from __future__ import annotations

import numpy as np
import pytest
import skrf as rf
from skrf.network import s2y

from cobra.stages.em_surrogate_stage import (
    LOW_FREQUENCIES,
    EMSurrogateStage,
    clean_dc_point,
    inference_frequencies,
)

FREQUENCIES = np.array([1e9, 2e9, 3e9])


def _reciprocal_s_matrix(n_ports: int) -> np.ndarray:
    """A distinct-valued symmetric S-matrix per frequency, shape ``(freq, n, n)``."""
    rng = np.random.default_rng(n_ports)
    s = rng.normal(size=(len(FREQUENCIES), n_ports, n_ports)) + 1j * rng.normal(
        size=(len(FREQUENCIES), n_ports, n_ports)
    )
    return (s + s.transpose(0, 2, 1)) / 2


def _outputs(s: np.ndarray, upper_triangle_only: bool) -> dict[str, np.ndarray]:
    """The model outputs for *s*, shaped ``(freq, 1)`` as ONNX Runtime returns them."""
    n_ports = s.shape[1]
    outputs = {}
    for i in range(n_ports):
        for j in range(i if upper_triangle_only else 0, n_ports):
            outputs[f"S{i + 1}{j + 1}_real"] = s[:, i, j].real.reshape(-1, 1)
            outputs[f"S{i + 1}{j + 1}_imag"] = s[:, i, j].imag.reshape(-1, 1)
    return outputs


@pytest.mark.parametrize("upper_triangle_only", [False, True], ids=["full", "upper_triangle"])
@pytest.mark.parametrize("n_ports", [2, 3, 6, 8])
def test_network_matches_outputs(n_ports: int, upper_triangle_only: bool) -> None:
    s = _reciprocal_s_matrix(n_ports)
    outputs = _outputs(s, upper_triangle_only)

    n, ntwk, merged = EMSurrogateStage([]).s_param_dict_to_network(outputs, FREQUENCIES)

    assert n == n_ports
    assert ntwk.nports == n_ports
    np.testing.assert_allclose(ntwk.s, s, rtol=1e-6)
    np.testing.assert_allclose(ntwk.f, FREQUENCIES)
    assert len(merged) == n_ports**2


def test_output_count_does_not_decide_port_count() -> None:
    """A full 6-port and an 8-port triangle both have 72 outputs."""
    full_six = _outputs(_reciprocal_s_matrix(6), upper_triangle_only=False)
    triangle_eight = _outputs(_reciprocal_s_matrix(8), upper_triangle_only=True)
    assert len(full_six) == len(triangle_eight)

    stage = EMSurrogateStage([])
    assert stage.s_param_dict_to_network(full_six, FREQUENCIES)[0] == 6
    assert stage.s_param_dict_to_network(triangle_eight, FREQUENCIES)[0] == 8


def test_grid_from_dc_adds_the_low_frequencies_below_the_first_step() -> None:
    grid = inference_frequencies((0.0, 500e9))
    assert grid[0] == 0.0
    np.testing.assert_array_equal(grid[1 : 1 + len(LOW_FREQUENCIES)], LOW_FREQUENCIES)
    assert grid[1 + len(LOW_FREQUENCIES)] == 1e9
    assert grid[-1] == 500e9
    assert len(grid) == 501 + len(LOW_FREQUENCIES)
    assert np.all(np.diff(grid) > 0)


def test_grid_above_the_low_frequencies_is_unchanged() -> None:
    np.testing.assert_array_equal(inference_frequencies((50e9, 60e9)), np.arange(50e9, 60.5e9, 1e9))


def test_grid_keeps_the_low_frequencies_inside_the_band() -> None:
    np.testing.assert_array_equal(inference_frequencies((0.0, 0.3e9)), [0.0, 10e6, 20e6, 100e6, 200e6, 300e6])


def _transformer_with_dc_errors() -> rf.Network:
    """A 4-port of two 1 Ohm windings (ports 1-2 and 3-4) whose 0 Hz point leaks 1e-4 S to
    ground at every port and couples the windings by 0.5 S, as a surrogate predicts it.
    """
    y_dc = np.zeros((4, 4))
    for a, b in ((0, 1), (2, 3)):
        y_dc[np.ix_([a, b], [a, b])] = [[1.0, -1.0], [-1.0, 1.0]]
    y_dc += np.eye(4) * 1e-4
    y_dc[0, 2] = y_dc[2, 0] = -0.5
    y_dc[0, 0] += 0.5
    y_dc[2, 2] += 0.5
    s_dc = np.linalg.solve(np.eye(4) + 50 * y_dc, np.eye(4) - 50 * y_dc)  # Y commutes with I + 50 Y
    s = np.repeat(s_dc[np.newaxis], 3, axis=0).astype(complex)
    s[1:] *= 0.9j
    return rf.Network(frequency=rf.Frequency.from_f(np.array([0, 1e9, 2e9]), unit="Hz"), s=s)


def test_clean_dc_point_keeps_only_the_conductance_inside_each_winding() -> None:
    network = _transformer_with_dc_errors()
    cleaned = clean_dc_point(network, [[1, 2], [3, 4]])

    y_dc = np.real(s2y(cleaned.s[:1], cleaned.z0[:1])[0])
    expected = np.zeros((4, 4))
    for a, b in ((0, 1), (2, 3)):
        expected[np.ix_([a, b], [a, b])] = [[1.0, -1.0], [-1.0, 1.0]]
    np.testing.assert_allclose(y_dc, expected, atol=1e-9)
    np.testing.assert_array_equal(cleaned.s[1:], network.s[1:])


def test_clean_dc_point_leaves_a_port_in_no_winding_open() -> None:
    cleaned = clean_dc_point(_transformer_with_dc_errors(), [[1, 2]])

    y_dc = np.real(s2y(cleaned.s[:1], cleaned.z0[:1])[0])
    np.testing.assert_allclose(y_dc[2:], 0.0, atol=1e-9)
    np.testing.assert_allclose(y_dc[:, 2:], 0.0, atol=1e-9)


def test_clean_dc_point_needs_a_0_hz_point() -> None:
    network = _transformer_with_dc_errors()[1:]
    assert clean_dc_point(network, [[1, 2], [3, 4]]) is network
