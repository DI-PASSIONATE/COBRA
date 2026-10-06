"""Tests for :meth:`EMSurrogateStage.s_param_dict_to_network`: turning a surrogate's
named ``S<i><j>_real``/``_imag`` outputs into a network, for both output layouts
ORCA exports (the full matrix and the upper triangle only).
"""

from __future__ import annotations

import numpy as np
import pytest

from cobra.stages.em_surrogate_stage import EMSurrogateStage

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
