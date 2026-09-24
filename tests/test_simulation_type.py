"""Tests for :class:`~cobra.spice_sim.simulation_type.SimulationType`."""

from __future__ import annotations

from cobra.spice_sim.simulation_type import SimulationType


def test_from_directive():
    cases = {
        ".AC": SimulationType.AC,
        ".HB": SimulationType.HB,
        ".TRAN": SimulationType.TRAN,
        # .LIN post-processes .AC for Touchstone output; it is not its own analysis.
        ".LIN": SimulationType.AC,
        "  .tran  ": SimulationType.TRAN,
        ".OP": SimulationType.UNKNOWN,
        "AC": SimulationType.UNKNOWN,
    }
    assert {directive: SimulationType.from_directive(directive) for directive in cases} == cases
