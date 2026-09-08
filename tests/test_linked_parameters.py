"""Tests for parameters that follow another parameter via ``linked_to``.

A linked parameter is not sampled: it takes its value, and the unit that value
is written in, from its master. Dropping the inherited unit is not cosmetic —
``19.0`` instead of ``19.0F`` is a capacitance fifteen orders of magnitude off,
which silently turns the reported best design into a different circuit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cobra.cobra import COBRA
from cobra.optimizers import GradientDescentOptimizer, OptunaOptimizer
from cobra.optimizers.base_optimizer import (
    OptimizationProperty,
    OptimizationType,
    netlist_unit,
    resolve_linked_master,
)
from cobra.optimizers.design_goal import DesignGoal, DesignParameter
from cobra.optimizers.design_goal_collection import calculate_array_penalty
from cobra.spice_sim.base_simulator import BaseSimulator, SimulationResult
from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
from cobra.spice_sim.simulation_type import SimulationType
from tests.conftest import netlist_path


def _property(name: str, unit: str | None = None, linked_to: str | None = None):
    return OptimizationProperty(
        name=name,
        type=OptimizationType.NETLIST_VARIABLE,
        min_value=0.0 if linked_to else 1.0,
        max_value=0.0 if linked_to else 20.0,
        unit=unit,
        linked_to=linked_to,
    )


# ---------------------------------------------------------------------------
# Unit resolution
# ---------------------------------------------------------------------------


def test_a_linked_parameter_inherits_its_masters_unit():
    master, follower = _property("C3", unit="F"), _property("C4", linked_to="C3")
    by_name = {p.name: p for p in (master, follower)}

    assert netlist_unit(follower, by_name) == "F"


def test_an_explicit_unit_wins_over_the_masters():
    master, follower = _property("C3", unit="F"), _property("C4", unit="p", linked_to="C3")
    by_name = {p.name: p for p in (master, follower)}

    assert netlist_unit(follower, by_name) == "p"


def test_a_unitless_parameter_gets_no_suffix():
    prop = _property("R1")
    assert netlist_unit(prop, {prop.name: prop}) == ""


def test_a_link_chain_resolves_to_the_far_end():
    a, b, c = _property("A", unit="F"), _property("B", linked_to="A"), _property("C", linked_to="B")
    by_name = {p.name: p for p in (a, b, c)}

    assert resolve_linked_master(c, by_name).name == "A"
    assert netlist_unit(c, by_name) == "F"


def test_a_link_to_nowhere_is_an_error():
    orphan = _property("C4", linked_to="missing")
    with pytest.raises(ValueError, match="unknown parameter"):
        resolve_linked_master(orphan, {orphan.name: orphan})


def test_a_circular_link_is_an_error():
    a, b = _property("A", linked_to="B"), _property("B", linked_to="A")
    by_name = {p.name: p for p in (a, b)}

    with pytest.raises(ValueError, match="Circular link"):
        resolve_linked_master(a, by_name)


# ---------------------------------------------------------------------------
# End to end: the netlist the run reports as best
# ---------------------------------------------------------------------------


class _CapturingSimulator(BaseSimulator):
    """Keeps the text of every netlist it is asked to simulate."""

    def __init__(self):
        self.netlists: list[str] = []

    def preprocess_ntwk(self, ntwk, name: str) -> str:
        return name

    def run_simulation(self, netlist_name: str) -> SimulationResult | None:
        self.netlists.append(Path(netlist_name).read_text(encoding="utf-8"))
        return SimulationResult(output_files=[netlist_name])


def _value_of(netlist: str, element: str) -> str:
    """The value token of *element* in *netlist*, e.g. ``"14.0F"`` for ``C1``."""
    line = next(line for line in netlist.splitlines() if line.startswith(f"{element} "))
    return line.split()[3]


@pytest.mark.parametrize(
    "optimizer_factory", [OptunaOptimizer, GradientDescentOptimizer], ids=["optuna", "gradient"]
)
def test_the_best_parameter_netlist_keeps_the_inherited_unit(
    tmp_path, monkeypatch, optimizer_factory
):
    """Regression: the best-parameter re-run wrote ``19.0`` where every trial
    had written ``19.0F``, so the design COBRA reported was not the one it
    evaluated.
    """
    monkeypatch.chdir(tmp_path)
    netlist = tmp_path / "circuit.cir"
    netlist.write_text(netlist_path("minimal_ac").read_text(encoding="utf-8"), encoding="utf-8")

    simulator = _CapturingSimulator()
    cobra = COBRA(
        netlist_parser=XyceNetlistParser().from_file(netlist_path("minimal_ac")),
        component_onnx_mapping={},
        optimizer=optimizer_factory(),
        circuit_simulator=simulator,
    )
    goal = DesignGoal(
        DesignParameter(
            name="unreachable",
            simulation_type=SimulationType.AC,
            formula=lambda _result, _freq=None: np.array([50.0]),
            loss=calculate_array_penalty,
        ),
        max_value=10.0,  # never satisfied, so the best-parameter re-run happens
    )

    context = cobra.run(
        netlist=str(netlist),
        design_goals=[goal],
        # C1 carries the unit; L1 follows it and declares none of its own.
        optimization_parameters=[_property("C1", unit="p"), _property("L1", linked_to="C1")],
        max_iterations=3,
    )

    assert not context.goal_achieved
    trial_netlist, final_netlist = simulator.netlists[0], simulator.netlists[-1]

    # The follower is written the same way in a trial and in the final netlist.
    assert _value_of(trial_netlist, "L1").endswith("p")
    assert _value_of(final_netlist, "L1").endswith("p")
    assert context.netlist_parameters["L1"].endswith("p")
