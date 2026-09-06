"""Shared fixtures for the COBRA test suite.

Everything here is hermetic: the fixture netlists are hand-written and free of
absolute paths, and configurations are materialised into ``tmp_path`` so that
``RunConfiguration.from_dict`` — which validates that referenced files exist —
can run without any of the gitignored example assets.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser

FIXTURE_DIR = Path(__file__).parent / "fixtures"
NETLIST_DIR = FIXTURE_DIR / "netlists"

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_NETLIST_DIR = REPO_ROOT / "examples" / "netlists"

# A valid 2-port Touchstone file, small enough to inline.
MINIMAL_S2P = """\
! Minimal 2-port Touchstone written by the COBRA test suite
# HZ S RI R 50
1e9  -0.1 0.0  0.9 0.0  0.9 0.0  -0.1 0.0
2e9  -0.2 0.0  0.8 0.0  0.8 0.0  -0.2 0.0
3e9  -0.3 0.0  0.7 0.0  0.7 0.0  -0.3 0.0
"""


def netlist_path(name: str) -> Path:
    """Absolute path of a fixture netlist, e.g. ``netlist_path("minimal_ac")``."""
    return NETLIST_DIR / f"{name}.cir"


@pytest.fixture
def parser_factory():
    """Return a callable that parses a fixture netlist into a fresh parser."""

    def _factory(name: str) -> XyceNetlistParser:
        return XyceNetlistParser().from_file(netlist_path(name))

    return _factory


@pytest.fixture
def minimal_ac(parser_factory) -> XyceNetlistParser:
    return parser_factory("minimal_ac")


@pytest.fixture
def editable_netlist(tmp_path: Path):
    """Copy a fixture netlist into ``tmp_path`` and parse it from there.

    Use this for mutation tests so the checked-in fixture is never touched.
    """

    def _factory(name: str) -> XyceNetlistParser:
        destination = tmp_path / f"{name}.cir"
        shutil.copy(netlist_path(name), destination)
        return XyceNetlistParser().from_file(destination)

    return _factory


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A directory holding a real netlist and a real model file.

    ``RunConfiguration`` validates that referenced paths exist, so configs under
    test need genuine files next to them.
    """
    shutil.copy(netlist_path("minimal_ac"), tmp_path / "circuit.cir")
    (tmp_path / "model.s2p").write_text(MINIMAL_S2P, encoding="utf-8")
    return tmp_path


def make_config_data(**overrides: Any) -> dict[str, Any]:
    """Build a minimal configuration payload, with per-test overrides applied."""
    data: dict[str, Any] = {
        "schema_version": 1,
        "netlist": "circuit.cir",
        "component_models": {"X1": "model.s2p"},
        "simulation_parameters": {".AC": {"points": "101", "start_freq": "1G"}},
        "optimizer": {"name": "OptunaOptimizer", "settings": {"multi_objective": False}},
        "simulator": {"name": "XyceSimulator", "settings": {"xyce_command": "Xyce"}},
        "max_iterations": 10,
        "optimization_parameters": [
            {
                "name": "R1",
                "type": "netlist_variable",
                "min_value": 10.0,
                "max_value": 100.0,
                "step": 1.0,
                "unit": None,
                "linked_to": None,
            },
        ],
        "design_goals": [
            {
                "parameter": "S11_dB",
                "frequency_range": "1-10GHz",
                "min_value": None,
                "max_value": -10.0,
                "weight": 1.0,
                "kind": "catalogue",
            },
        ],
        "fine_tuning": {"enabled": False},
    }
    data.update(overrides)
    return data


@pytest.fixture
def config_data() -> dict[str, Any]:
    return make_config_data()


@pytest.fixture
def written_config(config_dir: Path):
    """Write a configuration JSON into ``config_dir`` and return its path."""

    def _factory(data: dict[str, Any] | None = None, name: str = "config.json") -> Path:
        payload = make_config_data() if data is None else data
        path = config_dir / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    return _factory
