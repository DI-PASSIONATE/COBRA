"""Tests for the JSON run-configuration schema in :mod:`cobra.configuration.configuration`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cobra.configuration.configuration import (
    ConfigurationError,
    DesignGoalConfig,
    FineTuningConfig,
    OptimizationParameterConfig,
    RunConfiguration,
)
from tests.conftest import make_config_data

# ---------------------------------------------------------------------------
# Paths and round-tripping
# ---------------------------------------------------------------------------


def test_from_dict_resolves_paths_against_the_base_directory(config_dir: Path, config_data):
    config = RunConfiguration.from_dict(config_data, config_dir)

    assert Path(config.netlist) == (config_dir / "circuit.cir").resolve()
    assert Path(config.component_models["X1"]) == (config_dir / "model.s2p").resolve()


def test_save_writes_relative_paths_and_loads_back(config_dir: Path, config_data):
    config = RunConfiguration.from_dict(config_data, config_dir)
    written = config.save(config_dir / "nested" / "run.json")

    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["netlist"] == "../circuit.cir"
    assert payload["component_models"]["X1"] == "../model.s2p"

    reloaded = RunConfiguration.load(written)
    assert reloaded.to_dict(config_dir) == config.to_dict(config_dir)


def test_load_reports_invalid_json(config_dir: Path):
    broken = config_dir / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid JSON"):
        RunConfiguration.load(broken)


# ---------------------------------------------------------------------------
# Rejection before anything runs
# ---------------------------------------------------------------------------


def _parameter(**overrides) -> dict:
    data = {"name": "R1", "type": "netlist_variable", "min_value": 1.0, "max_value": 2.0}
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"unexpected": 1}, "Unknown configuration field"),
        ({"schema_version": 2}, "schema_version"),
        ({"netlist": "nowhere.cir"}, "Netlist file not found"),
        ({"component_models": {"X1": "absent.onnx"}}, "Model file for 'X1' not found"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"simulation_parameters": {".AC": {"points": 101}}}, "must be strings"),
        ({"optimization_parameters": [_parameter(), _parameter()]}, "names must be unique"),
        ({"optimization_parameters": [_parameter(linked_to="ghost")]}, "links to unknown parameter"),
        (
            {
                "optimization_parameters": [
                    _parameter(name="R1", linked_to="C1"),
                    _parameter(name="C1", linked_to="R1"),
                ]
            },
            "cycle",
        ),
    ],
    ids=[
        "unknown-field",
        "schema-version",
        "missing-netlist",
        "missing-model",
        "max-iterations",
        "numeric-simulation-parameter",
        "duplicate-parameter",
        "unknown-link",
        "circular-link",
    ],
)
def test_invalid_configuration_is_rejected(config_dir: Path, overrides, match):
    with pytest.raises(ConfigurationError, match=match):
        RunConfiguration.from_dict(make_config_data(**overrides), config_dir)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"min_value": 5.0, "max_value": 1.0}, "min_value greater than max_value"),
        ({"step": 0}, "step must be positive"),
        ({"type": "magic"}, "Unsupported optimization type"),
    ],
    ids=["min-above-max", "zero-step", "unknown-type"],
)
def test_invalid_optimization_parameter_is_rejected(overrides, match):
    with pytest.raises(ConfigurationError, match=match):
        OptimizationParameterConfig.from_dict(_parameter(**overrides)).validate()


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"parameter": "S11_dB"}, "needs a minimum or maximum"),
        ({"parameter": "S11_dB", "max_value": -10.0, "kind": "vibes"}, "Unsupported design goal kind"),
        ({"parameter": "P", "max_value": 0.0, "kind": "power_dbm"}, "requires an output node"),
        (
            {"parameter": "G", "min_value": 0.0, "kind": "gain_db", "node": "OUT"},
            "requires port and source_amplitude",
        ),
    ],
    ids=["no-bound", "unknown-kind", "power-without-node", "gain-without-port"],
)
def test_invalid_design_goal_is_rejected(data, match):
    with pytest.raises(ConfigurationError, match=match):
        DesignGoalConfig.from_dict(data).validate()


def test_complete_large_signal_goals_are_accepted():
    DesignGoalConfig.from_dict(
        {
            "parameter": "G",
            "min_value": 0.0,
            "kind": "gain_db",
            "node": "OUT",
            "port": "P1",
            "source_amplitude": 0.2,
            "impedance": 50.0,
        }
    ).validate()
    DesignGoalConfig.from_dict(
        {"parameter": "I", "min_value": 30.0, "kind": "isolation_db", "node": "OUT", "frequency_range": "35GHz"}
    ).validate()


# ---------------------------------------------------------------------------
# Fine tuning and geometries
# ---------------------------------------------------------------------------


def test_fine_tuning_optimizer_must_be_known():
    with pytest.raises(ConfigurationError, match="Unsupported fine-tuning optimizer"):
        FineTuningConfig.from_dict({"optimizer": "annealing"}).validate()


def test_custom_geometry_file_is_resolved_and_checked(config_dir: Path):
    data = make_config_data(
        fine_tuning={
            "enabled": True,
            "geometries": {
                "X1": {"source": "custom", "class_name": "Trafo", "file": "geom.py"}
            },
        }
    )
    with pytest.raises(ConfigurationError, match="Geometry file not found"):
        RunConfiguration.from_dict(data, config_dir)

    (config_dir / "geom.py").write_text("class Trafo: pass\n", encoding="utf-8")
    config = RunConfiguration.from_dict(data, config_dir)

    resolved = config.fine_tuning.geometries["X1"].file
    assert resolved is not None
    assert Path(resolved) == (config_dir / "geom.py").resolve()
