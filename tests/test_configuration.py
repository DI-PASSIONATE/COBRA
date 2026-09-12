"""Tests for the JSON run-configuration schema in :mod:`cobra.configuration.configuration`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cobra.configuration.configuration import (
    ConfigurationError,
    DesignGoalConfig,
    FineTuningConfig,
    GeometryConfig,
    OptimizationParameterConfig,
    RunConfiguration,
)
from tests.conftest import make_config_data

# ---------------------------------------------------------------------------
# Happy path and round-tripping
# ---------------------------------------------------------------------------


def test_from_dict_resolves_paths_against_the_base_directory(config_dir: Path, config_data):
    config = RunConfiguration.from_dict(config_data, config_dir)

    assert Path(config.netlist).is_absolute()
    assert Path(config.netlist) == (config_dir / "circuit.cir").resolve()
    assert Path(config.component_models["X1"]) == (config_dir / "model.s2p").resolve()


def test_to_dict_from_dict_roundtrip_is_stable(config_dir: Path, config_data):
    config = RunConfiguration.from_dict(config_data, config_dir)
    again = RunConfiguration.from_dict(config.to_dict(config_dir), config_dir)

    assert again.to_dict(config_dir) == config.to_dict(config_dir)


def test_save_writes_paths_relative_to_the_config_file(config_dir: Path, config_data):
    config = RunConfiguration.from_dict(config_data, config_dir)
    written = config.save(config_dir / "nested" / "run.json")

    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["netlist"] == "../circuit.cir"
    assert payload["component_models"]["X1"] == "../model.s2p"


def test_save_then_load_roundtrip(config_dir: Path, config_data):
    config = RunConfiguration.from_dict(config_data, config_dir)
    path = config.save(config_dir / "run.json")

    reloaded = RunConfiguration.load(path)

    assert Path(reloaded.netlist) == Path(config.netlist)
    assert Path(reloaded.component_models["X1"]) == Path(config.component_models["X1"])
    assert reloaded.max_iterations == config.max_iterations
    assert [g.parameter for g in reloaded.design_goals] == [
        g.parameter for g in config.design_goals
    ]


def test_load_reports_invalid_json_with_a_line_number(config_dir: Path):
    broken = config_dir / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid JSON"):
        RunConfiguration.load(broken)


def test_load_rejects_a_non_object_root(config_dir: Path):
    path = config_dir / "list.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="must be a JSON object"):
        RunConfiguration.load(path)


# ---------------------------------------------------------------------------
# Schema-level rejection
# ---------------------------------------------------------------------------


def test_unknown_top_level_field_is_rejected(config_dir: Path, config_data):
    config_data["unexpected"] = 1
    with pytest.raises(ConfigurationError, match="Unknown configuration field"):
        RunConfiguration.from_dict(config_data, config_dir)


@pytest.mark.parametrize("version", [0, 2, None, "1"])
def test_wrong_schema_version_is_rejected(config_dir: Path, version):
    data = make_config_data(schema_version=version)
    with pytest.raises(ConfigurationError, match="schema_version"):
        RunConfiguration.from_dict(data, config_dir)


@pytest.mark.parametrize("netlist", ["", "   ", 5, None])
def test_missing_or_blank_netlist_is_rejected(config_dir: Path, netlist):
    data = make_config_data(netlist=netlist)
    with pytest.raises(ConfigurationError, match="netlist must be a non-empty path"):
        RunConfiguration.from_dict(data, config_dir)


def test_missing_netlist_file_is_reported(config_dir: Path):
    data = make_config_data(netlist="nowhere.cir")
    with pytest.raises(ConfigurationError, match="Netlist file not found"):
        RunConfiguration.from_dict(data, config_dir)


def test_missing_model_file_is_reported(config_dir: Path):
    data = make_config_data(component_models={"X1": "absent.onnx"})
    with pytest.raises(ConfigurationError, match="Model file for 'X1' not found"):
        RunConfiguration.from_dict(data, config_dir)


def test_check_paths_false_skips_filesystem_validation():
    """Path checks are opt-out so configurations can be validated in the abstract."""
    config = RunConfiguration(netlist="/definitely/not/here.cir")

    config.validate(check_paths=False)
    with pytest.raises(ConfigurationError, match="Netlist file not found"):
        config.validate()


def test_non_object_section_is_rejected(config_dir: Path):
    data = make_config_data(component_models=["X1"])
    with pytest.raises(ConfigurationError, match="component_models must be an object"):
        RunConfiguration.from_dict(data, config_dir)


@pytest.mark.parametrize("iterations", [0, -1, 1.5])
def test_max_iterations_must_be_a_positive_integer(config_dir: Path, iterations):
    data = make_config_data(max_iterations=iterations)
    with pytest.raises(ConfigurationError, match="max_iterations"):
        RunConfiguration.from_dict(data, config_dir)


def test_simulation_parameters_must_map_strings_to_strings(config_dir: Path):
    data = make_config_data(simulation_parameters={".AC": {"points": 101}})
    with pytest.raises(ConfigurationError, match="must be strings"):
        RunConfiguration.from_dict(data, config_dir)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["", "   ", None, 3])
def test_backend_name_must_be_a_non_empty_string(config_dir: Path, name):
    data = make_config_data(optimizer={"name": name})
    with pytest.raises(ConfigurationError, match=r"optimizer\.name"):
        RunConfiguration.from_dict(data, config_dir)


def test_backend_settings_must_be_an_object(config_dir: Path):
    data = make_config_data(simulator={"name": "XyceSimulator", "settings": []})
    with pytest.raises(ConfigurationError, match=r"simulator\.settings must be an object"):
        RunConfiguration.from_dict(data, config_dir)


def test_backend_unknown_field_is_rejected(config_dir: Path):
    data = make_config_data(optimizer={"name": "OptunaOptimizer", "extra": 1})
    with pytest.raises(ConfigurationError, match="Unknown optimizer field"):
        RunConfiguration.from_dict(data, config_dir)


# ---------------------------------------------------------------------------
# Optimization parameters
# ---------------------------------------------------------------------------


def _parameter(**overrides) -> dict:
    data = {
        "name": "R1",
        "type": "netlist_variable",
        "min_value": 1.0,
        "max_value": 2.0,
    }
    data.update(overrides)
    return data


def test_optimization_parameter_unknown_field_is_rejected():
    with pytest.raises(ConfigurationError, match="Unknown optimization parameter field"):
        OptimizationParameterConfig.from_dict(_parameter(bogus=1))


def test_optimization_parameter_missing_field_is_reported():
    with pytest.raises(ConfigurationError, match="Invalid optimization parameter"):
        OptimizationParameterConfig.from_dict({"name": "R1"})


def test_unsupported_optimization_type_is_rejected():
    with pytest.raises(ConfigurationError, match="Unsupported optimization type"):
        OptimizationParameterConfig.from_dict(_parameter(type="magic")).validate()


def test_min_greater_than_max_is_rejected():
    parameter = OptimizationParameterConfig.from_dict(_parameter(min_value=5.0, max_value=1.0))
    with pytest.raises(ConfigurationError, match="min_value greater than max_value"):
        parameter.validate()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "5"])
def test_non_finite_bounds_are_rejected(value):
    parameter = OptimizationParameterConfig.from_dict(_parameter(min_value=value))
    with pytest.raises(ConfigurationError, match="min_value"):
        parameter.validate()


def test_step_must_be_positive():
    parameter = OptimizationParameterConfig.from_dict(_parameter(step=0))
    with pytest.raises(ConfigurationError, match="step must be positive"):
        parameter.validate()


def test_duplicate_parameter_names_are_rejected(config_dir: Path):
    data = make_config_data(optimization_parameters=[_parameter(), _parameter()])
    with pytest.raises(ConfigurationError, match="names must be unique"):
        RunConfiguration.from_dict(data, config_dir)


def test_link_to_unknown_parameter_is_rejected(config_dir: Path):
    data = make_config_data(optimization_parameters=[_parameter(linked_to="ghost")])
    with pytest.raises(ConfigurationError, match="links to unknown parameter"):
        RunConfiguration.from_dict(data, config_dir)


def test_self_link_is_rejected(config_dir: Path):
    data = make_config_data(optimization_parameters=[_parameter(linked_to="R1")])
    with pytest.raises(ConfigurationError, match="cannot link to itself"):
        RunConfiguration.from_dict(data, config_dir)


def test_circular_links_are_rejected(config_dir: Path):
    data = make_config_data(
        optimization_parameters=[
            _parameter(name="R1", linked_to="C1"),
            _parameter(name="C1", linked_to="L1"),
            _parameter(name="L1", linked_to="R1"),
        ]
    )
    with pytest.raises(ConfigurationError, match="cycle"):
        RunConfiguration.from_dict(data, config_dir)


def test_acyclic_link_chain_is_accepted(config_dir: Path):
    data = make_config_data(
        optimization_parameters=[
            _parameter(name="R1", linked_to="C1"),
            _parameter(name="C1", linked_to="L1"),
            _parameter(name="L1"),
        ]
    )
    config = RunConfiguration.from_dict(data, config_dir)
    assert [p.linked_to for p in config.optimization_parameters] == ["C1", "L1", None]


# ---------------------------------------------------------------------------
# Design goals
# ---------------------------------------------------------------------------


def _goal(**overrides) -> dict:
    data = {"parameter": "S11_dB", "max_value": -10.0}
    data.update(overrides)
    return data


def test_design_goal_unknown_field_is_rejected():
    with pytest.raises(ConfigurationError, match="Unknown design goal field"):
        DesignGoalConfig.from_dict(_goal(unexpected=True))


def test_design_goal_requires_a_bound():
    with pytest.raises(ConfigurationError, match="needs a minimum or maximum"):
        DesignGoalConfig.from_dict({"parameter": "S11_dB"}).validate()


def test_design_goal_kind_must_be_known():
    with pytest.raises(ConfigurationError, match="Unsupported design goal kind"):
        DesignGoalConfig.from_dict(_goal(kind="vibes")).validate()


def test_design_goal_weight_must_be_positive():
    with pytest.raises(ConfigurationError, match="weight must be positive"):
        DesignGoalConfig.from_dict(_goal(weight=0)).validate()


def test_power_goal_requires_a_node():
    with pytest.raises(ConfigurationError, match="requires an output node"):
        DesignGoalConfig.from_dict(_goal(kind="power_dbm")).validate()


def test_gain_goal_requires_port_and_amplitude():
    with pytest.raises(ConfigurationError, match="requires port and source_amplitude"):
        DesignGoalConfig.from_dict(_goal(kind="gain_db", node="OUT")).validate()


def test_gain_goal_requires_positive_impedance():
    goal = DesignGoalConfig.from_dict(
        _goal(kind="gain_db", node="OUT", port="P1", source_amplitude=0.2)
    )
    with pytest.raises(ConfigurationError, match="requires positive impedance"):
        goal.validate()


def test_valid_gain_goal_passes():
    DesignGoalConfig.from_dict(
        _goal(kind="gain_db", node="OUT", port="P1", source_amplitude=0.2, impedance=50.0)
    ).validate()


def test_isolation_goal_requires_a_node():
    with pytest.raises(ConfigurationError, match="requires an output node"):
        DesignGoalConfig.from_dict(_goal(kind="isolation_db")).validate()


def test_valid_isolation_goal_passes():
    DesignGoalConfig.from_dict(
        _goal(kind="isolation_db", node="OUT", frequency_range="35GHz")
    ).validate()


# ---------------------------------------------------------------------------
# Fine tuning and geometries
# ---------------------------------------------------------------------------


def test_fine_tuning_optimizer_must_be_known():
    with pytest.raises(ConfigurationError, match="Unsupported fine-tuning optimizer"):
        FineTuningConfig.from_dict({"optimizer": "annealing"}).validate()


@pytest.mark.parametrize("optimizer", ["reuse", "gradient_descent"])
def test_fine_tuning_accepts_documented_optimizers(optimizer):
    FineTuningConfig.from_dict({"optimizer": optimizer}).validate()


def test_fine_tuning_iterations_must_be_at_least_one():
    with pytest.raises(ConfigurationError, match="at least 1"):
        FineTuningConfig.from_dict({"iterations": 0}).validate()


def test_geometry_source_must_be_known():
    with pytest.raises(ConfigurationError, match="Unsupported geometry source"):
        GeometryConfig.from_dict({"source": "cloud", "class_name": "Trafo"}).validate()


def test_preset_geometry_requires_a_module():
    with pytest.raises(ConfigurationError, match="requires a module"):
        GeometryConfig.from_dict({"source": "preset", "class_name": "Trafo"}).validate()


def test_custom_geometry_requires_a_file():
    with pytest.raises(ConfigurationError, match="requires a file"):
        GeometryConfig.from_dict({"source": "custom", "class_name": "Trafo"}).validate()


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
