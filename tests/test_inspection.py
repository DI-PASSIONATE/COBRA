"""Tests for the ``cobra parse`` reporting engine in :mod:`cobra.configuration.inspection`."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from cobra.configuration.inspection import (
    ComponentModelReport,
    ConfigurationReport,
    all_issues,
    has_errors,
    inspect_netlist,
    inspect_path,
    render_report,
)
from tests.conftest import make_config_data, netlist_path

# ---------------------------------------------------------------------------
# Netlist and configuration reports
# ---------------------------------------------------------------------------


def test_netlist_report_describes_the_netlist_and_serialises():
    report = inspect_netlist(netlist_path("minimal_ac"))

    assert report.parsed
    assert report.num_ports == 2
    assert {c.name for c in report.components} == {"X1"}
    assert render_report(report, full=True).strip()
    json.dumps(report.to_dict())


def test_netlist_problems_are_reported_rather_than_raised():
    assert has_errors(inspect_netlist("definitely/not/here.cir"))
    # The fixture's .INCLUDE and .LIB targets do not exist on disk.
    report = inspect_netlist(netlist_path("subckt_and_includes"))
    assert any(not include.exists for include in report.includes)


def test_valid_configuration_has_no_errors(written_config):
    report = inspect_path(written_config(), check_models=False)

    assert isinstance(report, ConfigurationReport)
    assert not has_errors(report)
    assert report.netlist is not None
    assert report.netlist.num_ports == 2
    assert render_report(report).strip()
    json.dumps(report.to_dict())


def test_broken_configurations_are_reported_as_errors(config_dir: Path):
    broken = {
        "missing-netlist.json": json.dumps(make_config_data(netlist="missing.cir")),
        "unknown-goal.json": json.dumps(
            make_config_data(
                design_goals=[{"parameter": "S99_dB", "max_value": -10.0, "kind": "catalogue"}]
            )
        ),
        "invalid.json": "{ nope",
    }
    for name, text in broken.items():
        path = config_dir / name
        path.write_text(text, encoding="utf-8")
        assert has_errors(inspect_path(path, check_models=False)), name


# ---------------------------------------------------------------------------
# Harmonic-balance goal frequencies
# ---------------------------------------------------------------------------


def _hb_config(config_dir: Path, frequency_range: str, **overrides) -> Path:
    """A configuration whose only goal sits at *frequency_range* on the two-tone netlist."""
    shutil.copy(netlist_path("hb_two_tone"), config_dir / "mixer.cir")
    data = make_config_data(
        netlist="mixer.cir",
        simulation_parameters={},
        design_goals=[
            {
                "parameter": "Power_dBm[OUT]",
                "frequency_range": frequency_range,
                "min_value": None,
                "max_value": -30.0,
                "weight": 1.0,
                "kind": "power_dbm",
                "node": "OUT",
            },
        ],
    )
    data.update(overrides)
    path = config_dir / "hb.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _frequency_issues(path: Path) -> list[str]:
    return [
        issue.message
        for issue in all_issues(inspect_path(path, check_models=False))
        if "Goal frequency" in issue.message
    ]


def test_hb_goal_frequencies_are_checked_against_the_mixing_grid(config_dir: Path):
    # f1-f2 = 85 GHz is a real line of .HB 95E9 10E9 without being a fundamental.
    assert _frequency_issues(_hb_config(config_dir, "85GHz")) == []
    [message] = _frequency_issues(_hb_config(config_dir, "37GHz"))
    assert "not on the .HB grid" in message


def test_configured_numfreq_widens_the_hb_grid(config_dir: Path):
    # The netlist numfreq=5 cannot reach f1-6f2 = 35 GHz; the numfreq=4,40 that
    # the run actually writes can, so the goal must be judged against the latter.
    assert _frequency_issues(_hb_config(config_dir, "35GHz"))
    assert (
        _frequency_issues(
            _hb_config(
                config_dir,
                "35GHz",
                simulation_parameters={".OPTIONS:hbint": {"numfreq": "4,40"}},
            )
        )
        == []
    )


# ---------------------------------------------------------------------------
# Surrogate model frequency band
# ---------------------------------------------------------------------------


def _onnx_config(
    config_dir: Path,
    netlist: str,
    band: tuple[float, float] | None,
    monkeypatch,
    *,
    trained_w: tuple[float, float] | None = None,
    constraints: list[str] | None = None,
    bounds: tuple[float, float] = (1.0, 2.0),
) -> Path:
    """A configuration whose X1 is an ONNX model declaring *band*, on *netlist*.

    Loading a real ONNX file needs a model asset the suite does not ship, so the
    model inspection is stubbed with the metadata under test.
    """
    from cobra.configuration import inspection

    def inspect_onnx(_path: Path, entry: ComponentModelReport) -> None:
        entry.model_ports = 2
        entry.model_inputs = ["w", "frequency"]
        entry.model_frequency_range = band
        entry.model_input_ranges = {"w": trained_w} if trained_w else {}
        entry.model_constraints = constraints or []
        entry.model_guarantees = {"passive": False, "reciprocal": True}

    shutil.copy(netlist_path(netlist), config_dir / "circuit.cir")
    (config_dir / "model.onnx").write_bytes(b"")
    monkeypatch.setattr(inspection, "_inspect_onnx", inspect_onnx)
    data = make_config_data(
        component_models={"X1": "model.onnx"},
        simulation_parameters={},
        optimization_parameters=[
            {
                "name": "X1:w",
                "type": "model_input",
                "min_value": bounds[0],
                "max_value": bounds[1],
                "step": 0.1,
                "unit": None,
                "linked_to": None,
            },
        ],
    )
    path = config_dir / "onnx.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _band_issues(path: Path) -> list[str]:
    return [
        issue.message
        for issue in all_issues(inspect_path(path, check_models=True))
        if "the model only covers" in issue.message
    ]


def test_ac_sweep_is_checked_against_the_model_band(config_dir: Path, monkeypatch):
    # minimal_ac sweeps .AC LIN 101 1G 10G.
    assert _band_issues(_onnx_config(config_dir, "minimal_ac", (1e9, 10e9), monkeypatch)) == []

    [message] = _band_issues(_onnx_config(config_dir, "minimal_ac", (2e9, 8e9), monkeypatch))
    assert message.startswith(".AC evaluates the circuit from 1 to 10 GHz")
    assert "2-8 GHz" in message


def test_hb_harmonics_are_checked_against_the_model_band(config_dir: Path, monkeypatch):
    # .HB 95E9 10E9 with numfreq=5 reaches the fifth harmonic at 475 GHz.
    assert _band_issues(_onnx_config(config_dir, "hb_two_tone", (1e9, 500e9), monkeypatch)) == []
    messages = _band_issues(_onnx_config(config_dir, "hb_two_tone", (1e9, 400e9), monkeypatch))

    assert len(messages) == 1
    assert messages[0].startswith(".HB evaluates the circuit from 10 to 475 GHz")


def test_model_without_a_declared_band_is_an_error(config_dir: Path, monkeypatch):
    report = inspect_path(_onnx_config(config_dir, "minimal_ac", None, monkeypatch), check_models=True)

    assert has_errors(report)
    assert any("declares no usable frequency range" in issue.message for issue in all_issues(report))


def _range_warnings(path: Path) -> list[str]:
    return [
        issue.message
        for issue in all_issues(inspect_path(path, check_models=True))
        if "was trained on" in issue.message
    ]


def test_bounds_outside_the_trained_range_are_a_warning(config_dir: Path, monkeypatch):
    inside = _onnx_config(config_dir, "minimal_ac", (1e9, 10e9), monkeypatch, trained_w=(1.0, 2.0))
    assert _range_warnings(inside) == []

    beyond = _onnx_config(
        config_dir, "minimal_ac", (1e9, 10e9), monkeypatch, trained_w=(1.0, 2.0), bounds=(0.5, 2.0)
    )
    report = inspect_path(beyond, check_models=True)
    [message] = _range_warnings(beyond)
    assert "[0.5, 2] reach beyond [1, 2]" in message
    assert "'w'" in message
    assert not has_errors(report)


def test_report_lists_the_model_metadata(config_dir: Path, monkeypatch):
    path = _onnx_config(
        config_dir, "minimal_ac", (1e9, 10e9), monkeypatch, trained_w=(1.0, 2.0), constraints=["w <= 2"]
    )
    report = inspect_path(path, check_models=True)
    text = render_report(report, full=True)

    assert "trained ranges: w=[1, 2]" in text
    assert "constraints: w <= 2" in text
    assert "guarantees: reciprocal" in text
    [entry] = report.to_dict()["component_models"]
    assert entry["model_constraints"] == ["w <= 2"]


def test_malformed_constraints_are_an_error(config_dir: Path, monkeypatch):
    from cobra.configuration import inspection
    from cobra.configuration.configuration import ConfigurationError

    path = _onnx_config(config_dir, "minimal_ac", (1e9, 10e9), monkeypatch)

    def refuse(_path: Path, _entry: ComponentModelReport) -> None:
        raise ConfigurationError("Constraint '__import__(\"os\")' contains unsupported syntax (Call).")

    monkeypatch.setattr(inspection, "_inspect_onnx", refuse)
    report = inspect_path(path, check_models=True)

    messages = [issue.message for issue in all_issues(report)]
    assert has_errors(report)
    assert any(message.startswith("The run would refuse this model") for message in messages)
    assert not any("declares no usable frequency range" in message for message in messages)
