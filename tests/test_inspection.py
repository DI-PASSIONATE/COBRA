"""Tests for the ``cobra parse`` reporting engine in :mod:`cobra.configuration.inspection`."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from cobra.configuration.inspection import (
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


def _onnx_config(config_dir: Path, netlist: str, band: tuple[float, float] | None, monkeypatch) -> Path:
    """A configuration whose X1 is an ONNX model declaring *band*, on *netlist*.

    Loading a real ONNX file needs a model asset the suite does not ship, so the
    model inspection is stubbed with the band under test.
    """
    from cobra.configuration import inspection

    shutil.copy(netlist_path(netlist), config_dir / "circuit.cir")
    (config_dir / "model.onnx").write_bytes(b"")
    monkeypatch.setattr(
        inspection, "_inspect_onnx", lambda _path: (2, ["w", "frequency"], band, None)
    )
    data = make_config_data(
        component_models={"X1": "model.onnx"},
        simulation_parameters={},
        optimization_parameters=[
            {
                "name": "X1:w",
                "type": "model_input",
                "min_value": 1.0,
                "max_value": 2.0,
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
    assert any("declares no frequency range" in issue.message for issue in all_issues(report))
