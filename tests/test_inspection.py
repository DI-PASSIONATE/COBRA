"""Tests for the ``cobra parse`` reporting engine in :mod:`cobra.configuration.inspection`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cobra.configuration.configuration import ConfigurationError
from cobra.configuration.inspection import (
    ConfigurationReport,
    NetlistReport,
    Severity,
    all_issues,
    count_issues,
    has_errors,
    inspect_netlist,
    inspect_path,
    is_configuration_file,
    render_report,
)
from tests.conftest import make_config_data, netlist_path

# ---------------------------------------------------------------------------
# Netlist reports
# ---------------------------------------------------------------------------


def test_inspect_netlist_reports_structure():
    report = inspect_netlist(netlist_path("minimal_ac"))

    assert report.exists
    assert report.parsed
    assert report.num_ports == 2
    assert {c.name for c in report.components} == {"X1"}


def test_inspect_netlist_of_a_missing_file_does_not_raise():
    report = inspect_netlist("definitely/not/here.cir")

    assert not report.exists
    assert has_errors(report)


def test_netlist_report_renders_non_empty_text():
    report = inspect_netlist(netlist_path("minimal_ac"))

    assert render_report(report).strip()
    assert render_report(report, full=True).strip()


def test_netlist_report_is_json_serialisable():
    report = inspect_netlist(netlist_path("hb_two_tone"))
    json.dumps(report.to_dict())


def test_unresolved_include_is_reported():
    """The fixture's .INCLUDE and .LIB targets do not exist on disk."""
    report = inspect_netlist(netlist_path("subckt_and_includes"))

    assert any(not include.exists for include in report.includes)
    assert all_issues(report)


# ---------------------------------------------------------------------------
# Configuration reports
# ---------------------------------------------------------------------------


def test_valid_configuration_has_no_errors(written_config):
    path = written_config()
    report = inspect_path(path, check_models=False)

    assert isinstance(report, ConfigurationReport)
    assert not has_errors(report)


def test_configuration_report_includes_the_netlist_report(written_config):
    report = inspect_path(written_config(), check_models=False)

    assert isinstance(report, ConfigurationReport)
    assert report.netlist is not None
    assert report.netlist.num_ports == 2


def test_configuration_report_renders_and_serialises(written_config):
    report = inspect_path(written_config(), check_models=False)

    assert render_report(report).strip()
    json.dumps(report.to_dict())


def test_broken_configuration_is_reported_as_an_error(config_dir: Path):
    data = make_config_data(netlist="missing.cir")
    path = config_dir / "broken.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    report = inspect_path(path, check_models=False)

    assert has_errors(report)
    assert any(issue.severity is Severity.ERROR for issue in all_issues(report))


def test_invalid_json_is_reported_rather_than_raised(config_dir: Path):
    path = config_dir / "invalid.json"
    path.write_text("{ nope", encoding="utf-8")

    report = inspect_path(path)

    assert has_errors(report)


def test_unknown_design_goal_parameter_is_reported(config_dir: Path):
    data = make_config_data(
        design_goals=[{"parameter": "S99_dB", "max_value": -10.0, "kind": "catalogue"}]
    )
    path = config_dir / "bad_goal.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert has_errors(inspect_path(path, check_models=False))


# ---------------------------------------------------------------------------
# Issue counting
# ---------------------------------------------------------------------------


def test_count_issues_covers_every_severity():
    counts = count_issues(inspect_netlist(netlist_path("minimal_ac")))
    assert set(counts) == {s.value for s in Severity}


def test_has_errors_agrees_with_the_counts():
    report = inspect_netlist("nope.cir")
    assert has_errors(report) == (count_issues(report)["error"] > 0)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_is_configuration_file_detects_json_by_suffix(config_dir: Path):
    path = config_dir / "thing.json"
    path.write_text("{}", encoding="utf-8")
    assert is_configuration_file(path)


def test_is_configuration_file_detects_json_by_content(config_dir: Path):
    path = config_dir / "thing.txt"
    path.write_text('  {"schema_version": 1}', encoding="utf-8")
    assert is_configuration_file(path)


def test_is_configuration_file_rejects_a_netlist():
    assert not is_configuration_file(netlist_path("minimal_ac"))


def test_missing_file_falls_back_to_the_suffix():
    """The suffix decides before disk is touched, so a missing .json still
    dispatches as a configuration; anything else cannot be read and is not.
    """
    assert is_configuration_file("nowhere/at/all.json")
    assert not is_configuration_file("nowhere/at/all.cir")


def test_inspect_path_auto_detects_a_netlist():
    assert isinstance(inspect_path(netlist_path("minimal_ac")), NetlistReport)


def test_inspect_path_auto_detects_a_configuration(written_config):
    assert isinstance(inspect_path(written_config(), check_models=False), ConfigurationReport)


def test_inspect_path_honours_an_explicit_kind(written_config):
    """Forcing 'netlist' on a JSON file parses it as a netlist instead."""
    assert isinstance(inspect_path(written_config(), kind="netlist"), NetlistReport)


def test_inspect_path_rejects_an_unknown_kind():
    with pytest.raises(ConfigurationError, match="Unsupported parse kind"):
        inspect_path(netlist_path("minimal_ac"), kind="sideways")
