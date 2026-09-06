"""Tests for :mod:`cobra.diagnostics`, the data behind ``cobra doctor``.

The checks must survive a missing or broken dependency -- that is what they
exist to report -- so nothing here may import the modules it inspects.
"""

from __future__ import annotations

import pytest

from cobra import diagnostics
from cobra.console import Palette

MISSING = "cobra_not_a_module"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def test_missing_module_is_reported_not_raised():
    assert diagnostics.module_status(MISSING) == (False, "not installed")


def test_present_module_reports_a_detail():
    available, detail = diagnostics.module_status("json")

    assert available
    assert detail


def test_module_status_does_not_import_the_module(monkeypatch):
    """find_spec must be enough: importing gmsh or PySide6 here would be slow
    and could fail on a headless machine.
    """
    import sys

    monkeypatch.delitem(sys.modules, "this_module_is_not_imported", raising=False)
    diagnostics.module_status("gmsh")

    assert "gmsh" not in sys.modules


def test_version_falls_back_when_the_distribution_is_unknown():
    # The standard library ships no distribution metadata.
    assert diagnostics.module_version("json") == "installed"


def test_import_name_is_resolved_to_its_distribution():
    """scikit-rf installs as ``skrf``; the report must show a real version."""
    assert diagnostics.module_version("skrf") not in ("installed", "not installed")


def test_missing_executable_is_reported(monkeypatch):
    monkeypatch.setattr(diagnostics.shutil, "which", lambda _name: None)
    assert diagnostics.executable_status("Xyce") == (False, "not on PATH")


def test_found_executable_reports_its_path(monkeypatch):
    monkeypatch.setattr(diagnostics.shutil, "which", lambda _name: "/usr/bin/Xyce")
    assert diagnostics.executable_status("Xyce") == (True, "/usr/bin/Xyce")


# ---------------------------------------------------------------------------
# The assembled report
# ---------------------------------------------------------------------------


@pytest.fixture
def environment(monkeypatch):
    """Build a report from a controlled set of dependencies."""

    def _build(*, required=(), optional=(), executables=()):
        monkeypatch.setattr(diagnostics, "REQUIRED_MODULES", required)
        monkeypatch.setattr(diagnostics, "OPTIONAL_MODULES", optional)
        monkeypatch.setattr(diagnostics, "REQUIRED_EXECUTABLES", executables)
        monkeypatch.setattr(diagnostics, "OPTIONAL_EXECUTABLES", ())
        return diagnostics.check_environment()

    return _build


def test_report_is_ok_when_everything_required_is_present(environment):
    report = environment(required=(("json", "the standard library"),))

    assert report.ok
    assert report.missing_required == []
    assert report.version
    assert report.python


def test_report_names_what_is_missing(environment):
    report = environment(
        required=((MISSING, "testing"),), optional=((MISSING, "testing"),)
    )

    assert not report.ok
    assert report.missing_required == [MISSING]
    assert report.missing_optional == [MISSING]


def test_a_missing_optional_component_does_not_fail_the_report(environment):
    report = environment(
        required=(("json", "the standard library"),), optional=((MISSING, "testing"),)
    )

    assert report.ok
    assert report.missing_optional == [MISSING]


def test_rendered_report_lists_names_details_and_purposes(environment):
    report = environment(
        required=(("json", "the standard library"),), optional=((MISSING, "testing"),)
    )

    text = diagnostics.render_environment(report, Palette())

    assert "Required" in text
    assert "Optional" in text
    assert "the standard library" in text
    assert f"{MISSING}" in text
    assert "not installed" in text


def test_rendered_report_is_plain_text_without_a_palette(environment):
    report = environment(required=(("json", "the standard library"),))

    assert "\033[" not in diagnostics.render_environment(report)


def test_an_empty_section_renders_instead_of_crashing(environment):
    """Guard for the width calculation, which needs at least one row."""
    report = environment(required=(("json", "the standard library"),), optional=())

    assert "nothing to check" in diagnostics.render_environment(report)
