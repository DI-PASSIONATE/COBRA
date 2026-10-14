"""Tests for :mod:`cobra.diagnostics`, the data behind ``cobra doctor``.

The checks must survive a missing or broken dependency -- that is what they
exist to report -- so nothing here may import the modules it inspects.
"""

from __future__ import annotations

import pytest

from cobra import diagnostics

MISSING = "cobra_not_a_module"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def test_module_status_reports_without_importing():
    """find_spec must be enough: importing gmsh or PySide6 here would be slow
    and could fail on a headless machine.
    """
    import sys

    assert diagnostics.module_status(MISSING) == (False, "not installed")
    diagnostics.module_status("gmsh")
    assert "gmsh" not in sys.modules


def test_a_package_without_the_submodule_is_not_installed(tmp_path, monkeypatch):
    """Only PySide6_Addons installed: the package directory exists, QtWidgets does not."""
    import sys

    package = tmp_path / "cobra_partial_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('imported')\n")
    (package / "QtCharts.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert diagnostics.module_status("cobra_partial_pkg.QtWidgets") == (False, "not installed")
    assert diagnostics.module_status("cobra_partial_pkg.QtCharts")[0]
    assert "cobra_partial_pkg" not in sys.modules


# ---------------------------------------------------------------------------
# The assembled report
# ---------------------------------------------------------------------------


@pytest.fixture
def environment(monkeypatch):
    """Build a report from a controlled set of dependencies."""

    def _build(*, required=(), optional=()):
        monkeypatch.setattr(diagnostics, "REQUIRED_MODULES", required)
        monkeypatch.setattr(diagnostics, "OPTIONAL_MODULES", optional)
        monkeypatch.setattr(diagnostics, "REQUIRED_EXECUTABLES", ())
        monkeypatch.setattr(diagnostics, "OPTIONAL_EXECUTABLES", ())
        return diagnostics.check_environment()

    return _build


def test_report_names_what_is_missing(environment):
    report = environment(required=((MISSING, "testing"),), optional=((MISSING, "testing"),))

    assert not report.ok
    assert report.missing_required == [MISSING]
    assert report.missing_optional == [MISSING]
