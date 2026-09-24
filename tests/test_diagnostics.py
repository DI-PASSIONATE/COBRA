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
