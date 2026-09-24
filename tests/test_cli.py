"""Tests for the ``cobra`` command-line entry point.

Exit codes are a contract for scripts and CI, and ``cobra parse --json`` must
keep stdout machine-readable, so both are pinned here. ``cobra run`` is never
executed end-to-end (that needs Xyce); only its error mapping, header and
summary are covered, with the run pipeline stubbed out.

Diagnostics go to the ``cobra`` logger rather than stdout, so tests assert on
``caplog`` where the CLI reports a problem.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from cobra import __main__ as cli
from cobra import diagnostics
from cobra.configuration.configuration import ConfigurationError, RunConfiguration
from cobra.optimizers.base_optimizer import OptimizationProperty, OptimizationType
from cobra.spice_sim.simulation_type import SimulationType
from tests.conftest import make_config_data, make_context, netlist_path

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_INTERRUPTED = 130


def stub_configured_run(run):
    """A stand-in for :class:`ConfiguredRun` carrying what the header prints."""
    return SimpleNamespace(
        configuration=SimpleNamespace(
            netlist="designs/amp.cir",
            optimizer=SimpleNamespace(name="OptunaOptimizer"),
            simulator=SimpleNamespace(name="XyceSimulator"),
            max_iterations=250,
            parallel_trials=1,
            fine_tuning=SimpleNamespace(enabled=False, palace_command="palace", iterations=3),
        ),
        parser=SimpleNamespace(simulation_type=SimulationType.AC),
        design_goals=[object(), object()],
        optimization_parameters=[
            OptimizationProperty(
                name="R1",
                type=OptimizationType.NETLIST_VARIABLE,
                min_value=10.0,
                max_value=100.0,
            )
        ],
        run=run,
    )


@pytest.fixture
def stubbed_run(monkeypatch):
    """Replace the run pipeline, keeping ``_run_config``'s reporting intact."""

    def _install(run):
        monkeypatch.setattr(RunConfiguration, "load", staticmethod(lambda _path: object()))
        monkeypatch.setattr(
            "cobra.configuration.config_runner.build_configured_run",
            lambda _configuration: stub_configured_run(run),
        )

    return _install


# ---------------------------------------------------------------------------
# cobra parse
# ---------------------------------------------------------------------------


def test_parse_succeeds_for_a_netlist_and_a_valid_configuration(written_config, capsys):
    assert cli.main(["parse", str(netlist_path("minimal_ac"))]) == EXIT_OK
    assert capsys.readouterr().out.strip()
    assert cli.main(["parse", str(written_config()), "--no-model-check"]) == EXIT_OK


def test_parse_exits_two_on_errors(config_dir: Path):
    path = config_dir / "broken.json"
    path.write_text(json.dumps(make_config_data(netlist="gone.cir")), encoding="utf-8")

    assert cli.main(["parse", str(path), "--no-model-check"]) == EXIT_CONFIG
    assert cli.main(["parse", "no/such/file.cir"]) == EXIT_CONFIG


def test_parse_json_output_is_valid_json_on_stdout(capsys):
    """Regression guard: parser chatter is redirected to stderr so that stdout
    stays parseable. Without that redirect this test fails.
    """
    assert cli.main(["parse", str(netlist_path("minimal_ac")), "--json"]) == EXIT_OK

    payload = json.loads(capsys.readouterr().out)
    assert payload["num_ports"] == 2


def test_parse_does_not_import_qt():
    """The CLI must stay headless; importing PySide6 would make it unusable on
    a server and slow to start.
    """
    import sys

    cli.main(["parse", str(netlist_path("minimal_ac"))])

    assert not any(name.startswith("PySide6") for name in sys.modules)
    assert "gmsh" not in sys.modules


# ---------------------------------------------------------------------------
# cobra run — reporting and error mapping, with the pipeline stubbed out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (ConfigurationError("bad config"), EXIT_CONFIG),
        (KeyboardInterrupt(), EXIT_INTERRUPTED),
        (RuntimeError("boom"), EXIT_ERROR),
    ],
    ids=["configuration", "interrupted", "crash"],
)
def test_run_maps_failures_to_exit_codes(stubbed_run, exception, expected):
    def _raise():
        raise exception

    stubbed_run(_raise)
    assert cli._run_config("anything.json") == expected


def test_run_prints_the_header_and_the_summary(stubbed_run, capsys):
    stubbed_run(
        lambda: make_context(
            results_dir="results/2026-01-01_demo", goal_achieved=True, iteration=12, wall_time=90.0
        )
    )

    assert cli._run_config("anything.json") == EXIT_OK

    out = capsys.readouterr().out
    # The header tells the user what is about to run, before any simulation.
    assert "amp.cir" in out
    assert "up to 250" in out
    assert "results/2026-01-01_demo" in out
    assert "achieved at iteration 12" in out


def test_run_failure_is_logged_rather_than_printed(stubbed_run, capsys, caplog):
    def _raise():
        raise RuntimeError("Xyce exploded")

    stubbed_run(_raise)
    with caplog.at_level(logging.ERROR, logger="cobra"):
        assert cli._run_config("anything.json") == EXIT_ERROR

    assert "Xyce exploded" in caplog.text
    assert "Xyce exploded" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# cobra doctor and global flags
# ---------------------------------------------------------------------------


def test_doctor_fails_only_when_a_required_dependency_is_missing(monkeypatch, caplog):
    monkeypatch.setattr(diagnostics, "REQUIRED_EXECUTABLES", ())
    monkeypatch.setattr(diagnostics, "OPTIONAL_EXECUTABLES", ())
    monkeypatch.setattr(diagnostics, "OPTIONAL_MODULES", (("cobra_optional_missing", "testing"),))

    monkeypatch.setattr(diagnostics, "REQUIRED_MODULES", (("json", "the standard library"),))
    assert cli._doctor() == EXIT_OK

    monkeypatch.setattr(diagnostics, "REQUIRED_MODULES", (("cobra_required_missing", "testing"),))
    with caplog.at_level(logging.ERROR, logger="cobra"):
        assert cli._doctor() == EXIT_ERROR
    assert "cobra_required_missing" in caplog.text


def test_output_flags_are_accepted_before_or_after_the_command():
    """``cobra -v run CONFIG`` and ``cobra run -v CONFIG`` must behave alike."""
    parser = cli._parser()
    assert parser.parse_args(["-v", "run", "cfg.json"]).verbose == 1
    assert parser.parse_args(["run", "-v", "cfg.json"]).verbose == 1
