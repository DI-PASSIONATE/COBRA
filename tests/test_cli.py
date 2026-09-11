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


def test_parse_netlist_succeeds(capsys):
    assert cli.main(["parse", str(netlist_path("minimal_ac"))]) == EXIT_OK
    assert capsys.readouterr().out.strip()


def test_parse_valid_configuration_succeeds(written_config):
    assert cli.main(["parse", str(written_config()), "--no-model-check"]) == EXIT_OK


def test_parse_reports_errors_with_exit_code_two(config_dir: Path):
    path = config_dir / "broken.json"
    path.write_text(json.dumps(make_config_data(netlist="gone.cir")), encoding="utf-8")

    assert cli.main(["parse", str(path), "--no-model-check"]) == EXIT_CONFIG


def test_parse_missing_netlist_exits_two():
    assert cli.main(["parse", "no/such/file.cir"]) == EXIT_CONFIG


def test_forcing_config_kind_on_a_netlist_reports_rather_than_crashes(capsys):
    """A netlist read as a configuration yields a report full of errors (exit 2),
    not a traceback.
    """
    assert cli.main(["parse", str(netlist_path("minimal_ac")), "--kind", "config"]) == EXIT_CONFIG
    assert "Invalid JSON" in capsys.readouterr().out


def test_missing_target_reports_instead_of_crashing(capsys, tmp_path: Path):
    """Even a directory degrades into a report rather than a traceback."""
    assert cli.main(["parse", str(tmp_path)]) == EXIT_CONFIG
    assert "not parsed" in capsys.readouterr().out


@pytest.mark.parametrize("exception", [ConfigurationError("bad kind"), OSError("unreadable")])
def test_parse_boundary_maps_exceptions_to_exit_two(monkeypatch, capsys, caplog, exception):
    """inspect_path is defensive enough that this handler is hard to reach through
    normal input, so exercise it directly.
    """

    def _raise(*_args, **_kwargs):
        raise exception

    monkeypatch.setattr("cobra.configuration.inspection.inspect_path", _raise)

    args = cli._parser().parse_args(["parse", "whatever.cir"])
    with caplog.at_level(logging.ERROR, logger="cobra"):
        assert cli._parse_target(args) == EXIT_CONFIG
    assert str(exception) in caplog.text
    assert capsys.readouterr().out == ""


def test_parse_json_output_is_valid_json_on_stdout(capsys):
    """Regression guard: parser chatter is redirected to stderr so that stdout
    stays parseable. Without that redirect this test fails.
    """
    assert cli.main(["parse", str(netlist_path("minimal_ac")), "--json"]) == EXIT_OK

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["num_ports"] == 2


def test_parse_json_output_for_a_configuration(written_config, capsys):
    cli.main(["parse", str(written_config()), "--json", "--no-model-check"])

    payload = json.loads(capsys.readouterr().out)
    assert "netlist" in payload


def test_parse_full_flag_produces_at_least_as_much_text(capsys):
    cli.main(["parse", str(netlist_path("subckt_and_includes"))])
    short = capsys.readouterr().out

    cli.main(["parse", str(netlist_path("subckt_and_includes")), "--full"])
    full = capsys.readouterr().out

    assert len(full) >= len(short)


def test_parse_does_not_import_qt():
    """The CLI must stay headless; importing PySide6 would make it unusable on
    a server and slow to start.
    """
    import sys

    cli.main(["parse", str(netlist_path("minimal_ac"))])

    assert not any(name.startswith("PySide6") for name in sys.modules)
    assert "gmsh" not in sys.modules


# ---------------------------------------------------------------------------
# cobra run — error mapping only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (ConfigurationError("bad config"), EXIT_CONFIG),
        (FileNotFoundError("missing"), EXIT_CONFIG),
        (OSError("io"), EXIT_CONFIG),
        (KeyboardInterrupt(), EXIT_INTERRUPTED),
        (RuntimeError("boom"), EXIT_ERROR),
    ],
)
def test_run_maps_failures_to_exit_codes(stubbed_run, exception, expected):
    def _raise():
        raise exception

    stubbed_run(_raise)
    assert cli._run_config("anything.json") == expected


def test_run_reports_the_configuration_before_starting(stubbed_run, capsys):
    """The header tells the user what is about to run, before any simulation."""
    stubbed_run(lambda: make_context(results_dir="results/demo"))

    assert cli._run_config("anything.json") == EXIT_OK

    out = capsys.readouterr().out
    assert "amp.cir" in out
    assert "OptunaOptimizer" in out
    assert "up to 250" in out


def test_run_returns_zero_and_prints_the_results_directory(stubbed_run, capsys):
    stubbed_run(
        lambda: make_context(
            results_dir="results/2026-01-01_demo",
            goal_achieved=True,
            iteration=12,
            wall_time=90.0,
            # Concurrent trials make the stage times add up to more than the run
            # took; the summary must report the elapsed time, not this.
            times={"total_time": 300.0},
        )
    )

    assert cli._run_config("anything.json") == EXIT_OK

    out = capsys.readouterr().out
    assert "results/2026-01-01_demo" in out
    assert "achieved at iteration 12" in out
    assert "1m 30s" in out
    assert "5m" not in out


def test_run_summary_tolerates_a_context_without_timings(stubbed_run, capsys):
    """A run stopped early leaves ``times`` at zero; the summary still prints."""
    stubbed_run(lambda: make_context(results_dir="results/partial"))

    assert cli._run_config("anything.json") == EXIT_OK
    assert "results/partial" in capsys.readouterr().out


def test_run_failure_is_logged_rather_than_printed(stubbed_run, capsys, caplog):
    def _raise():
        raise RuntimeError("Xyce exploded")

    stubbed_run(_raise)
    with caplog.at_level(logging.ERROR, logger="cobra"):
        assert cli._run_config("anything.json") == EXIT_ERROR

    assert "Xyce exploded" in caplog.text
    assert "Xyce exploded" not in capsys.readouterr().out


def test_parse_defaults():
    args = cli._parser().parse_args(["parse", "x.cir"])

    assert args.kind == "auto"
    assert args.json is False
    assert args.full is False
    assert args.check_models is True


def test_no_model_check_clears_the_flag():
    args = cli._parser().parse_args(["parse", "x.cir", "--no-model-check"])
    assert args.check_models is False


def test_invalid_kind_is_rejected_by_argparse():
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["parse", "x.cir", "--kind", "sideways"])


# ---------------------------------------------------------------------------
# cobra doctor
# ---------------------------------------------------------------------------


def test_doctor_reports_both_sections(capsys):
    exit_code = cli._doctor()

    out = capsys.readouterr().out
    assert "Required" in out
    assert "Optional" in out
    assert "python" in out
    assert exit_code in (EXIT_OK, EXIT_ERROR)


def test_doctor_fails_when_a_required_dependency_is_missing(monkeypatch, caplog):
    monkeypatch.setattr(diagnostics, "REQUIRED_MODULES", (("cobra_not_a_module", "testing"),))
    monkeypatch.setattr(diagnostics, "REQUIRED_EXECUTABLES", ())

    with caplog.at_level(logging.ERROR, logger="cobra"):
        assert cli._doctor() == EXIT_ERROR

    assert "cobra_not_a_module" in caplog.text


def test_doctor_passes_when_only_optional_pieces_are_missing(monkeypatch, capsys):
    monkeypatch.setattr(diagnostics, "REQUIRED_MODULES", (("json", "the standard library"),))
    monkeypatch.setattr(diagnostics, "REQUIRED_EXECUTABLES", ())
    monkeypatch.setattr(diagnostics, "OPTIONAL_MODULES", (("cobra_not_a_module", "testing"),))
    monkeypatch.setattr(diagnostics, "OPTIONAL_EXECUTABLES", ())

    assert cli._doctor() == EXIT_OK
    assert "cobra_not_a_module" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Argument parsing and dispatch
# ---------------------------------------------------------------------------


def test_parser_exposes_the_documented_commands():
    parser = cli._parser()

    assert parser.parse_args(["run", "cfg.json"]).command == "run"
    assert parser.parse_args(["parse", "x.cir"]).command == "parse"
    assert parser.parse_args(["doctor"]).command == "doctor"
    assert parser.parse_args(["gui"]).command == "gui"
    assert parser.parse_args([]).command is None


@pytest.mark.parametrize("argv", [["-v", "run", "cfg.json"], ["run", "-v", "cfg.json"]])
def test_output_flags_are_accepted_before_or_after_the_command(argv):
    """``cobra -v run CONFIG`` and ``cobra run -v CONFIG`` must behave alike."""
    assert cli._parser().parse_args(argv).verbose == 1


def test_repeated_verbose_flags_accumulate():
    assert cli._parser().parse_args(["-vv", "doctor"]).verbose == 2


def test_version_flag_prints_the_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])

    assert exit_info.value.code == EXIT_OK
    assert "cobra" in capsys.readouterr().out


def test_no_command_opens_the_gui(monkeypatch):
    monkeypatch.setattr(cli, "_launch_gui", lambda: EXIT_OK)
    assert cli.main([]) == EXIT_OK


def test_main_configures_logging_from_the_flags(monkeypatch):
    monkeypatch.setattr(cli, "_doctor", lambda: EXIT_OK)
    cli.main(["-q", "doctor"])

    handlers = logging.getLogger("cobra").handlers
    assert handlers
    assert min(handler.level for handler in handlers) == logging.WARNING


def test_log_file_flag_writes_a_log(monkeypatch, tmp_path: Path):
    log_file = tmp_path / "logs" / "run.log"

    def _doctor():
        logging.getLogger("cobra.test").info("hello from the run")
        return EXIT_OK

    monkeypatch.setattr(cli, "_doctor", _doctor)
    cli.main(["--log-file", str(log_file), "doctor"])
    logging.shutdown()

    assert "hello from the run" in log_file.read_text(encoding="utf-8")
