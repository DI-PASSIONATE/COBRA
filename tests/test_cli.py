"""Tests for the ``cobra`` command-line entry point.

Exit codes are a contract for scripts and CI, and ``cobra parse --json`` must
keep stdout machine-readable, so both are pinned here. ``cobra run`` is never
executed end-to-end (that needs Xyce); only its error mapping is covered.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cobra import __main__ as cli
from cobra.configuration.configuration import ConfigurationError
from tests.conftest import make_config_data, netlist_path

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_INTERRUPTED = 130


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
def test_parse_boundary_maps_exceptions_to_exit_two(monkeypatch, capsys, exception):
    """inspect_path is defensive enough that this handler is hard to reach through
    normal input, so exercise it directly.
    """

    def _raise(*_args, **_kwargs):
        raise exception

    monkeypatch.setattr("cobra.configuration.inspection.inspect_path", _raise)

    args = cli._parser().parse_args(["parse", "whatever.cir"])
    assert cli._parse_target(args) == EXIT_CONFIG
    assert "COBRA error" in capsys.readouterr().err


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
def test_run_maps_failures_to_exit_codes(monkeypatch, exception, expected):
    def _raise(_path):
        raise exception

    monkeypatch.setattr(
        "cobra.configuration.config_runner.run_configuration_file", _raise
    )
    assert cli._run_config("anything.json") == expected


def test_run_returns_zero_and_prints_the_results_directory(monkeypatch, capsys):
    monkeypatch.setattr(
        "cobra.configuration.config_runner.run_configuration_file",
        lambda _path: {"results_dir": "results/2026-01-01_demo"},
    )

    assert cli._run_config("anything.json") == EXIT_OK
    assert "results/2026-01-01_demo" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def test_parser_exposes_the_documented_commands():
    parser = cli._parser()

    assert parser.parse_args(["run", "cfg.json"]).command == "run"
    assert parser.parse_args(["parse", "x.cir"]).command == "parse"


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
