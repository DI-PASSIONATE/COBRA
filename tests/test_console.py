"""Tests for :mod:`cobra.console`.

The package logs; only the CLI prints. These tests pin that split -- records
reach stderr, never stdout -- along with the verbosity mapping, the file log,
and the colour rules the formatter depends on.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest

from cobra import console

MESSAGE = "a message worth logging"


@pytest.fixture
def stream() -> io.StringIO:
    return io.StringIO()


def _log(level: int, message: str = MESSAGE, name: str = "cobra.test") -> None:
    logging.getLogger(name).log(level, message)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_console_level_mapping():
    assert console.console_level(0, False) == logging.INFO
    assert console.console_level(2, False) == logging.DEBUG
    assert console.console_level(1, True) == logging.WARNING  # --quiet wins over --verbose


def test_verbosity_is_applied_to_the_cobra_logger(stream):
    console.configure_logging(stream=stream, color=False)
    _log(logging.INFO, "shown")
    _log(logging.DEBUG, "hidden")

    console.configure_logging(stream=stream, color=False, quiet=True)
    _log(logging.INFO, "quiet info")
    _log(logging.WARNING, "quiet warning")

    output = stream.getvalue()
    assert "shown" in output
    assert "hidden" not in output
    assert "quiet info" not in output
    assert "quiet warning" in output


def test_only_double_verbose_captures_third_party_loggers(stream):
    """-vv attaches the handler to the root logger as well; -v does not."""
    console.configure_logging(stream=stream, color=False, verbose=1)
    _log(logging.DEBUG, "with -v", name="third_party.module")
    console.configure_logging(stream=stream, color=False, verbose=2)
    _log(logging.DEBUG, "with -vv", name="third_party.module")

    assert "with -v\n" not in stream.getvalue()
    assert "with -vv" in stream.getvalue()


def test_reconfiguring_does_not_duplicate_output(stream):
    console.configure_logging(stream=stream, color=False)
    console.configure_logging(stream=stream, color=False)

    _log(logging.WARNING)

    assert stream.getvalue().count(MESSAGE) == 1


def test_records_go_to_stderr_never_stdout(capsys):
    """The handler resolves ``sys.stderr`` per record, so a redirect is honoured."""
    console.configure_logging(color=False)

    _log(logging.ERROR)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert MESSAGE in captured.err


def test_info_is_unprefixed_and_severe_records_are_labelled(stream):
    console.configure_logging(stream=stream, color=False, verbose=1)

    _log(logging.INFO, "plain")
    _log(logging.WARNING, "careful")
    _log(logging.DEBUG, "details")

    assert stream.getvalue().splitlines() == ["plain", "warning: careful", "debug [test] details"]


def test_colour_only_for_a_terminal_without_no_color(monkeypatch, stream):
    class _Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert console.supports_color(_Tty()) is True
    assert console.supports_color(stream) is False  # redirected

    monkeypatch.setenv("NO_COLOR", "1")
    assert console.supports_color(_Tty()) is False


def test_log_file_records_debug_even_when_the_console_is_quiet(tmp_path: Path, stream):
    log_file = tmp_path / "nested" / "cobra.log"
    console.configure_logging(stream=stream, color=False, quiet=True, log_file=log_file)

    _log(logging.DEBUG, "deep detail")
    logging.shutdown()

    contents = log_file.read_text(encoding="utf-8")
    assert "deep detail" in contents
    assert "DEBUG" in contents
    assert "deep detail" not in stream.getvalue()


# ---------------------------------------------------------------------------
# stdout formatting
# ---------------------------------------------------------------------------
