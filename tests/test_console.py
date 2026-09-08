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
# Verbosity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verbose", "quiet", "expected"),
    [
        (0, False, logging.INFO),
        (1, False, logging.DEBUG),
        (2, False, logging.DEBUG),
        (0, True, logging.WARNING),
        (1, True, logging.WARNING),  # --quiet wins over --verbose
    ],
)
def test_console_level_mapping(verbose, quiet, expected):
    assert console.console_level(verbose, quiet) == expected


def test_default_configuration_shows_info_but_not_debug(stream):
    console.configure_logging(stream=stream, color=False)

    _log(logging.INFO, "shown")
    _log(logging.DEBUG, "hidden")

    output = stream.getvalue()
    assert "shown" in output
    assert "hidden" not in output


def test_quiet_hides_info(stream):
    console.configure_logging(stream=stream, color=False, quiet=True)

    _log(logging.INFO)
    _log(logging.WARNING, "still shown")

    output = stream.getvalue()
    assert MESSAGE not in output
    assert "still shown" in output


def test_double_verbose_also_captures_third_party_loggers(stream):
    """-vv attaches the handler to the root logger as well.

    Libraries that switch off propagation themselves (optuna does) stay silent
    either way; this covers the ordinary case.
    """
    console.configure_logging(stream=stream, color=False, verbose=2)

    _log(logging.DEBUG, "from a library", name="third_party.module")

    assert "from a library" in stream.getvalue()


def test_single_verbose_leaves_third_party_loggers_alone(stream):
    console.configure_logging(stream=stream, color=False, verbose=1)

    _log(logging.DEBUG, "from a library", name="third_party.module")

    assert "from a library" not in stream.getvalue()


# ---------------------------------------------------------------------------
# Handler management
# ---------------------------------------------------------------------------


def test_reconfiguring_does_not_duplicate_output(stream):
    console.configure_logging(stream=stream, color=False)
    console.configure_logging(stream=stream, color=False)

    _log(logging.WARNING)

    assert stream.getvalue().count(MESSAGE) == 1


def test_configuring_leaves_foreign_handlers_in_place(stream):
    foreign = logging.StreamHandler(io.StringIO())
    logging.getLogger(console.LOGGER_NAME).addHandler(foreign)

    console.configure_logging(stream=stream, color=False)

    assert foreign in logging.getLogger(console.LOGGER_NAME).handlers


def test_ensure_logging_keeps_an_existing_configuration(stream):
    console.configure_logging(stream=stream, color=False, quiet=True)
    console.ensure_logging(stream=io.StringIO(), color=False)

    _log(logging.INFO)

    assert MESSAGE not in stream.getvalue()


def test_ensure_logging_configures_when_nothing_is_installed():
    logger = console.ensure_logging(stream=io.StringIO(), color=False)

    assert logger.handlers


def test_records_never_reach_stdout(capsys, stream):
    console.configure_logging(stream=stream, color=False)

    _log(logging.ERROR)

    assert capsys.readouterr().out == ""
    assert MESSAGE in stream.getvalue()


def test_without_an_explicit_stream_records_follow_stderr(capsys):
    """The handler resolves ``sys.stderr`` per record, so a redirect is honoured."""
    console.configure_logging(color=False)

    _log(logging.WARNING)

    assert MESSAGE in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def test_info_is_unprefixed_and_severe_records_are_labelled(stream):
    console.configure_logging(stream=stream, color=False, verbose=1)

    _log(logging.INFO, "plain")
    _log(logging.WARNING, "careful")
    _log(logging.ERROR, "broken")
    _log(logging.DEBUG, "details")

    lines = stream.getvalue().splitlines()
    assert lines[0] == "plain"
    assert lines[1] == "warning: careful"
    assert lines[2] == "error: broken"
    assert lines[3] == "debug [test] details"


def test_colour_is_applied_only_when_enabled():
    coloured = io.StringIO()
    plain = io.StringIO()

    console.configure_logging(stream=coloured, color=True)
    _log(logging.ERROR)
    console.configure_logging(stream=plain, color=False)
    _log(logging.ERROR)

    assert "\033[" in coloured.getvalue()
    assert "\033[" not in plain.getvalue()


def test_palette_paints_only_when_enabled():
    assert console.Palette(enabled=False).red("x") == "x"
    assert console.Palette(enabled=True).red("x") == f"{console.Ansi.RED}x{console.Ansi.RESET}"


def test_no_color_environment_variable_disables_colour(monkeypatch):
    class _Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert console.supports_color(_Tty()) is True

    monkeypatch.setenv("NO_COLOR", "1")
    assert console.supports_color(_Tty()) is False


def test_a_redirected_stream_gets_no_colour(monkeypatch, stream):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert console.supports_color(stream) is False


# ---------------------------------------------------------------------------
# File logging
# ---------------------------------------------------------------------------


def test_log_file_records_debug_even_when_the_console_is_quiet(tmp_path: Path, stream):
    log_file = tmp_path / "nested" / "cobra.log"
    console.configure_logging(stream=stream, color=False, quiet=True, log_file=log_file)

    _log(logging.DEBUG, "deep detail")
    logging.shutdown()

    contents = log_file.read_text(encoding="utf-8")
    assert "deep detail" in contents
    assert "DEBUG" in contents
    assert "cobra.test" in contents
    assert "deep detail" not in stream.getvalue()


# ---------------------------------------------------------------------------
# stdout formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0, "0.0s"), (9.5, "9.5s"), (90.0, "1m 30s"), (3660.0, "1h 01m")],
)
def test_duration_formatting(seconds, expected):
    assert console.format_duration(seconds) == expected


def test_fields_are_aligned_and_empty_values_dropped():
    lines = console.format_fields(
        [("netlist", "amp.cir"), ("fine-tuning", ""), ("goals", "2")], console.Palette()
    )

    assert lines == ["  netlist  amp.cir", "  goals    2"]


def test_fields_render_nothing_when_every_value_is_empty():
    assert console.format_fields([("status", "")], console.Palette()) == []


def test_write_goes_to_stdout(capsys):
    console.write("a line")
    console.write_heading("Summary", console.Palette())

    out = capsys.readouterr().out
    assert out == "a line\nSummary\n"


# ---------------------------------------------------------------------------
# parameter sliders
# ---------------------------------------------------------------------------


def _marker_index(bar: str) -> int:
    """Position of the marker within the track, ignoring the end caps."""
    return bar[1:-1].index("┃")


def test_the_slider_marks_where_the_value_sits_in_its_range():
    width = 21
    low = console.format_slider(0.0, 100.0, 0.0, width=width)
    middle = console.format_slider(0.0, 100.0, 50.0, width=width)
    high = console.format_slider(0.0, 100.0, 100.0, width=width)

    assert _marker_index(low) == 0
    assert _marker_index(middle) == width // 2
    assert _marker_index(high) == width - 1


def test_a_value_outside_the_range_is_clamped_rather_than_dropped():
    assert _marker_index(console.format_slider(0.0, 10.0, -5.0, width=11)) == 0
    assert _marker_index(console.format_slider(0.0, 10.0, 99.0, width=11)) == 10


def test_a_pinned_parameter_does_not_divide_by_zero():
    assert _marker_index(console.format_slider(5.0, 5.0, 5.0, width=11)) == 0


def test_a_missing_value_leaves_the_track_empty():
    assert "┃" not in console.format_slider(0.0, 10.0, None, width=11)


def test_a_linked_parameter_shows_its_master_instead_of_a_track():
    spec = console.SliderSpec(name="C4", minimum=0.0, maximum=0.0, linked_to="C3")
    line = console.format_parameter_line(spec, 19.0, width=20, label_width=4)

    assert "follows C3" in line
    assert "┃" not in line
    assert "19" in line
