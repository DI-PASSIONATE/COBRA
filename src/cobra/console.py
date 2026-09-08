"""Terminal presentation for COBRA: stdout formatting, colour, and logging.

This module owns everything COBRA writes to a terminal. Commands compose their
output from :func:`write`, :func:`write_heading` and :func:`write_fields`, which
go to stdout; diagnostics go to stderr through the handlers installed by
:func:`configure_logging`.

Library code never writes to stdout.  Every module obtains a logger with
``logging.getLogger(__name__)`` and the entry points -- the ``cobra`` CLI and
:func:`cobra.gui.app.run_gui` -- install the handlers once through
:func:`configure_logging`.  Diagnostics therefore land on stderr, which keeps
stdout free for the output a caller asked for (``cobra parse --json`` is piped
into other tools).

Nothing here imports Qt, ORCA or a simulator, so the module stays importable in
a headless environment.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, ClassVar

import tqdm

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER_NAME = "cobra"
"""Root of the COBRA logger tree; every module logger is a child of it."""

_COBRA_HANDLER = "_cobra_handler"
"""Marker attribute set on handlers installed here, so reconfiguring only ever
removes COBRA's own handlers and leaves those of a host application alone."""

_FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


class Ansi:
    """The handful of SGR escapes used by the CLI."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"


def supports_color(stream: IO[str] | None = None) -> bool:
    """Whether ANSI escapes are safe to write to *stream*.

    Honours the ``NO_COLOR`` and ``FORCE_COLOR`` conventions before falling
    back to a TTY check, so redirected output stays plain text.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    stream = sys.stdout if stream is None else stream
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):  # detached or closed stream
        return False


def supports_unicode(stream: IO[str] | None = None) -> bool:
    """Whether *stream* can encode the status glyphs used in reports."""
    stream = sys.stdout if stream is None else stream
    encoding = getattr(stream, "encoding", None) or "ascii"
    try:
        "✓✗".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


@dataclass(frozen=True)
class Palette:
    """Colour helpers that collapse to plain text when colour is disabled."""

    enabled: bool = False

    @classmethod
    def for_stream(cls, stream: IO[str] | None = None, *, enabled: bool | None = None) -> Palette:
        """Palette for *stream*; *enabled* overrides the auto-detection."""
        return cls(supports_color(stream) if enabled is None else enabled)

    def paint(self, text: str, *codes: str) -> str:
        if not self.enabled or not codes:
            return text
        return f"{''.join(codes)}{text}{Ansi.RESET}"

    def bold(self, text: str) -> str:
        return self.paint(text, Ansi.BOLD)

    def dim(self, text: str) -> str:
        return self.paint(text, Ansi.DIM)

    def red(self, text: str) -> str:
        return self.paint(text, Ansi.RED)

    def green(self, text: str) -> str:
        return self.paint(text, Ansi.GREEN)

    def yellow(self, text: str) -> str:
        return self.paint(text, Ansi.YELLOW)

    def cyan(self, text: str) -> str:
        return self.paint(text, Ansi.CYAN)


# ---------------------------------------------------------------------------
# stdout
# ---------------------------------------------------------------------------


def write(text: str = "") -> None:
    """Write one line of requested output to stdout."""
    # Flushed line by line: stdout is block-buffered when redirected, and a
    # report would otherwise appear after the diagnostics logged to stderr.
    # T201 is deliberate here: this function is the package's only stdout writer,
    # and the rule stays on everywhere else so library code cannot start printing.
    print(text, flush=True)  # noqa: T201


def write_heading(text: str, palette: Palette) -> None:
    """Write a section heading, e.g. ``Summary``."""
    write(palette.bold(text))


def format_fields(rows: Sequence[tuple[str, str]], palette: Palette) -> list[str]:
    """Aligned ``label  value`` lines, skipping rows without a value."""
    filled = [(label, value) for label, value in rows if value]
    if not filled:
        return []
    width = max(len(label) for label, _ in filled)
    return [f"  {palette.dim(label.ljust(width))}  {value}" for label, value in filled]


def write_fields(rows: Sequence[tuple[str, str]], palette: Palette) -> None:
    """Write the lines of :func:`format_fields` to stdout."""
    for line in format_fields(rows, palette):
        write(line)


def write_rule(palette: Palette, width: int = 60, stream: IO[str] | None = None) -> None:
    """Write a horizontal rule, used to separate report sections."""
    glyph = "─" if supports_unicode(stream) else "-"
    write(palette.dim(glyph * width))


#: Track glyphs for :func:`format_slider`: (left cap, fill, marker, right cap).
_SLIDER_GLYPHS = ("├", "─", "┃", "┤")
_SLIDER_GLYPHS_ASCII = ("[", "-", "|", "]")


def format_slider(
    minimum: float,
    maximum: float,
    value: float | None,
    *,
    width: int = 24,
    palette: Palette | None = None,
    stream: IO[str] | None = None,
) -> str:
    """Render *value* as a marker on a ``minimum``-to-``maximum`` track.

    ``├────────┃───────────┤`` — the position of the marker within the track is
    where the parameter currently sits in its allowed range. A *value* outside
    the range is clamped to the nearest end rather than dropped, so a bad range
    stays visible. Returns an empty track when *value* is ``None``.
    """
    left, fill, marker, right = (
        _SLIDER_GLYPHS if supports_unicode(stream) else _SLIDER_GLYPHS_ASCII
    )
    width = max(width, 3)
    track = [fill] * width

    if value is not None:
        span = maximum - minimum
        # A degenerate range (a pinned parameter) has only one place to be.
        fraction = 0.0 if span <= 0 else (value - minimum) / span
        index = round(min(max(fraction, 0.0), 1.0) * (width - 1))
        track[index] = marker if palette is None else palette.cyan(marker)

    return f"{left}{''.join(track)}{right}"


def format_duration(seconds: float) -> str:
    """Render *seconds* as ``12.3s``, ``4m 12s`` or ``1h 03m``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


@dataclass(frozen=True)
class SliderSpec:
    """One optimization parameter, as much of it as the terminal needs."""

    name: str
    minimum: float
    maximum: float
    unit: str | None = None
    linked_to: str | None = None
    """Master parameter this one follows, if any; a linked parameter has no
    range of its own to place a marker in."""


def format_parameter_line(
    spec: SliderSpec,
    value: float | None,
    *,
    palette: Palette | None = None,
    width: int = 24,
    label_width: int = 24,
) -> str:
    """One ``name  min ├──┃──┤ max  value`` line for a parameter."""
    current = "—" if value is None else f"{value:g}{spec.unit or ''}"
    label = spec.name[:label_width].ljust(label_width)
    low = f"{spec.minimum:g}".rjust(8)
    high = f"{spec.maximum:g}".ljust(8)

    if spec.linked_to:
        # Its value is dictated by the master, so its own range says nothing:
        # showing a marker in it would invent a degree of freedom it lacks.
        track = f"follows {spec.linked_to}".center(width + 2)
        low = " " * 8
        high = " " * 8
    else:
        track = format_slider(spec.minimum, spec.maximum, value, width=width, palette=palette)

    if palette is not None:
        label = palette.bold(label)
        low, high = palette.dim(low), palette.dim(high)
        current = palette.cyan(current)
        if spec.linked_to:
            track = palette.dim(track)
    return f"{label} {low} {track} {high} {current}"


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


class ConsoleFormatter(logging.Formatter):
    """One line per record: ``level: message``, with INFO left unprefixed.

    INFO is the running commentary of a normal run and reads best as plain
    prose; anything more severe is labelled so it stands out next to the
    progress bar, and DEBUG additionally names the module it came from.
    """

    LABELS: ClassVar[dict[int, tuple[str, str]]] = {
        logging.WARNING: ("warning", Ansi.YELLOW),
        logging.ERROR: ("error", Ansi.RED),
        logging.CRITICAL: ("critical", Ansi.BOLD + Ansi.RED),
    }

    def __init__(self, palette: Palette | None = None):
        super().__init__("%(message)s")
        self.palette = palette if palette is not None else Palette()

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if record.levelno <= logging.DEBUG:
            origin = record.name.removeprefix(f"{LOGGER_NAME}.")
            return self.palette.dim(f"debug [{origin}] {text}")
        label = self.LABELS.get(record.levelno)
        if label is None:
            return text
        name, color = label
        return f"{self.palette.paint(f'{name}:', color)} {text}"


class TqdmStreamHandler(logging.StreamHandler):
    """Stream handler that writes through :func:`tqdm.write`.

    ``tqdm`` owns the last line of the terminal during a run; writing around it
    with :func:`tqdm.write` keeps the progress bar intact instead of leaving a
    trail of half-erased bars between log lines.

    Without an explicit *stream* the handler looks up ``sys.stderr`` for every
    record instead of holding on to the object installed at configuration time,
    so a later redirection -- a test harness, a captured subprocess -- is
    honoured rather than written into a stream that is already closed.
    """

    def __init__(self, stream: IO[str] | None = None):
        self._follow_stderr = stream is None
        super().__init__(sys.stderr if stream is None else stream)

    @property
    def target(self) -> IO[str]:
        """The stream this handler writes to right now."""
        return sys.stderr if self._follow_stderr else self.stream

    def emit(self, record: logging.LogRecord) -> None:
        try:
            stream = self.target
            tqdm.tqdm.write(self.format(record), file=stream)
            stream.flush()
        except RecursionError:
            raise
        except Exception:  # noqa: BLE001 - logging must never break the caller
            self.handleError(record)


def console_level(verbose: int = 0, quiet: bool = False) -> int:
    """Map the CLI's verbosity flags onto a logging level."""
    if quiet:
        return logging.WARNING
    if verbose >= 1:
        return logging.DEBUG
    return logging.INFO


def configure_logging(
    *,
    verbose: int = 0,
    quiet: bool = False,
    log_file: str | Path | None = None,
    stream: IO[str] | None = None,
    color: bool | None = None,
) -> logging.Logger:
    """Install COBRA's log handlers and return the package logger.

    Console output goes to *stream*, or to whatever ``sys.stderr`` is at the
    time each record is written, at the level implied by *verbose* and *quiet*.  A *log_file* additionally receives every record at
    DEBUG level with timestamps, so a long run can be diagnosed afterwards even
    when the console was quiet.  ``verbose >= 2`` also surfaces third-party
    loggers (optuna, scikit-rf, matplotlib).

    Calling this again replaces the handlers installed by an earlier call, so
    entry points can reconfigure without duplicating every line.
    """
    level = console_level(verbose, quiet)
    palette = Palette.for_stream(sys.stderr if stream is None else stream, enabled=color)

    logger = logging.getLogger(LOGGER_NAME)
    root = logging.getLogger()
    for target in (logger, root):
        _remove_cobra_handlers(target)

    console = TqdmStreamHandler(stream)
    console.setLevel(level)
    console.setFormatter(ConsoleFormatter(palette))
    _mark(console)
    logger.addHandler(console)

    levels = [level]
    if log_file is not None:
        path = Path(log_file)
        if path.parent != Path():
            path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        _mark(file_handler)
        logger.addHandler(file_handler)
        levels.append(logging.DEBUG)

    logger.setLevel(min(levels))
    # COBRA records stop here; the root handlers below are for other libraries.
    logger.propagate = False

    if verbose >= 2:
        for handler in logger.handlers:
            root.addHandler(handler)
        root.setLevel(min(levels))

    return logger


def ensure_logging(**kwargs) -> logging.Logger:
    """Configure logging unless an entry point already did.

    Used by :func:`cobra.gui.app.run_gui`, which can be reached either through
    the CLI (already configured, possibly with ``-v``) or directly.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if any(getattr(handler, _COBRA_HANDLER, False) for handler in logger.handlers):
        return logger
    return configure_logging(**kwargs)


def _mark(handler: logging.Handler) -> None:
    setattr(handler, _COBRA_HANDLER, True)


def _remove_cobra_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, _COBRA_HANDLER, False):
            logger.removeHandler(handler)
            if isinstance(handler, logging.FileHandler):
                handler.close()
