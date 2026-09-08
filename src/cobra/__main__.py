"""Command-line entry point for COBRA.

Output contract: stdout carries only what was asked for -- the ``parse``
report, the ``run`` summary, the ``doctor`` table -- so ``cobra parse --json``
stays pipeable.  Progress and diagnostics go to stderr through
:mod:`cobra.console`, which also owns the stdout formatting used here.

Exit codes are part of the interface and are pinned by the test suite:
``0`` success, ``1`` a run that started and failed, ``2`` invalid input or a
report containing errors, ``130`` interrupted.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from cobra.console import (
    Palette,
    configure_logging,
    format_duration,
    write,
    write_fields,
    write_heading,
)
from cobra.diagnostics import cobra_version

if TYPE_CHECKING:
    from cobra.configuration.config_runner import ConfiguredRun
    from cobra.optimization_context import OptimizationContext

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INVALID = 2
EXIT_INTERRUPTED = 130

_EPILOG = """\
examples:
  cobra                              open the graphical interface
  cobra parse design.cir             report what COBRA reads from a netlist
  cobra parse config.json --json     the same report as JSON, on stdout
  cobra parse config.json && cobra run config.json
  cobra run config.json -v --log-file run.log
  cobra doctor                       check simulators and optional packages

exit codes:
  0  success                         2  invalid input, or a report with errors
  1  the run failed                130  interrupted
"""


def _add_output_arguments(parser: argparse.ArgumentParser, *, inherit: bool = False) -> None:
    """Add the flags shared by every command.

    Subcommands repeat them with ``SUPPRESS`` defaults so that ``cobra -v run``
    and ``cobra run -v`` both work without the subparser resetting the value
    that the top-level parser already stored.
    """
    default = argparse.SUPPRESS if inherit else None
    group = parser.add_argument_group("output")
    group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0 if not inherit else default,
        help="Show debug output; repeat (-vv) to include third-party libraries",
    )
    group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=False if not inherit else default,
        help="Report warnings and errors only",
    )
    group.add_argument(
        "--log-file",
        metavar="PATH",
        default=None if not inherit else default,
        help="Also write a timestamped debug log to PATH",
    )
    group.add_argument(
        "--no-color",
        action="store_true",
        default=False if not inherit else default,
        help="Disable coloured output (also honours the NO_COLOR variable)",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cobra",
        description=(
            "COBRA - A Circuit-Level Open-Source Based RFIC AI-Assisted Optimizer.\n"
            "Run without a command to open the graphical interface."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"cobra {cobra_version()}")
    _add_output_arguments(parser)
    subparsers = parser.add_subparsers(dest="command", title="commands", metavar="COMMAND")

    run_parser = subparsers.add_parser(
        "run",
        help="Run a saved JSON configuration",
        description="Run a saved JSON configuration without opening the GUI.",
    )
    run_parser.add_argument("config", metavar="CONFIG", help="Path to a COBRA JSON configuration")
    _add_output_arguments(run_parser, inherit=True)

    parse_parser = subparsers.add_parser(
        "parse",
        help="Report the contents of a JSON configuration or a netlist",
        description=(
            "Report what COBRA reads from a configuration or a netlist without "
            "running it. Exits with 2 when the report contains an error."
        ),
    )
    parse_parser.add_argument(
        "target", metavar="TARGET", help="Path to a COBRA JSON configuration or a netlist"
    )
    parse_parser.add_argument(
        "--kind",
        choices=("auto", "config", "netlist"),
        default="auto",
        help="How to read the target file (default: auto, by suffix and content)",
    )
    parse_parser.add_argument(
        "--json", action="store_true", help="Print the report as JSON instead of text"
    )
    parse_parser.add_argument(
        "--full", action="store_true", help="Print long lists in full instead of truncating them"
    )
    parse_parser.add_argument(
        "--no-model-check",
        dest="check_models",
        action="store_false",
        help="Skip opening ONNX and Touchstone model files (faster, fewer checks)",
    )
    _add_output_arguments(parse_parser, inherit=True)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check simulators and optional packages",
        description=(
            "Report the runtime environment COBRA found. Exits with 1 when a "
            "required dependency is missing."
        ),
    )
    _add_output_arguments(doctor_parser, inherit=True)

    gui_parser = subparsers.add_parser(
        "gui",
        help="Open the graphical interface (the default with no command)",
        description="Open the graphical interface.",
    )
    _add_output_arguments(gui_parser, inherit=True)

    return parser


# ---------------------------------------------------------------------------
# cobra run
# ---------------------------------------------------------------------------


def _run_header(configured: ConfiguredRun, palette: Palette) -> None:
    configuration = configured.configuration
    fine_tuning = configuration.fine_tuning
    write_heading(f"COBRA {cobra_version()}", palette)
    write_fields(
        [
            ("netlist", Path(configuration.netlist).name),
            ("analysis", configured.parser.simulation_type.value),
            ("optimizer", configuration.optimizer.name),
            ("simulator", configuration.simulator.name),
            ("parameters", str(len(configured.optimization_parameters))),
            ("goals", str(len(configured.design_goals))),
            ("iterations", f"up to {configuration.max_iterations}"),
            (
                "fine-tuning",
                f"{fine_tuning.palace_command} ({fine_tuning.iterations} iterations)"
                if fine_tuning.enabled
                else "",
            ),
        ],
        palette,
    )
    write()


def _run_summary(context: OptimizationContext, palette: Palette) -> None:
    iteration = context.iteration
    if context.goal_achieved:
        status = palette.green(f"design goals achieved at iteration {iteration}")
    else:
        status = palette.yellow(f"design goals not achieved after {iteration} iterations")

    total = context.times.get("total_time") or 0.0
    write()
    write_heading("Summary", palette)
    write_fields(
        [
            ("status", status),
            ("wall time", format_duration(total) if total else ""),
            ("results", context.results_dir),
        ],
        palette,
    )


def _run_config(path: str) -> int:
    """Execute a saved configuration; return the process exit code."""
    from cobra.configuration import ConfigurationError, RunConfiguration
    from cobra.configuration.config_runner import build_configured_run

    palette = Palette.for_stream(sys.stdout)
    try:
        configured = build_configured_run(RunConfiguration.load(path))
        _run_header(configured, palette)
        context = configured.run()
    except (ConfigurationError, FileNotFoundError, OSError) as exc:
        # A bad path or configuration is user input, not a COBRA crash: report it as
        # one line and leave the traceback to -v.
        logger.error("%s", exc)  # noqa: TRY400
        logger.debug("Traceback for the failure above", exc_info=exc)
        return EXIT_INVALID
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return EXIT_INTERRUPTED
    except Exception as exc:  # CLI boundary: report and exit non-zero
        logger.error("Run failed: %s", exc)  # noqa: TRY400
        logger.debug("Traceback for the failure above", exc_info=exc)
        logger.info("Re-run with -v for the traceback, or `cobra doctor` to check the environment")
        return EXIT_FAILED

    _run_summary(context, palette)
    return EXIT_OK


# ---------------------------------------------------------------------------
# cobra parse
# ---------------------------------------------------------------------------


def _parse_target(args: argparse.Namespace) -> int:
    """Print a report for a configuration or netlist; return 2 when it has errors."""
    import json

    from cobra.configuration import ConfigurationError
    from cobra.configuration.inspection import has_errors, inspect_path, render_report

    try:
        report = inspect_path(args.target, kind=args.kind, check_models=args.check_models)
    except (ConfigurationError, OSError) as exc:
        logger.error("%s", exc)  # noqa: TRY400 - an unreadable target is user input, not a crash
        logger.debug("Traceback for the failure above", exc_info=exc)
        return EXIT_INVALID

    if args.json:
        write(json.dumps(report.to_dict(), indent=2))
    else:
        write(render_report(report, full=args.full))
    return EXIT_INVALID if has_errors(report) else EXIT_OK


# ---------------------------------------------------------------------------
# cobra doctor
# ---------------------------------------------------------------------------


def _doctor() -> int:
    """Report the runtime environment; return 1 when something required is missing."""
    from cobra.diagnostics import check_environment, render_environment

    report = check_environment()
    write(render_environment(report, Palette.for_stream(sys.stdout)))
    write()

    if not report.ok:
        logger.error("Missing required dependencies: %s", ", ".join(report.missing_required))
        return EXIT_FAILED
    if report.missing_optional:
        logger.info("Some optional components are missing; the features that use them are disabled")
    return EXIT_OK


# ---------------------------------------------------------------------------
# cobra gui
# ---------------------------------------------------------------------------


def _launch_gui() -> int:
    logger.info("Starting the COBRA %s graphical interface", cobra_version())
    try:
        from cobra.gui.app import run_gui
    except ImportError as exc:
        logger.error("The graphical interface is unavailable: %s", exc)  # noqa: TRY400
        logger.debug("Traceback for the failure above", exc_info=exc)
        logger.info(
            "Install the GUI dependencies (uv sync), or use `cobra run CONFIG` headlessly"
        )
        return EXIT_FAILED
    run_gui()
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Launch the GUI, execute a saved configuration, or report on an input file."""
    args = _parser().parse_args(argv)
    configure_logging(
        verbose=getattr(args, "verbose", 0),
        quiet=getattr(args, "quiet", False),
        log_file=getattr(args, "log_file", None),
        color=False if getattr(args, "no_color", False) else None,
    )

    if args.command == "parse":
        return _parse_target(args)
    if args.command == "run":
        return _run_config(args.config)
    if args.command == "doctor":
        return _doctor()
    return _launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())
