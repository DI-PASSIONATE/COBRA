from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cobra",
        description="COBRA: A Circuit-Level Open-Source Based RFIC AI-Assisted Optimizer",
    )
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="Run a saved JSON configuration")
    run_parser.add_argument("config", help="Path to a COBRA JSON configuration")
    parse_parser = subparsers.add_parser(
        "parse",
        help="Report the contents of a JSON configuration or a netlist without running it",
    )
    parse_parser.add_argument("target", help="Path to a COBRA JSON configuration or a netlist")
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
    return parser


def _print_dependency_status() -> None:
    """Report optional runtime dependencies without preventing COBRA startup."""
    print("COBRA dependency status:")
    try:
        importlib.import_module("orca")
    except ModuleNotFoundError as exc:
        print(f"  ORCA: missing ({exc.name or 'orca'} is not installed)")
        print("    EM fine-tuning is disabled; all other stages run normally.")
    except Exception as exc:  # noqa: BLE001 - a broken ORCA install must not stop startup
        print(f"  ORCA: unavailable ({type(exc).__name__}: {exc})")
        print("    EM fine-tuning is disabled; all other stages run normally.")
    else:
        print("  ORCA: found")


def _run_config(path: str) -> int:
    from cobra.configuration import ConfigurationError
    from cobra.configuration.config_runner import run_configuration_file

    try:
        context = run_configuration_file(path)
    except (ConfigurationError, FileNotFoundError, OSError) as exc:
        print(f">> COBRA error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(">> COBRA: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report and exit non-zero
        print(f">> COBRA: run failed: {exc}", file=sys.stderr)
        return 1

    print(f"Results: {context.get('results_dir', 'unknown')}")
    return 0


def _parse_target(args: argparse.Namespace) -> int:
    """Print a report for a configuration or netlist; return 2 when it has errors."""
    import contextlib
    import json

    from cobra.configuration import ConfigurationError
    from cobra.configuration.inspection import has_errors, inspect_path, render_report

    try:
        # Model loaders and the netlist parser print to stdout; keep the report alone there.
        with contextlib.redirect_stdout(sys.stderr):
            report = inspect_path(args.target, kind=args.kind, check_models=args.check_models)
    except (ConfigurationError, OSError) as exc:
        print(f">> COBRA error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_report(report, full=args.full))
    return 2 if has_errors(report) else 0


def main(argv: list[str] | None = None) -> int:
    """Launch the GUI, execute a saved configuration, or report on an input file."""
    args = _parser().parse_args(argv)
    if args.command == "parse":
        # Keep the report the only thing on stdout so it stays machine-readable.
        return _parse_target(args)
    _print_dependency_status()
    print(f"Starting COBRA V{importlib.metadata.version('cobra')}...")
    if args.command == "run":
        return _run_config(args.config)

    from cobra.gui.app import run_gui

    run_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
