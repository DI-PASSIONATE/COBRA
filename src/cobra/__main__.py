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
    return parser


def _print_dependency_status() -> None:
    """Report optional runtime dependencies without preventing COBRA startup."""
    print("COBRA dependency status:")
    try:
        importlib.import_module("orca")
    except ModuleNotFoundError as exc:
        print(f"  ORCA: missing ({exc.name or 'orca'} is not installed)")
        print("    EM fine-tuning is disabled; all other stages run normally.")
    except Exception as exc:
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
    except Exception as exc:
        print(f">> COBRA: run failed: {exc}", file=sys.stderr)
        return 1

    print(f"Results: {context.get('results_dir', 'unknown')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Launch the GUI or execute a saved configuration."""
    args = _parser().parse_args(argv)
    _print_dependency_status()
    print(f"Starting COBRA V{importlib.metadata.version('cobra')}...")
    if args.command == "run":
        return _run_config(args.config)

    from cobra.gui.app import run_gui

    run_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
