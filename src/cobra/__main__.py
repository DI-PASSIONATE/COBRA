from __future__ import annotations

import argparse
import importlib
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
    for package_name, import_name in (("torch", "torch"), ("ORCA", "orca")):
        try:
            importlib.import_module(import_name)
        except ModuleNotFoundError as exc:
            missing_name = exc.name or package_name
            print(f"  {package_name}: missing ({missing_name} is not installed)")
        except Exception as exc:
            print(f"  {package_name}: unavailable ({type(exc).__name__}: {exc})")
        else:
            print(f"  {package_name}: found")


def _run_config(path: str) -> int:
    from cobra.config_runner import run_configuration_file
    from cobra.configuration import ConfigurationError

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
