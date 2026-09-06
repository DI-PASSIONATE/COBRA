"""Runtime environment checks behind ``cobra doctor``.

Reports what COBRA found -- the interpreter, the packages and simulators a run
requires, and the optional components whose absence only disables a feature.
:func:`check_environment` collects it as data and :func:`render_environment`
turns it into text, mirroring ``inspection.render_report``; printing is left to
the caller.

Apart from :mod:`cobra.console` only the standard library is imported: a broken
or missing dependency must be reportable, not a crash.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import shutil
import sys
from dataclasses import dataclass

from cobra.console import Palette, format_fields, supports_unicode

# (import name, what COBRA needs it for)
REQUIRED_MODULES = (
    ("numpy", "numerics"),
    ("pandas", "simulation output parsing"),
    ("skrf", "S-parameter handling"),
    ("optuna", "the default optimizer"),
    ("tqdm", "progress reporting"),
)
OPTIONAL_MODULES = (
    ("PySide6", "graphical interface"),
    ("pyqtgraph", "live plots in the GUI"),
    ("matplotlib", "result plots"),
    ("onnxruntime", "ONNX surrogate models"),
    ("gmsh", "geometry meshing for EM fine-tuning"),
    ("huggingface_hub", "the model browser"),
    ("orca", "EM fine-tuning geometries"),
)
REQUIRED_EXECUTABLES = (("Xyce", "circuit simulation"),)
OPTIONAL_EXECUTABLES = (
    ("palace", "EM fine-tuning"),
    ("mpirun", "parallel Xyce and Palace runs"),
)


def cobra_version() -> str:
    """The installed COBRA version, or ``"unknown"`` outside an installation."""
    try:
        return importlib.metadata.version("cobra")
    except importlib.metadata.PackageNotFoundError:  # running from a source tree
        return "unknown"


@dataclass(frozen=True)
class Dependency:
    """One checked package or executable."""

    name: str
    purpose: str
    available: bool
    detail: str
    """Version, path, or the reason it is unavailable."""


@dataclass(frozen=True)
class EnvironmentReport:
    version: str
    python: str
    platform: str
    required: list[Dependency]
    optional: list[Dependency]

    @property
    def missing_required(self) -> list[str]:
        return [item.name for item in self.required if not item.available]

    @property
    def missing_optional(self) -> list[str]:
        return [item.name for item in self.optional if not item.available]

    @property
    def ok(self) -> bool:
        """Whether everything a run needs is present."""
        return not self.missing_required


def module_version(name: str) -> str:
    """Version of the distribution providing the module *name*, if it has one."""
    # The import name and the distribution name differ often enough to matter
    # here (skrf ships as scikit-rf), so ask which distribution owns the module.
    distributions = importlib.metadata.packages_distributions().get(name, [name])
    for distribution in distributions:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "installed"


def module_status(name: str) -> tuple[bool, str]:
    """Whether *name* is importable, plus its version, without importing it."""
    try:
        found = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # a broken or partially removed install
        return False, "not usable"
    if not found:
        return False, "not installed"
    return True, module_version(name)


def executable_status(name: str) -> tuple[bool, str]:
    """Whether *name* is on ``PATH``, plus where it was found."""
    path = shutil.which(name)
    return path is not None, path or "not on PATH"


def _check(
    modules: tuple[tuple[str, str], ...], executables: tuple[tuple[str, str], ...]
) -> list[Dependency]:
    checked = [(name, purpose, module_status(name)) for name, purpose in modules]
    checked += [(name, purpose, executable_status(name)) for name, purpose in executables]
    return [
        Dependency(name=name, purpose=purpose, available=available, detail=detail)
        for name, purpose, (available, detail) in checked
    ]


def check_environment() -> EnvironmentReport:
    """Inspect the current interpreter and its surroundings."""
    return EnvironmentReport(
        version=cobra_version(),
        python=f"{sys.version.split()[0]} ({sys.executable})",
        platform=sys.platform,
        required=_check(REQUIRED_MODULES, REQUIRED_EXECUTABLES),
        optional=_check(OPTIONAL_MODULES, OPTIONAL_EXECUTABLES),
    )


def _render_dependencies(
    title: str, dependencies: list[Dependency], *, required: bool, palette: Palette
) -> list[str]:
    """One section of the report: a status mark, the name, the detail, the purpose."""
    if not dependencies:
        return [palette.bold(title), palette.dim("  nothing to check")]

    unicode_ok = supports_unicode(sys.stdout)
    ok_mark = "✓" if unicode_ok else "+"
    missing_mark = ("✗" if unicode_ok else "!") if required else ("·" if unicode_ok else "-")

    name_width = max(len(item.name) for item in dependencies)
    detail_width = max(len(item.detail) for item in dependencies)
    lines = [palette.bold(title)]
    for item in dependencies:
        if item.available:
            mark = palette.green(ok_mark)
        elif required:
            mark = palette.red(missing_mark)
        else:
            mark = palette.dim(missing_mark)
        lines.append(
            f"  {mark} {item.name.ljust(name_width)}  {item.detail.ljust(detail_width)}  "
            f"{palette.dim(item.purpose)}"
        )
    return lines


def render_environment(report: EnvironmentReport, palette: Palette | None = None) -> str:
    """Render *report* as the text ``cobra doctor`` prints."""
    palette = Palette() if palette is None else palette
    lines = [palette.bold(f"COBRA {report.version}")]
    lines += format_fields([("python", report.python), ("platform", report.platform)], palette)
    lines.append("")
    lines += _render_dependencies("Required", report.required, required=True, palette=palette)
    lines.append("")
    lines += _render_dependencies("Optional", report.optional, required=False, palette=palette)
    return "\n".join(lines)
