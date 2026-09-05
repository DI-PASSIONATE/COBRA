"""
CobraSetting — metadata descriptor for configurable COBRA class parameters.

Classes that want their settings to be discoverable by the GUI (or other
tooling) should declare a class-level ``_settings: list[CobraSetting]``
attribute.  The GUI reads this list to build form widgets with correct types,
defaults, and tooltips — without requiring manual wiring.

Example::

    from cobra.configuration.setting import CobraSetting

    class MySimulator(BaseSimulator):
        _settings = [
            CobraSetting("xyce_command", str,  "Xyce",  "Path or command name to invoke Xyce."),
            CobraSetting("parallel",     bool, False,   "Run Xyce in parallel using MPI (mpirun). Requires an MPI-enabled Xyce build and mpirun on PATH. WARNING: Usually a lot slower than single-core Xyce for small and medium-sized circuits."),
        ]

        def __init__(self, xyce_command: str = "Xyce", parallel: bool = False):
            ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CobraSetting:
    """
    Metadata for a single configurable parameter of a COBRA component.

    Attributes
    ----------
    name:
        The ``__init__`` (or ``run()``) keyword-argument name this setting maps to.
    dtype:
        Expected Python type (``str``, ``int``, ``float``, ``bool``).
        Used by the GUI to pick the right widget type.
    default:
        Default value shown in the GUI widget.
    description:
        Human-readable explanation shown as a tooltip in the GUI.
    choices:
        Optional list of ``(label, value)`` pairs.  When provided the GUI
        renders a ``QComboBox`` instead of a plain text/spin widget.  The
        ``value`` stored against each item is what gets passed to the
        constructor; ``label`` is the human-readable display text.
    """

    name: str
    dtype: type
    default: Any
    description: str
    choices: list[tuple[str, Any]] | None = None
