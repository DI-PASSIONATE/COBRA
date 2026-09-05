"""
SimulationType – a lightweight enum that identifies the primary analysis directive
found in a SPICE netlist (.AC, .HB, .TRAN, .DC, …), plus a simulator-agnostic
metadata container.

Simulator-specific knowledge (parameter names, descriptions, defaults, options
categories) lives in each concrete simulator class, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SimulationType(Enum):
    """
    Primary simulation mode inferred from the netlist directives.

    In Xyce/SPICE, ``.AC`` performs a small-signal AC frequency sweep.
    ``.LIN`` is a *post-processing* directive that sits alongside ``.AC`` and
    instructs Xyce to extract S-parameters and write a Touchstone (``.s*p``)
    file — it is not a separate simulation.  Both directives map to
    ``SimulationType.AC`` here, since they share the same sweep parameters
    and produce compatible outputs.

    ``HB``  — Harmonic Balance (periodic steady-state, non-linear).
    ``TRAN``— Transient time-domain simulation.
    ``DC``  — DC operating-point sweep.
    """

    AC      = ".AC"    # AC frequency sweep; COBRA always pairs this with .LIN for S-parameter (Touchstone) output
    HB      = ".HB"    # Harmonic Balance
    TRAN    = ".TRAN"  # Transient
    DC      = ".DC"    # DC sweep
    UNKNOWN = "unknown"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_directive(cls, directive: str) -> SimulationType:
        """
        Convert a raw dot-directive string (e.g. ``.AC``, ``.tran``) to the
        corresponding ``SimulationType``.  Both ``.AC`` and ``.LIN`` map to
        ``SimulationType.AC`` because ``.LIN`` is a post-processor on ``.AC``,
        not a separate simulation.  Unrecognised directives map to ``UNKNOWN``.
        """
        normalized = directive.strip().upper()
        _map: dict[str, SimulationType] = {
            ".LIN":  cls.AC,   # .LIN post-processes .AC results; treat as the same type
            ".AC":   cls.AC,
            ".HB":   cls.HB,
            ".TRAN": cls.TRAN,
            ".DC":   cls.DC,
        }
        return _map.get(normalized, cls.UNKNOWN)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    @property
    def display_name(self) -> str:
        """Short label shown in the GUI (e.g. ``".AC"``, ``"unknown"``)."""
        return self.value

    # ------------------------------------------------------------------
    # Parameter discovery (catalogue-based, simulator-agnostic)
    # ------------------------------------------------------------------

    def available_parameters(self, num_ports: int) -> list[str]:
        from cobra.optimizers.design_goal_collection import (
            get_available_parameters,  # lazy – avoids circular import
        )
        return [p.name for p in get_available_parameters(num_ports, simulation_type=self)]

    @classmethod
    def for_parameter(cls, param_name: str) -> SimulationType:
        from cobra.optimizers.design_goal_collection import (
            find_parameter,  # lazy – avoids circular import
        )
        p = find_parameter(param_name)
        return p.simulation_type if p is not None else cls.UNKNOWN

    @classmethod
    def all_available_parameters(cls, num_ports: int) -> list[str]:
        from cobra.optimizers.design_goal_collection import (
            get_available_parameters,  # lazy – avoids circular import
        )
        return [p.name for p in get_available_parameters(num_ports)]


# ---------------------------------------------------------------------------
# SimulationTypeMetadata — simulator-specific knowledge about a SimulationType
# ---------------------------------------------------------------------------

@dataclass
class SimulationTypeMetadata:
    """
    All simulator-specific information for one ``SimulationType``.

    Instances are produced by ``BaseSimulator.get_simulation_metadata()`` and
    its subclass overrides.  Using a dataclass keeps the contract explicit and
    makes it easy for future simulators (ngspice, Spectre, …) to fill in their
    own values.
    """

    #: Named slots for the positional tokens that follow the directive keyword.
    #: E.g. ``.AC LIN 500 100G 170G`` → ``["sweep_type", "points", "start_freq", "stop_freq"]``
    positional_param_names: list[str] = field(default_factory=list)

    #: Human-readable tooltip for each positional param (keyed by param name).
    positional_param_descriptions: dict[str, str] = field(default_factory=dict)

    #: Sensible default value for each positional param (keyed by param name).
    positional_param_defaults: dict[str, str] = field(default_factory=dict)

    #: The ``.options`` subcategory for this sim type, if any (e.g. ``"hbint"`` for HB).
    options_category: str | None = None

    #: Human-readable tooltip for each ``.options <category>`` parameter.
    options_param_descriptions: dict[str, str] = field(default_factory=dict)
