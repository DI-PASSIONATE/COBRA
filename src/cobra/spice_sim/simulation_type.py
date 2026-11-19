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
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cobra.spice_sim.netlist_parsers.netlist import Netlist


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
    ``NOISE``   — Small-signal noise (noise figure).
    ``HBNOISE`` — Noise around a Harmonic Balance solution (mixer noise figure).
    """

    AC      = ".AC"    # AC frequency sweep; COBRA always pairs this with .LIN for S-parameter (Touchstone) output
    HB      = ".HB"    # Harmonic Balance
    TRAN    = ".TRAN"  # Transient
    DC      = ".DC"    # DC sweep
    NOISE   = ".NOISE"    # Small-signal noise
    HBNOISE = ".HBNOISE"  # (Quasi)periodic noise on a Harmonic Balance solution
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
            ".NOISE": cls.NOISE,
            ".HBNOISE": cls.HBNOISE,
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
# AnalysisVariant — a second analysis of a type, on a netlist with other values
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalysisVariant:
    """An analysis of *simulation_type* on the netlist with some values changed.

    A run holds one result per analysis type, so a netlist cannot carry two HB
    analyses at different drives. A variant adds one: the circuit simulation runs
    it as an analysis of its own, on a copy of the netlist with ``netlist_values``
    set (``"vp_rf:ampl"`` for an instance parameter, ``"R1"`` for an element value,
    as for ``netlist_variable`` parameters). Goals on it carry its name as a prefix,
    e.g. ``HB_m40:Power_dBm[Vdd]``.

    Like :class:`SimulationType`, it has a ``name`` (used in file names) and a
    ``value`` (used in messages), so the two can key the same mappings.
    """

    name: str
    simulation_type: SimulationType
    netlist_values: tuple[tuple[str, str], ...] = ()

    @classmethod
    def create(
        cls, name: str, simulation_type: SimulationType, netlist_values: Mapping[str, float | str]
    ) -> AnalysisVariant:
        """A variant setting *netlist_values*, given as a mapping."""
        return cls(name, simulation_type, tuple((key, str(value)) for key, value in netlist_values.items()))

    @property
    def value(self) -> str:
        return self.name

    @property
    def display_name(self) -> str:
        """Label shown in the GUI, e.g. ``"HB_m40"``."""
        return self.name

    def netlist_value(self, name: str) -> str | None:
        """The value the variant sets for *name* (case-insensitive), or ``None``."""
        return next(
            (value for key, value in self.netlist_values if key.casefold() == name.casefold()),
            None,
        )

    def apply(self, netlist: Netlist) -> Netlist:
        """A copy of *netlist* with the variant's values set."""
        changed = netlist.copy()
        changed.update_parameters(dict(self.netlist_values))
        return changed


#: What an analysis of a run is keyed by: a netlist analysis type or a variant of one.
Analysis: TypeAlias = SimulationType | AnalysisVariant


def base_simulation_type(analysis: Analysis) -> SimulationType:
    """The analysis type *analysis* runs: itself, or the type a variant is based on."""
    return analysis.simulation_type if isinstance(analysis, AnalysisVariant) else analysis


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
