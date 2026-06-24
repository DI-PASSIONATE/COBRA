"""
SimulationType – a lightweight enum that identifies the primary analysis directive
found in a SPICE/Xyce netlist (.AC, .HB, .TRAN, .DC, …).
"""
from __future__ import annotations

from enum import Enum
from typing import List


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
    def from_directive(cls, directive: str) -> "SimulationType":
        """
        Convert a raw dot-directive string (e.g. ``.AC``, ``.tran``) to the
        corresponding ``SimulationType``.  Both ``.AC`` and ``.LIN`` map to
        ``SimulationType.AC`` because ``.LIN`` is a post-processor on ``.AC``,
        not a separate simulation.  Unrecognised directives map to ``UNKNOWN``.
        """
        normalized = directive.strip().upper()
        _map: dict[str, "SimulationType"] = {
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
        """Short label shown in the GUI (e.g. ``".LIN"``, ``"unknown"``)."""
        return self.value

    def positional_param_names(self) -> List[str]:
        """
        Named slots for the positional tokens that follow this directive keyword.

        E.g. ``.AC LIN 500 100G 170G`` → ``["sweep_type", "points", "start_freq", "stop_freq"]``
        """
        _map: dict = {
            SimulationType.AC:   ["sweep_type", "points", "start_freq", "stop_freq"],
            SimulationType.TRAN: ["step", "stop_time", "start_time", "max_step"],
            SimulationType.HB:   ["fund_freq"],
            SimulationType.DC:   ["src_name", "start", "stop", "incr"],
        }
        return list(_map.get(self, []))

    def positional_param_descriptions(self) -> dict[str, str]:
        """Human-readable tooltip text for each positional parameter slot."""
        _map: dict = {
            SimulationType.AC: {
                "sweep_type": "Frequency sweep spacing: LIN (linear), DEC (decade), or OCT (octave).",
                "points":     "Number of frequency points in the sweep.",
                "start_freq": "Start frequency (e.g. 100G for 100 GHz).",
                "stop_freq":  "Stop frequency (e.g. 200G for 200 GHz).",
            },
            SimulationType.TRAN: {
                "step":       "Print/output time step.",
                "stop_time":  "Total simulation stop time.",
                "start_time": "Time at which output begins (default 0).",
                "max_step":   "Maximum internal time step (optional).",
            },
            SimulationType.HB: {
                "fund_freq": "Fundamental frequency for the Harmonic Balance analysis.",
            },
            SimulationType.DC: {
                "src_name": "Name of the voltage/current source to sweep.",
                "start":    "Sweep start value.",
                "stop":     "Sweep stop value.",
                "incr":     "Sweep increment step.",
            },
        }
        return dict(_map.get(self, {}))

    def positional_param_defaults(self) -> dict[str, str]:
        """Sensible default values for each positional parameter slot."""
        _map: dict = {
            SimulationType.AC:   {"sweep_type": "LIN", "points": "500", "start_freq": "1G", "stop_freq": "10G"},
            SimulationType.TRAN: {"step": "1n", "stop_time": "100n", "start_time": "0", "max_step": "1n"},
            SimulationType.HB:   {"fund_freq": "1G"},
            SimulationType.DC:   {"src_name": "V1", "start": "0", "stop": "1", "incr": "0.01"},
        }
        return dict(_map.get(self, {}))

    # ------------------------------------------------------------------
    # Parameter discovery
    # ------------------------------------------------------------------

    def available_parameters(self, num_ports: int) -> List[str]:
        """
        Return parameter *names* valid for this simulation type and port count.

        Iterates the global ``_ALL_PARAMETERS`` catalogue in
        ``cobra.optimizers.design_goal`` and returns the names of every entry
        whose ``simulation_type`` matches *self* and whose ``min_ports``
        requirement is met.

        For full ``DesignParameter`` objects use
        ``get_available_parameters(num_ports, simulation_type=self)``.
        """
        from cobra.optimizers.design_goal import get_available_parameters  # lazy – avoids circular import
        return [p.name for p in get_available_parameters(num_ports, simulation_type=self)]

    @classmethod
    def for_parameter(cls, param_name: str) -> "SimulationType":
        """
        Return the simulation type required to evaluate *param_name* by looking
        it up in the global ``_ALL_PARAMETERS`` catalogue.
        """
        from cobra.optimizers.design_goal import find_parameter  # lazy – avoids circular import
        p = find_parameter(param_name)
        return p.simulation_type if p is not None else cls.UNKNOWN

    @classmethod
    def all_available_parameters(cls, num_ports: int) -> List[str]:
        """Union of parameter names across all supported simulation types."""
        from cobra.optimizers.design_goal import get_available_parameters  # lazy – avoids circular import
        return [p.name for p in get_available_parameters(num_ports)]
