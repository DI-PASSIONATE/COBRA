from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from cobra.spice_sim.simulation_type import SimulationType


@dataclass
class NetlistElement:
    name: str
    etype: str                 # R, C, L, V, I, D, Q, M, X, P, Y, MODEL, ...
    subtype: str | None     # for Y-devices: second token (e.g. YLIN_Trafo1); else None
    line_index: int
    raw_line: str
    inline_comment: str
    tokens: list[str]
    nodes: list[str]
    value: str | None
    model: str | None
    params: dict[str, str]


@dataclass
class Component:
    """A subcircuit (X-instance) in the netlist that requires a surrogate model."""
    name: str                  # e.g. "X1"
    nodes: list[str]           # e.g. ["Port1", "Port2", "_net0"]
    model: str                 # e.g. "s_equivalent"
    params: dict[str, str]     # key=value parameters for the instance


@dataclass
class Include:
    """An .INCLUDE directive found in the netlist."""
    file_path: str             # e.g. "cobra_output.sp"
    line_index: int


@dataclass
class Library:
    """A ``.LIB`` directive found in the netlist."""
    file_path: str             # e.g. "cornerHBT.lib"
    entry: str | None          # library section, e.g. "hbt_typ"
    line_index: int


@dataclass
class SimulationDirective:
    """A simulation/analysis directive found in the netlist (e.g. .AC, .LIN, .HB)."""
    directive: str             # e.g. ".AC", ".LIN", ".TRAN"
    positional: list[str]      # positional tokens after the keyword, e.g. ["LIN", "500", "100G", "170G"]
    kv_params: dict[str, str]  # key=value params on the same line, e.g. {"format": "touchstone"}
    line_index: int


@dataclass
class PrintDirective:
    """A ``.PRINT`` directive found in the netlist (e.g. ``.PRINT hb format=csv v(Out) I(VOut)``)."""
    analysis: str              # analysis keyword in lower case, e.g. "hb", "ac", "tran"
    signals: list[str]         # printed signal tokens, e.g. ["v(Out)", "I(VOut)"]
    kv_params: dict[str, str]  # key=value params, e.g. {"format": "csv"}
    line_index: int


class BaseNetlistParser(ABC):
    """Abstract base class for SPICE netlist parsers."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._elements: dict[str, NetlistElement] = {}   # all elements, including MODEL entries
        self._components: dict[str, Component] = {}      # X-instances requiring surrogate models
        self._includes: list[Include] = []               # .INCLUDE directives
        self._libraries: list[Library] = []              # .LIB directives
        self._inline_subckt_names: set[str] = set()      # .SUBCKT names defined within this file
        self.netlist_path: str | Path | None = None
        self._simulation_type: SimulationType = SimulationType.UNKNOWN
        self._num_ports: int = 0
        self._simulation_directives: list[SimulationDirective] = []
        self._print_directives: list[PrintDirective] = []
        self._port_sources: dict[str, dict] = {} # Maps P-element name → {sin_amplitude, z0, ac_amplitude} for ports that carry a SIN source."""

    # -------------------------------------------------------------------------
    # Factory / loader methods
    # -------------------------------------------------------------------------

    def from_lines(self, lines: list[str]) -> Self:
        """Load the parser from a list of raw text lines and (re-)parse."""
        self._lines = lines[:]
        self._reset_state()
        self.parse_netlist()
        return self

    def from_file(self, path: str | Path, encoding: str = "utf-8") -> Self:
        """Load the parser from a file on disk and (re-)parse."""
        self.netlist_path = path
        text = Path(path).read_text(encoding=encoding, errors="replace")
        return self.from_lines(text.splitlines(keepends=True))

    def _reset_state(self) -> None:
        """Clear all previously parsed data so parse_netlist starts with a clean slate."""
        self._elements.clear()
        self._components.clear()
        self._includes.clear()
        self._libraries.clear()
        self._inline_subckt_names.clear()
        self._simulation_type = SimulationType.UNKNOWN
        self._num_ports = 0
        self._simulation_directives.clear()
        self._print_directives.clear()
        self._port_sources.clear()

    # -------------------------------------------------------------------------
    # Serialisation
    # -------------------------------------------------------------------------

    def to_string(self) -> str:
        """Return the current (possibly modified) netlist as a single string."""
        return "".join(self._lines)

    def save(self, path: str | Path, encoding: str = "utf-8") -> None:
        """Write the current netlist text to *path*."""
        Path(path).write_text(self.to_string(), encoding=encoding)

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------

    def list_elements(self, types: list[str] | None = None) -> list[NetlistElement]:
        """Return parsed elements in line order, optionally filtered by etype."""
        elems = sorted(self._elements.values(), key=lambda e: e.line_index)
        if types:
            want = {t.upper() for t in types}
            elems = [e for e in elems if e.etype in want]
        return elems

    def get_element(self, name: str) -> NetlistElement:
        """Look up a single element by name, raising KeyError if absent."""
        name = name.strip()
        if name not in self._elements:
            raise KeyError(f"Element '{name}' not found.")
        return self._elements[name]

    @property
    def components(self) -> dict[str, Component]:
        """Dict of X-instance components that require surrogate models."""
        return self._components.copy()

    @property
    def inline_subckt_names(self) -> set[str]:
        """Subcircuit names defined inline via .SUBCKT in this netlist."""
        return set(self._inline_subckt_names)

    @property
    def includes(self) -> list[Include]:
        """List of .INCLUDE directives found in the netlist."""
        return self._includes.copy()

    @property
    def libraries(self) -> list[Library]:
        """List of .LIB directives found in the netlist."""
        return self._libraries.copy()

    @property
    def simulation_type(self) -> SimulationType:
        """The primary analysis type detected in the netlist (e.g. LIN, AC, HB)."""
        return self._simulation_type

    @property
    def simulation_directives(self) -> list[SimulationDirective]:
        """All simulation/analysis directives found in the netlist."""
        return list(self._simulation_directives)

    @property
    def print_directives(self) -> list[PrintDirective]:
        """All ``.PRINT`` directives found in the netlist."""
        return list(self._print_directives)

    @property
    def num_ports(self) -> int:
        """Number of port elements (P-instances) found in the netlist."""
        return self._num_ports

    @property
    def port_sources(self) -> dict[str, dict]:
        """Dict mapping P-element name → waveform info (sin_amplitude, z0, ac_amplitude).

        Only ports that carry an explicit SIN or AC source declaration are included.
        These are used to compute Pin for Gain calculations.
        """
        return self._port_sources.copy()

    @property
    def available_design_parameters(self) -> list[str]:
        """
        Convenience shortcut: the design parameters available for the current
        simulation type and port count.
        """
        return self._simulation_type.available_parameters(self._num_ports)

    # -------------------------------------------------------------------------
    # Mutators (registration helpers)
    # -------------------------------------------------------------------------

    def add_component(self, component: Component) -> None:
        """Register a component that requires a surrogate model."""
        self._components[component.name] = component

    def add_include(self, include: Include) -> None:
        """Register an .INCLUDE directive."""
        self._includes.append(include)

    def add_library(self, library: Library) -> None:
        """Register a .LIB directive."""
        self._libraries.append(library)

    # -------------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # -------------------------------------------------------------------------

    @abstractmethod
    def parse_netlist(self) -> None:
        """Parse _lines and populate _elements, _components, and _includes."""

    @abstractmethod
    def update_parameters(self, parameters: dict[str, float]) -> None:
        """Apply a dict of {name: value} updates to the netlist held in memory."""

    # -------------------------------------------------------------------------
    # Optional interface — subclasses override these when the dialect supports it
    # -------------------------------------------------------------------------

    def set_model(self, name: str, new_model: str) -> None:
        """Replace the model reference of element *name* with *new_model*.

        Parsers for dialects without model substitution keep the default, which
        raises ``NotImplementedError`` so callers can skip the element.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support replacing the model of '{name}'."
        )
