from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union
from pathlib import Path
import skrf as rf


@dataclass
class NetlistElement:
    name: str
    etype: str                 # R, C, L, V, I, D, Q, M, X, P, Y, MODEL, ...
    subtype: Optional[str]     # for Y-devices: second token (e.g. YLIN_Trafo1); else None
    line_index: int
    raw_line: str
    inline_comment: str
    tokens: List[str]
    nodes: List[str]
    value: Optional[str]
    model: Optional[str]
    params: Dict[str, str]


@dataclass
class Component:
    """A subcircuit (X-instance) in the netlist that requires a surrogate model."""
    name: str                  # e.g. "X1"
    nodes: List[str]           # e.g. ["Port1", "Port2", "_net0"]
    model: str                 # e.g. "s_equivalent"
    params: Dict[str, str]     # key=value parameters for the instance


@dataclass
class Include:
    """An .INCLUDE directive found in the netlist."""
    file_path: str             # e.g. "cobra_output.sp"
    line_index: int


class BaseNetlistParser(ABC):
    """Abstract base class for SPICE netlist parsers."""

    def __init__(self) -> None:
        self._lines: List[str] = []
        self._elements: Dict[str, NetlistElement] = {}   # all elements, including MODEL entries
        self._components: Dict[str, Component] = {}      # X-instances requiring surrogate models
        self._includes: List[Include] = []               # .INCLUDE directives
        self._inline_subckt_names: Set[str] = set()      # .SUBCKT names defined within this file
        self.netlist_path: Optional[Union[str, Path]] = None

    # -------------------------------------------------------------------------
    # Factory / loader methods
    # -------------------------------------------------------------------------

    def from_lines(self, lines: List[str]) -> "BaseNetlistParser":
        """Load the parser from a list of raw text lines and (re-)parse."""
        self._lines = lines[:]
        self._reset_state()
        self.parse_netlist()
        return self

    def from_file(self, path: Union[str, Path], encoding: str = "utf-8") -> "BaseNetlistParser":
        """Load the parser from a file on disk and (re-)parse."""
        self.netlist_path = path
        text = Path(path).read_text(encoding=encoding, errors="replace")
        return self.from_lines(text.splitlines(keepends=True))

    def _reset_state(self) -> None:
        """Clear all previously parsed data so parse_netlist starts with a clean slate."""
        self._elements.clear()
        self._components.clear()
        self._includes.clear()
        self._inline_subckt_names.clear()

    # -------------------------------------------------------------------------
    # Serialisation
    # -------------------------------------------------------------------------

    def to_string(self) -> str:
        """Return the current (possibly modified) netlist as a single string."""
        return "".join(self._lines)

    def save(self, path: Union[str, Path], encoding: str = "utf-8") -> None:
        """Write the current netlist text to *path*."""
        Path(path).write_text(self.to_string(), encoding=encoding)

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------

    def list_elements(self, types: Optional[List[str]] = None) -> List[NetlistElement]:
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
    def components(self) -> Dict[str, Component]:
        """Dict of X-instance components that require surrogate models."""
        return self._components.copy()

    @property
    def inline_subckt_names(self) -> Set[str]:
        """Subcircuit names defined inline via .SUBCKT in this netlist."""
        return set(self._inline_subckt_names)

    @property
    def includes(self) -> List[Include]:
        """List of .INCLUDE directives found in the netlist."""
        return self._includes.copy()

    # -------------------------------------------------------------------------
    # Mutators (registration helpers)
    # -------------------------------------------------------------------------

    def add_component(self, component: Component) -> None:
        """Register a component that requires a surrogate model."""
        self._components[component.name] = component

    def add_include(self, include: Include) -> None:
        """Register an .INCLUDE directive."""
        self._includes.append(include)

    # -------------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # -------------------------------------------------------------------------

    @abstractmethod
    def parse_netlist(self) -> None:
        """Parse _lines and populate _elements, _components, and _includes."""

    @abstractmethod
    def update_parameters(self, parameters: Dict[str, float]) -> None:
        """Apply a dict of {name: value} updates to the netlist held in memory."""