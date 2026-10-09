"""The parsed netlist: a dialect-agnostic document COBRA queries and edits.

A :class:`Netlist` holds the statements its parser split the text into, and
one decoded record per statement.  Queries read the records; edits ask the
parser for a replacement statement and decode only that one, so editing never
re-parses the netlist.  Statements are immutable and records are frozen; an
edit replaces both instead of changing them, which makes :meth:`Netlist.copy`
cheap and safe to hand to another thread.  Treat the lists and dicts inside a
record as read-only, since copies share them.

Name lookups follow the parser's :meth:`~NetlistParser.fold`: case-insensitive
for SPICE dialects.

"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from cobra.spice_sim.simulation_type import SimulationType

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from cobra.spice_sim.netlist_parsers.netlist_parser import NetlistParser
    from cobra.spice_sim.netlist_parsers.statement import Statement

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Records: what a statement describes. ``line_index`` is the 0-based physical
# line the statement starts on; ``statement_index`` its position in
# ``Netlist.statements``.
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NetlistElement:
    name: str
    etype: str                 # R, C, L, V, I, D, Q, M, X, P, Y, MODEL, ...
    subtype: str | None        # for Y-devices: second token (e.g. YLIN_Trafo1); else None
    line_index: int
    statement_index: int
    nodes: list[str]
    value: str | None
    model: str | None
    params: dict[str, str]
    port: int | None = None    # port number when the element is a netlist port


@dataclass
class Component:
    """A subcircuit (X-instance) in the netlist that requires a surrogate model."""
    name: str                  # e.g. "X1"
    nodes: list[str]           # e.g. ["Port1", "Port2", "_net0"]
    model: str                 # e.g. "s_equivalent"
    params: dict[str, str]     # key=value parameters for the instance


@dataclass(frozen=True, slots=True)
class Subcircuit:
    """A subcircuit defined in the netlist itself."""
    name: str
    line_index: int
    statement_index: int


@dataclass(frozen=True, slots=True)
class Include:
    """An .INCLUDE directive found in the netlist."""
    file_path: str             # e.g. "cobra_output.sp"
    line_index: int
    statement_index: int


@dataclass(frozen=True, slots=True)
class Library:
    """A ``.LIB`` directive found in the netlist."""
    file_path: str             # e.g. "cornerHBT.lib"
    entry: str | None          # library section, e.g. "hbt_typ"
    line_index: int
    statement_index: int


@dataclass(frozen=True, slots=True)
class SimulationDirective:
    """A simulation/analysis directive found in the netlist (e.g. .AC, .LIN, .HB)."""
    directive: str             # e.g. ".AC", ".LIN", ".TRAN"
    positional: list[str]      # positional tokens after the keyword, e.g. ["LIN", "500", "100G", "170G"]
    kv_params: dict[str, str]  # key=value params on the same line, e.g. {"format": "touchstone"}
    line_index: int
    statement_index: int
    simulation_type: SimulationType
    # False for a directive that only modifies another analysis' output, such as .LIN.
    is_analysis: bool = True
    name: str | None = None    # analysis name in dialects that name them


@dataclass(frozen=True, slots=True)
class PrintDirective:
    """A ``.PRINT`` directive found in the netlist (e.g. ``.PRINT hb format=csv v(Out) I(VOut)``)."""
    analysis: str              # analysis keyword in lower case, e.g. "hb", "ac", "tran"
    signals: list[str]         # printed signal tokens, e.g. ["v(Out)", "I(VOut)"]
    kv_params: dict[str, str]  # key=value params, e.g. {"format": "csv"}
    line_index: int
    statement_index: int


@dataclass(frozen=True, slots=True)
class OptionsDirective:
    """A ``.OPTIONS`` directive (e.g. ``.options hbint numfreq=3``)."""
    category: str              # as written, e.g. "hbint"; "" when the line has none
    params: dict[str, str]
    line_index: int
    statement_index: int


NetlistRecord = (
    NetlistElement | Subcircuit | Include | Library | SimulationDirective | PrintDirective
    | OptionsDirective
)


@dataclass(frozen=True, slots=True)
class _Views:
    """Everything derived from the records, rebuilt lazily after an edit."""
    elements: list[NetlistElement]
    components: dict[str, Component]
    includes: list[Include]
    libraries: list[Library]
    simulation_directives: list[SimulationDirective]
    print_directives: list[PrintDirective]
    options: dict[str, OptionsDirective]  # keyed by lower-case category
    port_sources: dict[str, dict[str, float]]


class Netlist:
    """A parsed netlist: its statements, what they describe, and edits on them."""

    def __init__(
        self,
        parser: NetlistParser,
        statements: list[Statement],
    ) -> None:
        self.parser = parser
        self._statements = list(statements)
        self._records: list[NetlistRecord | None] = []
        line_index = 0
        for index, statement in enumerate(self._statements):
            self._records.append(parser.decode(statement, line_index, index))
            line_index += len(statement.lines)
        # Folded element name -> statement index. Edits never rename, so this
        # stays valid until the statements themselves change. An instance wins
        # over a .MODEL whose name folds to the same key.
        elements = [
            (index, record) for index, record in enumerate(self._records)
            if isinstance(record, NetlistElement)
        ]
        self._names: dict[str, int] = {
            parser.fold(record.name): index
            for index, record in sorted(elements, key=lambda item: item[1].etype != "MODEL")
        }
        self._views: _Views | None = None

    # -------------------------------------------------------------------------
    # Copies and rendering
    # -------------------------------------------------------------------------

    def copy(self) -> Netlist:
        """An independent netlist that shares the immutable statements and records."""
        clone = copy.copy(self)
        clone._statements = list(self._statements)  # noqa: SLF001 - same class
        clone._records = list(self._records)  # noqa: SLF001
        return clone

    def with_statements(self, statements: list[Statement]) -> Netlist:
        """A new netlist made of *statements*, decoded by the same parser."""
        return Netlist(self.parser, statements)

    @property
    def statements(self) -> tuple[Statement, ...]:
        return tuple(self._statements)

    def to_string(self) -> str:
        """Return the current (possibly modified) netlist as a single string."""
        return "".join(line for statement in self._statements for line in statement.lines)

    def save(self, path: str | Path, encoding: str = "utf-8") -> None:
        """Write the current netlist text to *path*."""
        Path(path).write_text(self.to_string(), encoding=encoding)

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def list_elements(self, types: list[str] | None = None) -> list[NetlistElement]:
        """Return parsed elements in line order, optionally filtered by etype."""
        elements = self._view().elements
        if types:
            wanted = {t.upper() for t in types}
            return [e for e in elements if e.etype in wanted]
        return list(elements)

    def get_element(self, name: str) -> NetlistElement:
        """Look up a single element by name, ignoring case; raises KeyError if absent."""
        record = self._records[self._index_of(name)]
        if not isinstance(record, NetlistElement):
            raise KeyError(f"Element '{name}' not found.")
        return record

    def has_element(self, name: str) -> bool:
        return self.parser.fold(name.strip()) in self._names

    @property
    def components(self) -> dict[str, Component]:
        """Dict of X-instance components that require surrogate models."""
        return dict(self._view().components)

    @property
    def inline_subckt_names(self) -> set[str]:
        """Subcircuit names defined inline via .SUBCKT in this netlist."""
        return {
            record.name for record in self._records if isinstance(record, Subcircuit)
        }

    @property
    def includes(self) -> list[Include]:
        """List of .INCLUDE directives found in the netlist."""
        return list(self._view().includes)

    @property
    def libraries(self) -> list[Library]:
        """List of .LIB directives found in the netlist."""
        return list(self._view().libraries)

    @property
    def simulation_directives(self) -> list[SimulationDirective]:
        """All simulation/analysis directives found in the netlist."""
        return list(self._view().simulation_directives)

    @property
    def simulation_type(self) -> SimulationType:
        """The primary analysis type: that of the first analysis directive."""
        directives = self._view().simulation_directives
        first = next((d for d in directives if d.is_analysis), None) or next(iter(directives), None)
        return first.simulation_type if first is not None else SimulationType.UNKNOWN

    @property
    def print_directives(self) -> list[PrintDirective]:
        """All ``.PRINT`` directives found in the netlist."""
        return list(self._view().print_directives)

    @property
    def options_directives(self) -> dict[str, dict[str, str]]:
        """Parsed ``.options`` lines, keyed by lower-case category name (e.g. ``'hbint'``)."""
        return {key: dict(entry.params) for key, entry in self._view().options.items()}

    @property
    def num_ports(self) -> int:
        """Number of port elements found in the netlist."""
        return sum(1 for element in self._view().elements if element.port is not None)

    @property
    def port_sources(self) -> dict[str, dict[str, float]]:
        """Dict mapping port name → waveform info (sin_amplitude, z0, ac_amplitude).

        Only ports that carry an explicit SIN or AC source declaration are included.
        These are used to compute Pin for Gain calculations.
        """
        return {name: dict(info) for name, info in self._view().port_sources.items()}

    @property
    def available_design_parameters(self) -> list[str]:
        """The design parameters available for the current simulation type and port count."""
        return self.simulation_type.available_parameters(self.num_ports)

    @property
    def probe_nodes(self) -> list[str]:
        """Node names that can serve as a large-signal (HB or transient) analysis point."""
        return self.parser.probe_nodes(self)

    # -------------------------------------------------------------------------
    # Element edits
    # -------------------------------------------------------------------------

    def set_value(self, name: str, value: str) -> None:
        """Replace the value of element *name*, e.g. the resistance of an R."""
        self._edit(self._index_of(name), lambda s: self.parser.set_value(s, value))

    def set_model(self, name: str, model: str) -> None:
        """Replace the model or subcircuit that element *name* refers to."""
        self._edit(self._index_of(name), lambda s: self.parser.set_model(s, model))

    def set_param(self, name: str, key: str, value: str) -> None:
        """Update or append a ``key=value`` parameter on element *name*."""
        self._edit(self._index_of(name), lambda s: self.parser.set_param(s, key.strip(), value))

    def update_parameters(self, parameters: Mapping[str, float | str]) -> None:
        """
        Bulk-update multiple parameters by name, ignoring case.

        Parameter names follow two conventions:

        * ``"ElementName"`` — updates the value of an element via :meth:`set_value`.
        * ``"InstanceName:ParamKey"`` — updates a key=value parameter on an
          instance via :meth:`set_param`.
        """
        for name, value in parameters.items():
            if ":" in name:
                instance_name, param_key = name.split(":", 1)
                if self.has_element(instance_name):
                    self.set_param(instance_name, param_key, str(value))
                else:
                    logger.warning("Instance '%s' not found in netlist elements.", instance_name)
            elif self.has_element(name):
                self.set_value(name, str(value))
            else:
                logger.warning("Parameter '%s' not found in netlist elements.", name)

    # -------------------------------------------------------------------------
    # Directive edits
    # -------------------------------------------------------------------------

    def update_simulation_directive(
        self, sim_type: SimulationType, params: Mapping[str, str]
    ) -> None:
        """
        Update named parameters of the *sim_type* analysis in-place.

        ``params`` keys can be:

        * **Positional names** from the parser's ``analysis_metadata``
          (e.g. ``"start_freq"``, ``"stop_freq"``, ``"points"`` for AC).
        * **Key=value names** (e.g. ``"format"``); missing ones are added.

        Example::

            netlist.update_simulation_directive(SimulationType.AC, {"start_freq": "100G"})
        """
        target = next(
            (
                d for d in self._view().simulation_directives
                if d.simulation_type is sim_type and d.is_analysis
            ),
            None,
        )
        if target is None:
            raise KeyError(f"No {sim_type.value} analysis found in netlist.")

        param_names = self.parser.analysis_metadata(sim_type).positional_param_names
        positional = list(target.positional)
        kv_params = dict(target.kv_params)
        fold = self.parser.fold
        for param_name, value in params.items():
            if param_name in param_names:
                index = param_names.index(param_name)
                # A value may hold several space-separated tokens (e.g. HB
                # "frequencies" = "95E9 10E9"), which fill consecutive slots.
                sub_tokens = str(value).split()
                for offset, token in enumerate(sub_tokens):
                    while len(positional) <= index + offset:
                        positional.append("")
                    positional[index + offset] = token
                # Drop slots that no longer exist
                positional = positional[:index + len(sub_tokens)]
            else:
                existing = next((k for k in kv_params if fold(k) == fold(param_name)), param_name)
                kv_params[existing] = str(value)

        self._edit(
            target.statement_index,
            lambda s: self.parser.set_directive(s, positional, kv_params),
        )

    def update_options_directive(self, category: str, params: Mapping[str, str]) -> None:
        """Update key=value parameters on a ``.options <category>`` line in-place.

        Keys match case-insensitively; missing ones are added. Example::

            netlist.update_options_directive("hbint", {"numfreq": "5"})
        """
        entry = self._view().options.get(category.lower())
        if entry is None:
            raise KeyError(f".options {category} not found in netlist.")
        merged = dict(entry.params)
        fold = self.parser.fold
        for key, value in params.items():
            existing = next((k for k in merged if fold(k) == fold(key)), key)
            merged[existing] = value
        positional = [entry.category] if entry.category else []
        self._edit(
            entry.statement_index,
            lambda s: self.parser.set_directive(s, positional, merged),
        )

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _index_of(self, name: str) -> int:
        index = self._names.get(self.parser.fold(name.strip()))
        if index is None:
            raise KeyError(f"Element '{name}' not found.")
        return index

    def _edit(self, index: int, edit: Callable[[Statement], Statement]) -> None:
        """Replace statement *index* with ``edit(statement)`` and decode only that one."""
        record = self._records[index]
        if record is None:
            raise ValueError(f"Statement {index} describes nothing that can be edited.")
        statement = edit(self._statements[index])
        self._statements[index] = statement
        self._records[index] = self.parser.decode(statement, record.line_index, index)
        self._views = None

    def _view(self) -> _Views:
        if self._views is None:
            self._views = self._build_views()
        return self._views

    def _build_views(self) -> _Views:
        records = [record for record in self._records if record is not None]
        elements = [r for r in records if isinstance(r, NetlistElement)]
        # Classified only now that every subcircuit name is known, so an instance
        # may come before the .SUBCKT it uses.
        subcircuit_names = {self.parser.fold(r.name) for r in records if isinstance(r, Subcircuit)}
        components = {
            e.name: Component(
                name=e.name, nodes=list(e.nodes), model=e.model or "", params=dict(e.params)
            )
            for e in elements
            if self.parser.is_component(e, subcircuit_names)
        }
        port_sources: dict[str, dict[str, float]] = {}
        for element in elements:
            if element.port is not None:
                source = self.parser.port_source(self._statements[element.statement_index])
                if source:
                    port_sources[element.name] = source
        options = {r.category.lower(): r for r in records if isinstance(r, OptionsDirective)}
        return _Views(
            elements=elements,
            components=components,
            includes=[r for r in records if isinstance(r, Include)],
            libraries=[r for r in records if isinstance(r, Library)],
            simulation_directives=[r for r in records if isinstance(r, SimulationDirective)],
            print_directives=[r for r in records if isinstance(r, PrintDirective)],
            options=options,
            port_sources=port_sources,
        )
