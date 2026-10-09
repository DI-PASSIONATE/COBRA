"""The contract every simulator's netlist dialect implements.

A :class:`NetlistParser` knows the syntax of one netlist language and holds no
state, so one instance serves every netlist; each simulator class owns one as
``netlist_parser``.  Parsing returns a :class:`Netlist`, the dialect-agnostic
document the rest of COBRA queries and edits.  The netlist calls back into its
parser for anything syntax-specific: decoding a statement into a record, or
rewriting a statement for an edit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from cobra.spice_sim.netlist_parsers.netlist import Netlist, NetlistElement, NetlistRecord

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cobra.spice_sim.netlist_parsers.statement import Statement, Token
    from cobra.spice_sim.simulation_type import SimulationType, SimulationTypeMetadata


class NetlistParser(ABC):
    """Syntax of one simulator's netlist language."""

    # -------------------------------------------------------------------------
    # Entry points
    # -------------------------------------------------------------------------

    def parse(self, text: str, *, has_title: bool = True) -> Netlist:
        """Parse netlist *text*.

        Pass ``has_title=False`` for a file that is included into a netlist,
        which has no title line.
        """
        return Netlist(self, self.normalize(self.split(text, has_title=has_title)))

    def parse_file(
        self, path: str | Path, encoding: str = "utf-8", *, has_title: bool = True
    ) -> Netlist:
        """Read and parse the netlist at *path*."""
        text = Path(path).read_text(encoding=encoding, errors="replace")
        return self.parse(text, has_title=has_title)

    def make_statements(self, text: str) -> list[Statement]:
        """Split a snippet, e.g. a directive to inject, which has no title line."""
        return self.split(text, has_title=False)

    # -------------------------------------------------------------------------
    # Reading
    # -------------------------------------------------------------------------

    @abstractmethod
    def split(self, text: str, *, has_title: bool = True) -> list[Statement]:
        """Split *text* into statements: lex, join continuation lines, classify, track scope.

        Joining every statement's lines must give back *text* unchanged.
        """

    @abstractmethod
    def decode(
        self, statement: Statement, line_index: int, statement_index: int
    ) -> NetlistRecord | None:
        """The record *statement* describes, or ``None`` when it is not one COBRA reads."""

    def fold(self, name: str) -> str:
        """The form of *name* used to compare names. Override for a case-sensitive dialect."""
        return name.casefold()

    def tokens(self, statement: Statement) -> tuple[Token, ...]:
        """The code tokens of *statement*, lexing it now if the split left it unlexed."""
        return statement.tokens if statement.tokens is not None else ()

    def normalize(self, statements: list[Statement]) -> list[Statement]:
        """Rewrite constructs COBRA cannot simulate as they are. Runs once per parse."""
        return statements

    @abstractmethod
    def is_component(self, element: NetlistElement, subcircuit_names: set[str]) -> bool:
        """Whether *element* needs a surrogate model.

        *subcircuit_names* holds the folded names (see :meth:`fold`) of the
        subcircuits defined in the netlist itself.
        """

    @abstractmethod
    def probe_nodes(self, netlist: Netlist) -> list[str]:
        """Nodes whose voltage and current the simulator writes for a large-signal
        (HB or transient) analysis, so they can serve as an analysis point.
        """

    def port_source(self, statement: Statement) -> dict[str, float]:  # noqa: ARG002
        """Source data of a port statement: any of ``z0``, ``ac_amplitude``,
        ``sin_amplitude`` and ``sin_frequency``. Empty when the port has no source.
        """
        return {}

    # -------------------------------------------------------------------------
    # Editing: each method returns a replacement for *statement* with the same
    # number of physical lines, so line numbers in the netlist stay valid.
    # -------------------------------------------------------------------------

    @abstractmethod
    def set_value(self, statement: Statement, value: str) -> Statement:
        """Replace the value of an element (e.g. a resistance or a source)."""

    @abstractmethod
    def set_model(self, statement: Statement, model: str) -> Statement:
        """Replace the model or subcircuit an element refers to."""

    @abstractmethod
    def set_param(self, statement: Statement, key: str, value: str) -> Statement:
        """Update parameter *key*, matched case-insensitively, or add it."""

    @abstractmethod
    def set_directive(
        self, statement: Statement, positional: list[str], params: Mapping[str, str]
    ) -> Statement:
        """Rewrite the arguments of an analysis or options directive."""

    # -------------------------------------------------------------------------
    # Analyses
    # -------------------------------------------------------------------------

    @abstractmethod
    def analysis_metadata(self, sim_type: SimulationType) -> SimulationTypeMetadata:
        """Names, descriptions and defaults of the arguments of *sim_type*'s directive."""
