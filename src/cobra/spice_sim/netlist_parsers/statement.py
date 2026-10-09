"""Statements: the lexical units a :class:`NetlistParser` splits a netlist into.

A statement is one logical line of a netlist: a device or directive line with
its continuation lines, a comment, or the title.  It keeps its physical lines
verbatim, so a netlist renders back byte for byte, and the code tokens with
their positions, so an edit can replace exactly one token and leave spacing,
comments and line breaks alone.

Statements are immutable.  An edit builds a replacement statement, which is
what lets copies of a :class:`~cobra.spice_sim.netlist_parsers.netlist.Netlist`
share them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The block a statement sits in, as ``(block kind, block name)``, e.g. ``("subckt", "buffer")``.
Scope = tuple[str, str]


class StatementKind(Enum):
    """What a statement is, as far as splitting can tell."""

    TITLE = "title"          # first line of a SPICE netlist, ignored by the simulator
    COMMENT = "comment"      # comment or blank line
    ELEMENT = "element"      # device or subcircuit instance
    DIRECTIVE = "directive"  # dot-directive or other control statement


@dataclass(frozen=True, slots=True)
class Token:
    """One code token and where it sits in its statement."""

    text: str
    line: int   # index into Statement.lines
    start: int  # column of the first character in that line
    end: int    # column after the last character


@dataclass(frozen=True, slots=True)
class Statement:
    """One logical netlist line."""

    lines: tuple[str, ...]           # verbatim physical lines, including newlines
    kind: StatementKind
    # Code tokens, without comments and continuation markers. ``None`` when not
    # lexed yet: a parser may classify statements it does not decode (e.g. the
    # body of a subcircuit) by keyword only; ``NetlistParser.tokens`` lexes them.
    tokens: tuple[Token, ...] | None = ()
    keyword: str = ""                # first token in upper case, e.g. ".AC" or "R1"
    scope: Scope | None = None       # enclosing block, None at top level
    # Data a rewrite removed from the text, e.g. the Touchstone path of a Qucs-S YLIN block.
    annotations: Mapping[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(self.lines)
