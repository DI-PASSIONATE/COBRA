"""Syntax shared by the SPICE family of netlists (Xyce, ngspice, ...).

Covers what these dialects have in common: the title line, ``*`` comment
lines, inline comments, ``+`` continuation lines, ``.SUBCKT``/``.ENDS`` blocks,
the positional layout of the standard devices, and ``.MODEL``, ``.INCLUDE``,
``.LIB``, ``.OPTIONS``, ``.PRINT`` and analysis directives.  A dialect
subclass adds its own devices, directives and analysis metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from cobra.spice_sim.netlist_parsers.netlist import (
    Include,
    Library,
    Netlist,
    NetlistElement,
    NetlistRecord,
    OptionsDirective,
    PrintDirective,
    SimulationDirective,
    Subcircuit,
)
from cobra.spice_sim.netlist_parsers.netlist_parser import NetlistParser
from cobra.spice_sim.netlist_parsers.statement import Scope, Statement, StatementKind, Token
from cobra.spice_sim.simulation_type import SimulationType

if TYPE_CHECKING:
    from collections.abc import Mapping

# Characters of a plain word: anything but whitespace, "=", brackets and quotes.
_WORD = r"""[^\s=(){}"']+"""
# A parenthesised group, allowing one nested level: "(0 0.2 95G)", "(V(a))".
_GROUP = r"\((?:[^()]|\([^()]*\))*\)"

# One code token. A parenthesised group is glued to the word right before it
# (v(out), SIN(0 1 1G)); a free-standing bracket is a token of its own, so a
# group can span continuation lines. "=" is always separate, so "k=v" and
# "k = v" both read as a parameter.
_TOKEN_RE = re.compile(
    rf"""
      "[^"]*"                               # double-quoted string
    | '[^']*'                               # single-quoted expression
    | \{{[^{{}}]*(?:\{{[^{{}}]*\}}[^{{}}]*)*\}}   # {{expression}}, one nested level
    | =
    | {_WORD}(?:{_GROUP}[^\s=(){{}}"']*)*      # word with glued groups
    | \S                                    # anything else, one character at a time
    """,
    re.VERBOSE,
)

_BRACKETS = ("(", ")")

# Printed signal tokens such as "v(Out)" or "I(VOut)".
_PROBE_RE = re.compile(r"^([VI])\(([^)]+)\)$", re.IGNORECASE)

# The first word of a statement, enough to classify one that is not lexed.
_KEYWORD_RE = re.compile(r"""\s*([^\s=(){}"';$]+)""")


@dataclass(frozen=True, slots=True)
class _Layout:
    """Where the parts of an element line sit, as token positions."""
    nodes: slice
    value: slice | None = None   # may be empty, meaning "insert here"
    model: int | None = None
    params: int = 1              # key=value parameters are read from this token on
    subtype: int | None = None


class SpiceNetlistParser(NetlistParser):
    """SPICE-family netlist syntax."""

    #: Characters that start an inline comment.
    inline_comment_markers: ClassVar[tuple[str, ...]] = (";", "$")
    #: Blocks whose statements are not top-level: opening keyword → (block kind, closing keyword).
    blocks: ClassVar[dict[str, tuple[str, str]]] = {".SUBCKT": ("subckt", ".ENDS")}
    #: Directives that run an analysis.
    analysis_keywords: ClassVar[frozenset[str]] = frozenset({".AC", ".DC", ".TRAN"})
    #: Directives that only modify an analysis' output (e.g. Xyce's .LIN).
    modifier_keywords: ClassVar[frozenset[str]] = frozenset()
    #: Element types whose value set_value replaces.
    value_types: ClassVar[frozenset[str]] = frozenset({"R", "C", "L", "V", "I"})
    #: Element types whose model set_model replaces.
    model_types: ClassVar[frozenset[str]] = frozenset({"D", "M", "Q", "X"})

    # -------------------------------------------------------------------------
    # Splitting
    # -------------------------------------------------------------------------

    def split(self, text: str, *, has_title: bool = True) -> list[Statement]:
        # Pass 1: group physical lines into statements. A "+" line joins the
        # previous code statement, together with any comment lines in between.
        groups: list[list[str]] = []
        kinds: list[StatementKind | None] = []  # None marks code
        last_code = -1
        for number, line in enumerate(text.splitlines(keepends=True)):
            stripped = line.lstrip()
            if number == 0 and has_title:
                kind: StatementKind | None = StatementKind.TITLE
            elif not stripped.strip() or stripped.startswith("*") or self._comment_only(line):
                kind = StatementKind.COMMENT
            elif stripped.startswith("+"):
                if last_code < 0:
                    kind = StatementKind.COMMENT  # nothing to continue
                else:
                    for absorbed in groups[last_code + 1:]:
                        groups[last_code].extend(absorbed)
                    del groups[last_code + 1:], kinds[last_code + 1:]
                    groups[last_code].append(line)
                    continue
            else:
                kind = None
            groups.append([line])
            kinds.append(kind)
            if kind is None:
                last_code = len(groups) - 1

        # Pass 2: tokenise code statements and track block scope. Statements
        # inside a block are never decoded, so they are only classified by
        # keyword; tokens() lexes them if needed. This keeps large inline
        # subcircuits (e.g. vector-fitted models) cheap to parse.
        closers = {closer for _, closer in self.blocks.values()}
        statements: list[Statement] = []
        stack: list[tuple[Scope, str]] = []  # (scope, closing keyword)
        for lines, kind in zip(groups, kinds, strict=True):
            scope = stack[-1][0] if stack else None
            if kind is not None:
                statements.append(Statement(tuple(lines), kind, scope=scope))
                continue
            statement = self._code_statement(tuple(lines), scope, lex=scope is None)
            keyword = statement.keyword
            if statement.tokens is None and (keyword in self.blocks or keyword in closers):
                statement = self._code_statement(tuple(lines), scope)
            if keyword in self.blocks:
                block_kind, closer = self.blocks[keyword]
                tokens = self.tokens(statement)
                stack.append(((block_kind, tokens[1].text if len(tokens) > 1 else ""), closer))
            elif stack and keyword == stack[-1][1]:
                stack.pop()
            statements.append(statement)
        return statements

    def _comment_only(self, line: str) -> bool:
        """Whether *line* holds nothing but an inline comment."""
        column = self.comment_start(line)
        return column is not None and not line[:column].strip()

    def comment_start(self, line: str) -> int | None:
        """Column where the inline comment of *line* starts, ignoring quoted text."""
        if not any(marker in line for marker in self.inline_comment_markers):
            return None
        quote = ""
        for column, char in enumerate(line):
            if quote:
                if char == quote:
                    quote = ""
            elif char in "\"'":
                quote = char
            elif char in self.inline_comment_markers:
                return column
        return None

    def _code_statement(
        self,
        lines: tuple[str, ...],
        scope: Scope | None,
        annotations: Mapping[str, str] | None = None,
        *,
        lex: bool = True,
    ) -> Statement:
        if not lex:
            match = _KEYWORD_RE.match(lines[0])
            keyword = match.group(1).upper() if match else ""
            return Statement(lines, _kind(keyword), None, keyword, scope, annotations or {})
        tokens: list[Token] = []
        for index, line in enumerate(lines):
            content = line.rstrip("\r\n")
            stripped = content.lstrip()
            if not stripped or stripped.startswith("*"):
                continue  # comment line between continuation lines
            begin = content.index("+") + 1 if index > 0 and stripped.startswith("+") else 0
            end = self.comment_start(content)
            tokens.extend(
                Token(match.group(), index, match.start(), match.end())
                for match in _TOKEN_RE.finditer(content, begin, len(content) if end is None else end)
            )
        keyword = tokens[0].text.upper() if tokens else ""
        return Statement(lines, _kind(keyword), tuple(tokens), keyword, scope, annotations or {})

    def tokens(self, statement: Statement) -> tuple[Token, ...]:
        if statement.tokens is not None:
            return statement.tokens
        return self._code_statement(statement.lines, statement.scope).tokens or ()

    # -------------------------------------------------------------------------
    # Decoding
    # -------------------------------------------------------------------------

    def decode(
        self, statement: Statement, line_index: int, statement_index: int
    ) -> NetlistRecord | None:
        if statement.keyword == ".SUBCKT":
            tokens = self.tokens(statement)
            # Every definition counts, nested ones included.
            if len(tokens) < 2:
                return None
            return Subcircuit(tokens[1].text, line_index, statement_index)
        if statement.scope is not None or not statement.tokens:
            return None  # devices inside a subcircuit are not top-level elements
        if statement.kind is StatementKind.ELEMENT:
            return self._decode_element(statement, line_index, statement_index)
        if statement.kind is StatementKind.DIRECTIVE:
            return self._decode_directive(statement, line_index, statement_index)
        return None

    def _decode_element(
        self, statement: Statement, line_index: int, statement_index: int
    ) -> NetlistElement:
        tokens = self.tokens(statement)
        texts = [token.text for token in tokens]
        name = texts[0]
        etype = name[0].upper()
        layout = self._layout(etype, tokens)
        nodes: list[str] = []
        value = model = subtype = None
        params: dict[str, str] = {}
        if layout is not None:
            nodes = texts[layout.nodes]
            if layout.value is not None:
                value = " ".join(texts[layout.value]) or None
            if layout.model is not None:
                model = texts[layout.model]
            if layout.subtype is not None:
                subtype = texts[layout.subtype]
            params = self._params(tokens[layout.params:])
        params.update(statement.annotations)
        return NetlistElement(
            name=name,
            etype=etype,
            subtype=subtype,
            line_index=line_index,
            statement_index=statement_index,
            nodes=nodes,
            value=value,
            model=model,
            params=params,
            port=self._port(etype, params),
        )

    def _decode_directive(
        self, statement: Statement, line_index: int, statement_index: int
    ) -> NetlistRecord | None:
        tokens = self.tokens(statement)
        keyword = statement.keyword
        if keyword == ".MODEL":
            flat = self._flatten(tokens)
            if len(flat) < 3:
                return None
            return NetlistElement(
                name=flat[1].text,
                etype="MODEL",
                subtype=None,
                line_index=line_index,
                statement_index=statement_index,
                nodes=[],
                value=None,
                model=flat[2].text,
                params=self._params(flat[3:]),
            )
        if keyword in (".INCLUDE", ".INC") and len(tokens) == 2:
            return Include(_unquote(tokens[1].text), line_index, statement_index)
        if keyword == ".LIB" and len(tokens) in (2, 3):
            entry = tokens[2].text if len(tokens) == 3 else None
            return Library(_unquote(tokens[1].text), entry, line_index, statement_index)
        if keyword == ".OPTIONS":
            positional, params = self._split_params(tokens[1:])
            category = positional[0].text if positional and positional[0] is tokens[1] else ""
            return OptionsDirective(category, _as_dict(params), line_index, statement_index)
        if keyword == ".PRINT" and len(tokens) >= 2:
            positional, params = self._split_params(tokens[2:])
            return PrintDirective(
                analysis=tokens[1].text.lower(),
                signals=[token.text for token in positional],
                kv_params=_as_dict(params),
                line_index=line_index,
                statement_index=statement_index,
            )
        if keyword in self.analysis_keywords or keyword in self.modifier_keywords:
            positional, params = self._split_params(tokens[1:])
            return SimulationDirective(
                directive=tokens[0].text,
                positional=[token.text for token in positional],
                kv_params=_as_dict(params),
                line_index=line_index,
                statement_index=statement_index,
                simulation_type=SimulationType.from_directive(keyword),
                is_analysis=keyword in self.analysis_keywords,
            )
        return None

    def is_component(self, element: NetlistElement, subcircuit_names: set[str]) -> bool:
        # An instance of a subcircuit defined in this file needs no surrogate.
        return element.etype == "X" and self.fold(element.model or "") not in subcircuit_names

    def probe_nodes(self, netlist: Netlist) -> list[str]:
        """Nodes printed with both ``V(X)`` and the current ``I(VX)`` of a 0 V probe source.

        This is the Qucs-S convention for a labelled node with a current probe.
        Both ``.PRINT hb`` and ``.PRINT tran`` lines count, since the same probe
        serves either analysis. Falls back to the netlist's V-elements when
        neither line exists.
        """
        voltages: dict[str, str] = {}   # upper-case node → node as written
        currents: set[str] = set()      # upper-case source name
        for directive in netlist.print_directives:
            if directive.analysis not in ("hb", "tran"):
                continue
            for token in directive.signals:
                match = _PROBE_RE.match(token)
                if not match:
                    continue
                kind, arg = match.group(1).upper(), match.group(2).strip()
                if kind == "V":
                    voltages.setdefault(arg.upper(), arg)
                else:
                    currents.add(arg.upper())

        if not voltages:
            # No .PRINT hb/tran line: derive candidates from the probe sources themselves.
            for element in netlist.list_elements(["V"]):
                if element.nodes:
                    node = element.nodes[0]
                    if element.name.upper() == f"V{node.upper()}":
                        voltages.setdefault(node.upper(), node)
                        currents.add(element.name.upper())

        return [node for key, node in voltages.items() if f"V{key}" in currents]

    def _layout(self, etype: str, tokens: tuple[Token, ...]) -> _Layout | None:
        """Token positions of an element line, or ``None`` if the line is too short."""
        count = len(tokens)
        if etype in ("R", "C", "L"):
            # <name> <n+> <n-> <value> [params...]
            if count >= 5 and tokens[4].text == "=":
                return _Layout(slice(1, 3), params=3)  # value given as a parameter, e.g. R=1k
            return _Layout(slice(1, 3), value=slice(3, 4), params=4) if count >= 4 else None
        if etype == "M":
            # <name> <drain> <gate> <source> <bulk> <model> [params...]
            return _Layout(slice(1, 5), model=5, params=6) if count >= 6 else None
        if etype == "Q":
            # <name> <collector> <base> <emitter> <model> [params...]
            return _Layout(slice(1, 4), model=4, params=5) if count >= 5 else None
        if etype == "D":
            # <name> <anode> <cathode> <model> [params...]
            return _Layout(slice(1, 3), model=3, params=4) if count >= 4 else None
        if etype == "X":
            # <name> [nodes...] <subckt_name> [key=value...]
            first = self._first_param(tokens)
            end = count if first is None else first
            return _Layout(slice(1, end - 1), model=end - 1, params=end) if end >= 3 else None
        if etype in ("V", "I"):
            # <name> <n+> <n-> [value or waveform...]: the value is all of the rest
            return _Layout(slice(1, 3), value=slice(3, count), params=count) if count >= 3 else None
        # Fallback for unrecognised types.
        return _Layout(
            slice(1, 3) if count >= 3 else slice(0, 0),
            value=slice(count - 1, count) if count >= 4 else None,
            params=3,
        )

    def _port(self, etype: str, params: dict[str, str]) -> int | None:  # noqa: ARG002
        """Port number of an element that is a netlist port, else ``None``."""
        return None

    # -------------------------------------------------------------------------
    # Editing
    # -------------------------------------------------------------------------

    def set_value(self, statement: Statement, value: str) -> Statement:
        etype = _etype(statement)
        if etype not in self.value_types:
            raise ValueError(
                f"set_value is for R/C/L and simple V/I. Use set_param or set_model for '{etype}'."
            )
        layout = self._layout(etype, self.tokens(statement))
        if layout is None or layout.value is None:
            raise ValueError(
                f"{etype} line too short, or its value is a key=value parameter; "
                "use set_param for that."
            )
        return self._replace(statement, layout.value.start, layout.value.stop, value)

    def set_model(self, statement: Statement, model: str) -> Statement:
        etype = _etype(statement)
        if etype not in self.model_types:
            raise ValueError(f"set_model not supported for type '{etype}'")
        layout = self._layout(etype, self.tokens(statement))
        if layout is None or layout.model is None:
            raise ValueError(f"{etype} line too short.")
        return self._replace(statement, layout.model, layout.model + 1, model)

    def set_param(self, statement: Statement, key: str, value: str) -> Statement:
        tokens = self.tokens(statement)
        is_model = statement.keyword == ".MODEL"
        # .MODEL lines have two positional tokens before parameters start.
        candidates = self._flatten(tokens)[3:] if is_model else tokens[1:]
        for name, current in self._split_params(candidates)[1]:
            if name.text.casefold() == key.casefold():
                return self._splice(statement, [(current.line, current.start, current.end, value)])
        last = tokens[-1]
        if is_model and last.text.endswith(")"):
            # Keep the new parameter inside the parenthesised list.
            return self._splice(statement, [(last.line, last.end - 1, last.end - 1, f" {key}={value}")])
        return self._splice(statement, [(last.line, last.end, last.end, f" {key}={value}")])

    def set_directive(
        self, statement: Statement, positional: list[str], params: Mapping[str, str]
    ) -> Statement:
        arguments = [item for item in positional if item]
        arguments.extend(f"{key}={value}" for key, value in params.items())
        return self._replace(statement, 1, len(self.tokens(statement)), " ".join(arguments))

    def _replace(self, statement: Statement, start: int, stop: int, text: str) -> Statement:
        """Replace tokens ``[start, stop)`` with *text*; an empty range inserts *text* there."""
        tokens = self.tokens(statement)
        if start >= stop:
            if not text:
                return statement
            if start < len(tokens):
                token = tokens[start]
                return self._splice(statement, [(token.line, token.start, token.start, f"{text} ")])
            token = tokens[-1]
            return self._splice(statement, [(token.line, token.end, token.end, f" {text}")])
        # The replaced tokens may span continuation lines: the text goes where the
        # first one was, and the others are cut from their lines.
        by_line: dict[int, list[Token]] = {}
        for token in tokens[start:stop]:
            by_line.setdefault(token.line, []).append(token)
        edits = [
            (line, group[0].start, group[-1].end, text if position == 0 else "")
            for position, (line, group) in enumerate(by_line.items())
        ]
        return self._splice(statement, edits)

    def _splice(self, statement: Statement, edits: list[tuple[int, int, int, str]]) -> Statement:
        """Apply ``(line, start, end, text)`` replacements and re-tokenise the statement."""
        lines = list(statement.lines)
        for line, start, end, text in sorted(edits, reverse=True):
            lines[line] = lines[line][:start] + text + lines[line][end:]
        return self._code_statement(tuple(lines), statement.scope, statement.annotations)

    # -------------------------------------------------------------------------
    # Token helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _first_param(tokens: tuple[Token, ...]) -> int | None:
        """Index of the first ``key`` of a ``key=value`` pair, if any."""
        return next(
            (i for i in range(1, len(tokens) - 1) if tokens[i + 1].text == "="),
            None,
        )

    @staticmethod
    def _split_params(
        tokens: tuple[Token, ...] | list[Token],
    ) -> tuple[list[Token], list[tuple[Token, Token]]]:
        """Split *tokens* into positional tokens and ``(key, value)`` pairs; brackets are dropped.

        A value is every token that touches the one after ``=`` (``-{dv}``), merged
        into a single token spanning them.
        """
        positional: list[Token] = []
        params: list[tuple[Token, Token]] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if (
                index + 2 < len(tokens)
                and tokens[index + 1].text == "="
                and token.text not in ("=", *_BRACKETS)
            ):
                value = tokens[index + 2]
                index += 3
                while (
                    index < len(tokens)
                    and tokens[index].line == value.line
                    and tokens[index].start == value.end
                    and tokens[index].text != "="
                ):
                    following = tokens[index]
                    value = Token(value.text + following.text, value.line, value.start, following.end)
                    index += 1
                params.append((token, value))
                continue
            if token.text not in ("=", *_BRACKETS):
                positional.append(token)
            index += 1
        return positional, params

    def _params(self, tokens: tuple[Token, ...] | list[Token]) -> dict[str, str]:
        return _as_dict(self._split_params(tokens)[1])

    @staticmethod
    def _flatten(tokens: tuple[Token, ...] | list[Token]) -> list[Token]:
        """Open glued groups (``SIN(0 1 1G)`` → ``SIN 0 1 1G``) and drop free brackets."""
        flat: list[Token] = []
        for token in tokens:
            text = token.text
            if text in _BRACKETS:
                continue
            opening = text.find("(")
            if opening <= 0 or not text.endswith(")") or text[0] in "{\"'":
                flat.append(token)
                continue
            flat.append(Token(text[:opening], token.line, token.start, token.start + opening))
            flat.extend(
                Token(match.group(), token.line, token.start + match.start(), token.start + match.end())
                for match in _TOKEN_RE.finditer(text, opening + 1, len(text) - 1)
                if match.group() not in _BRACKETS
            )
        return flat

    def _rewrite(
        self, statement: Statement, code: str, annotations: Mapping[str, str] | None = None
    ) -> Statement:
        """A one-line statement with *code*, keeping the inline comment and line ending."""
        content = statement.lines[0].rstrip("\r\n")
        column = self.comment_start(content)
        comment = content[column:].rstrip() if column is not None else ""
        last = statement.lines[-1]
        newline = last[len(last.rstrip("\r\n")):]
        line = f"{code} {comment}{newline}" if comment else f"{code}{newline}"
        return self._code_statement((line,), statement.scope, annotations or statement.annotations)


def _kind(keyword: str) -> StatementKind:
    if not keyword:
        return StatementKind.COMMENT  # e.g. a line holding only an inline comment
    return StatementKind.ELEMENT if keyword[0].isalpha() else StatementKind.DIRECTIVE


def _etype(statement: Statement) -> str:
    return "MODEL" if statement.keyword == ".MODEL" else statement.keyword[:1]


def _as_dict(params: list[tuple[Token, Token]]) -> dict[str, str]:
    return {key.text: value.text for key, value in params}


def _unquote(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text
