"""VACASK netlist dialect.

VACASK (https://codeberg.org/arpadbuermen/VACASK) reads a Spectre-like
netlist: instances are written ``name (node ...) master key=value ...``,
devices are Verilog-A modules bound by ``model <name> <module>``, and the
analyses, options and ``save`` requests sit in a ``control ... endc`` block.
Names are case-sensitive and the first line of a netlist is a title VACASK
ignores.

COBRA's conventions for a VACASK netlist:

* A port is a ``(vsource, resistor)`` pair listed in an ``acsp`` analysis,
  ``ports=["vp1", "rp1", "vp2", "rp2"]``: port *k* is the *k*-th pair, named
  after its source, with the resistor as its reference impedance.
* A surrogate component is an instance whose master is defined neither in the
  netlist nor in a file it includes (see :meth:`VacaskNetlistParser.parse_file`).
* Analysis arguments are named, but the editable ones are exposed in the same
  slots as their Xyce counterparts (e.g. ``mode, points, from, to`` for an AC
  sweep), so configuration checks read either dialect alike.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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
from cobra.spice_sim.simulation_type import SimulationType, SimulationTypeMetadata

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

logger = logging.getLogger(__name__)

# One code token. Comments are cut before lexing; "/" is no word character so
# that "//" and "/*" are always seen.
_TOKEN_RE = re.compile(
    r"""
      "(?:[^"\\]|\\.)*"            # double-quoted string with escapes
    | <<<\w+                       # start of a heredoc (embedded file)
    | [<>!=]=                      # comparison operators
    | [=()\[\]{},]                 # punctuation
    | [^\s=()\[\]{},"<>!/]+        # word: name, number or expression
    | \S                           # anything else, one character at a time
    """,
    re.VERBOSE,
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_$][\w$.:]*$")
_OPENING = frozenset("([{")
_CLOSING = frozenset(")]}")
_QUOTED_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
_PROBE_RE = re.compile(r"^([vi])\((.+)\)$", re.IGNORECASE)

#: Statements that are not instances even when a "(" follows the keyword.
_KEYWORDS = frozenset({
    "model", "subckt", "ends", "include", "load", "ground", "global", "parameters",
    "control", "endc", "embed", "section", "endsection", "@if", "@elseif", "@else", "@end",
})
#: Blocks whose statements are not top-level: opening keyword → (block kind, closing keyword).
_BLOCKS = {"subckt": ("subckt", "ends"), "control": ("control", "endc")}
_CONTROL: Scope = ("control", "")
#: Prefix of the lines of a statement COBRA switched off, see VacaskNetlistParser.disable.
DISABLED_PREFIX = "//cobra-off "

#: Verilog-A modules of the VACASK device library → SPICE-style element letter.
_MODULE_ETYPES = {
    "resistor": "R", "capacitor": "C", "inductor": "L",
    "vsource": "V", "isource": "I",
    "vcvs": "E", "vccs": "G", "ccvs": "H", "cccs": "F", "mutual": "K",
    "diode": "D",
}
#: The instance parameter :meth:`VacaskNetlistParser.set_value` replaces, per element letter.
_VALUE_PARAMS = {"R": "r", "C": "c", "L": "l", "V": "dc", "I": "dc"}

#: Analysis keyword → simulation type.
_ANALYSES = {
    "acsp": SimulationType.AC,
    "hb": SimulationType.HB,
    "tran": SimulationType.TRAN,
    "noise": SimulationType.NOISE,
    "hbnoise": SimulationType.HBNOISE,
}
#: Analysis parameters whose value is a string literal.
_STRING_PARAMS = frozenset({"mode", "in", "out", "truncate", "sample", "icmode", "store"})
#: Analysis parameters whose value is always a vector.
_VECTOR_PARAMS = frozenset({"freq", "values", "ports"})
#: Analysis parameters that are a scalar or, with several items, a vector.
_SCALAR_OR_VECTOR_PARAMS = frozenset({"nharm"})
#: Spur selectors: integers are tone weights (a vector), anything else a frequency in Hz.
_SPUR_PARAMS = frozenset({"inspur", "outspur"})
_INTEGER_RE = re.compile(r"^[+-]?\d+$")

# VACASK SI prefixes: case matters ("m" is milli, "M" is mega).
_NUMBER_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)([A-Za-z_]*)$")
_PREFIXES = {
    "a": 1e-18, "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
    "k": 1e3, "K": 1e3, "M": 1e6, "x": 1e6, "X": 1e6, "G": 1e9, "T": 1e12,
}

_SWEEP = {
    "mode":   "Frequency sweep spacing: lin, dec or oct.",
    "points": "Points per sweep (lin: intervals, dec/oct: points per decade/octave).",
    "from":   "Start frequency (e.g. 100G for 100 GHz).",
    "to":     "Stop frequency (e.g. 200G for 200 GHz).",
}
_SWEEP_DEFAULTS = {"mode": "lin", "points": "500", "from": "1G", "to": "10G"}

_VACASK_METADATA: dict[SimulationType, SimulationTypeMetadata] = {
    SimulationType.AC: SimulationTypeMetadata(
        positional_param_names=["mode", "points", "from", "to"],
        positional_param_descriptions=_SWEEP,
        positional_param_defaults=_SWEEP_DEFAULTS,
    ),
    SimulationType.HB: SimulationTypeMetadata(
        positional_param_names=["freq"],
        positional_param_descriptions={
            "freq": "Space-separated list of fundamental frequencies.\n"
                    "Single tone: e.g. 130G. Multi-tone: e.g. 95G 10G.\n"
                    "Set the harmonics per tone with nharm (default 4).",
        },
        positional_param_defaults={"freq": "1G"},
    ),
    SimulationType.TRAN: SimulationTypeMetadata(
        positional_param_names=["step", "stop", "start", "maxstep"],
        positional_param_descriptions={
            "step":    "Initial time step.",
            "stop":    "Simulation stop time.",
            "start":   "Time at which results start being recorded (default 0). Everything\n"
                       "before it is the start-up transient and is left out of the spectrum.\n"
                       "stop - start sets the FFT resolution and should be a whole number of\n"
                       "signal periods.",
            "maxstep": "Maximum time step (0: no limit beyond step).",
        },
        positional_param_defaults={"step": "1p", "stop": "100n", "start": "0", "maxstep": "1p"},
    ),
    SimulationType.NOISE: SimulationTypeMetadata(
        positional_param_names=["mode", "points", "from", "to"],
        positional_param_descriptions=_SWEEP,
        positional_param_defaults=_SWEEP_DEFAULTS,
    ),
    SimulationType.HBNOISE: SimulationTypeMetadata(
        positional_param_names=["mode", "points", "from", "to"],
        positional_param_descriptions={
            **_SWEEP,
            "from": "Start offset (output sideband) frequency.",
            "to":   "Stop offset (output sideband) frequency.",
        },
        positional_param_defaults=_SWEEP_DEFAULTS,
    ),
}


def vacask_float(text: str) -> float:
    """Convert a VACASK number such as ``130G``, ``1.5meg`` or ``2m`` to a float.

    Raises :class:`ValueError` for anything else, e.g. an expression.
    """
    match = _NUMBER_RE.match(text.strip())
    if not match:
        raise ValueError(f"Not a VACASK number: {text!r}")
    mantissa, suffix = float(match.group(1)), match.group(2)
    if not suffix:
        return mantissa
    for prefix, factor in (("meg", 1e6), ("mil", 25.4e-6)):
        if suffix.startswith(prefix):
            return mantissa * factor
    if suffix[0] in _PREFIXES:
        return mantissa * _PREFIXES[suffix[0]]
    return mantissa  # a unit without a prefix, e.g. "50Ohm"


def _unquote(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return text[1:-1]
    return text


@dataclass(frozen=True, slots=True)
class _LexState:
    """What a line leaves open for the next one."""
    depth: int = 0              # open brackets
    block_comment: bool = False  # inside /* ... */
    heredoc: str | None = None   # marker of an open <<<MARKER block
    continued: bool = False      # the line ended with a backslash

    @property
    def closed(self) -> bool:
        return self.depth <= 0 and not self.block_comment and self.heredoc is None and not self.continued


class VacaskNetlistParser(NetlistParser):
    """VACASK netlist syntax."""

    # -------------------------------------------------------------------------
    # Entry points
    # -------------------------------------------------------------------------

    def parse_file(
        self, path: str | Path, encoding: str = "utf-8", *, has_title: bool = True
    ) -> Netlist:
        """Read and parse the netlist at *path*, noting the masters its includes define.

        Include files are looked up the way VACASK does: next to the including
        file, in the current directory, then on the include path from
        ``SIM_INCLUDE_PATH`` and the ``.vacaskrc.toml`` files.  An instance of a
        master found there (a PDK device, say) is not a surrogate component.
        """
        path = Path(path)
        netlist = super().parse_file(path, encoding, has_title=has_title)
        masters = self.included_masters(netlist, path.resolve().parent)
        if not masters:
            return netlist
        return Netlist(self, list(netlist.statements), masters)

    def fold(self, name: str) -> str:
        return name  # VACASK names are case-sensitive

    # -------------------------------------------------------------------------
    # Splitting
    # -------------------------------------------------------------------------

    def split(self, text: str, *, has_title: bool = True) -> list[Statement]:
        # Pass 1: group physical lines into statements. A statement runs on
        # while a bracket, block comment or heredoc is open, or after a line
        # that ends with a backslash.
        groups: list[tuple[list[str], list[Token], bool]] = []  # (lines, tokens, is_title)
        lines: list[str] = []
        tokens: list[Token] = []
        state = _LexState()
        for number, line in enumerate(text.splitlines(keepends=True)):
            if number == 0 and has_title:
                groups.append(([line], [], True))
                continue
            line_tokens, state = self._scan(line, state, len(lines))
            lines.append(line)
            tokens.extend(line_tokens)
            if state.closed:
                groups.append((lines, tokens, False))
                lines, tokens, state = [], [], _LexState()
        if lines:
            groups.append((lines, tokens, False))

        # Pass 2: classify and track block scope.
        statements: list[Statement] = []
        stack: list[tuple[Scope, str]] = []  # (scope, closing keyword)
        for group_lines, group_tokens, is_title in groups:
            scope = stack[-1][0] if stack else None
            if is_title:
                statements.append(Statement(tuple(group_lines), StatementKind.TITLE))
                continue
            statement = self._statement(tuple(group_lines), tuple(group_tokens), scope)
            keyword = statement.keyword
            if keyword in _BLOCKS:
                block_kind, closer = _BLOCKS[keyword]
                name = group_tokens[1].text if block_kind == "subckt" and len(group_tokens) > 1 else ""
                stack.append(((block_kind, name), closer))
            elif stack and keyword == stack[-1][1]:
                stack.pop()
            statements.append(statement)
        return statements

    def _scan(self, line: str, state: _LexState, line_number: int) -> tuple[list[Token], _LexState]:
        """Tokens of one physical line, given what the previous lines left open."""
        content = line.rstrip("\r\n")
        tokens: list[Token] = []
        depth, block_comment, heredoc = state.depth, state.block_comment, state.heredoc
        position = 0
        if heredoc is not None:
            end = content.find(">>>" + heredoc)
            if end < 0:
                return tokens, dataclasses.replace(state, continued=False)
            position, heredoc = end + 3 + len(heredoc), None
        while position < len(content):
            if block_comment:
                end = content.find("*/", position)
                if end < 0:
                    break
                position, block_comment = end + 2, False
                continue
            while position < len(content) and content[position].isspace():
                position += 1
            if position >= len(content) or content.startswith("//", position):
                break
            if content.startswith("/*", position):
                position, block_comment = position + 2, True
                continue
            match = _TOKEN_RE.match(content, position)
            if match is None:  # cannot happen: "\S" matches any character
                break
            text = match.group()
            tokens.append(Token(text, line_number, match.start(), match.end()))
            position = match.end()
            if text.startswith("<<<"):
                heredoc = text[3:]
                break  # the embedded content starts on the next line
            if text in _OPENING:
                depth += 1
            elif text in _CLOSING:
                depth -= 1
        continued = False
        if tokens and tokens[-1].text.endswith("\\") and not block_comment and heredoc is None:
            last = tokens.pop()
            if last.text != "\\":
                tokens.append(Token(last.text[:-1], last.line, last.start, last.end - 1))
            continued = True
        return tokens, _LexState(depth, block_comment, heredoc, continued)

    def _lex(self, lines: tuple[str, ...]) -> tuple[Token, ...]:
        tokens: list[Token] = []
        state = _LexState()
        for number, line in enumerate(lines):
            line_tokens, state = self._scan(line, state, number)
            tokens.extend(line_tokens)
        return tuple(tokens)

    def _statement(
        self,
        lines: tuple[str, ...],
        tokens: tuple[Token, ...],
        scope: Scope | None,
        annotations: Mapping[str, str] | None = None,
    ) -> Statement:
        keyword = tokens[0].text if tokens else ""
        if not keyword:
            kind = StatementKind.COMMENT
        elif (
            scope != _CONTROL
            and keyword not in _KEYWORDS
            and len(tokens) > 1
            and tokens[1].text == "("
        ):
            kind = StatementKind.ELEMENT
        else:
            kind = StatementKind.DIRECTIVE
        return Statement(lines, kind, tokens, keyword, scope, annotations or {})

    def normalize(self, statements: list[Statement]) -> list[Statement]:
        """Note on each top-level instance the device module its master stands for.

        The ``model`` statements bind master names to modules anywhere in the
        netlist, so this needs every statement; decoding then reads the module
        from the annotation.
        """
        models: dict[str, str] = {}
        for statement in statements:
            tokens = statement.tokens or ()
            if statement.scope is None and statement.keyword == "model" and len(tokens) >= 3:
                models[tokens[1].text] = tokens[2].text
        result = []
        for statement in statements:
            module = None
            if statement.kind is StatementKind.ELEMENT and statement.scope is None:
                master = self._master(statement)
                module = models.get(master or "") or (master if master in _MODULE_ETYPES else None)
            result.append(
                statement if module is None
                else dataclasses.replace(statement, annotations={"module": module})
            )
        return result

    # -------------------------------------------------------------------------
    # Decoding
    # -------------------------------------------------------------------------

    def decode(
        self, statement: Statement, line_index: int, statement_index: int
    ) -> NetlistRecord | None:
        tokens = self.tokens(statement)
        keyword = statement.keyword
        if keyword == "subckt":
            # Every definition counts, nested ones included.
            if len(tokens) < 2:
                return None
            pins = tuple(_unquote(token.text) for token in tokens[3:self._closing(tokens, 2)])
            return Subcircuit(tokens[1].text, line_index, statement_index, pins)
        if statement.scope == _CONTROL:
            return self._decode_control(statement, line_index, statement_index)
        if statement.scope is not None:
            return None  # devices inside a subcircuit are not top-level elements
        if statement.kind is StatementKind.ELEMENT:
            return self._decode_element(statement, line_index, statement_index)
        if keyword == "model" and len(tokens) >= 3:
            return NetlistElement(
                name=tokens[1].text,
                etype="MODEL",
                subtype=None,
                line_index=line_index,
                statement_index=statement_index,
                nodes=[],
                value=None,
                model=tokens[2].text,
                params=self._params(tokens, 3),
            )
        if keyword == "include" and len(tokens) >= 2:
            path = _unquote(tokens[1].text)
            section = self._params(tokens, 2).get("section")
            if section is not None:
                return Library(path, _unquote(section), line_index, statement_index)
            return Include(path, line_index, statement_index)
        return None

    def _decode_element(
        self, statement: Statement, line_index: int, statement_index: int
    ) -> NetlistElement:
        tokens = self.tokens(statement)
        close = self._closing(tokens, 1)
        nodes = [_unquote(token.text) for token in tokens[2:close] if token.text != ","]
        master_index = close + 1
        master = tokens[master_index].text if master_index < len(tokens) else None
        params = self._params(tokens, master_index + 1)
        module = statement.annotations.get("module")
        etype = _MODULE_ETYPES.get(module.removeprefix("sp_"), "DEVICE") if module else "X"
        value_param = _VALUE_PARAMS.get(etype)
        return NetlistElement(
            name=tokens[0].text,
            etype=etype,
            subtype=None,
            line_index=line_index,
            statement_index=statement_index,
            nodes=nodes,
            value=params.get(value_param) if value_param else None,
            model=master,
            params=params,
        )

    def _decode_control(
        self, statement: Statement, line_index: int, statement_index: int
    ) -> NetlistRecord | None:
        tokens = self.tokens(statement)
        keyword = statement.keyword
        if keyword == "analysis" and len(tokens) >= 3:
            analysis = tokens[2].text
            sim_type = _ANALYSES.get(analysis, SimulationType.UNKNOWN)
            params = self._params(tokens, 3)
            names = self.analysis_metadata(sim_type).positional_param_names
            positional = [self._slot_value(name, params.pop(name, "")) for name in names]
            if names == ["freq"]:
                positional = positional[0].split() if positional else []
            while positional and not positional[-1]:
                positional.pop()
            return SimulationDirective(
                directive=analysis,
                positional=positional,
                kv_params=params,
                line_index=line_index,
                statement_index=statement_index,
                simulation_type=sim_type,
                # Analyses COBRA does not run (op, ac, ...) never set the netlist's type.
                is_analysis=sim_type is not SimulationType.UNKNOWN,
                name=tokens[1].text,
            )
        if keyword == "options":
            return OptionsDirective("", self._params(tokens, 1), line_index, statement_index)
        if keyword == "save":
            return PrintDirective(
                analysis="",
                signals=self._words(tokens[1:]),
                kv_params={},
                line_index=line_index,
                statement_index=statement_index,
            )
        return None

    @staticmethod
    def _slot_value(name: str, text: str) -> str:
        """The editable form of an analysis argument: unquoted, a vector as spaced items."""
        text = _unquote(text.strip())
        if name in _VECTOR_PARAMS and text.startswith("[") and text.endswith("]"):
            return " ".join(item.strip() for item in text[1:-1].split(",") if item.strip())
        return text

    def instance_master(self, statement: Statement) -> str | None:
        return self._master(statement) if statement.kind is StatementKind.ELEMENT else None

    def surrogate_include(self, component: str) -> str:
        return f'include "{component}.inc"\n'

    def is_component(self, element: NetlistElement, subcircuit_names: set[str]) -> bool:
        # An instance of a subcircuit defined in this file or an include needs no surrogate.
        return element.etype == "X" and (element.model or "") not in subcircuit_names

    def probe_nodes(self, netlist: Netlist) -> list[str]:
        """Nodes saved with both ``v(X)`` and the current ``i(vX)`` of a 0 V probe source,
        followed by the port sources.

        Falls back to the vsources named ``v<node>`` (or ``V<node>``) on that
        node when the control block saves neither. A port is measured by name of
        its source (e.g. ``Vout``): the power into it is ``2 R |I|^2``, from the
        current of its source and its resistor ``R`` (see :class:`VacaskSimulator`).
        """
        voltages: dict[str, str] = {}
        currents: set[str] = set()
        for directive in netlist.print_directives:
            for signal in directive.signals:
                match = _PROBE_RE.match(signal)
                if not match:
                    continue
                arg = _unquote(match.group(2).strip())
                if match.group(1).lower() == "v":
                    voltages.setdefault(arg, arg)
                else:
                    currents.add(arg)
        if not voltages:
            for element in netlist.list_elements(["V"]):
                if element.nodes and element.name[:1] in "vV" and element.name[1:] == element.nodes[0]:
                    voltages.setdefault(element.nodes[0], element.nodes[0])
                    currents.add(element.name)
        nodes = [
            node for node in voltages
            if f"v{node}" in currents or f"V{node}" in currents
        ]
        return nodes + [source for source in self.port_resistances(netlist) if source not in nodes]

    def port_resistances(self, netlist: Netlist) -> dict[str, float]:
        """The reference resistance of each port, by source name, where it is a number."""
        resistances: dict[str, float] = {}
        for source, resistor in self.port_pairs(netlist):
            if netlist.has_element(resistor):
                try:
                    resistances[source] = vacask_float(netlist.get_element(resistor).params.get("r", ""))
                except ValueError:
                    continue
        return resistances

    # -------------------------------------------------------------------------
    # Ports
    # -------------------------------------------------------------------------

    def port_pairs(self, netlist: Netlist) -> list[tuple[str, str]]:
        """The ``(vsource, resistor)`` pairs of the ports, in port order.

        The first ``acsp`` analysis declares them; one COBRA switched off (see
        :meth:`disable`) still counts. Without one, every top-level vsource
        from a ground to a node that only a resistor shares is a port, in
        netlist order: supplies and bias sources feed more than that.
        """
        for candidate in (netlist, self._disabled(netlist)):
            for directive in candidate.simulation_directives:
                if directive.directive == "acsp" and "ports" in directive.kv_params:
                    names = _QUOTED_RE.findall(directive.kv_params["ports"])
                    return list(zip(names[0::2], names[1::2], strict=False))
        return self._detected_ports(netlist)

    def _detected_ports(self, netlist: Netlist) -> list[tuple[str, str]]:
        grounds = self.grounds(netlist)
        elements = [element for element in netlist.list_elements() if element.etype != "MODEL"]
        by_node: dict[str, list[NetlistElement]] = {}
        for element in elements:
            for node in set(element.nodes):
                by_node.setdefault(node, []).append(element)
        pairs: list[tuple[str, str]] = []
        for source in elements:
            if source.etype != "V" or len(source.nodes) != 2:
                continue
            signal = [node for node in source.nodes if node not in grounds]
            if len(signal) != 1:
                continue
            others = [element for element in by_node[signal[0]] if element is not source]
            if len(others) == 1 and others[0].etype == "R" and len(others[0].nodes) == 2:
                pairs.append((source.name, others[0].name))
        return pairs

    def grounds(self, netlist: Netlist) -> set[str]:
        """The node names the netlist declares as ground (``ground GND``); ``{"0"}`` without any."""
        names = {
            token.text
            for statement in netlist.statements
            if statement.keyword == "ground" and statement.scope is None
            for token in self.tokens(statement)[1:]
        }
        return names or {"0"}

    def disable(self, statement: Statement) -> list[str]:
        """The lines of *statement* commented out, in a form :meth:`port_pairs` still reads."""
        return [DISABLED_PREFIX + line if line.strip() else line for line in statement.lines]

    def _disabled(self, netlist: Netlist) -> Netlist:
        """The control statements COBRA switched off in *netlist*, parsed on their own."""
        text = "".join(
            line[len(DISABLED_PREFIX):]
            for statement in netlist.statements
            for line in statement.lines
            if line.startswith(DISABLED_PREFIX)
        )
        return self.parse(f"control\n{text}\nendc\n" if text else "", has_title=False)

    def ports(self, netlist: Netlist) -> dict[str, int]:
        return {source: number for number, (source, _) in enumerate(self.port_pairs(netlist), 1)}

    def port_sources(self, netlist: Netlist) -> dict[str, dict[str, float]]:
        """Drive and reference impedance of each port, from its vsource and resistor."""
        sources: dict[str, dict[str, float]] = {}
        for source, resistor in self.port_pairs(netlist):
            if not netlist.has_element(source):
                continue
            params = netlist.get_element(source).params
            result: dict[str, float] = {}
            fields = {"ac_amplitude": params.get("mag")}
            if _unquote(params.get("type", "")) == "sine":
                fields |= {"sin_amplitude": params.get("ampl"), "sin_frequency": params.get("freq")}
            if netlist.has_element(resistor):
                fields["z0"] = netlist.get_element(resistor).params.get("r")
            for key, text in fields.items():
                try:
                    result[key] = vacask_float(text or "")
                except ValueError:
                    continue
            if "sin_amplitude" in result or "ac_amplitude" in result:
                sources[source] = result
        return sources

    # -------------------------------------------------------------------------
    # Included files
    # -------------------------------------------------------------------------

    def resolve_include(self, file_path: str, directory: Path) -> Path:
        """Search like VACASK: next to the netlist, the current directory, then the include path."""
        found = _resolve(file_path, directory, _include_path(directory))
        return found if found is not None else super().resolve_include(file_path, directory)

    def included_masters(self, netlist: Netlist, directory: Path) -> dict[str, tuple[str, ...]]:
        """The subcircuits and models the files *netlist* includes define, with their pins.

        A model has no pins.
        """
        masters: dict[str, tuple[str, ...]] = {}
        seen: set[Path] = set()
        search = _include_path(directory)
        pending = [(statement, directory) for statement in netlist.statements]
        while pending:
            statement, base = pending.pop()
            if statement.keyword != "include" or len(self.tokens(statement)) < 2:
                continue
            tokens = self.tokens(statement)
            name = _unquote(tokens[1].text)
            path = _resolve(name, base, search)
            if path is None:
                logger.debug("Include '%s' not found; its masters stay unknown", name)
                continue
            if path in seen:
                continue
            seen.add(path)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.debug("Could not read include '%s': %s", path, exc)
                continue
            if "lang" in self._params(tokens, 2):
                masters.update(_foreign_masters(text, path.parent, search, seen))
                continue
            included = self.parse(text, has_title=False)
            for index, inner in enumerate(included.statements):
                record = self.decode(inner, 0, index)
                if isinstance(record, Subcircuit):
                    masters[record.name] = record.pins
                elif inner.keyword == "model" and len(self.tokens(inner)) > 1:
                    masters.setdefault(self.tokens(inner)[1].text, ())
            pending.extend((inner, path.parent) for inner in included.statements)
        return masters

    # -------------------------------------------------------------------------
    # Editing
    # -------------------------------------------------------------------------

    def set_value(self, statement: Statement, value: str) -> Statement:
        module = statement.annotations.get("module", "")
        param = _VALUE_PARAMS.get(_MODULE_ETYPES.get(module.removeprefix("sp_"), ""))
        if statement.kind is not StatementKind.ELEMENT or param is None:
            raise ValueError(
                f"set_value is for resistors, capacitors, inductors and DC sources; "
                f"use set_param for '{statement.keyword}'."
            )
        return self.set_param(statement, param, value)

    def set_model(self, statement: Statement, model: str) -> Statement:
        if statement.kind is not StatementKind.ELEMENT:
            raise ValueError(f"set_model is for instances, not '{statement.keyword}'")
        tokens = self.tokens(statement)
        index = self._closing(tokens, 1) + 1
        if index >= len(tokens):
            raise ValueError(f"Instance '{statement.keyword}' has no master.")
        token = tokens[index]
        return self._splice(statement, [(token.line, token.start, token.end, model)])

    def set_param(self, statement: Statement, key: str, value: str) -> Statement:
        tokens = self.tokens(statement)
        span = self._param_spans(tokens, self._params_start(statement)).get(key)
        if span is not None:
            return self._replace(statement, tokens[span[0]], tokens[span[1] - 1], value)
        last = tokens[-1]
        return self._splice(statement, [(last.line, last.end, last.end, f" {key}={value}")])

    def set_directive(
        self, statement: Statement, positional: list[str], params: Mapping[str, str]
    ) -> Statement:
        """Rewrite the arguments of an ``analysis`` or ``options`` statement.

        The slots of an analysis are written as ``name=value``; a multi-token
        ``freq`` slot (several HB tones) becomes a vector.
        """
        tokens = self.tokens(statement)
        start = 3 if statement.keyword == "analysis" else 1
        arguments: dict[str, str] = {}
        if statement.keyword == "analysis" and len(tokens) >= 3:
            sim_type = _ANALYSES.get(tokens[2].text, SimulationType.UNKNOWN)
            names = self.analysis_metadata(sim_type).positional_param_names
            if names == ["freq"]:
                positional = [" ".join(item for item in positional if item)]
            arguments = {
                name: value for name, value in zip(names, positional, strict=False) if value
            }
        arguments.update(params)
        text = " ".join(f"{key}={_format(key, value)}" for key, value in arguments.items())
        if len(tokens) <= start:
            last = tokens[-1]
            return self._splice(statement, [(last.line, last.end, last.end, f" {text}")])
        return self._replace(statement, tokens[start], tokens[-1], text)

    def _replace(self, statement: Statement, first: Token, last: Token, text: str) -> Statement:
        """Replace the code from *first* to *last* with *text*, keeping the line count.

        The text goes where *first* was; the rest of the span is cut from the
        lines it covers, so continuation lines are left empty, not removed.
        """
        if first.line == last.line:
            return self._splice(statement, [(first.line, first.start, last.end, text)])
        edits = [(first.line, first.start, _code_end(statement.lines[first.line]), text)]
        edits.extend(
            (line, 0, last.end if line == last.line else _code_end(statement.lines[line]), "")
            for line in range(first.line + 1, last.line + 1)
        )
        return self._splice(statement, edits)

    def _splice(self, statement: Statement, edits: list[tuple[int, int, int, str]]) -> Statement:
        """Apply ``(line, start, end, text)`` replacements and re-lex the statement."""
        lines = list(statement.lines)
        for line, start, end, text in sorted(edits, reverse=True):
            lines[line] = lines[line][:start] + text + lines[line][end:]
        new_lines = tuple(lines)
        return self._statement(new_lines, self._lex(new_lines), statement.scope, statement.annotations)

    # -------------------------------------------------------------------------
    # Analyses
    # -------------------------------------------------------------------------

    def analysis_metadata(self, sim_type: SimulationType) -> SimulationTypeMetadata:
        return _VACASK_METADATA.get(sim_type, SimulationTypeMetadata())

    @staticmethod
    def analysis_keyword(sim_type: SimulationType) -> str:
        """The VACASK analysis that runs *sim_type*, e.g. ``acsp`` for AC."""
        for keyword, candidate in _ANALYSES.items():
            if candidate is sim_type:
                return keyword
        raise ValueError(f"VACASK has no analysis for {sim_type.name}")

    # -------------------------------------------------------------------------
    # Token helpers
    # -------------------------------------------------------------------------

    def _master(self, statement: Statement) -> str | None:
        tokens = statement.tokens or ()
        index = self._closing(tokens, 1) + 1
        return tokens[index].text if index < len(tokens) else None

    def _params_start(self, statement: Statement) -> int:
        """Index of the first token that may be a ``key=value`` parameter."""
        tokens = self.tokens(statement)
        if statement.kind is StatementKind.ELEMENT:
            return self._closing(tokens, 1) + 2
        return {"analysis": 3, "model": 3, "include": 2}.get(statement.keyword, 1)

    @staticmethod
    def _closing(tokens: tuple[Token, ...], start: int) -> int:
        """Index of the bracket closing the one at *start*, or ``start`` if none opens there."""
        if start >= len(tokens) or tokens[start].text not in _OPENING:
            return start
        depth = 0
        for index in range(start, len(tokens)):
            text = tokens[index].text
            if text in _OPENING:
                depth += 1
            elif text in _CLOSING:
                depth -= 1
                if depth == 0:
                    return index
        return len(tokens) - 1

    @staticmethod
    def _param_spans(tokens: tuple[Token, ...], start: int) -> dict[str, tuple[int, int]]:
        """``key → (first, stop)`` token range of the value of each ``key=value`` from *start* on.

        A value runs until the next top-level ``key=`` or the end of the
        statement, so expressions and vectors with spaces stay whole.
        """
        keys: list[int] = []
        depth = 0
        for index in range(start, len(tokens)):
            text = tokens[index].text
            if text in _OPENING:
                depth += 1
            elif text in _CLOSING:
                depth -= 1
            elif (
                depth == 0
                and index + 1 < len(tokens)
                and tokens[index + 1].text == "="
                and _IDENTIFIER_RE.match(text)
            ):
                keys.append(index)
        spans: dict[str, tuple[int, int]] = {}
        for position, key in enumerate(keys):
            stop = keys[position + 1] if position + 1 < len(keys) else len(tokens)
            if key + 2 < stop:
                spans[tokens[key].text] = (key + 2, stop)
        return spans

    def _params(self, tokens: tuple[Token, ...], start: int) -> dict[str, str]:
        return {
            key: _span_text(tokens[first:stop])
            for key, (first, stop) in self._param_spans(tokens, start).items()
        }

    @staticmethod
    def _words(tokens: tuple[Token, ...]) -> list[str]:
        """Tokens glued back together where no space separates them, e.g. ``v(out)``."""
        words: list[str] = []
        previous: Token | None = None
        for token in tokens:
            if previous is not None and token.line == previous.line and token.start == previous.end:
                words[-1] += token.text
            else:
                words.append(token.text)
            previous = token
        return words


def _span_text(tokens: tuple[Token, ...]) -> str:
    """The source text of *tokens*: their own spacing on a line, one space across lines."""
    text = ""
    previous: Token | None = None
    for token in tokens:
        if previous is None:
            text = token.text
        elif token.line == previous.line:
            text += " " * (token.start - previous.end) + token.text
        else:
            text += " " + token.text
        previous = token
    return text


def _code_end(line: str) -> int:
    """Column after the last code character of *line*, ignoring its line ending."""
    return len(line.rstrip("\r\n"))


def _format(key: str, value: str) -> str:
    """*value* as VACASK expects it for analysis parameter *key*."""
    value = str(value).strip()
    if key in _STRING_PARAMS and not value.startswith('"'):
        return f'"{value}"'
    items = value.replace(",", " ").split()
    if not value.startswith("[") and (
        key in _VECTOR_PARAMS
        or (key in _SCALAR_OR_VECTOR_PARAMS and len(items) > 1)
        or (key in _SPUR_PARAMS and all(_INTEGER_RE.match(item) for item in items))
    ):
        return "[" + ", ".join(items) + "]"
    return value


# ---------------------------------------------------------------------------
# Include lookup
# ---------------------------------------------------------------------------


def _include_path(directory: Path) -> list[Path]:
    """VACASK's configured include directories: environment first, then config files."""
    paths = [Path(item) for item in os.environ.get("SIM_INCLUDE_PATH", "").split(os.pathsep) if item]
    prefix: list[Path] = []
    suffix: list[Path] = []
    for config in (
        Path("/etc/vacask/vacaskrc.toml"),
        Path.home() / ".vacaskrc.toml",
        Path.cwd() / ".vacaskrc.toml",
        directory / ".vacaskrc.toml",
    ):
        try:
            data = tomllib.loads(config.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        section = data.get("Paths", {})
        prefix = [Path(_expand(item)) for item in section.get("include_path_prefix", [])] + prefix
        suffix += [Path(_expand(item)) for item in section.get("include_path_suffix", [])]
    return [*paths, *prefix, *suffix]


def _expand(text: str) -> str:
    """Interpolate ``$(VAR)`` environment references as VACASK's config files do."""
    return re.sub(r"\$\((\w+)\)", lambda match: os.environ.get(match.group(1), ""), text)


def _resolve(name: str, base: Path, search: list[Path]) -> Path | None:
    path = Path(name).expanduser()
    if path.is_absolute():
        return path if path.is_file() else None
    for directory in (base, Path.cwd(), *search):
        candidate = directory / path
        if candidate.is_file():
            return candidate.resolve()
    return None


_FOREIGN_DEFINITION_RE = re.compile(r"^\s*\.(subckt|model)\s+(\S+)(.*)$", re.IGNORECASE | re.MULTILINE)
_FOREIGN_INCLUDE_RE = re.compile(r"""^\s*\.(?:include|inc|lib)\s+["']?([^"'\s]+)["']?""", re.IGNORECASE | re.MULTILINE)


def _foreign_masters(
    text: str, base: Path, search: list[Path], seen: set[Path]
) -> dict[str, tuple[str, ...]]:
    """Masters a SPICE file (``include ... lang=ngspice``) defines, following its own includes.

    VACASK registers a SPICE ``.model`` as ``m_<name>``. SPICE names are
    case-insensitive, and VACASK lower-cases them, so both spellings are added.
    """
    masters: dict[str, tuple[str, ...]] = {}
    for text_part in _iter_foreign(text, base, search, seen):
        for kind, written, rest in _FOREIGN_DEFINITION_RE.findall(text_part):
            name = written.split("(")[0]
            pins: list[str] = []
            if kind.lower() == "subckt":
                for word in rest.split():
                    if "=" in word or word.lower() == "params:":
                        break
                    pins.append(word.lower())
            for spelling in {name, name.lower()}:
                masters[spelling if kind.lower() == "subckt" else f"m_{spelling}"] = tuple(pins)
    return masters


def _iter_foreign(text: str, base: Path, search: list[Path], seen: set[Path]) -> Iterator[str]:
    yield text
    for name in _FOREIGN_INCLUDE_RE.findall(text):
        path = _resolve(name, base, search)
        if path is None or path in seen:
            continue
        seen.add(path)
        try:
            inner = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        yield from _iter_foreign(inner, path.parent, search, seen)
