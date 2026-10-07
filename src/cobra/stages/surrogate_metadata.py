"""What an ORCA-exported ONNX surrogate declares about itself.

ORCA writes three ``metadata_props`` entries into every model it exports:

- ``input_parameter_ranges``: ``{input: {"min": ..., "max": ...}}`` for every
  input, ``frequency`` included. The ``frequency`` entry is the band the
  surrogate is valid over; the others are the ranges it was trained on.
- ``input_constraints``: boolean expressions over the inputs, such as
  ``"bottom_linewidth <= bottom_winding_diameter / 3"``, that every buildable
  geometry satisfies. The model was only trained inside them.
- ``physics_guarantees``: ``{"passive": bool, "reciprocal": bool, ...}``, the
  properties the model enforces by construction.

The constraints are evaluated without ``eval``: an expression is parsed with
:func:`ast.parse` and interpreted node by node over the small grammar ORCA
defines in ``orca.geometry.constraints`` (numbers, input names, arithmetic,
comparisons, ``and``/``or``/``not``, ``a if c else b``, a fixed set of math
functions and the constants ``pi`` and ``sqrt2``). Anything else is rejected
when the model is loaded, so a model from an untrusted source cannot run code.
Expressions are also capped in length, and ``**`` is computed in floating point,
so they cannot exhaust memory or time either.

Nothing here imports ``onnxruntime``, so the module stays cheap to import.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from cobra.configuration.configuration import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from onnxruntime import InferenceSession

#: Longest constraint expression accepted, in characters.
MAX_CONSTRAINT_LENGTH: Final = 1000

#: Functions a constraint may call.
FUNCTIONS: Final[dict[str, Callable[..., Any]]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "radians": math.radians,
    "ceil": math.ceil,
    "floor": math.floor,
    "round": round,
}

#: Named constants a constraint may use.
CONSTANTS: Final[dict[str, float]] = {"pi": math.pi, "sqrt2": math.sqrt(2)}


def _power(base: float, exponent: float) -> float:
    # In floating point, so a huge result overflows instead of growing an integer without bound.
    return float(base) ** float(exponent)


_BINARY: Final[dict[type[ast.operator], Callable[[Any, Any], Any]]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _power,
}

_COMPARE: Final[dict[type[ast.cmpop], Callable[[Any, Any], bool]]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}

#: A model output: the real or imaginary part of one S-parameter entry.
_S_OUTPUT_RE = re.compile(r"^S(\d)(\d)_(real|imag)$")


def surrogate_port_count(output_names: Iterable[str]) -> int | None:
    """Number of ports of a surrogate with these outputs, or ``None`` if they are not S-parameters.

    The outputs are ``S<i><j>_real``/``_imag`` for either every entry of the
    N x N matrix or its upper triangle only (ORCA's ``UpperTriangleReImCodec``,
    for a reciprocal model). The port count is the highest index in the names:
    the number of outputs cannot tell the two layouts apart (72 is a full
    6-port or an 8-port triangle). The single-digit indices limit the naming to
    nine ports.
    """
    entries: set[tuple[int, int, str]] = set()
    for name in output_names:
        match = _S_OUTPUT_RE.match(name)
        if match is None:
            return None
        entries.add((int(match[1]), int(match[2]), match[3]))
    if not entries:
        return None
    n_ports = max(max(i, j) for i, j, _ in entries)
    full = {(i, j, part) for i in range(1, n_ports + 1) for j in range(1, n_ports + 1) for part in ("real", "imag")}
    upper_triangle = {(i, j, part) for i, j, part in full if i <= j}
    return n_ports if entries in (full, upper_triangle) else None


@dataclass(frozen=True)
class SurrogateMetadata:
    """The ``metadata_props`` ORCA writes into an exported surrogate."""

    input_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    """Trained ``(min, max)`` of every input, ``frequency`` (Hz) included."""
    constraints: list[str] = field(default_factory=list)
    """Feasibility expressions over the inputs; empty means the whole range box."""
    guarantees: dict[str, bool] = field(default_factory=dict)
    """Physical properties the model enforces by construction."""

    @classmethod
    def from_session(cls, session: InferenceSession) -> SurrogateMetadata:
        """Read the metadata of *session*.

        Missing entries read as empty, and a malformed range or guarantee is
        skipped. Malformed constraints raise :class:`ConfigurationError`: without
        them COBRA cannot tell a buildable geometry from one the model never saw.
        """
        props = session.get_modelmeta().custom_metadata_map
        return cls(
            input_ranges=_parse_ranges(props.get("input_parameter_ranges")),
            constraints=_parse_constraints(props.get("input_constraints")),
            guarantees=_parse_guarantees(props.get("physics_guarantees")),
        )

    @property
    def frequency_range(self) -> tuple[float, float] | None:
        """The band (Hz) the surrogate is valid over, or ``None`` if it declares no usable one."""
        band = self.input_ranges.get("frequency")
        if band is None or not 0 < band[0] < band[1]:
            return None
        return band

    def summary(self) -> str:
        """The band, trained ranges, constraints and guarantees, one line each."""
        band = self.frequency_range
        ranges = ", ".join(
            f"{name} [{low:g}, {high:g}]" for name, (low, high) in self.input_ranges.items() if name != "frequency"
        )
        guarantees = ", ".join(name for name, held in self.guarantees.items() if held)
        return "\n".join(
            [
                f"Band: {band[0] / 1e9:g}-{band[1] / 1e9:g} GHz" if band else "Band: not declared",
                f"Trained ranges: {ranges or 'not declared'}",
                f"Constraints: {'; '.join(self.constraints) or 'none'}",
                f"Guarantees: {guarantees or 'none'}",
            ]
        )


def _load_json(raw: str | None) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _parse_ranges(raw: str | None) -> dict[str, tuple[float, float]]:
    data = _load_json(raw)
    if not isinstance(data, dict):
        return {}
    ranges = {}
    for name, bounds in data.items():
        try:
            ranges[name] = (float(bounds["min"]), float(bounds["max"]))
        except (KeyError, TypeError, ValueError):
            continue
    return ranges


def _parse_constraints(raw: str | None) -> list[str]:
    if raw is None:
        return []
    data = _load_json(raw)
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ConfigurationError(
            "Its input_constraints metadata is not a JSON list of expressions; re-export the "
            "model with ORCA."
        )
    return data


def _parse_guarantees(raw: str | None) -> dict[str, bool]:
    data = _load_json(raw)
    if not isinstance(data, dict):
        return {}
    return {name: value for name, value in data.items() if isinstance(value, bool)}


class FeasibilityConstraints:
    """A surrogate's ``input_constraints``, parsed once and checked per trial.

    Raises :class:`ConfigurationError` when an expression is longer than
    :data:`MAX_CONSTRAINT_LENGTH`, falls outside the grammar, or names something
    other than *input_names* and the constants.
    """

    def __init__(self, expressions: Iterable[str], input_names: Iterable[str]) -> None:
        names = set(input_names)
        self._parsed = [(expression, _parse(expression, names)) for expression in expressions]

    @property
    def expressions(self) -> list[str]:
        return [expression for expression, _ in self._parsed]

    def violated(self, params: Mapping[str, float]) -> list[str]:
        """The expressions *params* do not satisfy.

        An expression that cannot be evaluated for *params* (a division by zero,
        the square root of a negative number, an overflow) counts as violated.
        """
        return [expression for expression, tree in self._parsed if not _holds(expression, tree, params)]


def _parse(expression: str, names: set[str]) -> ast.expr:
    if len(expression) > MAX_CONSTRAINT_LENGTH:
        raise ConfigurationError(
            f"Constraint {expression[:60]!r}... is longer than {MAX_CONSTRAINT_LENGTH} characters."
        )
    try:
        tree = ast.parse(expression, mode="eval").body
        _check(tree, expression, names)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"Constraint {expression!r} is not a valid expression: {exc}") from exc
    return tree


def _holds(expression: str, tree: ast.expr, params: Mapping[str, float]) -> bool:
    try:
        result = _eval(tree, params)
    except KeyError as exc:
        raise ConfigurationError(
            f"Constraint {expression!r} needs a value for model input {exc.args[0]!r}."
        ) from exc
    except (ArithmeticError, ValueError, TypeError, RecursionError):
        return False
    if not isinstance(result, bool):
        raise ConfigurationError(f"Constraint {expression!r} evaluates to {result!r}, not to a boolean.")
    return result


def _check(node: ast.expr, expression: str, names: set[str]) -> None:
    """Walk *node* and reject anything outside the grammar."""
    match node:
        case ast.Constant(value=bool() | int() | float()):
            pass
        case ast.Name(id=name):
            if name not in names and name not in CONSTANTS:
                raise ConfigurationError(
                    f"Constraint {expression!r} uses {name!r}, which is not a geometry input "
                    f"({', '.join(sorted(names)) or 'none'}) or a constant."
                )
        case ast.UnaryOp(op=ast.USub() | ast.UAdd() | ast.Not(), operand=operand):
            _check(operand, expression, names)
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINARY:
            _check(left, expression, names)
            _check(right, expression, names)
        case ast.BoolOp(values=values):
            for value in values:
                _check(value, expression, names)
        case ast.Compare(left=left, ops=ops, comparators=comparators):
            if any(type(op) not in _COMPARE for op in ops):
                raise ConfigurationError(f"Constraint {expression!r} uses an unsupported comparison.")
            for value in (left, *comparators):
                _check(value, expression, names)
        case ast.IfExp(test=test, body=body, orelse=orelse):
            for value in (test, body, orelse):
                _check(value, expression, names)
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]):
            if name not in FUNCTIONS:
                raise ConfigurationError(
                    f"Constraint {expression!r} calls {name!r}; allowed functions are "
                    f"{', '.join(FUNCTIONS)}."
                )
            for value in args:
                _check(value, expression, names)
        case _:
            raise ConfigurationError(
                f"Constraint {expression!r} contains unsupported syntax ({type(node).__name__})."
            )


def _eval(node: ast.expr, params: Mapping[str, float]) -> Any:
    """Evaluate a node that passed :func:`_check`."""
    match node:
        case ast.Constant(value=value):
            return value
        case ast.Name(id=name):
            return params[name] if name in params or name not in CONSTANTS else CONSTANTS[name]
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            return -_eval(operand, params)
        case ast.UnaryOp(op=ast.UAdd(), operand=operand):
            return +_eval(operand, params)
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            return not _eval(operand, params)
        case ast.BinOp(left=left, op=op, right=right):
            return _BINARY[type(op)](_eval(left, params), _eval(right, params))
        case ast.BoolOp(op=ast.And(), values=values):
            return all(_eval(value, params) for value in values)
        case ast.BoolOp(op=ast.Or(), values=values):
            return any(_eval(value, params) for value in values)
        case ast.Compare(left=left, ops=ops, comparators=comparators):
            current = _eval(left, params)
            for op, comparator in zip(ops, comparators, strict=True):
                following = _eval(comparator, params)
                if not _COMPARE[type(op)](current, following):
                    return False
                current = following
            return True
        case ast.IfExp(test=test, body=body, orelse=orelse):
            return _eval(body, params) if _eval(test, params) else _eval(orelse, params)
        case ast.Call(func=ast.Name(id=name), args=args):
            return FUNCTIONS[name](*(_eval(value, params) for value in args))
    raise ConfigurationError(f"Unsupported node {type(node).__name__}.")  # unreachable after _check
