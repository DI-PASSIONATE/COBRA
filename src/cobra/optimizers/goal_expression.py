"""Arithmetic expressions over named goal values, e.g. ``a - b`` for gain compression.

An expression is parsed once into a restricted syntax tree: numbers, variable
names, ``+ - * / **``, unary signs, and the functions in :data:`FUNCTIONS`. It
is never passed to ``eval``. Values are numpy arrays (one entry per frequency
point) and combine elementwise.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def _extremum(reduce: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    return lambda *values: reduce(np.broadcast_arrays(*values))


#: Functions an expression may call: name -> (implementation, minimum argument count,
#: maximum argument count or ``None`` for any).
FUNCTIONS: dict[str, tuple[Callable[..., np.ndarray], int, int | None]] = {
    "abs": (np.abs, 1, 1),
    "sqrt": (np.sqrt, 1, 1),
    "exp": (np.exp, 1, 1),
    "ln": (np.log, 1, 1),
    "log10": (np.log10, 1, 1),
    "min": (_extremum(np.minimum.reduce), 2, None),
    "max": (_extremum(np.maximum.reduce), 2, None),
}

_OPERATORS: dict[type[ast.operator], Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
}


class GoalExpression:
    """A parsed expression over named values.

    Raises ``ValueError`` for anything outside the grammar, with the offending part.
    """

    def __init__(self, text: str):
        self.text = text
        try:
            tree = ast.parse(text.strip(), mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Expression {text!r} is not valid: {exc.msg}") from exc
        self._body = tree.body
        variables: list[str] = []
        self._check(self._body, variables)
        #: The variable names the expression reads, in order of first use.
        self.variables: tuple[str, ...] = tuple(dict.fromkeys(variables))

    def __repr__(self) -> str:
        return f"GoalExpression({self.text!r})"

    def _check(self, node: ast.expr, variables: list[str]) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(
            node.value, bool
        ):
            return
        if isinstance(node, ast.Name):
            if node.id in FUNCTIONS:
                raise ValueError(f"'{node.id}' is a function; call it, e.g. {node.id}(a)")
            variables.append(node.id)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            self._check(node.operand, variables)
            return
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            self._check(node.left, variables)
            self._check(node.right, variables)
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            spec = FUNCTIONS.get(node.func.id)
            if spec is None:
                raise ValueError(
                    f"Unknown function '{node.func.id}'; available: {', '.join(sorted(FUNCTIONS))}"
                )
            _, least, most = spec
            count = len(node.args)
            if count < least or (most is not None and count > most):
                expected = str(least) if most == least else f"at least {least}"
                raise ValueError(f"{node.func.id}() takes {expected} argument(s), got {count}")
            for argument in node.args:
                self._check(argument, variables)
            return
        raise ValueError(
            f"'{ast.unparse(node)}' is not allowed in an expression: use numbers, variables, "
            f"+ - * / **, and {', '.join(sorted(FUNCTIONS))}"
        )

    def evaluate(self, values: Mapping[str, np.ndarray]) -> np.ndarray:
        """The expression's value for *values*, one array per variable, elementwise."""
        with np.errstate(all="ignore"):
            return np.asarray(self._evaluate(self._body, values), dtype=float)

    def _evaluate(self, node: ast.expr, values: Mapping[str, np.ndarray]) -> np.ndarray:
        if isinstance(node, ast.Constant):
            return np.asarray(node.value, dtype=float)
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.UnaryOp):
            operand = self._evaluate(node.operand, values)
            return -operand if isinstance(node.op, ast.USub) else operand
        if isinstance(node, ast.BinOp):
            operator = _OPERATORS[type(node.op)]
            return operator(self._evaluate(node.left, values), self._evaluate(node.right, values))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = FUNCTIONS[node.func.id][0]
            return function(*(self._evaluate(argument, values) for argument in node.args))
        raise TypeError(f"Unexpected node {ast.dump(node)}")  # excluded by _check
