"""Tests for :mod:`cobra.stages.surrogate_metadata` and how :class:`EMSurrogateStage`
uses it: reading ORCA's metadata, counting ports, and evaluating the feasibility
constraints without ever executing code taken from a model.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from cobra.configuration.configuration import ConfigurationError
from cobra.stages import em_surrogate_stage
from cobra.stages.em_surrogate_stage import EMSurrogateStage
from cobra.stages.surrogate_metadata import (
    MAX_CONSTRAINT_LENGTH,
    FeasibilityConstraints,
    SurrogateMetadata,
    surrogate_port_count,
)

if TYPE_CHECKING:
    from onnxruntime import InferenceSession

INPUTS = ["bottom_linewidth", "bottom_winding_diameter", "turns"]


class _FakeSession:
    """The part of an ``InferenceSession`` COBRA reads when it loads a model."""

    def __init__(self, metadata: dict[str, str], inputs: list[str] | None = None):
        self._metadata = metadata
        self._inputs = [*(inputs or INPUTS), "frequency"]

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in self._inputs]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self._metadata)


def _session(metadata: dict[str, str]) -> InferenceSession:
    return cast("InferenceSession", _FakeSession(metadata))


def _metadata(**entries) -> dict[str, str]:
    """ORCA-style metadata: a valid band plus *entries*, each encoded as JSON."""
    metadata = {"input_parameter_ranges": json.dumps({"frequency": {"min": 1e9, "max": 1e10}})}
    metadata.update({key: json.dumps(value) for key, value in entries.items()})
    return metadata


# ---------------------------------------------------------------------------
# Port count
# ---------------------------------------------------------------------------


def _names(n_ports: int, upper_triangle_only: bool) -> list[str]:
    return [
        f"S{i}{j}_{part}"
        for i in range(1, n_ports + 1)
        for j in range(i if upper_triangle_only else 1, n_ports + 1)
        for part in ("real", "imag")
    ]


@pytest.mark.parametrize("upper_triangle_only", [False, True])
@pytest.mark.parametrize("n_ports", [1, 3, 6, 9])
def test_port_count_of_full_and_upper_triangle_outputs(n_ports, upper_triangle_only):
    assert surrogate_port_count(_names(n_ports, upper_triangle_only)) == n_ports


@pytest.mark.parametrize(
    "names",
    [
        [],
        ["S11_real"],  # no imaginary part
        [*_names(2, False), "loss"],  # an output that is not an S-parameter
        [name for name in _names(3, False) if not name.startswith("S13")],  # a hole
        [name for name in _names(3, False) if not name.startswith("S12")],  # neither layout
    ],
)
def test_port_count_rejects_other_outputs(names):
    assert surrogate_port_count(names) is None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_is_read_from_the_session():
    session = _session(
        {
            "input_parameter_ranges": json.dumps(
                {"turns": {"min": 1, "max": 4}, "frequency": {"min": 1e9, "max": 1e11}}
            ),
            "input_constraints": json.dumps(["turns <= 3"]),
            "physics_guarantees": json.dumps({"passive": False, "reciprocal": True}),
        }
    )

    metadata = SurrogateMetadata.from_session(session)

    assert metadata.input_ranges == {"turns": (1.0, 4.0), "frequency": (1e9, 1e11)}
    assert metadata.frequency_range == (1e9, 1e11)
    assert metadata.constraints == ["turns <= 3"]
    assert metadata.guarantees == {"passive": False, "reciprocal": True}


def test_missing_or_malformed_metadata_reads_as_empty():
    metadata = SurrogateMetadata.from_session(
        _session(
            {
                "input_parameter_ranges": json.dumps(
                    {"turns": {"min": "a"}, "frequency": {"min": 5e9, "max": 1e9}}
                ),
                "physics_guarantees": "not json",
            }
        )
    )

    assert metadata.input_ranges == {"frequency": (5e9, 1e9)}
    assert metadata.frequency_range is None  # an empty band is not usable
    assert metadata.constraints == []
    assert metadata.guarantees == {}


@pytest.mark.parametrize("raw", ["not json", json.dumps("w < 1"), json.dumps([1])])
def test_malformed_constraints_are_refused(raw):
    with pytest.raises(ConfigurationError, match="input_constraints"):
        SurrogateMetadata.from_session(_session({"input_constraints": raw}))


# ---------------------------------------------------------------------------
# Constraint evaluation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "params", "holds"),
    [
        ("bottom_linewidth <= bottom_winding_diameter / 3", {"bottom_linewidth": 2, "bottom_winding_diameter": 9}, True),
        ("bottom_linewidth <= bottom_winding_diameter / 3", {"bottom_linewidth": 4, "bottom_winding_diameter": 9}, False),
        ("1 < turns <= 3", {"turns": 3}, True),
        ("1 < turns <= 3", {"turns": 1}, False),
        ("not (turns > 2 and turns < 4) or turns == 3", {"turns": 3}, True),
        ("(turns if turns >= 2 else 10) < 5", {"turns": 1}, False),
        ("max(turns, 2) * sqrt2 >= 2 * sqrt(2) - 1e-9", {"turns": 1}, True),
        ("ceil(turns) % 2 == 0 and round(pi, 2) == 3.14", {"turns": 1.5}, True),
        ("-turns ** 2 // 1 < +abs(turns)", {"turns": 2}, True),
    ],
)
def test_constraints_follow_orcas_grammar(expression, params, holds):
    constraints = FeasibilityConstraints([expression], INPUTS)
    assert (constraints.violated(params) == []) is holds


@pytest.mark.parametrize(
    "expression",
    [
        "sqrt(turns - 5) > 0",  # square root of a negative number
        "1 / (turns - 2) > 0",  # division by zero
        "(-8) ** 0.5 < 1",  # a complex result cannot be compared
        "10 ** 10 ** 10 > 0",  # overflows in floating point
    ],
)
def test_an_expression_that_cannot_be_evaluated_counts_as_violated(expression):
    assert FeasibilityConstraints([expression], INPUTS).violated({"turns": 2}) == [expression]


def test_a_non_boolean_constraint_is_a_configuration_error():
    with pytest.raises(ConfigurationError, match="not to a boolean"):
        FeasibilityConstraints(["turns + 1"], INPUTS).violated({"turns": 2})


# ---------------------------------------------------------------------------
# Safety: metadata may come from an untrusted model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd') is None",
        "eval('1') == 1",
        "exec('import os') is None",
        "turns.__class__.__bases__ == ()",
        "().__class__.__base__.__subclasses__() == []",
        "[x for x in (1,)] == [1]",
        "(lambda: True)()",
        "(turns := 1) == 1",
        "'a' < 'b'",
        "f'{turns}' == '1'",
        "turns[0] == 1",
        "1j == 1j",
        "turns << 2 > 1",
        "turns in (1, 2)",
        "turns is 1",
        "max(*[turns]) > 0",
        "max(turns, key=abs) > 0",
        "math.sqrt(turns) > 0",
        "__builtins__ is None",
        "frequency < 1e10",  # COBRA checks a geometry, not a frequency
        "unknown_input < 1",
        "turns < 1; import os",
        "",
    ],
)
def test_anything_outside_the_grammar_is_refused_at_load(expression):
    with pytest.raises(ConfigurationError):
        FeasibilityConstraints([expression], INPUTS)


def test_overlong_and_deeply_nested_expressions_are_refused():
    with pytest.raises(ConfigurationError, match="longer than"):
        FeasibilityConstraints(["turns < 1" + " " * MAX_CONSTRAINT_LENGTH], INPUTS)
    with pytest.raises(ConfigurationError):
        FeasibilityConstraints(["(" * 400 + "turns" + ")" * 400 + " < 1"], INPUTS)


def test_huge_powers_are_not_computed_as_integers():
    """``9 ** 9 ** 9`` as an integer has some 370 million digits; in floats it overflows at once."""
    started = time.monotonic()
    assert FeasibilityConstraints(["9 ** 9 ** 9 > 0"], INPUTS).violated({}) == ["9 ** 9 ** 9 > 0"]
    assert time.monotonic() - started < 1.0


# ---------------------------------------------------------------------------
# EMSurrogateStage
# ---------------------------------------------------------------------------


def _stage(monkeypatch, metadata: dict[str, str], *, components: int = 1) -> EMSurrogateStage:
    monkeypatch.setattr(em_surrogate_stage, "InferenceSession", lambda _path: _FakeSession(metadata))
    names = [f"X{index + 1}" for index in range(components)]
    return EMSurrogateStage([f"{name}.onnx" for name in names], component_names=names)


def test_stage_reports_the_violated_constraints_per_component(monkeypatch):
    stage = _stage(
        monkeypatch,
        _metadata(input_constraints=["bottom_linewidth <= bottom_winding_diameter / 3", "turns <= 3"]),
        components=2,
    )

    infeasible = stage.infeasible_components(
        {
            "bottom_winding_diameter": 9.0,  # unscoped: shared by both components
            "X1:bottom_linewidth": 2.0,
            "X1:turns": 4,
            "X2:bottom_linewidth": 2.0,
            "X2:turns": 2,
        }
    )

    assert infeasible == {"X1": ["turns <= 3"]}


def test_stage_refuses_a_model_with_unsafe_constraints(monkeypatch):
    with pytest.raises(ConfigurationError, match=r"X1\.onnx: Constraint .*unsupported syntax"):
        _stage(monkeypatch, _metadata(input_constraints=["__import__('os').getcwd() == ''"]))


def test_summary_lists_what_the_model_declares():
    session = _session(
        {
            "input_parameter_ranges": json.dumps(
                {"turns": {"min": 1, "max": 4}, "frequency": {"min": 1e9, "max": 1.1e11}}
            ),
            "input_constraints": json.dumps(["turns <= 3"]),
            "physics_guarantees": json.dumps({"passive": False, "reciprocal": True}),
        }
    )

    assert SurrogateMetadata.from_session(session).summary().splitlines() == [
        "Band: 1-110 GHz",
        "Trained ranges: turns [1, 4]",
        "Constraints: turns <= 3",
        "Guarantees: reciprocal",
    ]
