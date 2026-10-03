"""Keep the examples in the docs and the agent skill in step with the code.

Users and agents copy these snippets verbatim, so a renamed field or flag must
fail here rather than in someone's run.  Every JSON example is validated against
the configuration schema, and every ``cobra`` command in a shell block is parsed
by the real argument parser.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

import pytest

from cobra import __main__ as cli
from cobra.configuration.configuration import (
    DesignGoalConfig,
    OptimizationParameterConfig,
    RunConfiguration,
)
from cobra.spice_sim.simulation_type import SimulationType
from cobra.spice_sim.xyce_simulator import XyceSimulator
from tests.conftest import REPO_ROOT

DOCUMENTS = sorted(
    [
        REPO_ROOT / "README.md",
        *(REPO_ROOT / "docs").rglob("*.md"),
        *(REPO_ROOT / ".claude" / "skills").rglob("*.md"),
    ]
)
_FENCE = re.compile(r"^```(\w*)\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _blocks(language: str) -> list[tuple[str, str]]:
    found = []
    for document in DOCUMENTS:
        found.extend(
            (str(document.relative_to(REPO_ROOT)), match.group(2))
            for match in _FENCE.finditer(document.read_text(encoding="utf-8"))
            if match.group(1) == language
        )
    return found


def _parse_json_block(block: str) -> list[Any]:
    """A block holds one object, one object per line, or a bare ``"key": value`` member."""
    try:
        return [json.loads(block)]
    except json.JSONDecodeError:
        pass
    if block.lstrip().startswith('"'):
        return [json.loads("{" + block + "}")]
    return [json.loads(line) for line in block.splitlines() if line.strip()]


def _json_examples() -> list[Any]:
    examples = []
    for source, block in _blocks("json"):
        objects = _parse_json_block(block)
        examples.extend(pytest.param(item, id=f"{source}:{index}") for index, item in enumerate(objects))
    return examples


def _cobra_commands() -> list[Any]:
    """Every ``cobra`` invocation in a bash block, as the argument list after the executable."""
    commands = []
    for source, block in _blocks("bash"):
        for line in block.replace("\\\n", " ").splitlines():
            # `<design>` is a placeholder, not a redirection.
            command = re.sub(r"<([\w -]+)>", r"__\1__", line)
            lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
            lexer.commenters = "#"
            lexer.whitespace_split = True
            words = list(lexer)
            for start, word in enumerate(words):
                if word != "cobra" and not word.endswith("/bin/cobra"):
                    continue
                arguments = []
                for argument in words[start + 1 :]:
                    if set(argument) <= set("&|;<>()"):
                        break
                    arguments.append(argument)
                if arguments and arguments[0].startswith("__"):
                    continue  # a placeholder such as `cobra <command> --help`
                commands.append(pytest.param(arguments, id=f"{source}:{' '.join(arguments)}"))
    return commands


def _check_simulation_parameters(parameters: dict[str, dict[str, str]]) -> None:
    """Directive entries must name parameters the simulator knows; ``.OPTIONS`` entries are free-form."""
    for directive, values in parameters.items():
        assert all(isinstance(value, str) for value in values.values()), directive
        if directive.upper().startswith(".OPTIONS:"):
            continue
        simulation_type = SimulationType.from_directive(directive)
        assert simulation_type is not SimulationType.UNKNOWN, directive
        known = XyceSimulator.get_simulation_metadata(simulation_type).positional_param_names
        assert set(values) <= set(known), f"{directive}: {sorted(set(values) - set(known))}"


@pytest.mark.parametrize("example", _json_examples())
def test_json_example_matches_the_schema(example: dict[str, Any], tmp_path: Path):
    if all(key.startswith(".") for key in example):
        _check_simulation_parameters(example)
    elif "schema_version" in example:
        base = tmp_path / "config"
        base.mkdir()
        referenced = [example["netlist"], *example.get("component_models", {}).values()]
        referenced += [
            geometry["file"]
            for geometry in example.get("fine_tuning", {}).get("geometries", {}).values()
            if geometry.get("file")
        ]
        for path in referenced:
            target = (base / path).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
        RunConfiguration.from_dict(example, base)
        _check_simulation_parameters(example.get("simulation_parameters", {}))
    elif "parameter" in example:
        DesignGoalConfig.from_dict(example).validate()
    elif {"name", "type"} <= example.keys():
        OptimizationParameterConfig.from_dict(example).validate()
    else:
        pytest.fail(f"Unrecognised JSON example: {sorted(example)}")


@pytest.mark.parametrize("arguments", _cobra_commands())
def test_documented_command_is_accepted(arguments: list[str]):
    if {"--help", "--version"} & set(arguments):
        with pytest.raises(SystemExit) as exit_info:
            cli._parser().parse_args(arguments)
        assert exit_info.value.code == 0
    else:
        # A usage error raises SystemExit(2), which fails the test with argparse's message.
        cli._parser().parse_args(arguments)
