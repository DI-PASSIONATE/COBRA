---
name: cobra
description: "Use when answering COBRA RFIC optimizer questions, reading its docs, creating or validating JSON configs, running COBRA from Python or the CLI, or checking a long-running optimization."
argument-hint: "Ask about COBRA or describe an optimization target and netlist."
user-invocable: true
---

# COBRA Agent Skill

Use this skill for COBRA documentation, configuration, COBRA runs, and CLI tasks.
COBRA combines Xyce simulation, ONNX or Touchstone models, and Optuna for RFIC
optimization.

## Rules

- Read repository docs first; inspect source when docs are unclear.
- Inspect every netlist and config with `cobra parse` before writing a config or
  starting a run. It is the sanctioned source for component names, ports, HB
  nodes, goal parameters, and tunable variables.
- Do not invent paths, component names, ports, HB nodes, bounds, frequencies, or
  simulator settings.
- Ask for missing inputs before writing or running anything.
- Use the repository `.venv/` environment (preferably with uv); do not silently
  use a global install.
- Never run simulations or optimizations in the foreground. Capture output and
  report PID, log path, config/script path, and results location.
- Do not restart an interrupted job automatically. Preserve unrelated changes.

## CLI Surface

| Command | Purpose | Exit codes |
| --- | --- | --- |
| `cobra` | Launch the GUI | `0` |
| `cobra run CONFIG` | Run a saved JSON configuration | `0` done, `2` bad config, `1` run failed, `130` interrupted |
| `cobra parse TARGET` | Report a config or netlist without running it | `0` no error, `2` at least one error |

`cobra parse` accepts `--json`, `--kind {auto,config,netlist}`, `--full`, and
`--no-model-check`. `auto` treats `.json` files and JSON objects as
configurations and everything else as a netlist.

Only `parse` keeps stdout clean: the report is the sole thing on stdout while
parser and model-loader chatter goes to stderr, so `--json` can be piped
straight into a JSON reader. `run` and the GUI print a dependency and version
banner first.

## Inspect Before Writing or Running

`cobra parse` is the ground truth for everything the netlist decides. Run it
before writing a config, and again to gate a run:

```bash
.venv/bin/cobra parse /absolute/path/design.cir
.venv/bin/cobra parse /absolute/path/config.json && .venv/bin/cobra run /absolute/path/config.json
```

A **netlist** report gives the primary analysis, ports with `z0` and source
amplitude, HB probe nodes, surrogate components needing a `component_models`
entry, included and library files, the design-goal parameters the netlist
supports (split into available now, with `.AC`, and with `.HB`), and every
tunable netlist variable with its current value.

A **config** report adds the cross-checks against the netlist it references:
model files exist, load, and have as many ports as the instance has nodes;
`model_input` names match the mapped ONNX inputs; `netlist_variable` names
resolve to an element or `instance:parameter`; goals are buildable and their
frequency ranges parse; `simulation_parameters` name real directives and
`.options` categories; fine-tuning geometries resolve and cover every ONNX
component.

Read the `Issues` block, which is grouped by severity. Fix every ERROR; review
WARNINGs before running. Long lists are truncated — pass `--full` when a name
you need may have been cut off. JSON syntax alone is not validation.

The same reports are available in Python from `cobra.configuration.inspection`:
`inspect_path`, `inspect_netlist`, `inspect_configuration`, `render_report`,
`has_errors`, and `count_issues`.

## Choose a Workflow

Load only the relevant reference:

- Documentation questions: [documentation.md](./references/documentation.md)
- JSON creation or validation: [configuration.md](./references/configuration.md)
- Python script runs: [python-runs.md](./references/python-runs.md)
- CLI, Xyce, background runs, and monitoring: [cli-runs.md](./references/cli-runs.md)

Use `.claude/CLAUDE.md` for repository-wide development rules, code structure,
coding style, and general implementation practices.

## Shared Request Rules

For optimization requests, extract the netlist path, target and direction,
signed value, frequency/range, variables and bounds, component model mapping,
iteration count, analysis type, and optional fine-tuning settings. Ask only for
unknown values. A goal does not say what may change.

Use COBRA names such as `S11_dB`, `MODEL_INPUT`, `NETLIST_VARIABLE`, `.AC`,
`.HB`, `OptunaOptimizer`, and `XyceSimulator`. Resolve “S11 less than 10 dB”
with the user: it usually means `S11_dB <= -10`, but the signed convention must
be confirmed. Do not silently choose a frequency or full sweep.

For HB, check output node probes and driven ports. For fine-tuning, every ONNX
component needs an ORCA geometry. Report the docs page, config path, run PID,
log, results directory, and current status when applicable.
