---
name: cobra
description: "Use when answering COBRA RFIC optimizer questions, reading its docs, creating or validating JSON configs, running COBRA from Python or the CLI, checking a long-running optimization, or designing and optimizing a circuit end to end with surrogate models."
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
- Ask for missing inputs before writing or running anything, all in one batch.
  For end-to-end design tasks, follow the step order in
  [design-workflow.md](./references/design-workflow.md).
- Use the repository `.venv/` environment (preferably with uv); do not silently
  use a global install.
- Never run simulations or optimizations in the foreground. Capture output and
  report the PID or task id, log path, config/script path, and results location.
- Do not restart an interrupted job automatically. Preserve unrelated changes.

## CLI Surface

| Command | Purpose | Exit codes |
| --- | --- | --- |
| `cobra` | Launch the GUI | `0` |
| `cobra doctor` | Check packages, Xyce, and Palace | `0` ok, `1` required dependency missing |
| `cobra init NETLIST` | Write a starter config from a netlist | `0` written, `2` bad netlist, model, or existing output |
| `cobra run CONFIG` | Run a saved JSON configuration | `0` done, `2` bad config, `1` run failed, `130` interrupted |
| `cobra parse TARGET` | Report a config or netlist without running it | `0` no error, `2` at least one error |

`cobra init` accepts `-o PATH`, repeated `--model NAME=PATH`, and `--force`. It
fills in the analysis parameters, default backend settings, and models, and
leaves `design_goals` and `optimization_parameters` empty. `parse` and `run`
reject a config until both have at least one entry.

`cobra parse` accepts `--json`, `--kind {auto,config,netlist}`, `--full`, and
`--no-model-check`. `auto` treats `.json` files and JSON objects as
configurations and everything else as a netlist.

Stdout carries only the requested output: the `parse` report, the `run`
header and summary, the `doctor` table, or the path `init` wrote. Progress and
diagnostics go to stderr, so `parse --json` can be piped straight into a JSON
reader.

## Inspect Before Writing or Running

`cobra parse` is the ground truth for everything the netlist decides. Run it on
the netlist before writing a config, and on the config to gate every run
(`cobra parse config.json && cobra run config.json`). Fix every ERROR and review
every WARNING. What the reports contain and check is in
[configuration.md](./references/configuration.md).

## Choose a Workflow

Load only the relevant reference:

- End-to-end design or optimization (spec → validated config → runs →
  refinement → optional EM verification): [design-workflow.md](./references/design-workflow.md)
- Documentation questions: [documentation.md](./references/documentation.md)
- JSON creation or validation: [configuration.md](./references/configuration.md)
- Python script runs: [python-runs.md](./references/python-runs.md)
- CLI, Xyce, background runs, and monitoring: [cli-runs.md](./references/cli-runs.md)

Use `.claude/CLAUDE.md` for repository-wide development rules, code structure,
coding style, and general implementation practices.

## Shared Request Rules

For optimization requests, extract the netlist path, target and direction,
signed value, frequency/range, variables with bounds, step, and unit, component
model mapping, iteration count, analysis type, and optional fine-tuning settings
(Palace command and ORCA geometries). Ask only for unknown values. A goal does
not say what may change.

Use COBRA names such as `S11_dB`, `model_input`, `netlist_variable`, `.AC`,
`.HB`, `OptunaOptimizer`, and `XyceSimulator`. Resolve “S11 less than 10 dB”
with the user: it usually means `S11_dB <= -10`, but the signed convention must
be confirmed. Do not silently choose a frequency or full sweep.

For HB, check output node probes and driven ports. For fine-tuning, every ONNX
component needs an ORCA geometry. Report the docs page, config path, run PID,
log, results directory, and current status when applicable.
