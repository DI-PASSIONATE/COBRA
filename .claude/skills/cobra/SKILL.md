---
name: cobra
description: "Use when answering COBRA RFIC optimizer questions, reading its docs, creating or validating JSON configs, running COBRA from Python or the CLI, or checking a long-running optimization."
argument-hint: "Ask about COBRA or describe an optimization target and netlist."
user-invocable: true
---

# COBRA Agent Skill

Use this skill for COBRA documentation, configuration, COBRA runs, and CLI tasks.
COBRA combines Xyce simulation, ONNX or Touchstone models, and Optuna for RFIC optimization. 

## Rules

- Read repository docs first; inspect source when docs are unclear.
- Do not invent paths, component names, ports, HB nodes, bounds, frequencies, or
  simulator settings.
- Ask for missing inputs before writing or running anything.
- Use the repository `.venv/` environment (preferably with uv); do not silently use a global install.
- Never run simulations or optimizations in the foreground. Capture output and
  report PID, log path, config/script path, and results location.
- Do not restart an interrupted job automatically. Preserve unrelated changes.

## Choose a Workflow

Load only the relevant reference:

- Documentation questions: [documentation.md](./references/documentation.md)
- JSON creation or validation: [configuration.md](./references/configuration.md)
- Python script runs: [python-runs.md](./references/python-runs.md)
- CLI, Xyce, background runs, and monitoring: [cli-runs.md](./references/cli-runs.md)

Use `AGENTS.md` for repository-wide development rules, code structure, coding
style, tests, and general implementation practices.

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
