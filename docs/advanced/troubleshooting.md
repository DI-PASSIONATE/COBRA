---
title: Troubleshooting – Fix Xyce, ORCA and Palace Errors in COBRA
description: >-
  Fix common COBRA problems: cobra command not found, Xyce simulation failures, component mapping and frequency range errors, ORCA and Palace issues, and slow runs.
---

# Troubleshooting

## Start With `cobra doctor`

`cobra doctor` reports the interpreter, the required packages and simulators,
and the optional components COBRA found. It exits with `1` when a requirement
is missing, and it is the quickest way to tell a missing dependency from a
misconfigured run.

When a run itself misbehaves, repeat it with more detail:

```bash
cobra run config.json -v                       # debug output on the console
cobra run config.json --log-file results/run.log   # a full debug log on disk
```

`--log-file` records at debug level regardless of what the console shows, so it
can be combined with `-q`.

## `cobra` Command Not Found

- Activate your virtual environment.
- Reinstall in editable mode:

```bash
pip install -e .
```

## Xyce Execution Fails

COBRA treats the two kinds of Xyce failure differently.

**A simulation that fails** — Xyce exits non-zero (for example because the
sampled parameters describe a circuit that will not converge) or writes no
output — is a property of the parameters. COBRA logs a warning, assigns a very
high loss to every design goal that needed that analysis, and continues, so the
optimizer steers away from those parameters:

```
WARNING  Xyce failed for SimulationType.AC (return code 1) in results/...
WARNING  No AC result for these parameters; the design goals that need it are penalised so the optimizer avoids them
```

Occasional warnings of this kind are normal. If *every* iteration produces them,
the netlist itself is at fault:

- Check your netlist compatibility.
- Confirm generated include/subcircuit files exist in the run output folder.
- Narrow the parameter ranges so the optimizer cannot reach unphysical values.

**A simulator that cannot be run at all** — Xyce is not installed, not on
`PATH`, or not executable — raises `SimulatorError` and stops the run, because
no choice of parameters can fix it:

- Verify `Xyce` is installed and in `PATH` (`cobra doctor` reports this).
- Or set the simulator's `xyce_command` setting to the absolute path of the
  executable.
- With `parallel_xyce` enabled, `mpirun` must be on `PATH` too. The rank count
  comes from the `parallel_xyce_processes` setting (default: the machine's core
  count); `mpirun` fails when it exceeds the available slots.

## Component Mapping Errors

If COBRA reports missing model files for components:

- Ensure every parsed component name has an entry in `component_onnx_mapping`.
- Match exact instance names from the netlist (for example `X1`, `X2`).

## Frequency Range Errors

If goal range parsing fails:

- Use format similar to `125-135ghz`.
- Avoid malformed ranges like trailing separators.

If `cobra parse` warns that an analysis evaluates the circuit outside the band the
model covers, the surrogate is only predicted inside the `frequency` range of its
ONNX metadata; beyond it the vector-fitted subcircuit extrapolates. Narrow the
`.AC` sweep or `numfreq`, or use a model trained over the wider band. A model
without that metadata is rejected: re-export it with `input_parameter_ranges`
(ORCA writes it) so the band is known.

## Trained Ranges and Infeasible Geometries

ORCA records, in each ONNX model's metadata, the range every geometry input was
trained on and the feasibility constraints a buildable geometry satisfies (for
example `bottom_linewidth <= bottom_winding_diameter / 3`). `cobra parse` lists
both for each component, together with the physical properties the model
guarantees.

- If `cobra parse` warns that a parameter's bounds reach beyond the trained
  range, the surrogate extrapolates there. Narrow the bounds unless you mean to.
- During a run, a trial whose parameters violate a constraint is not simulated
  and does not count as an iteration: Optuna records it as pruned and suggests
  another one, and the progress display, plots and run log never see it.
  Optimizers without a pruned state, such as gradient descent, get the same
  penalty as a failed simulation instead. Run with `--verbose` to see which
  constraint each skipped trial violated; the number skipped is logged at the
  end of the optimization. After 1000 skipped trials in a row the run stops,
  because the bounds cover mostly unbuildable geometries; narrow them.
- A model whose constraints are not valid JSON, or use anything beyond ORCA's
  expression grammar, is refused before the run starts. COBRA parses the
  expressions itself and never executes them as Python code, so a model from an
  untrusted source cannot run code through them.

## ORCA Geometry Import Errors

- Ensure ORCA is installed in the same Python environment.
- Validate import path used in your script.

## Palace Fine-Tuning Errors

- Verify Palace command is valid and executable.
- Confirm geometry/mesh prerequisites are available.
- "The Palace simulation for X1 failed": ORCA logs Palace's last output lines just above
  this message. The Palace model is in `fine_tuning/iteration_NN/palace_sims/` inside
  the results folder.

## Large Runtime or Slow Progress

- Reduce `max_iterations` during early debugging.
- Narrow parameter search ranges.
- Start with fewer design goals, then add constraints incrementally.
