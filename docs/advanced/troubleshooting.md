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
- With `parallel_xyce` enabled, `mpirun` must be on `PATH` too.

## Component Mapping Errors

If COBRA reports missing model files for components:

- Ensure every parsed component name has an entry in `component_onnx_mapping`.
- Match exact instance names from the netlist (for example `X1`, `X2`).

## Frequency Range Errors

If goal range parsing fails:

- Use format similar to `125-135ghz`.
- Avoid malformed ranges like trailing separators.

## ORCA Geometry Import Errors

- Ensure ORCA is installed in the same Python environment.
- Validate import path used in your script.

## Palace Fine-Tuning Errors

- Verify Palace command is valid and executable.
- Confirm geometry/mesh prerequisites are available.

## Large Runtime or Slow Progress

- Reduce `max_iterations` during early debugging.
- Narrow parameter search ranges.
- Start with fewer design goals, then add constraints incrementally.
