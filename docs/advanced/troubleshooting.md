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

- Verify `Xyce` is installed and in `PATH` (`cobra doctor` reports this).
- Check your netlist compatibility.
- Confirm generated include/subcircuit files exist in the run output folder.

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
