# Quickstart

This page walks through a first successful COBRA run.

## 1. Prepare Inputs

You need:

- A netlist file (`.cir` / `.sp`) compatible with Xyce parsing.
- At least one component mapping to either:
  - ONNX surrogate model (`.onnx`), or
  - fixed Touchstone model (`.sNp`).

For an immediate starting point, use files in `examples/`.

## 2. Run GUI Mode

```bash
cobra
```

In GUI mode:

1. Load netlist.
2. Map each parsed component to model source files.
3. Define design goals.
4. Define optimization parameters.
5. Start optimization.

## 3. Run Script Mode (Optional)

```bash
python examples/main.py
```

This script demonstrates:

- mixed model sources (`.onnx` + `.sNp`),
- `MODEL_INPUT` and `NETLIST_VARIABLE` optimization,
- linked variables,
- `OptunaOptimizer` + `XyceSimulator`,
- optional ORCA geometry argument.

For concept details (workflow, goals, parameter types), see **Home**.

## 4. Inspect Results

Each run creates a timestamped folder under `results/`:

```text
results/<timestamp>_<name>/
```

Typical contents include:

- `cobra_optimization_context.json`
- `<component>_predicted.sNp`
- `surrogate_s_params_<component>.sNp`
- vector-fitted `.sp` files for simulation

!!! tip
    Keep the generated context JSON when comparing optimization runs. It contains trial history and best parameters.

## 5. Common First-Run Checks

- `cobra` command opens the GUI.
- `Xyce` is discoverable in `PATH`.
- Netlist component names match mapping keys (e.g. `X1`, `X2`).
- File paths in script mode are absolute or correctly resolved.
