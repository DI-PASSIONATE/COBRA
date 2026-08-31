# Optimizers API

## OptimizationProperty

Defines one optimization variable.

Fields include:

- `name`
- `type` (`MODEL_INPUT` or `NETLIST_VARIABLE`)
- `min_value`
- `max_value`
- `step`
- optional `unit`
- optional `linked_to`

## OptimizationType

- `MODEL_INPUT`: variable passed to ONNX surrogate inputs.
- `NETLIST_VARIABLE`: variable patched directly in parsed netlist content.

## DesignGoal and DesignParameter

`DesignGoal` combines:

- parameter enum (`DesignParameter`),
- optional min/max bounds,
- optional `frequency_range`,
- optional weight.

Penalty is computed from normalized squared bound violations.

Each `DesignParameter` declares the analysis it needs, so goals are grouped by simulation type and evaluated against the matching result. Small-signal parameters (S-parameters, `Lp`, `Qs`, `k`, `mu`, `SRF`, …) read the `.AC` network; the Harmonic Balance parameters read the HB spectrum:

| Factory | Parameter name | Description |
|---------|----------------|-------------|
| `make_power_dbm(node)` | `Power_dBm[<node>]` | Output power in dBm at an HB analysis point |
| `make_gain_db(port, amplitude, z0, node)` | `Gain_dB[<port>@<node>]` | Transducer gain in dB referred to a port's drive level |

These two are built per netlist rather than taken from the static catalogue, because the available nodes and ports differ per circuit. See **Advanced -> Harmonic Balance**.

## OptunaOptimizer

Default optimizer wrapper for Optuna.

Typical options:

- `sampler`: `tpe`, `random`, `simulated_annealing`, `auto`
- `pruner`: `median`, `successive_halving`, `hyperband`
- `multi_objective`: boolean

## GradientDescentOptimizer

Used for local refinement workflows, including optional fine-tuning mode in COBRA.

!!! tip
    Start with `OptunaOptimizer` and move to gradient descent only when you are refining near a known good region.
