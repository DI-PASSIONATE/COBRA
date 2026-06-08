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
