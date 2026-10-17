---
title: Optimizers API – Goals, Parameters and Optuna in COBRA
description: >-
  API reference for COBRA optimization: OptimizationProperty, OptimizationType, DesignGoal, DesignParameter, OptunaOptimizer, and GradientDescentOptimizer.
---

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
- optional `initial_value`: the value the optimizer evaluates first

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

Each `DesignParameter` declares the analysis it needs, so goals are grouped by simulation type and evaluated against the matching result. Small-signal parameters (S-parameters, `Lp`, `Qs`, `k`, `mu`, `SRF`, …) read the `.AC` network; the spectrum parameters read the HB or transient spectrum:

| Factory | Parameter name | Description |
|---------|----------------|-------------|
| `make_power_dbm(node, simulation_type)` | `Power_dBm[<node>]` | Output power in dBm at an analysis point |
| `make_gain_db(port, amplitude, z0, node, simulation_type)` | `Gain_dB[<port>@<node>]` | Transducer gain in dB referred to a port's drive level |
| `make_isolation_db(node, simulation_type)` | `Isolation_dB[<node>]` | Margin in dB between the target line and the strongest other line |

`simulation_type` defaults to `SimulationType.HB`; passing `SimulationType.TRAN` evaluates the same quantity on the transient spectrum and prefixes the name with `TRAN:`. These are built per netlist rather than taken from the static catalogue, because the available nodes and ports differ per circuit. See **Advanced -> Harmonic Balance** and **Advanced -> Transient Analysis**.

## OptunaOptimizer

Default optimizer wrapper for Optuna.

Typical options:

- `sampler`: `tpe`, `random`, `simulated_annealing`, `auto`. TPE uses Optuna's
  defaults: multivariate for a single objective, independent for several;
  override them through `sampler_kwargs`, e.g. `{"multivariate": False}`
- `pruner`: `median`, `successive_halving`, `hyperband`
- `multi_objective`: boolean

A parameter's `initial_value` makes the first trial evaluate it; the others
are sampled.

## GradientDescentOptimizer

Used for local refinement workflows, including optional fine-tuning mode in COBRA.
It is simultaneous-perturbation (SPSA) gradient descent in coordinates
normalised to each parameter's range: every step probes the current point at
± a small distance along a random direction (two trials), then moves against
the estimated gradient. Both distances shrink with every step. The point is kept
unrounded, so steps below a parameter's grid still add up; only the proposals
are rounded to it.

- `learning_rate`: first step as a fraction of the range (default `0.2`)
- `exploration_scale`: probe distance; `1` probes at 10 % of the range, at least
  one grid step (default `0.5`)
- `random_seed`

It starts at the context's values (in EM fine-tuning, the design just
evaluated), else at each parameter's `initial_value`, else in the middle of the
range; a start not taken from the context is evaluated first. It needs the
result of one trial before it suggests the next, so `parallel_trials` must be 1.

!!! tip
    Start with `OptunaOptimizer` and move to gradient descent only when you are refining near a known good region.
