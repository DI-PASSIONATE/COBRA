---
title: EM Fine-Tuning – Refine Designs with AWS Palace and ORCA
description: >-
  Optional EM fine-tuning in COBRA: refine surrogate-optimized designs with full-wave AWS Palace simulations and ORCA geometry classes.
---

# Fine-Tuning

Fine-tuning is an optional phase after surrogate-based optimization.

## Purpose

Surrogate optimization is fast. Fine-tuning adds higher-fidelity EM verification and refinement.

## Requirements

- Palace installed and callable.
- ORCA 2.2 or newer, installed with `pip install "cobra-rfic[orca]"`, and an ORCA geometry
  for every ONNX component.

## Enabling Fine-Tuning

Fine-tuning behavior is configured via `COBRA(...)` constructor arguments and `cobra.run(...)` context.

Key options:

- `palace_fine_tuning_command`
- `fine_tuning_iterations`
- `fine_tuning_optimizer`
- `retune_trials`, `retune_range`, `retune_min_steps` (see [Retune](#retune-with-the-em-models-fixed))

In a JSON configuration they are the fields of `fine_tuning`: `palace_command`,
`iterations`, `optimizer`, `retune_trials`, `retune_range` and `retune_min_steps`.

## Fine-Tuning Optimizer Modes

- `reuse`: continue using the surrogate-phase optimizer.
- `gradient_descent`: switch to local gradient-descent refinement.

## High-Level Loop

1. The first iteration verifies the parameters the surrogate optimization ended with.
2. For each ONNX component, ORCA's `simulate_geometry` draws, meshes and simulates the
   geometry as for the model's training data. COBRA uses the Palace result the model was
   trained on (its `touchstone_type` metadata, else `dc`, the raw result with ORCA's 0 Hz
   point). Touchstone components keep their file.
3. Simulate the circuit with the Palace results and check the design goals.
4. With `retune_trials` set, an iteration that misses the goals is followed by a
   retune (below), which may meet them without another Palace run.
5. Stop when the goals are met or the iteration budget is exhausted. Otherwise the
   fine-tuning optimizer suggests the next parameters, geometry and netlist values alike.

COBRA returns the best iteration it evaluated, which is not necessarily the last one,
and writes its parameters into the run's netlist.

Each iteration runs in its own folder, `fine_tuning/iteration_NN/` inside the results
folder, holding its netlist, GDS file, Palace model and Touchstone results.

An iteration that violates a feasibility constraint of an ONNX model is not sent to Palace;
it is scored like a failed simulation. If Palace cannot simulate a component otherwise,
fine-tuning stops with an error that names the component and the reason.

## Retune with the EM Models Fixed

The circuit values were tuned to the surrogates' networks, so a design often misses its
goals narrowly in the first Palace iteration. The retune tunes them to Palace's networks
without running Palace again:

- Every component is fixed: the Palace result of an ONNX component, the file of a
  Touchstone one.
- Only `netlist_variable` parameters are searched, each within its value ±
  `max(retune_range × |value|, retune_min_steps × step)`, clipped to its range and kept on
  its grid. Variables linked to a model input keep their values.
- The search starts at the iteration's values and stops after `retune_trials` simulations
  or at the first that meets the goals.

If the best values meet the goals, fine-tuning stops; otherwise the fine-tuning optimizer
continues from them.

Each retune writes to `fine_tuning/retune_NN/`: the Palace networks as Touchstone files,
the netlist it started from, its search (a COBRA results folder) and `result/`, the
rescored values. In the GUI, the retune's trials show as part of their fine-tuning
iteration, and stopping the run stops the retune too.

## Frequency Range

Palace simulates the band of the geometry's simconfig file (`fstart`, `fstop`, `fstep`, in
GHz, plus the frequencies in `fpoint`), not the circuit's `.AC` sweep. ORCA's presets sweep
0–500 GHz, which is slow. To shorten it, subclass the geometry with a narrower simconfig and
select it as a custom geometry:

```python
from dataclasses import dataclass

from orca.geometry.presets import TransformerOcta


@dataclass
class NarrowBandTransformer(TransformerOcta):
    # A copy of tf_octa_c_ports.simcfg with fstart=120, fstop=150, fstep=5
    simconfig_filename: str = "/path/to/tf_octa_c_ports_narrow.simcfg"
```

Keep the band wide enough to cover the circuit's analyses, since the vector fit extrapolates
outside it. ORCA writes the `dc` result only when the sweep starts at or below 1 GHz with
more than 20 points; otherwise COBRA uses the raw result, without a 0 Hz point.

!!! warning
    Fine-tuning requires a correctly configured external EM environment and may be significantly slower than surrogate-only optimization.
