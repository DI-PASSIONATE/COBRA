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
- ORCA 2.1 or newer, installed with `pip install "cobra-rfic[orca]"`, and an ORCA geometry
  for every ONNX component.

## Enabling Fine-Tuning

Fine-tuning behavior is configured via `COBRA(...)` constructor arguments and `cobra.run(...)` context.

Key options:

- `palace_fine_tuning_command`
- `fine_tuning_iterations`
- `fine_tuning_optimizer`

## Fine-Tuning Optimizer Modes

- `reuse`: continue using the surrogate-phase optimizer.
- `gradient_descent`: switch to local gradient-descent refinement.

## High-Level Loop

1. The first iteration verifies the parameters the surrogate optimization ended with.
2. For each ONNX component, ORCA's `simulate_geometry` draws the geometry from the current
   parameters, meshes it with that sample's own ports and runs Palace, exactly as for the
   model's training data. Touchstone (`.snp`) components keep their file.
3. Simulate the circuit with the Palace results and check the design goals.
4. Stop when the goals are met or the iteration budget is exhausted. Otherwise the
   fine-tuning optimizer suggests the next parameters, geometry and netlist values alike.

COBRA returns the best iteration it evaluated, which is not necessarily the last one,
and writes its parameters into the run's netlist.

Each iteration runs in its own folder, `fine_tuning/iteration_NN/` inside the results
folder, holding its netlist, GDS file, Palace model and Touchstone results.

An iteration whose parameters violate a feasibility constraint of a component's ONNX model
is not sent to Palace; it is scored like a failed simulation and the optimizer moves on.
If a component cannot be simulated otherwise, fine-tuning stops with an error that names the
component and the reason: parameters the geometry rejects as infeasible (when the model
declares no constraints), a degenerate mesh that Palace could not solve, or a failed Palace
run, whose log ORCA prints just above.

## Frequency Range

Palace simulates the band set in the geometry's simconfig file (`fstart`, `fstop`,
`fstep`, in GHz, plus the single frequencies in `fpoint`), not the circuit's `.AC` sweep.
ORCA's presets, for example, sweep 0–500 GHz in 1 GHz steps plus extra points below 10 GHz,
which is slow. To shorten fine-tuning, subclass
the geometry with a narrower simconfig and select it as a custom geometry:

```python
from dataclasses import dataclass

from orca.geometry.presets import TransformerOcta


@dataclass
class NarrowBandTransformer(TransformerOcta):
    # A copy of tf_octa_c_ports.simcfg with fstart=120, fstop=150, fstep=5
    simconfig_filename: str = "/path/to/tf_octa_c_ports_narrow.simcfg"
```

Keep the band wide enough to cover the circuit's analyses: outside it, the vector fit
extrapolates. COBRA reads back the DC-extrapolated, de-embedded result. ORCA only writes
that one when the sweep starts at or below 1 GHz with more than 20 points; otherwise
COBRA uses the de-embedded result, which leaves the behaviour near DC to the vector fit.

!!! warning
    Fine-tuning requires a correctly configured external EM environment and may be significantly slower than surrogate-only optimization.
