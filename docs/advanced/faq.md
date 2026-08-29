# FAQ

## Can I use COBRA without the GUI?

Yes. Script mode is fully supported. Start with **Getting Started -> Quickstart**.

## Can I mix ONNX and Touchstone components?

Yes. Map each component to either `.onnx` or fixed `.sNp` in the same run.

## Can I optimize large-signal behavior?

Yes, through Harmonic Balance. If the netlist contains a `.HB` analysis, COBRA exposes output-power and gain goals at any node that has both a voltage label and a current probe, and plots the resulting spectrum live. See **Advanced -> Harmonic Balance**.

## Can S-parameter and Harmonic Balance goals be combined?

Yes. COBRA runs one simulation per required analysis type and aggregates all penalties into a single loss, so matching (`.AC`) and conversion gain or isolation (`.HB`) can be optimized together.

## How do I target a single frequency instead of a band?

Set the same value for the minimum and maximum frequency of the goal. The nearest point of the sweep or HB spectrum is used.

## Is transient simulation supported?

Transient netlists are parsed and simulated, but time-domain spectrum plotting is not implemented yet. Use Harmonic Balance to obtain a spectrum.

## Is ORCA required?

Not for core surrogate optimization with existing ONNX files.

ORCA is required if you need ORCA geometry classes/workflows, especially for fine-tuning scenarios.

## Is Palace required?

No. Palace is optional and only needed for EM fine-tuning.

## Which optimizer should I start with?

Start with `OptunaOptimizer` (`sampler="tpe"`) for most cases.

## Where are outputs saved?

COBRA writes each run to a timestamped directory under `results/`.

## How do I keep symmetric constraints?

Use `linked_to` in `OptimizationProperty` so one variable mirrors another.
