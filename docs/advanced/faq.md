# FAQ

## Can I use COBRA without the GUI?

Yes. Script mode is fully supported. Start with **Getting Started -> Quickstart**.

## Can I mix ONNX and Touchstone components?

Yes. Map each component to either `.onnx` or fixed `.sNp` in the same run.

## Can I optimize large-signal behavior?

Yes, through Harmonic Balance or a transient analysis. If the netlist contains a `.HB` or `.TRAN` analysis, COBRA exposes output-power, gain and isolation goals at any node that has both a voltage label and a current probe, and plots the resulting spectrum live. See **Advanced -> Harmonic Balance** and **Advanced -> Transient Analysis**.

## Can S-parameter and Harmonic Balance goals be combined?

Yes. COBRA runs one simulation per required analysis type and aggregates all penalties into a single loss, so matching (`.AC`) and conversion gain or isolation (`.HB`) can be optimized together.

## How do I target a single frequency instead of a band?

Set the same value for the minimum and maximum frequency of the goal. The nearest point of the sweep or spectrum is used.

## Is transient simulation supported?

Yes. The settled part of the waveform (from the `.TRAN` start time on) is turned into a spectrum by an FFT, and the `TRAN:`-prefixed power, gain and isolation goals read it exactly like their HB counterparts. The `.PRINT tran` line must use `format=csv`. See **Advanced -> Transient Analysis**.

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
