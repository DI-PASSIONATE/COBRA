# FAQ

## Can I use COBRA without the GUI?

Yes. Script mode is fully supported. Start with **Getting Started -> Quickstart**.

## Can I mix ONNX and Touchstone components?

Yes. Map each component to either `.onnx` or fixed `.sNp` in the same run.

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
