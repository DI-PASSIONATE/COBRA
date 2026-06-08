# Examples

COBRA includes runnable assets in the `examples/` folder.

## Main Script Example

`examples/main.py` demonstrates a mixed-source optimization setup.

It includes:

- ONNX surrogate mapping for one component,
- fixed SNP mapping for another component,
- model-input and netlist-variable optimization,
- linked netlist variable constraints,
- optional ORCA geometry integration for fine-tuning-ready workflows.

## Example Data Files

You will find sample files such as:

- `.cir` netlists
- `.onnx` surrogate model files
- `.sNp` Touchstone files

!!! tip
    Start by running `examples/main.py` unchanged, then clone it into your own experiment script and modify one section at a time.

## Suggested Adaptation Order

1. Replace netlist path.
2. Replace component mapping files.
3. Keep design goals unchanged for first run.
4. Expand optimization parameter ranges after baseline run succeeds.
5. Add fine-tuning options only after surrogate-only flow works.

## Expected Artifacts

After successful runs, compare generated files under `results/` to understand how model predictions and simulation outputs evolve over iterations.
