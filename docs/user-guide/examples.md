---
title: Examples – Runnable COBRA Netlists and Models
description: >-
  Runnable COBRA examples with netlists and component models, plus a suggested order for adapting them to your own RFIC design.
---

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

## LNA With an Inductor Surrogate

`examples/configs/lna_inductor_config.json` optimizes a 130 GHz cascode LNA
(`examples/netlists/LNA/lna_inductor.cir`, IHP SG13G2, Xyce) whose output inductor is
the ORCA octagonal-inductor surrogate `models/DavidL-11/inductor_octa`. It tunes the
inductor geometry (turns, width, space, diameter), the transistor sizes and the
matching network for S11, S22, S21 and the noise figure at 125–135 GHz, keeping the
amplifier stable (`mu` ≥ 1 from 110 to 170 GHz):

```bash
cobra parse examples/configs/lna_inductor_config.json
cobra run examples/configs/lna_inductor_config.json
```

The netlist holds an optimized design (inductor: 2 turns, width 10.5 µm, space 3 µm,
diameter 229 µm): S11 −12.2 to −11.9 dB, S21 6.7 to 7.8 dB, NF 4.79 dB at 130 GHz.
Optuna samples at random, so another run may end elsewhere. It needs Xyce, the IHP
PDK path in the netlist and the ONNX model from Hugging Face.

## VACASK Example

`examples/configs/vacask_trafo_acsp.json` runs the transformer example with
`VacaskSimulator` on the native VACASK netlist `examples/netlists/VACASK/trafo_acsp.sim`:

```bash
cobra parse examples/configs/vacask_trafo_acsp.json
cobra run examples/configs/vacask_trafo_acsp.json
```

It needs `vacask` on `PATH`; see [VACASK Simulator](../advanced/vacask.md).

`examples/configs/vacask_lna_trafo_hb.json` optimizes the 130 GHz LNA
(`examples/netlists/VACASK/lna_trafo_hb.sim`) for harmonic-balance gain. It uses
the IHP SG13G2 PDK converted for VACASK; see
[PDK models](../advanced/vacask.md#pdk-models-ihp-sg13g2).

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
