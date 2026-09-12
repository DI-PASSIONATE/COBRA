# GUI Mode

The COBRA GUI is launched with:

```bash
cobra
```

## What You Can Configure

- Netlist source
- Component-to-model mapping (`.onnx` or `.sNp`)
- Design goals (S-parameter and RF metrics)
- Analysis point, when the run includes a Harmonic Balance or transient analysis
- Optimization parameters and ranges
- Iteration control and stop behavior
- Optional fine-tuning settings

## Typical GUI Flow

```mermaid
flowchart TD
    A[Load Netlist] --> B[Map Components]
    B --> C[Define Design Goals]
    C --> D[Set Optimization Parameters]
    D --> E[Run Optimization]
    E --> F[Inspect Live Plots and Loss]
    F --> G[Review Saved Results]
```

## Design Goals in Practice

Use goal constraints with optional frequency ranges, for example:

- S11 <= -9 dB in `125-135ghz`
- -3 dB <= S21 <= 0 dB in `125-135ghz`
- `Power_dBm[Out]` >= 10 dB at `35ghz` (single spectral line of an HB run)

Setting the same value for min and max frequency selects one point; the nearest frequency of the sweep or HB spectrum is used.

!!! note
    Goals are transformed into penalty values. Satisfying goals yields zero or negative penalty (reward), violations increase positive penalty.

## Parameter Types

- `MODEL_INPUT`: geometry/model inputs for ONNX surrogate components.
- `NETLIST_VARIABLE`: parsed netlist values patched directly in netlist text.

## During Optimization

The GUI displays:

- live S-parameter traces,
- the HB or transient output spectrum at the selected analysis point, as power, gain, voltage or current,
- per-goal penalty/loss behavior,
- current and best trial values,
- progress against maximum iterations.

When a run produces more than one of the S-parameter, HB and transient plots, a selector switches the left plot between them. See **Advanced -> Harmonic Balance** for the spectrum markers and mixing-product labels, and **Advanced -> Transient Analysis** for the FFT window.

## Outputs

Results are stored in a timestamped folder under `results/` and include generated surrogate/predicted touchstone files, SPICE subcircuits, and run context.
