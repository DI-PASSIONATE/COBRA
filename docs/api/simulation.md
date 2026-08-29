# Simulation and Netlist Parsing API

## XyceSimulator

`XyceSimulator` executes circuit-level simulation and works with generated SPICE-compatible surrogate subcircuits.

Responsibilities:

- run Xyce backend,
- coordinate vector fitting flow,
- return simulated network data for goal checking.

One simulation is run per analysis type required by the goals. Results are returned as a `SimulationResult` holding an S-parameter `Network` for `.AC` runs and, for `.HB` runs, the frequency-domain table (`.HB.FD.csv` or `.HB.FD.prn`) as a DataFrame.

## Netlist Parsing

`XyceNetlistParser` handles netlist ingestion and controlled updates.

Capabilities include:

- parse input netlist,
- identify component instances to map,
- patch netlist variables during optimization,
- read simulation, `.options` and `.PRINT` directives,
- expose `hb_probe_nodes`, the nodes carrying both `V(node)` and `I(Vnode)` in an HB run,
- expose `port_sources`, the SIN/AC drive level and impedance per port,
- save updated netlist for simulation stage.

When a goal requires an analysis the netlist does not declare, the stage writes a copy with the missing directive injected — `.LIN` for S-parameter output, or `.HB` together with a matching `.PRINT HB format=csv` line.

## Harmonic Balance Spectra

`hb_spectrum` turns an HB result table into a spectrum and is shared by the design-goal formulas and the GUI plot, so both always agree.

- `probe_nodes(df)` lists the usable analysis points in a result.
- `spectrum(df, node, quantity, frequency_range, pin_dbm)` returns `(frequencies, values)` as power (dBm), gain (dB), voltage (dBV) or current (dBmA).
- `classify_bins(freqs, fundamentals, max_order)` labels each line as `DC`, a harmonic (`H2`), or a mixing product (`2f1-f2`).
- `available_power_dbm(amplitude, z0)` converts a port's SIN amplitude to available input power.

See **Advanced -> Harmonic Balance** for the underlying conventions.

## Component Model Strategy

Each parsed component maps to one source path:

- ONNX surrogate model for optimizable geometry behavior, or
- fixed Touchstone file when component response is static.

## Vector Fit Integration

COBRA uses vector fitting so surrogate/frequency-domain responses can be represented in SPICE-compatible subcircuit form for Xyce simulation.

!!! note
    This conversion is central for integrating surrogate S-parameter behavior into a standard circuit simulation loop.
