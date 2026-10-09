---
title: Simulation API – XyceSimulator and Netlist Parsing
description: >-
  API reference for XyceSimulator, netlist parsing, Harmonic Balance and transient spectra, component model strategies, and vector fitting in COBRA.
---

# Simulation and Netlist Parsing API

## XyceSimulator

`XyceSimulator` executes circuit-level simulation and works with generated SPICE-compatible surrogate subcircuits.

Responsibilities:

- own the Xyce netlist parser (`XyceSimulator.netlist_parser`),
- prepare the netlist for each analysis (`prepare_netlist`),
- run Xyce backend (`run_simulation(netlist_path, netlist)`),
- coordinate vector fitting flow,
- return simulated network data for goal checking.

One simulation is run per analysis type required by the goals. Results are returned as a `SimulationResult` holding an S-parameter `Network` for `.AC` runs and, for `.HB` runs, the frequency-domain table (`.HB.FD.csv` or `.HB.FD.prn`) as a DataFrame.

`run_simulation` returns `None` when the simulation itself failed — a non-zero
Xyce exit code or no output files. `CircuitSimulationStage` records no result
for that analysis type, and `DesignGoalChecker` gives every goal that needed it
`FAILED_SIMULATION_PENALTY`, so the optimizer avoids those parameters. A
simulator that cannot be started at all (Xyce missing or not executable) raises
`SimulatorError` instead and aborts the run.

## Netlist Parsing

Parsing and the parsed document are separate. A stateless **parser** knows one dialect; the **`Netlist`** it returns holds the document and all queries and edits.

| Class | Module (`cobra.spice_sim.netlist_parsers`) | Role |
| --- | --- | --- |
| `NetlistParser` | `netlist_parser` | Abstract dialect contract: `parse(text, *, has_title=True)`, `parse_file(path, *, has_title=True)`, `analysis_metadata`, and the token-level edit hooks. |
| `SpiceNetlistParser` | `spice_netlist_parser` | SPICE-family syntax: title line, `*` `;` `$` comments, `+` continuations, `.SUBCKT` scope, `.MODEL` `.INCLUDE` `.LIB` `.OPTIONS` `.PRINT` and analyses. |
| `XyceNetlistParser` | `xyce_netlist_parser` | Xyce additions: `P` ports, `Y` devices, `.HB`, `.LIN`, analysis metadata, and Qucs-S `YLIN` / `TSTONEFILE` normalisation. |
| `Netlist` | `netlist` | The parsed document (built from immutable `Statement`s) plus its records: `Component`, `Include`, `Library`, `SimulationDirective`, `PrintDirective`, `OptionsDirective`, `Subcircuit`. |

```python
from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser

netlist = XyceNetlistParser().parse_file("examples/netlists/Mixer/mixer_hb.cir")
netlist.components                  # instances to map to models
netlist.set_value("C1", "2p")       # replaces only that token
netlist.save("patched.cir")
```

`Netlist` provides:

- queries: `list_elements`, `get_element`, `components`, `inline_subckt_names`, `includes`, `libraries`, `simulation_type`, `simulation_directives`, `print_directives`, `options_directives`, `num_ports`, `port_sources` (the SIN/AC drive level and impedance per port), `probe_nodes` (nodes carrying both `V(node)` and `I(Vnode)` in an HB or transient run), `available_design_parameters`,
- edits: `set_value`, `set_model`, `set_param`, `update_parameters`, `update_simulation_directive(SimulationType, params)`, `update_options_directive(category, params)`,
- output: `copy()`, `to_string()`, `save(path)`.

Behaviour worth knowing:

- An edit replaces only the edited token. Spacing, inline comments and continuation layout are kept, and untouched lines render unchanged.
- `+` continuation lines belong to their statement: parameters on them are read and edited, and removing a directive removes them too.
- Element and parameter names match case-insensitively (`c1` updates `C1`, `X1:WIDTH` updates `width`); the netlist keeps its own spelling. Component-model mapping keys stay exact-case.
- The first line is the title, as in Xyce. Included files are parsed without one (`has_title=False`).
- Nothing is re-parsed after an edit. Each trial renders `template.copy()` plus edits, and the circuit stage parses the rendered file once and derives every analysis variant in memory.

When a goal requires an analysis the netlist does not declare, `XyceSimulator.prepare_netlist(netlist, sim_type, sim_params)` returns a copy with the missing directive injected — `.LIN` for S-parameter output, or `.HB` / `.TRAN` together with a matching `.PRINT <analysis> format=csv` line. It returns the netlist itself when the analysis is already declared.

## Adding a Simulator Dialect

A simulator subclasses `BaseSimulator` and owns its parser through the class attribute `netlist_parser`.

1. Pick a parser base: subclass `SpiceNetlistParser` for a SPICE-like dialect (for example ngspice), or `NetlistParser` for a different syntax (for example VACASK). Dialect rules are hooks on the parser: `fold` (name comparison; case-insensitive by default), `probe_nodes` (large-signal analysis points), and for SPICE dialects the `blocks` class attribute (scopes such as `.SUBCKT`/`.ENDS`, or ngspice's `.CONTROL`/`.ENDC`) and `comment_start`.
2. Set it on the simulator: `netlist_parser = MyDialectParser()`. `get_simulation_metadata` delegates to its `analysis_metadata`.
3. Implement `prepare_netlist(netlist, sim_type, sim_params) -> Netlist` and `run_simulation(netlist_path, netlist) -> SimulationResult | None`.

## Harmonic Balance Spectra

`hb_spectrum` turns an HB result table into a spectrum and is shared by the design-goal formulas and the GUI plot, so both always agree.

- `probe_nodes(df)` lists the usable analysis points in a result.
- `spectrum(df, node, quantity, frequency_range, pin_dbm)` returns `(frequencies, values)` as power (dBm), gain (dB), voltage (dBV) or current (dBmA).
- `classify_bins(freqs, fundamentals, max_order)` labels each line as `DC`, a harmonic (`H2`), or a mixing product (`2f1-f2`).
- `available_power_dbm(amplitude, z0)` converts a port's SIN amplitude to available input power.

## Transient Spectra

`tran_spectrum` turns a `.PRINT tran` time series into the same table layout, so everything above applies to a transient result too.

- `to_frequency_domain(df)` resamples the waveform onto a uniform grid and returns the one-sided FFT, normalised so each bin holds amplitude/2 like an HB phasor. `XyceSimulator` calls it for every transient result and writes the table as `<netlist>.TRAN.FD.csv`.
- `on_fft_grid(window, low, high)` tells whether a `.TRAN` window of the given length puts a bin inside a goal's frequency range.

See **Advanced -> Harmonic Balance** for the underlying conventions.

## Component Model Strategy

Each parsed component maps to one source path:

- ONNX surrogate model for optimizable geometry behavior, or
- fixed Touchstone file when component response is static.

## Vector Fit Integration

COBRA uses vector fitting so surrogate/frequency-domain responses can be represented in SPICE-compatible subcircuit form for Xyce simulation.

!!! note
    This conversion is central for integrating surrogate S-parameter behavior into a standard circuit simulation loop.
