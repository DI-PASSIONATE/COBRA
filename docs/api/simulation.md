---
title: Simulation API – XyceSimulator, VacaskSimulator and Netlist Parsing
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

## VacaskSimulator

`VacaskSimulator` runs native [VACASK](../advanced/vacask.md) netlists with the
same responsibilities as `XyceSimulator`: it owns a `VacaskNetlistParser`, injects
or switches analyses in `prepare_netlist`, and `run_simulation` calls
`vacask -sp -qp -n N` and converts the raw files into a `SimulationResult`.
Settings: `vacask_command`, `vacask_threads`, `vector_fit_max_order`, `enforce_passivity`.

- `read_raw(path) -> list[RawPlot]` (`cobra.spice_sim.raw_file`) reads a SPICE raw file;
  it raises `ValueError` for a file it cannot read.
- `vector_fit_vacask(nw, name, enforce_passivity=False, max_order=12)` (`cobra.spice_sim.vector_fit`)
  models a network with [snp2le](https://github.com/iic-jku/snp2le) (universal mode) and
  writes the VACASK subcircuit `<basename>_subct` to `<name>.inc`; it returns the file name.
  `vacask_include(subcircuit_text)` makes that text self-contained: `ground GND`, the
  `load` lines it needs, and `model` bindings inside the subcircuit. All resistors are
  `noisy=0`. The Xyce path still uses scikit-rf's SPICE writer.
- `add_port_signals(frame, resistances)` (`cobra.spice_sim.vacask_simulator`) adds
  `V(<src>) = R*I(<src>)` and `I(V<src>) = I(<src>)` to an HB or transient table, so
  each port is a probe point named after its source.
- `SimulationType.NOISE` (`.NOISE`) and `SimulationType.HBNOISE` (`.HBNOISE`) are the
  noise analyses; the catalogue goals `NF` (NOISE), `NF_SSB` and `NF_DSB` (HBNOISE) read them.
  `XyceSimulator` runs `.NOISE` and adds an `NF` column to its noise table;
  `VacaskSimulator` runs both.
- `noiseless_resistors(subcircuit)` (`cobra.spice_sim.vector_fit`) rewrites every
  `R n1 n2 r` of a SPICE subcircuit as `GR n1 n2 n1 n2 1/r`: the same conductance, but
  noiseless. `vector_fit` applies it to the Xyce surrogates.
- `BaseSimulator.prepare_netlist(netlist, sim_type, sim_params, goal_band=None)`:
  *goal_band* is the frequency span the goals on that analysis evaluate; both
  simulators sweep it in a noise analysis they add.

`BaseSimulator` subclasses declare two more class attributes:

| Attribute | Meaning |
| --- | --- |
| `supported_simulation_types` | `frozenset[SimulationType]` the simulator can run; `build_design_goals(configs, netlist, simulator=None)` rejects goals needing another analysis |
| `command_setting` | Name of the setting that holds the executable, e.g. `"vacask_command"` |

`SIMULATOR_REGISTRY` in `cobra.configuration.config_runner` maps the configuration's
`simulator.name` to the class.

## Netlist Parsing

Parsing and the parsed document are separate. A stateless **parser** knows one dialect; the **`Netlist`** it returns holds the document and all queries and edits.

| Class | Module (`cobra.spice_sim.netlist_parsers`) | Role |
| --- | --- | --- |
| `NetlistParser` | `netlist_parser` | Abstract dialect contract: `parse(text, *, has_title=True)`, `parse_file(path, *, has_title=True)`, `analysis_metadata`, and the token-level edit hooks. |
| `SpiceNetlistParser` | `spice_netlist_parser` | SPICE-family syntax: title line, `*` `;` `$` comments, `+` continuations, `.SUBCKT` scope, `.MODEL` `.INCLUDE` `.LIB` `.OPTIONS` `.PRINT` and analyses. |
| `XyceNetlistParser` | `xyce_netlist_parser` | Xyce additions: `P` ports, `Y` devices, `.HB`, `.LIN`, analysis metadata, and Qucs-S `YLIN` / `TSTONEFILE` normalisation. |
| `VacaskNetlistParser` | `vacask_netlist_parser` | VACASK syntax: title line, `//` and `/* */` comments, `model`, `load`, `include`, `subckt`, `control ... endc`; ports from `acsp`; scans included files for external masters. |
| `Netlist` | `netlist` | The parsed document (built from immutable `Statement`s) plus its records: `Component`, `Include`, `Library`, `SimulationDirective`, `PrintDirective`, `OptionsDirective`, `Subcircuit`. |

Surrogates by subcircuit name use these members:

| Member | Meaning |
| --- | --- |
| `Netlist.masters` | Subcircuits (defined here or in an include) as name → pins |
| `Netlist.select_surrogates(names)` | Marks those of `names` that name a subcircuit, not an instance, as surrogate masters |
| `Netlist.mark_surrogate_masters(names)` | Same for subcircuit names only; `KeyError` for an unknown one |
| `Netlist.use_surrogate(component, subcircuit)` | Points an instance at the fitted `subcircuit`; for a master, renames every instance at any depth and adds the include |
| `Subcircuit.pins` | Pin names in definition order |
| parser `instance_master(statement)` | Subcircuit a statement instantiates, or `None` |
| parser `surrogate_include(component)` | The include statement: `include "<name>.inc"` (VACASK), `.INCLUDE "<name>.sp"` (Xyce) |
| `VacaskNetlistParser.grounds(netlist)` | Declared ground names, `{"0"}` by default |

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

A parser can override these hooks for its dialect:

- `ports(netlist)` → port element name to port number (`Netlist.ports`),
- `port_sources(netlist)` → SIN/AC drive level and impedance per port (`Netlist.port_sources`),
- `resolve_include(file_path, directory)` → where the simulator finds an included file (default: relative to `directory`).

`Netlist.external_masters` holds the folded names of subcircuits and models defined in included files; instances of them are not surrogate components.

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
