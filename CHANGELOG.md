# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed
- **Breaking:** the netlist API is split into a parser and a parsed document.
  `XyceNetlistParser().parse_file("x.cir")` returns a `Netlist`, which is passed
  as `COBRA(netlist=...)` (was `from_file` and `COBRA(netlist_parser=...)`).
  `from_file`, `from_lines`, `lines`, `parse_netlist` and `BaseNetlistParser`
  are removed, with no compatibility shim.
  `ConfiguredRun.parser` is now `ConfiguredRun.netlist`, and
  `inspection.load_netlist_parser` is now `load_netlist(path, parser=None, *, has_title=True)`.
- **Breaking:** a simulator owns its parser (`XyceSimulator.netlist_parser`).
  `BaseSimulator` requires `prepare_netlist(netlist, sim_type, sim_params)`
  (moved from `CircuitSimulationStage`) and `run_simulation(netlist_path, netlist)`.
- Edits replace only the edited token, so spacing, inline comments and
  continuation layout are kept and untouched lines render unchanged.
- Performance: a netlist is no longer re-parsed after every edit. Rendering 10
  parameters of the 4 869-line mixer netlist went from about 50 ms to about 0.2 ms.

### Fixed
- `+` continuation lines are part of their statement: their parameters are read
  and edited, and removing a directive removes its continuation lines.
- An `X` instance may come before its `.SUBCKT`.
- Element and parameter names match case-insensitively (`c1` updates `C1`,
  `X1:WIDTH` updates `width`); the netlist keeps its own spelling.
- The first line of a netlist is its title, as in Xyce and ngspice; included
  files are parsed without one.
- Port sources accept `SIN(0 a f)` and `SIN (...)`.
- `key = value` with spaces and `{expr with spaces}` are read correctly.

## 2.0.0 - 2026-10-07

### Added
- `cobra init NETLIST` writes a starter configuration from a netlist: its
  analysis parameters, default settings and the models given with `--model`.
- COBRA reads ORCA's model metadata:
  - The surrogate is evaluated over the band the model declares instead of a
    fixed 1–200 GHz sweep; `cobra parse` warns when `.AC` or `.HB` reach beyond it.
  - Trials whose geometry violates the model's `input_constraints` are skipped
    without simulation and do not count as iterations (Optuna marks them
    pruned). A model with malformed constraints is refused before the run.
  - `cobra parse` and the GUI's model-field tooltip show each model's trained
    ranges, constraints and guarantees; `cobra parse` warns when a
    `model_input`'s bounds exceed the trained range.
- ONNX surrogates that output only the upper triangle of the S-matrix (ORCA's
  `UpperTriangleReImCodec`) are accepted.
- GUI: light and dark themes that follow the system, a single *Parallelism*
  selector (independent trials or MPI), and a design-goal dialog with a tab per
  analysis and whole-sweep, single-frequency or range goals.

### Changed
- **Breaking:** ONNX models without a frequency range in their
  `input_parameter_ranges` metadata are rejected before the run starts.
- **Breaking:** COBRA requires Optuna 5.0. Single-objective TPE is now
  multivariate; pass `{"multivariate": false}` in `sampler_kwargs` to turn it off.
- **Breaking:** EM fine-tuning requires ORCA 2.1 or newer (`pip install
  "cobra-rfic[orca]"`) and runs each component through ORCA's
  `simulate_geometry`. Presets load from `orca.geometry.presets`; configurations
  saved with an older preset module must be updated.
- Each fine-tuning iteration runs in its own folder, `fine_tuning/iteration_NN/`.
- The Hugging Face browser lists models tagged `orca-rfic` instead of
  `orca-surrogate`.
- `gmsh` is no longer a COBRA dependency, and the other dependencies have
  minimum versions.
- The citation points to the published SBCCI 2026 paper
  ([10.1109/SBCCI69725.2026.11706804](https://doi.org/10.1109/SBCCI69725.2026.11706804)).

### Fixed
- EM fine-tuning works with current ORCA: it no longer imports the dropped
  `ihp` PDK, reads results for the geometry's actual port count, handles sweeps
  starting above 1 GHz, and no longer fails on the second iteration of the
  `"reuse"` optimizer. A failed Palace run stops with an error naming the
  component.
- Netlist parameters suggested during fine-tuning are written into the
  simulated netlist; before, only geometry parameters changed.
- Stopping at the fine-tuning prompt still writes
  `cobra_optimization_context.json`.
- The gradient-descent optimizer's gradient had the wrong sign and scale.
- GUI number fields accept dot decimals on systems with a comma-decimal locale.
- `cobra --version`, `cobra doctor` and the run report showed the version as
  `unknown`.

# 1.8.2 - 2026-09-15
### Fixed
- Descriptions, PyPI README and installation instructions updated to reflect the new `cobra-rfic` release

## 1.8.1 - 2026-09-13

### Changed
- The distribution is published on PyPI as `cobra-rfic` (`pip install
  cobra-rfic`); the import package and the `cobra` CLI are unchanged. The
  `orca` extra is disabled until ORCA is published as `orca-rfic`.
- A GitHub release now builds and uploads the package to PyPI via trusted
  publishing (`.github/workflows/publish.yml`).
- `requires-python` is now `>=3.11,<3.14`: PySide6, gmsh and onnxruntime ship
  no wheels for 3.14 yet, so installs on it failed anyway.

## 1.8.0 - 2026-09-13

### Added
- Transient (`.TRAN`) analysis support: COBRA takes the FFT of the settled part
  of the waveform and writes the spectrum as `<netlist>.TRAN.FD.csv`, so the
  power, gain and isolation goals are available as `TRAN:`-prefixed variants
  (`TRAN:Power_dBm[Out]`, `TRAN:Gain_dB[P2@Out]`, `TRAN:Isolation_dB[Out]`) and
  can be mixed with HB goals in one run. Requires `.PRINT tran format=csv`.
- The `.TRAN` arguments are editable under **Simulation Parameters**, and the
  transient spectrum is plotted live like the HB one.
- `cobra parse` warns about a raw-format `.PRINT tran` line and about goal
  frequencies that do not land on an FFT bin.
- `docs/advanced/transient.md` and the `examples/configs/mixer_tran_config.json`
  example.

### Changed
- Spectrum goals in configurations carry an `analysis` field (`"HB"`, the
  default, or `"TRAN"`).
- GUI: the **HB Analysis Point** dropdown became **Analysis Point** and the plot
  selector switches between S-parameter, HB and transient plots.

## 1.7.0 - 2026-09-12

### Added
- `Isolation_dB[<node>]` design goal: the margin in dB between the wanted line
  and the strongest other line in the spectrum (DC excluded), so one goal
  replaces a `Power_dBm` cap per spur. A frequency range is required.
- `examples/configs/mixer_hb_config.json`, a downconverting mixer example using
  the isolation goal.
- `cobra parse` checks that HB goal frequencies lie on the grid the run will
  actually use, including `simulation_parameters` overrides.

### Changed
- GUI: clicking a spectral line toggles its marker, and the click snaps to the
  nearest line.

## 1.6.0 - 2026-09-11

### Added
- `parallel_trials` setting: evaluate several optimization trials concurrently,
  each in its own `trials/trial_<n>/` directory (only the most recent one is
  kept). Supported by Optuna; gradient descent rejects values above 1.
- `parallel_xyce_processes` simulator setting and `fine_tuning.palace_processes`
  replace the hardcoded MPI rank counts; both default to the machine's core
  count.
- `cobra run` prints the final parameters, the goals with their reached values,
  and a summary at the end of a run (`cobra.run_report`).

### Changed
- `COBRA.run()` and the iteration `callback` now return and receive an
  `OptimizationContext` dataclass instead of a dict; fields are accessed as
  attributes (`context.goal_achieved`). The same object is written to
  `cobra_optimization_context.json`.
- A failing Xyce simulation (non-zero exit, no output) is logged as a warning
  and penalised so the optimizer steers away from those parameters; a Xyce that
  cannot be started at all raises `SimulatorError` and stops the run.
- Results directory timestamps use `-` instead of `:` for Windows compatibility.

### Fixed
- Parameters using `linked_to` inherit their master's unit, so the netlist no
  longer receives e.g. `19.0` in place of `19.0F`.
- Touchstone result files were not found by the Xyce simulator (`.s[0-9]p`
  glob missed the file stem), and the Optuna `auto`/`autosampler` sampler
  name now resolves to the OptunaHub `AutoSampler`.

## 1.5.0 - 2026-09-06

### Added
- Pytest test suite covering the netlist parsers, `RunConfiguration`, design goals
  and goal collections, simulation-type resolution, optimizer aggregation, HB
  spectrum handling, config inspection, and the `cobra` CLI, run in CI against
  Python 3.11, 3.12 and 3.13.
- New logging-based CLI with new commands similar to other modern command-line tools (`-v`/`--verbose` (`-vv`
  also includes third-party libraries), `-q`/`--quiet`, `--log-file PATH`, and
  `--no-color` and `cobra doctor`, reporting the requirement status)
- `cobra gui` as an explicit command for the default no-command behaviour, and
  `cobra --version`.
- `docs/user-guide/cli.md`, a reference for the commands, options, exit codes,
  and the stdout/stderr split.

### Changed
- Library code logs instead of printing: every `print()` outside the CLI is now a
  `logging` call on the `cobra` logger tree, so callers control verbosity and can
  capture output. `COBRA.print_time()` became `COBRA.log_stage_times()` and is
  called at the end of a run.
- `cobra run` prints a header describing the run before it starts and a summary
  with the status, wall time, and results directory when it finishes.
- Progress bars follow the log level: `--quiet` hides them.
- The IHP PDK is imported dynamically in `EMFineTuningStage`, keeping the PDK
  optional like the other integrations.

## 1.4.1 - 2026-09-06

### Changed
- Expanded the ruff rule set and applied the resulting style and typing fixes
  across the package.

## 1.4.0 - 2026-09-05

### Added
- `cobra parse TARGET` command that reports the contents of a JSON configuration
  or a netlist without running it, with `--kind`, `--json`, `--full`, and
  `--no-model-check` options.
- Configuration inspection module (`cobra.configuration.inspection`) backing the
  new command, and the netlist-parser accessors it reports from.

### Changed
- Configuration modules moved into the `cobra.configuration` package
  (`configuration.py`, `config_runner.py`, `geometry_loader.py`, `setting.py`).

## 1.3.0 - 2026-08-31

### Added
- Save and load complete run configurations as JSON from the GUI, so a GUI session
  and a headless `cobra run CONFIG` describe the same run.
- `config_runner` module that executes a saved configuration file end to end.
- `geometry_loader` module for resolving built-in and custom geometries referenced
  by a configuration.
- Startup dependency status report that flags a missing or broken ORCA install
  without preventing COBRA from starting.

### Changed
- Example folder restructured into `examples/netlists/` and `examples/configs/`.

## 1.2.0 - 2026-08-31

### Added
- Harmonic balance support: `hb_analysis` module, HB-aware result queries, and
  power and gain design goals evaluated from HB output.
- Gain design goal computed as Pout - Pin by parsing the input power from the
  netlist.
- `SimulationResult` class as the common container for simulation output.
- HB spectrum plot in the GUI, backed by a new `hb_spectrum` module.
- Parsing of additional Xyce print files, so multiple analyses from one run are
  read back.
- `gmsh` as a COBRA dependency.

### Changed
- `DesignGoal` reworked again into a per-goal formula and loss function with
  per-simulation-type queries, and goals are now managed by a
  `DesignGoalCollection`.

### Fixed
- Single-frequency goals, where `min_freq` equals `max_freq`.

## 1.1.0 - 2026-08-31

### Added
- Dynamic GUI fields for simulator and optimizer settings, with hover tooltips.
- Netlist simulation type and port count are detected and displayed in the configuration panel.
- Design goals are populated automatically from the loaded netlist (port count and simulation type).
- GUI: Add Goal dialog shows the required simulation type for the selected parameter.
- Reworked `DesignParameter` and `DesignGoal` to be more modular, making it straightforward to add custom goal types in the future.
- Non-dB S-parameter variants in goal definitions, for future custom goal calculations.
- `.options` statements are parsed from the netlist.

### Changed
- GUI layout restructured: configuration occupies the full left panel; optimization parameters and design goals are stacked on the right.
- The netlist provides the primary simulation type; an `.AC` analysis is added
  automatically when an S-parameter goal is selected.

### Removed
- Hardcoded S-parameter and lumped-element entries from the design goal selector.
- `.LIN` handling, now implicit in `.AC`.
