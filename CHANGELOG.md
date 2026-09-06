# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.5.0 [Unreleased]

### Added
- Pytest test suite covering the netlist parsers, `RunConfiguration`, design goals
  and goal collections, simulation-type resolution, optimizer aggregation, HB
  spectrum handling, config inspection, and the `cobra` CLI, run in CI against
  Python 3.11, 3.12 and 3.13.

### Changed
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
