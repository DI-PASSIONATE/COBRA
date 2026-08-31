# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.1.0 [Unreleased]

### Added
- Dynamic GUI fields for simulator and optimizer settings, with hover tooltips.
- Netlist simulation type and port count are detected and displayed in the configuration panel.
- Design goals are populated automatically from the loaded netlist (port count and simulation type).
- GUI: Add Goal dialog shows the required simulation type for the selected parameter.
- Reworked `DesignParameter` and `DesignGoal` to be more modular, making it straightforward to add custom goal types in the future.

### Changed
- GUI layout restructured: configuration occupies the full left panel; optimization parameters and design goals are stacked on the right.

### Removed
- Hardcoded S-parameter and lumped-element entries from the design goal selector.