# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.1.0 [Unreleased]

### Added
- CobraSetting class added to define dynamic GUI fields for simulator and optimizer settings, including type, default value, and description.
- GUI: simulator and optimizer settings are now shown as dynamic form fields with hover tooltips
- GUI: simulation type (e.g. `.AC`, `.LIN`, `.HB`) and port count are detected from the netlist and displayed in the configuration panel. Parameters parsed directly from the netlist (e.g. start/stop frequency, number of points), with tooltips describing each field.
- Design goals are now populated dynamically from the netlist (port count, simulation type), including both dB-magnitude and linear-magnitude S-parameters.
- Lumped element parameters (`Lp`, `Rp`, `Qp`, `SRF`, `Ls`, `Rs`, `Qs`, `k`) are only offered as design goals when the simulation type and port count support them.

### Changed
- GUI layout restructured: configuration occupies the full left panel; optimization parameters and design goals are stacked on the right.
- S-parameter design goals are no longer hardcoded — any `S{i}{j}_dB` or `S{i}{j}` combination valid for the loaded netlist is available automatically.

### Removed
- Hardcoded S-parameter entries from the design goal selector.