# Simulation and Netlist Parsing API

## XyceSimulator

`XyceSimulator` executes circuit-level simulation and works with generated SPICE-compatible surrogate subcircuits.

Responsibilities:

- run Xyce backend,
- coordinate vector fitting flow,
- return simulated network data for goal checking.

## Netlist Parsing

`XyceNetlistParser` handles netlist ingestion and controlled updates.

Capabilities include:

- parse input netlist,
- identify component instances to map,
- patch netlist variables during optimization,
- save updated netlist for simulation stage.

## Component Model Strategy

Each parsed component maps to one source path:

- ONNX surrogate model for optimizable geometry behavior, or
- fixed Touchstone file when component response is static.

## Vector Fit Integration

COBRA uses vector fitting so surrogate/frequency-domain responses can be represented in SPICE-compatible subcircuit form for Xyce simulation.

!!! note
    This conversion is central for integrating surrogate S-parameter behavior into a standard circuit simulation loop.
