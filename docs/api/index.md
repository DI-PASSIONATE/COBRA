# API Reference

This section documents the public API used in normal COBRA workflows.

## Recommended Reading Order

1. Read **Home** first for workflow and concept overview.
2. Use this API reference for signatures and object responsibilities.
3. Use `examples/main.py` as an integration template.

## API Sections

- **Core**: top-level orchestration (`COBRA`).
- **Optimizers**: goal and parameter definitions plus optimizer choices.
- **Simulation and Netlist Parsing**: simulator backend and parser behavior.

!!! tip
    Prefer stable imports from the package root when available:

    ```python
    from cobra import COBRA, DesignGoal, OptimizationProperty, OptunaOptimizer, XyceSimulator
    ```
