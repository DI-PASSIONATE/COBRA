# Core API

## COBRA

`COBRA` is the orchestration class for the optimization pipeline.

### Constructor

```python
COBRA(
    netlist_parser,
    component_onnx_mapping,
    optimizer=None,              # defaults to a fresh OptunaOptimizer()
    circuit_simulator=None,      # defaults to a fresh XyceSimulator()
    palace_fine_tuning_command=None,
    fine_tuning_iterations=3,
    fine_tuning_optimizer="reuse",
)
```

### Required Inputs

- `netlist_parser`: parsed netlist object with component discovery.
- `component_onnx_mapping`: dictionary mapping every parsed component name to a model path (`.onnx` or `.sNp`).

### Main Method

```python
cobra.run(
    netlist,
    design_goals,
    optimization_parameters,
    max_iterations=500,
    orca_geometries=None,
    callback=None,
    results_name=None,
    sim_params_by_type=None,
    run_configuration=None,
)
```

### `run(...)` Responsibilities

- create a timestamped run directory,
- copy and patch netlist for simulation,
- execute iterative optimization stages,
- track timings and iteration history,
- save context JSON and generated artifacts.

## Practical Notes

!!! warning
    COBRA validates that every parsed component has a corresponding mapping entry. Missing mappings stop execution early.

!!! note
    If goals are not achieved within budget, COBRA re-evaluates best known parameters for final context consistency.
