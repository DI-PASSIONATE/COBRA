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

### Return value: `OptimizationContext`

`run(...)` returns an `OptimizationContext` — the dataclass that carries state
through every stage. It is also what `callback` receives on each iteration, and
what is written to `cobra_optimization_context.json`.

```python
context = cobra.run(...)
context.goal_achieved      # bool: were all goals met?
context.iteration          # iterations actually run
context.netlist_parameters # {name: "2.5n"} applied to the netlist
context.model_parameters   # {name: value} fed to the surrogate
context.goals              # DesignGoal objects, carrying current_value / current_penalty
context.simulation_results # {SimulationType: SimulationResult} — a type that failed is absent
context.times              # seconds per stage, plus "total_time"
context.iterations         # one record per optimizer step
context.results_dir        # where everything was written
```

A field a stage does not own should be treated as read-only. `max_iterations`
is the exception: raising it mid-run is how the GUI continues past the original
budget.

!!! note
    Fields are accessed as attributes, not as dict keys. `context["goal_achieved"]`
    raises `TypeError`; use `context.goal_achieved`.

## Practical Notes

!!! warning
    COBRA validates that every parsed component has a corresponding mapping entry. Missing mappings stop execution early.

!!! note
    If goals are not achieved within budget, COBRA re-evaluates best known parameters for final context consistency.
