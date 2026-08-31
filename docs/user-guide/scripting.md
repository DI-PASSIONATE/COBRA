# Script Mode

Script mode provides full automation and reproducibility.

## Minimal Flow

1. Parse netlist using `XyceNetlistParser`.
2. Create `COBRA` with component mappings and optimizer/simulator.
3. Define design goals.
4. Define optimization properties.
5. Call `cobra.run(...)`.

## Reference Example

Use `examples/main.py` as the canonical end-to-end script.

```python
from cobra import (
    COBRA,
    DesignGoal,
    OptimizationProperty,
    OptimizationType,
    OptunaOptimizer,
    XyceSimulator,
)
from cobra.optimizers.design_goal_collection import find_parameter
from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
```

## Key Construction Pattern

```python
parser = XyceNetlistParser().from_file("your_netlist.cir")

cobra = COBRA(
    netlist_parser=parser,
    component_onnx_mapping={
        "X1": "model.onnx",
        "X2": "fixed_component.s6p",
    },
    optimizer=OptunaOptimizer(multi_objective=False, sampler="tpe", pruner="median"),
    circuit_simulator=XyceSimulator(),
)
```

## Defining Goals and Parameters

A goal binds one `DesignParameter` to min/max limits. Look parameters up by name with
`find_parameter(...)`, or list the ones valid for a netlist with `get_available_parameters(num_ports)`.

=== "DesignGoal"

    ```python
    goals = [
        DesignGoal(find_parameter("S11_dB"), max_value=-9, frequency_range="125-135ghz"),
        DesignGoal(find_parameter("S21_dB"), min_value=-3, max_value=0, frequency_range="125-135ghz"),
    ]
    ```

=== "Harmonic Balance goal"

    ```python
    from cobra.optimizers.design_goal_collection import make_power_dbm

    # Output power at one spectral line of an .HB run
    goals.append(
        DesignGoal(make_power_dbm("Out"), min_value=10.0, frequency_range="35ghz")
    )
    ```

    Power and gain parameters depend on the nodes and ports of the circuit, so they are
    built per netlist instead of being looked up by name. See **Advanced -> Harmonic Balance**.

=== "OptimizationProperty"

    ```python
    params = [
        OptimizationProperty(
            name="X1:bottom_winding_diameter",
            type=OptimizationType.MODEL_INPUT,
            min_value=20.0,
            max_value=100.0,
            step=0.1,
        ),
        OptimizationProperty(
            name="Cshunt_p",
            type=OptimizationType.NETLIST_VARIABLE,
            unit="F",
            min_value=0.0,
            max_value=20.0,
            step=1.0,
        ),
    ]
    ```

## Running

```python
context = cobra.run(
    netlist="your_netlist.cir",
    design_goals=goals,
    optimization_parameters=params,
    max_iterations=200,
    results_name="your_experiment_name",
)
```

## Optional Fine-Tuning

You can configure optional EM fine-tuning by providing Palace command and ORCA geometry. See **Advanced -> Fine-Tuning**.
