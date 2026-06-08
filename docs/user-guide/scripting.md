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
    DesignParameter,
    OptimizationProperty,
    OptimizationType,
    OptunaOptimizer,
    XyceSimulator,
)
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

=== "DesignGoal"

    ```python
    goals = [
        DesignGoal(DesignParameter.S11_dB, max_value=-9, frequency_range="125-135ghz"),
        DesignGoal(DesignParameter.S21_dB, min_value=-3, max_value=0, frequency_range="125-135ghz"),
    ]
    ```

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
