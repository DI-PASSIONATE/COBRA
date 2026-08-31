# JSON Configuration

COBRA configurations are versioned JSON files containing the inputs required to
reproduce an optimization. They can be created in the GUI and executed either in
the GUI or from a terminal.

## Save and Load in the GUI

Use **Save Config** to write the current run inputs to a JSON file. Use **Load
Config** to restore them. Loading rebuilds fields that depend on the netlist,
including component model selectors, available HB nodes and ports, simulation
parameters, optimization properties, and goals.

Pressing **START OPTIMIZATION** automatically saves the exact input configuration
as `cobra_config.json` in the new timestamped results directory. The snapshot is
created before optimization starts, so it remains available if a run fails or is
stopped.

## Run from the Command Line

```bash
cobra run path/to/cobra_config.json
```

The command prints iteration progress and the final results directory. Running
`cobra` without a subcommand continues to open the GUI.

## Path Rules

Relative paths are interpreted relative to the directory containing the JSON
file, not the process working directory. When the GUI saves a configuration,
netlist, component model, and custom geometry paths are rewritten relative to the
new JSON destination where possible. This makes a directory containing a config
and its inputs portable as a unit.

## Schema Version 1

The following example shows the supported top-level structure. Fields that do not
apply to a run may use empty objects or arrays.

```json
{
  "schema_version": 1,
  "netlist": "../circuits/lna.cir",
  "component_models": {
    "X1": "../models/lna.onnx"
  },
  "simulation_parameters": {
    ".AC": {
      "points": "500",
      "start_freq": "100G",
      "stop_freq": "150G"
    },
    ".HB": {
      "frequencies": "130G"
    },
    ".OPTIONS:hbint": {
      "numfreq": "5",
      "startupperiods": "2"
    }
  },
  "optimizer": {
    "name": "OptunaOptimizer",
    "settings": {
      "multi_objective": false,
      "sampler": "tpe",
      "pruner": null
    }
  },
  "simulator": {
    "name": "XyceSimulator",
    "settings": {}
  },
  "max_iterations": 500,
  "optimization_parameters": [
    {
      "name": "X1:width",
      "type": "model_input",
      "min_value": 10.0,
      "max_value": 30.0,
      "step": 0.1,
      "unit": null,
      "linked_to": null
    }
  ],
  "design_goals": [
    {
      "parameter": "S21_dB",
      "frequency_range": "125-135ghz",
      "min_value": -3.0,
      "max_value": null,
      "weight": 1.0,
      "kind": "catalogue",
      "node": null,
      "port": null,
      "source_amplitude": null,
      "impedance": null
    },
    {
      "parameter": "Power_dBm[Out]",
      "frequency_range": "130ghz",
      "min_value": 10.0,
      "max_value": null,
      "weight": 1.0,
      "kind": "power_dbm",
      "node": "Out",
      "port": null,
      "source_amplitude": null,
      "impedance": null
    }
  ],
  "fine_tuning": {
    "enabled": false,
    "palace_command": "palace",
    "iterations": 3,
    "optimizer": "reuse",
    "geometries": {}
  }
}
```

Optimization parameter types are `model_input` and `netlist_variable`. Dynamic HB
goals use `power_dbm` or `gain_db` as their `kind`; gain goals additionally store
the input port, source amplitude, and impedance required to reconstruct their
power reference.

Fine-tuning presets store their Python module and class name. Custom geometries
store a JSON-relative Python file and class name. ORCA is imported only when an
enabled configuration requires a geometry.

Unknown fields, unsupported schema versions or backends, broken linked-parameter
references, unavailable HB nodes or ports, and missing input files are rejected
before optimization starts.