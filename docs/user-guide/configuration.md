---
title: JSON Configuration – COBRA Config File Reference
description: >-
  Schema reference for COBRA JSON configuration files: netlist, component models, design goals, optimization parameters, optimizer settings, and fine-tuning.
---

# JSON Configuration

COBRA configurations are versioned JSON files containing the inputs required to
reproduce an optimization. They can be created in the GUI and executed either in
the GUI or from a terminal.

## Save and Load in the GUI

Use **Save configuration** to write the current run inputs to a JSON file. Use
**Load configuration** to restore them. Loading rebuilds fields that depend on the netlist,
including component model selectors, available HB nodes and ports, simulation
parameters, optimization properties, and goals.

Pressing **Start optimization** automatically saves the exact input configuration
as `cobra_config.json` in the new timestamped results directory. The snapshot is
created before optimization starts, so it remains available if a run fails or is
stopped.

## Start from a Netlist

`cobra init` writes a starter configuration for a netlist, with its analysis
parameters, default optimizer and simulator settings, and the component models
you name:

```bash
cobra init path/to/design.cir --model X1=path/to/model.onnx -o cobra_config.json
```

Add `design_goals` and `optimization_parameters` before running; a
configuration without either is rejected by `cobra parse` and `cobra run`. See
[Command Line](cli.md#starting-a-configuration) for the options.

## Run from the Command Line

```bash
cobra run path/to/cobra_config.json
```

The command prints a header describing the run, shows iteration progress, and
ends with a summary naming the results directory. Add `-v` for debug output,
`-q` for warnings and errors only, or `--log-file PATH` to keep a full debug
log of the run. Running `cobra` without a subcommand continues to open the GUI.
See [Command Line](cli.md) for the full command and option reference.

## Inspect Before Running

`cobra parse` reports what COBRA reads from an input file without starting a
simulation. It accepts either a JSON configuration or a netlist:

```bash
cobra parse path/to/cobra_config.json
cobra parse path/to/design.cir
```

For a netlist it lists the detected analysis, ports and their sources, HB probe
nodes, surrogate components, included and library files, the design-goal
parameters the netlist supports, and every tunable netlist variable with its
current value.

For a configuration it additionally cross-checks the configuration against the
netlist it references:

- every surrogate component in the netlist has a `component_models` entry, and
  every mapped model file exists, loads, and has as many ports as the instance
  has nodes; the `.AC` sweep and `.HB` harmonics stay inside the band an ONNX
  model declares in its metadata (the surrogate is evaluated over that band);
  the model's feasibility constraints are well-formed (see below);
- every `netlist_variable` resolves to an element (or an `instance:parameter`)
  that exists, and every `model_input` matches an input of the mapped ONNX model;
  a warning is raised when a `model_input`'s bounds (its master's, if linked) reach
  beyond the range the model was trained on;
- at least one design goal and one optimization parameter are defined;
- every design goal is buildable, its frequency range parses, and the analysis it
  needs is present or will be injected;
- `simulation_parameters` name directives and `.options` categories the netlist
  actually has;
- fine-tuning geometries resolve and cover every ONNX component.

Findings are grouped by severity. The command exits with `0` when no error was
found and `2` when at least one error would stop a run, so it can gate a run:

```bash
cobra parse config.json && cobra run config.json
```

Useful options:

| Option | Effect |
| --- | --- |
| `--json` | Print the report as JSON instead of text |
| `--kind {auto,config,netlist}` | Override the automatic file-type detection |
| `--full` | Print long lists in full instead of truncating them |
| `--no-model-check` | Skip opening ONNX and Touchstone model files |

The report is written to stdout and diagnostics to stderr, so
`cobra parse config.json --json > report.json` yields a clean JSON file.

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
  "parallel_trials": 1,
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
    "palace_processes": 16,
    "iterations": 3,
    "optimizer": "reuse",
    "geometries": {}
  }
}
```

### Simulator

`simulator.name` is `XyceSimulator` (default) or `VacaskSimulator`. Goals whose
analysis the chosen simulator cannot run are rejected before the run starts.

| Simulator | Setting | Default | Meaning |
| --- | --- | --- | --- |
| `XyceSimulator` | `xyce_command`, `parallel_xyce`, `parallel_xyce_processes`, `enforce_passivity` | — | Xyce executable, MPI ranks, passivity |
| `VacaskSimulator` | `vacask_command` | `"vacask"` | Command name or absolute path |
| | `vacask_threads` | `1` | `vacask -n N`; `0` = auto (`OMP_NUM_THREADS`); keep `1` with `parallel_trials` above 1 |
| | `vector_fit_max_order` | `12` | Highest model order snp2le may use when it fits a surrogate (integer, at least 2) |
| | `enforce_passivity` | `false` | Enforce passivity on the vector fit |

VACASK netlists use VACASK analysis names as `simulation_parameters` keys
(`"AC"`, `"HB"`, `"TRAN"`, `"NOISE"`, `"HBNOISE"`) and VACASK parameter names
(`from`, `to`, `points`, `freq`, `nharm`, ...). Xyce keys keep their leading dot
(`".AC"`). See [VACASK Simulator](../advanced/vacask.md#analyses-and-simulation_parameters).

```json
"simulator": {"name": "VacaskSimulator", "settings": {"vacask_command": "vacask", "vacask_threads": 1, "vector_fit_max_order": 12}},
"simulation_parameters": {"AC": {"points": "500", "from": "1G", "to": "200G"}}
```

A `component_models` key is normally an instance (`X1`). With either simulator
it may instead name a *subcircuit* defined in the netlist or an included file;
COBRA then replaces every instance of it, at any depth, by the surrogate, whose
ports are the subcircuit's pins in order. See
[VACASK Simulator](../advanced/vacask.md#style-2-xschem-netlists).

```text
"component_models": {"ISM_le_new": "../models/ism.onnx"}
```

### Optimization Parameters

| Field | Meaning |
| --- | --- |
| `type` | `model_input` (a geometry input of an ONNX surrogate) or `netlist_variable` (an element value or instance parameter in the netlist) |
| `name` | `<component>:<onnx input>` for `model_input`, e.g. `X1:width`; an element (`C1`) or `<instance>:<parameter>` (`Xq1:Nx`) for `netlist_variable`. `cobra parse` lists both |
| `min_value`, `max_value` | Search bounds. Keep `model_input` bounds inside the range the model was trained on (its `input_parameter_ranges` metadata); the surrogate extrapolates outside it, and `cobra parse` warns |
| `step` | Grid the values snap to; `null` for continuous |
| `unit` | Scale suffix appended verbatim to `netlist_variable` values: `"p"` writes `1.5` as `1.5p`. `null` for `model_input`. VACASK prefixes are case-sensitive: use `"f"`, not `"F"`, for femto (`cobra parse` warns) |
| `linked_to` | Name of another parameter this one mirrors, e.g. for symmetric windings. Links must not form a cycle |

### Design Goals

Every goal needs a `parameter`, a `kind`, and at least one of `min_value` and
`max_value`. `weight` (default `1.0`, must be positive) sets its share of the
penalty. `frequency_range` is a single frequency (`"130GHz"`, the nearest point
or spectral line) or a band (`"125-135GHz"`, every point or line inside counts);
`null` uses the whole sweep.

| `kind` | `parameter` | Also required |
| --- | --- | --- |
| `catalogue` | An `.AC` parameter such as `S21_dB`, `Qp`, `k` or `K`; also `NF` (small-signal noise, Xyce and VACASK) and, with VACASK, `NF_SSB`, `NF_DSB` (hbnoise), in dB | — |
| `power_dbm` | `Power_dBm[<node>]` | `node` |
| `gain_db` | `Gain_dB[<port>@<node>]` | `node`; `port`, a driven port; `source_amplitude`, its SIN amplitude in volts; `impedance`, its `z0` |
| `isolation_db` | `Isolation_dB[<node>]` | `node`; `frequency_range`, naming the wanted line |

`NF` is the small-signal noise figure referred to the input port at 290 K. Both
simulators run it; the mixer goals `NF_SSB` and `NF_DSB` need `VacaskSimulator`
(see [VACASK Simulator](../advanced/vacask.md#noise-figure)). With Xyce, COBRA adds
`.NOISE V(<out>) <port> ...` and `.PRINT NOISE format=csv ONOISE INOISE` when the
netlist has no `.NOISE`. Set the output node in `simulation_parameters`:

```json
"simulation_parameters": {".NOISE": {"out": "Out", "in": "P1"}}
```

`in` defaults to port 1 and must be a `P` port. Without `start_freq`/`stop_freq`
the sweep spans the frequencies of the `NF` goals. Xyce computes its input-referred
noise from an AC solve with every source at its AC magnitude, so for the noise run
COBRA sets the input port to `AC 1` and every other AC magnitude to 0. Xyce's ports
are noiseless, so the noise figure is `F = 1 + INOISE / (4 k T0 z0)` with the input
port's `z0`: the output port terminates the circuit and is left out, as with VACASK.
The vector-fitted surrogates are noiseless in both simulators (in the Xyce SPICE
subcircuit every resistor is written as a conductance controlled by its own voltage).

`node` must be a probe node and `port` a port of the netlist, spelled exactly as
`cobra parse` lists them: names are case-sensitive, so `OUT` and `Out` differ.

The spectrum goals (`power_dbm`, `gain_db`, `isolation_db`) name the analysis
they read in `analysis`: `"HB"` (the default) or `"TRAN"`, in which case the
parameter name carries a `TRAN:` prefix, e.g. `TRAN:Power_dBm[Out]`. Gain goals
store the drive level so their power reference can be reconstructed. An
isolation goal's value is the margin in dB down to the strongest other line in
the spectrum, DC excluded. See [Harmonic Balance](../advanced/harmonic-balance.md)
and [Transient](../advanced/transient.md).

`fine_tuning.palace_processes` is the number of MPI ranks Palace uses per EM
simulation; the Xyce simulator takes the equivalent `parallel_xyce_processes`
setting, used when `parallel_xyce` is enabled. Both must be at least 1 and both
default to the number of cores available on the machine writing the
configuration, so set them explicitly when a configuration is shared between
machines.

`parallel_trials` is how many optimization trials are evaluated at the same
time, each with its own single-threaded simulation in its own directory. It
defaults to 1. For small and medium-sized circuits this is the setting that
actually shortens a run: Xyce is often slower with several MPI ranks than with
one, so N concurrent trials beat one N-rank simulation. Do not combine it with
`parallel_xyce` unless the machine has cores for both — `parallel_trials` times
`parallel_xyce_processes` is the peak load, and COBRA warns when both are set.
The GUI exposes the two as a single *Parallelism* selector and refuses to load
a configuration that sets both.
It requires an optimizer that can suggest a trial before the previous one
reported back: Optuna can, gradient descent cannot and rejects any value above
1. One consequence to be aware of: once the design goals are met, the trials
already running are still finished, so up to `parallel_trials - 1` extra
evaluations may be performed.
With `parallel_trials` above 1 the surrogates are also vector-fitted in that many
worker processes, each using one BLAS thread, because fits in threads of one
process cannot overlap. A Python script that runs COBRA this way must start it
under `if __name__ == "__main__":`, since the worker processes import the script.

The summary's `wall time` is always the run's elapsed time. The per-stage
percentages beside it are shares of the summed stage times, which measure work
done rather than time elapsed and therefore add up to more than `wall time` when
trials ran concurrently — the log line names both numbers so they cannot be
confused.

Fine-tuning presets store their class name and the module ORCA exports its presets
from, `orca.geometry.presets`. Custom geometries
store a JSON-relative Python file and class name. ORCA is imported only when an
enabled configuration requires a geometry.

Unknown fields, unsupported schema versions or backends, broken linked-parameter
references, unavailable analysis-point nodes or ports, and missing input files
are rejected before optimization starts.