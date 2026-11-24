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
    "retune_trials": 0,
    "retune_range": 0.3,
    "retune_min_steps": 2,
    "geometries": {}
  }
}
```

### Optimizer

`optimizer.name` is `OptunaOptimizer` or `GradientDescentOptimizer`.

`OptunaOptimizer`'s settings are `multi_objective`, `sampler` (`tpe`, `cmaes`,
`random`, `simulated_annealing`), `pruner` and `random_seed` (default `-1`, no
seed). A seed makes the run repeatable when `parallel_trials` is 1; it reaches the
`tpe`, `cmaes` and `random` samplers, and an optunahub sampler that accepts a seed.

`cmaes` is Optuna's CMA-ES sampler in its margin variant, which keeps moving on the
grid of parameters with a `step`. It adapts one search distribution to the penalty
landscape, which suits continuous circuit values and many parallel trials; in the
OJCAS ablation on the LNA it met the goals as often as TPE. It optimizes a single
objective, so `multi_objective` must be off, and with only one parameter Optuna falls
back to independent sampling.

```json
"optimizer": {"name": "OptunaOptimizer", "settings": {"sampler": "tpe", "random_seed": 42}}
```

`GradientDescentOptimizer` refines a design locally, one trial at a time
(`parallel_trials` must be 1). Its settings are `learning_rate` (first step as a
fraction of each parameter's range, default `0.2`; later steps shrink),
`exploration_scale` (distance of its gradient probes: `1` probes at 10 % of the
range, default `0.5`) and `random_seed`. Give it a start point with the
parameters' `initial_value`; otherwise it starts in the middle of the ranges.

```json
"optimizer": {"name": "GradientDescentOptimizer", "settings": {"learning_rate": 0.05}}
```

When no trial meets every goal, the run ends at the trial that misses them by
the least: once a goal is missed, over-fulfilling the others does not count.
That trial's netlist is the run's netlist, and its folder under `trials/` is
kept. To aim past the specification rather than at its edge (the first trial
that meets every goal often meets one of them by a hair), tighten the goals'
`min_value` and `max_value` by the margin you want.

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
| | `surrogate_thermal_noise` | `false` | Add the surrogates' thermal noise to noise analyses; requires `enforce_passivity`. A fit without it is retried at lower orders, then the trial is penalised. Poor fits (RMS above 3e-3) are retried at lower orders too |
| | `force_dc_pin` | `true` | Pin each surrogate fit to its 0 Hz point also where snp2le skips the pin for passivity below the band |

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
| `initial_value` | Optional value the optimizer evaluates first, e.g. a known design to retune: Optuna's first trial uses it (parameters without one are sampled), gradient descent starts there. Must lie within the bounds and on the `step` grid; a linked parameter takes its master's |

A model may declare that an input only takes values on a grid, such as whole
turns. `cobra parse` then warns when a `model_input` parameter's `step` and
`min_value` allow values off that grid: the model never saw them, and EM
fine-tuning would round them to a different geometry.

```json
{"name": "C5", "type": "netlist_variable", "min_value": 300, "max_value": 400, "step": 1, "unit": "f", "initial_value": 377}
```

### Design Goals

Every goal needs a `parameter`, a `kind`, and at least one of `min_value` and
`max_value`. `weight` (default `1.0`, must be positive) sets its share of the
penalty. A goal's penalty is the mean, over the points of its frequency range, of the
squared miss of its bound divided by `scale`. Without `scale` the miss is relative to
the bound (a miss of 1 dB at −10 dB counts as 0.1), or measured in units of 1 for a
bound of 0, such as a conversion gain of at least 0 dB. Give `scale` (must be positive),
for example `1.0` for dB goals, to measure every goal of a run in the same unit.

`frequency_range` is a single frequency (`"130GHz"`, the nearest point
or spectral line) or a band (`"125-135GHz"`, every point or line inside counts);
`null` uses the whole sweep.

| `kind` | `parameter` | Also required |
| --- | --- | --- |
| `catalogue` | An `.AC` parameter such as `S21_dB`, `Qp`, `k` or `K`; also `NF` (small-signal noise, Xyce and VACASK) and, with VACASK, `NF_SSB`, `NF_DSB` (hbnoise), in dB | — |
| `power_dbm` | `Power_dBm[<node>]` | `node` |
| `gain_db` | `Gain_dB[<port>@<node>]` | `node`; `port`, a driven port; `source_amplitude`, its SIN amplitude in volts; `impedance`, its `z0` |
| `isolation_db` | `Isolation_dB[<node>]` | `node`; `frequency_range`, naming the wanted line |
| `expression` | a name of your choice, e.g. `Compression_dB` | `expression`, a formula; `terms`, its variables and the goal parameters they stand for |

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
subcircuit every resistor is written as a conductance controlled by its own voltage),
unless VACASK's `surrogate_thermal_noise` adds their thermal noise (see
[VACASK Simulator](../advanced/vacask.md#surrogate-noise)).

`node` must be a probe node and `port` a port of the netlist, spelled exactly as
`cobra parse` lists them: names are case-sensitive, so `OUT` and `Out` differ.

The spectrum goals (`power_dbm`, `gain_db`, `isolation_db`) name the analysis
they read in `analysis`: `"HB"` (the default) or `"TRAN"`, in which case the
parameter name carries a `TRAN:` prefix, e.g. `TRAN:Power_dBm[Out]`. Gain goals
store the drive level so their power reference can be reconstructed. An
isolation goal's value is the margin in dB down to the strongest other line in
the spectrum, DC excluded. See [Harmonic Balance](../advanced/harmonic-balance.md)
and [Transient](../advanced/transient.md).

### Analysis Variants

A run holds one result per analysis type, so one netlist cannot carry two HB
analyses at different drives. `analysis_variants` adds them: each is an analysis
of type `analysis` (`AC`, `HB`, `TRAN`, `DC`, `NOISE` or `HBNOISE`), run on a copy
of the netlist with `netlist_values` set. The values use the names of
`netlist_variable` parameters: an element (`"R1"`) or
`"<instance>:<parameter>"` (`"vp_rf:ampl"`, a VACASK sine source's amplitude in
volts). A variant name is an identifier other than an analysis type.

```json
"analysis_variants": [
  {"name": "HB_m40", "analysis": "HB", "netlist_values": {"vp_rf:ampl": 0.00632}},
  {"name": "HB_m15", "analysis": "HB", "netlist_values": {"vp_rf:ampl": 0.112}}
]
```

A goal reads a variant by naming it in `analysis`, and its `parameter` takes the
variant's name as a prefix: `HB_m40:Power_dBm[Vdd]`, `HB_m15:Gain_dB[vp_rf@Out]`,
or a catalogue goal such as `AC_cold:S21_dB` on an `AC` variant. The variant runs
with its type's `simulation_parameters`. A gain goal on a variant refers to the
amplitude the variant sets for its port (`<port>:ampl`); its `source_amplitude`
must be that amplitude. Each variant is one more simulation per trial. In a VACASK
netlist, an `alter` in the control block that sets the same value later wins over
the variant.

Typical uses: the supply power at the operating point (`Power_dBm[Vdd]` at
`0-0.1GHz` on a low-drive HB variant) next to the gain at the nominal drive, or
the gain at a test drive for compression.

### Expression Goals

An `expression` goal bounds a formula over the values of other goals. `terms`
maps each variable of the `expression` to the `parameter` of another goal in the
configuration; that goal keeps its own bounds (give it a loose one when only the
expression matters). Every term is read at the expression goal's own
`frequency_range`, and the formula is applied elementwise. The goal's `parameter`
is a name of your choice, unique among the goals, so that other expressions can
use it as a term: expressions chain, as long as none depends on itself.

The formula takes numbers, the variables, `+ - * / **`, and the functions `abs`,
`sqrt`, `exp`, `ln`, `log10`, and `min`/`max` of two or more values (elementwise).
Terms from several analyses need a single frequency, since their frequency
points need not match. A trial missing an analysis a term reads, or whose formula
gives no finite value (e.g. `ln` of a negative number), fails the goal like any
other.

Gain compression at a test drive, the gain at the netlist's drive minus the gain
on a high-drive variant, at most 1 dB, and the same relative to the gain:

```json
{"parameter": "Compression_dB", "kind": "expression", "expression": "g - g_test",
 "terms": {"g": "Gain_dB[vp_rf@Out]", "g_test": "HB_m15:Gain_dB[vp_rf@Out]"},
 "frequency_range": "35GHz", "max_value": 1.0}
```

```json
{"parameter": "Compression_pct", "kind": "expression", "expression": "100 * c / g",
 "terms": {"c": "Compression_dB", "g": "Gain_dB[vp_rf@Out]"},
 "frequency_range": "35GHz", "max_value": 10.0}
```

`fine_tuning.retune_trials` (default `0`, off) is the budget of the circuit-only
retune that follows a fine-tuning iteration missing the goals: the circuit values are
optimized again, with that iteration's Palace networks as fixed models, in ranges of
± `retune_range` (relative, default `0.3`) around their values, at least
`retune_min_steps` steps wide (default `2`). See
[Fine-Tuning](../advanced/fine-tuning.md#retune-with-the-em-models-fixed).

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
under `if __name__ == "__main__":`, since the worker processes import the script;
without it they run the script again and die, and COBRA stops with an error saying so.

A component whose S-parameters are the same as in an earlier trial is not fitted again:
a fixed Touchstone component, every component of a [retune](../advanced/fine-tuning.md#retune-with-the-em-models-fixed),
or a geometry the optimizer returns to. The last 128 fits of a run are kept; the log
gives the number of fits made and reused at the end.

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