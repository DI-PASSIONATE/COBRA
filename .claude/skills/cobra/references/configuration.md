# COBRA Configuration Workflow

Use schema version 1. This file is the skill's complete reference for what
`cobra parse` reports and for the exact shape of goals and optimization
parameters, so `docs/user-guide/configuration.md` only needs reading for
questions this file does not answer. What to ask the user is in SKILL.md.

## Inspect the Netlist

`cobra parse` is the ground truth for everything the netlist decides. Do not
choose names, variables, or bounds by guesswork:

```bash
.venv/bin/cobra parse /absolute/path/design.cir --json --full
```

The netlist report lists the primary analysis; ports with `z0` and source
amplitude; HB and transient probe nodes; surrogate components that need a
`component_models` entry; included and library files; the goal parameters the
netlist supports (available now, with `.AC`, and with `.HB`); and every tunable
netlist variable with its current value. Without `--full`, long lists are
truncated.

## Scaffold

Start the JSON from the netlist instead of writing it by hand:

```bash
.venv/bin/cobra init /absolute/path/design.cir --model X1=/absolute/path/x1.onnx -o /absolute/path/config.json
```

This fills in the netlist, analysis parameters, default optimizer and simulator
settings, and models, and leaves `design_goals` and `optimization_parameters`
empty. Relative paths in a config resolve from the config file's directory.
Simulation directive values are strings, keyed by `.AC`, `.HB`, `.TRAN`, `.DC`,
or `.OPTIONS:<category>` (e.g. `.OPTIONS:hbint`).

## Design Goals

Every goal object accepts exactly these keys: `parameter`, `kind`,
`frequency_range`, `min_value`, `max_value`, `weight`, `node`, `port`,
`source_amplitude`, `impedance`, `analysis`. Unknown keys are rejected.

| Spec | Field |
| --- | --- |
| "at least X" | `min_value: X` |
| "at most X", "below X" | `max_value: X` |
| a window | both `min_value` and `max_value` |
| relative priority | `weight` > 0 (default `1.0`); set it only when the user ranks goals |
| one frequency | `"frequency_range": "130GHz"` (the nearest point or spectral line) |
| a band | `"frequency_range": "125-135GHz"` (every point or line inside counts) |

At least one of `min_value` and `max_value` is required. Confirm the sign of every
dB spec: "S11 below 10 dB" usually means `S11_dB` with `max_value: -10`. Ask for
the frequency or band rather than leaving `frequency_range` null, which means
the whole sweep.

`parameter` must equal the name COBRA builds, and `node` and `port` must match
the netlist report exactly, including case (`OUT` and `Out` are different
nodes). Copy the names from `cobra parse`.

**Small-signal (`.AC`) goals** use `kind: "catalogue"` and a name from the
report: `S11_dB`, `S21_dB`, `S12_dB`, `S22_dB`, the complex `Sij`, `Lp`, `Ls`,
`Rp`, `Rs`, `Qp`, `Qs`, `k`, `SRF`, `mu`, `mu_prime`, `K`, `Gmax`. `analysis`
is ignored; leave it at `"HB"`.

```json
{"parameter": "S21_dB", "kind": "catalogue", "frequency_range": "125-135GHz", "min_value": 5.0, "max_value": null, "weight": 1.0, "node": null, "port": null, "source_amplitude": null, "impedance": null, "analysis": "HB"}
```

**Large-signal goals** read the HB spectrum (`analysis: "HB"`) or the FFT of a
transient (`analysis: "TRAN"`). A TRAN goal's parameter carries a `TRAN:` prefix
(e.g. `TRAN:Power_dBm[OUT]`). Every large-signal goal needs a `node` that is a
probe node. If the netlist lacks the analysis, COBRA adds it from
`simulation_parameters`.

| Kind | Parameter | Extra keys |
| --- | --- | --- |
| `power_dbm` | `Power_dBm[<node>]` | `node` |
| `gain_db` | `Gain_dB[<port>@<node>]` | `node`; `port` (a driven port); `source_amplitude` (that port's SIN amplitude in V from the report); `impedance` (its `z0`, > 0) |
| `isolation_db` | `Isolation_dB[<node>]` | `node`; `frequency_range` is required and names the wanted line, so the value is the dB margin to the strongest other line (DC excluded) |

```json
{"parameter": "Gain_dB[P1@OUT]", "kind": "gain_db", "frequency_range": "85GHz", "min_value": 0.0, "max_value": null, "weight": 1.0, "node": "OUT", "port": "P1", "source_amplitude": 0.2, "impedance": 50.0, "analysis": "HB"}
{"parameter": "Isolation_dB[OUT]", "kind": "isolation_db", "frequency_range": "85GHz", "min_value": 20.0, "max_value": null, "weight": 1.0, "node": "OUT", "port": null, "source_amplitude": null, "impedance": null, "analysis": "HB"}
{"parameter": "TRAN:Power_dBm[OUT]", "kind": "power_dbm", "frequency_range": "10GHz", "min_value": -10.0, "max_value": null, "weight": 1.0, "node": "OUT", "port": null, "source_amplitude": null, "impedance": null, "analysis": "TRAN"}
```

Details: `docs/advanced/harmonic-balance.md` and `docs/advanced/transient.md`.

## Optimization Parameters

Every parameter object accepts exactly `name`, `type`, `min_value`, `max_value`,
`step`, `unit`, `linked_to`.

| Type | `name` | Bounds | `unit` |
| --- | --- | --- | --- |
| `model_input` | `<component>:<onnx input>`, e.g. `XYLIN_X1:top_linewidth` | inside the model's `input_parameter_ranges` metadata ([design-workflow.md](./design-workflow.md) step 4) | `null` |
| `netlist_variable` | an element (`C1`) or `<instance>:<param>` (`Xnpn13G1:Nx`) from the report | from the user | SPICE suffix appended in the netlist: `"p"` turns `1.5` into `1.5p` |

`step` > 0 snaps values to a grid; `null` means continuous. `linked_to` names
another parameter that this one mirrors (e.g. symmetric windings); links must
not form a cycle.

```json
{"name": "XYLIN_X1:top_linewidth", "type": "model_input", "min_value": 2.0, "max_value": 12.0, "step": 0.1, "unit": null, "linked_to": null}
{"name": "C1", "type": "netlist_variable", "min_value": 0.1, "max_value": 2.0, "step": 0.05, "unit": "p", "linked_to": null}
```

## Validate

Before execution:

1. Validate without running optimization:

   ```bash
   .venv/bin/cobra parse /absolute/path/config.json
   ```

2. Read the `Issues` section, which is grouped by severity. Exit code `0` means no
   error, `2` means at least one error would stop the run. Fix every error and
   review the warnings.
3. Show the config and command; run only after user confirmation.

`cobra parse` loads the configuration with `RunConfiguration.load` (JSON, schema,
paths, fields, bounds, types, links). It then cross-checks it against the parsed
netlist: at least one goal and one parameter exist; model files exist, load, and
have as many ports as the instance has nodes; ONNX inputs match the `model_input`
names; the analysis sweep and harmonics stay inside each ONNX model's band;
optimization parameters resolve; goals are buildable and their frequencies parse;
`simulation_parameters` name real directives; included and library files exist;
and fine-tuning geometries cover every ONNX component. JSON syntax alone is not
enough. The same checks are available in Python from
`cobra.configuration.inspection` (`inspect_path`, `render_report`, `has_errors`).
