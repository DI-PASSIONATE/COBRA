# COBRA Configuration Workflow

Use schema version 1 and follow `docs/user-guide/configuration.md`.

## Required Information

Ask for missing values before creating JSON:

- netlist path;
- component-to-model mapping (`.onnx` or fixed `.sNp`);
- optimization variables, type, bounds, step, and unit;
- goal parameter, bound, signed value, and frequency/range;
- iteration count and optional optimizer/simulator settings;
- HB nodes, ports, and settings when needed;
- Palace command and ORCA geometries for fine-tuning.

Inspect the netlist first when it can reveal component names, ports, variables, or
HB probes. Do not choose variables or bounds by guesswork:

```bash
.venv/bin/cobra parse /absolute/path/design.cir
```

The report lists surrogate components needing a model, ports and their source
amplitude and impedance, HB probe nodes, the design-goal parameters the netlist
supports, and every tunable netlist variable with its current value. Add `--json`
for machine-readable output.

## Goal Translation

Catalogue parameters include `S11_dB`, `S21_dB`, `S12_dB`, `S22_dB`, `Lp`, `Ls`,
`Rp`, `Rs`, `Qp`, `Qs`, `k`, `SRF`, and supported stability factors. HB parameters
are `Power_dBm[<node>]` and `Gain_dB[<port>@<node>]`.

- “At least X” -> `min_value: X`.
- “At most X” or “less than X” -> `max_value: X`.
- AC goals -> `kind: "catalogue"`.
- HB power -> `kind: "power_dbm"`, matching `node`.
- HB gain -> `kind: "gain_db"`, `node`, `port`, `source_amplitude`, and positive
  `impedance`.
- Frequency values include `130GHz` and ranges such as `125-135GHz`.

Confirm signed dB wording. “S11 less than 10 dB” usually means
`S11_dB <= -10`, not `+10 dB`. Ask for a frequency/range; do not silently use
the full sweep.

## Config Shape

```json
{
  "schema_version": 1,
  "netlist": "./design.cir",
  "component_models": {"X1": "./models/x1.onnx"},
  "simulation_parameters": {},
  "optimizer": {"name": "OptunaOptimizer", "settings": {}},
  "simulator": {"name": "XyceSimulator", "settings": {}},
  "max_iterations": 500,
  "optimization_parameters": [],
  "design_goals": [],
  "fine_tuning": {"enabled": false, "palace_command": "palace", "iterations": 3, "optimizer": "reuse", "geometries": {}}
}
```

Use `model_input` or `netlist_variable`. Relative netlist, model, and custom
geometry paths are relative to the config file. Simulation directive values are
strings, such as `.AC`, `.HB`, and `.OPTIONS:hbint`.

## Validate

Before execution:

1. Use `cobra --help`, `cobra run --help`, and `cobra parse --help` if needed.
2. Validate without running optimization:

   ```bash
   .venv/bin/cobra parse /absolute/path/config.json
   ```

3. Read the `Issues` section. Exit code `0` means no error, `2` means at least one
   error would stop the run. Fix every error and review the warnings.
4. Show the config and command; run only after user confirmation.

`cobra parse` loads the configuration with `RunConfiguration.load` (JSON, schema,
paths, fields, bounds, types, links) and then cross-checks it against the parsed
netlist: model mappings and their port counts, ONNX model inputs, resolvable
optimization parameters, buildable design goals, simulation-parameter directives,
included and library files, and fine-tuning geometries. JSON syntax alone is not
enough.
