---
title: Command Line – cobra init, run, parse and doctor
description: >-
  Reference for the cobra command line: start a configuration from a netlist, run a JSON configuration, parse a netlist, check the environment with cobra doctor, output options, and exit codes.
---

# Command Line

The `cobra` command starts a configuration from a netlist, runs a saved
configuration, inspects an input file, checks the environment, or opens the GUI.

```bash
cobra                              # open the graphical interface
cobra init design.cir              # write a starter configuration for a netlist
cobra run config.json              # run a saved configuration headlessly
cobra parse design.cir             # report what COBRA reads from an input file
cobra doctor                       # check simulators and optional packages
cobra --version
cobra <command> --help
```

## Commands

| Command | Purpose |
| --- | --- |
| `init NETLIST` | Write a starter JSON configuration for a netlist |
| `run CONFIG` | Execute a saved JSON configuration without the GUI |
| `parse TARGET` | Report a configuration or netlist without running it |
| `doctor` | Report the Python packages and simulators COBRA found |
| `gui` | Open the graphical interface (also the default with no command) |

## Output Options

These options are accepted before or after the command, so both
`cobra -v run config.json` and `cobra run -v config.json` work.

| Option | Effect |
| --- | --- |
| `-v`, `--verbose` | Show debug output; `-vv` also includes third-party libraries |
| `-q`, `--quiet` | Report warnings and errors only, and hide the progress bar |
| `--log-file PATH` | Also write a timestamped debug log to `PATH` |
| `--no-color` | Disable coloured output (the `NO_COLOR` variable does the same) |

`--log-file` always records at debug level, whatever the console shows, so a
long run can be diagnosed afterwards:

```bash
cobra run config.json --quiet --log-file results/run.log
```

## What Goes Where

Requested output — the `parse` report, the `run` header and summary, the
`doctor` table, the path `init` wrote — is written to **stdout**. Progress and
diagnostics are written to **stderr**. Piping a report therefore stays safe:

```bash
cobra parse config.json --json > report.json
```

## Starting a Configuration

`cobra init` writes a configuration filled in from a netlist, to be completed
and checked before a run:

```bash
cobra init design.cir --model X1=models/coil.onnx -o design_config.json
# add design_goals and optimization_parameters, then:
cobra parse design_config.json && cobra run design_config.json
```

| Option | Effect |
| --- | --- |
| `-o`, `--output PATH` | Where to write the file (default: `<netlist stem>_config.json` in the current directory) |
| `--model NAME=PATH` | ONNX or Touchstone model for component `NAME`; repeat for each component |
| `--force` | Overwrite an existing output file |

The file contains the netlist, the parameters of its analysis directives (`.AC`,
`.HB`, `.TRAN`, `.DC`), the default optimizer and simulator settings, and a
model for every component given with `--model` or whose netlist `TSTONEFILE`
exists. Paths are written relative to the output file. Components still without
a model are listed on stderr, together with the design-goal parameters the
netlist supports.

`design_goals` and `optimization_parameters` are left empty: they are design
decisions COBRA cannot infer. `cobra parse` and `cobra run` reject a
configuration until both have at least one entry.

## Running a Configuration

```bash
cobra run path/to/cobra_config.json
```

The run starts with a header describing what is about to happen, shows a
progress bar while it iterates, and ends with a summary:

```text
COBRA 1.5.0
  netlist     amplifier.cir
  analysis    AC
  optimizer   OptunaOptimizer
  simulator   XyceSimulator
  parameters  4
  goals       2
  iterations  up to 500

COBRA optimization:  14%|█████                | 68/500 [02:41<17:03,  2.37s/it]

Summary
  status      design goals achieved at iteration 68
  wall time   2m 41s
  results     results/2026-09-06_11-04-22_amplifier
```

## Checking the Environment

`cobra doctor` reports the interpreter, the required packages and simulators,
and the optional components whose absence only disables a feature. It exits
with `1` when something required is missing, which makes it usable as a
pre-flight check in CI.

```bash
cobra doctor
```

Xyce is required for circuit simulation; when it comes from Spack, load it in
the same shell first:

```bash
. ../spack/share/spack/setup-env.sh && spack load xyce
cobra doctor
```

## Exit Codes

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | The run failed, or `doctor` found a missing requirement |
| `2` | Invalid input, or a `parse` report containing errors |
| `130` | Interrupted (Ctrl+C) |

This makes `parse` usable as a gate:

```bash
cobra parse config.json && cobra run config.json
```
