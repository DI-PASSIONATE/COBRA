# Autonomous Design Workflow

Follow these steps in order when asked to design or optimize a circuit end to
end. Do not skip or reorder them; each one produces something the next relies on.

COBRA sizes a circuit at schematic level: it tunes surrogate-modelled passives
(ONNX geometry inputs) and netlist values until the design goals are met, and
can verify the passives with full-wave EM. It does not produce a chip layout or
run DRC, LVS, or PDK sign-off. Say so when a request implies more.

## Decision Boundaries

The agent works alone between the approval gate (step 7) and the final report,
inside these limits:

| May decide alone | Must come from the user |
| --- | --- |
| Analysis parameters, via `cobra init` | Netlist, specs, and frequency ranges |
| `model_input` bounds within the model's trained range | Which components and variables may change |
| Goal parameter names that `cobra parse` lists | The sign of every dB spec |
| Fixing `cobra parse` errors | `netlist_variable` bounds |
| More iterations within the budget | Compute budget and number of rounds |
| Widening a `model_input` bound up to its trained range | Relaxing a goal, changing a weight, adding a variable |

Never relax a goal, change goal weights, or tune a variable the user did not
allow. Those are design decisions; report the trade-off instead.

## Run Journal

A design can take hours over several rounds, longer than one conversation or
context window. Keep its state in `runs/<design>/journal.md`, not only in
memory. Create it in step 5, update it at every step marked **Journal**, and
write each entry *before* acting on it, so the last line always says what
happened last.

```markdown
# lna journal

## Spec (approved 2026-10-03)
- netlist: /abs/examples/netlists/LNA/lna_with_trafo_sp.cir
- goals: S21_dB >= 5 @ 125-135GHz; S11_dB <= -10 @ 125-135GHz
- parameters: XYLIN_X1 geometry inputs within trained ranges; C1 0.1-2 p
- budget: max_iterations 500, parallel_trials 8, 3 rounds, 12 h; fine-tuning no

## Round 1
- config: runs/lna/config_r1.json (parse exit 0)
- smoke: passed, log runs/lna/smoke_r1.log
- run: started 2026-10-03 14:02, PID 12345, log runs/lna/run_r1.log
- results: results/2026-10-03_14-02-11_lna_with_trafo_sp, exit 0
- outcome: goal_achieved false; S21_dB penalty 0.8; XYLIN_X1:top_linewidth at max bound
- next: widen top_linewidth to trained max 12.0 in config_r2.json
```

**Resuming.** Before starting a design, and whenever asked about one, look for
`runs/*/journal.md`. If one exists, read it and continue from its last entry
instead of starting over. If the last entry is a started run, check that process
first ([cli-runs.md](./cli-runs.md#monitor)). Never launch a run that the journal
shows may still be active. If the run died without a `COBRA_EXIT` line, report
that and ask before starting it again. An approval recorded in the journal still holds
after a context reset, as long as the work stays within the recorded spec and
budget.

## 1. Check the Environment

```bash
.venv/bin/cobra doctor
```

Exit `1` means a required dependency is missing: stop and report it. With Spack,
run `spack load xyce` in the same shell first (see [cli-runs.md](./cli-runs.md)).
Note whether Palace is found; without it, skip step 12.

## 2. Collect the Specification

Ask once, in a single batch, for whatever is missing:

- netlist path;
- every spec with direction, signed value, and frequency or range
  (confirm that "S11 below 10 dB" means `S11_dB <= -10`);
- which components and netlist variables may change, with bounds for netlist
  variables; a goal never says what may change;
- a model for each surrogate component: an ORCA `.onnx`, a fixed `.sNp`, or a
  Hugging Face `orca-rfic` model downloaded to `./models/<owner>/<repo>/`;
- compute budget: `max_iterations`, `parallel_trials`, wall-clock limit, and how
  many refinement rounds step 11 may run (suggest 3);
- whether EM fine-tuning is wanted, with the ORCA geometry for each ONNX component.

## 3. Inspect the Netlist

```bash
.venv/bin/cobra parse /abs/design.cir --json --full
```

Read the analysis type, ports and their `z0` and sources, surrogate components,
HB probe nodes, supported goal parameters, and tunable variables with their
current values. Check that every spec maps to a listed goal parameter and every
permitted variable exists. For HB goals, the output node must be probed and the
input port driven (`docs/advanced/harmonic-balance.md`).
If a spec cannot be expressed, stop and say why.

## 4. Inspect the Surrogate Models

`cobra parse` lists ONNX input names but not their trained ranges; read them
from the model metadata:

```bash
.venv/bin/python -c "import onnxruntime as ort; m = ort.InferenceSession('/abs/model.onnx', providers=['CPUExecutionProvider']).get_modelmeta().custom_metadata_map; [print(k, '=', v) for k, v in m.items()]"
```

- `input_parameter_ranges`: the trained range of every geometry input. These are
  the outer limits for `model_input` bounds, since the surrogate is not valid
  outside them.
- `input_parameter_ranges.frequency`: the model's band. It must cover the
  analysis sweep, the HB harmonics, and every goal range, or the vector fit
  extrapolates. A model without it is rejected.
- `physics_guarantees`: if `passive` is `false`, consider the simulator setting
  `enforce_passivity: true`.

## 5. Scaffold the Configuration

Give each design its own directory, e.g. `runs/<design>/`, and write the starter
config there:

```bash
.venv/bin/cobra init /abs/design.cir --model X1=/abs/model.onnx -o runs/<design>/config_r1.json
```

This fills in the netlist, analysis parameters, default optimizer and simulator
settings, and component models. Every component listed as unmapped on stderr
needs a model before the config can run.

**Journal:** create `runs/<design>/journal.md` with the spec collected in step 2.

## 6. Fill In Goals and Parameters

Edit the config following [configuration.md](./configuration.md):

- `design_goals`: one entry per spec, using the exact names from step 3.
- `optimization_parameters`:
  - `model_input` named `<component>:<input>` for each permitted geometry input,
    bounded by the user's range or else by the trained range from step 4;
  - `netlist_variable` for each permitted variable, with the user's bounds;
  - `linked_to` for inputs that must stay equal, such as symmetric windings;
  - a `step` that matches the process grid when the user gave one.
- `max_iterations` and `parallel_trials` from the budget. The run stops early
  once every goal is met, so `max_iterations` is an upper limit, not a target.

Then validate in a loop until the exit code is `0`:

```bash
.venv/bin/cobra parse runs/<design>/config_r1.json
```

Fix every ERROR. Read every WARNING. Band, sweep, and goal-frequency warnings
usually mean a spec or a model does not fit the netlist.

## 7. Approval Gate

Show the user, once, in a compact table: the goals, the parameters and bounds
(marking which came from model metadata), the budget, the round limit, the
config path, and the run command. Wait for approval. Everything up to step 13
then runs without further questions, unless a decision from the table above
comes up.

**Journal:** record the approval date and the approved spec and budget.

## 8. Smoke Run

Copy the config to `config_r1_smoke.json` with `max_iterations: 2` and
`parallel_trials: 1`, then run it in the background with its log at
`runs/<design>/smoke_r1.log`. This catches Xyce, vector-fitting, and HB
convergence failures in minutes instead of hours. If it fails, diagnose with
`docs/advanced/troubleshooting.md`, fix the config, and go back to step 6. If
the fix changes anything approved in step 7, ask again.

**Journal:** record the smoke result and its log.

## 9. Full Run

Start the approved config in the background, as shown in
[cli-runs.md](./cli-runs.md), with its log at `runs/<design>/run_r<N>.log`.
Never run it in the foreground, and never start a second run on the same config
while one is active.

**Journal:** record the start time, the PID or background task id, and the log,
before you wait on anything.

## 10. Monitor and Read the Results

Wait for the run as described in [cli-runs.md](./cli-runs.md#monitor): an
exit notification where the agent environment provides one, otherwise checks at
intervals that suit the run length, never busy polling. When it exits, read
`results/<timestamp>_<netlist>/cobra_optimization_context.json`:

- `goal_achieved`: whether every goal was met;
- `iteration`, `model_parameters`, `netlist_parameters`: the best trial. In a
  `multi_objective` run that missed its goals these describe the last trial, so
  find the best ones in `iterations` instead;
- `goals`: each goal's `current_penalty`, where zero or negative means met and
  positive means missed by that much;
- `iterations`: one record per step with its parameters and `losses`, one
  penalty per goal in the order of `goals`. Use it to judge convergence and to
  see which goal limits the design.

Do not trust a results directory alone; confirm the process exited with code `0`.

**Journal:** record the results directory, the exit code, and the outcome:
`goal_achieved`, the goals that missed with their penalties, and any parameter
whose best value sits on a bound.

## 11. Refine When Goals Are Missed

Diagnose before changing anything, then write the next round as a new file
(`config_r2.json`, ...) and go back to step 6's validation. Stop after the
agreed number of rounds.

| Symptom in the results | Action |
| --- | --- |
| Best value of a parameter sits at its bound | Widen it up to the trained range for `model_input`; ask first for `netlist_variable` |
| Loss still falling at the end of the run | Raise `max_iterations` within the budget |
| Loss flat, and one goal is met only when another fails | The goals conflict. Show the user the trade-off from `iterations` (the non-dominated `losses`). Offer a round with `optimizer.settings.multi_objective: true` to map it properly; COBRA does not write the Pareto front, so compute it from `iterations`. Let the user choose before going on |
| A goal never moves | Check that its frequency lies inside the sweep and that a permitted variable affects it |
| Simulations fail | Fix them using the troubleshooting guide, then repeat step 8 |

**Journal:** start a `## Round N` section with the diagnosis and the change,
before writing the new config.

## 12. EM Fine-Tuning (Optional)

Only when the surrogate run met the goals and the user asked for EM
verification. In a new config round, set `fine_tuning.enabled: true` and add a
geometry for every ONNX component: either `{"source": "preset", "module": ...,
"class_name": ...}` or `{"source": "custom", "file": ..., "class_name": ...}`. A
Hugging Face model provides its class in `<repo>.py`. Palace sweeps the band in
the geometry's simconfig file, not the circuit sweep; a narrower simconfig
shortens the run (`docs/advanced/fine-tuning.md`).
Validate, run in the background, and read the results as in step 10.
Fine-tuning iterations are stored under `fine_tuning/iteration_NN/`.

**Journal:** record it as its own round, marked `fine-tuning`.

## 13. Final Report

Report:

- whether every goal was met, with each goal's achieved value and margin;
- the final parameter values, the config that produced them, and the results
  directory (its netlist holds the best parameters);
- whether the result is surrogate-only or EM-verified;
- the rounds used and what changed between them;
- open trade-offs or decisions waiting for the user.

Do not describe a surrogate-only result as verified, and do not describe any
result as a finished chip.

**Journal:** end it with `## Done` and the same summary, so a later session does
not resume a finished design.
