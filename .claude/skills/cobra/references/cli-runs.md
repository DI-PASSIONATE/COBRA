# COBRA CLI and Long Runs

Use this workflow for saved JSON configs, Xyce setup, background execution, and
progress checks. Read `docs/user-guide/configuration.md` for config details.

## Environment and Xyce

Prefer the repository environment:

```bash
.venv/bin/cobra --help
# or
.venv/bin/python -m cobra --help
```

Gate every run with `cobra parse CONFIG` (exit `0`); what it checks is in
[configuration.md](./configuration.md#validate). `cobra doctor` checks the
packages, Xyce, and Palace at once.

Check Xyce independently:

```bash
command -v Xyce && Xyce --version
```

With Spack, keep setup, loading, verification, and COBRA in one shell:

```bash
source "/absolute/path/to/spack/share/spack/setup-env.sh" && spack load xyce && command -v Xyce && Xyce --version && .venv/bin/cobra run /absolute/path/config.json
```

In this repository Spack is checked out next to it, at `../spack` relative to
the repository root (see `.claude/CLAUDE.md`); use that directory's absolute
path. Replace the package spec when needed. Without Spack, use:

```bash
command -v Xyce && Xyce --version && .venv/bin/cobra run /absolute/path/config.json
```

Ask for a direct Xyce path or Spack setup command if discovery fails. Do not
assume Spack is usable because it is installed.

## Background Execution

Always run in the background; large circuits, HB, fine-tuning, high iteration
counts, difficult goals, complex surrogates, and vector fitting can take hours.
Use absolute paths and keep the log outside `results/` (in a design workflow,
under `runs/<design>/`). Every command below ends the log with a
`COBRA_EXIT=<code>` line, so the outcome can be read from the log even when
nothing else recorded it.

The run command, with Spack (without Spack, drop the `source` and `spack load`
part):

```bash
source "/absolute/path/to/spack/share/spack/setup-env.sh" && spack load xyce && command -v Xyce && Xyce --version && "/absolute/repo/.venv/bin/cobra" run "/absolute/config.json"; echo "COBRA_EXIT=$?"
```

Choose how to start it by how long it may take:

- **Can finish within the agent environment's background-task limit** (smoke
  runs, short runs): start it as a background task that notifies on exit, e.g.
  Claude Code's Bash `run_in_background` with a `timeout` above the expected
  duration. Claude Code stops a background task at its timeout, at most 2 hours.
  Run `bash -lc '<run command>' >"$log" 2>&1` as that task, then wait for the
  notification without polling.
- **May run longer, or no such facility**: detach it so it survives the session,
  and record the PID:

  ```bash
  log="/absolute/runs/<design>/run_r1.log"
  nohup bash -lc '<run command>' >"$log" 2>&1 &
  echo "PID=$! LOG=$log RESULTS=$PWD/results"
  ```

Report the PID or task id, log, config, `results/`, and the new
`results/<timestamp>_<name>/` directory. Do not claim success before
`COBRA_EXIT=0` appears.

## Monitor

When asked for progress, do not start another run. For a detached run, wait
with the environment's wait facility where it has one (in Claude Code, a
`Monitor` until-loop on `! kill -0 <PID>`). Otherwise check at intervals that
suit the run length; never busy-poll in the foreground:

```bash
ps -p <PID> -o pid=,etime=,stat=,cmd=
grep -a "COBRA_EXIT=" /absolute/path/run.log
tail -n 80 /absolute/path/run.log
find results -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort | tail -n 40
```

Read `results/<run>/cobra_optimization_context.json` when present. Report
`goal_achieved`, `iteration`, final parameters, goals/penalties, artifacts, and
timings. Distinguish running, completed, stopped, failed, and no-results states:
no process and no `COBRA_EXIT` line means the run was killed. A directory alone
does not prove success.
