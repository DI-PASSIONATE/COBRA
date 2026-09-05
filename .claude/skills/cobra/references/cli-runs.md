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

Report what COBRA reads from a config or netlist without running it:

```bash
.venv/bin/cobra parse /absolute/path/config.json
.venv/bin/cobra parse /absolute/path/design.cir
```

It exits with `2` when it reports an error that would stop a run, so it can gate
execution. Run it before every `cobra run` and before writing a config from a
netlist. Add `--json` for machine-readable output.

Check Xyce independently:

```bash
command -v Xyce && Xyce --version
```

With Spack, keep setup, loading, verification, and COBRA in one shell:

```bash
source "$HOME/path/to/spack/share/spack/setup-env.sh" && spack load xyce && command -v Xyce && Xyce --version && .venv/bin/cobra run /absolute/path/config.json
```

Replace the setup path and package spec when needed. Without Spack, use:

```bash
command -v Xyce && Xyce --version && .venv/bin/cobra run /absolute/path/config.json
```

Ask for a direct Xyce path or Spack setup command if discovery fails. Do not
assume Spack is usable because it is installed.

## Background Execution

Always detach runs; large circuits, HB, fine-tuning, high iteration counts,
difficult goals, complex surrogates, and vector fitting can take hours. Keep
the log outside `results/` and use absolute paths.

Without Spack:

```bash
log="$PWD/cobra-run-$(date +%Y%m%d-%H%M%S).log"
nohup bash -lc 'command -v Xyce && Xyce --version && exec "/absolute/repo/.venv/bin/cobra" run "/absolute/config.json"' >"$log" 2>&1 &
echo "PID=$! LOG=$log RESULTS=$PWD/results"
```

With Spack:

```bash
log="$PWD/cobra-run-$(date +%Y%m%d-%H%M%S).log"
nohup bash -lc 'source "$HOME/path/to/spack/share/spack/setup-env.sh" && spack load xyce && command -v Xyce && Xyce --version && exec "/absolute/repo/.venv/bin/cobra" run "/absolute/config.json"' >"$log" 2>&1 &
echo "PID=$! LOG=$log RESULTS=$PWD/results"
```

Report PID, log, config, `results/`, and the new
`results/<timestamp>_<name>/` directory. Do not claim success before exit.

## Monitor

When asked for progress, do not start another run:

```bash
ps -p <PID> -o pid=,etime=,stat=,cmd=
tail -n 80 /absolute/path/cobra-run-YYYYMMDD-HHMMSS.log
find results -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort | tail -n 40
```

Read `results/<run>/cobra_optimization_context.json` when present. Report
`goal_achieved`, `iteration`, final parameters, goals/penalties, artifacts, and
timings. Distinguish running, completed, stopped, failed, and no-results states.
A directory alone does not prove success.
