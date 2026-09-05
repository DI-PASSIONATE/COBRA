# COBRA Python Runs

Use this workflow when the user wants to run or create a Python script.
Read `docs/user-guide/scripting.md` and `docs/api/core.md` first. Use existing
examples as templates; do not invent APIs.

## Environment

Run from the repository environment:

```bash
.venv/bin/python --version
.venv/bin/python path/to/script.py
```

If `.venv/` is missing, ask which environment to use. Do not silently use a
global interpreter. Check Xyce separately with `command -v Xyce` and
`Xyce --version`.

## Script Review

Before running, check that the script supplies a parser, component model mapping,
design goals, optimization parameters, and a simulator. Verify model paths,
component names, goal signed values, and parameter bounds. For HB and
fine-tuning, also verify probes, ports, and ORCA geometries.

## Long Runs

A Python optimization may take minutes or hours. Run it in the background,
redirect output, and report PID, absolute script path, log path, and results
path. Use the same Spack setup in the child shell when required:

```bash
log="$PWD/cobra-python-$(date +%Y%m%d-%H%M%S).log"
nohup bash -lc 'source "$HOME/path/to/spack/share/spack/setup-env.sh" && spack load xyce && command -v Xyce && Xyce --version && exec "/absolute/repo/.venv/bin/python" "/absolute/script.py"' >"$log" 2>&1 &
echo "PID=$! LOG=$log RESULTS=$PWD/results"
```

Without Spack, remove the `source` and `spack load` commands. Do not claim
success before the process exits.
