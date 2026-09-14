---
title: Installation – Install COBRA, Xyce, Qucs-S, ORCA and Palace
description: >-
  Install COBRA from PyPI or source, set up Xyce, Qucs-S, and optional ORCA and Palace, and verify the environment with cobra doctor.
---

# Installation

## Requirements

- Python 3.11 to 3.13
- Xyce simulator available in your `PATH`
- Qucs-S for generating/inspecting compatible netlists

Optional:

- ORCA installed/importable if you use ORCA geometry presets/classes
- Palace if you want EM fine-tuning

!!! note
	COBRA supports Python 3.11 to 3.13; 3.14 has no PySide6, gmsh or onnxruntime wheels yet.

## Install COBRA

=== "Developers: uv (recommended)"

	`uv sync` creates `.venv/` from `uv.lock` with COBRA in editable mode and the
	dev tools (`pytest`, `ruff`, `ty`). Use `uv run <command>` or activate the
	environment.

	```bash
	git clone https://github.com/DI-PASSIONATE/COBRA
	cd COBRA
	curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is missing
	uv sync --python 3.12
	source .venv/bin/activate
	```
=== "Users: from PyPI"

	COBRA is published as [`cobra-rfic`](https://pypi.org/project/cobra-rfic/);
	the import package and the CLI are still called `cobra`.
	This is the easiest way to install COBRA for users, but it doesn't allow any modifications to the existing code or the provided examples.

	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	pip install cobra-rfic
	```

	Or as an isolated `uv` tool that puts `cobra` on your `PATH`:

	```bash
	uv tool install cobra-rfic
	```

=== "Developers: venv + pip"

	```bash
	git clone https://github.com/DI-PASSIONATE/COBRA
	cd COBRA
	python3 -m venv .venv
	source .venv/bin/activate
	pip install -U pip
	pip install -e . pytest pytest-cov
	```

## Verify Setup

Run the following checks in your activated environment:

```bash
cobra
```

Expected behavior: COBRA GUI starts.

To verify script workflow:

```bash
python examples/main.py
```

!!! warning
	`examples/main.py` requires external tools (including Xyce), valid model files, and compatible netlist inputs.

## Running the Tests

The test suite is a quick way to confirm an installation is sound.

=== "uv"

	```bash
	uv run pytest
	```

=== "venv + pip"

	```bash
	pytest
	```

For a coverage report:

```bash
uv run pytest --cov
```

!!! note
	The GUI is not covered by the suite; verify `cobra` interactively after
	changing anything under `src/cobra/gui/`.

## External Tool Notes

### Xyce

- COBRA currently uses Xyce as the circuit simulator backend.
- Ensure `Xyce` executable is on `PATH`.

### Qucs-S

- Qucs-S is used to create netlists (`.cir`) that COBRA parses.

### Palace (Optional)

- If you use EM fine-tuning, install Palace and ensure your command invocation is valid.
- See official instructions: <https://awslabs.github.io/palace/stable/install/index.html>

### ORCA (Optional)

- ORCA is used to generate surrogates and optional geometry workflows.

## Next Steps

- Continue with **Getting Started -> Quickstart** for a first run.
- Use **User Guide -> Script Mode** if you prefer GUI-free automation.
