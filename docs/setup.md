## Installation

## Requirements

- Python 3.11+
- Xyce simulator available in your `PATH`
- Qucs-S for generating/inspecting compatible netlists

Optional:

- ORCA installed/importable if you use ORCA geometry presets/classes
- Palace if you want EM fine-tuning

!!! note
	COBRA supports Python 3.11 to 3.14.

## Clone Repository

```bash
git clone https://github.com/DI-PASSIONATE/COBRA
cd COBRA
```

## Install COBRA

=== "Option A: uv (recommended)"

	```bash
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv python install 3.13
	uv venv --python 3.13
	source .venv/bin/activate
	uv pip install -e .
	```

=== "Option B: venv + pip"

	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	pip install -U pip
	pip install -e .
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