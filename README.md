# COBRA

COBRA (Circuit-Level Open-Source Based RFIC AI-Assisted Optimizer) is an optimization framework for RFIC workflows.
It combines:

- surrogate-model S-parameter prediction (from ORCA-generated ONNX models),
- circuit-level SPICE simulation,
- and goal-driven optimization (Optuna-based).

## How COBRA Fits with ORCA

COBRA is the optimization/runtime side of the flow.

- ORCA creates the surrogate model (for example, `tf_octa_c_ports.onnx`) from EM simulation data.
- COBRA loads that ONNX model and predicts S-parameters during optimization loops.
- Optional fine-tuning in COBRA can call full-wave EM simulations (Palace) and uses ORCA geometry classes (preset or custom).

In short: ORCA builds the model, COBRA uses it to optimize circuits quickly and can verify/refine with EM fine-tuning.

## Requirements

- Python 3.11+
- Xyce (current circuit simulator backend)
- A valid netlist (`.cir`, exported from Qucs-S)
- An ORCA-generated surrogate ONNX model (`.onnx`)

Optional:

- Palace, if you want EM fine-tuning
- ORCA installed/importable in your Python environment if you use ORCA geometry presets/classes in scripts or GUI fine-tuning

## Installation

### Option A: Using `uv` (recommended)

1. Clone the repository:

```bash
git clone https://github.com/DavidL-11/COBRA
cd COBRA
```

2. Install `uv` (if needed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install a supported Python version:

```bash
uv python install 3.13
```

4. Create and activate a virtual environment:

```bash
uv venv --python 3.13
source .venv/bin/activate
```

5. Install COBRA in editable mode:

```bash
uv pip install -e .
```

### Option B: Using standard `venv` + `pip`

1. Clone the repository:

```bash
git clone https://github.com/DavidL-11/COBRA
cd COBRA
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install COBRA:

```bash
pip install -U pip
pip install -e .
```

## Running COBRA

COBRA supports two main usage modes.

### 1. GUI mode

After installation, start the GUI with:

```bash
cobra
```

The GUI lets you:

- select ONNX model and netlist,
- configure optimization parameters and design goals,
- run/pause/stop optimization,
- visualize S-parameters and goal losses,
- optionally enable fine-tuning with Palace and select ORCA geometry presets or custom geometry classes.

### 2. Python script mode

Use the provided example:

```bash
python examples/main.py
```

The example in `examples/main.py` shows how to:

- define design goals,
- configure `OptunaOptimizer` and `XyceSimulator`,
- pass optimization properties,
- connect ORCA geometry (`TransformerOcta`) for optional fine-tuning,
- call `cobra.run(...)`.

## Minimal Python Usage Pattern

```python
from cobra import COBRA, DesignGoal, DesignParameter, OptimizationProperty, OptimizationType, XyceSimulator, OptunaOptimizer

cobra = COBRA(
	em_surrogate_model="path/to/orca_model.onnx",
	optimizer=OptunaOptimizer(sampler="tpe", pruner="median"),
	circuit_simulator=XyceSimulator(),
)

context = cobra.run(
	netlist="path/to/netlist.cir",
	design_goals=[
		DesignGoal(DesignParameter.S11_dB, max_value=-8, frequency_range="125-135ghz"),
	],
	optimization_parameters=[
		OptimizationProperty(
			name="my_param",
			type=OptimizationType.MODEL_INPUT,
			min_value=0.0,
			max_value=1.0,
			step=0.01,
		)
	],
	max_iterations=200,
)
```

## Typical Inputs and Outputs

Inputs:

- surrogate model `.onnx` (from ORCA)
- circuit netlist `.cir`/`.sp`
- design goals and optimization parameter definitions

Generated artifacts commonly include:

- optimization context JSON (`cobra_optimization_context.json`)
- predicted surrogate S-parameters (for example `surrogate_s_params.sNp`)
- simulator output touchstone files and result netlists

## EM Fine-Tuning Notes (Optional)

If you enable fine-tuning:

- set `palace_fine_tuning_command` in script mode or enable it in GUI,
- provide an ORCA geometry object/class (preset or custom),
- COBRA will run EM verification/fine-tuning iterations in addition to surrogate-based steps.

## Troubleshooting

- If `cobra` command is not found, ensure your virtual environment is activated and reinstall with `pip install -e .`.
- If geometry presets fail to load, verify ORCA is installed and importable in the same environment.
- If circuit simulation fails, verify Xyce is installed and available in your `PATH`.
