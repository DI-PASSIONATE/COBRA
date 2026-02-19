## Setup

Required tools:

- Python 3.11 or higher
- Python package manager (Recommended: `uv`, but `pip` can also be used)
- virtual environment tool (e.g., venv or conda)
- SPICE simulator (currently supported: Xyce), can be installed e.g. via `spack`
- Qucs-S (Schematic editor) for netlist generation
- Optional: Palace EM simulation software (open-source, available at https://github.com/awslabs/palace) for finetuning

### Installation Steps

- Clone this repository

```bash
git clone https://github.com/DavidL-11/COBRA && cd COBRA
```

- (Install UV (![https://docs.astral.sh/uv/getting-started/installation/]))

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Download Python 3.13

```bash
uv python install 3.13
```

- Create and activate a virtual environment

```bash
uv venv --python 3.13
```

- Install COBRA

```bash
uv pip install -e .
```

- If you plan to use Palace for EM simulations, install Palace on your system by following [the Palace installation instructions](https://awslabs.github.io/palace/stable/install/index.html).

You can then use cobra by running `cobra` in your terminal. 