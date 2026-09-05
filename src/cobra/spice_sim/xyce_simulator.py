import glob
import os
import re
import subprocess

import pandas as pd
import skrf as rf

from cobra.configuration.setting import CobraSetting
from cobra.spice_sim.base_simulator import BaseSimulator, SimulationResult
from cobra.spice_sim.netlist_parsers.netlist_parser import BaseNetlistParser
from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
from cobra.spice_sim.simulation_type import SimulationType, SimulationTypeMetadata
from cobra.spice_sim.vector_fit import vector_fit

_PRINT_FILE_RE = re.compile(r'\bfile=(\S+)', re.IGNORECASE)

# ---------------------------------------------------------------------------
# Xyce-specific simulation metadata
# ---------------------------------------------------------------------------

_XYCE_METADATA: dict[SimulationType, SimulationTypeMetadata] = {
    SimulationType.AC: SimulationTypeMetadata(
        positional_param_names=["sweep_type", "points", "start_freq", "stop_freq"],
        positional_param_descriptions={
            "sweep_type": "Frequency sweep spacing: LIN (linear), DEC (decade), or OCT (octave).",
            "points":     "Number of frequency points in the sweep.",
            "start_freq": "Start frequency (e.g. 100G for 100 GHz).",
            "stop_freq":  "Stop frequency (e.g. 200G for 200 GHz).",
        },
        positional_param_defaults={"sweep_type": "LIN", "points": "500", "start_freq": "1G", "stop_freq": "10G"},
    ),
    SimulationType.HB: SimulationTypeMetadata(
        positional_param_names=["frequencies"],
        positional_param_descriptions={
            "frequencies": "Space-separated list of fundamental frequencies for the Harmonic Balance analysis.\n"
                           "Single tone: e.g. 130G. Multi-tone: e.g. 95E9 10E9.",
        },
        positional_param_defaults={"frequencies": "1G"},
        options_category="hbint",
        options_param_descriptions={
            "numfreq":        "Number of harmonic frequencies (e.g. 3 = DC + 2 harmonics).",
            "startupperiods": "Transient startup periods before HB steady-state. Increase if convergence is difficult.",
            "freq":           "Fundamental frequency (alternative to the positional .HB argument).",
            "maxsteps":       "Maximum Newton iterations per HB solve.",
            "abstol":         "Absolute convergence tolerance for the HB residual.",
            "reltol":         "Relative convergence tolerance for the HB residual.",
            "voltlim":        "Enable voltage limiting during HB Newton iterations (0 = off, 1 = on).",
        },
    ),
    SimulationType.TRAN: SimulationTypeMetadata(
        positional_param_names=["step", "stop_time", "start_time", "max_step"],
        positional_param_descriptions={
            "step":       "Print/output time step.",
            "stop_time":  "Total simulation stop time.",
            "start_time": "Time at which output begins (default 0).",
            "max_step":   "Maximum internal time step (optional).",
        },
        positional_param_defaults={"step": "1n", "stop_time": "100n", "start_time": "0", "max_step": "1n"},
        options_category="timeint",
        options_param_descriptions={
            "abstol":  "Absolute local truncation error tolerance for the time integrator.",
            "reltol":  "Relative local truncation error tolerance for the time integrator.",
            "method":  "Integration method: gear or trap (trapezoid).",
            "maxord":  "Maximum order for the Gear integration method (1–6).",
            "newlte":  "Enable new local truncation error algorithm (0 = off, 1 = on).",
            "delmax":  "Maximum allowed internal time step size.",
        },
    ),
    SimulationType.DC: SimulationTypeMetadata(
        positional_param_names=["src_name", "start", "stop", "incr"],
        positional_param_descriptions={
            "src_name": "Name of the voltage/current source to sweep.",
            "start":    "Sweep start value.",
            "stop":     "Sweep stop value.",
            "incr":     "Sweep increment step.",
        },
        positional_param_defaults={"src_name": "V1", "start": "0", "stop": "1", "incr": "0.01"},
    ),
}

class XyceSimulator(BaseSimulator):
    netlist_parser: BaseNetlistParser = XyceNetlistParser()

    @classmethod
    def get_simulation_metadata(cls, sim_type: SimulationType) -> SimulationTypeMetadata:
        """Return Xyce-specific metadata for *sim_type*."""
        return _XYCE_METADATA.get(sim_type, SimulationTypeMetadata())

    _settings = [
        CobraSetting(
            name="xyce_command",
            dtype=str,
            default="Xyce",
            description=(
                "Command used to invoke the Xyce simulator.\n"
                "Can be a plain command name (e.g. 'Xyce') if it is on PATH,\n"
                "or an absolute path to the Xyce executable."
            ),
        ),
        CobraSetting(
            name="parallel_xyce",
            dtype=bool,
            default=False,
            description=(
                "Run Xyce in parallel using MPI (mpirun -np 8).\n"
                "Requires an MPI-enabled Xyce build and mpirun on PATH.\n"
                "WARNING: Usually a lot slower than single-core Xyce for small and medium-sized circuits."
            ),
        ),
        CobraSetting(
            name="enforce_passivity",
            dtype=bool,
            default=False,
            description=(
                "Enforce passivity on the vector-fitted surrogate model.\n"
                "Increases pre-processing time but prevents non-physical\n"
                "active behaviour in the circuit simulator."
            ),
        ),
    ]

    def __init__(self, xyce_command: str = "Xyce", parallel_xyce: bool = False, enforce_passivity: bool = False):
        self.xyce_command = xyce_command
        self.parallel = parallel_xyce
        self.enforce_passivity = enforce_passivity

    def preprocess_ntwk(self, ntwk, name="cobra_output"):
        # Preprocess the network by vector fitting the S-parameters to create a compact model that can be included in the netlist for circuit simulation.
        return vector_fit(ntwk, name=name, enforce_passivity=self.enforce_passivity)

    def run_simulation(self, netlist_name: str) -> SimulationResult | None:
        results_dir = os.path.dirname(netlist_name)
        netlist_base = os.path.basename(netlist_name)  # e.g. "circuit_hb.cir"

        # --- Determine expected output files from the netlist -----------------
        parser = XyceNetlistParser().from_file(netlist_name)
        sim_type = parser.simulation_type  # SimulationType enum

        # Collect any custom filenames declared via ".PRINT ... file=X"
        custom_print_files: list[str] = []
        for line in parser._lines:
            stripped = line.strip()
            if stripped.lower().startswith(".print"):
                m = _PRINT_FILE_RE.search(stripped)
                if m:
                    custom_print_files.append(os.path.join(results_dir, m.group(1)))

        # --- Run Xyce --------------------------------------------------------
        parallel_command = ["mpirun", "-np", "8"] if self.parallel else []
        command = parallel_command + [
            self.xyce_command, netlist_base
        ]
        proc = subprocess.run(command, capture_output=True, text=True, cwd=results_dir)

        if proc.returncode != 0:
            print(f"Simulation Failed! Return code: {proc.returncode}")
            print(proc.stderr)
            return None

        # --- Collect output files --------------------------------------------
        found: list[str] = []

        if sim_type is SimulationType.AC:
            # AC sweep output is written to <netlist>.s*p (Touchstone) 
            # To avoid .sp files we don't use * to match but rather use regex to match .s followed by a single digit and then p (e.g. .s1p, .s2p, etc.)
            matched_files = glob.glob(os.path.join(results_dir, "*.s[0-9]p"))
            found.extend(matched_files)

        elif sim_type is SimulationType.HB:
            # Xyce HB writes <netlist>.HB.FD.prn (freq-domain) and
            # <netlist>.HB.TD.prn (time-domain); ".PRINT hb format=csv" yields .csv instead.
            found.extend(glob.glob(os.path.join(results_dir, "*.HB.FD.csv")))
            found.extend(glob.glob(os.path.join(results_dir, "*.HB.FD.prn")))

        elif sim_type in (SimulationType.TRAN, SimulationType.DC):
            # Default PRN output for transient / DC sweeps
            found.extend(glob.glob(os.path.join(results_dir, f"{netlist_base}.prn")))

        else:
            # Unknown / UNKNOWN — accept any .prn or .s*p produced nearby
            found.extend(glob.glob(os.path.join(results_dir, "*.prn")))
            found.extend(glob.glob(os.path.join(results_dir, ".s[0-9]p")))

        # Add any files explicitly named in .PRINT file= directives
        for path in custom_print_files:
            if os.path.isfile(path) and path not in found:
                found.append(path)

        if not found:
            print(f"Simulation completed but no output files were found for {sim_type} in {results_dir}")
            return None

        # --- Load Touchstone output as rf.Network (AC only) ------------------
        network: rf.Network | None = None
        sp_files = [f for f in found if re.search(r'\.s\d+p$', f, re.IGNORECASE)]
        if sp_files:
            network = rf.Network(sp_files[0])

        # --- Parse all PRN / table output files with pandas ------------------
        dataframes: dict[str, pd.DataFrame] = {}
        prn_files = [f for f in found if not re.search(r'\.s\d+p$', f, re.IGNORECASE)]
        for prn_path in prn_files:
            try:
                separator = "," if prn_path.lower().endswith(".csv") else r"\s+"
                df = pd.read_csv(prn_path, sep=separator, engine="python")
                df.columns = df.columns.str.strip()
                # Xyce PRN files end with a trailing "End of Xyce(TM) Simulation" line;
                # drop any rows where the first column is not numeric.
                first_col = df.columns[0]
                df = df[pd.to_numeric(df[first_col], errors="coerce").notna()].reset_index(drop=True)
                dataframes[prn_path] = df.apply(pd.to_numeric, errors="coerce")
            except Exception as exc:
                print(f"Warning: could not parse {prn_path}: {exc}")

        return SimulationResult(output_files=found, network=network, dataframes=dataframes)