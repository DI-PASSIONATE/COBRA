import glob
import logging
import os
import re
import subprocess
from typing import ClassVar

import numpy as np
import pandas as pd
import skrf as rf

from cobra.configuration.configuration import ConfigurationError
from cobra.configuration.setting import CobraSetting
from cobra.spice_sim import tran_spectrum
from cobra.spice_sim.base_simulator import (
    BOLTZMANN,
    T0,
    BaseSimulator,
    SimulationResult,
    SimulatorError,
)
from cobra.spice_sim.hb_spectrum import spice_float
from cobra.spice_sim.netlist_parsers.netlist import Netlist
from cobra.spice_sim.netlist_parsers.statement import Statement
from cobra.spice_sim.netlist_parsers.xyce_netlist_parser import XyceNetlistParser
from cobra.spice_sim.simulation_type import SimulationType
from cobra.spice_sim.vector_fit import vector_fit

logger = logging.getLogger(__name__)

#: Default MPI rank count for parallel Xyce runs: one per available core.
DEFAULT_XYCE_PROCESSES: int = os.cpu_count() or 1

#: The noise output run_simulation reads: totals referred to the output and to the input.
_NOISE_PRINT = ".PRINT NOISE format=csv ONOISE INOISE\n"

# Output and solver directives that belong to one analysis; injecting another drops them.
_ANALYSIS_COMPANIONS = frozenset({".PRINT", ".OPTIONS", ".MEASURE", ".FOUR"})

def _is_device_options(statement: Statement, parser: XyceNetlistParser) -> bool:
    """Whether *statement* is ``.OPTIONS DEVICE`` (e.g. TEMP), which applies to every analysis."""
    tokens = parser.tokens(statement)
    return statement.keyword == ".OPTIONS" and len(tokens) > 1 and tokens[1].text.upper() == "DEVICE"


class XyceSimulator(BaseSimulator):
    netlist_parser: ClassVar[XyceNetlistParser] = XyceNetlistParser()
    supported_simulation_types: ClassVar[frozenset[SimulationType]] = frozenset(
        {SimulationType.AC, SimulationType.HB, SimulationType.TRAN, SimulationType.DC, SimulationType.NOISE}
    )
    command_setting: ClassVar[str] = "xyce_command"

    _settings: ClassVar[list[CobraSetting]] = [
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
                "Run Xyce in parallel using MPI (mpirun).\n"
                "Requires an MPI-enabled Xyce build and mpirun on PATH.\n"
                "WARNING: Usually a lot slower than single-core Xyce for small and medium-sized circuits."
            ),
        ),
        CobraSetting(
            name="parallel_xyce_processes",
            dtype=int,
            default=DEFAULT_XYCE_PROCESSES,
            description=(
                "Number of MPI ranks used when parallel_xyce is enabled (mpirun -np N).\n"
                "Defaults to the number of cores available on this machine.\n"
                "Requesting more ranks than the machine has slots makes mpirun fail."
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

    def __init__(
        self,
        xyce_command: str = "Xyce",
        parallel_xyce: bool = False,
        enforce_passivity: bool = False,
        parallel_xyce_processes: int = DEFAULT_XYCE_PROCESSES,
    ):
        if isinstance(parallel_xyce_processes, bool) or not isinstance(parallel_xyce_processes, int):
            raise ConfigurationError("parallel_xyce_processes must be an integer")
        if parallel_xyce_processes < 1:
            raise ConfigurationError(
                f"parallel_xyce_processes must be at least 1, got {parallel_xyce_processes}"
            )
        self.xyce_command = xyce_command
        self.parallel = parallel_xyce
        self.enforce_passivity = enforce_passivity
        self.parallel_processes = parallel_xyce_processes

    def preprocess_ntwk(self, ntwk, name="cobra_output"):
        # Preprocess the network by vector fitting the S-parameters to create a compact model that can be included in the netlist for circuit simulation.
        return vector_fit(ntwk, name=name, enforce_passivity=self.enforce_passivity)

    def prepare_netlist(
        self,
        netlist: Netlist,
        sim_type: SimulationType,
        sim_params: dict[str, str],
        goal_band: tuple[float, float] | None = None,
    ) -> Netlist:
        if any(d.simulation_type is sim_type for d in netlist.simulation_directives):
            if sim_type is SimulationType.NOISE:
                return self._noise_excitation(self._with_noise_print(netlist))
            return netlist  # the existing directive is assumed correct

        # Merge the given params over the built-in defaults
        meta = self.get_simulation_metadata(sim_type)
        merged = {**meta.positional_param_defaults, **sim_params}
        if sim_type is SimulationType.NOISE:
            merged = self._noise_params(netlist, merged, sim_params, goal_band)

        # Build the new directive line(s). A param value may contain several
        # space-separated tokens (e.g. HB frequencies = "95E9 10E9").
        tokens = [sim_type.value]
        for param_name in meta.positional_param_names:
            tokens.extend(merged.get(param_name, "").split())
        lines = [" ".join(tokens) + "\n"]

        # The output files run_simulation collects: Touchstone for AC, CSV otherwise.
        if sim_type is SimulationType.AC:
            lines.append(".LIN format=touchstone sparcalc=1\n")
        elif sim_type in (SimulationType.HB, SimulationType.TRAN):
            probes = " ".join(f"V({n}) I(V{n})" for n in netlist.probe_nodes)
            if probes:
                lines.append(f".PRINT {sim_type.name} format=csv {probes}\n")
        elif sim_type is SimulationType.NOISE:
            lines.append(_NOISE_PRINT)
        injected = netlist.parser.make_statements("".join(lines))

        # Drop the top-level directives of other analyses, each with its
        # continuation lines, and insert the new ones before .END.
        parser = self.netlist_parser
        dropped = parser.analysis_keywords | parser.modifier_keywords | _ANALYSIS_COMPANIONS
        kept = [
            statement for statement in netlist.statements
            if statement.scope is not None
            or statement.keyword not in dropped
            or _is_device_options(statement, parser)
        ]
        end = next(
            (i for i, s in enumerate(kept) if s.keyword == ".END" and s.scope is None),
            len(kept),
        )
        prepared = netlist.with_statements([*kept[:end], *injected, *kept[end:]])
        return self._noise_excitation(prepared) if sim_type is SimulationType.NOISE else prepared

    @staticmethod
    def _noise_params(
        netlist: Netlist,
        merged: dict[str, str],
        sim_params: dict[str, str],
        goal_band: tuple[float, float] | None,
    ) -> dict[str, str]:
        """The arguments of a ``.NOISE`` line COBRA adds: output, input port and sweep."""
        params = dict(merged)
        if not params.get("out"):
            raise ConfigurationError(
                'A .NOISE analysis needs its output node: set simulation_parameters[".NOISE"]["out"].'
            )
        if not params["out"].upper().startswith("V("):
            params["out"] = f"V({params['out']})"
        if not params.get("in"):
            ports = sorted(netlist.ports.items(), key=lambda item: item[1])
            if not ports:
                raise ConfigurationError("A .NOISE analysis needs an input port (P element).")
            params["in"] = ports[0][0]
        if goal_band is not None and not {"start_freq", "stop_freq"} & set(sim_params):
            # Sweep what the noise goals evaluate.
            low, high = goal_band
            params["start_freq"], params["stop_freq"] = f"{low:g}", f"{high:g}"
            if low == high and "points" not in sim_params:
                params["points"] = "1"
        return params

    def _noise_excitation(self, netlist: Netlist) -> Netlist:
        """*netlist* driven by its ``.NOISE`` input alone, with ``AC 1``.

        Xyce refers the noise to the input with the gain of an AC solve in which
        every source keeps its AC magnitude, so a second driven port or an input
        magnitude other than 1 would corrupt INOISE and the noise figure.
        """
        directive = next(
            (d for d in netlist.simulation_directives if d.simulation_type is SimulationType.NOISE),
            None,
        )
        if directive is None or len(directive.positional) < 2:
            return netlist
        source = self.netlist_parser.fold(directive.positional[1])
        parser = self.netlist_parser
        statements = list(netlist.statements)
        for element in netlist.list_elements(["P", "V", "I"]):
            statement = statements[element.statement_index]
            is_input = parser.fold(element.name) == source
            has_ac = any(token.text.upper() == "AC" for token in parser.tokens(statement)[3:])
            if is_input or has_ac:
                statements[element.statement_index] = parser.set_ac_magnitude(
                    statement, "1" if is_input else "0"
                )
        if all(new.text == old.text for new, old in zip(statements, netlist.statements, strict=True)):
            return netlist
        return netlist.with_statements(statements)

    def _with_noise_print(self, netlist: Netlist) -> Netlist:
        """*netlist* with the ``.PRINT NOISE`` line the noise figure needs, if it lacks one."""
        if any(
            directive.analysis == "noise"
            and any(signal.upper() == "INOISE" for signal in directive.signals)
            for directive in netlist.print_directives
        ):
            return netlist
        statements = list(netlist.statements)
        end = next(
            (i for i, s in enumerate(statements) if s.keyword == ".END" and s.scope is None),
            len(statements),
        )
        injected = netlist.parser.make_statements(_NOISE_PRINT)
        return netlist.with_statements([*statements[:end], *injected, *statements[end:]])

    def run_simulation(self, netlist_path: str, netlist: Netlist) -> SimulationResult | None:
        results_dir = os.path.dirname(netlist_path)
        netlist_base = os.path.basename(netlist_path)  # e.g. "circuit_hb.cir"
        sim_type = netlist.simulation_type

        # Collect any custom filenames declared via ".PRINT ... file=X"
        custom_print_files = [
            os.path.join(results_dir, value)
            for directive in netlist.print_directives
            for key, value in directive.kv_params.items()
            if key.lower() == "file"
        ]

        # --- Run Xyce --------------------------------------------------------
        parallel_command = (
            ["mpirun", "-np", str(self.parallel_processes)] if self.parallel else []
        )
        command = [*parallel_command, self.xyce_command, netlist_base]
        # check=False: a non-zero return code is reported below, not raised.
        try:
            proc = subprocess.run(
                command, capture_output=True, text=True, cwd=results_dir, check=False
            )
        except OSError as exc:
            # The executable is missing or cannot be started: no choice of design
            # parameters can fix this, so abort instead of penalising the trial.
            raise SimulatorError(
                f"Could not run '{' '.join(command)}': {exc}. "
                "Check that Xyce is installed and on PATH (or set the xyce_command "
                "setting to its absolute path); run `cobra doctor` to inspect the environment."
            ) from exc

        if proc.returncode != 0:
            # A convergence failure is a property of the parameters, not of the setup:
            # report it and let the caller penalise this trial.
            logger.warning(
                "Xyce failed for %s (return code %s) in %s",
                sim_type,
                proc.returncode,
                results_dir,
            )
            if proc.stderr:
                logger.warning("Xyce stderr:\n%s", proc.stderr.strip())
            return None

        # --- Collect output files --------------------------------------------
        found: list[str] = []

        def outputs(*suffixes: str) -> list[str]:
            """Files Xyce wrote for this netlist, matching any of *suffixes*.

            Xyce names its output after the netlist, so those matches are tried
            first: the surrogate stage writes its own Touchstone predictions into
            the same directory, and a bare ``*.s[0-9]p`` would pick one of those
            up instead of the simulation result. Only when nothing matches does
            this fall back to any file in the directory.
            """
            for prefix in (glob.escape(netlist_base), "*"):
                matches: list[str] = []
                for suffix in suffixes:
                    matches.extend(glob.glob(os.path.join(results_dir, prefix + suffix)))
                if matches:
                    return matches
            return []

        if sim_type is SimulationType.AC:
            # AC sweep output is written to <netlist>.s*p (Touchstone)
            # To avoid .sp files we don't use * to match but rather use regex to match .s followed by a single digit and then p (e.g. .s1p, .s2p, etc.)
            found.extend(outputs(".s[0-9]p"))

        elif sim_type is SimulationType.HB:
            # Xyce HB writes <netlist>.HB.FD.prn (freq-domain) and
            # <netlist>.HB.TD.prn (time-domain); ".PRINT hb format=csv" yields .csv instead.
            found.extend(outputs(".HB.FD.csv", ".HB.FD.prn"))

        elif sim_type is SimulationType.NOISE:
            # ".PRINT noise format=csv" writes <netlist>.NOISE.csv, the default format <netlist>.NOISE.prn.
            found.extend(outputs(".NOISE.csv", ".NOISE.prn"))

        elif sim_type in (SimulationType.TRAN, SimulationType.DC):
            # ".PRINT tran format=csv" writes <netlist>.csv; the default format writes <netlist>.prn.
            for extension in (".csv", ".prn"):
                found.extend(glob.glob(os.path.join(results_dir, netlist_base + extension)))

        else:
            # Unknown / UNKNOWN — accept any .prn or .s*p produced nearby
            found.extend(outputs(".prn", ".s[0-9]p"))

        # Add any files explicitly named in .PRINT file= directives
        for path in custom_print_files:
            if os.path.isfile(path) and path not in found:
                found.append(path)

        if not found:
            logger.warning(
                "Xyce completed but produced no output files for %s in %s",
                sim_type,
                results_dir,
            )
            return None

        # --- Load Touchstone output as rf.Network (AC only) ------------------
        network: rf.Network | None = None
        sp_files = [f for f in found if re.search(r"\.s\d+p$", f, re.IGNORECASE)]
        if sp_files:
            network = rf.Network(sp_files[0])

        # --- Parse all PRN / table output files with pandas ------------------
        dataframes: dict[str, pd.DataFrame] = {}
        prn_files = [f for f in found if not re.search(r"\.s\d+p$", f, re.IGNORECASE)]
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
            except Exception as exc:  # noqa: BLE001 - one unreadable output file must not abort the run
                logger.warning("Could not parse simulation output %s: %s", prn_path, exc)

        # --- Transient: derive the spectrum once, next to Xyce's own output -----
        # Mirrors the <netlist>.HB.FD.csv Xyce writes for HB, so goals and the GUI
        # read a transient result exactly like a Harmonic Balance one.
        if sim_type is SimulationType.TRAN:
            for prn_path, df in list(dataframes.items()):
                if not tran_spectrum.is_time_domain(df):
                    continue
                fd_path = os.path.splitext(prn_path)[0] + ".TRAN.FD.csv"
                try:
                    spectrum = tran_spectrum.to_frequency_domain(df)
                except (KeyError, ValueError) as exc:
                    logger.warning("Could not compute the spectrum of %s: %s", prn_path, exc)
                    continue
                spectrum.to_csv(fd_path, index=False)
                dataframes[fd_path] = spectrum
                found.append(fd_path)

        if sim_type is SimulationType.NOISE:
            resistance = self._input_port_resistance(netlist)
            for frame in dataframes.values():
                columns = {str(column).strip().upper(): column for column in frame.columns}
                if "INOISE" in columns and resistance is not None:
                    frame["NF"] = 10 * np.log10(
                        1.0 + frame[columns["INOISE"]] / (4 * BOLTZMANN * T0 * resistance)
                    )

        return SimulationResult(output_files=found, network=network, dataframes=dataframes)

    @staticmethod
    def _input_port_resistance(netlist: Netlist) -> float | None:
        """The reference impedance of the ``.NOISE`` input port, the source of the noise figure.

        Xyce's P ports are noiseless, so its INOISE is the noise the circuit adds,
        referred to the input; the source's own noise is 4 k T0 z0. The output
        port terminates the circuit and, being noiseless, is left out as well.
        """
        directive = next(
            (d for d in netlist.simulation_directives if d.simulation_type is SimulationType.NOISE),
            None,
        )
        if directive is None or len(directive.positional) < 2:
            return None
        source = directive.positional[1]
        if not netlist.has_element(source) or netlist.get_element(source).port is None:
            logger.warning(
                "The .NOISE input '%s' is not a P port, so COBRA cannot refer a noise figure to it",
                source,
            )
            return None
        params = netlist.get_element(source).params
        z0 = next((value for key, value in params.items() if key.lower() == "z0"), "50")
        try:
            return spice_float(z0)
        except ValueError:
            logger.warning("The z0 of port '%s' is not a number (%r); no noise figure", source, z0)
            return None
