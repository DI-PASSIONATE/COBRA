"""VACASK circuit simulator backend.

Runs a native VACASK netlist (see
:mod:`cobra.spice_sim.netlist_parsers.vacask_netlist_parser` for COBRA's
conventions) and turns its raw files into the results the design goals read,
in the same layout as the Xyce backend:

* ``acsp`` → an :class:`skrf.Network`, also written as ``<netlist>.sNp``;
* ``hb``   → ``FREQ``, ``Re(V(n))``, ``Im(V(n))``, ``Re(I(inst))``, … in
  ``<netlist>.HB.FD.csv``, with the phasors halved: VACASK reports peak
  amplitudes, COBRA (like Xyce) amplitude/2;
* ``tran`` → ``TIME``, ``V(n)``, ``I(inst)``, … in ``<netlist>.csv`` plus its
  spectrum in ``<netlist>.TRAN.FD.csv``.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import skrf as rf

from cobra.configuration.configuration import ConfigurationError
from cobra.configuration.setting import CobraSetting
from cobra.spice_sim import tran_spectrum
from cobra.spice_sim.base_simulator import BaseSimulator, SimulationResult, SimulatorError
from cobra.spice_sim.netlist_parsers.vacask_netlist_parser import (
    VacaskNetlistParser,
    vacask_float,
)
from cobra.spice_sim.raw_file import RawPlot, read_raw
from cobra.spice_sim.simulation_type import SimulationType
from cobra.spice_sim.vector_fit import vector_fit_vacask

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cobra.spice_sim.netlist_parsers.netlist import Netlist, SimulationDirective

logger = logging.getLogger(__name__)

_FLOW_SUFFIX = ":flow(br)"
_S_PARAMETER_RE = re.compile(r"^s\((\d+),(\d+)\)$")
#: Name suffix of the image-sideband twin COBRA adds to an hbnoise analysis.
_IMAGE_SUFFIX = "_img"
#: Reference temperature of the noise figure (IEEE), in kelvin.
T0 = 290.0
_KELVIN = 273.15
#: Defaults for analysis parameters outside the editable slots.
_EXTRA_DEFAULTS: dict[SimulationType, dict[str, str]] = {
    # LO at 1 GHz; signal in the upper sideband of the first LO harmonic, output at the offset.
    SimulationType.HBNOISE: {"freq": "1G", "inspur": "1", "outspur": "0"},
}


class VacaskSimulator(BaseSimulator):
    netlist_parser: ClassVar[VacaskNetlistParser] = VacaskNetlistParser()
    supported_simulation_types: ClassVar[frozenset[SimulationType]] = frozenset({
        SimulationType.AC,
        SimulationType.HB,
        SimulationType.TRAN,
        SimulationType.NOISE,
        SimulationType.HBNOISE,
    })
    command_setting: ClassVar[str] = "vacask_command"
    named_analysis_parameters: ClassVar[bool] = True

    _settings: ClassVar[list[CobraSetting]] = [
        CobraSetting(
            name="vacask_command",
            dtype=str,
            default="vacask",
            description=(
                "Command used to invoke the VACASK simulator.\n"
                "Can be a plain command name (e.g. 'vacask') if it is on PATH,\n"
                "or an absolute path to the vacask executable."
            ),
        ),
        CobraSetting(
            name="vacask_threads",
            dtype=int,
            default=1,
            description=(
                "Threads VACASK uses for device evaluation and its linear solver (vacask -n N).\n"
                "0 lets VACASK pick (honours OMP_NUM_THREADS). With parallel trials,\n"
                "keep this at 1 so the trials do not compete for cores."
            ),
        ),
        CobraSetting(
            name="vector_fit_max_order",
            dtype=int,
            default=12,
            description=(
                "Highest model order snp2le may use when it vector-fits a surrogate.\n"
                "Higher follows the S-parameters more closely, but very high orders can\n"
                "place poles far out of band, which VACASK then solves inaccurately."
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
        vacask_command: str = "vacask",
        vacask_threads: int = 1,
        enforce_passivity: bool = False,
        vector_fit_max_order: int = 12,
    ):
        if isinstance(vacask_threads, bool) or not isinstance(vacask_threads, int):
            raise ConfigurationError("vacask_threads must be an integer")
        if vacask_threads < 0:
            raise ConfigurationError(f"vacask_threads must be at least 0, got {vacask_threads}")
        self.vacask_command = vacask_command
        self.threads = vacask_threads
        self.enforce_passivity = enforce_passivity
        if (
            isinstance(vector_fit_max_order, bool)
            or not isinstance(vector_fit_max_order, int)
            or vector_fit_max_order < 2
        ):
            raise ConfigurationError(
                f"vector_fit_max_order must be an integer of at least 2, got {vector_fit_max_order!r}"
            )
        self.vector_fit_max_order = vector_fit_max_order

    def preprocess_ntwk(self, ntwk, name="cobra_output"):
        # snp2le models the S-parameters as a VACASK subcircuit in <name>.inc.
        return vector_fit_vacask(
            ntwk, name=name, enforce_passivity=self.enforce_passivity, max_order=self.vector_fit_max_order
        )

    # -------------------------------------------------------------------------
    # Netlist preparation
    # -------------------------------------------------------------------------

    def prepare_netlist(
        self,
        netlist: Netlist,
        sim_type: SimulationType,
        sim_params: dict[str, str],
        goal_band: tuple[float, float] | None = None,
    ) -> Netlist:
        """Keep only the *sim_type* analyses of the control block, adding one if there is none.

        The other analyses are switched off (commented out), so one run produces
        exactly the result *sim_type* needs.  An ``hbnoise`` analysis gets an
        image-sideband twin for the SSB noise figure.  A noise analysis COBRA
        adds sweeps *goal_band* unless *sim_params* give ``from`` or ``to``.
        """
        parser = self.netlist_parser
        analyses = [d for d in netlist.simulation_directives if d.name is not None]
        wanted = [d for d in analyses if d.simulation_type is sim_type]
        names = {d.name for d in analyses}
        images = {
            d.statement_index: line
            for d in wanted
            if sim_type is SimulationType.HBNOISE
            and not (d.name or "").endswith(_IMAGE_SUFFIX)
            and f"{d.name}{_IMAGE_SUFFIX}" not in names
            and (line := self._image_line(d.name or "", d.positional, d.kv_params)) is not None
        }
        if wanted and len(wanted) == len(analyses) and not images:
            return netlist  # the netlist runs exactly this analysis already

        if not wanted and sim_type is SimulationType.AC:
            raise ConfigurationError(
                "The VACASK netlist has no acsp analysis, so COBRA does not know its ports. "
                'Declare them with one, e.g. analysis sp1 acsp ports=["vp1", "rp1", "vp2", "rp2"] '
                'from=1G to=10G mode="lin" points=100'
            )
        injected = [] if wanted else self._injected_lines(netlist, sim_type, sim_params, goal_band)

        dropped = {d.statement_index for d in analyses if d.simulation_type is not sim_type}
        lines: list[str] = []
        for index, statement in enumerate(netlist.statements):
            if index in dropped:
                lines.extend(parser.disable(statement))
                continue
            lines.extend(statement.lines)
            if index in images:
                lines.append(images[index])
            if statement.keyword == "control" and statement.scope is None:
                lines.extend(injected)
                injected = []
        if injected:  # the netlist has no control block
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.extend(["control\n", *injected, "endc\n"])

        statements = parser.normalize(parser.split("".join(lines)))
        return netlist.with_statements(statements)

    def _injected_lines(
        self,
        netlist: Netlist,
        sim_type: SimulationType,
        sim_params: dict[str, str],
        goal_band: tuple[float, float] | None,
    ) -> list[str]:
        """The analysis lines for a *sim_type* the netlist lacks, from defaults and *sim_params*."""
        meta = self.get_simulation_metadata(sim_type)
        params = {
            **meta.positional_param_defaults,
            **_EXTRA_DEFAULTS.get(sim_type, {}),
            **sim_params,
        }
        if sim_type in (SimulationType.NOISE, SimulationType.HBNOISE):
            if goal_band is not None and not {"from", "to"} & set(sim_params):
                # Sweep what the goals evaluate (for hbnoise: the output offset).
                low, high = goal_band
                params["from"], params["to"] = f"{low:g}", f"{high:g}"
                if low == high and "points" not in sim_params:
                    params["points"] = "1"
            if "out" not in params:
                raise ConfigurationError(
                    f"A {sim_type.name} analysis needs its output node: add one to the netlist "
                    f'or set simulation_parameters["{sim_type.name}"]["out"].'
                )
            if "in" not in params:
                pairs = self.netlist_parser.port_pairs(netlist)
                if not pairs:
                    raise ConfigurationError(
                        f"A {sim_type.name} analysis needs its input port source: declare ports "
                        f'with an acsp analysis or set simulation_parameters["{sim_type.name}"]["in"].'
                    )
                params["in"] = pairs[0][0]
        positional = [params.pop(name, "") for name in meta.positional_param_names]
        name = f"cobra_{sim_type.name.lower()}"
        lines = [self._analysis_line(name, sim_type, positional, params)]
        if sim_type is SimulationType.HBNOISE:
            image = self._image_line(name, positional, params)
            if image is not None:
                lines.append(image)
        return lines

    def _analysis_line(
        self, name: str, sim_type: SimulationType, positional: list[str], params: dict[str, str]
    ) -> str:
        """An ``analysis`` statement named *name* with the given slot values and parameters."""
        parser = self.netlist_parser
        stub = parser.make_statements(f"analysis {name} {parser.analysis_keyword(sim_type)}\n")[0]
        if self.get_simulation_metadata(sim_type).positional_param_names == ["freq"]:
            positional = " ".join(positional).split()
        return "  " + parser.set_directive(stub, positional, params).text

    def _image_line(self, name: str, positional: list[str], params: Mapping[str, str]) -> str | None:
        """The image-sideband twin of the ``hbnoise`` analysis *name*: its input spur negated.

        ``None`` when the input spur is not given as tone weights; the SSB noise
        figure is then unavailable.
        """
        spur = _image_spur(params.get("inspur", ""))
        if spur is None:
            logger.warning(
                "hbnoise analysis '%s' gives inspur as %r, not as tone weights (e.g. [1]); "
                "NF_SSB needs the image sideband and is left out",
                name,
                params.get("inspur", ""),
            )
            return None
        return self._analysis_line(
            f"{name}{_IMAGE_SUFFIX}", SimulationType.HBNOISE, list(positional), {**params, "inspur": spur}
        )

    # -------------------------------------------------------------------------
    # Running
    # -------------------------------------------------------------------------

    def run_simulation(self, netlist_path: str, netlist: Netlist) -> SimulationResult | None:
        results_dir = Path(netlist_path).parent
        netlist_base = Path(netlist_path).name
        sim_type = netlist.simulation_type
        analyses = [d for d in netlist.simulation_directives if d.name is not None]

        # A raw file left from an earlier run would be read as this run's result.
        for directive in analyses:
            (results_dir / f"{directive.name}.raw").unlink(missing_ok=True)

        command = [self.vacask_command, "-sp", "-qp", "-n", str(self.threads), netlist_base]
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=results_dir,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise SimulatorError(
                f"Could not run '{' '.join(command)}': {exc}. "
                "Check that VACASK is installed and on PATH (or set the vacask_command "
                "setting to its absolute path); run `cobra doctor` to inspect the environment."
            ) from exc

        if proc.returncode != 0:
            # A convergence failure is a property of the parameters, not of the setup:
            # report it and let the caller penalise this trial.
            logger.warning(
                "VACASK failed for %s (return code %s) in %s",
                sim_type,
                proc.returncode,
                results_dir,
            )
            message = (proc.stderr or proc.stdout).strip()
            if message:
                logger.warning("VACASK output:\n%s", message[-2000:])
            return None

        result = SimulationResult()
        stem = results_dir / netlist_base
        for directive in analyses:
            if directive.simulation_type is not sim_type or (directive.name or "").endswith(_IMAGE_SUFFIX):
                continue  # an image-sideband twin is read along with its hbnoise analysis
            raw_path = results_dir / f"{directive.name}.raw"
            if not raw_path.is_file():
                continue
            try:
                plot = read_raw(raw_path)[0]
                result.output_files.append(str(raw_path))
                self._collect(plot, directive, netlist, stem, result)
            except (ValueError, KeyError) as exc:
                logger.warning("Could not read VACASK result %s: %s", raw_path, exc)

        if not result.dataframes and result.network is None:
            logger.warning(
                "VACASK completed but produced no %s result in %s", sim_type, results_dir
            )
            return None
        return result

    def _collect(
        self,
        plot: RawPlot,
        directive: SimulationDirective,
        netlist: Netlist,
        stem: Path,
        result: SimulationResult,
    ) -> None:
        """Add the result of one analysis to *result*, in the layout the goals read."""
        sim_type = directive.simulation_type
        if sim_type is SimulationType.AC:
            network = self._network(plot, directive, netlist)
            touchstone = Path(f"{stem}.s{network.nports}p")
            # Touchstone 1.0 has a single reference impedance; 2.0 keeps one per port.
            version = "1.0" if np.allclose(network.z0, network.z0[0, 0]) else "2.0"
            # The full name: given "deck.sim", scikit-rf would take .sim as the extension.
            network.write_touchstone(str(touchstone), skrf_comment=False, version=version)
            result.network = network
            result.output_files.append(str(touchstone))
        elif sim_type is SimulationType.HB:
            path = Path(f"{stem}.HB.FD.csv")
            frame = add_port_signals(hb_dataframe(plot), self.netlist_parser.port_resistances(netlist))
            frame.to_csv(path, index=False)
            result.dataframes[str(path)] = frame
            result.output_files.append(str(path))
        elif sim_type is SimulationType.TRAN:
            path = Path(f"{stem}.csv")
            frame = add_port_signals(tran_dataframe(plot), self.netlist_parser.port_resistances(netlist))
            frame.to_csv(path, index=False)
            result.dataframes[str(path)] = frame
            result.output_files.append(str(path))
            # Mirrors the HB result, so goals and the GUI read both alike.
            spectrum_path = Path(f"{stem}.TRAN.FD.csv")
            try:
                spectrum = tran_spectrum.to_frequency_domain(frame)
            except (KeyError, ValueError) as exc:
                logger.warning("Could not compute the spectrum of %s: %s", path, exc)
                return
            spectrum.to_csv(spectrum_path, index=False)
            result.dataframes[str(spectrum_path)] = spectrum
            result.output_files.append(str(spectrum_path))
        elif sim_type in (SimulationType.NOISE, SimulationType.HBNOISE):
            image_path = stem.parent / f"{directive.name}{_IMAGE_SUFFIX}.raw"
            image = read_raw(image_path)[0] if image_path.is_file() else None
            path = Path(f"{stem}.{sim_type.name}.csv")
            frame = self._noise_frame(plot, directive, netlist, image)
            frame.to_csv(path, index=False)
            result.dataframes[str(path)] = frame
            result.output_files.append(str(path))

    def _noise_frame(
        self,
        plot: RawPlot,
        directive: SimulationDirective,
        netlist: Netlist,
        image: RawPlot | None,
    ) -> pd.DataFrame:
        """The noise figure of a ``noise`` or ``hbnoise`` plot, as ``FREQ`` and ``NF*`` columns.

        The source is the input port's series resistor; the resistors of the
        other ports terminate the circuit, so their noise is left out (as in
        Spectre and ADS).  For ``hbnoise`` the image-sideband twin gives the
        image gain, which turns the DSB into the SSB noise figure.
        """
        source = directive.kv_params.get("in", "").strip('"')
        pairs = dict(self.netlist_parser.port_pairs(netlist))
        if source not in pairs:
            raise KeyError(
                f"noise input '{source}' is not a port source of an acsp ports list; the noise "
                "figure is referenced to that port's series resistor"
            )
        variables = plot.variables
        resistor = pairs[source]
        if f"n({resistor})" not in variables:
            raise KeyError(f"no noise from the source resistor '{resistor}'; it must be noisy")

        def psd(name: str) -> np.ndarray:
            return np.real(variables.get(f"n({name})", np.zeros(len(variables["frequency"]))))

        onoise = np.real(variables["onoise"])
        terminations = sum((psd(other) for port, other in pairs.items() if port != source), start=np.zeros_like(onoise))
        factor = noise_factor(onoise, psd(resistor), terminations, self._temperature(netlist))
        gain = np.real(variables["gain"])
        columns: dict[str, np.ndarray] = {"FREQ": np.real(variables["frequency"]), "ONOISE": onoise}
        if directive.simulation_type is SimulationType.NOISE:
            columns |= {"GAIN": gain, "NF": 10 * np.log10(factor)}
            return pd.DataFrame(columns)
        columns |= {"GAIN_SIG": gain, "NF_DSB": 10 * np.log10(factor)}
        if image is not None:
            image_gain = np.real(image.variables["gain"])
            columns |= {
                "GAIN_IMG": image_gain,
                "NF_SSB": 10 * np.log10(factor * (gain + image_gain) / gain),
            }
        return pd.DataFrame(columns)

    @staticmethod
    def _temperature(netlist: Netlist) -> float:
        """The simulation temperature in kelvin: option ``temp`` (°C), else VACASK's 27 °C."""
        text = netlist.options_directives.get("", {}).get("temp")
        if text is None:
            return 27.0 + _KELVIN
        try:
            return vacask_float(text) + _KELVIN
        except ValueError:
            logger.warning("Cannot read options temp=%s; assuming 27 °C for the noise figure", text)
            return 27.0 + _KELVIN

    def _network(self, plot: RawPlot, directive: SimulationDirective, netlist: Netlist) -> rf.Network:
        """The S-parameters of an ``acsp`` plot, referenced to each port's resistor."""
        indices = [
            (int(match.group(1)), int(match.group(2)))
            for name in plot.variables
            if (match := _S_PARAMETER_RE.match(name))
        ]
        nports = max(max(pair) for pair in indices)
        frequencies = np.real(plot.variables["frequency"])
        s = np.zeros((len(frequencies), nports, nports), dtype=complex)
        for i, j in indices:
            s[:, i - 1, j - 1] = plot.variables[f"s({i},{j})"]
        z0 = []
        for _, resistor in self.netlist_parser.port_pairs(netlist):
            text = netlist.get_element(resistor).params.get("r", "") if netlist.has_element(resistor) else ""
            try:
                z0.append(vacask_float(text))
            except ValueError:
                logger.warning(
                    "Port resistor '%s' has no numeric r (%r); assuming 50 Ohm for the "
                    "S-parameter reference", resistor, text,
                )
                z0.append(50.0)
        z0 = (z0 + [50.0] * nports)[:nports]
        return rf.Network(
            frequency=rf.Frequency.from_f(frequencies, unit="Hz"),
            s=s,
            # One row per frequency: a bare list would be read per frequency, not per port.
            z0=np.tile(z0, (len(frequencies), 1)),
            name=directive.name or "",
        )


# ---------------------------------------------------------------------------
# Raw plots → COBRA tables
# ---------------------------------------------------------------------------


def noise_factor(
    onoise: np.ndarray, source: np.ndarray, terminations: np.ndarray | float, temperature: float
) -> np.ndarray:
    """Noise factor F from output noise PSDs simulated at *temperature* kelvin.

    *source* is the part of *onoise* the source resistor contributes and
    *terminations* the part of the other port resistors. The noise the
    circuit adds is referred to the source noise at the IEEE reference
    temperature T0 = 290 K: ``F = 1 + (onoise - source - terminations) * T / (source * T0)``.
    """
    added = onoise - source - terminations
    return 1.0 + added * temperature / (source * T0)


def _image_spur(text: str) -> str | None:
    """*text* (an hbnoise spur as tone weights, e.g. ``[1, 0]``) with every weight negated."""
    items = text.strip().strip("[]").replace(",", " ").split()
    try:
        weights = [int(item) for item in items]
    except ValueError:
        return None
    if not any(weights):
        return None
    return "[" + ", ".join(str(-weight) for weight in weights) + "]"


def _signal(name: str) -> str:
    """COBRA's column name for a raw variable: ``V(node)`` or ``I(instance)``."""
    if name.endswith(_FLOW_SUFFIX):
        return f"I({name[: -len(_FLOW_SUFFIX)]})"
    return f"V({name})"


def hb_dataframe(plot: RawPlot) -> pd.DataFrame:
    """An ``hb`` plot as COBRA's HB table: ``FREQ``, then ``Re(..)``/``Im(..)`` per signal.

    VACASK reports the peak amplitude of each harmonic; COBRA's phasors (like
    Xyce's) hold amplitude/2, so every bin but DC is halved.
    """
    frequencies = np.real(plot.variables["frequency"])
    scale = np.where(frequencies == 0.0, 1.0, 0.5)
    columns: dict[str, np.ndarray] = {"FREQ": frequencies}
    for name, values in plot.variables.items():
        if name == "frequency":
            continue
        phasor = np.asarray(values, dtype=complex) * scale
        signal = _signal(name)
        columns[f"Re({signal})"] = phasor.real
        columns[f"Im({signal})"] = phasor.imag
    return pd.DataFrame(columns)


def add_port_signals(frame: pd.DataFrame, resistances: Mapping[str, float]) -> pd.DataFrame:
    """Make every port a probe point named after its source.

    For a port with source ``S`` and resistor ``R`` it adds ``V(S) = R * I(S)``
    and ``I(VS) = I(S)``, the columns a probe node has. The power COBRA computes
    from them, ``2 |V I|``, is then ``2 R |I|^2``: the power into the port's
    resistor, the one that terminates an output port.
    """
    columns: dict[str, np.ndarray] = {}
    for source, resistance in resistances.items():
        # HB tables hold Re(...)/Im(...) columns, transient tables the signals themselves.
        for wrap in ("Re({})", "Im({})", "{}"):
            current, voltage = wrap.format(f"I({source})"), wrap.format(f"V({source})")
            if current in frame.columns and voltage not in frame.columns:
                columns[voltage] = resistance * frame[current].to_numpy()
                columns[wrap.format(f"I(V{source})")] = frame[current].to_numpy()
    return pd.concat([frame, pd.DataFrame(columns, index=frame.index)], axis=1) if columns else frame


def tran_dataframe(plot: RawPlot) -> pd.DataFrame:
    """A ``tran`` plot as COBRA's transient table: ``TIME``, then one column per signal."""
    columns: dict[str, np.ndarray] = {"TIME": np.real(plot.variables["time"])}
    for name, values in plot.variables.items():
        if name != "time":
            columns[_signal(name)] = np.real(values)
    return pd.DataFrame(columns)

