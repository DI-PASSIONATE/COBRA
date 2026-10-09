"""Xyce netlist dialect.

Adds to the SPICE base what only Xyce has: ``P`` port elements with their
AC/SIN sources, ``Y`` devices, the ``.HB`` analysis, the ``.LIN`` S-parameter
post-processor, the analysis metadata COBRA uses to read and write directive
arguments, and the rewrite of Qucs-S ``TSTONEFILE`` blocks into plain
subcircuit calls.
"""

from __future__ import annotations

import contextlib
from typing import ClassVar

from cobra.spice_sim.netlist_parsers.spice_netlist_parser import (
    SpiceNetlistParser,
    _Layout,
    _unquote,
)
from cobra.spice_sim.netlist_parsers.statement import Statement, StatementKind, Token
from cobra.spice_sim.simulation_type import SimulationType, SimulationTypeMetadata

# ---------------------------------------------------------------------------
# Xyce-specific analysis metadata
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
            "start_time": "Time at which output begins (default 0). Everything before it is the\n"
                          "start-up transient and is left out of the spectrum, so raise it until\n"
                          "the circuit has settled. stop_time - start_time sets the FFT resolution\n"
                          "and should be a whole number of signal periods.",
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
    SimulationType.NOISE: SimulationTypeMetadata(
        positional_param_names=["out", "in", "sweep_type", "points", "start_freq", "stop_freq"],
        positional_param_descriptions={
            "out":        "Output node, e.g. Out (written as V(Out)), or V(a,b) for a node pair.",
            "in":         "Input port (P element) the noise figure is referred to, e.g. P1.",
            "sweep_type": "Frequency sweep spacing: LIN, DEC or OCT.",
            "points":     "Number of frequency points (LIN) or points per decade/octave.",
            "start_freq": "Start frequency (e.g. 120G).",
            "stop_freq":  "Stop frequency (e.g. 140G).",
        },
        positional_param_defaults={"sweep_type": "LIN", "points": "21", "start_freq": "1G", "stop_freq": "10G"},
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


class XyceNetlistParser(SpiceNetlistParser):
    """
    Xyce-flavoured SPICE netlist parser.

    Responsibilities beyond the SPICE base:
    - Rewrites Qucs-S ``TSTONEFILE`` transformer blocks (``YLIN`` instances)
      into ordinary ``X`` subcircuit calls so the rest of COBRA can treat them
      as standard surrogate components.
    """

    analysis_keywords: ClassVar[frozenset[str]] = frozenset({".AC", ".DC", ".TRAN", ".HB", ".NOISE"})
    # .LIN post-processes .AC into Touchstone output; it is not an analysis of its own.
    modifier_keywords: ClassVar[frozenset[str]] = frozenset({".LIN"})
    model_types: ClassVar[frozenset[str]] = frozenset({"D", "M", "Q", "X", "Y"})

    def analysis_metadata(self, sim_type: SimulationType) -> SimulationTypeMetadata:
        return _XYCE_METADATA.get(sim_type, SimulationTypeMetadata())

    # -------------------------------------------------------------------------
    # Xyce devices
    # -------------------------------------------------------------------------

    def _layout(self, etype: str, tokens: tuple[Token, ...]) -> _Layout | None:
        count = len(tokens)
        if etype == "P":
            # <name> <n1> <n2> [params...] [AC <mag>] [SIN <offset> <amplitude> <freq> ...]
            return _Layout(slice(1, 3), params=3) if count >= 3 else None
        if etype == "Y":
            # Qucs-S format: <name> <subtype> [port_nodes...] <model>
            return _Layout(slice(2, count - 1), model=count - 1, params=count, subtype=1) if count >= 4 else None
        return super()._layout(etype, tokens)

    def _port(self, etype: str, params: dict[str, str]) -> int | None:
        if etype != "P":
            return None
        number = next((value for key, value in params.items() if key.lower() == "port"), "0")
        try:
            return int(number)
        except ValueError:
            return 0  # still a port, just without a usable number

    def port_source(self, statement: Statement) -> dict[str, float]:
        """Extract AC amplitude, SIN amplitude/frequency and z0 from a P-element.

        Handles lines such as::

            P2 _net28 0 port=1 z0=100 AC 0.089442719 SIN 0 0.089442719 130G

        as well as the bracketed forms ``SIN(0 0.089 130G)`` and ``SIN (...)``.
        Only populated when at least one of SIN or AC amplitude is found.
        """
        # Imported here: hb_spectrum pulls in numpy and pandas.
        from cobra.spice_sim.hb_spectrum import spice_float

        positional, params = self._split_params(self.tokens(statement)[3:])
        result: dict[str, float] = {}
        for key, value in params:
            if key.text.lower() == "z0":
                with contextlib.suppress(ValueError):
                    result["z0"] = float(value.text)

        words = [token.text for token in self._flatten(positional)]
        index = 0
        while index < len(words):
            upper = words[index].upper()
            if upper == "AC" and index + 1 < len(words):
                with contextlib.suppress(ValueError):
                    result["ac_amplitude"] = float(words[index + 1])
                index += 2
                continue
            if upper == "SIN" and index + 2 < len(words):
                # SIN <offset> <amplitude> [freq] [td] [theta]
                with contextlib.suppress(ValueError):
                    result["sin_amplitude"] = float(words[index + 2])
                if index + 3 < len(words):
                    with contextlib.suppress(ValueError):
                        result["sin_frequency"] = spice_float(words[index + 3])
                index += 3
                continue
            index += 1

        if "sin_amplitude" not in result and "ac_amplitude" not in result:
            return {}
        return result

    # -------------------------------------------------------------------------
    # Qucs-S TSTONEFILE → X-subcircuit normalisation
    # -------------------------------------------------------------------------

    def normalize(self, statements: list[Statement]) -> list[Statement]:
        """
        Convert Qucs-S ``YLIN``/``TSTONEFILE`` device blocks into
        Xyce-compatible subcircuit instances (``X...``).

        1. Collect every ``.MODEL <name> LIN TSTONEFILE=...`` entry.
        2. Rewrite the matching ``Y`` instance into an ``X`` call, keeping the
           Touchstone path as an annotation, and replace the model line with a
           ``.INCLUDE`` for the vector-fitted ``.sp`` file.
        """
        # folded model name → (statement index, Touchstone path)
        models: dict[str, tuple[int, str]] = {}
        for index, statement in enumerate(statements):
            if statement.keyword != ".MODEL":
                continue
            flat = self._flatten(self.tokens(statement))
            if len(flat) < 3 or flat[2].text.upper() != "LIN":
                continue
            touchstone = next(
                (v.text for k, v in self._split_params(flat[3:])[1] if k.text.upper() == "TSTONEFILE"),
                None,
            )
            if touchstone:
                models[self.fold(flat[1].text)] = (index, _unquote(touchstone))
        if not models:
            return statements

        result = list(statements)
        converted: dict[str, str] = {}  # model name → X instance name
        for index, statement in enumerate(statements):
            if statement.kind is not StatementKind.ELEMENT or not statement.keyword.startswith("Y"):
                continue
            tokens = self.tokens(statement)
            model = self.fold(tokens[-1].text) if len(tokens) >= 4 else ""
            if model not in models:
                continue
            model_index, touchstone = models[model]
            x_name = converted.get(model)
            if x_name is None:
                # Prefix the original instance name with "X" to produce a
                # standard subcircuit call that Xyce understands.
                x_name = converted[model] = f"X{tokens[1].text}"
                # The fitted subcircuit replaces the placeholder .MODEL.
                result[model_index] = self._rewrite(statements[model_index], f'.INCLUDE "{x_name}.sp"')
            # Qucs-S emits each port as <signal_node> 0; drop the literal zeros
            # because the fitted subcircuit only expects the signal nodes.
            ports = [token.text for token in tokens[2:-1] if token.text != "0"]
            result[index] = self._rewrite(
                statement,
                " ".join([x_name, *ports, f"{x_name}_subct"]),
                annotations={"TSTONEFILE": touchstone},
            )
        return result
