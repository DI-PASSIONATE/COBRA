from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union
import re

import numpy as np
import skrf as rf

from cobra.spice_sim.simulation_type import SimulationType

# ---------------------------------------------------------------------------
# DesignParameter — an object-oriented descriptor for a single measurable goal
# ---------------------------------------------------------------------------


@dataclass
class DesignParameter:
    """
    A named, self-describing design parameter.

    Attributes
    ----------
    name:
        Unique identifier used in the GUI, context dict, and log output
        (e.g. ``"S21_dB"``, ``"Lp"``).
    simulation_type:
        Which Xyce analysis type must be run to produce the result.
    formula:
        Callable ``(ntwk: rf.Network) -> np.ndarray | float``.
        Receives the (possibly frequency-sliced) skrf Network and returns
        the parameter values over the sweep.
    description:
        Human-readable explanation shown as a tooltip in the GUI.
    """

    name: str
    simulation_type: SimulationType
    formula: Callable[[rf.Network], np.ndarray]
    description: str = ""
    min_ports: int = 1
    """Minimum number of netlist ports required to evaluate this parameter."""

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, DesignParameter):
            return self.name == other.name
        return NotImplemented


# ---------------------------------------------------------------------------
# Helper: Z-matrix accessor used by several lumped-parameter formulas
# ---------------------------------------------------------------------------


def _mm_z(ntwk: rf.Network):
    """Return (mm_ntwk, z11, z22, z21, omega) after mixed-mode conversion if ≥ 4 ports."""
    mm = ntwk.copy()
    if mm.nports >= 4:
        mm.se2gmm(p=2)
    return mm, mm.z[:, 0, 0], mm.z[:, 1, 1], mm.z[:, 1, 0], 2 * np.pi * mm.f


# ---------------------------------------------------------------------------
# Predefined lumped-element parameters (AC/Z-matrix based)
# ---------------------------------------------------------------------------


def _lp(ntwk):
    _, z11, *_, omega = _mm_z(ntwk)
    return np.imag(z11) / omega * 1e9


def _ls(ntwk):
    _, _, z22, *_, omega = _mm_z(ntwk)
    return np.imag(z22) / omega * 1e9


def _rp(ntwk):
    _, z11, *_ = _mm_z(ntwk)
    return np.real(z11)


def _rs(ntwk):
    _, _, z22, *_ = _mm_z(ntwk)
    return np.real(z22)


def _qp(ntwk):
    _, z11, *_ = _mm_z(ntwk)
    return np.imag(z11) / np.real(z11)


def _qs(ntwk):
    _, _, z22, *_ = _mm_z(ntwk)
    return np.imag(z22) / np.real(z22)


def _k(ntwk):
    _, z11, z22, z21, _ = _mm_z(ntwk)
    return np.abs(np.imag(z21) / np.sqrt(np.imag(z11) * np.imag(z22)))


def _srf(ntwk):
    _, z11, *_ = _mm_z(ntwk)
    freq_ghz = ntwk.f / 1e9
    srf_idx = np.where(np.diff(np.sign(np.imag(z11))))[0]
    return freq_ghz[srf_idx[0]] if len(srf_idx) > 0 else None


# ---------------------------------------------------------------------------
# Two-port stability figures (small-signal, AC)
# ---------------------------------------------------------------------------


def _stability_terms(ntwk: rf.Network):
    """Return (s11, s12, s21, s22, delta) for a 2-port network."""
    s11 = ntwk.s[:, 0, 0]
    s12 = ntwk.s[:, 0, 1]
    s21 = ntwk.s[:, 1, 0]
    s22 = ntwk.s[:, 1, 1]
    delta = s11 * s22 - s12 * s21
    return s11, s12, s21, s22, delta


def _mu(ntwk):
    s11, s12, s21, s22, delta = _stability_terms(ntwk)
    return (1 - np.abs(s11) ** 2) / (
        np.abs(s22 - delta * np.conj(s11)) + np.abs(s12 * s21)
    )


def _mu_prime(ntwk):
    s11, s12, s21, s22, delta = _stability_terms(ntwk)
    return (1 - np.abs(s22) ** 2) / (
        np.abs(s11 - delta * np.conj(s22)) + np.abs(s12 * s21)
    )


def _rollett_k(ntwk):
    s11, s12, s21, s22, delta = _stability_terms(ntwk)
    return (1 - np.abs(s11) ** 2 - np.abs(s22) ** 2 + np.abs(delta) ** 2) / (
        2 * np.abs(s12 * s21)
    )


def _gmax(ntwk):
    s11, s12, s21, s22, delta = _stability_terms(ntwk)
    K = (1 - np.abs(s11) ** 2 - np.abs(s22) ** 2 + np.abs(delta) ** 2) / (
        2 * np.abs(s12 * s21)
    )
    ratio = np.abs(s21 / s12)
    # Gmax is only defined where K ≥ 1; clamp K² - 1 at 0 to avoid NaN
    return ratio * (K - np.sqrt(np.maximum(K**2 - 1, 0)))


# ---------------------------------------------------------------------------
# Factory for dynamic S-parameter DesignParameters
# ---------------------------------------------------------------------------


def make_s_param_db(i: int, j: int) -> DesignParameter:
    """Return a ``S{i}{j}_dB`` DesignParameter (dB magnitude)."""
    return DesignParameter(
        name=f"S{i}{j}_dB",
        simulation_type=SimulationType.AC,
        formula=lambda ntwk, _i=i, _j=j: ntwk.s_db[:, _i - 1, _j - 1],
        description=f"S{i}{j} magnitude in dB.",
        min_ports=max(i, j),
    )


def make_s_param_linear(i: int, j: int) -> DesignParameter:
    """Return a ``S{i}{j}`` DesignParameter (linear magnitude)."""
    return DesignParameter(
        name=f"S{i}{j}",
        simulation_type=SimulationType.AC,
        formula=lambda ntwk, _i=i, _j=j: np.abs(ntwk.s[:, _i - 1, _j - 1]),
        description=f"S{i}{j} linear magnitude.",
        min_ports=max(i, j),
    )


# --------------------------------------------------------------------------
# DesignParameter catalogue 
# --------------------------------------------------------------------------

# All single-port lumped parameters
_SINGLE_PORT_LUMPED: list[DesignParameter] = [
    DesignParameter(
        "Lp",
        SimulationType.AC,
        _lp,
        "Primary (single-port) inductance in nH.",
        min_ports=1,
    ),
    DesignParameter(
        "Rp",
        SimulationType.AC,
        _rp,
        "Primary (single-port) series resistance in Ω.",
        min_ports=1,
    ),
    DesignParameter(
        "Qp",
        SimulationType.AC,
        _qp,
        "Primary quality factor (Im(Z₁₁) / Re(Z₁₁)).",
        min_ports=1,
    ),
    DesignParameter(
        "SRF", SimulationType.AC, _srf, "Self-resonance frequency in GHz.", min_ports=1
    ),
]

# Two-port lumped parameters (secondary winding / coupling)
_TWO_PORT_LUMPED: list[DesignParameter] = [
    DesignParameter(
        "Ls",
        SimulationType.AC,
        _ls,
        "Secondary (two-port) inductance in nH.",
        min_ports=2,
    ),
    DesignParameter(
        "Rs",
        SimulationType.AC,
        _rs,
        "Secondary (two-port) series resistance in Ω.",
        min_ports=2,
    ),
    DesignParameter(
        "Qs",
        SimulationType.AC,
        _qs,
        "Secondary quality factor (Im(Z₂₂) / Re(Z₂₂)).",
        min_ports=2,
    ),
    DesignParameter(
        "k",
        SimulationType.AC,
        _k,
        "Magnetic coupling coefficient between primary and secondary.",
        min_ports=2,
    ),
]


_STABILITY: list[DesignParameter] = [
    DesignParameter(
        "mu",
        SimulationType.AC,
        _mu,
        "Mu stability factor (source). > 1 indicates unconditional small-signal stability.",
        min_ports=2,
    ),
    DesignParameter(
        "mu_prime",
        SimulationType.AC,
        _mu_prime,
        "Mu' stability factor (load). > 1 indicates unconditional small-signal stability.",
        min_ports=2,
    ),
    DesignParameter(
        "K",
        SimulationType.AC,
        _rollett_k,
        "Rollett stability factor. K > 1 together with |Δ| < 1 guarantees unconditional stability.",
        min_ports=2,
    ),
    DesignParameter(
        "Gmax",
        SimulationType.AC,
        _gmax,
        "Maximum available / stable gain |S21/S12| · (K − √(K²−1)), defined where K ≥ 1.",
        min_ports=2,
    ),
]


# ---------------------------------------------------------------------------
# Global catalogue — all known DesignParameters up to MAX_PORTS S-param ports.
# Use get_available_parameters() to obtain the subset valid for a given netlist.
# ---------------------------------------------------------------------------

MAX_PORTS: int = 8

_ALL_PARAMETERS: list[DesignParameter] = (
    [
        make_s_param_db(i, j)
        for i in range(1, MAX_PORTS + 1)
        for j in range(1, MAX_PORTS + 1)
    ]
    + [
        make_s_param_linear(i, j)
        for i in range(1, MAX_PORTS + 1)
        for j in range(1, MAX_PORTS + 1)
    ]
    + _SINGLE_PORT_LUMPED
    + _TWO_PORT_LUMPED
    + _STABILITY
)


def get_available_parameters(
    num_ports: int,
    simulation_type: "SimulationType | None" = None,
) -> list[DesignParameter]:
    """
    Return all DesignParameter instances whose constraints are satisfied.

    Parameters
    ----------
    num_ports:
        Number of ports in the netlist.  Only parameters whose
        ``min_ports <= num_ports`` are included.
    simulation_type:
        When given, further restricts results to parameters that require
        exactly this simulation type.  ``None`` means no restriction.
    """
    if num_ports < 1:
        return []
    return [
        p
        for p in _ALL_PARAMETERS
        if p.min_ports <= num_ports
        and (simulation_type is None or p.simulation_type is simulation_type)
    ]


def find_parameter(name: str) -> Optional[DesignParameter]:
    """Resolve a parameter name to its DesignParameter descriptor from the global catalogue."""
    for p in _ALL_PARAMETERS:
        if p.name == name:
            return p
    return None


# ---------------------------------------------------------------------------
# DesignGoal — a constraint on a DesignParameter over a frequency range
# ---------------------------------------------------------------------------


class DesignGoal:
    """
    A single optimisation goal: ``parameter`` must satisfy ``[min_value, max_value]``
    within the given ``frequency_range``.
    """

    def __init__(
        self,
        parameter: DesignParameter,
        frequency_range: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        weight: float = 1.0,
    ):
        if not isinstance(parameter, DesignParameter):
            raise TypeError(
                f"'parameter' must be a DesignParameter instance, got {type(parameter).__name__}. "
                "Use find_parameter() or get_available_parameters() to look up parameters."
            )
        self.parameter = parameter
        self.frequency_range = frequency_range
        self.min_value = min_value
        self.max_value = max_value
        self.weight = weight
        self._eps = 1e-9

    @property
    def parameter_name(self) -> str:
        return self.parameter.name

    @property
    def required_simulation_type(self) -> SimulationType:
        return self.parameter.simulation_type

    def penalty(self, values: np.ndarray) -> float:
        """Calculate the penalty for the given measured values."""
        loss_val = 0.0

        if self.min_value is not None and self.max_value is not None:
            below_mask = values < self.min_value
            above_mask = values > self.max_value
            if np.any(below_mask):
                diff = (self.min_value - values[below_mask]) / (
                    np.abs(self.min_value) + self._eps
                )
                loss_val += np.sum(diff**2)
            if np.any(above_mask):
                diff = (values[above_mask] - self.max_value) / (
                    np.abs(self.max_value) + self._eps
                )
                loss_val += np.sum(diff**2)
            return loss_val

        elif self.min_value is not None:
            denom = np.abs(self.min_value) + self._eps
            if np.any(values < self.min_value):
                violating = values[values < self.min_value]
                return np.sum(((self.min_value - violating) / denom) ** 2)
            return -np.sum(((values - self.min_value) / denom) ** 2)

        elif self.max_value is not None:
            denom = np.abs(self.max_value) + self._eps
            if np.any(values > self.max_value):
                violating = values[values > self.max_value]
                return np.sum(((violating - self.max_value) / denom) ** 2)
            return -np.sum(((self.max_value - values) / denom) ** 2)

        raise ValueError("At least one of min_value or max_value must be provided.")


# ---------------------------------------------------------------------------
# DesignGoalChecker — evaluates all goals against simulation results
# ---------------------------------------------------------------------------


class DesignGoalChecker:
    """Evaluates a list of DesignGoals against one or more simulation results."""

    def __init__(self, design_goals: list[DesignGoal]):
        self.design_goals = design_goals

    def check_goals(self, context: dict) -> dict:
        """
        Evaluate all goals and update *context* with results.

        Reads ``context["simulated_networks"]`` (``dict[SimulationType, rf.Network]``).
        """
        networks: dict = context.get("simulated_networks") or {}

        params, penalties = self.loss(networks)
        context["goal_achieved"] = all(p <= 0.0 for p in penalties)
        context["electrical_parameters"] = params
        context["penalties"] = penalties
        return context

    def loss(self, networks: dict) -> tuple[dict, list[float]]:
        """
        Compute penalties for each goal using the appropriate network.

        Parameters
        ----------
        networks:
            ``{SimulationType: rf.Network}`` — one entry per required analysis type.
        """
        penalties = []
        design_state = {}

        for goal in self.design_goals:
            sim_type = goal.required_simulation_type
            ntwk = networks.get(sim_type)
            if ntwk is None:
                raise ValueError(
                    f"No simulation result for '{sim_type.value}', "
                    f"required by goal '{goal.parameter_name}'."
                )

            # Slice to frequency range if specified
            if goal.frequency_range is not None:
                try:
                    ntwk_sliced = ntwk[goal.frequency_range]
                except ValueError:
                    raise ValueError(
                        f"Invalid frequency range '{goal.frequency_range}'. "
                        "Expected format e.g. '10-20ghz'."
                    )
            else:
                ntwk_sliced = ntwk

            values = goal.parameter.formula(ntwk_sliced)
            if values is None:
                raise ValueError(
                    f"Formula returned None for parameter '{goal.parameter_name}'."
                )

            design_state[goal.parameter_name] = values
            penalties.append(goal.penalty(values) * goal.weight)

        return design_state, penalties

    def __str__(self):
        parts = [
            f"{g.parameter_name}: [{g.min_value}, {g.max_value}]"
            f" in {g.frequency_range or 'full range'} (w={g.weight})"
            for g in self.design_goals
        ]
        return "Design Goals: " + " | ".join(parts)
