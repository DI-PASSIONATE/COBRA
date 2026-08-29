import numpy as np
import skrf as rf
from typing import Callable, Optional, Union

from cobra.optimizers.design_goal import DesignGoal, DesignGoalChecker, DesignParameter
from cobra.spice_sim import hb_spectrum
from cobra.spice_sim.base_simulator import SimulationResult
from cobra.spice_sim.simulation_type import SimulationType

# ---------------------------------------------------------------------------
# Helper: Z-matrix accessor used by several lumped-parameter formulas
# ---------------------------------------------------------------------------


def _network(
    sim_result: SimulationResult, frequency_range: Optional[str] = None
) -> rf.Network:
    """Return the simulated network, sliced to *frequency_range* when one is given.
    """
    if sim_result.network is None:
        raise ValueError("Simulation result does not contain a network.")
    return sim_result.network[frequency_range] if frequency_range else sim_result.network


def _mm_z(sim_result: SimulationResult, frequency_range: Optional[str] = None):
    """Return (mm_ntwk, z11, z22, z21, omega) after mixed-mode conversion if ≥ 4 ports.

    The number of differential pairs follows the netlist's port count, so an N-port
    network is converted as ``N // 2`` pairs (any leftover odd port stays single-ended).
    """
    mm = _network(sim_result, frequency_range).copy()
    if mm.nports >= 4:
        mm.se2gmm(p=mm.nports // 2)
    return mm, mm.z[:, 0, 0], mm.z[:, 1, 1], mm.z[:, 1, 0], 2 * np.pi * mm.f


# ---------------------------------------------------------------------------
# Predefined lumped-element parameters (AC/Z-matrix based)
# ---------------------------------------------------------------------------


def _lp(sim_result: SimulationResult, frequency_range: Optional[str] = None):
    _, z11, *_, omega = _mm_z(sim_result, frequency_range)
    return np.imag(z11) / omega * 1e9


def _ls(sim_result: SimulationResult, frequency_range: Optional[str] = None):
    _, _, z22, *_, omega = _mm_z(sim_result, frequency_range)
    return np.imag(z22) / omega * 1e9


def _rp(sim_result: SimulationResult, frequency_range: Optional[str] = None):
    _, z11, *_ = _mm_z(sim_result, frequency_range)
    return np.real(z11)


def _rs(sim_result: SimulationResult, frequency_range: Optional[str] = None):
    _, _, z22, *_ = _mm_z(sim_result, frequency_range)
    return np.real(z22)


def _qp(sim_result: SimulationResult, frequency_range: Optional[str] = None):
    _, z11, *_ = _mm_z(sim_result, frequency_range)
    return np.imag(z11) / np.real(z11)


def _qs(sim_result: SimulationResult, frequency_range: Optional[str] = None):
    _, _, z22, *_ = _mm_z(sim_result, frequency_range)
    return np.imag(z22) / np.real(z22)


def _k(sim_result: SimulationResult, frequency_range: Optional[str] = None):
    _, z11, z22, z21, _ = _mm_z(sim_result, frequency_range)
    return np.abs(np.imag(z21) / np.sqrt(np.imag(z11) * np.imag(z22)))


def _srf(sim_result: SimulationResult, frequency_range: Optional[str] = None):
    ntwk = _network(sim_result, frequency_range)
    if ntwk.nports < 2:
        raise ValueError("SRF calculation requires a 2-port network in the simulation result.")
    _, z11, *_ = _mm_z(sim_result, frequency_range)
    freq_ghz = ntwk.f / 1e9
    srf_idx = np.where(np.diff(np.sign(np.imag(z11))))[0]
    return freq_ghz[srf_idx[0]] if len(srf_idx) > 0 else np.nan

def _power(
    sim_result: SimulationResult,
    frequency_range: Optional[str] = None,
    node: str = "OUT",
) -> np.ndarray:
    """Output power in dBm at *node* from a Harmonic Balance result.

    Requires the node voltage ``V(<node>)`` and the current ``I(V<node>)`` of the
    0 V probe source in series with it to be present in the HB output.
    """
    df = hb_spectrum.find_dataframe(sim_result.dataframes, node, "power")
    if df is None:
        raise KeyError(
            f"No HB result containing V({node}) and I(V{node}) was found. "
            f"Add both to the netlist's '.PRINT hb' line."
        )
    _, p_dbm = hb_spectrum.spectrum(
        df, node, "power", DesignGoal.str_to_frequency_range(frequency_range)
    )
    return p_dbm


def make_power_dbm(node: str) -> "DesignParameter":
    """Return a ``Power_dBm[<node>]`` DesignParameter for HB simulations."""

    def _power_formula(
        sim_result: SimulationResult, frequency_range: Optional[str] = None
    ) -> np.ndarray:
        return _power(sim_result, frequency_range, node)

    return DesignParameter(
        name=f"Power_dBm[{node}]",
        simulation_type=SimulationType.HB,
        formula=_power_formula,
        loss=calculate_array_penalty,
        description=f"Output power in dBm at node '{node}' (from V({node}) and I(V{node})).",
        min_ports=1,
    )


def make_gain_db(
    port_name: str, sin_amplitude: float, z0: float = 50.0, node: str = "OUT"
) -> "DesignParameter":
    """Return a ``Gain_dB[<port_name>@<node>]`` DesignParameter for HB simulations.

    Gain is defined as ``Pout[dBm] − Pin[dBm]`` where:

    * ``Pout`` is the HB output power at *node* (see :func:`_power`).
    * ``Pin`` (available power at the input port) is the constant::

          P_avail = A² / (8 · z0)   →   Pin_dBm = 10·log10(P_avail / 1 mW)

      with *A* being the SIN peak amplitude read from the netlist.

    Parameters
    ----------
    port_name:
        Name of the input P-element in the netlist (e.g. ``"P2"``).
    sin_amplitude:
        Peak amplitude of the SIN source on that port (Volts).
    z0:
        Port impedance in Ohms (default 50 Ω).
    node:
        Output node the gain is measured at.
    """
    pin_dbm = hb_spectrum.available_power_dbm(sin_amplitude, z0)

    def _gain_formula(
        sim_result: SimulationResult, frequency_range: Optional[str] = None
    ) -> np.ndarray:
        return _power(sim_result, frequency_range, node) - pin_dbm

    return DesignParameter(
        name=f"Gain_dB[{port_name}@{node}]",
        simulation_type=SimulationType.HB,
        formula=_gain_formula,
        loss=calculate_array_penalty,
        description=(
            f"Transducer gain in dB at node '{node}' using {port_name} as input port "
            f"(A={sin_amplitude:.6g} V, z0={z0:.4g} Ω → P_in={pin_dbm:.2f} dBm)."
        ),
        min_ports=1,
    )


# ---------------------------------------------------------------------------
# Two-port stability figures (small-signal, AC)
# ---------------------------------------------------------------------------


def _stability_terms(sim_result: SimulationResult, frequency_range: Optional[str] = None):
    """Return (s11, s12, s21, s22, delta) for a 2-port network."""
    ntwk = _network(sim_result, frequency_range)
    if ntwk.nports < 2:
        raise ValueError("Stability terms require a 2-port network in the simulation result.")
    s11 = ntwk.s[:, 0, 0]
    s12 = ntwk.s[:, 0, 1]
    s21 = ntwk.s[:, 1, 0]
    s22 = ntwk.s[:, 1, 1]
    delta = s11 * s22 - s12 * s21
    return s11, s12, s21, s22, delta


def _mu(sim_result: SimulationResult, frequency_range: Optional[str] = None):
    s11, s12, s21, s22, delta = _stability_terms(sim_result, frequency_range)
    return (1 - np.abs(s11) ** 2) / (
        np.abs(s22 - delta * np.conj(s11)) + np.abs(s12 * s21)
    )


def _mu_prime(sim_result: SimulationResult, frequency_range: Optional[str] = None):
    s11, s12, s21, s22, delta = _stability_terms(sim_result, frequency_range)
    return (1 - np.abs(s22) ** 2) / (
        np.abs(s11 - delta * np.conj(s22)) + np.abs(s12 * s21)
    )


def _rollett_k(sim_result: SimulationResult, frequency_range: Optional[str] = None):
    s11, s12, s21, s22, delta = _stability_terms(sim_result, frequency_range)
    return (1 - np.abs(s11) ** 2 - np.abs(s22) ** 2 + np.abs(delta) ** 2) / (
        2 * np.abs(s12 * s21)
    )


def _gmax(sim_result: SimulationResult, frequency_range: Optional[str] = None):
    s11, s12, s21, s22, delta = _stability_terms(sim_result, frequency_range)
    K = (1 - np.abs(s11) ** 2 - np.abs(s22) ** 2 + np.abs(delta) ** 2) / (
        2 * np.abs(s12 * s21)
    )
    ratio = np.abs(s21 / s12)
    # Gmax is only defined where K ≥ 1; clamp K² - 1 at 0 to avoid NaN
    return ratio * (K - np.sqrt(np.maximum(K**2 - 1, 0)))

def s_param_formula(i: int, j: int, in_db: bool = True) -> Callable[[SimulationResult, Optional[str]], np.ndarray]:
    """Return a formula function for S{i}{j} or S{i}{j}_dB."""
    if in_db:
        return lambda sim_result, frequency_range=None, _i=i, _j=j: np.asarray(_network(sim_result, frequency_range).s_db[:, _i - 1, _j - 1])
    else:
        return lambda sim_result, frequency_range=None, _i=i, _j=j: np.abs(_network(sim_result, frequency_range).s[:, _i - 1, _j - 1])

def make_s_param_db(i: int, j: int) -> DesignParameter:
    """Return a ``S{i}{j}_dB`` DesignParameter (dB magnitude)."""
    return DesignParameter(
        name=f"S{i}{j}_dB",
        simulation_type=SimulationType.AC,
        formula=s_param_formula(i, j, in_db=True),
        loss=calculate_array_penalty,
        description=f"S{i}{j} magnitude in dB.",
        min_ports=max(i, j),
    )


def make_s_param_linear(i: int, j: int) -> DesignParameter:
    """Return a ``S{i}{j}`` DesignParameter (linear magnitude)."""
    return DesignParameter(
        name=f"S{i}{j}",
        simulation_type=SimulationType.AC,
        formula=s_param_formula(i, j, in_db=False),
        loss=calculate_array_penalty,
        description=f"S{i}{j} linear magnitude.",
        min_ports=max(i, j),
    )

# ----------------------------------------------------------------------------
# Penalty functions for design goals
# ----------------------------------------------------------------------------
def calculate_array_penalty(min_value: Optional[float], max_value: Optional[float], values: Union[np.ndarray, float]) -> float:
        """
        Calculate a penalty for an array of values based on the provided min and max constraints.
        The penalty is calculated as the sum of squared normalized deviations from the constraints.
        If a value is within the range [min_value, max_value], it contributes 0 to the penalty.
        If it is outside this range, it contributes a positive value proportional to the square of its normalized deviation from the nearest bound.
        If it is within the range, it contributes a negative value proportional to the square of its normalized deviation from the nearest bound.

        Parameters
        ----------
        min_value : Optional[float]
            The minimum acceptable value. If None, no lower bound is enforced.
        max_value : Optional[float]
            The maximum acceptable value. If None, no upper bound is enforced.
        values : Union[np.ndarray, float]
            The value(s) to evaluate. Scalars — as returned by single-valued parameters
            such as SRF — are treated as a one-element array.

        """
        values = np.atleast_1d(np.asarray(values, dtype=float))

        eps = 1e-9
        loss_val = 0.0

        if min_value is not None and max_value is not None:
            below_mask = values < min_value
            above_mask = values > max_value
            if np.any(below_mask):
                diff = (min_value - values[below_mask]) / (
                    np.abs(min_value) + eps
                )
                loss_val += np.sum(diff**2)
            if np.any(above_mask):
                diff = (values[above_mask] - max_value) / (
                    np.abs(max_value) + eps
                )
                loss_val += np.sum(diff**2)
            return float(loss_val)

        elif min_value is not None:
            denom = np.abs(min_value) + eps
            if np.any(values < min_value):
                violating = values[values < min_value]
                return float(np.sum(((min_value - violating) / denom) ** 2))
            return float(-np.sum(((values - min_value) / denom) ** 2))

        elif max_value is not None:
            denom = np.abs(max_value) + eps
            if np.any(values > max_value):
                violating = values[values > max_value]
                return float(np.sum(((violating - max_value) / denom) ** 2))
            return float(-np.sum(((max_value - values) / denom) ** 2))

        raise ValueError("At least one of min_value or max_value must be provided.")

# --------------------------------------------------------------------------
# DesignParameter catalogue 
# --------------------------------------------------------------------------
MAX_PORTS: int = 8

_ALL_PARAMETERS: list[DesignParameter] = [
    *(make_s_param_db(i, j) for i in range(1, MAX_PORTS + 1) for j in range(1, MAX_PORTS + 1)),
    *(make_s_param_linear(i, j) for i in range(1, MAX_PORTS + 1) for j in range(1, MAX_PORTS + 1)),
    DesignParameter(
        "Lp",
        SimulationType.AC,
        _lp,
        calculate_array_penalty,
        "Primary (single-port) inductance in nH.",
        min_ports=1,
    ),
    DesignParameter(
        "Rp",
        SimulationType.AC,
        _rp,
        calculate_array_penalty,
        "Primary (single-port) series resistance in Ω.",
        min_ports=1,
    ),
    DesignParameter(
        "Qp",
        SimulationType.AC,
        _qp,
        calculate_array_penalty,
        "Primary quality factor (Im(Z₁₁) / Re(Z₁₁)).",
        min_ports=1,
    ),
    DesignParameter(
        "SRF", 
        SimulationType.AC, 
        _srf, 
        calculate_array_penalty, 
        "Self-resonance frequency in GHz.", 
        min_ports=1
    ),
    DesignParameter(
        "Ls",
        SimulationType.AC,
        _ls,
        calculate_array_penalty,
        "Secondary (two-port) inductance in nH.",
        min_ports=2,
    ),
    DesignParameter(
        "Rs",
        SimulationType.AC,
        _rs,
        calculate_array_penalty,
        "Secondary (two-port) series resistance in Ω.",
        min_ports=2,
    ),
    DesignParameter(
        "Qs",
        SimulationType.AC,
        _qs,
        calculate_array_penalty,
        "Secondary quality factor (Im(Z₂₂) / Re(Z₂₂)).",
        min_ports=2,
    ),
    DesignParameter(
        "k",
        SimulationType.AC,
        _k,
        calculate_array_penalty,
        "Magnetic coupling coefficient between primary and secondary.",
        min_ports=2,
    ),
    DesignParameter(
        "mu",
        SimulationType.AC,
        _mu,
        calculate_array_penalty,
        "Mu stability factor (source). > 1 indicates unconditional small-signal stability.",
        min_ports=2,
    ),
    DesignParameter(
        "mu_prime",
        SimulationType.AC,
        _mu_prime,
        calculate_array_penalty,
        "Mu' stability factor (load). > 1 indicates unconditional small-signal stability.",
        min_ports=2,
    ),
    DesignParameter(
        "K",
        SimulationType.AC,
        _rollett_k,
        calculate_array_penalty,
        "Rollett stability factor. K > 1 together with |Δ| < 1 guarantees unconditional stability.",
        min_ports=2,
    ),
    DesignParameter(
        "Gmax",
        SimulationType.AC,
        _gmax,
        calculate_array_penalty,
        "Maximum available / stable gain |S21/S12| · (K − √(K²−1)), defined where K ≥ 1.",
        min_ports=2,
    ),
]


# ---------------------------------------------------------------------------
# Global catalogue — all known DesignParameters up to MAX_PORTS S-param ports.
# Use get_available_parameters() to obtain the subset valid for a given netlist.
# ---------------------------------------------------------------------------

MAX_PORTS: int = 8

ALL_PARAMETERS: dict[str, DesignParameter] = {p.name: p for p in _ALL_PARAMETERS}

def get_available_parameters(
    num_ports: int,
    simulation_type: SimulationType | None = None,
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
    return ALL_PARAMETERS.get(name)