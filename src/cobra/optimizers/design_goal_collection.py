import numpy as np
import skrf as rf
from typing import Callable, Optional, Union

from cobra.optimizers.design_goal import DesignGoal, DesignGoalChecker, DesignParameter
from cobra.spice_sim.base_simulator import SimulationResult
from cobra.spice_sim.simulation_type import SimulationType

# ---------------------------------------------------------------------------
# Helper: Z-matrix accessor used by several lumped-parameter formulas
# ---------------------------------------------------------------------------


def _mm_z(sim_result: SimulationResult, frequency_range: Optional[str] = None):
    """Return (mm_ntwk, z11, z22, z21, omega) after mixed-mode conversion if ≥ 4 ports."""
    if sim_result.network is None:
        raise ValueError("Simulation result does not contain a network.")
    ntwk = sim_result.network[frequency_range] if frequency_range else sim_result.network
    mm = ntwk.copy()
    if mm.nports >= 4:
        mm.se2gmm(p=2)
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
    if sim_result.network is None or sim_result.network.nports < 2:
        raise ValueError("SRF calculation requires a 2-port network in the simulation result.")
    ntwk: rf.Network = sim_result.network
    _, z11, *_ = _mm_z(sim_result, frequency_range)
    freq_ghz = ntwk.f / 1e9
    srf_idx = np.where(np.diff(np.sign(np.imag(z11))))[0]
    return freq_ghz[srf_idx[0]] if len(srf_idx) > 0 else np.nan

def _power(sim_result: SimulationResult, frequency_range: Optional[str] = None):
    """Isolation in dB for a HB analysis result"""
    for df in sim_result.dataframes.values():
        if "FREQ" in df.columns and "Re(V(OUT))" in df.columns and "Im(V(OUT))" in df.columns:
            df = df.copy()
            df.columns = df.columns.str.strip()

            freq_range = DesignGoal.str_to_frequency_range(frequency_range)

            # Change frequency column to float for comparison from scientific e notation
            df["FREQ"] = df["FREQ"].astype(float)

            df_filtered = df
            if freq_range[0] is not None:
                df_filtered = df_filtered[df_filtered["FREQ"] >= freq_range[0]]
            if freq_range[1] is not None:
                df_filtered = df_filtered[df_filtered["FREQ"] <= freq_range[1]]

            df = df_filtered
            if df.empty:
                raise ValueError(f"No data points found in the specified frequency range: {frequency_range}")
            
            # Standardize to uppercase to match Xyce output convention
            pt_upper = "OUT"
            
            # Construct target column names based on user example: V(Out) and I(VOut)
            v_re_col = f"Re(V({pt_upper}))"
            v_im_col = f"Im(V({pt_upper}))"
            i_re_col = f"Re(I(V{pt_upper}))"
            i_im_col = f"Im(I(V{pt_upper}))"
            
            # Validation check
            missing_cols = [c for c in [v_re_col, v_im_col, i_re_col, i_im_col] if c not in df.columns]
            if missing_cols:
                raise KeyError(
                    f"Could not find required columns for point 'OUT'. Missing: {missing_cols}\n"
                    f"Available columns: {list(df.columns)}"
                )
            
            # Recombine real and imaginary parts into complex phasors
            v_pk = df[v_re_col] + 1j * df[v_im_col]
            i_pk = df[i_re_col] + 1j * df[i_im_col]
            
            # Power formula matching hb_analysis.py:
            # S = 2 * |V_pk * I_pk| due to two-sided phasor convention
            p_w = 2 * np.abs(v_pk * i_pk)
            
            # Convert to dBm with a floor to prevent log10(0) runtime errors
            p_dbm = 10 * np.log10(np.maximum(p_w / 1e-3, 1e-30))

            # Convert to numpy array for consistency
            p_dbm = np.array(p_dbm)
            
            return p_dbm
    return np.nan  # Return NaN if no suitable dataframe is found
    


# ---------------------------------------------------------------------------
# Two-port stability figures (small-signal, AC)
# ---------------------------------------------------------------------------


def _stability_terms(sim_result: SimulationResult, frequency_range: Optional[str] = None):
    """Return (s11, s12, s21, s22, delta) for a 2-port network."""
    if sim_result.network is None or sim_result.network.nports < 2:
        raise ValueError("Stability terms require a 2-port network in the simulation result.")
    ntwk: rf.Network = sim_result.network
    if frequency_range:
        ntwk = ntwk[frequency_range]
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
        return lambda sim_result, frequency_range=None, _i=i, _j=j: sim_result.network[frequency_range].s_db[:, _i - 1, _j - 1]
    else:
        return lambda sim_result, frequency_range=None, _i=i, _j=j: np.abs(sim_result.network[frequency_range].s[:, _i - 1, _j - 1])

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
            The array of values to evaluate. Can be a numpy array or a single float.

        """
        if isinstance(values, float):
            raise TypeError("calculate_array_penalty expects 'values' to be a numpy array, not a float.")
        
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
            return loss_val

        elif min_value is not None:
            denom = np.abs(min_value) + eps
            if np.any(values < min_value):
                violating = values[values < min_value]
                return np.sum(((min_value - violating) / denom) ** 2)
            return -np.sum(((values - min_value) / denom) ** 2)

        elif max_value is not None:
            denom = np.abs(max_value) + eps
            if np.any(values > max_value):
                violating = values[values > max_value]
                return np.sum(((violating - max_value) / denom) ** 2)
            return -np.sum(((max_value - values) / denom) ** 2)

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
    DesignParameter(
        "Power_dBm",
        SimulationType.HB,
        _power,
        calculate_array_penalty,
        "Output power in dBm for a given circuit point.",
        min_ports=1,
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