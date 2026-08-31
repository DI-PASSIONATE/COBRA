"""Spectrum extraction from Xyce Harmonic Balance frequency-domain results.

Works directly on the DataFrames stored in :class:`~cobra.spice_sim.base_simulator.SimulationResult`
(columns ``FREQ``, ``Re(V(X))``, ``Im(V(X))``, ``Re(I(VX))``, ``Im(I(VX))``), so both the
design-goal formulas and the GUI plot share one implementation.

Phasor convention: Xyce stores two-sided spectra, i.e. each bin holds amplitude/2.
The apparent power therefore is ``S = Vrms * Irms = 2 * |V_pk * I_pk|``.
"""

from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple
import re

import numpy as np
import pandas as pd

# Axis metadata per quantity, mirroring hb_analysis._QUANTITY_META.
QUANTITY_META: Dict[str, Dict[str, str]] = {
    "power":   {"label": "Power",   "unit": "dBm"},
    "gain":    {"label": "Gain",    "unit": "dB"},
    "voltage": {"label": "Voltage", "unit": "dBV"},
    "current": {"label": "Current", "unit": "dBmA"},
}

_PROBE_RE = re.compile(r"^Re\(([VI])\(([^)]+)\)\)$", re.IGNORECASE)

# SPICE engineering suffixes; MEG must be matched before M.
_SPICE_SUFFIXES: Sequence[Tuple[str, float]] = (
    ("MEG", 1e6), ("MIL", 25.4e-6),
    ("T", 1e12), ("G", 1e9), ("K", 1e3),
    ("M", 1e-3), ("U", 1e-6), ("N", 1e-9), ("P", 1e-12), ("F", 1e-15),
)

_FLOOR = 1e-30


def spice_float(token: str) -> float:
    """Convert a SPICE number such as ``130G`` or ``95E9`` to a float."""
    text = token.strip()
    if not text:
        raise ValueError("Empty numeric token.")
    try:
        return float(text)
    except ValueError:
        pass
    upper = text.upper()
    for suffix, factor in _SPICE_SUFFIXES:
        if upper.endswith(suffix):
            return float(upper[: -len(suffix)]) * factor
        # Trailing unit letters (e.g. "130GHz") are ignored by SPICE.
        idx = upper.find(suffix)
        if idx > 0 and not upper[:idx].endswith("E"):
            return float(upper[:idx]) * factor
    raise ValueError(f"Cannot interpret '{token}' as a SPICE number.")


def parse_fundamentals(text: Optional[str]) -> List[float]:
    """Parse the ``.HB`` frequency tokens (e.g. ``"95E9 10E9"``) into Hz values."""
    if not text:
        return []
    fundamentals = []
    for token in text.replace(",", " ").split():
        try:
            fundamentals.append(spice_float(token))
        except ValueError:
            continue
    return fundamentals


def available_power_dbm(sin_amplitude: float, z0: float = 50.0) -> float:
    """Available power of a port source in dBm: ``P = A² / (8·z0)`` for a SIN peak amplitude."""
    p_avail_w = sin_amplitude ** 2 / (8.0 * z0)
    return 10.0 * np.log10(max(p_avail_w / 1e-3, _FLOOR))


def probe_nodes(df: pd.DataFrame) -> List[str]:
    """Return node names that have both a ``V(X)`` and an ``I(VX)`` column."""
    voltages: Dict[str, str] = {}
    currents: set = set()
    for column in df.columns:
        match = _PROBE_RE.match(str(column).strip())
        if not match:
            continue
        kind, arg = match.group(1).upper(), match.group(2).strip()
        if kind == "V":
            voltages.setdefault(arg.upper(), arg)
        else:
            currents.add(arg.upper())
    return [node for key, node in voltages.items() if f"V{key}" in currents]


def has_probe(df: pd.DataFrame, node: str, quantity: str = "power") -> bool:
    """Whether *df* holds the columns required to evaluate *quantity* at *node*."""
    columns = {str(c).strip().upper() for c in df.columns}
    if "FREQ" not in columns:
        return False
    needed = []
    if quantity in ("power", "gain", "voltage"):
        needed.append(f"RE(V({node.upper()}))")
    if quantity in ("power", "gain", "current"):
        needed.append(f"RE(I(V{node.upper()}))")
    return all(name in columns for name in needed)


def find_dataframe(
    dataframes: Dict[str, pd.DataFrame], node: str, quantity: str = "power"
) -> Optional[pd.DataFrame]:
    """Return the first DataFrame that can supply *quantity* at *node*."""
    return next((df for df in dataframes.values() if has_probe(df, node, quantity)), None)


def spectrum(
    df: pd.DataFrame,
    node: str,
    quantity: str = "power",
    frequency_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
    pin_dbm: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(frequencies_Hz, values)`` of the one-sided HB spectrum at *node*.

    Values are dBm for ``power``, dB for ``gain`` (output power referred to *pin_dbm*),
    dBV for ``voltage`` and dBmA for ``current``.
    """
    if quantity not in QUANTITY_META:
        raise ValueError(f"quantity must be one of {sorted(QUANTITY_META)}, got '{quantity}'")

    lookup = {str(c).strip().upper(): c for c in df.columns}
    if "FREQ" not in lookup:
        raise KeyError(f"No 'FREQ' column in HB result; found: {list(df.columns)}")

    freqs = pd.to_numeric(df[lookup["FREQ"]], errors="coerce").to_numpy(dtype=float)
    # The negative half of the two-sided spectrum is redundant for real signals.
    mask = freqs >= 0
    if frequency_range:
        f_min, f_max = frequency_range
        if f_min is not None and f_max is not None and f_min == f_max and mask.any():
            # Single frequency point: snap to the nearest line, matching skrf network slicing.
            candidates = np.flatnonzero(mask)
            nearest = candidates[int(np.argmin(np.abs(freqs[candidates] - f_min)))]
            mask = np.zeros(freqs.shape, dtype=bool)
            mask[nearest] = True
        else:
            if f_min is not None:
                mask &= freqs >= f_min
            if f_max is not None:
                mask &= freqs <= f_max
    if not mask.any():
        raise ValueError(f"No HB frequency bins left after filtering to {frequency_range}.")

    def phasor(signal: str) -> np.ndarray:
        re_key, im_key = f"RE({signal.upper()})", f"IM({signal.upper()})"
        if re_key not in lookup:
            raise KeyError(
                f"Signal '{signal}' not found in HB result. "
                f"Available probe nodes: {probe_nodes(df)}"
            )
        real = pd.to_numeric(df[lookup[re_key]], errors="coerce").to_numpy(dtype=float)[mask]
        imag = (
            pd.to_numeric(df[lookup[im_key]], errors="coerce").to_numpy(dtype=float)[mask]
            if im_key in lookup
            else np.zeros_like(real)
        )
        return real + 1j * imag

    if quantity in ("power", "gain"):
        p_w = 2 * np.abs(phasor(f"V({node})") * phasor(f"I(V{node})"))
        values = 10 * np.log10(np.maximum(p_w / 1e-3, _FLOOR))
        if quantity == "gain":
            values = values - pin_dbm
    elif quantity == "voltage":
        values = 20 * np.log10(np.maximum(np.abs(phasor(f"V({node})")), _FLOOR))
    else:
        values = 20 * np.log10(np.maximum(np.abs(phasor(f"I(V{node})")) / 1e-3, _FLOOR))

    return freqs[mask], values


def classify_bins(
    freqs: np.ndarray,
    fundamentals: Sequence[float],
    max_order: int = 10,
    rtol: float = 1e-3,
) -> List[str]:
    """Label every bin as ``DC``, a harmonic (``H2``) or a mixing product (``2f1-f2``).

    Each bin is matched against ``sum(m_i * f_i)`` over the fundamentals, choosing
    the combination with the smallest total order. Unmatched bins get an empty label.
    """
    tones = [f for f in fundamentals if f > 0]
    labels = ["" for _ in freqs]
    if not tones:
        return labels

    tolerance = max(tones) * rtol
    orders = range(-max_order, max_order + 1)
    # Sort candidates by total order so the simplest description wins.
    combos = sorted(product(orders, repeat=len(tones)), key=lambda c: sum(abs(m) for m in c))

    for i, freq in enumerate(freqs):
        if abs(freq) <= tolerance:
            labels[i] = "DC"
            continue
        for combo in combos:
            if not any(combo):
                continue
            if abs(sum(m * f for m, f in zip(combo, tones)) - freq) <= tolerance:
                labels[i] = _format_combo(combo, len(tones))
                break
    return labels


def is_fundamental(label: str) -> bool:
    """Whether *label* denotes a fundamental tone (``H1``, ``f1``, ``f2``, ...)."""
    return label in ("H1",) or bool(re.fullmatch(r"f\d+", label))


def _format_combo(combo: Sequence[int], n_tones: int) -> str:
    if n_tones == 1:
        return f"H{abs(combo[0])}"
    parts = []
    for i, m in enumerate(combo, start=1):
        if m == 0:
            continue
        sign = "-" if m < 0 else ("+" if parts else "")
        magnitude = "" if abs(m) == 1 else str(abs(m))
        parts.append(f"{sign}{magnitude}f{i}")
    return "".join(parts)
