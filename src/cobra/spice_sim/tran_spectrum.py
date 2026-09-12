"""Spectrum extraction from Xyce transient (time-domain) results.

A ``.PRINT tran`` table (columns ``TIME``, ``V(X)``, ``I(VX)``, …) is turned into
the same frequency-domain layout Xyce writes for Harmonic Balance (``FREQ``,
``Re(V(X))``, ``Im(V(X))``, …), so :mod:`cobra.spice_sim.hb_spectrum`, the
design-goal formulas and the GUI plot all work on a transient result unchanged.

Phasor convention: the one-sided FFT is divided by the sample count, so every
bin holds amplitude/2 exactly like an HB phasor, and the DC bin holds the mean.
No window is applied: the simulator covers a whole number of periods when the
``.TRAN`` window is chosen that way, and a window would only smear the lines.

Xyce prints at the time steps its integrator actually took, which are neither
exactly the ``.TRAN`` step nor perfectly uniform, so the samples are first
interpolated onto a uniform grid spanning the printed window.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

#: The fewest samples a window needs before an FFT says anything useful.
MIN_SAMPLES = 4


def is_time_domain(df: pd.DataFrame) -> bool:
    """Whether *df* is a transient table (has a ``TIME`` column)."""
    return any(str(column).strip().upper() == "TIME" for column in df.columns)


def to_frequency_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Return the one-sided spectrum of every signal column in a transient table.

    The window is ``TIME[-1] - TIME[0]``; the frequency resolution is its
    reciprocal. Each signal becomes a ``Re(<signal>)``/``Im(<signal>)`` pair.
    """
    lookup = {str(column).strip().upper(): column for column in df.columns}
    if "TIME" not in lookup:
        raise KeyError(f"No 'TIME' column in transient result; found: {list(df.columns)}")

    times = pd.to_numeric(df[lookup["TIME"]], errors="coerce").to_numpy(dtype=float)
    order = np.argsort(times, kind="stable")
    times = times[order]
    # A breakpoint can make Xyce print the same instant twice; keep the first.
    keep = np.concatenate(([True], np.diff(times) > 0.0))
    times = times[keep]
    if times.size < MIN_SAMPLES:
        raise ValueError(
            f"Transient result holds only {times.size} distinct time points; "
            f"at least {MIN_SAMPLES} are needed for a spectrum."
        )

    window = times[-1] - times[0]
    # The last printed sample closes the window, so the uniform grid stops one
    # step before it: for a periodic signal that keeps the FFT free of leakage.
    n_samples = times.size - 1
    step = window / n_samples
    grid = times[0] + np.arange(n_samples) * step

    spectrum: dict[str, np.ndarray] = {"FREQ": np.fft.rfftfreq(n_samples, d=step)}
    for column in df.columns:
        name = str(column).strip()
        if name.upper() in ("TIME", "INDEX"):
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)[order][keep]
        phasors = np.fft.rfft(np.interp(grid, times, values)) / n_samples
        spectrum[f"Re({name})"] = phasors.real
        spectrum[f"Im({name})"] = phasors.imag
    return pd.DataFrame(spectrum)


def on_fft_grid(window: float, low: float, high: float, rtol: float = 1e-3) -> bool:
    """Whether a ``.TRAN`` window of *window* seconds puts an FFT bin inside ``[low, high]``.

    The bins sit at multiples of ``1 / window``. A single frequency (``low == high``)
    must coincide with a bin to within *rtol* of the resolution; a band only needs
    to contain one. Returns ``True`` for an unusable window so callers that only
    warn stay quiet instead of guessing.
    """
    if window <= 0.0 or not math.isfinite(window):
        return True
    resolution = 1.0 / window
    if low == high:
        bins = low / resolution
        return abs(bins - round(bins)) <= rtol
    return math.floor(high / resolution + rtol) >= math.ceil(low / resolution - rtol)
