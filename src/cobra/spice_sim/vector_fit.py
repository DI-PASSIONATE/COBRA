import contextlib
import logging
import os
import re
import threading
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, cast

import skrf
from skrf.vectorFitting import VectorFitting

if TYPE_CHECKING:
    from snp2le.core.dcpin import DCPinHealth
    from snp2le.core.ir import CircuitIR
    from snp2le.core.noise import NoiseHealth

logger = logging.getLogger(__name__)

# TODO: make configurable, find better values, or implement some sort of dynamic strategy
_MAX_MATRIX_OPS = 1_000_000
_INIT_FACTOR = 5


def _calc_n_samples(n_poles: int, init: bool = False) -> int:
    """Compute n_samples from matrix ops budget and current pole count."""
    n = int(_MAX_MATRIX_OPS // (n_poles ** 2 * (_INIT_FACTOR if init else 1)))
    return max(n, 1)


def _try_enforce(vf: VectorFitting, n_samples: int):
    """Run passivity_enforce and return skrf-recommended n_samples if found in warnings, else None."""
    caught = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vf.passivity_enforce(n_samples=n_samples)
    caught.extend(w)

    for warning in caught:
        m = re.search(r"n_samples\s*>\s*(\d+)", str(warning.message))
        if m:
            return int(m.group(1))
    return None


def _pole_count(vf: VectorFitting) -> int:
    """Number of poles in *vf*, or 0 before a fit has been performed."""
    return 0 if vf.poles is None else len(vf.poles)


def _enforce_passivity(vf: VectorFitting, nw: skrf.Network) -> VectorFitting:
    """Apply the iterative passivity enforcement strategy.

    1. Cheap initial enforcement attempt.
    2. Full-budget enforcement if still not passive.
    3. Iteratively reduce pole count by 2 and refit until passive.

    Returns the (possibly re-fitted) VectorFitting object.
    """
    n_poles = _pole_count(vf)
    logger.debug(
        "Vector fit: passive before enforcement=%s (poles=%d, RMS=%.4e)",
        vf.is_passive(),
        n_poles,
        vf.get_rms_error(),
    )

    if vf.is_passive():
        return vf

    # Step 1 – cheap init attempt
    recommended = _try_enforce(vf, n_samples=_calc_n_samples(n_poles, init=True))
    logger.debug(
        "Vector fit: after init enforcement passive=%s, RMS=%.4e",
        vf.is_passive(),
        vf.get_rms_error(),
    )

    if not vf.is_passive():
        # Step 2 – use skrf recommendation or full budget
        n_full = _calc_n_samples(n_poles, init=False)
        n_next = min(recommended, n_full) if recommended else n_full
        _try_enforce(vf, n_samples=n_next)
        logger.debug(
            "Vector fit: after full enforcement passive=%s, RMS=%.4e",
            vf.is_passive(),
            vf.get_rms_error(),
        )

    if not vf.is_passive():
        # Step 3 – reduce poles iteratively
        logger.debug("Vector fit: reducing poles iteratively (start=%d, step=-2)", n_poles)
        n_poles_iter = n_poles - 2
        found = False

        while n_poles_iter >= 2:
            vf_iter = VectorFitting(nw)
            vf_iter.vector_fit(n_poles_real=n_poles_iter // 2, n_poles_cmplx=n_poles_iter // 2)

            recommended = _try_enforce(vf_iter, n_samples=_calc_n_samples(n_poles_iter, init=True))
            if not vf_iter.is_passive():
                n_full = _calc_n_samples(n_poles_iter, init=False)
                n_next = min(recommended, n_full) if recommended else n_full
                _try_enforce(vf_iter, n_samples=n_next)

            rms = vf_iter.get_rms_error()
            passive_now = vf_iter.is_passive()
            logger.debug(
                "Vector fit: poles=%d, RMS=%.4e, passive=%s", n_poles_iter, rms, passive_now
            )

            if passive_now:
                vf = vf_iter
                found = True
                break

            n_poles_iter -= 2

        if not found:
            logger.warning(
                "Vector fit could not achieve passivity; using the best auto-fit result"
            )

    logger.info(
        "Vector fit: passive=%s, poles=%d, RMS=%.4e",
        vf.is_passive(),
        _pole_count(vf),
        vf.get_rms_error(),
    )
    return vf


def vector_fit(nw: skrf.Network, name: str, enforce_passivity: bool = False) -> str:
    vf = VectorFitting(nw)
    try:
        with _bounded_pole_relocation(_MAX_POLE_RELOCATIONS):
            vf.auto_fit()
    except VectorFitError as exc:
        raise VectorFitError(f"could not model {os.path.basename(name)}: {exc}") from exc

    if enforce_passivity:
        vf = _enforce_passivity(vf, nw)

    # write SPICE netlist
    netlist_filename = name + ".sp"
    subcircuit_name = os.path.basename(name) + "_subct"
    vf.write_spice_subcircuit_s(netlist_filename, fitted_model_name=subcircuit_name)
    path = Path(netlist_filename)
    path.write_text(noiseless_resistors(path.read_text(encoding="utf-8")), encoding="utf-8")

    return netlist_filename


_SPICE_RESISTOR_RE = re.compile(r"^(R\S*)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", re.IGNORECASE | re.MULTILINE)


def noiseless_resistors(subcircuit: str) -> str:
    """Write every resistor of a SPICE subcircuit as a noiseless equivalent.

    The resistors of a vector-fit realisation are not losses, yet a SPICE
    resistor always adds thermal noise. ``R n1 n2 r`` becomes a conductance
    controlled by its own voltage, ``G n1 n2 n1 n2 1/r``: the same current and
    the same matrix entries, but a controlled source is noiseless.
    """
    def conductance(match: re.Match[str]) -> str:
        name, node1, node2, value = match.groups()
        return f"G{name} {node1} {node2} {node1} {node2} {1.0 / float(value)!r}"

    return _SPICE_RESISTOR_RE.sub(conductance, subcircuit)


def vector_fit_vacask(
    nw: skrf.Network,
    name: str,
    enforce_passivity: bool = False,
    max_order: int = 12,
    thermal_noise: bool = False,
) -> str:
    """Model *nw* with snp2le and write it as the VACASK subcircuit ``<basename>_subct`` to ``<name>.inc``.

    snp2le's universal mode vector-fits the S-parameters into a lumped-element
    macromodel whose resistors are noiseless (``noisy=0``): they realise poles,
    not losses.  With *thermal_noise* snp2le appends a generator of the
    passive's thermal noise, kT(I - S S^H) at the ports.  It exists only for a
    strictly passive fit, which an order often misses where a lower one does
    not.  A high order can also leave a poor fit: snp2le's passivity enforcement
    then perturbs it, or falls back to a reduced order, and the optimizer
    exploits the error.  A fit without the generator, or with an RMS error above
    :data:`_FIT_RMS_OK`, is therefore retried at lower orders (see
    :data:`_RETRY_ORDERS`), and the usable fit with the lowest RMS error is
    kept.  If no fit is usable, a :class:`VectorFitError` is raised: a noiseless
    surrogate would make the noise figure optimistic.  *max_order* bounds the
    model order of the fit.  The file is self-contained, see :func:`vacask_include`.
    """
    # Imported here: snp2le is only needed by the VACASK backend.
    from snp2le.core.engine import convert
    from snp2le.core.netlist import render_vacask
    from snp2le.core.state import ConverterState

    surrogate = os.path.basename(name)
    best = None
    tried: list[str] = []
    reason = ""
    cap: int | None = max_order
    while cap is not None:
        state = ConverterState(
            mode="universal", enforce_passivity=enforce_passivity, max_order=cap, thermal_noise=thermal_noise
        )
        result = convert(state, nw)
        if not result.ok:
            reason = f"snp2le could not model {surrogate}: {result.error}"
            tried.append(f"{cap}: no fit")
            cap = _next_cap(cap, cap)
            continue
        for message in result.messages:
            logger.debug("snp2le %s: %s", surrogate, message)
        noise = cast("NoiseHealth | None", result.noise)
        if noise is not None and not noise.ok:
            reason = (
                f"no fit of surrogate {surrogate} gets its thermal noise (passivity enforcement: "
                f"{result.passivity_path}): {noise.message}"
            )
            tried.append(f"{cap}: no thermal noise")
        else:
            tried.append(f"{cap}: RMS {result.rms_error:.1e}")
            if best is None or result.rms_error < best.rms_error:
                best = result
            if result.rms_error <= _FIT_RMS_OK:
                break
        cap = _next_cap(cap, result.model_order)
    if best is None:
        raise VectorFitError(f"{reason} (max orders tried: {'; '.join(tried)})")
    if len(tried) > 1:
        logger.info(
            "Surrogate %s: kept the fit with model order %d, RMS %.2e (max orders tried: %s)",
            surrogate, best.model_order, best.rms_error, "; ".join(tried),
        )
    pin = cast("DCPinHealth | None", best.dc_pin)
    if pin is not None and pin.message.startswith("DC pin skipped"):
        # The model keeps the fit's extrapolated DC admittance, which can connect windings.
        logger.warning("Surrogate %s does not meet its 0 Hz point: %s", surrogate, pin.message)
    circuit = cast("CircuitIR", best.ir)
    circuit.name = os.path.basename(name) + "_subct"
    netlist_filename = name + ".inc"
    Path(netlist_filename).write_text(vacask_include(render_vacask(circuit)), encoding="utf-8")
    return netlist_filename


#: Max orders, below the configured one, at which a fit without its thermal noise
#: generator or with an RMS error above :data:`_FIT_RMS_OK` is retried.  Whether a fit is
#: strictly passive scatters across orders: of 24 inductor geometries at orders 20/30/40,
#: 3 to 6 failed per order, 1 at all.  On 40 + 40 random ORCA inductor and transformer
#: geometries, the best of these instead of a single fit at order 40 cut the worst in-band
#: error from 0.080 to 0.027 (inductor) and 0.065 to 0.014 (transformer).
_RETRY_ORDERS = (30, 20, 15, 12, 10)
#: RMS error of a fit good enough to stop retrying at lower orders.
_FIT_RMS_OK = 3e-3


def _next_cap(cap: int, model_order: int) -> int | None:
    """The max order of the retry after a fit at *cap*, or ``None`` when none is left.

    A cap at or above *model_order*, the order the last fit stopped at, would repeat
    that fit, so the cap stays at least 2 below it.
    """
    lower = [order for order in _RETRY_ORDERS if order < cap]
    if not lower:
        return None
    next_cap = min(lower[0], model_order - 2)
    return next_cap if next_cap >= 2 else None


#: Pole relocations one fit of the Xyce backend may run (snp2le bounds its own fits).
#: scikit-rf's auto_fit has no iteration limit:
#: at the order cap, skimming can remove as many poles as it adds while the error
#: flips between two values, and the loop never ends. Converging fits need about
#: 20 to 120 relocations.
_MAX_POLE_RELOCATIONS = 1000
#: The static method auto_fit calls once per relocation: the only hook inside its loop.
_RELOCATION_HOOK = "_pole_relocation"
#: The hook is a class attribute, so two bounded fits in threads would restore each
#: other's stand-ins; they run one at a time.
_RELOCATION_LOCK = threading.Lock()


class VectorFitError(ValueError):
    """A surrogate's S-parameters could not be fitted into a subcircuit."""


@contextlib.contextmanager
def _bounded_pole_relocation(limit: int):
    """Make a scikit-rf fit that exceeds *limit* pole relocations raise instead of looping forever.

    Holds ``_RELOCATION_LOCK`` while active: the hook is shared by every fit in the
    process, so bounded fits in threads run one at a time.
    """
    calls = 0

    def bounded(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > limit:
            raise VectorFitError(f"vector fit did not converge in {limit} pole relocations")
        return relocate(*args, **kwargs)

    with _RELOCATION_LOCK:
        relocate = getattr(VectorFitting, _RELOCATION_HOOK)
        setattr(VectorFitting, _RELOCATION_HOOK, staticmethod(bounded))
        try:
            yield
        finally:
            setattr(VectorFitting, _RELOCATION_HOOK, staticmethod(relocate))


#: Device module of each master an snp2le subcircuit uses; ``None`` for a VACASK builtin.
_SNP2LE_MODULES = {
    "resistor": "resistor.osdi",
    "capacitor": "capacitor.osdi",
    "inductor": "inductor.osdi",
    "vsource": None,
    "vccs": None,
    "vcvs": None,
    "cccs": None,
    "mutual": None,
}
_SUBCKT_RE = re.compile(r"^subckt\s.*$", re.MULTILINE)
_MASTER_RE = re.compile(r"^\s+\S+\s+\([^)]*\)\s+(\w+)", re.MULTILINE)


def vacask_include(subcircuit: str) -> str:
    """Make an snp2le VACASK subcircuit independent of the netlist that includes it.

    snp2le leaves the device models to the testbench and names ground ``GND``.
    The include declares ``GND`` as a ground (VACASK allows several names), loads
    the modules it needs, and binds its masters inside the subcircuit, so the
    netlist's own ``model`` lines (e.g. a PDK resistor) do not apply to it and
    ``noisy=0`` reaches VACASK's resistor.  It must be included at the top level.
    """
    masters = sorted({m for m in _MASTER_RE.findall(subcircuit) if m in _SNP2LE_MODULES})
    loads = sorted({module for master in masters if (module := _SNP2LE_MODULES[master])})
    header = ["ground GND", *(f'load "{module}"' for module in loads), ""]
    models = "".join(f"\n  model {master} {master}" for master in masters)
    body = _SUBCKT_RE.sub(lambda match: match.group(0) + models, subcircuit, count=1)
    return "\n".join(header) + body
