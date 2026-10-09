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
    from snp2le.core.ir import CircuitIR

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
    vf.auto_fit()

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
    nw: skrf.Network, name: str, enforce_passivity: bool = False, max_order: int = 12
) -> str:
    """Model *nw* with snp2le and write it as the VACASK subcircuit ``<basename>_subct`` to ``<name>.inc``.

    snp2le's universal mode vector-fits the S-parameters into a lumped-element
    macromodel whose resistors are noiseless (``noisy=0``): they realise poles,
    not losses.  *max_order* bounds the model order of the fit.  The file is
    self-contained, see :func:`vacask_include`.
    """
    # Imported here: snp2le is only needed by the VACASK backend.
    from snp2le.core.engine import convert
    from snp2le.core.netlist import render_vacask
    from snp2le.core.state import ConverterState

    state = ConverterState(mode="universal", enforce_passivity=enforce_passivity, max_order=max_order)
    with _SNP2LE_LOCK, _bounded_pole_relocation(_MAX_POLE_RELOCATIONS):
        result = convert(state, nw)
    if not result.ok:
        raise VectorFitError(f"snp2le could not model {os.path.basename(name)}: {result.error}")
    for message in result.messages:
        logger.debug("snp2le %s: %s", os.path.basename(name), message)
    circuit = cast("CircuitIR", result.ir)
    circuit.name = os.path.basename(name) + "_subct"
    netlist_filename = name + ".inc"
    Path(netlist_filename).write_text(vacask_include(render_vacask(circuit)), encoding="utf-8")
    return netlist_filename


#: snp2le swaps sys.stdout/sys.stderr for the duration of a fit. Two fits of
#: parallel trials interleaving would restore each other's stand-ins and leave the
#: console redirected for good, so the fits run one at a time.
_SNP2LE_LOCK = threading.Lock()

#: Pole relocations one fit may run. scikit-rf's auto_fit has no iteration limit:
#: at the order cap, skimming can remove as many poles as it adds while the error
#: flips between two values, and the loop never ends. Converging fits need about
#: 20 to 120 relocations.
_MAX_POLE_RELOCATIONS = 1000
#: The static method auto_fit calls once per relocation: the only hook inside its loop.
_RELOCATION_HOOK = "_pole_relocation"


class VectorFitError(ValueError):
    """A surrogate's S-parameters could not be fitted into a subcircuit."""


@contextlib.contextmanager
def _bounded_pole_relocation(limit: int):
    """Make a scikit-rf fit that exceeds *limit* pole relocations raise instead of looping forever.

    The count is shared by every fit while this is active, so use it around one fit
    at a time (snp2le fits hold ``_SNP2LE_LOCK``).
    """
    relocate = getattr(VectorFitting, _RELOCATION_HOOK)
    calls = 0

    def bounded(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > limit:
            raise VectorFitError(f"vector fit did not converge in {limit} pole relocations")
        return relocate(*args, **kwargs)

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
