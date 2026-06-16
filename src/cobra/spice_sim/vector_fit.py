import os
import re
import warnings
import skrf
import numpy as np
from skrf.vectorFitting import VectorFitting

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


def _enforce_passivity(vf: VectorFitting, nw: skrf.Network) -> VectorFitting:
    """Apply the iterative passivity enforcement strategy.

    1. Cheap initial enforcement attempt.
    2. Full-budget enforcement if still not passive.
    3. Iteratively reduce pole count by 2 and refit until passive.

    Returns the (possibly re-fitted) VectorFitting object.
    """
    n_poles = len(vf.poles)
    print(f"  Passive before enforcement: {vf.is_passive()}  (poles={n_poles}, RMS={vf.get_rms_error():.4e})")

    if vf.is_passive():
        return vf

    # Step 1 – cheap init attempt
    recommended = _try_enforce(vf, n_samples=_calc_n_samples(n_poles, init=True))
    print(f"  After init enforcement: Passive={vf.is_passive()}, RMS={vf.get_rms_error():.4e}")

    if not vf.is_passive():
        # Step 2 – use skrf recommendation or full budget
        n_full = _calc_n_samples(n_poles, init=False)
        n_next = min(recommended, n_full) if recommended else n_full
        _try_enforce(vf, n_samples=n_next)
        print(f"  After full enforcement:  Passive={vf.is_passive()}, RMS={vf.get_rms_error():.4e}")

    if not vf.is_passive():
        # Step 3 – reduce poles iteratively
        print(f"  Reducing poles iteratively (start={n_poles}, step=-2)...")
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
            print(f"    Poles={n_poles_iter}, RMS={rms:.4e}, Passive={passive_now}")

            if passive_now:
                vf = vf_iter
                found = True
                break

            n_poles_iter -= 2

        if not found:
            print("  Could not achieve passivity — using best auto-fit result.")

    print(f"  Final: Passive={vf.is_passive()}, poles={len(vf.poles)}, RMS={vf.get_rms_error():.4e}")
    return vf


def vector_fit(nw: skrf.Network, name: str, enforce_passivity: bool = False) -> str:
    vf = VectorFitting(nw)
    vf.auto_fit()

    if enforce_passivity:
        vf = _enforce_passivity(vf, nw)

    # write SPICE netlist
    netlist_filename = name + '.sp'
    subcircuit_name = os.path.basename(name) + '_subct'
    vf.write_spice_subcircuit_s(netlist_filename, fitted_model_name=subcircuit_name)

    return netlist_filename