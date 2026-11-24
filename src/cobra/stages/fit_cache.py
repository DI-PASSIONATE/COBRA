"""Reuse of surrogate fits: the same network under the same name is fitted once."""

import hashlib
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import skrf as rf

#: Fits kept at once; the one used least recently is dropped first.
MAX_FITS = 128


class FitCache:
    """The files surrogate fits wrote, by the network they were made from.

    Safe for concurrent trials: a trial that needs a fit another one is making
    waits for it. A fit that fails is not kept.
    """

    def __init__(self, max_fits: int = MAX_FITS):
        self.max_fits = max_fits
        #: Fits made and fits reused since the last :meth:`clear`.
        self.fitted = 0
        self.reused = 0
        self._fits: OrderedDict[str, Future[tuple[str, str] | None]] = OrderedDict()
        self._lock = threading.Lock()

    def clear(self) -> None:
        """Forget every fit and reset the counters."""
        with self._lock:
            self._fits.clear()
            self.fitted = self.reused = 0

    def get(self, ntwk: "rf.Network", name: str, fit: Callable[["rf.Network", str], str]) -> str:
        """The file *fit* writes for *ntwk* at the path prefix *name*, from an earlier fit if there is one.

        *fit* takes the network and the prefix and returns the path of the one file
        it wrote, whose content depends only on the network and the prefix's last part.
        """
        key = _fit_key(ntwk, name)
        with self._lock:
            kept = self._fits.get(key)
            if kept is None:
                own: Future[tuple[str, str] | None] = Future()
                self._fits[key] = own
                self.fitted += 1
                while len(self._fits) > self.max_fits:
                    self._fits.popitem(last=False)
            else:
                self._fits.move_to_end(key)
        if kept is None:
            return self._fit(key, own, ntwk, name, fit)

        file = kept.result()
        if file is None:
            return fit(ntwk, name)
        with self._lock:
            self.reused += 1
        suffix, text = file
        path = name + suffix
        Path(path).write_text(text, encoding="utf-8")
        return path

    def _fit(
        self,
        key: str,
        own: "Future[tuple[str, str] | None]",
        ntwk: "rf.Network",
        name: str,
        fit: Callable[["rf.Network", str], str],
    ) -> str:
        """Make the fit of *key* and hand it to *own*, which other trials may be waiting for."""
        try:
            path = fit(ntwk, name)
            file = None
            if path.startswith(name) and Path(path).is_file():
                file = (path[len(name) :], Path(path).read_text(encoding="utf-8"))
        except BaseException as exc:
            self._forget(key, own)
            own.set_exception(exc)
            raise
        if file is None:
            self._forget(key, own)
        own.set_result(file)
        return path

    def _forget(self, key: str, own: "Future[tuple[str, str] | None]") -> None:
        with self._lock:
            if self._fits.get(key) is own:
                del self._fits[key]


def _fit_key(ntwk: "rf.Network", name: str) -> str:
    """A digest of everything a fit of *ntwk* at the path prefix *name* depends on."""
    digest = hashlib.sha256(f"{os.path.basename(name)}|{ntwk.s_def}|{ntwk.s.shape}".encode())
    for values in (ntwk.f, ntwk.s, ntwk.z0):
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()
