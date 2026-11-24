"""Background acquisition: one thread owns the spectrometer and hands the GUI the newest spectrum."""
from __future__ import annotations

import threading
import time

import numpy as np


class Acquirer:
    """Reads spectra in a worker thread.

    The GUI sets `params` (a fresh dict each time), toggles `running`, or sets `single` for one
    acquisition, and collects results with `take()`. Every device call happens under `lock`.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.dev = None
        self.params = {"int_us": 100_000, "avg": 1, "edark": False, "nonlin": False}
        self.running = False
        self.single = False
        self.error: str | None = None
        self.delivered = 0
        self._applied_int: int | None = None
        self._latest = None
        self._slot = threading.Lock()
        self._quit = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="acquire", daemon=True)
        self._thread.start()

    # ---------------------------------------------------------- GUI side
    def attach(self, dev) -> None:
        with self.lock:
            self.dev = dev
            self._applied_int = None
            self.error = None

    def detach(self) -> None:
        self.running = False
        self.single = False
        with self.lock:
            dev, self.dev = self.dev, None
            self._applied_int = None
        if dev is not None:
            dev.close()
        with self._slot:
            self._latest = None

    def take(self):
        """Newest (counts, info) since the last call, or None."""
        with self._slot:
            item, self._latest = self._latest, None
        return item

    def stop(self) -> None:
        self._quit.set()
        self._thread.join(timeout=3)
        self.detach()

    # ---------------------------------------------------------- worker
    def _wanted(self) -> bool:
        return self.dev is not None and (self.running or self.single) and not self._quit.is_set()

    def _loop(self) -> None:
        while not self._quit.is_set():
            if not self._wanted():
                time.sleep(0.02)
                continue
            p = self.params
            try:
                with self.lock:
                    dev = self.dev
                    if dev is None:
                        continue
                    if self._applied_int != p["int_us"]:
                        dev.set_integration_us(p["int_us"])
                        self._applied_int = p["int_us"]
                        # the first spectrum after an integration-time change can still carry
                        # the old exposure, so read it and throw it away
                        dev.intensities(p["edark"], p["nonlin"])
                    t_start = time.time()
                    want = max(1, int(p["avg"]))
                    acc = None
                    n = 0
                    for _ in range(want):
                        y = dev.intensities(p["edark"], p["nonlin"])
                        acc = np.array(y, dtype=float) if acc is None else acc + y
                        n += 1
                        if not self._wanted() or self.params is not p:
                            break
            except Exception as exc:  # USB unplugged, timeout, ...
                self.error = f"{type(exc).__name__}: {exc}"
                self.running = False
                self.single = False
                continue
            if n < want:  # paused or settings changed part-way: drop the partial average
                continue
            info = {
                "t_start": t_start,
                "t_end": time.time(),
                "int_us": p["int_us"],
                "avg": n,
                "edark": p["edark"],
                "nonlin": p["nonlin"],
            }
            with self._slot:
                self._latest = (acc / n, info)
            self.delivered += 1
            self.single = False
