"""A per-rank watchdog that turns a device hang into a loud, attributed failure.

Every rank executes the calls rank 0 broadcasts, in the same order (ModelRunner._exec / serve). If one rank stops
taking part (a host error between calls, a logic bug that skips a call), the others wait inside the Neuron runtime
for its collectives, and nothing raises: measured on trn1.32xlarge at 32 ranks with the runtime's defaults
(tools/probe_mismatch.py --case missing, docs/neuron-notes.md "Upstream harvest (2026-10)"), 31 ranks were still
blocked when the probe gave up after 120 s. This module bounds that wait.

The main thread marks the sections where it blocks on the device: a graph call's launch (which blocks once the
runtime's execution queue is full) and a read-back of a call's output. A daemon thread checks every few seconds; a
section that has lasted longer than the timeout prints, on stderr, this rank's section and its last calls (graph
name and key, with their age), then ends the process with EXIT_HANG. A read-back blocked inside the runtime cannot
be interrupted from Python, so exiting is the only way to fail it. Each rank reports its own calls; comparing the
ranks' lists names the call where they diverged.

A graph's FIRST call is not watched: it may compile (minutes on a device box) or load a NEFF. Neither is rank 0's
broadcast of the next call (a peer may still be compiling the previous one) nor the idle time between calls.

KILN_EXEC_TIMEOUT_S: the timeout in seconds (default 300; 0 turns the watchdog off). The longest healthy section
measured is a few seconds (a 4096-token prefill call is ~0.5 s on trn1.32xlarge, and a read-back waits behind at most
the 63 executions the runtime queues), so the default leaves two orders of magnitude.

Not a licence to turn off the runtime's per-execution barrier (NEURON_RT_DISABLE_EXECUTION_BARRIER=1, which
vllm-neuron 0.24 sets): without it a collective mismatch returns silently wrong numbers, and ranks alternating a
world-collective graph with an attention-group one under host jitter deadlock; see the notes. Kiln leaves it on.
"""

from __future__ import annotations

import collections
import os
import sys
import threading
import time
from contextlib import contextmanager

EXIT_HANG = 86  # the process exit code of a rank the watchdog ended
DEFAULT_TIMEOUT_S = 300.0
RECENT = 16  # calls kept per rank for the report


def timeout_s() -> float:
    v = os.environ.get("KILN_EXEC_TIMEOUT_S")
    return DEFAULT_TIMEOUT_S if v is None or v == "" else float(v)


class ExecWatchdog:
    def __init__(self, rank: int, timeout: float, poll: float | None = None, exit_fn=None):
        self.rank = rank
        self.timeout = float(timeout)
        self.poll = poll if poll is not None else max(0.05, min(5.0, self.timeout / 4))
        self.recent: collections.deque = collections.deque(maxlen=RECENT)
        self._lock = threading.Lock()
        self._section: tuple[str, float] | None = None
        self._depth = 0
        self._exit = exit_fn or os._exit
        self._stop = threading.Event()
        self.fired = False
        self._thread = threading.Thread(target=self._loop, name=f"kiln-exec-watchdog-r{rank}", daemon=True)
        self._thread.start()

    def launched(self, name: str, key) -> None:
        """Record a call this rank is about to launch (rank 0: also the one it is about to broadcast)."""
        self.recent.append((time.monotonic(), name, key))

    @contextmanager
    def blocking(self, what: str):
        """A section in which the main thread may wait on the device or on a peer. Nested sections keep the outer
        one's start (the outer section is what is blocked)."""
        with self._lock:
            if self._depth == 0:
                self._section = (what, time.monotonic())
            self._depth += 1
        try:
            yield
        except BaseException as e:  # an error raised by the runtime inside a watched section: say where we were
            print(f"KILN EXEC WATCHDOG rank {self.rank}: {type(e).__name__} in {what}; {self._calls()}",
                  file=sys.stderr, flush=True)
            raise
        finally:
            with self._lock:
                self._depth -= 1
                if self._depth == 0:
                    self._section = None

    def _calls(self) -> str:
        now = time.monotonic()
        items = list(self.recent)
        if not items:
            return "no call launched yet"
        return "last calls (oldest first): " + "; ".join(f"{n} {k} ({now - t:.1f} s ago)" for t, n, k in items)

    def report(self, what: str, dt: float) -> str:
        return (f"KILN EXEC WATCHDOG rank {self.rank}: {what} blocked for {dt:.0f} s (KILN_EXEC_TIMEOUT_S="
                f"{self.timeout:g}); a rank that stopped taking part in the calls leaves the others waiting in "
                f"their collectives: compare the ranks' call lists. {self._calls()}")

    def close(self) -> None:
        """Stop the checking thread (idempotent)."""
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.poll):
            with self._lock:
                sec = self._section
            if sec is None:
                continue
            dt = time.monotonic() - sec[1]
            if dt > self.timeout:
                self.fired = True
                print(self.report(sec[0], dt), file=sys.stderr, flush=True)
                self._exit(EXIT_HANG)
                return


def make(rank: int, device_type: str) -> ExecWatchdog | None:
    """The watchdog for a rank on a Neuron device, or None (off: KILN_EXEC_TIMEOUT_S=0, or not a Neuron device)."""
    t = timeout_s()
    if device_type != "neuron" or t <= 0:
        return None
    return ExecWatchdog(rank, t)


@contextmanager
def watched(wd: ExecWatchdog | None, what: str):
    if wd is None:
        yield
    else:
        with wd.blocking(what):
            yield
