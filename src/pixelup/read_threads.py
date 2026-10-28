"""Background reads on one shared library pool.

Each read reports through its owner's signal, and the owner applies a result only
when it answers the latest request. Quit never waits for a read: one still blocked
on storage when the quit's bound ends goes with the process.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from pixelup.session_log import log

_pool = ThreadPoolExecutor(thread_name_prefix="pixelup-read")


def run_in_background(read: Callable[[], None]) -> None:
    def run() -> None:
        try:
            read()
        except Exception:  # noqa: BLE001 - the pool would otherwise keep it silently.
            log.exception("background.read_failed")

    _pool.submit(run)
