"""Watch the real Records window over about 10,000 records with new ones arriving.

    uv run python tests/records_demo.py [count]

Seeds ``count`` fake records (10,000 by default) into a throwaway data folder,
never the real ``~/.pixelup``, opens the Records window alone over it and logs
three new records a second through PixelUp's own records writer. Close the
window to end; the folder is deleted. ``test_records_volume.py`` drives the same
window and seed headlessly.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from itertools import count
from pathlib import Path

LIVE_INTERVAL_MS = 333
_SEED_START = datetime(2026, 1, 1, tzinfo=UTC)
_LEVELS = ("info", "info", "info", "debug", "warn", "error")


def seed(database: Path, total: int) -> None:
    """``total`` records, one second apart and older than any live record."""
    from pixelup.session_log import _open_records
    from pixelup.timestamps import to_utc_iso_ms

    connection = _open_records(database)
    try:
        connection.execute("BEGIN")
        connection.executemany(
            "INSERT INTO logs (session, time, level, message, job_id, operation_id, fields)"
            " VALUES (?, ?, ?, ?, NULL, NULL, ?)",
            (
                (
                    to_utc_iso_ms(_SEED_START),
                    to_utc_iso_ms(_SEED_START + timedelta(seconds=i)),
                    _LEVELS[i % len(_LEVELS)],
                    f"seed.record_{i}",
                    json.dumps({"n": i}),
                )
                for i in range(total)
            ),
        )
        connection.commit()
    finally:
        connection.close()


def main() -> int:
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    with tempfile.TemporaryDirectory(prefix="pixelup-records-demo-") as home:
        os.environ["PIXELUP_DATA_DIR"] = home
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication

        from pixelup.records_window import RecordsReads, RecordsWindow, stored_signal
        from pixelup.session_log import (
            RECORDS_FILE_NAME,
            configure_session_logging,
            log,
            records_store,
            set_stored_listener,
        )
        from pixelup.window_placement import WindowState

        seed(Path(home) / RECORDS_FILE_NAME, total)
        app = QApplication(sys.argv)
        configure_session_logging()
        set_stored_listener(stored_signal().emit)
        store = records_store()
        assert store is not None
        window = RecordsWindow(RecordsReads(store), WindowState(Path(home) / "window.ini"))
        window.closed.connect(app.quit)
        numbers = count(1)
        live = QTimer()
        live.timeout.connect(lambda: log.info("demo.live_record", n=next(numbers)))
        live.start(LIVE_INTERVAL_MS)
        window.bring_forward()
        app.exec()
        live.stop()
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
