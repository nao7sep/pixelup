from __future__ import annotations

import logging
import os
import sys

import pytest
from PySide6.QtWidgets import QApplication

from pixelup.backup_store import close_backup_store
from pixelup.session_log import LOGGER_NAME


@pytest.fixture(autouse=True)
def _reset_pixelup_logging():
    """Restore the shared logger and excepthook after every test.

    PixelUp's session logger is a process-wide singleton; tests call
    configure_session_logging() which mutates its handlers/level and installs a
    sys.excepthook. Without this teardown that state leaks across tests and makes
    the suite order-dependent.
    """
    saved_excepthook = sys.excepthook
    yield
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)
    sys.excepthook = saved_excepthook


@pytest.fixture(autouse=True)
def _reset_backup_store():
    """Close the write-through backup store after every test.

    The store is a process-wide singleton opened lazily against whatever
    PIXELUP_HOME resolves to at first use. Tests redirect PIXELUP_HOME to a
    throwaway root; without this teardown the singleton would stay bound to the
    first test's root (and hold that file handle open) for the whole session. This
    closes it and resets the singleton so the next test's first save re-opens the
    store under that test's own root (data-backup-conventions test migration).
    """
    yield
    close_backup_store()


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """A single offscreen QApplication shared by all GUI tests.

    Widget construction needs a running QApplication. The offscreen platform
    keeps the suite headless so it never opens real windows.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
