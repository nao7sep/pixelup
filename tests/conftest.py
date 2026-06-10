from __future__ import annotations

import os

import pytest
from PySide6.QtWidgets import QApplication


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
