from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from pixelup.app_config import AppConfig
from pixelup.gui import ImagePreview, MainWindow
from pixelup.runner import JobRunner
from pixelup.session_log import configure_session_logging
from pixelup.ui_common import apply_scrollbar_style

# Window-chrome conformance per the window-chrome-conventions: a thin rounded
# palette-themed scroll bar (Fusion's default is thick and square), and a window
# minimum derived from the panes' content-based minimums rather than a hand-typed
# constant, so no pane can be crushed below its useful size.


@pytest.fixture
def make_window(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    # No real scheduling (no threads / inference), and a clean in-memory config
    # rather than the developer's real ~/.pixelup/config.json.
    monkeypatch.setattr(JobRunner, "schedule", lambda self, max_concurrent_jobs: None)
    monkeypatch.setattr("pixelup.gui.load_app_config", lambda: AppConfig())
    log_file = tmp_path / "logs" / "session.log"
    configure_session_logging(log_file)

    created: list[MainWindow] = []

    def _make() -> MainWindow:
        window = MainWindow(log_file=log_file)
        created.append(window)
        return window

    yield _make

    for window in created:
        window._session_shutdown = True
        window.close()
        window.deleteLater()
    qapp.processEvents()


def test_image_preview_floor_is_deliberate(qapp: QApplication) -> None:
    """The preview floor is the deliberate 320x240, not the old 160x120 stub."""
    preview = ImagePreview()
    assert preview.minimumSize().width() == 320
    assert preview.minimumSize().height() == 240


def test_window_minimum_is_derived_not_the_old_constant(make_window) -> None:
    """The window minimum is derived from the layout, not a hand-typed number.

    It must equal (or exceed) the central widget's size hint and clearly differ
    from the old preview-stub value of 160x120.
    """
    window = make_window()
    hint = window.centralWidget().sizeHint()

    assert window.minimumSize() != (160, 120)
    assert window.minimumWidth() >= hint.width()
    assert window.minimumHeight() >= hint.height()


def test_window_minimum_covers_both_panes_so_neither_is_crushed(make_window) -> None:
    """The minimum width is at least the sum of the two side-by-side panes.

    The image table (left) and the queue table (right) sit in a horizontal
    layout; the window minimum must cover both their minimum widths so widening
    one can never squeeze the other out of view.
    """
    window = make_window()
    panes_min_width = (
        window.image_table.minimumWidth() + window.queue_table.minimumWidth()
    )

    assert window.minimumWidth() >= panes_min_width


def test_apply_scrollbar_style_installs_thin_rounded_qss(qapp: QApplication) -> None:
    """apply_scrollbar_style installs a thin, rounded scroll-bar stylesheet."""
    saved = qapp.styleSheet()
    try:
        qapp.setStyleSheet("")
        apply_scrollbar_style(qapp)
        qss = QApplication.instance().styleSheet()

        assert "QScrollBar" in qss
        assert "border-radius" in qss  # rounded pill handle
        assert "12px" in qss  # slim gutter
        assert "palette(mid)" in qss  # palette-themed, not hard-coded hex
    finally:
        qapp.setStyleSheet(saved)


def test_apply_scrollbar_style_merges_with_existing_stylesheet(
    qapp: QApplication,
) -> None:
    """Existing application QSS is preserved, not clobbered, when merging."""
    saved = qapp.styleSheet()
    try:
        qapp.setStyleSheet("QLabel { color: red; }")
        apply_scrollbar_style(qapp)
        qss = QApplication.instance().styleSheet()

        assert "QLabel { color: red; }" in qss
        assert "QScrollBar" in qss
    finally:
        qapp.setStyleSheet(saved)
