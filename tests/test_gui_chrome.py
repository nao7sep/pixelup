from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QStyle, QStyleFactory

from pixelup.app_config import AppConfig, ConfigLoadResult
from pixelup.fonts import DEFAULT_UI_FONT_SIZE
from pixelup.gui import ImagePreview, MainWindow
from pixelup.runner import JobRunner
from pixelup.session_log import configure_session_logging
from pixelup.ui_common import title_label

# Window-chrome conformance per the window-chrome-conventions: a window minimum
# derived from the panes' content-based minimums rather than a hand-typed constant,
# so no pane can be crushed below its useful size. Scrollbars remain toolkit-owned.


@pytest.fixture
def make_window(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    # No real scheduling (no threads / inference), and a clean in-memory config
    # rather than the developer's real ~/.pixelup/config.json. PIXELUP_HOME is
    # redirected as well as the load stubbed, because the window also writes
    # config.json (first-run materialize, and the Parameters panel's own save).
    monkeypatch.setenv("PIXELUP_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(JobRunner, "schedule", lambda self, max_concurrent_jobs: None)
    monkeypatch.setattr("pixelup.gui.load_app_config_result", lambda: ConfigLoadResult(AppConfig()))
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


def test_content_minimum_covers_both_panes_so_neither_is_crushed(make_window) -> None:
    """The inner floor is at least the sum of the two side-by-side panes.

    The image table (left) and the queue table (right) sit in a horizontal
    layout; when the screen cannot fit their floor, the outer viewport scrolls
    without squeezing either pane below its useful width.
    """
    window = make_window()
    panes_min_width = (
        window.image_table.minimumWidth() + window.queue_table.minimumWidth()
    )

    assert window.centralWidget().widget().minimumWidth() >= panes_min_width


def test_fusion_scrollbars_are_non_transient(qapp: QApplication) -> None:
    """The app's selected toolkit style keeps needed scrollbars discoverable."""
    style = QStyleFactory.create("Fusion")
    assert style is not None
    try:
        assert style.styleHint(QStyle.StyleHint.SH_ScrollBar_Transient) == 0
    finally:
        style.deleteLater()


def test_title_label_derives_a_logical_pixel_size(qapp: QApplication) -> None:
    label = title_label("PixelUp")
    try:
        assert label.font().pixelSize() == round(DEFAULT_UI_FONT_SIZE * 1.6)
        assert label.font().bold()
    finally:
        label.deleteLater()
