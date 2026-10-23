from __future__ import annotations

import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QFrame, QLabel, QScrollArea, QWidget

from pixelup.about_dialog import AboutDialog
from pixelup.app_config import AppConfig
from pixelup.dialog_shell import (
    ABOUT_WIDTH,
    DIALOG_HEIGHT_FRACTION,
    FORM_WIDTH,
    NOTICE_WIDTH,
    TABLE_WIDTH,
    DialogShell,
)
from pixelup.managed_models_dialog import ManagedModelsDialog
from pixelup.message_dialogs import MessageDialog, StartupFailureDialog
from pixelup.model_manager import ModelManager
from pixelup.parameters_help_dialog import ParametersHelpDialog
from pixelup.quit_dialog import QuitConfirmDialog, QuitSaveFailedDialog
from pixelup.settings_dialog import SettingsDialog
from pixelup.shortcuts_dialog import ShortcutsDialog

# Every blocking surface PixelUp draws itself. Named, so a failure says which one,
# and built per test so no dialog inherits another's geometry.
BUILDERS = {
    "about": lambda: AboutDialog(),
    "quit": lambda: QuitConfirmDialog(2),
    "quit-save-failed": lambda: QuitSaveFailedDialog(),
    "shortcuts": lambda: ShortcutsDialog(),
    "settings": lambda: SettingsDialog(AppConfig()),
    "parameters-help": lambda: ParametersHelpDialog(),
    "message": lambda: MessageDialog("PixelUp", "A short one-shot notice."),
    "startup-failure": lambda: StartupFailureDialog("PixelUp could not start.", None),
    "managed-models": lambda: ManagedModelsDialog(
        ModelManager(Path(tempfile.mkdtemp()))
    ),
}


@pytest.fixture(params=sorted(BUILDERS))
def dialog(request: pytest.FixtureRequest, qapp: QApplication) -> Iterator[DialogShell]:
    surface = BUILDERS[request.param]()
    if isinstance(surface, ManagedModelsDialog):
        # Geometry assertions compare the same loaded state before and after show.
        deadline = time.monotonic() + 5
        while not surface._manager.readiness_known:
            assert time.monotonic() < deadline
            qapp.processEvents()
    yield surface
    surface.close()
    surface.deleteLater()


def test_every_dialog_uses_the_shared_shell(dialog: DialogShell) -> None:
    # The conventions ask for one shared shell per app so the chrome is decided in
    # one place. A dialog that subclasses QDialog directly would be deciding its
    # own bands, bound and sizing again, which is how the seven drifted apart.
    assert isinstance(dialog, DialogShell)


def test_cmd_or_ctrl_w_closes_the_dialog_like_its_close_button(
    dialog: DialogShell, qapp: QApplication, process_until
) -> None:
    dialog.show()
    dialog.activateWindow()
    process_until(dialog.isActiveWindow, timeout_s=5, what="Dialog becomes the key window")

    QTest.keyClick(dialog, Qt.Key.Key_W, Qt.KeyboardModifier.ControlModifier)

    assert not dialog.isVisible()
    assert dialog.result() == QDialog.DialogCode.Rejected


def test_a_dialog_takes_focus_above_its_opener_and_returns_it(
    dialog: DialogShell, qapp: QApplication, process_until
) -> None:
    # window-conventions: a window opened at the user's request becomes the key
    # window above the one it was opened from. Owned by its opener, the OS keeps
    # it above and hands focus back when it closes; the offscreen test platform
    # reactivates nothing on close, so that last step is the OS's to qualify.
    opener = QWidget()
    opener.show()
    opener.activateWindow()
    process_until(opener.isActiveWindow, timeout_s=5, what="Opener becomes the key window")
    dialog.setParent(opener, dialog.windowFlags())
    try:
        dialog.show()
        process_until(dialog.isActiveWindow, timeout_s=5, what="Dialog becomes the key window")
        assert dialog.windowHandle().transientParent() is opener.windowHandle()
    finally:
        dialog.setParent(None)
        opener.close()
        opener.deleteLater()


def test_the_title_bar_is_the_header(dialog: DialogShell) -> None:
    # A window the OS draws a title bar for takes two bands, not three: the bar
    # already names the surface, so no heading inside repeats it. The about
    # surface is the stated exception and names the product instead.
    body_text = {label.text() for label in dialog.body.findChildren(QLabel)}
    assert dialog.windowTitle() not in body_text


def test_the_footer_band_opens_with_a_line_across_the_dialog(
    dialog: DialogShell, qapp: QApplication
) -> None:
    dialog.show()
    qapp.processEvents()

    line = dialog.findChild(QFrame, "dialogFooterLine")
    assert line is not None
    assert line.height() == 1
    # Across the whole surface, not inset from it: a line that stops short of the
    # edges is a rule inside a band, not the boundary between two.
    assert line.width() == dialog.width()
    assert line.y() == dialog.body_scroll.height()


def test_the_body_is_the_only_scroll_region(dialog: DialogShell) -> None:
    # The footer is docked outside it, so it stays reachable at whatever height
    # the body is bounded to, and nothing nests a second scroller inside.
    areas = dialog.findChildren(QScrollArea)
    assert areas == [dialog.body_scroll]
    assert dialog.body_scroll.widget() is dialog.body


def test_a_dialog_is_fixed_and_takes_a_chosen_width(dialog: DialogShell) -> None:
    # None of these is a surface anyone settles into and works in, so none offers
    # an edge to drag; what replaces the handle is a width that was chosen rather
    # than inherited from whatever the content happened to measure.
    assert dialog.minimumSize() == dialog.maximumSize()
    assert dialog.width() in {ABOUT_WIDTH, NOTICE_WIDTH, FORM_WIDTH, TABLE_WIDTH}


def test_the_bound_is_applied_before_the_window_is_shown(
    dialog: DialogShell, qapp: QApplication
) -> None:
    # Qt's adjustSize applies no screen cap of its own, and a dialog placed at its
    # unbounded height is placed wrong even if it is shrunk afterwards. So the
    # size is already settled here, with nothing shown yet.
    assert not dialog.isVisible()
    settled = dialog.size()
    assert settled == dialog.minimumSize()

    screen = dialog.screen() or QApplication.primaryScreen()
    assert settled.height() <= screen.availableGeometry().height() * DIALOG_HEIGHT_FRACTION

    dialog.show()
    qapp.processEvents()
    assert dialog.size() == settled


def test_the_scroll_bar_never_takes_a_column_out_of_the_body(
    dialog: DialogShell, qapp: QApplication
) -> None:
    # The body is laid out at the viewport's own width, so whether the bar is
    # there or not, nothing in the body is left outside what is drawn — the bar
    # sits in the padding at the dialog's edge rather than over a control.
    dialog.show()
    qapp.processEvents()

    assert dialog.body.width() == dialog.body_scroll.viewport().width()
    assert dialog.body_scroll.horizontalScrollBar().isVisible() is False


def test_only_opted_in_reference_bodies_are_focus_targets(dialog: DialogShell) -> None:
    expected = (
        Qt.FocusPolicy.StrongFocus
        if isinstance(dialog, (ParametersHelpDialog, ShortcutsDialog))
        else Qt.FocusPolicy.NoFocus
    )
    assert dialog.body_scroll.focusPolicy() == expected
    assert dialog.body_scroll.frameShape() == QFrame.Shape.NoFrame


def test_every_dialog_opens_on_a_control_it_chose(
    dialog: DialogShell, qapp: QApplication
) -> None:
    # Left unsaid, Qt hands focus to whatever comes first in the tab order, which
    # is rarely what the reader wants next and never a deliberate choice
    # (modal-dialog-conventions).
    dialog.show()
    qapp.processEvents()

    focused = dialog.focusWidget()
    assert focused is not None, "nothing takes focus when this dialog opens"
    assert focused is not dialog.body_scroll
    assert dialog.isAncestorOf(focused)


def test_opted_in_body_preserves_descendant_editing_keys(qapp: QApplication) -> None:
    from PySide6.QtWidgets import QLineEdit, QPushButton, QSpinBox

    dialog = DialogShell("help.title", width=FORM_WIDTH, body_height_limit=160)
    dialog.enable_keyboard_scrolling()
    field = QLineEdit("alpha beta")
    number = QSpinBox()
    number.setValue(4)
    dialog.body_layout.addWidget(field)
    dialog.body_layout.addWidget(number)
    for _ in range(12):
        dialog.body_layout.addWidget(QLabel("Reference text"))
    close = QPushButton("Close")
    close.clicked.connect(dialog.reject)
    dialog.add_footer_widget(close)
    dialog.set_initial_focus(close)
    dialog.fit()
    try:
        dialog.show()
        dialog.activateWindow()
        qapp.processEvents()
        field.setFocus()
        bar = dialog.body_scroll.verticalScrollBar()
        bar.setValue(0)
        QTest.keyClick(field, Qt.Key.Key_End)
        assert field.cursorPosition() == len(field.text())
        QTest.keyClick(field, Qt.Key.Key_Home)
        assert field.cursorPosition() == 0
        QTest.keyClick(field, Qt.Key.Key_Space)
        assert field.text() == " alpha beta"
        QTest.keyClick(field, Qt.Key.Key_PageDown)
        assert bar.value() == 0
        number.setFocus()
        bar.setValue(0)
        QTest.keyClick(number, Qt.Key.Key_Up)
        assert number.value() == 5
        assert bar.value() == 0
    finally:
        dialog.close()
        dialog.deleteLater()
