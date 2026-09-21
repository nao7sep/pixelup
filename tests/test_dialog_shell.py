from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QScrollArea

from pixelup.about_dialog import AboutDialog
from pixelup.app_config import AppConfig
from pixelup.dialog_shell import (
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
from pixelup.quit_dialog import QuitConfirmDialog
from pixelup.settings_dialog import SettingsDialog
from pixelup.shortcuts_dialog import ShortcutsDialog

# Every blocking surface PixelUp draws itself. Named, so a failure says which one,
# and built per test so no dialog inherits another's geometry.
BUILDERS = {
    "about": lambda: AboutDialog(),
    "quit": lambda: QuitConfirmDialog(2),
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
    yield surface
    surface.close()
    surface.deleteLater()


def test_every_dialog_uses_the_shared_shell(dialog: DialogShell) -> None:
    # The conventions ask for one shared shell per app so the chrome is decided in
    # one place. A dialog that subclasses QDialog directly would be deciding its
    # own bands, bound and sizing again, which is how the seven drifted apart.
    assert isinstance(dialog, DialogShell)


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
    assert dialog.width() in {NOTICE_WIDTH, FORM_WIDTH, TABLE_WIDTH}


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


def test_no_dialog_body_is_a_focus_target(dialog: DialogShell) -> None:
    # The body reaches the dialog's own edges, so any focus treatment it carried
    # would draw a second border just inside the window — which is exactly what a
    # scroll region that was also a focus target drew here. The controls inside
    # keep their own focus; the region itself takes none.
    assert dialog.body_scroll.focusPolicy() == Qt.FocusPolicy.NoFocus


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
