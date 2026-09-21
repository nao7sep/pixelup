from __future__ import annotations

from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from pixelup import __version__
from pixelup.about_dialog import AboutDialog


def test_about_dialog_includes_required_metadata(qapp: QApplication) -> None:
    dialog = AboutDialog()
    try:
        text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
        assert dialog.windowTitle() == "About PixelUp"
        # The title bar is the header, so nothing inside repeats it — but an about
        # surface does name the product, which is what it is for, and shows the
        # version beside it (modal-dialog-conventions).
        assert dialog.windowTitle() not in text
        assert "PixelUp" in text
        assert __version__ in text
        assert "© 2026 Yoshinao Inoguchi · GNU GPL v3 or later" in text
    finally:
        dialog.deleteLater()


def test_external_open_failure_stays_in_about_and_resizes_for_authored_result(
    qapp: QApplication,
) -> None:
    hostile = "EACCES IPC /private/tmp/PIXELUP-ABOUT-SENTINEL"

    def fail(_url: str) -> None:
        raise RuntimeError(hostile)

    dialog = AboutDialog(opener=fail)
    try:
        dialog.show()
        qapp.processEvents()
        before = dialog.height()
        github = next(
            button
            for button in dialog.findChildren(QPushButton)
            if button.text() == "GitHub"
        )

        github.click()
        qapp.processEvents()

        assert dialog.launch_result.isVisibleTo(dialog)
        assert hostile not in dialog.launch_result.message_label.text()
        assert dialog.height() > before
        assert dialog.launch_result.dismiss_button.isVisibleTo(dialog)

        dialog.launch_result.dismiss_button.click()
        qapp.processEvents()
        assert not dialog.launch_result.isVisible()
        # And back to where it started. The surface is fixed, so a dialog that
        # only grows keeps the dismissed banner's room for as long as it is open.
        assert dialog.height() == before
    finally:
        dialog.close()
        dialog.deleteLater()
