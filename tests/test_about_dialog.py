from __future__ import annotations

from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from pixelup import __version__
from pixelup.about_dialog import AboutDialog


def test_about_dialog_includes_required_metadata(qapp: QApplication) -> None:
    dialog = AboutDialog()
    try:
        text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
        assert dialog.windowTitle() == "About PixelUp"
        assert dialog.windowTitle() in text
        assert "PixelUp" in text
        assert __version__ in text
        assert "© 2026 Yoshinao Inoguchi · MIT License" in text
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
        assert not dialog.launch_result.isVisible()
    finally:
        dialog.close()
        dialog.deleteLater()
