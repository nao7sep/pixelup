from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from pixelup.app_config import MAX_CONCURRENT_JOBS, MIN_CONCURRENT_JOBS, AppConfig
from pixelup.fonts import DEFAULT_UI_FONT_FAMILY, normalize_font_family
from pixelup.ui_common import secondary_label, title_label, use_dialog_spacing, use_regular_spacing
from pixelup.widgets import NoWheelSpinBox

# A settled column width for the value field so the control and its wrapped
# caption share one edge, and the caption wraps predictably instead of stretching
# the dialog to the widest single line.
_FIELD_WIDTH = 320


def _captioned(control: QWidget, caption: str) -> QWidget:
    """Group a control with a muted caption directly beneath it.

    The caption reads as sub-text of its control (a tight 2px gap) rather than a
    peer form row a full row-gap away. The control spans the column; the parameters
    live in the main window's panel rather than this dialog.
    """
    container = QWidget()
    container.setFixedWidth(_FIELD_WIDTH)
    box = QVBoxLayout(container)
    box.setContentsMargins(0, 0, 0, 0)
    box.setSpacing(2)
    box.addWidget(control)
    cap = secondary_label(caption)
    cap.setWordWrap(True)
    box.addWidget(cap)
    return container


class SettingsDialog(QDialog):
    """Modal settings editor: everything PixelUp persists that the main window does not show.

    One home per thing. The image-processing parameters live in the main window's
    Parameters panel, which persists its own edits and resets to its own built-ins,
    so this dialog deliberately holds only the leftovers: the UI font and the
    concurrent job count. It carries no reset button — neither setting is a stale-able
    built-in or a tuned, interacting set worth returning to, so there is no default
    worth a control (config-seeding-conventions).

    The widgets hold a draft; the incoming config is never mutated. The commit
    (OK) button stays disabled until the draft differs from the config the
    dialog was opened with, so accepting always means "apply a real change".
    """

    def __init__(
        self,
        config: AppConfig,
        parent: QWidget | None = None,
        *,
        try_save: Callable[[AppConfig], bool] | None = None,
    ) -> None:
        super().__init__(parent)
        self._initial = config
        self._try_save = try_save
        self.setWindowTitle("Settings")
        self.setModal(True)

        layout = QVBoxLayout(self)
        use_dialog_spacing(layout)
        layout.addWidget(title_label("Settings"))

        form_widget = QWidget()
        form = QGridLayout(form_widget)
        use_regular_spacing(form)

        self.concurrent = NoWheelSpinBox()
        self.concurrent.setRange(MIN_CONCURRENT_JOBS, MAX_CONCURRENT_JOBS)
        self.concurrent.setValue(config.max_concurrent_jobs)

        self.font_family = QLineEdit()
        self.font_family.setText(config.font_family)
        self.font_family.setPlaceholderText("Platform default")
        self.font_family.setMinimumWidth(260)

        # Each caption groups under its own control (see _captioned) rather than
        # floating a full row-gap away, so the label reads as sub-text of the
        # field it explains. Row labels top-align to sit beside the control, not
        # the caption.
        label_align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        row = 0
        form.addWidget(QLabel("UI font"), row, 0, label_align)
        form.addWidget(
            _captioned(
                self.font_family,
                "Blank uses the platform UI face. Otherwise, the first installed font is used.",
            ),
            row,
            1,
        )
        row += 1
        form.addWidget(QLabel("Concurrent jobs"), row, 0)
        form.addWidget(self.concurrent, row, 1, Qt.AlignmentFlag.AlignLeft)
        form.setColumnStretch(0, 0)
        form.setColumnStretch(1, 0)
        form.setColumnStretch(2, 1)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.ok_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.error_message = QLabel()
        self.error_message.setWordWrap(True)
        self.error_message.setStyleSheet("color: palette(bright-text);")
        self.error_message.setAccessibleName("Settings save error")
        self.error_message.hide()
        self.buttons.accepted.connect(self._save)
        self.buttons.rejected.connect(self.reject)

        layout.addWidget(form_widget, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.error_message)
        layout.addWidget(self.buttons)
        self.adjustSize()
        self.setMinimumSize(self.sizeHint())

        for changed in (
            self.concurrent.valueChanged,
            self.font_family.textChanged,
        ):
            changed.connect(self._update_commit_enabled)
        self._update_commit_enabled()

    def config(self) -> AppConfig:
        """The draft config: the opened one with exactly this dialog's two fields replaced.

        Built by ``replace`` rather than a fresh ``AppConfig(...)`` so the settings this
        dialog does not show — today the Parameters panel — pass through untouched.
        Constructing one here would silently reset the panel to its built-ins every time
        the user pressed OK on a font change.
        """
        return replace(
            self._initial,
            max_concurrent_jobs=self.concurrent.value(),
            font_family=normalize_font_family(self.font_family.text(), DEFAULT_UI_FONT_FAMILY),
        )

    def is_dirty(self) -> bool:
        return self.config() != self._initial

    def _update_commit_enabled(self) -> None:
        self.ok_button.setEnabled(self.is_dirty())

    def _save(self) -> None:
        candidate = self.config()
        self.error_message.clear()
        self.error_message.hide()
        if self._try_save is not None and not self._try_save(candidate):
            self.error_message.setText(
                "PixelUp could not save your settings. Your changes are still here; try again."
            )
            self.error_message.show()
            return
        self.accept()
