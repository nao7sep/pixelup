from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
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
from pixelup.ui_common import secondary_label, use_regular_spacing
from pixelup.widgets import NoWheelSpinBox

# A settled column width for the value field so the control and its wrapped
# caption share one edge, and the caption wraps predictably instead of stretching
# the dialog to the widest single line.
_FIELD_WIDTH = 320


def _captioned(control: QWidget, caption: str) -> QWidget:
    """Group a control with a muted caption directly beneath it.

    The caption reads as sub-text of its control (a tight 2px gap) rather than a
    peer form row a full row-gap away. The control spans the column: the font line
    edit is the only captioned field left here, now that the parameters live in the
    main window's panel.
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
    so this dialog deliberately holds only the leftovers: the UI font, the
    model-download toggle, and the concurrent job count. It carries no reset button
    — none of these three is a stale-able built-in or a tuned, interacting set worth
    returning to (a font is a preference, a toggle is a toggle, a job count is one
    number), so there is no default worth a control (config-seeding-conventions).

    The widgets hold a draft; the incoming config is never mutated. The commit
    (OK) button stays disabled until the draft differs from the config the
    dialog was opened with, so accepting always means "apply a real change".
    """

    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._initial = config
        self.setWindowTitle("Settings")
        self.setModal(True)

        layout = QVBoxLayout(self)
        use_regular_spacing(layout)

        form_widget = QWidget()
        form = QGridLayout(form_widget)
        use_regular_spacing(form)

        self.concurrent = NoWheelSpinBox()
        self.concurrent.setRange(MIN_CONCURRENT_JOBS, MAX_CONCURRENT_JOBS)
        self.concurrent.setValue(config.max_concurrent_jobs)

        self.auto_download = QCheckBox("Download missing models automatically")
        self.auto_download.setChecked(config.auto_download)

        self.font_family = QLineEdit()
        self.font_family.setText(config.font_family)
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
                "Comma-separated; the first font installed on this system is used.",
            ),
            row,
            1,
        )
        row += 1
        form.addWidget(QLabel(""), row, 0)
        form.addWidget(self.auto_download, row, 1, Qt.AlignmentFlag.AlignLeft)
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
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout.addWidget(form_widget, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.buttons)
        self.adjustSize()
        self.setMinimumSize(self.sizeHint())

        for changed in (
            self.concurrent.valueChanged,
            self.auto_download.toggled,
            self.font_family.textChanged,
        ):
            changed.connect(self._update_commit_enabled)
        self._update_commit_enabled()

    def config(self) -> AppConfig:
        """The draft config: the opened one with exactly this dialog's three fields replaced.

        Built by ``replace`` rather than a fresh ``AppConfig(...)`` so the settings this
        dialog does not show — today the Parameters panel — pass through untouched.
        Constructing one here would silently reset the panel to its built-ins every time
        the user pressed OK on a font change.
        """
        return replace(
            self._initial,
            max_concurrent_jobs=self.concurrent.value(),
            auto_download=self.auto_download.isChecked(),
            font_family=normalize_font_family(self.font_family.text(), DEFAULT_UI_FONT_FAMILY),
        )

    def is_dirty(self) -> bool:
        return self.config() != self._initial

    def _update_commit_enabled(self) -> None:
        self.ok_button.setEnabled(self.is_dirty())
