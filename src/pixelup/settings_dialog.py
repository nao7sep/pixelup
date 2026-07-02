from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from pixelup.app_config import (
    MAX_CONCURRENT_JOBS,
    MAX_QUALITY,
    MAX_TILE,
    MIN_CONCURRENT_JOBS,
    MIN_QUALITY,
    MIN_TILE,
    TILE_STEP,
    AppConfig,
)
from pixelup.fonts import DEFAULT_UI_FONT_FAMILY, normalize_font_family
from pixelup.paths import OutputFormat
from pixelup.ui_common import secondary_label, use_regular_spacing
from pixelup.widgets import NoWheelSpinBox, device_combo, output_format_combo

# A settled column width for the value fields so a control and its wrapped
# caption share one edge, and captions wrap predictably instead of stretching
# the dialog to the widest single line.
_FIELD_WIDTH = 320


def _captioned(control: QWidget, caption: str, *, fill: bool = False) -> QWidget:
    """Group a control with a muted caption directly beneath it.

    The caption reads as sub-text of its control (a tight 2px gap) rather than a
    peer form row a full row-gap away. ``fill`` lets a wide field (the font line
    edit) span the column, while compact controls (spin boxes, combos) stay left.
    """
    container = QWidget()
    container.setFixedWidth(_FIELD_WIDTH)
    box = QVBoxLayout(container)
    box.setContentsMargins(0, 0, 0, 0)
    box.setSpacing(2)
    if fill:
        box.addWidget(control)
    else:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(control)
        row.addStretch()
        box.addLayout(row)
    cap = secondary_label(caption)
    cap.setWordWrap(True)
    box.addWidget(cap)
    return container


class SettingsDialog(QDialog):
    """Modal settings editor.

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

        self.format = output_format_combo()
        self.format.setCurrentIndex(self.format.findData(config.output_format.value))

        self.quality = NoWheelSpinBox()
        self.quality.setRange(MIN_QUALITY, MAX_QUALITY)
        self.quality.setValue(config.quality)

        self.tile = NoWheelSpinBox()
        self.tile.setRange(MIN_TILE, MAX_TILE)
        self.tile.setSingleStep(TILE_STEP)
        self.tile.setValue(config.tile)

        self.device = device_combo()
        self.device.setCurrentIndex(self.device.findData(config.device))

        self.font_family = QLineEdit()
        self.font_family.setText(config.font_family)
        self.font_family.setMinimumWidth(260)

        # Each caption groups under its own control (see _captioned) rather than
        # floating a full row-gap away, so the label reads as sub-text of the
        # field it explains. Row labels top-align to sit beside the control, not
        # the caption. UI font leads as the one app-appearance setting, set apart
        # from the image-processing job defaults that follow.
        label_align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        row = 0
        form.addWidget(QLabel("UI font"), row, 0, label_align)
        form.addWidget(
            _captioned(
                self.font_family,
                "Comma-separated; the first font installed on this system is used.",
                fill=True,
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
        row += 1
        form.addWidget(QLabel("Output format"), row, 0)
        form.addWidget(self.format, row, 1, Qt.AlignmentFlag.AlignLeft)
        row += 1
        form.addWidget(QLabel("Quality"), row, 0, label_align)
        form.addWidget(
            _captioned(self.quality, "Used for JPG and WebP. Ignored for PNG."),
            row,
            1,
        )
        row += 1
        form.addWidget(QLabel("Tile size"), row, 0, label_align)
        form.addWidget(
            _captioned(self.tile, "Lower uses less memory; 0 processes the whole image at once."),
            row,
            1,
        )
        row += 1
        form.addWidget(QLabel("Device"), row, 0, label_align)
        form.addWidget(
            _captioned(self.device, "Auto lets Real-ESRGAN choose the best available device."),
            row,
            1,
        )
        form.setColumnStretch(0, 0)
        form.setColumnStretch(1, 0)
        form.setColumnStretch(2, 1)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.ok_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        restore = self.buttons.addButton("Restore defaults", QDialogButtonBox.ButtonRole.ResetRole)
        restore.clicked.connect(self._restore_defaults)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout.addWidget(form_widget, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.buttons)
        self.adjustSize()
        self.setMinimumSize(self.sizeHint())

        for changed in (
            self.concurrent.valueChanged,
            self.auto_download.toggled,
            self.format.currentIndexChanged,
            self.quality.valueChanged,
            self.tile.valueChanged,
            self.device.currentIndexChanged,
            self.font_family.textChanged,
        ):
            changed.connect(self._update_commit_enabled)
        self._update_commit_enabled()

    def config(self) -> AppConfig:
        return AppConfig(
            max_concurrent_jobs=self.concurrent.value(),
            output_format=OutputFormat(self.format.currentData()),
            quality=self.quality.value(),
            tile=self.tile.value(),
            device=self.device.currentData(),
            auto_download=self.auto_download.isChecked(),
            font_family=normalize_font_family(self.font_family.text(), DEFAULT_UI_FONT_FAMILY),
        )

    def is_dirty(self) -> bool:
        return self.config() != self._initial

    def _update_commit_enabled(self) -> None:
        self.ok_button.setEnabled(self.is_dirty())

    def _restore_defaults(self) -> None:
        defaults = AppConfig()
        self.concurrent.setValue(defaults.max_concurrent_jobs)
        self.auto_download.setChecked(defaults.auto_download)
        self.format.setCurrentIndex(self.format.findData(defaults.output_format.value))
        self.quality.setValue(defaults.quality)
        self.tile.setValue(defaults.tile)
        self.device.setCurrentIndex(self.device.findData(defaults.device))
        self.font_family.setText(defaults.font_family)
