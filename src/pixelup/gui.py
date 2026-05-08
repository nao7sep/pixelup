from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from itertools import count
from pathlib import Path

from PySide6.QtCore import (
    QObject,
    QRect,
    QSize,
    Qt,
    QThread,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QGuiApplication,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLayoutItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QSplitterHandle,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible, QElidingLabel

from pixelup import __version__
from pixelup.app_config import CONFIG_PATH, AppConfig, load_app_config, save_app_config
from pixelup.config import resolve_runtime_dirs
from pixelup.errors import ErrorCode, PixelupError
from pixelup.imaging import read_image_size, register_image_plugins
from pixelup.models import KNOWN_MODELS
from pixelup.paths import OutputFormat, default_output_path
from pixelup.session_log import configure_session_logging, get_logger
from pixelup.sidecar import write_sidecar
from pixelup.upscale import UpscaleOptions, run_upscale

LOGGER = get_logger()
PROJECT_URL = "https://github.com/nao7sep/pixelup"
ISSUES_URL = "https://github.com/nao7sep/pixelup/issues"
MODEL_ORDER = (
    "realesr-general-x4v3",
    "RealESRGAN_x4plus",
    "RealESRNet_x4plus",
    "RealESRGAN_x2plus",
    "RealESRGAN_x4plus_anime_6B",
    "realesr-animevideov3",
)
MODEL_DESCRIPTIONS = {
    "realesr-general-x4v3": "Best default for mixed images.",
    "RealESRGAN_x4plus": "General 4x with stronger detail enhancement.",
    "RealESRNet_x4plus": "General 4x with a softer, less GAN-heavy look.",
    "RealESRGAN_x2plus": "2x upscale when 4x is too much.",
    "RealESRGAN_x4plus_anime_6B": "Best for anime and flat illustrations.",
    "realesr-animevideov3": "Good for anime and video-frame style images.",
}
KNOWN_MODEL_NAMES = {model.name for model in KNOWN_MODELS}
UPSCALE_MODELS = tuple(name for name in MODEL_ORDER if name in KNOWN_MODEL_NAMES)
TAB_CHIP_MAX_WIDTH = 220
LEFT_PANE_MIN_WIDTH = 300
LEFT_PANE_START_WIDTH = 330
APP_STYLESHEET = """
QWidget {
    background: transparent;
    color: #24324d;
    font-size: 13px;
}
QMainWindow,
QDialog,
QWidget#appRoot,
QWidget#dialogRoot {
    background: #eff4ff;
}
QFrame#topBarCard,
QFrame#tabStripCard,
QFrame#panelCard,
QFrame#dialogCard,
QFrame#sectionCard,
QFrame#modelCard {
    background: #ffffff;
    border: 1px solid #d9e3ff;
    border-radius: 14px;
}
QFrame#topBarCard {
    background: #f4f1ff;
    border-color: #d7ccff;
}
QFrame#tabStripCard {
    background: #f6f9ff;
}
QFrame#modelCard {
    background: #f8fbff;
}
QLabel#windowTitle {
    font-size: 20px;
    font-weight: 700;
    color: #24324d;
}
QLabel#mutedText,
QLabel#pathText,
QLabel#modelNote,
QLabel#dialogSubtitle {
    color: #66789c;
}
QLabel#mutedText {
    font-size: 13px;
}
QLabel#sectionTitle {
    font-size: 15px;
    font-weight: 700;
    color: #4050a8;
}
QLabel {
    border: none;
}
QPushButton,
QToolButton {
    background: #ffffff;
    border: 1px solid #c9d6ff;
    border-radius: 10px;
    padding: 7px 12px;
}
QPushButton:hover,
QToolButton:hover {
    border-color: #91a6ff;
}
QPushButton:disabled,
QToolButton:disabled {
    background: #f8faff;
    border-color: #dbe3f5;
    color: #9aa8c5;
}
QPushButton:pressed,
QToolButton:pressed {
    background: #eef2ff;
}
QPushButton#primaryButton {
    background: #4f46e5;
    color: #ffffff;
    border-color: #4f46e5;
}
QPushButton#primaryButton:hover {
    background: #4338ca;
    border-color: #4338ca;
}
QPushButton#primaryButton:disabled {
    background: #c7d2fe;
    border-color: #c7d2fe;
    color: #eef2ff;
}
QPushButton#chipButton {
    padding: 6px 12px;
}
QPushButton#advancedToggle {
    background: transparent;
    border: none;
    padding: 0 0 0 2px;
    font-weight: 600;
}
QPushButton#advancedToggle[modified="true"] {
    color: #c2410c;
}
QPushButton#advancedToggle:hover {
    color: #344054;
}
QComboBox,
QSpinBox,
QDoubleSpinBox {
    background: #ffffff;
    border: 1px solid #c9d6ff;
    border-radius: 10px;
    padding: 6px 10px;
    min-height: 22px;
}
QComboBox:hover,
QSpinBox:hover,
QDoubleSpinBox:hover {
    border-color: #91a6ff;
}
QComboBox:disabled,
QSpinBox:disabled,
QDoubleSpinBox:disabled {
    background: #f8faff;
    border-color: #dbe3f5;
    color: #9aa8c5;
}
QAbstractItemView {
    background: #ffffff;
    color: #24324d;
    border: 1px solid #c9d6ff;
    outline: 0;
    selection-background-color: #e8eeff;
    selection-color: #24324d;
}
QToolButton#helpButton {
    background: transparent;
    border: none;
    color: #5c6c92;
    font-weight: 700;
    padding: 0 2px;
}
QToolButton#helpButton:hover {
    color: #4050a8;
}
QScrollArea,
QScrollArea > QWidget > QWidget {
    border: none;
    background: transparent;
}
QTableWidget {
    background: #ffffff;
    border: 1px solid #d9e3ff;
    border-radius: 12px;
    gridline-color: #edf2ff;
    font-size: 13px;
}
QTableWidget::item {
    padding: 6px 10px;
}
QHeaderView::section {
    background: #f3f6ff;
    color: #5c6c92;
    border: none;
    border-bottom: 1px solid #d9e3ff;
    padding: 6px 10px;
    font-weight: 600;
}
QSplitter::handle {
    background: transparent;
    margin: 12px 0;
    border-radius: 4px;
}
"""


@dataclass(frozen=True, slots=True)
class AdvancedSettings:
    face_enhance: bool = False
    denoise_strength: float = 0.5
    alpha_mode: str = "realesrgan"
    device: str = "auto"
    output_format: OutputFormat = OutputFormat.PNG
    quality: int = 95
    tile: int = 0
    strip_metadata: bool = False
    target_profile: str | None = None


@dataclass(slots=True)
class Job:
    id: int
    input_path: Path
    model: str
    scale: int
    output_path: Path
    advanced: AdvancedSettings
    auto_download: bool
    status: str = "pending"
    message: str = ""
    warnings: list[str] = field(default_factory=list)


class JobSignals(QObject):
    progress = Signal(int, str)
    finished = Signal(int, bool, str, object, object)


class JobWorker(QObject):
    def __init__(self, job: Job) -> None:
        super().__init__()
        self.job = job
        self.signals = JobSignals()
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return self._cancel_requested

    @Slot()
    def run(self) -> None:
        warnings: list[str] = []
        LOGGER.info(
            "Starting job %s details=%s",
            self.job.id,
            _job_log_payload(self.job),
        )
        try:
            options = _options_for_job(self.job)
            result = run_upscale(
                options,
                resolve_runtime_dirs(),
                on_progress=lambda phase: self.signals.progress.emit(
                    self.job.id,
                    _progress_text(phase),
                ),
                on_tile=lambda done, total: self.signals.progress.emit(
                    self.job.id,
                    _tile_progress_text(done, total),
                ),
                on_download=lambda model, done, total: self.signals.progress.emit(
                    self.job.id,
                    _download_text(model, done, total),
                ),
                on_warning=warnings.append,
                should_cancel=self._is_cancelled,
            )
            sidecar = write_sidecar(
                input_path=self.job.input_path,
                output_path=self.job.output_path,
                options=options,
                result=result,
                warnings=warnings,
            )
            result["sidecar"] = str(sidecar)
            LOGGER.info(
                "Finished job %s output=%s sidecar=%s warnings=%s",
                self.job.id,
                self.job.output_path,
                sidecar,
                warnings,
            )
            self.signals.finished.emit(self.job.id, True, "Done", result, warnings)
        except PixelupError as exc:
            if exc.code == ErrorCode.JOB_CANCELLED:
                LOGGER.info("Job %s cancelled", self.job.id)
                self.signals.finished.emit(
                    self.job.id, False, "Cancelled", {"cancelled": True}, warnings
                )
                return
            LOGGER.warning(
                "Job %s failed message=%s warnings=%s details=%s",
                self.job.id,
                exc.message,
                warnings,
                _job_log_payload(self.job),
            )
            self.signals.finished.emit(self.job.id, False, exc.message, {}, warnings)
        except Exception as exc:
            LOGGER.exception(
                "Job %s failed unexpectedly details=%s",
                self.job.id,
                _job_log_payload(self.job),
            )
            self.signals.finished.emit(self.job.id, False, f"Unexpected error: {exc}", {}, warnings)


class FlowLayout(QLayout):
    def __init__(self, parent: QWidget | None = None, margin: int = 0, spacing: int = 8) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> QLayoutItem | None:
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientations:
        return Qt.Orientations()

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        x = rect.x()
        y = rect.y()
        line_height = 0
        right = rect.right()

        for item in self._items:
            spacing = self.spacing()
            size = item.sizeHint()
            next_x = x + size.width() + spacing
            if line_height > 0 and next_x - spacing > right + 1:
                x = rect.x()
                y = y + line_height + spacing
                next_x = x + size.width() + spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(x, y, size.width(), size.height()))
            x = next_x
            line_height = max(line_height, size.height())
        return y + line_height - rect.y()


class PreviewLabel(QLabel):
    def __init__(self) -> None:
        super().__init__("Preview unavailable")
        self._source: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(340)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image_path(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        self._source = pixmap if not pixmap.isNull() else None
        self._update_pixmap()

    def resizeEvent(self, event: object) -> None:
        self._update_pixmap()
        super().resizeEvent(event)

    def _update_pixmap(self) -> None:
        if self._source is None:
            self.setPixmap(QPixmap())
            self.setText("Preview unavailable")
            return
        size = self.contentsRect().size() - QSize(4, 4)
        if size.width() <= 0 or size.height() <= 0:
            return
        self.setText("")
        self.setPixmap(
            self._source.scaled(
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class IndicatorCheckBox(QCheckBox):
    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(max(hint.width(), self.fontMetrics().horizontalAdvance(self.text()) + 28), 24)

    def paintEvent(self, event: object) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        indicator = QRect(0, (rect.height() - 16) // 2, 16, 16)
        border = QColor("#4f46e5") if self.isChecked() else QColor("#9aa8c5")
        fill = QColor("#4f46e5") if self.isChecked() else QColor("#ffffff")
        painter.setPen(QPen(border, 1.4))
        painter.setBrush(fill)
        painter.drawRoundedRect(indicator.adjusted(1, 1, -1, -1), 4, 4)
        if self.isChecked():
            painter.setPen(
                QPen(QColor("#ffffff"), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            )
            painter.drawLine(4, indicator.center().y(), 7, indicator.bottom() - 4)
            painter.drawLine(7, indicator.bottom() - 4, 13, indicator.top() + 4)
        painter.setPen(QPen(QColor("#24324d")))
        painter.drawText(
            QRect(26, 0, rect.width() - 26, rect.height()),
            Qt.AlignmentFlag.AlignVCenter,
            self.text(),
        )


class IndicatorRadioButton(QRadioButton):
    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(max(hint.width(), self.fontMetrics().horizontalAdvance(self.text()) + 28), 24)

    def paintEvent(self, event: object) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        indicator = QRect(0, (rect.height() - 16) // 2, 16, 16)
        border = QColor("#4f46e5") if self.isChecked() else QColor("#9aa8c5")
        painter.setPen(QPen(border, 1.4))
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(indicator.adjusted(1, 1, -1, -1))
        if self.isChecked():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#4f46e5"))
            painter.drawEllipse(indicator.adjusted(5, 5, -5, -5))
        painter.setPen(QPen(QColor("#24324d")))
        painter.drawText(
            QRect(26, 0, rect.width() - 26, rect.height()),
            Qt.AlignmentFlag.AlignVCenter,
            self.text(),
        )


class PaneSplitterHandle(QSplitterHandle):
    def __init__(self, orientation: Qt.Orientation, parent: QSplitter) -> None:
        super().__init__(orientation, parent)
        self.setCursor(Qt.CursorShape.SplitHCursor)


class PaneSplitter(QSplitter):
    def __init__(self) -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self.setHandleWidth(10)
        self.setChildrenCollapsible(False)

    def createHandle(self) -> QSplitterHandle:  # noqa: N802
        return PaneSplitterHandle(self.orientation(), self)


class ModelOptionRow(QFrame):
    def __init__(self, model: str, note: str) -> None:
        super().__init__()
        self.setObjectName("modelCard")
        self.checkbox = IndicatorCheckBox(model)
        note_label = QLabel(note)
        note_label.setObjectName("modelNote")
        note_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)
        layout.addWidget(self.checkbox)
        layout.addWidget(note_label)

    def is_checked(self) -> bool:
        return self.checkbox.isChecked()

    def set_checked(self, checked: bool) -> None:
        self.checkbox.setChecked(checked)


class WrappedTabChip(QFrame):
    activated = Signal(object)
    close_requested = Signal(object)

    def __init__(self, text: str) -> None:
        super().__init__()
        self._full_text = text
        self._selected = False
        self._color = QColor("black")
        self._label = QPushButton()
        self._label.setObjectName("chipButton")
        self._label.clicked.connect(lambda: self.activated.emit(self))
        self._close = QToolButton()
        self._close.setText("×")
        self._close.clicked.connect(lambda: self.close_requested.emit(self))
        self._close.setAutoRaise(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(self._label)
        layout.addWidget(self._close)

        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setMaximumWidth(TAB_CHIP_MAX_WIDTH + 40)
        self.setMinimumWidth(120)
        self.set_text(text)
        self._refresh_style()

    def sizeHint(self) -> QSize:
        return QSize(TAB_CHIP_MAX_WIDTH + 20, 42)

    def set_text(self, text: str) -> None:
        self._full_text = text
        self.setToolTip(text)
        self._update_text()

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._refresh_style()

    def set_color(self, color: QColor) -> None:
        self._color = color
        self._refresh_style()

    def resizeEvent(self, event: object) -> None:
        self._update_text()
        super().resizeEvent(event)

    def _update_text(self) -> None:
        available = max(60, min(TAB_CHIP_MAX_WIDTH, self.width() - 42))
        self._label.setText(
            self.fontMetrics().elidedText(
                self._full_text,
                Qt.TextElideMode.ElideRight,
                available,
            )
        )

    def _refresh_style(self) -> None:
        fg = self._color.name()
        bg = "#eef4ff" if self._selected else "#ffffff"
        border = "#7aa2ff" if self._selected else "#d0d5dd"
        self.setStyleSheet(
            f"""
            QFrame {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 12px;
            }}
            QPushButton {{
                color: {fg};
                font-size: 14px;
                font-weight: 600;
                background: transparent;
                border: none;
                padding: 0;
                text-align: left;
            }}
            QToolButton {{
                color: {fg};
                background: transparent;
                border: none;
                padding: 0 4px;
                font-size: 14px;
            }}
            """
        )


class WrappedTabs(QWidget):
    tabCloseRequested = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self._entries: list[dict[str, object]] = []

        tab_card = QFrame()
        tab_card.setObjectName("tabStripCard")
        tab_layout = QVBoxLayout(tab_card)
        tab_layout.setContentsMargins(10, 10, 10, 10)
        tab_layout.setSpacing(0)

        self._tab_hint = QLabel("No images open yet. Open files or drop them here.")
        self._tab_hint.setObjectName("mutedText")
        self._tab_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._tab_row = QWidget()
        self._flow = FlowLayout(self._tab_row, spacing=8)
        tab_layout.addWidget(self._tab_hint)
        tab_layout.addWidget(self._tab_row)

        self._stack = QStackedWidget()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(tab_card)
        layout.addWidget(self._stack, 1)
        self._update_empty_state()

    def addTab(self, widget: QWidget, text: str) -> int:  # noqa: N802
        chip = WrappedTabChip(text)
        chip.activated.connect(self._activate_chip)
        chip.close_requested.connect(self._request_close_chip)
        self._entries.append(
            {"widget": widget, "chip": chip, "text": text, "color": QColor("black")}
        )
        self._flow.addWidget(chip)
        self._stack.addWidget(widget)
        if len(self._entries) == 1:
            self.setCurrentWidget(widget)
        self._update_empty_state()
        return len(self._entries) - 1

    def count(self) -> int:
        return len(self._entries)

    def widget(self, index: int) -> QWidget | None:
        if 0 <= index < len(self._entries):
            return self._entries[index]["widget"]  # type: ignore[return-value]
        return None

    def indexOf(self, widget: QWidget) -> int:  # noqa: N802
        for index, entry in enumerate(self._entries):
            if entry["widget"] is widget:
                return index
        return -1

    def setCurrentWidget(self, widget: QWidget) -> None:  # noqa: N802
        self._stack.setCurrentWidget(widget)
        self._sync_selection()

    def removeTab(self, index: int) -> None:  # noqa: N802
        if not 0 <= index < len(self._entries):
            return
        entry = self._entries.pop(index)
        widget = entry["widget"]
        chip = entry["chip"]
        self._stack.removeWidget(widget)  # type: ignore[arg-type]
        self._flow.removeWidget(chip)  # type: ignore[arg-type]
        chip.deleteLater()  # type: ignore[union-attr]
        if self._entries:
            next_index = min(index, len(self._entries) - 1)
            self.setCurrentWidget(self._entries[next_index]["widget"])  # type: ignore[arg-type]
        self._update_empty_state()

    def setTabText(self, index: int, text: str) -> None:  # noqa: N802
        if not 0 <= index < len(self._entries):
            return
        self._entries[index]["text"] = text
        chip = self._entries[index]["chip"]
        chip.set_text(text)  # type: ignore[union-attr]

    def setTabToolTip(self, index: int, text: str) -> None:  # noqa: N802
        if not 0 <= index < len(self._entries):
            return
        chip = self._entries[index]["chip"]
        chip.setToolTip(text)  # type: ignore[union-attr]

    def setTabTextColor(self, index: int, color: QColor) -> None:  # noqa: N802
        if not 0 <= index < len(self._entries):
            return
        self._entries[index]["color"] = color
        chip = self._entries[index]["chip"]
        chip.set_color(color)  # type: ignore[union-attr]

    def _activate_chip(self, chip: WrappedTabChip) -> None:
        for entry in self._entries:
            if entry["chip"] is chip:
                self.setCurrentWidget(entry["widget"])  # type: ignore[arg-type]
                return

    def _request_close_chip(self, chip: WrappedTabChip) -> None:
        for index, entry in enumerate(self._entries):
            if entry["chip"] is chip:
                self.tabCloseRequested.emit(index)
                return

    def _sync_selection(self) -> None:
        current = self._stack.currentWidget()
        for entry in self._entries:
            chip = entry["chip"]
            chip.set_selected(entry["widget"] is current)  # type: ignore[union-attr]
            chip.set_color(entry["color"])  # type: ignore[union-attr]

    def _update_empty_state(self) -> None:
        has_tabs = bool(self._entries)
        self._tab_row.setVisible(has_tabs)
        self._tab_hint.setVisible(not has_tabs)


class ImageTab(QWidget):
    enqueue_requested = Signal(object, object, int)
    retry_requested = Signal(object)
    cancel_requested = Signal(object)
    left_width_changed = Signal(int)

    def __init__(
        self,
        input_path: Path,
        *,
        defaults: AdvancedSettings,
        left_pane_width: int,
    ) -> None:
        super().__init__()
        self.input_path = input_path
        self.jobs: list[Job] = []
        self._rows_by_job: dict[int, int] = {}
        self._default_advanced = defaults
        self._advanced_initialized = False
        self._input_size = _safe_image_size(input_path)

        self.model_rows: dict[str, ModelOptionRow] = {}
        for model in UPSCALE_MODELS:
            row = ModelOptionRow(model, MODEL_DESCRIPTIONS.get(model, ""))
            self.model_rows[model] = row
            row.checkbox.toggled.connect(self._update_action_buttons)

        self.scale_group = QButtonGroup(self)
        self.scale_2 = IndicatorRadioButton("2x")
        self.scale_4 = IndicatorRadioButton("4x")
        self.scale_4.setChecked(True)
        self.scale_group.addButton(self.scale_2)
        self.scale_group.addButton(self.scale_4)

        self.enqueue_button = QPushButton("Enqueue selected")
        self.enqueue_button.setObjectName("primaryButton")
        self.enqueue_button.setEnabled(False)
        self.enqueue_button.clicked.connect(self._enqueue_selected)
        self.enqueue_all_button = QPushButton("Enqueue all models")
        self.enqueue_all_button.clicked.connect(self._enqueue_all_models)
        self.retry_button = QPushButton("Retry failed")
        self.retry_button.setEnabled(False)
        self.retry_button.clicked.connect(lambda: self.retry_requested.emit(self))
        self.cancel_button = QPushButton("Cancel queue")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(lambda: self.cancel_requested.emit(self))

        actions = QWidget()
        actions_layout = FlowLayout(actions, spacing=8)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.addWidget(self.enqueue_button)
        actions_layout.addWidget(self.enqueue_all_button)
        actions_layout.addWidget(self.retry_button)
        actions_layout.addWidget(self.cancel_button)

        left_card = QFrame()
        left_card.setObjectName("panelCard")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)

        self.name_label = _elided_label(
            self.input_path.name,
            mode=Qt.TextElideMode.ElideRight,
            object_name="sectionTitle",
        )
        self.dir_label = _elided_label(
            str(self.input_path.parent),
            mode=Qt.TextElideMode.ElideMiddle,
            object_name="pathText",
        )
        left_layout.addWidget(self.name_label)
        left_layout.addWidget(self.dir_label)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(12)

        controls_layout.addWidget(_section_label("Models"))
        models_box = QWidget()
        models_box_layout = QVBoxLayout(models_box)
        models_box_layout.setContentsMargins(0, 0, 0, 0)
        models_box_layout.setSpacing(8)
        for row in self.model_rows.values():
            models_box_layout.addWidget(row)
        controls_layout.addWidget(models_box)

        scale_row = QWidget()
        scale_layout = QHBoxLayout(scale_row)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.setSpacing(10)
        scale_layout.addWidget(_section_label("Scale"))
        scale_layout.addWidget(self.scale_2)
        scale_layout.addWidget(self.scale_4)
        scale_layout.addStretch()
        controls_layout.addWidget(scale_row)

        self.advanced_box = QCollapsible(
            "Advanced options - defaults",
            expandedIcon="▾",
            collapsedIcon="▸",
        )
        self.advanced_box.setObjectName("sectionCard")
        self.advanced_box.layout().setContentsMargins(12, 6, 12, 4)
        self.advanced_box.layout().setSpacing(4)
        self.advanced_box.toggleButton().setObjectName("advancedToggle")
        self.advanced_box.toggleButton().setToolTip("Show or hide advanced options.")
        self.advanced_box.toggled.connect(self._on_advanced_toggled)

        advanced_content = QWidget()
        advanced_content_layout = QVBoxLayout(advanced_content)
        advanced_content_layout.setContentsMargins(0, 0, 0, 0)
        advanced_content_layout.setSpacing(0)
        advanced_content_layout.addLayout(self._build_advanced_form())
        self.advanced_box.addWidget(advanced_content)
        self.advanced_box.collapse()
        controls_layout.addWidget(self.advanced_box)
        controls_layout.addWidget(actions)
        controls_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(controls)
        left_layout.addWidget(left_scroll, 1)

        preview_card = QFrame()
        preview_card.setObjectName("panelCard")
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(16, 16, 16, 16)
        preview_layout.setSpacing(8)
        preview_layout.addWidget(_section_label("Original image"))
        self.preview_meta = _muted_label(_preview_meta_text(self._input_size))
        preview_layout.addWidget(self.preview_meta)
        self.preview = PreviewLabel()
        self.preview.set_image_path(self.input_path)
        preview_layout.addWidget(self.preview, 1)

        queue_card = QFrame()
        queue_card.setObjectName("panelCard")
        queue_layout = QVBoxLayout(queue_card)
        queue_layout.setContentsMargins(16, 16, 16, 16)
        queue_layout.setSpacing(8)
        queue_layout.addWidget(_section_label("Queue"))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Model", "Scale", "Output", "Status"])
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        widths = _queue_column_widths(self.table.fontMetrics())
        self.table.setColumnWidth(0, widths["model"])
        self.table.setColumnWidth(1, widths["scale"])
        self.table.setColumnWidth(3, widths["status"])
        self.table.setAlternatingRowColors(True)
        self.table.setMaximumHeight(150)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(30)
        queue_layout.addWidget(self.table)

        right_stack = QWidget()
        right_layout = QVBoxLayout(right_stack)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        right_layout.addWidget(preview_card, 1)
        right_layout.addWidget(queue_card, 0)

        self.splitter = PaneSplitter()
        self.splitter.addWidget(left_card)
        self.splitter.addWidget(right_stack)
        self.splitter.setSizes([left_pane_width, max(500, 1100 - left_pane_width)])
        self.splitter.splitterMoved.connect(self._emit_left_width_changed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)

        self.set_left_pane_width(left_pane_width)
        self.set_default_advanced(defaults)
        self._update_action_buttons()

    def add_jobs(self, jobs: list[Job]) -> None:
        for job in jobs:
            self.jobs.append(job)
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._rows_by_job[job.id] = row
            self.table.setItem(row, 0, _item(job.model, tooltip=job.model))
            self.table.setItem(row, 1, _item(f"{job.scale}x"))
            self.table.setItem(row, 2, _item(job.output_path.name, tooltip=str(job.output_path)))
            self.table.setItem(row, 3, _item(_status_text(job.status)))
        self._update_action_buttons()

    def update_job(self, job: Job) -> None:
        row = self._rows_by_job[job.id]
        self.table.item(row, 2).setText(job.output_path.name)
        self.table.item(row, 2).setToolTip(str(job.output_path))
        status_text = job.message or _status_text(job.status)
        if job.status == "cancelling":
            status_text = _status_text("cancelling")
        self.table.item(row, 3).setText(status_text)
        tooltip = "\n".join(job.warnings) if job.warnings else status_text
        self.table.item(row, 3).setToolTip(tooltip)
        color = {
            "pending": QColor("#eef5ff"),
            "running": QColor("#fff4cc"),
            "succeeded": QColor("#e8f7e8"),
            "failed": QColor("#ffe8e8"),
            "cancelling": QColor("#fff4cc"),
            "cancelled": QColor("#f0f0f0"),
        }.get(job.status, QColor("white"))
        for column in range(self.table.columnCount()):
            self.table.item(row, column).setBackground(color)
        self._update_action_buttons()

    def counts(self) -> tuple[int, int, int, int]:
        total = len(self.jobs)
        done = sum(1 for job in self.jobs if job.status == "succeeded")
        failed = sum(1 for job in self.jobs if job.status == "failed")
        running = sum(1 for job in self.jobs if job.status in {"running", "cancelling"})
        return total, done, failed, running

    def all_succeeded(self) -> bool:
        return bool(self.jobs) and all(job.status == "succeeded" for job in self.jobs)

    def has_active_jobs(self) -> bool:
        return any(job.status in {"pending", "running", "cancelling"} for job in self.jobs)

    def has_cancellable_jobs(self) -> bool:
        return any(job.status in {"pending", "running"} for job in self.jobs)

    def current_scale(self) -> int:
        return 2 if self.scale_2.isChecked() else 4

    def current_advanced(self) -> AdvancedSettings:
        profile = self.target_profile.currentData()
        return AdvancedSettings(
            face_enhance=self.face_enhance.isChecked(),
            denoise_strength=self.denoise_strength.value(),
            alpha_mode=self.alpha_mode.currentData(),
            device=self.device.currentData(),
            output_format=_coerce_output_format(self.output_format.currentData()),
            quality=self.quality.value(),
            tile=self.tile.value(),
            strip_metadata=self.strip_metadata.isChecked(),
            target_profile=profile,
        )

    def set_default_advanced(self, defaults: AdvancedSettings) -> None:
        if not self._advanced_initialized:
            self._default_advanced = defaults
            self._apply_advanced(defaults)
            self._advanced_initialized = True
            return
        current = self.current_advanced()
        old_defaults = self._default_advanced
        self._default_advanced = defaults
        if current == old_defaults:
            self._apply_advanced(defaults)
        self._update_advanced_summary()

    def set_left_pane_width(self, width: int) -> None:
        width = max(LEFT_PANE_MIN_WIDTH, width)
        total = max(sum(self.splitter.sizes()), width + 500)
        self.splitter.blockSignals(True)
        self.splitter.setSizes([width, max(500, total - width)])
        self.splitter.blockSignals(False)

    def _build_advanced_form(self) -> QFormLayout:
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)

        self.face_enhance = IndicatorCheckBox("Enabled")
        self.face_enhance.toggled.connect(self._update_advanced_summary)

        self.denoise_strength = QDoubleSpinBox()
        self.denoise_strength.setRange(0.0, 1.0)
        self.denoise_strength.setSingleStep(0.1)
        self.denoise_strength.setDecimals(2)
        self.denoise_strength.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.denoise_strength.valueChanged.connect(self._update_advanced_summary)

        self.alpha_mode = QComboBox()
        self.alpha_mode.addItem("Real-ESRGAN", "realesrgan")
        self.alpha_mode.addItem("Bicubic", "bicubic")
        self.alpha_mode.currentIndexChanged.connect(self._update_advanced_summary)

        self.output_format = QComboBox()
        for fmt in OutputFormat:
            self.output_format.addItem(fmt.value.upper(), fmt.value)
        self.output_format.currentIndexChanged.connect(self._update_advanced_summary)

        self.quality = QSpinBox()
        self.quality.setRange(0, 100)
        self.quality.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.quality.valueChanged.connect(self._update_advanced_summary)

        self.tile = QSpinBox()
        self.tile.setRange(0, 4096)
        self.tile.setSingleStep(64)
        self.tile.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.tile.valueChanged.connect(self._update_advanced_summary)

        self.device = QComboBox()
        self.device.addItem("Auto", "auto")
        self.device.addItem("MPS", "mps")
        self.device.addItem("CUDA", "cuda")
        self.device.addItem("CPU", "cpu")
        self.device.currentIndexChanged.connect(self._update_advanced_summary)

        self.strip_metadata = IndicatorCheckBox("Enabled")
        self.strip_metadata.toggled.connect(self._update_advanced_summary)

        self.target_profile = QComboBox()
        self.target_profile.addItem("Default", None)
        self.target_profile.addItem("sRGB", "srgb")
        self.target_profile.addItem("Display P3", "p3")
        self.target_profile.addItem("Adobe RGB", "adobergb")
        self.target_profile.currentIndexChanged.connect(self._update_advanced_summary)

        restore = QPushButton("Restore defaults")
        restore.clicked.connect(self._restore_advanced_defaults)

        form.addRow("Face enhancement", self.face_enhance)
        form.addRow("Denoise", self.denoise_strength)
        form.addRow("", _muted_label("Only affects realesr-general-x4v3."))
        form.addRow("Alpha mode", self.alpha_mode)
        form.addRow("Output format", self.output_format)
        form.addRow("Quality", self.quality)
        form.addRow("", _muted_label("Used for JPG and WebP. Ignored for PNG."))
        form.addRow("Tile size", self.tile)
        form.addRow("", _muted_label("0 uses the whole image. Raise this only if memory is tight."))
        form.addRow("Device", self.device)
        form.addRow("", _muted_label("Auto is best for most people."))
        form.addRow("Strip metadata", self.strip_metadata)
        form.addRow("Target profile", self.target_profile)
        form.addRow("", _button_row(restore))
        return form

    def _apply_advanced(self, settings: AdvancedSettings) -> None:
        self.face_enhance.setChecked(settings.face_enhance)
        self.denoise_strength.setValue(settings.denoise_strength)
        self.alpha_mode.setCurrentIndex(self.alpha_mode.findData(settings.alpha_mode))
        self.output_format.setCurrentIndex(
            self.output_format.findData(_coerce_output_format(settings.output_format).value)
        )
        self.quality.setValue(settings.quality)
        self.tile.setValue(settings.tile)
        self.device.setCurrentIndex(self.device.findData(settings.device))
        self.strip_metadata.setChecked(settings.strip_metadata)
        self.target_profile.setCurrentIndex(self.target_profile.findData(settings.target_profile))
        self._update_advanced_summary()

    def _restore_advanced_defaults(self) -> None:
        self._apply_advanced(self._default_advanced)
        LOGGER.info(
            "Restored advanced defaults input=%s defaults=%s",
            self.input_path,
            _advanced_log_payload(self._default_advanced),
        )

    def _on_advanced_toggled(self, expanded: bool) -> None:
        LOGGER.info(
            "Advanced options %s input=%s current=%s",
            "expanded" if expanded else "collapsed",
            self.input_path,
            _advanced_log_payload(self.current_advanced()),
        )

    def _update_advanced_summary(self) -> None:
        modified = self.current_advanced() != self._default_advanced
        toggle = self.advanced_box.toggleButton()
        toggle.setProperty("modified", modified)
        toggle.setText(f"Advanced options - {'modified' if modified else 'defaults'}")
        toggle.style().unpolish(toggle)
        toggle.style().polish(toggle)

    def _emit_left_width_changed(self, position: int, index: int) -> None:
        del position, index
        self.left_width_changed.emit(self.splitter.sizes()[0])

    def _selected_models(self) -> list[str]:
        return [name for name, row in self.model_rows.items() if row.is_checked()]

    def _update_action_buttons(self) -> None:
        self.enqueue_button.setEnabled(bool(self._selected_models()))
        self.retry_button.setEnabled(any(job.status == "failed" for job in self.jobs))
        self.cancel_button.setEnabled(self.has_cancellable_jobs())

    def _enqueue_selected(self) -> None:
        models = self._selected_models()
        if not models:
            self._update_action_buttons()
            return
        self.enqueue_requested.emit(self, models, self.current_scale())

    def _enqueue_all_models(self) -> None:
        self.enqueue_requested.emit(self, list(UPSCALE_MODELS), self.current_scale())


class MainWindow(QMainWindow):
    def __init__(self, *, log_file: Path) -> None:
        super().__init__()
        self.config = load_app_config()
        self.log_file = log_file
        self._job_ids = count(1)
        self._threads: dict[int, tuple[QThread, JobWorker]] = {}
        self._active_jobs = 0
        self._tabs_by_path: dict[Path, ImageTab] = {}
        self._left_pane_width = LEFT_PANE_START_WIDTH

        self._session_shutdown = False
        self.setWindowTitle("PixelUp")
        self.resize(1260, 860)
        self.setAcceptDrops(True)
        self._build_ui()
        app = QApplication.instance()
        if app is not None:
            commit = getattr(app, "commitDataRequest", None)
            if commit is not None:
                commit.connect(self._on_commit_data_request)
        LOGGER.info(
            "Loaded config path=%s values=%s",
            CONFIG_PATH,
            _config_log_payload(self.config),
        )

    def _on_commit_data_request(self, _manager: object) -> None:
        self._session_shutdown = True

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._session_shutdown or QGuiApplication.isSavingSession():
            LOGGER.info("Accepting close during session shutdown")
            self._cleanup_workers_for_quit()
            event.accept()
            return
        if not self._tabs_by_path:
            self._cleanup_workers_for_quit()
            event.accept()
            return
        running = sum(1 for tab in self._tabs_by_path.values() if tab.has_active_jobs())
        if running:
            text = (
                f"PixelUp has {running} tab(s) with running or pending jobs. "
                "Quit and abandon them?"
            )
        else:
            text = "PixelUp has open images. Quit anyway?"
        choice = QMessageBox.question(
            self,
            "Quit PixelUp?",
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice == QMessageBox.StandardButton.Yes:
            LOGGER.info("User confirmed quit with open tabs running=%s", running)
            self._cleanup_workers_for_quit()
            event.accept()
        else:
            LOGGER.info("User cancelled quit")
            event.ignore()

    def _cleanup_workers_for_quit(self) -> None:
        if not self._threads:
            return
        entries = list(self._threads.values())
        for _thread, worker in entries:
            try:
                worker.request_cancel()
            except Exception:
                pass
            for signal in (worker.signals.progress, worker.signals.finished):
                try:
                    signal.disconnect()
                except (RuntimeError, TypeError):
                    pass
        for thread, _worker in entries:
            try:
                thread.wait(2000)
            except Exception:
                pass
        LOGGER.info("Cleaned up %s worker thread(s) for quit", len(entries))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        self.open_paths(paths)

    def open_paths(self, paths: list[Path]) -> None:
        LOGGER.info("Open requested paths=%s", [str(path) for path in paths])
        for path in paths:
            resolved = path.expanduser().resolve()
            if not resolved.is_file():
                LOGGER.warning("Ignored non-file open request path=%s resolved=%s", path, resolved)
                continue
            tab = self._tabs_by_path.get(resolved)
            if tab is None:
                tab = ImageTab(
                    resolved,
                    defaults=_advanced_defaults(self.config),
                    left_pane_width=self._left_pane_width,
                )
                tab.enqueue_requested.connect(self._enqueue_jobs)
                tab.retry_requested.connect(self._retry_failed)
                tab.cancel_requested.connect(self._cancel_queue)
                tab.left_width_changed.connect(self._sync_left_width)
                self._tabs_by_path[resolved] = tab
                self.tabs.addTab(tab, resolved.name)
                LOGGER.info(
                    "Created tab input=%s left_pane_width=%s defaults=%s",
                    resolved,
                    self._left_pane_width,
                    _advanced_log_payload(_advanced_defaults(self.config)),
                )
            else:
                LOGGER.info("Focused existing tab input=%s", resolved)
            self.tabs.setCurrentWidget(tab)
            self._update_tab_state(tab)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        root_layout.addWidget(self._build_header())

        self.tabs = WrappedTabs()
        self.tabs.tabCloseRequested.connect(self._close_tab_at)
        root_layout.addWidget(self.tabs, 1)

        self.setCentralWidget(root)

    def _build_header(self) -> QWidget:
        card = QFrame()
        card.setObjectName("topBarCard")

        layout = QHBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        title = QLabel("PixelUp")
        title.setObjectName("windowTitle")

        open_button = QPushButton("Open images…")
        open_button.setObjectName("primaryButton")
        open_button.clicked.connect(self._open_dialog)

        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self._settings_dialog)
        about_button = QPushButton("About")
        about_button.clicked.connect(self._about_dialog)
        logs_button = QPushButton("Reveal log")
        logs_button.clicked.connect(self._reveal_log_file)

        action_row = QWidget()
        action_layout = QHBoxLayout(action_row)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(8)
        action_layout.addWidget(open_button)
        action_layout.addWidget(logs_button)
        action_layout.addWidget(settings_button)
        action_layout.addWidget(about_button)

        layout.addWidget(title, 1, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(action_row, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return card

    def _open_dialog(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Open images")
        LOGGER.info("Open dialog returned count=%s", len(files))
        self.open_paths([Path(file) for file in files])

    def _settings_dialog(self) -> None:
        LOGGER.info("Opened settings dialog")
        dialog = SettingsDialog(self.config, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            previous_config = self.config
            previous_defaults = _advanced_defaults(self.config)
            self.config = dialog.config()
            save_app_config(self.config)
            LOGGER.info(
                "Saved settings path=%s previous=%s current=%s",
                CONFIG_PATH,
                _config_log_payload(previous_config),
                _config_log_payload(self.config),
            )
            for tab in self._tabs_by_path.values():
                if tab.current_advanced() == previous_defaults:
                    tab.set_default_advanced(_advanced_defaults(self.config))
                else:
                    tab._default_advanced = _advanced_defaults(self.config)
                    tab._update_advanced_summary()
            self._schedule()
            return
        LOGGER.info("Closed settings dialog without saving")

    def _about_dialog(self) -> None:
        LOGGER.info("Opened about dialog")
        AboutDialog(self).exec()

    def _reveal_log_file(self) -> None:
        _reveal_in_file_browser(self.log_file)
        LOGGER.info("Revealed log file %s", self.log_file)

    @Slot(object, object, int)
    def _enqueue_jobs(self, tab: ImageTab, models: list[str], scale: int) -> None:
        if not models:
            LOGGER.warning(
                "Rejected enqueue request input=%s reason=no-models-selected",
                tab.input_path,
            )
            QMessageBox.information(self, "PixelUp", "Choose at least one model.")
            return
        advanced = tab.current_advanced()
        jobs: list[Job] = []
        reserved = {job.output_path for job in tab.jobs}
        for model in models:
            model_advanced = _advanced_for_model(advanced, model)
            output_path = default_output_path(
                tab.input_path,
                model=model,
                scale=scale,
                output_format=model_advanced.output_format,
                reserved=reserved,
            )
            reserved.add(output_path)
            jobs.append(
                Job(
                    id=next(self._job_ids),
                    input_path=tab.input_path,
                    model=model,
                    scale=scale,
                    output_path=output_path,
                    advanced=model_advanced,
                    auto_download=self.config.auto_download,
                )
            )
        LOGGER.info(
            "Accepted enqueue request input=%s models=%s scale=%s advanced=%s auto_download=%s",
            tab.input_path,
            models,
            scale,
            _advanced_log_payload(advanced),
            self.config.auto_download,
        )
        for job in jobs:
            LOGGER.info("Queued job %s details=%s", job.id, _job_log_payload(job))
        tab.add_jobs(jobs)
        self._update_tab_state(tab)
        self._schedule()

    @Slot(object)
    def _cancel_queue(self, tab: ImageTab) -> None:
        cancelled_pending: list[int] = []
        signalled_running: list[int] = []
        for job in tab.jobs:
            if job.status == "pending":
                job.status = "cancelled"
                job.message = "Cancelled"
                tab.update_job(job)
                cancelled_pending.append(job.id)
            elif job.status == "running":
                entry = self._threads.get(job.id)
                if entry is not None:
                    _, worker = entry
                    worker.request_cancel()
                job.status = "cancelling"
                tab.update_job(job)
                signalled_running.append(job.id)
        LOGGER.info(
            "Cancel queue input=%s cancelled_pending=%s signalled_running=%s",
            tab.input_path,
            cancelled_pending,
            signalled_running,
        )
        self._update_tab_state(tab)

    @Slot(object)
    def _retry_failed(self, tab: ImageTab) -> None:
        reserved = {job.output_path for job in tab.jobs if job.status != "failed"}
        changed = False
        retried_jobs: list[int] = []
        for job in tab.jobs:
            if job.status != "failed":
                continue
            job.output_path = default_output_path(
                tab.input_path,
                model=job.model,
                scale=job.scale,
                output_format=job.advanced.output_format,
                reserved=reserved,
            )
            reserved.add(job.output_path)
            job.status = "pending"
            job.message = ""
            tab.update_job(job)
            changed = True
            retried_jobs.append(job.id)
        if changed:
            LOGGER.info("Retrying failed jobs input=%s job_ids=%s", tab.input_path, retried_jobs)
            self._update_tab_state(tab)
            self._schedule()

    @Slot(int)
    def _sync_left_width(self, width: int) -> None:
        self._left_pane_width = max(LEFT_PANE_MIN_WIDTH, width)
        for tab in self._tabs_by_path.values():
            tab.set_left_pane_width(self._left_pane_width)

    def _schedule(self) -> None:
        limit = max(1, self.config.max_concurrent_jobs)
        while self._active_jobs < limit:
            next_job = self._next_pending_job()
            if next_job is None:
                return
            tab, job = next_job
            self._start_job(tab, job)

    def _next_pending_job(self) -> tuple[ImageTab, Job] | None:
        for index in range(self.tabs.count()):
            tab = self.tabs.widget(index)
            if not isinstance(tab, ImageTab):
                continue
            for job in tab.jobs:
                if job.status == "pending":
                    return tab, job
        return None

    def _start_job(self, tab: ImageTab, job: Job) -> None:
        job.status = "running"
        job.message = "Starting"
        tab.update_job(job)
        self._update_tab_state(tab)
        self._active_jobs += 1

        thread = QThread(self)
        worker = JobWorker(job)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.signals.progress.connect(self._job_progress)
        worker.signals.finished.connect(self._job_finished)
        worker.signals.finished.connect(thread.quit)
        worker.signals.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda job_id=job.id: self._threads.pop(job_id, None))
        self._threads[job.id] = (thread, worker)
        thread.start()

    @Slot(int, str)
    def _job_progress(self, job_id: int, message: str) -> None:
        tab, job = self._find_job(job_id)
        if message != job.message:
            LOGGER.info(
                "Job %s progress input=%s model=%s output=%s message=%s",
                job.id,
                tab.input_path,
                job.model,
                job.output_path,
                message,
            )
        job.message = message
        tab.update_job(job)

    @Slot(int, bool, str, object, object)
    def _job_finished(
        self,
        job_id: int,
        ok: bool,
        message: str,
        result: object,
        warnings: object,
    ) -> None:
        tab, job = self._find_job(job_id)
        cancelled = isinstance(result, dict) and bool(result.get("cancelled"))
        if ok:
            job.status = "succeeded"
        elif cancelled:
            job.status = "cancelled"
        else:
            job.status = "failed"
        job.message = message
        job.warnings = list(warnings) if isinstance(warnings, list) else []
        tab.update_job(job)
        self._active_jobs = max(0, self._active_jobs - 1)
        self._update_tab_state(tab)
        if self.config.close_tab_on_success and tab.all_succeeded():
            self._close_tab(tab, reason="all-jobs-succeeded")
        QTimer.singleShot(0, self._schedule)

    def _find_job(self, job_id: int) -> tuple[ImageTab, Job]:
        for tab in self._tabs_by_path.values():
            for job in tab.jobs:
                if job.id == job_id:
                    return tab, job
        raise RuntimeError(f"Unknown job id: {job_id}")

    def _update_tab_state(self, tab: ImageTab) -> None:
        index = self.tabs.indexOf(tab)
        if index < 0:
            return
        total, done, failed, running = tab.counts()
        self.tabs.setTabText(index, tab.input_path.name)
        self.tabs.setTabToolTip(
            index,
            f"{tab.input_path}\n{done}/{total} done, {failed} failed, {running} running",
        )
        if failed:
            color = QColor("#b00020")
        elif running or any(job.status == "pending" for job in tab.jobs):
            color = QColor("#2563eb")
        elif total and done == total:
            color = QColor("#15803d")
        else:
            color = QColor("#111827")
        self.tabs.setTabTextColor(index, color)

    def _close_tab_at(self, index: int) -> None:
        tab = self.tabs.widget(index)
        if isinstance(tab, ImageTab) and tab.has_active_jobs():
            LOGGER.info("Blocked tab close input=%s reason=active-jobs", tab.input_path)
            QMessageBox.information(self, "PixelUp", "This tab still has pending or running jobs.")
            return
        if isinstance(tab, ImageTab):
            self._close_tab(tab, reason="user")

    def _close_tab(self, tab: ImageTab, *, reason: str) -> None:
        index = self.tabs.indexOf(tab)
        if index >= 0:
            self.tabs.removeTab(index)
        self._tabs_by_path.pop(tab.input_path, None)
        LOGGER.info("Closed tab input=%s reason=%s", tab.input_path, reason)
        tab.deleteLater()


class SettingsDialog(QDialog):
    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)

        root = QWidget()
        root.setObjectName("dialogRoot")
        root.setMinimumWidth(540)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("Settings")
        title.setObjectName("windowTitle")
        subtitle = QLabel("Default behavior for new jobs and tabs.")
        subtitle.setObjectName("dialogSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        general_card = _section_card("General")
        general_form = QFormLayout()
        general_form.setSpacing(10)

        self.concurrent = QSpinBox()
        self.concurrent.setRange(1, 8)
        self.concurrent.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.concurrent.setValue(config.max_concurrent_jobs)

        self.close_tabs = IndicatorCheckBox("Close tabs after all jobs succeed")
        self.close_tabs.setChecked(config.close_tab_on_success)

        self.auto_download = IndicatorCheckBox("Download missing models automatically")
        self.auto_download.setChecked(config.auto_download)

        general_form.addRow("Concurrent jobs", self.concurrent)
        general_form.addRow("", self.close_tabs)
        general_form.addRow("", self.auto_download)
        general_card.layout().addLayout(general_form)

        defaults_card = _section_card("Advanced defaults")
        defaults_form = QFormLayout()
        defaults_form.setSpacing(10)

        self.format = QComboBox()
        self.format.addItems([item.value.upper() for item in OutputFormat])
        self.format.setCurrentText(config.output_format.value.upper())

        self.quality = QSpinBox()
        self.quality.setRange(0, 100)
        self.quality.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.quality.setValue(config.quality)

        self.tile = QSpinBox()
        self.tile.setRange(0, 4096)
        self.tile.setSingleStep(64)
        self.tile.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.tile.setValue(config.tile)

        self.device = QComboBox()
        self.device.addItem("Auto", "auto")
        self.device.addItem("MPS", "mps")
        self.device.addItem("CUDA", "cuda")
        self.device.addItem("CPU", "cpu")
        self.device.setCurrentIndex(self.device.findData(config.device))

        defaults_form.addRow("Output format", self.format)
        defaults_form.addRow("Quality", self.quality)
        defaults_form.addRow("", _muted_label("Used for JPG and WebP. Ignored for PNG."))
        defaults_form.addRow(_field_label("Tile size", self._show_tile_help), self.tile)
        defaults_form.addRow(
            "",
            _muted_label("0 uses the whole image. Raise this only if memory is tight."),
        )
        defaults_form.addRow(_field_label("Device", self._show_device_help), self.device)
        defaults_form.addRow("", _muted_label("Auto is best for most people."))
        defaults_card.layout().addLayout(defaults_form)

        layout.addWidget(general_card)
        layout.addWidget(defaults_card)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        restore = buttons.addButton(
            "Restore defaults",
            QDialogButtonBox.ButtonRole.ResetRole,
        )
        restore.clicked.connect(self._restore_defaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        shell = QVBoxLayout(self)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        shell.addWidget(root)

    def config(self) -> AppConfig:
        return AppConfig(
            max_concurrent_jobs=self.concurrent.value(),
            close_tab_on_success=self.close_tabs.isChecked(),
            output_format=OutputFormat(self.format.currentText().lower()),
            quality=self.quality.value(),
            tile=self.tile.value(),
            device=self.device.currentData(),
            auto_download=self.auto_download.isChecked(),
        )

    def _restore_defaults(self) -> None:
        defaults = AppConfig()
        self.concurrent.setValue(defaults.max_concurrent_jobs)
        self.close_tabs.setChecked(defaults.close_tab_on_success)
        self.auto_download.setChecked(defaults.auto_download)
        self.format.setCurrentText(defaults.output_format.value.upper())
        self.quality.setValue(defaults.quality)
        self.tile.setValue(defaults.tile)
        self.device.setCurrentIndex(self.device.findData(defaults.device))

    def _show_tile_help(self) -> None:
        HelpDialog(
            "Tile size",
            [
                "Tile size controls whether PixelUp splits a large image into smaller pieces "
                "before running the AI model.",
                "0 means no tiling: PixelUp processes the whole image at once. This is usually "
                "best because it avoids seams between tiles.",
                "If an upscale fails because memory is tight, try a tile size such as 256 or "
                "512. Smaller tiles use less memory, but may be slower.",
                "If you are not troubleshooting a memory problem, leave this at 0.",
            ],
            self,
        ).exec()

    def _show_device_help(self) -> None:
        HelpDialog(
            "Device",
            [
                "Device controls where the AI model runs.",
                "Auto is recommended. PixelUp lets the underlying AI stack choose the best "
                "available option.",
                "MPS uses Apple GPU acceleration on supported Macs. It is usually faster than "
                "CPU on Apple Silicon, but some model operations may still fall back internally.",
                "CUDA uses an NVIDIA GPU on systems where the CUDA-enabled PyTorch stack is "
                "available.",
                "CPU is the compatibility option. It is slower, but useful when GPU acceleration "
                "is unavailable or unreliable.",
            ],
            self,
        ).exec()


class AboutDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About PixelUp")
        self.setModal(True)

        root = QWidget()
        root.setObjectName("dialogRoot")
        root.setMinimumWidth(520)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("About PixelUp")
        title.setObjectName("windowTitle")
        layout.addWidget(title)

        info_card = QFrame()
        info_card.setObjectName("sectionCard")
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(18, 18, 18, 18)
        info_layout.setSpacing(8)
        name = QLabel("PixelUp")
        name.setObjectName("windowTitle")
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version = QLabel(f"Version {__version__}")
        version.setObjectName("dialogSubtitle")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        copy = QLabel("Upscale local images with Real-ESRGAN in a simple desktop workflow.")
        copy.setObjectName("mutedText")
        copy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        copy.setWordWrap(True)
        links = QWidget()
        links_layout = QHBoxLayout(links)
        links_layout.setContentsMargins(0, 4, 0, 0)
        links_layout.setSpacing(10)
        github_button = QPushButton("GitHub")
        github_button.clicked.connect(lambda: _open_url(PROJECT_URL))
        issues_button = QPushButton("Report issue")
        issues_button.clicked.connect(lambda: _open_url(ISSUES_URL))
        links_layout.addStretch()
        links_layout.addWidget(github_button)
        links_layout.addWidget(issues_button)
        links_layout.addStretch()
        meta = QLabel("© 2026 Yoshinao Inoguchi · MIT License")
        meta.setObjectName("mutedText")
        meta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(name)
        info_layout.addWidget(version)
        info_layout.addWidget(copy)
        info_layout.addWidget(links)
        info_layout.addSpacing(4)
        info_layout.addWidget(meta)
        layout.addWidget(info_card)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.accept)
        layout.addWidget(buttons)
        shell = QVBoxLayout(self)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        shell.addWidget(root)


class HelpDialog(QDialog):
    def __init__(
        self,
        title_text: str,
        paragraphs: list[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title_text)
        self.setModal(True)

        root = QWidget()
        root.setObjectName("dialogRoot")
        root.setMinimumWidth(520)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel(title_text)
        title.setObjectName("windowTitle")
        layout.addWidget(title)

        card = QFrame()
        card.setObjectName("sectionCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 18, 18, 18)
        card_layout.setSpacing(10)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)
        for paragraph in paragraphs:
            label = _muted_label(paragraph)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            content_layout.addWidget(label)
        content_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(240)
        scroll.setWidget(content)
        card_layout.addWidget(scroll)
        layout.addWidget(card)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.accept)
        layout.addWidget(buttons)

        shell = QVBoxLayout(self)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        shell.addWidget(root)


def _advanced_defaults(config: AppConfig) -> AdvancedSettings:
    return AdvancedSettings(
        device=config.device,
        output_format=config.output_format,
        quality=config.quality,
        tile=config.tile,
    )


def _advanced_for_model(settings: AdvancedSettings, model: str) -> AdvancedSettings:
    if model != "realesr-general-x4v3" and settings.denoise_strength != 1.0:
        return replace(settings, denoise_strength=1.0)
    return settings


def _options_for_job(job: Job) -> UpscaleOptions:
    return UpscaleOptions(
        input_path=job.input_path,
        output_arg=str(job.output_path),
        model=job.model,
        scale=job.scale,
        tile=job.advanced.tile,
        tile_pad=10,
        pre_pad=0,
        fp32=False,
        face_enhance=job.advanced.face_enhance,
        denoise_strength=job.advanced.denoise_strength,
        alpha_mode=job.advanced.alpha_mode,
        gpu_id=None,
        device=job.advanced.device,
        output_format=job.advanced.output_format,
        quality=job.advanced.quality,
        background="white",
        strip_metadata=job.advanced.strip_metadata,
        target_profile=job.advanced.target_profile,
        overwrite=False,
        auto_download=job.auto_download,
        download_timeout=600,
        lock_timeout=600,
    )


def _section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionTitle")
    return label


def _field_label(text: str, on_help: Callable[[], None]) -> QWidget:
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    label = QLabel(text)
    button = QToolButton()
    button.setObjectName("helpButton")
    button.setText("?")
    button.setToolTip(f"What does {text.lower()} mean?")
    button.clicked.connect(on_help)
    layout.addWidget(label)
    layout.addWidget(button)
    layout.addStretch()
    return container


def _button_row(button: QPushButton) -> QWidget:
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 8)
    layout.setSpacing(0)
    layout.addWidget(button)
    layout.addStretch()
    return container


def _muted_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("mutedText")
    label.setWordWrap(True)
    return label


def _section_card(title: str) -> QFrame:
    card = QFrame()
    card.setObjectName("sectionCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(14, 14, 14, 14)
    layout.setSpacing(10)
    layout.addWidget(_section_label(title))
    return card


def _elided_label(
    text: str,
    *,
    mode: Qt.TextElideMode,
    object_name: str | None = None,
) -> QElidingLabel:
    label = QElidingLabel(text)
    label.setElideMode(mode)
    label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    label.setToolTip(text)
    if object_name:
        label.setObjectName(object_name)
    return label


def _item(text: str, *, tooltip: str | None = None) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    if tooltip:
        item.setToolTip(tooltip)
    return item


def _download_text(model: str, done: int, total: int | None) -> str:
    if total:
        return f"{done * 100 // total}% — downloading {model}"
    return f"Downloading {model}"


def _progress_text(phase: str) -> str:
    match phase:
        case "upscale":
            return "Upscaling"
        case "encode":
            return "Saving"
        case _:
            return phase.replace("-", " ").replace("_", " ").capitalize()


def _tile_progress_text(done: int, total: int) -> str:
    return f"Tiles {done}/{total} — processing"


def _status_text(status: str) -> str:
    return {
        "pending": "Pending",
        "running": "Running",
        "succeeded": "Done",
        "failed": "Failed",
        "cancelling": "Cancelling…",
        "cancelled": "Cancelled",
    }.get(status, status.replace("-", " ").replace("_", " ").capitalize())


def _queue_column_widths(metrics: object) -> dict[str, int]:
    longest_model = max(UPSCALE_MODELS, key=len) if UPSCALE_MODELS else "RealESRGAN_x4plus_anime_6B"
    samples = {
        "model": longest_model,
        "scale": "4x",
        "status": _tile_progress_text(9999, 9999),
    }
    padding = 32
    floors = {"model": 180, "scale": 56, "status": 180}
    return {
        key: max(floors[key], metrics.horizontalAdvance(text) + padding)  # type: ignore[attr-defined]
        for key, text in samples.items()
    }


def _safe_image_size(path: Path) -> tuple[int, int] | None:
    try:
        return read_image_size(path) if path.exists() else None
    except PixelupError:
        return None


def _preview_meta_text(size: tuple[int, int] | None) -> str:
    if size is None:
        return "Size: unavailable"
    width, height = size
    return f"Size: {width} × {height}"


def _coerce_output_format(value: OutputFormat | str | object) -> OutputFormat:
    if isinstance(value, OutputFormat):
        return value
    if isinstance(value, str):
        return OutputFormat(value)
    raise ValueError(f"Unsupported output format: {value!r}")


def _advanced_log_payload(settings: AdvancedSettings) -> dict[str, object]:
    return {
        "face_enhance": settings.face_enhance,
        "denoise_strength": settings.denoise_strength,
        "alpha_mode": settings.alpha_mode,
        "device": settings.device,
        "output_format": _coerce_output_format(settings.output_format).value,
        "quality": settings.quality,
        "tile": settings.tile,
        "strip_metadata": settings.strip_metadata,
        "target_profile": settings.target_profile,
    }


def _config_log_payload(config: AppConfig) -> dict[str, object]:
    return {
        "max_concurrent_jobs": config.max_concurrent_jobs,
        "close_tab_on_success": config.close_tab_on_success,
        "output_format": config.output_format.value,
        "quality": config.quality,
        "tile": config.tile,
        "device": config.device,
        "auto_download": config.auto_download,
    }


def _job_log_payload(job: Job) -> dict[str, object]:
    return {
        "input_path": str(job.input_path),
        "model": job.model,
        "scale": job.scale,
        "output_path": str(job.output_path),
        "advanced": _advanced_log_payload(job.advanced),
        "auto_download": job.auto_download,
    }


def _reveal_in_file_browser(path: Path) -> None:
    target = path if path.exists() else path.parent
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(target)], check=False)
        return
    if sys.platform == "win32":
        subprocess.run(["explorer", f"/select,{target}"], check=False)
        return
    if target.is_file():
        target = target.parent
    if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(target))):
        raise RuntimeError(f"Could not reveal path: {target}")


def _open_url(url: str) -> None:
    if not QDesktopServices.openUrl(QUrl(url)):
        raise RuntimeError(f"Could not open URL: {url}")


def main() -> int:
    register_image_plugins()
    log_file = configure_session_logging()
    runtime_dirs = resolve_runtime_dirs()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("PixelUp")
    app.setApplicationDisplayName("PixelUp")
    app.setStyleSheet(APP_STYLESHEET)
    LOGGER.info(
        "PixelUp started python=%s platform=%s log_file=%s runtime_dirs=%s argv=%s",
        sys.version.split()[0],
        sys.platform,
        log_file,
        {
            "models_dir": str(runtime_dirs.models_dir),
            "temp_dir": str(runtime_dirs.temp_dir),
        },
        sys.argv[1:],
    )

    window = MainWindow(log_file=log_file)
    window.show()
    paths = [Path(arg) for arg in sys.argv[1:] if not arg.startswith("-")]
    if paths:
        window.open_paths(paths)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
