from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path

from PySide6.QtCore import QObject, QPoint, QRect, QSize, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import (
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
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pixelup import __version__
from pixelup.app_config import CONFIG_PATH, AppConfig, load_app_config, save_app_config
from pixelup.config import resolve_runtime_dirs
from pixelup.errors import PixelupError
from pixelup.imaging import register_image_plugins
from pixelup.models import KNOWN_MODELS
from pixelup.paths import OutputFormat, default_output_path
from pixelup.session_log import configure_session_logging, get_logger
from pixelup.sidecar import write_sidecar
from pixelup.upscale import UpscaleOptions, run_upscale

LOGGER = get_logger()
MODEL_ORDER = (
    "realesr-general-x4v3",
    "RealESRGAN_x4plus",
    "RealESRNet_x4plus",
    "RealESRGAN_x2plus",
    "RealESRGAN_x4plus_anime_6B",
    "realesr-animevideov3",
)
KNOWN_MODEL_NAMES = {model.name for model in KNOWN_MODELS}
UPSCALE_MODELS = tuple(name for name in MODEL_ORDER if name in KNOWN_MODEL_NAMES)
TAB_TEXT_MAX_WIDTH = 180
LEFT_PANE_MIN_WIDTH = 210
LEFT_PANE_START_WIDTH = 220


@dataclass(frozen=True, slots=True)
class AdvancedSettings:
    face_enhance: bool = False
    denoise_strength: float = 1.0
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

    @Slot()
    def run(self) -> None:
        warnings: list[str] = []
        LOGGER.info(
            "Starting job %s for %s with %s",
            self.job.id,
            self.job.input_path,
            self.job.model,
        )
        try:
            options = _options_for_job(self.job)
            result = run_upscale(
                options,
                resolve_runtime_dirs(),
                on_progress=lambda phase: self.signals.progress.emit(self.job.id, phase),
                on_tile=lambda done, total: self.signals.progress.emit(
                    self.job.id, f"tile {done}/{total}"
                ),
                on_download=lambda model, done, total: self.signals.progress.emit(
                    self.job.id, _download_text(model, done, total)
                ),
                on_warning=warnings.append,
            )
            sidecar = write_sidecar(
                input_path=self.job.input_path,
                output_path=self.job.output_path,
                options=options,
                result=result,
                warnings=warnings,
            )
            result["sidecar"] = str(sidecar)
            LOGGER.info("Finished job %s successfully -> %s", self.job.id, self.job.output_path)
            self.signals.finished.emit(self.job.id, True, "done", result, warnings)
        except PixelupError as exc:
            LOGGER.warning("Job %s failed: %s", self.job.id, exc.message)
            self.signals.finished.emit(self.job.id, False, exc.message, {}, warnings)
        except Exception as exc:
            LOGGER.exception("Job %s failed unexpectedly", self.job.id)
            self.signals.finished.emit(self.job.id, False, f"Unexpected error: {exc}", {}, warnings)


class FlowLayout(QLayout):
    def __init__(self, parent: QWidget | None = None, margin: int = 0, spacing: int = 6) -> None:
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
            widget = item.widget()
            spacing = self.spacing()
            size = item.sizeHint()
            next_x = x + size.width() + spacing
            if line_height > 0 and next_x - spacing > right + 1:
                x = rect.x()
                y = y + line_height + spacing
                next_x = x + size.width() + spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), size))
                if widget is not None:
                    widget.updateGeometry()
            x = next_x
            line_height = max(line_height, size.height())
        return y + line_height - rect.y()


class ElidedLabel(QLabel):
    def __init__(
        self,
        text: str = "",
        *,
        mode: Qt.TextElideMode = Qt.TextElideMode.ElideRight,
    ) -> None:
        super().__init__()
        self._full_text = ""
        self._mode = mode
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802
        self._full_text = text
        self.setToolTip(text)
        self._update_text()

    def resizeEvent(self, event: object) -> None:
        self._update_text()
        super().resizeEvent(event)

    def _update_text(self) -> None:
        width = max(10, self.contentsRect().width())
        text = self.fontMetrics().elidedText(self._full_text, self._mode, width)
        super().setText(text)


class WrappedTabChip(QFrame):
    activated = Signal(object)
    close_requested = Signal(object)

    def __init__(self, text: str) -> None:
        super().__init__()
        self._full_text = text
        self._selected = False
        self._color = QColor("black")

        self._button = QToolButton()
        self._button.setAutoRaise(True)
        self._button.setCheckable(True)
        self._button.clicked.connect(lambda: self.activated.emit(self))

        self._close_button = QToolButton()
        self._close_button.setAutoRaise(True)
        self._close_button.setText("×")
        self._close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_button.clicked.connect(lambda: self.close_requested.emit(self))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 6, 4)
        layout.setSpacing(4)
        layout.addWidget(self._button, 1)
        layout.addWidget(self._close_button)

        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setMaximumWidth(230)
        self.setMinimumWidth(90)
        self.set_text(text)
        self._refresh_style()

    def sizeHint(self) -> QSize:
        return QSize(200, 32)

    def set_text(self, text: str) -> None:
        self._full_text = text
        self.setToolTip(text)
        self._update_button_text()

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._button.setChecked(selected)
        self._refresh_style()

    def set_color(self, color: QColor) -> None:
        self._color = color
        self._refresh_style()

    def resizeEvent(self, event: object) -> None:
        self._update_button_text()
        super().resizeEvent(event)

    def _update_button_text(self) -> None:
        available = max(40, min(TAB_TEXT_MAX_WIDTH, self.width() - 34))
        self._button.setText(
            self.fontMetrics().elidedText(
                self._full_text,
                Qt.TextElideMode.ElideRight,
                available,
            )
        )

    def _refresh_style(self) -> None:
        fg = self._color.name()
        bg = "#edf4ff" if self._selected else "#f5f5f5"
        border = "#7aa2ff" if self._selected else "#d0d0d0"
        self.setStyleSheet(
            f"""
            QFrame {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 8px;
            }}
            QToolButton {{
                color: {fg};
                background: transparent;
                border: none;
                padding: 0;
            }}
            """
        )


class WrappedTabs(QWidget):
    tabCloseRequested = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self._entries: list[dict[str, object]] = []

        self._tab_row = QWidget()
        self._tab_layout = FlowLayout(self._tab_row)

        self._stack = QStackedWidget()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._tab_row)
        layout.addWidget(self._stack, 1)

    def addTab(self, widget: QWidget, text: str) -> int:  # noqa: N802
        chip = WrappedTabChip(text)
        chip.activated.connect(self._activate_chip)
        chip.close_requested.connect(self._request_close_chip)
        self._entries.append(
            {"widget": widget, "chip": chip, "text": text, "color": QColor("black")}
        )
        self._tab_layout.addWidget(chip)
        self._stack.addWidget(widget)
        if len(self._entries) == 1:
            self.setCurrentWidget(widget)
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

    def removeTab(self, index: int) -> None:  # noqa: N802
        if not 0 <= index < len(self._entries):
            return
        entry = self._entries.pop(index)
        widget = entry["widget"]
        chip = entry["chip"]
        current_index = self._stack.currentIndex()
        self._stack.removeWidget(widget)  # type: ignore[arg-type]
        self._tab_layout.removeWidget(chip)  # type: ignore[arg-type]
        chip.deleteLater()  # type: ignore[union-attr]
        if self._entries:
            next_index = min(index, len(self._entries) - 1)
            if current_index == index:
                self.setCurrentWidget(self._entries[next_index]["widget"])  # type: ignore[arg-type]
            else:
                self._sync_selection()

    def setCurrentWidget(self, widget: QWidget) -> None:  # noqa: N802
        self._stack.setCurrentWidget(widget)
        self._sync_selection()

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
            selected = entry["widget"] is current
            chip = entry["chip"]
            chip.set_selected(selected)  # type: ignore[union-attr]
            chip.set_color(entry["color"])  # type: ignore[union-attr]


class ImageTab(QWidget):
    enqueue_requested = Signal(object, object, int)
    retry_requested = Signal(object)
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

        self.model_checks: dict[str, QCheckBox] = {}
        for model in UPSCALE_MODELS:
            check = QCheckBox(model)
            self.model_checks[model] = check

        self.scale_group = QButtonGroup(self)
        self.scale_2 = QRadioButton("2x")
        self.scale_4 = QRadioButton("4x")
        self.scale_4.setChecked(True)
        self.scale_group.addButton(self.scale_2)
        self.scale_group.addButton(self.scale_4)

        enqueue_button = QPushButton("Enqueue selected")
        enqueue_button.clicked.connect(self._enqueue_selected)
        try_all_button = QPushButton("Try all models")
        try_all_button.clicked.connect(self._try_all_models)
        retry_button = QPushButton("Retry failed")
        retry_button.clicked.connect(lambda: self.retry_requested.emit(self))

        self.name_label = ElidedLabel(self.input_path.name, mode=Qt.TextElideMode.ElideRight)
        self.dir_label = ElidedLabel(str(self.input_path.parent), mode=Qt.TextElideMode.ElideMiddle)

        left = QWidget()
        left.setMinimumWidth(LEFT_PANE_MIN_WIDTH)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.addWidget(self.name_label)
        left_layout.addWidget(self.dir_label)
        left_layout.addSpacing(4)
        left_layout.addWidget(QLabel("Models"))

        model_box = QWidget()
        model_layout = QVBoxLayout(model_box)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(2)
        for check in self.model_checks.values():
            model_layout.addWidget(check)
        model_layout.addStretch()
        model_scroll = QScrollArea()
        model_scroll.setWidgetResizable(True)
        model_scroll.setMaximumHeight(170)
        model_scroll.setWidget(model_box)
        left_layout.addWidget(model_scroll)

        scale_row = QWidget()
        scale_layout = QHBoxLayout(scale_row)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.setSpacing(8)
        scale_layout.addWidget(QLabel("Scale"))
        scale_layout.addWidget(self.scale_2)
        scale_layout.addWidget(self.scale_4)
        scale_layout.addStretch()
        left_layout.addWidget(scale_row)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        left_layout.addWidget(self.advanced_toggle)

        self.advanced_content = self._build_advanced_content()
        self.advanced_content.setVisible(False)
        left_layout.addWidget(self.advanced_content)

        left_layout.addWidget(enqueue_button)
        left_layout.addWidget(try_all_button)
        left_layout.addWidget(retry_button)
        left_layout.addStretch()

        self.preview = QLabel("Preview unavailable")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(260)
        pixmap = QPixmap(str(self.input_path))
        if not pixmap.isNull():
            self.preview.setPixmap(
                pixmap.scaled(
                    620,
                    460,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Model", "Scale", "Output", "Status"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(220)
        self.table.verticalHeader().setDefaultSectionSize(24)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        right_layout.addWidget(self.preview, 5)
        right_layout.addWidget(self.table, 2)

        self.splitter = QSplitter()
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(left)
        self.splitter.addWidget(right)
        self.splitter.setSizes([left_pane_width, 900 - left_pane_width])
        self.splitter.splitterMoved.connect(self._emit_left_width_changed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)

        self.set_default_advanced(defaults)

    def add_jobs(self, jobs: list[Job]) -> None:
        for job in jobs:
            self.jobs.append(job)
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._rows_by_job[job.id] = row
            self.table.setItem(row, 0, _item(job.model, tooltip=job.model))
            self.table.setItem(row, 1, _item(f"{job.scale}x"))
            self.table.setItem(row, 2, _item(job.output_path.name, tooltip=str(job.output_path)))
            self.table.setItem(row, 3, _item(job.status))

    def update_job(self, job: Job) -> None:
        row = self._rows_by_job[job.id]
        self.table.item(row, 2).setText(job.output_path.name)
        self.table.item(row, 2).setToolTip(str(job.output_path))
        self.table.item(row, 3).setText(job.message or job.status)
        tooltip = "\n".join(job.warnings) if job.warnings else (job.message or job.status)
        self.table.item(row, 3).setToolTip(tooltip)
        color = {
            "pending": QColor("#eef5ff"),
            "running": QColor("#fff4cc"),
            "succeeded": QColor("#e8f7e8"),
            "failed": QColor("#ffe8e8"),
        }.get(job.status, QColor("white"))
        for column in range(self.table.columnCount()):
            self.table.item(row, column).setBackground(color)

    def counts(self) -> tuple[int, int, int, int]:
        total = len(self.jobs)
        done = sum(1 for job in self.jobs if job.status == "succeeded")
        failed = sum(1 for job in self.jobs if job.status == "failed")
        running = sum(1 for job in self.jobs if job.status == "running")
        return total, done, failed, running

    def all_succeeded(self) -> bool:
        return bool(self.jobs) and all(job.status == "succeeded" for job in self.jobs)

    def has_active_jobs(self) -> bool:
        return any(job.status in {"pending", "running"} for job in self.jobs)

    def current_scale(self) -> int:
        return 2 if self.scale_2.isChecked() else 4

    def current_advanced(self) -> AdvancedSettings:
        profile_text = self.target_profile.currentText()
        return AdvancedSettings(
            face_enhance=self.face_enhance.isChecked(),
            denoise_strength=self.denoise_strength.value(),
            alpha_mode=self.alpha_mode.currentData(),
            device=self.device.currentData(),
            output_format=self.output_format.currentData(),
            quality=self.quality.value(),
            tile=self.tile.value(),
            strip_metadata=self.strip_metadata.isChecked(),
            target_profile=profile_text if profile_text else None,
        )

    def set_default_advanced(self, defaults: AdvancedSettings) -> None:
        previous = self._default_advanced
        current = self.current_advanced() if hasattr(self, "face_enhance") else defaults
        was_default = current == previous
        self._default_advanced = defaults
        if was_default:
            self._apply_advanced(defaults)
        self._update_advanced_summary()

    def set_left_pane_width(self, width: int) -> None:
        total = max(sum(self.splitter.sizes()), width + 200)
        self.splitter.blockSignals(True)
        self.splitter.setSizes([width, max(200, total - width)])
        self.splitter.blockSignals(False)

    def _build_advanced_content(self) -> QWidget:
        box = QWidget()
        form = QFormLayout(box)
        form.setContentsMargins(8, 0, 0, 0)
        form.setSpacing(6)

        self.face_enhance = QCheckBox("Enable face enhancement")
        self.face_enhance.toggled.connect(self._update_advanced_summary)

        self.denoise_strength = QDoubleSpinBox()
        self.denoise_strength.setRange(0.0, 1.0)
        self.denoise_strength.setSingleStep(0.1)
        self.denoise_strength.setDecimals(2)
        self.denoise_strength.valueChanged.connect(self._update_advanced_summary)

        self.alpha_mode = QComboBox()
        self.alpha_mode.addItem("Real-ESRGAN", "realesrgan")
        self.alpha_mode.addItem("Bicubic", "bicubic")
        self.alpha_mode.currentIndexChanged.connect(self._update_advanced_summary)

        self.output_format = QComboBox()
        for fmt in OutputFormat:
            self.output_format.addItem(fmt.value.upper(), fmt)
        self.output_format.currentIndexChanged.connect(self._update_advanced_summary)

        self.quality = QSpinBox()
        self.quality.setRange(0, 100)
        self.quality.valueChanged.connect(self._update_advanced_summary)

        self.tile = QSpinBox()
        self.tile.setRange(0, 4096)
        self.tile.setSingleStep(64)
        self.tile.valueChanged.connect(self._update_advanced_summary)

        self.device = QComboBox()
        self.device.addItem("Auto", "auto")
        self.device.addItem("MPS", "mps")
        self.device.addItem("CUDA", "cuda")
        self.device.addItem("CPU", "cpu")
        self.device.currentIndexChanged.connect(self._update_advanced_summary)

        self.strip_metadata = QCheckBox("Strip metadata")
        self.strip_metadata.toggled.connect(self._update_advanced_summary)

        self.target_profile = QComboBox()
        self.target_profile.addItem("", None)
        self.target_profile.addItem("srgb", "srgb")
        self.target_profile.addItem("p3", "p3")
        self.target_profile.addItem("adobergb", "adobergb")
        self.target_profile.currentIndexChanged.connect(self._update_advanced_summary)

        restore_button = QPushButton("Restore defaults")
        restore_button.clicked.connect(self._restore_advanced_defaults)

        form.addRow("", self.face_enhance)
        form.addRow("Denoise", self.denoise_strength)
        form.addRow("Alpha mode", self.alpha_mode)
        form.addRow("Output format", self.output_format)
        form.addRow("Quality", self.quality)
        form.addRow("Tile size", self.tile)
        form.addRow("Device", self.device)
        form.addRow("", self.strip_metadata)
        form.addRow("Target profile", self.target_profile)
        form.addRow("", restore_button)
        return box

    def _apply_advanced(self, settings: AdvancedSettings) -> None:
        self.face_enhance.setChecked(settings.face_enhance)
        self.denoise_strength.setValue(settings.denoise_strength)
        self.alpha_mode.setCurrentIndex(self.alpha_mode.findData(settings.alpha_mode))
        self.output_format.setCurrentIndex(self.output_format.findData(settings.output_format))
        self.quality.setValue(settings.quality)
        self.tile.setValue(settings.tile)
        self.device.setCurrentIndex(self.device.findData(settings.device))
        self.strip_metadata.setChecked(settings.strip_metadata)
        profile_index = self.target_profile.findText(settings.target_profile or "")
        self.target_profile.setCurrentIndex(max(0, profile_index))
        self._update_advanced_summary()

    def _restore_advanced_defaults(self) -> None:
        self._apply_advanced(self._default_advanced)

    def _toggle_advanced(self, expanded: bool) -> None:
        self.advanced_content.setVisible(expanded)
        self.advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )

    def _update_advanced_summary(self) -> None:
        modified = self.current_advanced() != self._default_advanced
        status = "modified" if modified else "defaults"
        self.advanced_toggle.setText(f"Advanced options ({status})")
        color = "#b45f06" if modified else "#555555"
        self.advanced_toggle.setStyleSheet(f"QToolButton {{ color: {color}; }}")

    def _emit_left_width_changed(self, *_: object) -> None:
        self.left_width_changed.emit(self.splitter.sizes()[0])

    def _enqueue_selected(self) -> None:
        models = [name for name, check in self.model_checks.items() if check.isChecked()]
        self.enqueue_requested.emit(self, models, self.current_scale())

    def _try_all_models(self) -> None:
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

        self.setWindowTitle("PixelUp")
        self.resize(1080, 760)
        self.setAcceptDrops(True)

        self.tabs = WrappedTabs()
        self.tabs.tabCloseRequested.connect(self._close_tab_at)
        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage("Drop image files here, or choose File > Open.")
        self._build_menu()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        self.open_paths(paths)

    def open_paths(self, paths: list[Path]) -> None:
        LOGGER.info("Opening paths: %s", [str(path) for path in paths])
        for path in paths:
            resolved = path.expanduser().resolve()
            if not resolved.is_file():
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
                tab.left_width_changed.connect(self._sync_left_pane_width)
                self._tabs_by_path[resolved] = tab
                self.tabs.addTab(tab, resolved.name)
            self.tabs.setCurrentWidget(tab)
            self._update_tab_state(tab)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_action = file_menu.addAction("&Open images...")
        open_action.triggered.connect(self._open_dialog)
        settings_action = file_menu.addAction("&Settings...")
        settings_action.triggered.connect(self._settings_dialog)
        reveal_logs_action = file_menu.addAction("Reveal current &log")
        reveal_logs_action.triggered.connect(self._reveal_log_file)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("&Quit")
        quit_action.triggered.connect(self.close)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = help_menu.addAction("&About PixelUp")
        about_action.triggered.connect(self._about_dialog)

    def _open_dialog(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Open images")
        self.open_paths([Path(file) for file in files])

    def _settings_dialog(self) -> None:
        dialog = SettingsDialog(self.config, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            previous_defaults = _advanced_defaults(self.config)
            self.config = dialog.config()
            save_app_config(self.config)
            for tab in self._tabs_by_path.values():
                if tab.current_advanced() == previous_defaults:
                    tab.set_default_advanced(_advanced_defaults(self.config))
                else:
                    tab.set_default_advanced(_advanced_defaults(self.config))
            LOGGER.info("Saved settings to %s", CONFIG_PATH)
            self.statusBar().showMessage("Settings saved.", 3000)
            self._schedule()

    def _about_dialog(self) -> None:
        AboutDialog(self.log_file, self).exec()

    def _reveal_log_file(self) -> None:
        _reveal_in_file_browser(self.log_file)
        LOGGER.info("Revealed log file %s", self.log_file)

    @Slot(object, object, int)
    def _enqueue_jobs(self, tab: ImageTab, models: list[str], scale: int) -> None:
        if not models:
            QMessageBox.information(self, "PixelUp", "Choose at least one model.")
            return
        advanced = tab.current_advanced()
        jobs: list[Job] = []
        reserved = {job.output_path for job in tab.jobs}
        for model in models:
            output_path = default_output_path(
                tab.input_path,
                model=model,
                scale=scale,
                output_format=advanced.output_format,
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
                    advanced=advanced,
                    auto_download=self.config.auto_download,
                )
            )
        LOGGER.info(
            "Enqueued %s job(s) for %s with models=%s scale=%s advanced=%s",
            len(jobs),
            tab.input_path,
            models,
            scale,
            advanced,
        )
        tab.add_jobs(jobs)
        self._update_tab_state(tab)
        self._schedule()

    @Slot(object)
    def _retry_failed(self, tab: ImageTab) -> None:
        reserved = {job.output_path for job in tab.jobs if job.status != "failed"}
        changed = False
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
        if changed:
            LOGGER.info("Retrying failed jobs for %s", tab.input_path)
            self._update_tab_state(tab)
            self._schedule()

    @Slot(int)
    def _sync_left_pane_width(self, width: int) -> None:
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
        job.message = "running"
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
        job.message = message
        tab.update_job(job)

    @Slot(int, bool, str, object, object)
    def _job_finished(
        self,
        job_id: int,
        ok: bool,
        message: str,
        _result: object,
        warnings: object,
    ) -> None:
        tab, job = self._find_job(job_id)
        job.status = "succeeded" if ok else "failed"
        job.message = message
        job.warnings = list(warnings) if isinstance(warnings, list) else []
        tab.update_job(job)
        self._active_jobs = max(0, self._active_jobs - 1)
        self._update_tab_state(tab)
        if self.config.close_tab_on_success and tab.all_succeeded():
            self._close_tab(tab)
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
            color = QColor("#0057b8")
        elif total and done == total:
            color = QColor("#148a14")
        else:
            color = QColor("black")
        self.tabs.setTabTextColor(index, color)

    def _close_tab_at(self, index: int) -> None:
        tab = self.tabs.widget(index)
        if isinstance(tab, ImageTab) and tab.has_active_jobs():
            QMessageBox.information(self, "PixelUp", "This tab still has pending or running jobs.")
            return
        if isinstance(tab, ImageTab):
            self._close_tab(tab)

    def _close_tab(self, tab: ImageTab) -> None:
        index = self.tabs.indexOf(tab)
        if index >= 0:
            self.tabs.removeTab(index)
        self._tabs_by_path.pop(tab.input_path, None)
        tab.deleteLater()


class SettingsDialog(QDialog):
    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.concurrent = QSpinBox()
        self.concurrent.setRange(1, 8)
        self.concurrent.setValue(config.max_concurrent_jobs)

        self.close_tabs = QCheckBox("Close successful tabs")
        self.close_tabs.setChecked(config.close_tab_on_success)

        self.format = QComboBox()
        self.format.addItems([item.value for item in OutputFormat])
        self.format.setCurrentText(config.output_format.value)

        self.quality = QSpinBox()
        self.quality.setRange(0, 100)
        self.quality.setValue(config.quality)

        self.tile = QSpinBox()
        self.tile.setRange(0, 4096)
        self.tile.setSingleStep(64)
        self.tile.setValue(config.tile)

        self.device = QComboBox()
        self.device.addItems(["auto", "mps", "cuda", "cpu"])
        self.device.setCurrentText(config.device)

        self.auto_download = QCheckBox("Download missing models")
        self.auto_download.setChecked(config.auto_download)

        form.addRow("Concurrent jobs", self.concurrent)
        form.addRow("", self.close_tabs)
        form.addRow("Default output format", self.format)
        form.addRow("Default quality", self.quality)
        form.addRow("Default tile size", self.tile)
        form.addRow("Default device", self.device)
        form.addRow("", self.auto_download)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        restore_defaults = buttons.addButton(
            "Restore defaults",
            QDialogButtonBox.ButtonRole.ResetRole,
        )
        restore_defaults.clicked.connect(self._restore_defaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def config(self) -> AppConfig:
        return AppConfig(
            max_concurrent_jobs=self.concurrent.value(),
            close_tab_on_success=self.close_tabs.isChecked(),
            output_format=OutputFormat(self.format.currentText()),
            quality=self.quality.value(),
            tile=self.tile.value(),
            device=self.device.currentText(),
            auto_download=self.auto_download.isChecked(),
        )

    def _restore_defaults(self) -> None:
        defaults = AppConfig()
        self.concurrent.setValue(defaults.max_concurrent_jobs)
        self.close_tabs.setChecked(defaults.close_tab_on_success)
        self.format.setCurrentText(defaults.output_format.value)
        self.quality.setValue(defaults.quality)
        self.tile.setValue(defaults.tile)
        self.device.setCurrentText(defaults.device)
        self.auto_download.setChecked(defaults.auto_download)


class AboutDialog(QDialog):
    def __init__(self, log_file: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.log_file = log_file
        self.setWindowTitle("About PixelUp")
        self.setModal(True)

        layout = QVBoxLayout(self)
        title = QLabel(f"<b>PixelUp</b> {__version__}")
        subtitle = QLabel("Simple Real-ESRGAN desktop upscaler")
        config_label = ElidedLabel(str(CONFIG_PATH), mode=Qt.TextElideMode.ElideMiddle)
        log_label = ElidedLabel(str(log_file), mode=Qt.TextElideMode.ElideMiddle)

        reveal_button = QPushButton("Reveal current log")
        reveal_button.clicked.connect(lambda: _reveal_in_file_browser(self.log_file))

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(8)
        layout.addWidget(QLabel("Config file"))
        layout.addWidget(config_label)
        layout.addWidget(QLabel("Current session log"))
        layout.addWidget(log_label)
        layout.addWidget(reveal_button)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.accept)
        layout.addWidget(buttons)


def _advanced_defaults(config: AppConfig) -> AdvancedSettings:
    return AdvancedSettings(
        device=config.device,
        output_format=config.output_format,
        quality=config.quality,
        tile=config.tile,
    )


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


def _download_text(model: str, done: int, total: int | None) -> str:
    if total:
        return f"download {model} {done * 100 // total}%"
    return f"download {model}"


def _item(text: str, *, tooltip: str | None = None) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    if tooltip:
        item.setToolTip(tooltip)
    return item


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


def main() -> int:
    register_image_plugins()
    log_file = configure_session_logging()
    app = QApplication(sys.argv)
    window = MainWindow(log_file=log_file)
    LOGGER.info("PixelUp started with log file %s", log_file)
    window.show()
    paths = [Path(arg) for arg in sys.argv[1:] if not arg.startswith("-")]
    if paths:
        window.open_paths(paths)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
