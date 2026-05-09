from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field, replace
from itertools import count
from pathlib import Path

from PySide6.QtCore import QObject, QSize, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import (
    QCloseEvent,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QGuiApplication,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
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
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

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


@dataclass(frozen=True, slots=True)
class ImageEntry:
    input_path: Path
    input_size: tuple[int, int] | None


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
        LOGGER.info("Starting job %s details=%s", self.job.id, _job_log_payload(self.job))
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
                    self.job.id,
                    False,
                    "Cancelled",
                    {"cancelled": True},
                    warnings,
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


class PreviewLabel(QLabel):
    def __init__(self) -> None:
        super().__init__("No image selected")
        self._source: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(112, 112)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def clear_image(self) -> None:
        self._source = None
        self.setPixmap(QPixmap())
        self.setText("No image selected")

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


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event: object) -> None:
        if self.hasFocus():
            super().wheelEvent(event)  # type: ignore[arg-type]
            return
        event.ignore()  # type: ignore[attr-defined]


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event: object) -> None:
        if self.hasFocus():
            super().wheelEvent(event)  # type: ignore[arg-type]
            return
        event.ignore()  # type: ignore[attr-defined]


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event: object) -> None:
        if self.hasFocus():
            super().wheelEvent(event)  # type: ignore[arg-type]
            return
        event.ignore()  # type: ignore[attr-defined]


class MainWindow(QMainWindow):
    def __init__(self, *, log_file: Path) -> None:
        super().__init__()
        self.config = load_app_config()
        self.log_file = log_file
        self._job_ids = count(1)
        self._threads: dict[int, tuple[QThread, JobWorker]] = {}
        self._active_jobs = 0
        self._images_by_path: dict[Path, ImageEntry] = {}
        self._image_order: list[Path] = []
        self._image_rows: dict[Path, int] = {}
        self._queue_rows: dict[int, int] = {}
        self.jobs: list[Job] = []
        self._session_shutdown = False

        self.setWindowTitle("PixelUp")
        self.resize(1260, 860)
        self.setAcceptDrops(True)
        self._build_ui()
        self._apply_advanced(_advanced_defaults(self.config))
        self._update_selected_image()
        self._update_action_buttons()

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
        app = QGuiApplication.instance()
        is_saving_session = app.isSavingSession() if app is not None else False
        if self._session_shutdown or is_saving_session:
            LOGGER.info("Accepting close during session shutdown")
            self._cleanup_workers_for_quit()
            event.accept()
            return
        if not self._images_by_path:
            self._cleanup_workers_for_quit()
            event.accept()
            return
        active = sum(1 for job in self.jobs if job.status in {"pending", "running", "cancelling"})
        text = (
            f"PixelUp has {active} running or pending job(s). Quit and abandon them?"
            if active
            else "PixelUp has open images. Quit anyway?"
        )
        choice = QMessageBox.question(
            self,
            "Quit PixelUp?",
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice == QMessageBox.StandardButton.Yes:
            LOGGER.info("User confirmed quit active_jobs=%s", active)
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
        selected: Path | None = None
        for path in paths:
            resolved = path.expanduser().resolve()
            if not resolved.is_file():
                LOGGER.warning("Ignored non-file open request path=%s resolved=%s", path, resolved)
                continue
            if resolved not in self._images_by_path:
                entry = ImageEntry(resolved, _safe_image_size(resolved))
                self._images_by_path[resolved] = entry
                self._image_order.append(resolved)
                self._add_image_row(entry)
                LOGGER.info("Added image input=%s size=%s", resolved, entry.input_size)
            else:
                LOGGER.info("Focused existing image input=%s", resolved)
            selected = resolved
        if selected is not None:
            self._select_image(selected)
        self._update_action_buttons()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)
        root_layout.addWidget(self._build_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_image_panel())
        splitter.addWidget(self._build_work_panel())
        splitter.setSizes([340, 1180])
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(root)

    def _build_header(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QLabel("PixelUp")
        title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        open_button = QPushButton("Open images...")
        open_button.clicked.connect(self._open_dialog)
        logs_button = QPushButton("Reveal log")
        logs_button.clicked.connect(self._reveal_log_file)
        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self._settings_dialog)
        about_button = QPushButton("About")
        about_button.clicked.connect(self._about_dialog)

        layout.addWidget(title)
        layout.addWidget(open_button)
        layout.addWidget(logs_button)
        layout.addWidget(settings_button)
        layout.addWidget(about_button)
        return row

    def _build_image_panel(self) -> QWidget:
        group = QGroupBox("Images")
        layout = QVBoxLayout(group)

        self.image_table = QTableWidget(0, 3)
        self.image_table.setHorizontalHeaderLabels(["Image", "Size", "Jobs"])
        self.image_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.image_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.image_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.image_table.verticalHeader().setVisible(False)
        image_header = self.image_table.horizontalHeader()
        image_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        image_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        image_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.image_table.itemSelectionChanged.connect(self._update_selected_image)
        layout.addWidget(self.image_table, 1)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        self.open_images_button = QPushButton("Open...")
        self.open_images_button.clicked.connect(self._open_dialog)
        self.remove_image_button = QPushButton("Remove")
        self.remove_image_button.clicked.connect(self._remove_selected_image)
        button_layout.addWidget(self.open_images_button)
        button_layout.addWidget(self.remove_image_button)
        layout.addWidget(button_row)

        layout.addWidget(self._build_selected_image_group())
        return group

    def _build_work_panel(self) -> QWidget:
        container = QWidget()
        layout = QGridLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        layout.addWidget(self._build_models_group(), 0, 0)
        layout.addWidget(self._build_parameters_group(), 0, 1)
        layout.addWidget(self._build_actions_group(), 0, 2)
        layout.addWidget(self._build_queue_panel(), 1, 0, 1, 3)
        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 3)
        layout.setColumnStretch(2, 2)
        layout.setRowStretch(0, 0)
        layout.setRowStretch(1, 1)
        return container

    def _build_selected_image_group(self) -> QWidget:
        group = QGroupBox("Selected image")
        layout = QHBoxLayout(group)
        layout.setSpacing(8)
        self.preview = PreviewLabel()

        details = QWidget()
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(4)
        self.selected_name = QLabel("No image selected")
        self.selected_name.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.selected_path = QLabel("")
        self.selected_path.setWordWrap(True)
        self.selected_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.selected_size = QLabel("")
        details_layout.addWidget(self.selected_name)
        details_layout.addWidget(self.selected_path)
        details_layout.addWidget(self.selected_size)
        details_layout.addStretch()

        layout.addWidget(self.preview)
        layout.addWidget(details, 1)
        group.setMaximumHeight(160)
        return group

    def _build_models_group(self) -> QWidget:
        group = QGroupBox("Models")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        self.model_checks: dict[str, QCheckBox] = {}
        for model in UPSCALE_MODELS:
            checkbox = QCheckBox(model)
            checkbox.setToolTip(MODEL_DESCRIPTIONS.get(model, ""))
            checkbox.toggled.connect(self._update_action_buttons)
            self.model_checks[model] = checkbox
            layout.addWidget(checkbox)
        layout.addStretch()
        group.setMinimumWidth(260)
        return group

    def _build_parameters_group(self) -> QWidget:
        group = QGroupBox("Parameters")
        form = QFormLayout(group)
        form.setSpacing(6)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        scale_row = QWidget()
        scale_layout = QHBoxLayout(scale_row)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        self.scale_group = QButtonGroup(self)
        self.scale_2 = QRadioButton("2x")
        self.scale_4 = QRadioButton("4x")
        self.scale_4.setChecked(True)
        self.scale_group.addButton(self.scale_2)
        self.scale_group.addButton(self.scale_4)
        scale_layout.addWidget(self.scale_2)
        scale_layout.addWidget(self.scale_4)
        scale_layout.addStretch()

        self.face_enhance = QCheckBox("Enable")
        self.denoise_strength = NoWheelDoubleSpinBox()
        self.denoise_strength.setRange(0.0, 1.0)
        self.denoise_strength.setSingleStep(0.1)
        self.denoise_strength.setDecimals(2)
        self.denoise_strength.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.alpha_mode = NoWheelComboBox()
        self.alpha_mode.addItem("Real-ESRGAN", "realesrgan")
        self.alpha_mode.addItem("Bicubic", "bicubic")

        self.output_format = NoWheelComboBox()
        for fmt in OutputFormat:
            self.output_format.addItem(fmt.value.upper(), fmt.value)

        self.quality = NoWheelSpinBox()
        self.quality.setRange(0, 100)
        self.quality.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.tile = NoWheelSpinBox()
        self.tile.setRange(0, 4096)
        self.tile.setSingleStep(64)
        self.tile.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        self.device = NoWheelComboBox()
        self.device.addItem("Auto", "auto")
        self.device.addItem("MPS", "mps")
        self.device.addItem("CUDA", "cuda")
        self.device.addItem("CPU", "cpu")

        self.strip_metadata = QCheckBox("Enable")

        self.target_profile = NoWheelComboBox()
        self.target_profile.addItem("Default", None)
        self.target_profile.addItem("sRGB", "srgb")
        self.target_profile.addItem("Display P3", "p3")
        self.target_profile.addItem("Adobe RGB", "adobergb")

        restore = QPushButton("Restore defaults")
        restore.clicked.connect(self._restore_advanced_defaults)

        form.addRow("Scale", scale_row)
        form.addRow("Face enhancement", self.face_enhance)
        form.addRow("Denoise", self.denoise_strength)
        form.addRow("Alpha mode", self.alpha_mode)
        form.addRow("Output format", self.output_format)
        form.addRow("Quality", self.quality)
        form.addRow("Tile size", self.tile)
        form.addRow("Device", self.device)
        form.addRow("Strip metadata", self.strip_metadata)
        form.addRow("Target profile", self.target_profile)
        form.addRow("", restore)
        group.setMinimumWidth(320)
        return group

    def _build_actions_group(self) -> QWidget:
        group = QGroupBox("Queue actions")
        layout = QGridLayout(group)
        layout.setVerticalSpacing(6)
        self.queue_selected_button = QPushButton("Queue selected image")
        self.queue_selected_button.clicked.connect(self._queue_selected_image)
        self.queue_selected_all_models_button = QPushButton("Queue selected image with all models")
        self.queue_selected_all_models_button.clicked.connect(self._queue_selected_image_all_models)
        self.queue_all_selected_models_button = QPushButton("Queue all images with selected models")
        self.queue_all_selected_models_button.clicked.connect(self._queue_all_images_selected_models)
        self.queue_all_all_models_button = QPushButton("Queue all images with all models")
        self.queue_all_all_models_button.clicked.connect(self._queue_all_images_all_models)
        self.retry_button = QPushButton("Retry failed jobs")
        self.retry_button.clicked.connect(self._retry_failed)
        self.cancel_button = QPushButton("Cancel queue")
        self.cancel_button.clicked.connect(self._cancel_queue)

        layout.addWidget(self.queue_selected_button, 0, 0)
        layout.addWidget(self.queue_selected_all_models_button, 1, 0)
        layout.addWidget(self.queue_all_selected_models_button, 2, 0)
        layout.addWidget(self.queue_all_all_models_button, 3, 0)
        layout.addWidget(self.retry_button, 4, 0)
        layout.addWidget(self.cancel_button, 5, 0)
        layout.setRowStretch(6, 1)
        group.setMinimumWidth(250)
        return group

    def _build_queue_panel(self) -> QWidget:
        group = QGroupBox("Queue")
        layout = QVBoxLayout(group)
        self.queue_table = QTableWidget(0, 5)
        self.queue_table.setHorizontalHeaderLabels(["Image", "Model", "Scale", "Output", "Status"])
        self.queue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.queue_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.queue_table.verticalHeader().setVisible(False)
        header = self.queue_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)
        widths = _queue_column_widths(self.queue_table.fontMetrics())
        self.queue_table.setColumnWidth(0, 160)
        self.queue_table.setColumnWidth(1, widths["model"])
        self.queue_table.setColumnWidth(4, widths["status"])
        layout.addWidget(self.queue_table)
        return group

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
            if self.current_advanced() == previous_defaults:
                self._apply_advanced(_advanced_defaults(self.config))
            self._schedule()
            return
        LOGGER.info("Closed settings dialog without saving")

    def _about_dialog(self) -> None:
        LOGGER.info("Opened about dialog")
        AboutDialog(self).exec()

    def _reveal_log_file(self) -> None:
        _reveal_in_file_browser(self.log_file)
        LOGGER.info("Revealed log file %s", self.log_file)

    def _add_image_row(self, entry: ImageEntry) -> None:
        row = self.image_table.rowCount()
        self.image_table.insertRow(row)
        self._image_rows[entry.input_path] = row
        self.image_table.setItem(
            row,
            0,
            _item(entry.input_path.name, tooltip=str(entry.input_path)),
        )
        self.image_table.setItem(row, 1, _item(_image_size_text(entry.input_size)))
        self.image_table.setItem(row, 2, _item("0 jobs"))

    def _select_image(self, path: Path) -> None:
        row = self._image_rows.get(path)
        if row is None:
            return
        self.image_table.selectRow(row)

    def _selected_path(self) -> Path | None:
        row = self.image_table.currentRow()
        if row < 0:
            return None
        for path, image_row in self._image_rows.items():
            if image_row == row:
                return path
        return None

    def _selected_paths(self) -> list[Path]:
        path = self._selected_path()
        return [path] if path is not None else []

    def _update_selected_image(self) -> None:
        path = self._selected_path()
        if path is None:
            self.selected_name.setText("No image selected")
            self.selected_path.setText("")
            self.selected_size.setText("")
            self.preview.clear_image()
            self.remove_image_button.setEnabled(False)
            self._update_action_buttons()
            return
        entry = self._images_by_path[path]
        self.selected_name.setText(path.name)
        self.selected_path.setText(str(path))
        self.selected_size.setText(f"Size: {_image_size_text(entry.input_size)}")
        self.preview.set_image_path(path)
        self.remove_image_button.setEnabled(not self._has_active_jobs(path))
        self._update_action_buttons()

    def _remove_selected_image(self) -> None:
        path = self._selected_path()
        if path is None:
            return
        if self._has_active_jobs(path):
            QMessageBox.information(
                self,
                "PixelUp",
                "This image still has pending or running jobs.",
            )
            return
        row = self._image_rows.pop(path)
        self.image_table.removeRow(row)
        self._images_by_path.pop(path, None)
        self._image_order = [item for item in self._image_order if item != path]
        self._rebuild_image_rows()
        LOGGER.info("Removed image input=%s", path)
        self._update_selected_image()
        self._update_action_buttons()

    def _rebuild_image_rows(self) -> None:
        self._image_rows = {}
        for row, path in enumerate(self._image_order):
            self._image_rows[path] = row

    def _selected_models(self) -> list[str]:
        return [model for model, checkbox in self.model_checks.items() if checkbox.isChecked()]

    def _current_scale(self) -> int:
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

    def _restore_advanced_defaults(self) -> None:
        defaults = _advanced_defaults(self.config)
        self._apply_advanced(defaults)
        LOGGER.info("Restored parameter defaults defaults=%s", _advanced_log_payload(defaults))

    def _queue_selected_image(self) -> None:
        self._enqueue_jobs(self._selected_paths(), self._selected_models(), self._current_scale())

    def _queue_selected_image_all_models(self) -> None:
        self._enqueue_jobs(self._selected_paths(), list(UPSCALE_MODELS), self._current_scale())

    def _queue_all_images_selected_models(self) -> None:
        self._enqueue_jobs(list(self._image_order), self._selected_models(), self._current_scale())

    def _queue_all_images_all_models(self) -> None:
        self._enqueue_jobs(list(self._image_order), list(UPSCALE_MODELS), self._current_scale())

    def _enqueue_jobs(self, input_paths: list[Path], models: list[str], scale: int) -> None:
        if not input_paths:
            QMessageBox.information(self, "PixelUp", "Open or select at least one image.")
            return
        if not models:
            QMessageBox.information(self, "PixelUp", "Choose at least one model.")
            return
        advanced = self.current_advanced()
        reserved_by_input: dict[Path, set[Path]] = defaultdict(set)
        for job in self.jobs:
            reserved_by_input[job.input_path].add(job.output_path)

        new_jobs: list[Job] = []
        for input_path in input_paths:
            reserved = reserved_by_input[input_path]
            for model in models:
                model_advanced = _advanced_for_model(advanced, model)
                output_path = default_output_path(
                    input_path,
                    model=model,
                    scale=scale,
                    output_format=model_advanced.output_format,
                    reserved=reserved,
                )
                reserved.add(output_path)
                new_jobs.append(
                    Job(
                        id=next(self._job_ids),
                        input_path=input_path,
                        model=model,
                        scale=scale,
                        output_path=output_path,
                        advanced=model_advanced,
                        auto_download=self.config.auto_download,
                    )
                )
        LOGGER.info(
            "Accepted enqueue request inputs=%s models=%s scale=%s advanced=%s auto_download=%s",
            [str(path) for path in input_paths],
            models,
            scale,
            _advanced_log_payload(advanced),
            self.config.auto_download,
        )
        for job in new_jobs:
            LOGGER.info("Queued job %s details=%s", job.id, _job_log_payload(job))
        self.jobs.extend(new_jobs)
        self._add_queue_rows(new_jobs)
        self._refresh_image_job_summaries()
        self._update_action_buttons()
        self._schedule()

    def _add_queue_rows(self, jobs: list[Job]) -> None:
        for job in jobs:
            row = self.queue_table.rowCount()
            self.queue_table.insertRow(row)
            self._queue_rows[job.id] = row
            self.queue_table.setItem(
                row,
                0,
                _item(job.input_path.name, tooltip=str(job.input_path)),
            )
            self.queue_table.setItem(row, 1, _item(job.model, tooltip=job.model))
            self.queue_table.setItem(row, 2, _item(f"{job.scale}x"))
            self.queue_table.setItem(
                row,
                3,
                _item(job.output_path.name, tooltip=str(job.output_path)),
            )
            self.queue_table.setItem(row, 4, _item(_status_text(job.status)))

    def _cancel_queue(self) -> None:
        cancelled_pending: list[int] = []
        signalled_running: list[int] = []
        for job in self.jobs:
            if job.status == "pending":
                job.status = "cancelled"
                job.message = "Cancelled"
                self._update_job(job)
                cancelled_pending.append(job.id)
            elif job.status == "running":
                entry = self._threads.get(job.id)
                if entry is not None:
                    _, worker = entry
                    worker.request_cancel()
                job.status = "cancelling"
                self._update_job(job)
                signalled_running.append(job.id)
        LOGGER.info(
            "Cancel queue cancelled_pending=%s signalled_running=%s",
            cancelled_pending,
            signalled_running,
        )
        self._refresh_image_job_summaries()
        self._update_action_buttons()

    def _retry_failed(self) -> None:
        reserved_by_input: dict[Path, set[Path]] = defaultdict(set)
        for job in self.jobs:
            if job.status != "failed":
                reserved_by_input[job.input_path].add(job.output_path)
        retried_jobs: list[int] = []
        for job in self.jobs:
            if job.status != "failed":
                continue
            reserved = reserved_by_input[job.input_path]
            job.output_path = default_output_path(
                job.input_path,
                model=job.model,
                scale=job.scale,
                output_format=job.advanced.output_format,
                reserved=reserved,
            )
            reserved.add(job.output_path)
            job.status = "pending"
            job.message = ""
            job.warnings = []
            self._update_job(job)
            retried_jobs.append(job.id)
        if retried_jobs:
            LOGGER.info("Retrying failed jobs job_ids=%s", retried_jobs)
            self._refresh_image_job_summaries()
            self._update_action_buttons()
            self._schedule()

    def _schedule(self) -> None:
        limit = max(1, self.config.max_concurrent_jobs)
        while self._active_jobs < limit:
            job = self._next_pending_job()
            if job is None:
                return
            self._start_job(job)

    def _next_pending_job(self) -> Job | None:
        for job in self.jobs:
            if job.status == "pending":
                return job
        return None

    def _start_job(self, job: Job) -> None:
        job.status = "running"
        job.message = "Starting"
        self._update_job(job)
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
        job = self._find_job(job_id)
        if message != job.message:
            LOGGER.info(
                "Job %s progress input=%s model=%s output=%s message=%s",
                job.id,
                job.input_path,
                job.model,
                job.output_path,
                message,
            )
        job.message = message
        self._update_job(job)

    @Slot(int, bool, str, object, object)
    def _job_finished(
        self,
        job_id: int,
        ok: bool,
        message: str,
        result: object,
        warnings: object,
    ) -> None:
        job = self._find_job(job_id)
        cancelled = isinstance(result, dict) and bool(result.get("cancelled"))
        if ok:
            job.status = "succeeded"
        elif cancelled:
            job.status = "cancelled"
        else:
            job.status = "failed"
        job.message = message
        job.warnings = list(warnings) if isinstance(warnings, list) else []
        self._update_job(job)
        self._active_jobs = max(0, self._active_jobs - 1)
        self._refresh_image_job_summaries()
        self._update_action_buttons()
        QTimer.singleShot(0, self._schedule)

    def _find_job(self, job_id: int) -> Job:
        for job in self.jobs:
            if job.id == job_id:
                return job
        raise RuntimeError(f"Unknown job id: {job_id}")

    def _update_job(self, job: Job) -> None:
        row = self._queue_rows[job.id]
        self.queue_table.item(row, 3).setText(job.output_path.name)
        self.queue_table.item(row, 3).setToolTip(str(job.output_path))
        status_text = job.message or _status_text(job.status)
        if job.status == "cancelling":
            status_text = _status_text("cancelling")
        self.queue_table.item(row, 4).setText(status_text)
        tooltip = "\n".join(job.warnings) if job.warnings else status_text
        self.queue_table.item(row, 4).setToolTip(tooltip)

    def _refresh_image_job_summaries(self) -> None:
        counts: dict[Path, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for job in self.jobs:
            counts[job.input_path]["total"] += 1
            counts[job.input_path][job.status] += 1
        for path in self._image_order:
            row = self._image_rows[path]
            total = counts[path]["total"]
            done = counts[path]["succeeded"]
            failed = counts[path]["failed"]
            running = counts[path]["running"] + counts[path]["cancelling"]
            pending = counts[path]["pending"]
            if total == 0:
                text = "0 jobs"
            else:
                parts = [f"{done}/{total} done"]
                if failed:
                    parts.append(f"{failed} failed")
                if running:
                    parts.append(f"{running} running")
                if pending:
                    parts.append(f"{pending} pending")
                text = ", ".join(parts)
            self.image_table.item(row, 2).setText(text)
        self._update_selected_image()

    def _has_active_jobs(self, path: Path) -> bool:
        return any(
            job.input_path == path and job.status in {"pending", "running", "cancelling"}
            for job in self.jobs
        )

    def _update_action_buttons(self) -> None:
        has_images = bool(self._image_order)
        has_selected_image = self._selected_path() is not None
        has_selected_models = bool(self._selected_models())
        has_failed = any(job.status == "failed" for job in self.jobs)
        has_cancellable = any(job.status in {"pending", "running"} for job in self.jobs)
        self.queue_selected_button.setEnabled(has_selected_image and has_selected_models)
        self.queue_selected_all_models_button.setEnabled(
            has_selected_image and bool(UPSCALE_MODELS)
        )
        self.queue_all_selected_models_button.setEnabled(has_images and has_selected_models)
        self.queue_all_all_models_button.setEnabled(has_images and bool(UPSCALE_MODELS))
        self.retry_button.setEnabled(has_failed)
        self.cancel_button.setEnabled(has_cancellable)


class SettingsDialog(QDialog):
    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        general_group = QGroupBox("General")
        general_form = QFormLayout(general_group)

        self.concurrent = QSpinBox()
        self.concurrent.setRange(1, 8)
        self.concurrent.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.concurrent.setValue(config.max_concurrent_jobs)

        self.auto_download = QCheckBox("Download missing models automatically")
        self.auto_download.setChecked(config.auto_download)

        general_form.addRow("Concurrent jobs", self.concurrent)
        general_form.addRow("", self.auto_download)

        defaults_group = QGroupBox("Parameter defaults")
        defaults_form = QFormLayout(defaults_group)

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
        defaults_form.addRow("Tile size", self.tile)
        defaults_form.addRow("Device", self.device)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        restore = buttons.addButton("Restore defaults", QDialogButtonBox.ButtonRole.ResetRole)
        restore.clicked.connect(self._restore_defaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(general_group)
        layout.addWidget(defaults_group)
        layout.addWidget(buttons)

    def config(self) -> AppConfig:
        return AppConfig(
            max_concurrent_jobs=self.concurrent.value(),
            output_format=OutputFormat(self.format.currentText().lower()),
            quality=self.quality.value(),
            tile=self.tile.value(),
            device=self.device.currentData(),
            auto_download=self.auto_download.isChecked(),
        )

    def _restore_defaults(self) -> None:
        defaults = AppConfig()
        self.concurrent.setValue(defaults.max_concurrent_jobs)
        self.auto_download.setChecked(defaults.auto_download)
        self.format.setCurrentText(defaults.output_format.value.upper())
        self.quality.setValue(defaults.quality)
        self.tile.setValue(defaults.tile)
        self.device.setCurrentIndex(self.device.findData(defaults.device))


class AboutDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About PixelUp")
        self.setModal(True)

        layout = QVBoxLayout(self)
        name = QLabel("PixelUp")
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version = QLabel(f"Version {__version__}")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        copy = QLabel("Upscale local images with Real-ESRGAN in a simple desktop workflow.")
        copy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        copy.setWordWrap(True)

        links = QWidget()
        links_layout = QHBoxLayout(links)
        github_button = QPushButton("GitHub")
        github_button.clicked.connect(lambda: _open_url(PROJECT_URL))
        issues_button = QPushButton("Report issue")
        issues_button.clicked.connect(lambda: _open_url(ISSUES_URL))
        links_layout.addStretch()
        links_layout.addWidget(github_button)
        links_layout.addWidget(issues_button)
        links_layout.addStretch()

        meta = QLabel("(c) 2026 Yoshinao Inoguchi - MIT License")
        meta.setAlignment(Qt.AlignmentFlag.AlignCenter)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.accept)

        layout.addWidget(name)
        layout.addWidget(version)
        layout.addWidget(copy)
        layout.addWidget(links)
        layout.addWidget(meta)
        layout.addWidget(buttons)


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


def _item(text: str, *, tooltip: str | None = None) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    if tooltip:
        item.setToolTip(tooltip)
    return item


def _download_text(model: str, done: int, total: int | None) -> str:
    if total:
        return f"{done * 100 // total}% - downloading {model}"
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
    return f"Tiles {done}/{total} - processing"


def _status_text(status: str) -> str:
    return {
        "pending": "Pending",
        "running": "Running",
        "succeeded": "Done",
        "failed": "Failed",
        "cancelling": "Cancelling...",
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


def _image_size_text(size: tuple[int, int] | None) -> str:
    if size is None:
        return "unavailable"
    width, height = size
    return f"{width} x {height}"


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
    app.setApplicationName("PixelUp")
    app.setApplicationDisplayName("PixelUp")
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
