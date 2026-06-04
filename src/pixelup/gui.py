from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from itertools import count
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Slot
from PySide6.QtGui import (
    QCloseEvent,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QGuiApplication,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
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
    QSpinBox,
    QStyleFactory,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pixelup import __version__
from pixelup.app_config import CONFIG_PATH, AppConfig, load_app_config, save_app_config
from pixelup.config import resolve_runtime_dirs
from pixelup.errors import PixelupError
from pixelup.imaging import read_image_size, register_image_plugins
from pixelup.jobs import (
    ImageEntry,
    Job,
    JobSettings,
    coerce_output_format,
    config_log_payload,
    create_jobs,
    job_log_payload,
    job_settings_defaults,
    job_settings_log_payload,
    job_status_summary,
    retry_failed_jobs,
)
from pixelup.models import KNOWN_MODELS
from pixelup.paths import OutputFormat
from pixelup.runner import JobRunner
from pixelup.session_log import configure_session_logging, get_logger

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
KNOWN_MODEL_NAMES = {model.name for model in KNOWN_MODELS}
UPSCALE_MODELS = tuple(name for name in MODEL_ORDER if name in KNOWN_MODEL_NAMES)
REGULAR_SPACING = 10


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
            return
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
            return
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
            return
        event.ignore()


class ImagePreview(QLabel):
    def __init__(self) -> None:
        super().__init__("No image selected")
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(160, 120)

    def set_image(self, path: Path | None) -> None:
        if path is None:
            self._pixmap = None
            self.clear()
            self.setText("No image selected")
            return
        pixmap = QPixmap(str(path))
        self._pixmap = pixmap if not pixmap.isNull() else None
        if self._pixmap is None:
            self.clear()
            self.setText("Preview unavailable")
            return
        self.setText("")
        self._update_pixmap()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._pixmap is None or self.width() <= 0 or self.height() <= 0:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class MainWindow(QMainWindow):
    def __init__(self, *, log_file: Path) -> None:
        super().__init__()
        self.config = load_app_config()
        self.log_file = log_file
        self._job_ids = count(1)
        self._images_by_path: dict[Path, ImageEntry] = {}
        self._image_order: list[Path] = []
        self._image_rows: dict[Path, int] = {}
        self._queue_rows: dict[int, int] = {}
        self.jobs: list[Job] = []
        self.runner = JobRunner(self.jobs, self)
        self.runner.progress.connect(self._job_progress)
        self.runner.finished.connect(self._job_finished)
        self._session_shutdown = False

        self.setWindowTitle("PixelUp")
        self.resize(1260, 860)
        self.setAcceptDrops(True)
        self._build_ui()
        self._apply_job_settings(job_settings_defaults(self.config))
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
            config_log_payload(self.config),
        )

    def _on_commit_data_request(self, _manager: object) -> None:
        self._session_shutdown = True

    def closeEvent(self, event: QCloseEvent) -> None:
        app = QGuiApplication.instance()
        is_saving_session = app.isSavingSession() if app is not None else False
        if self._session_shutdown or is_saving_session:
            LOGGER.info("Accepting close during session shutdown")
            self.runner.cleanup_for_quit()
            event.accept()
            return
        if not self._images_by_path:
            self.runner.cleanup_for_quit()
            event.accept()
            return
        active = sum(1 for job in self.jobs if job.status in {"pending", "running", "cancelling"})
        text = (
            f"{active} {_plural(active, 'running or pending job')} will be abandoned. Quit PixelUp?"
            if active
            else "Open images will be closed. Quit PixelUp?"
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
            self.runner.cleanup_for_quit()
            event.accept()
        else:
            LOGGER.info("User cancelled quit")
            event.ignore()

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
        _use_regular_spacing(root_layout)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        _use_regular_spacing(content_layout, margins=False)
        content_layout.addWidget(self._build_image_panel(), 1)
        content_layout.addWidget(self._build_work_panel(), 1)
        root_layout.addWidget(content, 1)
        self.setCentralWidget(root)

    def _build_window_actions(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        _use_regular_spacing(layout, margins=False)

        logs_button = QPushButton("Reveal log")
        logs_button.clicked.connect(self._reveal_log_file)
        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self._settings_dialog)
        about_button = QPushButton("About")
        about_button.clicked.connect(self._about_dialog)

        layout.addStretch()
        layout.addWidget(logs_button)
        layout.addWidget(settings_button)
        layout.addWidget(about_button)
        return row

    def _build_image_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        _use_regular_spacing(layout, margins=False)

        layout.addWidget(self._build_images_group(), 1)
        layout.addWidget(self._build_selected_image_group(), 1)
        return panel

    def _build_images_group(self) -> QWidget:
        group = QGroupBox("Images")
        layout = QVBoxLayout(group)
        _use_regular_spacing(layout)

        self.image_table = QTableWidget(0, 3)
        self.image_table.setHorizontalHeaderLabels(["Image", "Size", "Jobs"])
        self.image_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.image_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.image_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.image_table.itemSelectionChanged.connect(self._update_selected_image)
        image_header = self.image_table.horizontalHeader()
        image_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        image_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        image_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        image_header.resizeSection(1, 120)
        image_header.resizeSection(2, 320)
        layout.addWidget(self.image_table, 1)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        _use_regular_spacing(button_layout, margins=False)
        self.open_images_button = QPushButton("Open")
        self.open_images_button.clicked.connect(self._open_dialog)
        self.remove_image_button = QPushButton("Remove")
        self.remove_image_button.clicked.connect(self._remove_selected_image)
        button_layout.addWidget(self.open_images_button)
        button_layout.addWidget(self.remove_image_button)
        layout.addWidget(button_row)
        return group

    def _build_work_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        _use_regular_spacing(layout, margins=False)

        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        _use_regular_spacing(controls_layout, margins=False)
        controls_layout.addWidget(self._build_models_group(), 0, Qt.AlignmentFlag.AlignTop)
        controls_layout.addWidget(self._build_parameters_group(), 0, Qt.AlignmentFlag.AlignTop)
        controls_layout.addWidget(self._build_action_column(), 0, Qt.AlignmentFlag.AlignTop)
        controls_layout.addStretch()

        layout.addWidget(controls)
        layout.addWidget(self._build_queue_panel(), 1)
        return container

    def _build_action_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        _use_regular_spacing(layout, margins=False)
        layout.addWidget(self._build_window_actions())
        layout.addWidget(self._build_actions_group())
        return column

    def _build_selected_image_group(self) -> QWidget:
        group = QGroupBox("Preview")
        layout = QVBoxLayout(group)
        _use_regular_spacing(layout)
        self.preview = ImagePreview()
        layout.addWidget(self.preview, 1)
        return group

    def _build_models_group(self) -> QWidget:
        group = QGroupBox("Models")
        layout = QVBoxLayout(group)
        _use_regular_spacing(layout)
        self.model_checks: dict[str, QCheckBox] = {}
        for model in UPSCALE_MODELS:
            checkbox = QCheckBox(model)
            checkbox.toggled.connect(self._update_action_buttons)
            self.model_checks[model] = checkbox
            layout.addWidget(checkbox)
        layout.addStretch()
        return group

    def _build_parameters_group(self) -> QWidget:
        group = QGroupBox("Parameters")
        form = QFormLayout(group)
        _use_regular_spacing(form)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)

        scale_row = QWidget()
        scale_layout = QHBoxLayout(scale_row)
        self.scale_group = QButtonGroup(self)
        self.scale_2 = QRadioButton("2x")
        self.scale_4 = QRadioButton("4x")
        self.scale_4.setChecked(True)
        self.scale_group.addButton(self.scale_2)
        self.scale_group.addButton(self.scale_4)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.addWidget(self.scale_2)
        scale_layout.addWidget(self.scale_4)
        scale_layout.addStretch()

        self.face_enhance = QCheckBox("Face enhancement")
        self.denoise_strength = NoWheelDoubleSpinBox()
        self.denoise_strength.setRange(0.0, 1.0)
        self.denoise_strength.setSingleStep(0.1)
        self.denoise_strength.setDecimals(2)

        self.alpha_mode = NoWheelComboBox()
        self.alpha_mode.addItem("Real-ESRGAN", "realesrgan")
        self.alpha_mode.addItem("Bicubic", "bicubic")

        self.output_format = NoWheelComboBox()
        for fmt in OutputFormat:
            self.output_format.addItem(fmt.value.upper(), fmt.value)

        self.quality = NoWheelSpinBox()
        self.quality.setRange(0, 100)

        self.tile = NoWheelSpinBox()
        self.tile.setRange(0, 4096)
        self.tile.setSingleStep(256)

        self.device = NoWheelComboBox()
        self.device.addItem("Auto", "auto")
        self.device.addItem("MPS", "mps")
        self.device.addItem("CUDA", "cuda")
        self.device.addItem("CPU", "cpu")

        self.strip_metadata = QCheckBox("Strip metadata")

        self.target_profile = NoWheelComboBox()
        self.target_profile.addItem("Default", None)
        self.target_profile.addItem("sRGB", "srgb")
        self.target_profile.addItem("Display P3", "p3")
        self.target_profile.addItem("Adobe RGB", "adobergb")

        restore = QPushButton("Restore defaults")
        restore.clicked.connect(self._restore_job_settings_defaults)

        form.addRow("Scale", scale_row)
        form.addRow("", self.face_enhance)
        form.addRow("Denoise", self.denoise_strength)
        form.addRow("", QLabel("Applies only to realesr-general-x4v3."))
        form.addRow("Alpha mode", self.alpha_mode)
        form.addRow("Output format", self.output_format)
        form.addRow("Quality", self.quality)
        form.addRow("Tile size", self.tile)
        form.addRow("", QLabel("If memory is limited, try 512 before 256."))
        form.addRow("Device", self.device)
        form.addRow("", self.strip_metadata)
        form.addRow("Target profile", self.target_profile)
        form.addRow("", restore)
        return group

    def _build_actions_group(self) -> QWidget:
        group = QGroupBox("Queue actions")
        layout = QVBoxLayout(group)
        _use_regular_spacing(layout)
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

        layout.addWidget(self.queue_selected_button, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(
            self.queue_selected_all_models_button,
            0,
            Qt.AlignmentFlag.AlignLeft,
        )
        layout.addWidget(
            self.queue_all_selected_models_button,
            0,
            Qt.AlignmentFlag.AlignLeft,
        )
        layout.addWidget(self.queue_all_all_models_button, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.retry_button, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.cancel_button, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addStretch()
        return group

    def _build_queue_panel(self) -> QWidget:
        group = QGroupBox("Queue")
        layout = QVBoxLayout(group)
        _use_regular_spacing(layout)
        self.queue_table = QTableWidget(0, 5)
        self.queue_table.setHorizontalHeaderLabels(["Image", "Model", "Scale", "Output", "Status"])
        self.queue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.queue_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        queue_header = self.queue_table.horizontalHeader()
        queue_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        queue_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        queue_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        queue_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        queue_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        queue_header.resizeSection(0, 180)
        queue_header.resizeSection(1, 230)
        queue_header.resizeSection(2, 70)
        queue_header.resizeSection(4, 180)
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
            previous_defaults = job_settings_defaults(self.config)
            self.config = dialog.config()
            save_app_config(self.config)
            LOGGER.info(
                "Saved settings path=%s previous=%s current=%s",
                CONFIG_PATH,
                config_log_payload(previous_config),
                config_log_payload(self.config),
            )
            if self.current_job_settings() == previous_defaults:
                self._apply_job_settings(job_settings_defaults(self.config))
            self.runner.schedule(self.config.max_concurrent_jobs)
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
        self.image_table.setItem(row, 2, _item("No jobs"))

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
            self.preview.set_image(None)
            self.remove_image_button.setEnabled(False)
            self._update_action_buttons()
            return
        self.preview.set_image(path)
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
                "Pending or running jobs still use this image.",
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

    def current_job_settings(self) -> JobSettings:
        profile = self.target_profile.currentData()
        return JobSettings(
            face_enhance=self.face_enhance.isChecked(),
            denoise_strength=self.denoise_strength.value(),
            alpha_mode=self.alpha_mode.currentData(),
            device=self.device.currentData(),
            output_format=coerce_output_format(self.output_format.currentData()),
            quality=self.quality.value(),
            tile=self.tile.value(),
            strip_metadata=self.strip_metadata.isChecked(),
            target_profile=profile,
        )

    def _apply_job_settings(self, settings: JobSettings) -> None:
        self.face_enhance.setChecked(settings.face_enhance)
        self.denoise_strength.setValue(settings.denoise_strength)
        self.alpha_mode.setCurrentIndex(self.alpha_mode.findData(settings.alpha_mode))
        self.output_format.setCurrentIndex(
            self.output_format.findData(coerce_output_format(settings.output_format).value)
        )
        self.quality.setValue(settings.quality)
        self.tile.setValue(settings.tile)
        self.device.setCurrentIndex(self.device.findData(settings.device))
        self.strip_metadata.setChecked(settings.strip_metadata)
        self.target_profile.setCurrentIndex(self.target_profile.findData(settings.target_profile))

    def _restore_job_settings_defaults(self) -> None:
        defaults = job_settings_defaults(self.config)
        self._apply_job_settings(defaults)
        LOGGER.info("Restored parameter defaults defaults=%s", job_settings_log_payload(defaults))

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
            QMessageBox.information(self, "PixelUp", "Select at least one model.")
            return
        settings = self.current_job_settings()
        new_jobs = create_jobs(
            input_paths=input_paths,
            models=models,
            scale=scale,
            settings=settings,
            existing_jobs=self.jobs,
            auto_download=self.config.auto_download,
            job_ids=self._job_ids,
        )
        LOGGER.info(
            "Accepted enqueue request inputs=%s models=%s scale=%s settings=%s auto_download=%s",
            [str(path) for path in input_paths],
            models,
            scale,
            job_settings_log_payload(settings),
            self.config.auto_download,
        )
        for job in new_jobs:
            LOGGER.info("Queued job %s details=%s", job.id, job_log_payload(job))
        self.jobs.extend(new_jobs)
        self._add_queue_rows(new_jobs)
        self._refresh_image_job_summaries()
        self._update_action_buttons()
        self.runner.schedule(self.config.max_concurrent_jobs)

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
                self.runner.request_cancel(job.id)
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
        retried_jobs = retry_failed_jobs(self.jobs)
        retried = set(retried_jobs)
        for job in self.jobs:
            if job.id in retried:
                self._update_job(job)
        if retried_jobs:
            LOGGER.info("Retrying failed jobs job_ids=%s", retried_jobs)
            self._refresh_image_job_summaries()
            self._update_action_buttons()
            self.runner.schedule(self.config.max_concurrent_jobs)

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
        self._refresh_image_job_summaries()
        self._update_action_buttons()

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
        statuses_by_input: dict[Path, list[str]] = defaultdict(list)
        for job in self.jobs:
            statuses_by_input[job.input_path].append(job.status)
        for path in self._image_order:
            row = self._image_rows[path]
            self.image_table.item(row, 2).setText(job_status_summary(statuses_by_input[path]))
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
        _use_regular_spacing(layout)

        form_widget = QWidget()
        form = QGridLayout(form_widget)
        _use_regular_spacing(form)

        self.concurrent = QSpinBox()
        self.concurrent.setRange(1, 8)
        self.concurrent.setValue(config.max_concurrent_jobs)

        self.auto_download = QCheckBox("Download missing models automatically")
        self.auto_download.setChecked(config.auto_download)

        self.format = QComboBox()
        self.format.addItems([item.value.upper() for item in OutputFormat])
        self.format.setCurrentText(config.output_format.value.upper())

        self.quality = QSpinBox()
        self.quality.setRange(0, 100)
        self.quality.setValue(config.quality)

        self.tile = QSpinBox()
        self.tile.setRange(0, 4096)
        self.tile.setSingleStep(256)
        self.tile.setValue(config.tile)

        self.device = QComboBox()
        self.device.addItem("Auto", "auto")
        self.device.addItem("MPS", "mps")
        self.device.addItem("CUDA", "cuda")
        self.device.addItem("CPU", "cpu")
        self.device.setCurrentIndex(self.device.findData(config.device))

        row = 0
        form.addWidget(QLabel("Concurrent jobs"), row, 0)
        form.addWidget(self.concurrent, row, 1, Qt.AlignmentFlag.AlignLeft)
        row += 1
        form.addWidget(QLabel(""), row, 0)
        form.addWidget(self.auto_download, row, 1, Qt.AlignmentFlag.AlignLeft)
        row += 1
        form.addWidget(QLabel("Output format"), row, 0)
        form.addWidget(self.format, row, 1, Qt.AlignmentFlag.AlignLeft)
        row += 1
        form.addWidget(QLabel("Quality"), row, 0)
        form.addWidget(self.quality, row, 1, Qt.AlignmentFlag.AlignLeft)
        row += 1
        form.addWidget(QLabel(""), row, 0)
        form.addWidget(
            QLabel("Used for JPG and WebP. Ignored for PNG."),
            row,
            1,
            Qt.AlignmentFlag.AlignLeft,
        )
        row += 1
        form.addWidget(QLabel("Tile size"), row, 0)
        form.addWidget(self.tile, row, 1, Qt.AlignmentFlag.AlignLeft)
        row += 1
        form.addWidget(QLabel(""), row, 0)
        form.addWidget(
            QLabel("0 processes the whole image. If memory is limited, try 512 before 256."),
            row,
            1,
            Qt.AlignmentFlag.AlignLeft,
        )
        row += 1
        form.addWidget(QLabel("Device"), row, 0)
        form.addWidget(self.device, row, 1, Qt.AlignmentFlag.AlignLeft)
        row += 1
        form.addWidget(QLabel(""), row, 0)
        form.addWidget(
            QLabel("Auto lets Real-ESRGAN choose the best available device."),
            row,
            1,
            Qt.AlignmentFlag.AlignLeft,
        )
        form.setColumnStretch(0, 0)
        form.setColumnStretch(1, 0)
        form.setColumnStretch(2, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        restore = buttons.addButton("Restore defaults", QDialogButtonBox.ButtonRole.ResetRole)
        restore.clicked.connect(self._restore_defaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(form_widget, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(buttons)
        self.adjustSize()
        self.setMinimumSize(self.sizeHint())

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
        _use_regular_spacing(layout)
        name = QLabel("PixelUp")
        version = QLabel(f"Version {__version__}")
        copy = QLabel("Upscale local images with Real-ESRGAN in a simple desktop workflow.")
        copy.setWordWrap(True)

        links = QWidget()
        links_layout = QHBoxLayout(links)
        _use_regular_spacing(links_layout, margins=False)
        github_button = QPushButton("GitHub")
        github_button.clicked.connect(lambda: _open_url(PROJECT_URL))
        issues_button = QPushButton("Report issue")
        issues_button.clicked.connect(lambda: _open_url(ISSUES_URL))
        links_layout.addStretch()
        links_layout.addWidget(github_button)
        links_layout.addWidget(issues_button)
        links_layout.addStretch()

        meta = QLabel("(c) 2026 Yoshinao Inoguchi - MIT License")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.accept)

        layout.addWidget(name)
        layout.addWidget(version)
        layout.addWidget(copy)
        layout.addWidget(links)
        layout.addWidget(meta)
        layout.addWidget(buttons)


def _item(text: str, *, tooltip: str | None = None) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    if tooltip:
        item.setToolTip(tooltip)
    return item


def _use_regular_spacing(layout: QLayout, *, margins: bool = True) -> None:
    margin = REGULAR_SPACING if margins else 0
    layout.setContentsMargins(margin, margin, margin, margin)
    layout.setSpacing(REGULAR_SPACING)


def _status_text(status: str) -> str:
    return {
        "pending": "Pending",
        "running": "Running",
        "succeeded": "Done",
        "failed": "Failed",
        "cancelling": "Cancelling...",
        "cancelled": "Cancelled",
    }.get(status, status.replace("-", " ").replace("_", " ").capitalize())


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


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    if count == 1:
        return singular
    return plural if plural is not None else f"{singular}s"


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
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")
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
