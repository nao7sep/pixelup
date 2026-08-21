from __future__ import annotations

import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import replace
from itertools import count
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl, Slot
from PySide6.QtGui import (
    QCloseEvent,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QGuiApplication,
    QIcon,
    QKeySequence,
    QPixmap,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QStyleFactory,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pixelup import __version__
from pixelup.about_dialog import AboutDialog
from pixelup.app_config import (
    config_log_payload,
    config_path,
    ensure_app_config,
    load_app_config_result,
    save_app_config,
)
from pixelup.config import RuntimeDirs, resolve_runtime_dirs
from pixelup.errors import PixelupError
from pixelup.fonts import apply_ui_font
from pixelup.imaging import read_image_size, register_image_plugins
from pixelup.jobs import (
    ImageEntry,
    Job,
    JobSettings,
    coerce_output_format,
    create_jobs,
    job_log_payload,
    job_settings_log_payload,
    job_status_summary,
    retry_failed_jobs,
)
from pixelup.message_dialogs import (
    warn_config_reset,
    warn_image_in_use,
    warn_no_images,
    warn_no_models,
)
from pixelup.model_registry import KNOWN_MODELS
from pixelup.parameters import (
    ALPHA_MODE_CHOICES,
    DEFAULT_SCALE,
    DENOISE_STRENGTH_STEP,
    MAX_DENOISE_STRENGTH,
    MAX_QUALITY,
    MIN_DENOISE_STRENGTH,
    MIN_QUALITY,
    SCALE_CHOICES,
    TARGET_PROFILE_CHOICES,
    TILE_CHOICES,
)
from pixelup.parameters_help_dialog import ParametersHelpDialog
from pixelup.quit_dialog import QuitConfirmDialog
from pixelup.runner import JobRunner
from pixelup.session_log import configure_session_logging, log
from pixelup.settings_dialog import SettingsDialog
from pixelup.ui_common import (
    apply_palette_fixes,
    apply_scrollbar_style,
    use_regular_spacing,
)
from pixelup.widgets import (
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
    device_combo,
    output_format_combo,
)

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


# Horizontal slack added to a measured string so cell text never touches the
# column edges (covers Qt's default cell margins plus a little breathing room).
_CELL_PADDING = 24
# Minimum readable width for a filename column: a fixed filename column elides to
# this (with a tooltip), and a stretch filename column uses it as its floor.
_NAME_MIN_WIDTH = 180

# How long the Parameters panel waits after the last edit before writing config.json.
# The panel saves as it is edited rather than only at a well-behaved quit, so a crash
# or a force-quit cannot cost the user their parameters — but the quality and denoise
# spin boxes emit a change per typed digit, and every save is an atomic rewrite plus a
# backup record, so the writes are coalesced. Long enough to absorb typing "100" or
# "0.75", short enough that the save is landed by the time the user has moved on.
_PARAMETERS_SAVE_DELAY_MS = 500


def _fit_columns(widget: QWidget, *samples: str) -> int:
    """Width that shows the widest of ``samples`` in ``widget``'s font without
    eliding. Pass the header label plus the worst-case cell strings that must stay
    fully visible; rarer, longer values elide and carry a tooltip."""
    metrics = widget.fontMetrics()
    return max(metrics.horizontalAdvance(text) for text in samples) + _CELL_PADDING


class ImagePreview(QLabel):
    def __init__(self) -> None:
        super().__init__("No image selected")
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # A 320x240 (4:3 QVGA) floor keeps the preview genuinely useful: a
        # scaled-down image is still large enough to judge upscale quality,
        # unlike the previous 160x120 which crushed to a thumbnail.
        self.setMinimumSize(320, 240)

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
        # Create config.json from the built-in defaults on first run so the settings file exists
        # on disk immediately, not only after the first save (storage-path conventions).
        # Create-if-absent — never overwrites an existing file — and before the config is
        # loaded below.
        ensure_app_config()
        # A corrupt config.json quarantines-then-resets rather than crashing startup
        # (storage-path conventions); the loader returns where the corrupt file went so
        # the notice below can tell the user. It is surfaced only after the window is
        # built, so the message box has a real parent.
        load_result = load_app_config_result()
        self.config = load_result.config
        self._config_quarantined_to = load_result.quarantined_to
        # Apply the configured UI font (family-only; the explicit size lives in
        # fonts.py) before building the UI so every widget inherits it. A fresh
        # install resolves the canonical default stack. setFont propagates app-
        # wide, so this is the single place the UI font is established.
        apply_ui_font(QApplication.instance(), self.config.font_family)
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
        # Coalesces the Parameters panel's edits into one save (see
        # _PARAMETERS_SAVE_DELAY_MS). Built before the UI, because building the panel
        # connects the widget-change signals that start it.
        self._parameters_save_timer = QTimer(self)
        self._parameters_save_timer.setSingleShot(True)
        self._parameters_save_timer.setInterval(_PARAMETERS_SAVE_DELAY_MS)
        self._parameters_save_timer.timeout.connect(self._save_parameters)

        self.setWindowTitle("PixelUp")
        self.setAcceptDrops(True)
        self._build_ui()
        # Window minimum = the layout's content-based size hint: the central widget
        # sums the panes' needs, and each table's minimum is its measured column
        # widths plus a filename floor (see _fit_columns). Open a little roomier than
        # that floor so the stretch (filename) columns have slack on first launch.
        # The old fixed resize(1260, …) sat below this minimum, so it was dead.
        hint = self.centralWidget().sizeHint()
        self.setMinimumSize(hint)
        self.resize(hint.width() + 160, hint.height() + 120)
        # The panel opens on what the user last left it at, not on a defaults layer:
        # config.parameters is the persisted panel, and on a fresh install the loader
        # has already filled it with JobSettings() — the built-ins.
        self._apply_job_settings(self.config.parameters)
        self._update_selected_image()
        self._update_action_buttons()

        app = QApplication.instance()
        if app is not None:
            commit = getattr(app, "commitDataRequest", None)
            if commit is not None:
                commit.connect(self._on_commit_data_request)
        log.info(
            "config.loaded",
            path=str(config_path()),
            values=config_log_payload(self.config),
        )
        if self._config_quarantined_to is not None:
            log.warning(
                "config.corrupt_reset",
                path=str(config_path()),
                quarantined_to=str(self._config_quarantined_to),
            )
            # Deferred to the event loop so the notice appears over the shown window
            # (build_app calls window.show() after __init__ returns) instead of
            # blocking construction ahead of the first paint. Non-fatal: the app is
            # already running on freshly reset defaults; this only tells the user.
            QTimer.singleShot(0, self._notify_config_reset)

    def _notify_config_reset(self) -> None:
        if self._config_quarantined_to is not None:
            warn_config_reset(self, self._config_quarantined_to.name)

    def _on_commit_data_request(self, _manager: object) -> None:
        self._session_shutdown = True

    def closeEvent(self, event: QCloseEvent) -> None:
        # Land any edit still inside the debounce window before the app can go away.
        # Safe on every branch below, including a cancelled quit: the user made the
        # edit, and deciding not to quit is no reason to drop it.
        self._flush_parameters_save()
        app = QGuiApplication.instance()
        is_saving_session = app.isSavingSession() if app is not None else False
        if self._session_shutdown or is_saving_session:
            log.info("quit.session_shutdown")
            self.runner.cleanup_for_quit()
            event.accept()
            return
        if not self._images_by_path:
            self.runner.cleanup_for_quit()
            event.accept()
            return
        active = sum(1 for job in self.jobs if job.status in {"pending", "running", "cancelling"})
        if QuitConfirmDialog(active, self).exec() == QDialog.DialogCode.Accepted:
            log.info("quit.confirmed", active_jobs=active)
            self.runner.cleanup_for_quit()
            event.accept()
        else:
            log.info("quit.cancelled")
            event.ignore()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        self.open_paths(paths)

    def open_paths(self, paths: list[Path]) -> None:
        log.info("open.requested", paths=[str(path) for path in paths])
        selected: Path | None = None
        for path in paths:
            resolved = path.expanduser().resolve()
            if not resolved.is_file():
                log.warning("open.ignored_non_file", path=str(path), resolved=str(resolved))
                continue
            if resolved not in self._images_by_path:
                entry = ImageEntry(resolved, _safe_image_size(resolved))
                self._images_by_path[resolved] = entry
                self._image_order.append(resolved)
                self._add_image_row(entry)
                log.info("image.added", input=str(resolved), size=entry.input_size)
            else:
                log.info("image.focused_existing", input=str(resolved))
            selected = resolved
        if selected is not None:
            self._select_image(selected)
        self._update_action_buttons()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        use_regular_spacing(root_layout)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        use_regular_spacing(content_layout, margins=False)
        content_layout.addWidget(self._build_image_panel(), 1)
        content_layout.addWidget(self._build_work_panel(), 1)
        root_layout.addWidget(content, 1)
        self.setCentralWidget(root)

    def _build_window_actions(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        use_regular_spacing(layout, margins=False)

        logs_button = QPushButton("Reveal log")
        logs_button.clicked.connect(self._reveal_log_file)
        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self._settings_dialog)
        settings_shortcut = QKeySequence("Ctrl+,")
        settings_button.setShortcut(settings_shortcut)
        settings_button.setToolTip(
            f"Settings ({settings_shortcut.toString(QKeySequence.SequenceFormat.NativeText)})"
        )
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
        use_regular_spacing(layout, margins=False)

        layout.addWidget(self._build_images_group(), 1)
        layout.addWidget(self._build_selected_image_group(), 1)
        return panel

    def _build_images_group(self) -> QWidget:
        group = QGroupBox("Images")
        layout = QVBoxLayout(group)
        use_regular_spacing(layout)

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
        # Fixed widths measured from the worst-case string each column must show in
        # the current UI font (see _fit_columns): Size holds "99999 x 99999"; Jobs
        # holds a common two-state roll-up and elides + tooltips the rarer longer
        # summaries. "Image" stretches (filename, elides to the floor with a tooltip).
        size_width = _fit_columns(self.image_table, "Size", "99999 x 99999", "unavailable")
        jobs_width = _fit_columns(self.image_table, "Jobs", "12 done, 5 failed")
        image_header.resizeSection(1, size_width)
        image_header.resizeSection(2, jobs_width)
        self.image_table.setMinimumWidth(_NAME_MIN_WIDTH + size_width + jobs_width)
        self.image_table.setMinimumHeight(180)
        layout.addWidget(self.image_table, 1)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        use_regular_spacing(button_layout, margins=False)
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
        use_regular_spacing(layout, margins=False)

        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        use_regular_spacing(controls_layout, margins=False)
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
        use_regular_spacing(layout, margins=False)
        layout.addWidget(self._build_window_actions())
        layout.addWidget(self._build_actions_group())
        return column

    def _build_selected_image_group(self) -> QWidget:
        group = QGroupBox("Preview")
        layout = QVBoxLayout(group)
        use_regular_spacing(layout)
        self.preview = ImagePreview()
        layout.addWidget(self.preview, 1)
        return group

    def _build_models_group(self) -> QWidget:
        group = QGroupBox("Models")
        layout = QVBoxLayout(group)
        use_regular_spacing(layout)
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
        use_regular_spacing(form)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)

        scale_row = QWidget()
        scale_layout = QHBoxLayout(scale_row)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        self.scale_group = QButtonGroup(self)
        # Built from SCALE_CHOICES, like the enumerated combos below: the selectable
        # scales are enumerated once, beside JobSettings, for the panel and the config
        # loader both. The built-in is checked here so the group is never all-unchecked;
        # the startup seed (_apply_job_settings) then applies the persisted value.
        self.scale_buttons: dict[int, QRadioButton] = {}
        for label, scale in SCALE_CHOICES:
            button = QRadioButton(label)
            button.setChecked(scale == DEFAULT_SCALE)
            self.scale_group.addButton(button)
            self.scale_buttons[scale] = button
            scale_layout.addWidget(button)
        scale_layout.addStretch()

        self.face_enhance = QCheckBox("Face enhancement")
        self.denoise_strength = NoWheelDoubleSpinBox()
        self.denoise_strength.setRange(MIN_DENOISE_STRENGTH, MAX_DENOISE_STRENGTH)
        self.denoise_strength.setSingleStep(DENOISE_STRENGTH_STEP)
        self.denoise_strength.setDecimals(2)

        self.alpha_mode = NoWheelComboBox()
        for label, alpha_mode in ALPHA_MODE_CHOICES:
            self.alpha_mode.addItem(label, alpha_mode)

        self.output_format = output_format_combo()

        self.quality = NoWheelSpinBox()
        self.quality.setRange(MIN_QUALITY, MAX_QUALITY)

        self.tile = NoWheelComboBox()
        for label, tile in TILE_CHOICES:
            self.tile.addItem(label, tile)

        self.device = device_combo()

        self.strip_metadata = QCheckBox("Strip metadata")

        self.target_profile = NoWheelComboBox()
        for label, profile in TARGET_PROFILE_CHOICES:
            self.target_profile.addItem(label, profile)

        reset = QPushButton("Reset parameters")
        reset.clicked.connect(self._reset_parameters_to_defaults)

        # Per-control captions live in the Parameters Help dialog, not the panel:
        # always-visible help text was the main driver of the window's minimum
        # width, which had outgrown small screens.
        help_button = QPushButton("Help")
        help_button.clicked.connect(self._parameters_help_dialog)

        form.addRow("Scale", scale_row)
        form.addRow("", self.face_enhance)
        form.addRow("Denoise", self.denoise_strength)
        form.addRow("Alpha mode", self.alpha_mode)
        form.addRow("Output format", self.output_format)
        form.addRow("Quality", self.quality)
        form.addRow("Tile size", self.tile)
        form.addRow("Device", self.device)
        form.addRow("", self.strip_metadata)
        form.addRow("Target profile", self.target_profile)
        # Reset and Help stack as two rows: side by side they would be the widest
        # field in the form and re-widen the group the Help dialog exists to slim.
        form.addRow("", reset)
        form.addRow("", help_button)

        # Every control the panel persists, wired to the debounced save. Connected
        # after the panel is populated so building it (addItem, setChecked) does not
        # schedule a save of values nobody edited.
        for changed in (
            self.scale_group.buttonToggled,
            self.face_enhance.toggled,
            self.denoise_strength.valueChanged,
            self.alpha_mode.currentIndexChanged,
            self.output_format.currentIndexChanged,
            self.quality.valueChanged,
            self.tile.currentIndexChanged,
            self.device.currentIndexChanged,
            self.strip_metadata.toggled,
            self.target_profile.currentIndexChanged,
        ):
            changed.connect(self._parameters_edited)
        return group

    def _build_actions_group(self) -> QWidget:
        group = QGroupBox("Queue actions")
        layout = QVBoxLayout(group)
        use_regular_spacing(layout)
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
        use_regular_spacing(layout)
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
        # Widths measured from worst-case content in the current UI font: Model
        # shows the longest model name in full; Scale fits its header; Status holds
        # "Cancelling...". "Image" elides a long filename to a readable floor (with a
        # tooltip) and "Output" stretches.
        model_width = _fit_columns(self.queue_table, "Model", max(MODEL_ORDER, key=len))
        scale_width = _fit_columns(self.queue_table, "Scale", "4x")
        status_width = _fit_columns(self.queue_table, "Status", "Cancelling...")
        queue_header.resizeSection(0, _NAME_MIN_WIDTH)
        queue_header.resizeSection(1, model_width)
        queue_header.resizeSection(2, scale_width)
        queue_header.resizeSection(4, status_width)
        self.queue_table.setMinimumWidth(
            _NAME_MIN_WIDTH + model_width + scale_width + _NAME_MIN_WIDTH + status_width
        )
        self.queue_table.setMinimumHeight(180)
        layout.addWidget(self.queue_table)
        return group

    def _open_dialog(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Open images")
        log.info("open.dialog_returned", count=len(files))
        self.open_paths([Path(file) for file in files])

    def _settings_dialog(self) -> None:
        # Land any pending panel edit first, so the config the dialog opens on — and
        # carries the unshown settings through from — is the current one. Otherwise a
        # debounce firing behind the modal would be undone on OK.
        self._flush_parameters_save()
        log.info("settings.dialog_opened")
        dialog = SettingsDialog(self.config, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            previous_config = self.config
            self.config = dialog.config()
            save_app_config(self.config)
            # Re-apply the UI font so a changed family takes effect immediately,
            # no restart needed.
            apply_ui_font(QApplication.instance(), self.config.font_family)
            log.info(
                "settings.saved",
                path=str(config_path()),
                previous=config_log_payload(previous_config),
                current=config_log_payload(self.config),
            )
            self.runner.schedule(self.config.max_concurrent_jobs)
            return
        log.info("settings.dialog_cancelled")

    def _parameters_help_dialog(self) -> None:
        log.info("parameters_help.dialog_opened")
        ParametersHelpDialog(self).exec()

    def _about_dialog(self) -> None:
        log.info("about.dialog_opened")
        AboutDialog(self).exec()

    def _reveal_log_file(self) -> None:
        try:
            revealed = _reveal_in_file_browser(self.log_file)
        except OSError as exc:
            log.warning("log.reveal_failed", log_file=str(self.log_file), reason=str(exc))
            return
        if revealed:
            log.info("log.revealed", log_file=str(self.log_file))
        else:
            log.warning("log.reveal_failed", log_file=str(self.log_file))

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
            warn_image_in_use(self)
            return
        row = self._image_rows.pop(path)
        self.image_table.removeRow(row)
        self._images_by_path.pop(path, None)
        self._image_order = [item for item in self._image_order if item != path]
        self._rebuild_image_rows()
        log.info("image.removed", input=str(path))
        self._update_selected_image()
        self._update_action_buttons()

    def _rebuild_image_rows(self) -> None:
        self._image_rows = {}
        for row, path in enumerate(self._image_order):
            self._image_rows[path] = row

    def _selected_models(self) -> list[str]:
        return [model for model, checkbox in self.model_checks.items() if checkbox.isChecked()]

    def _current_scale(self) -> int:
        for scale, button in self.scale_buttons.items():
            if button.isChecked():
                return scale
        return DEFAULT_SCALE

    def current_job_settings(self) -> JobSettings:
        profile = self.target_profile.currentData()
        return JobSettings(
            scale=self._current_scale(),
            face_enhance=self.face_enhance.isChecked(),
            denoise_strength=self.denoise_strength.value(),
            alpha_mode=self.alpha_mode.currentData(),
            device=self.device.currentData(),
            output_format=coerce_output_format(self.output_format.currentData()),
            quality=self.quality.value(),
            tile=self.tile.currentData(),
            strip_metadata=self.strip_metadata.isChecked(),
            target_profile=profile,
        )

    def _apply_job_settings(self, settings: JobSettings) -> None:
        # Checking one button of an exclusive group unchecks the other, so the matching
        # button is all that is set. An out-of-domain scale falls back to the built-in
        # rather than raising: the loader already coerces against SCALE_VALUES, so this
        # only guards a programmatic caller.
        self.scale_buttons.get(settings.scale, self.scale_buttons[DEFAULT_SCALE]).setChecked(True)
        self.face_enhance.setChecked(settings.face_enhance)
        self.denoise_strength.setValue(settings.denoise_strength)
        self.alpha_mode.setCurrentIndex(self.alpha_mode.findData(settings.alpha_mode))
        self.output_format.setCurrentIndex(
            self.output_format.findData(coerce_output_format(settings.output_format).value)
        )
        self.quality.setValue(settings.quality)
        self.tile.setCurrentIndex(self.tile.findData(settings.tile))
        self.device.setCurrentIndex(self.device.findData(settings.device))
        self.strip_metadata.setChecked(settings.strip_metadata)
        self.target_profile.setCurrentIndex(self.target_profile.findData(settings.target_profile))
        # Applying settings is not a user edit: the setters above fire the panel's
        # change signals (a freshly built spin box sits at 0, so seeding it to the
        # persisted value reads as a change), which would leave a save armed for values
        # nobody touched. Reset flushes explicitly, so cancelling here costs it nothing.
        self._parameters_save_timer.stop()

    def _parameters_edited(self) -> None:
        self._parameters_save_timer.start()

    def _flush_parameters_save(self) -> None:
        """Save any pending panel edit now, cancelling the debounce."""
        self._parameters_save_timer.stop()
        self._save_parameters()

    def _save_parameters(self) -> None:
        """Persist the Parameters panel as the user has it.

        The panel is durable user intent, so it lives in config.json like every other
        preference. This never touches an already-queued job: a job snapshots the panel
        through current_job_settings() at the moment it is created (see _enqueue_jobs),
        and holds its own frozen JobSettings from then on. Saving is a no-op when the
        panel already matches what is on disk, which is what makes it safe to call from
        the startup apply, from close, and from the settings dialog.
        """
        parameters = self.current_job_settings()
        if parameters == self.config.parameters:
            return
        self.config = replace(self.config, parameters=parameters)
        save_app_config(self.config)
        log.info(
            "parameters.saved",
            path=str(config_path()),
            values=job_settings_log_payload(parameters),
        )

    def _reset_parameters_to_defaults(self) -> None:
        """Restore the panel to PixelUp's built-in parameters.

        ``JobSettings()`` is the built-ins, and the only source of them — not the
        user's persisted config, which is what the reset exists to get *away* from.
        The restored values are then persisted like any other panel edit: pressing
        reset is a decision, so it is flushed rather than left to the debounce.
        """
        defaults = JobSettings()
        self._apply_job_settings(defaults)
        log.info("parameters.reset", defaults=job_settings_log_payload(defaults))
        self._flush_parameters_save()

    def _queue_selected_image(self) -> None:
        self._enqueue_jobs(self._selected_paths(), self._selected_models())

    def _queue_selected_image_all_models(self) -> None:
        self._enqueue_jobs(self._selected_paths(), list(UPSCALE_MODELS))

    def _queue_all_images_selected_models(self) -> None:
        self._enqueue_jobs(list(self._image_order), self._selected_models())

    def _queue_all_images_all_models(self) -> None:
        self._enqueue_jobs(list(self._image_order), list(UPSCALE_MODELS))

    def _enqueue_jobs(self, input_paths: list[Path], models: list[str]) -> None:
        if not input_paths:
            warn_no_images(self)
            return
        if not models:
            warn_no_models(self)
            return
        # The enqueue snapshot: the panel captured whole, scale included, at the moment
        # the user pressed the button. Later panel edits do not reach these jobs.
        settings = self.current_job_settings()
        new_jobs = create_jobs(
            input_paths=input_paths,
            models=models,
            settings=settings,
            existing_jobs=self.jobs,
            auto_download=self.config.auto_download,
            job_ids=self._job_ids,
        )
        log.info(
            "enqueue.requested",
            inputs=[str(path) for path in input_paths],
            models=models,
            settings=job_settings_log_payload(settings),
            auto_download=self.config.auto_download,
        )
        for job in new_jobs:
            log.debug("job.queued", job_id=job.id, details=job_log_payload(job))
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
            self.queue_table.setItem(row, 2, _item(f"{job.settings.scale}x"))
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
        log.info(
            "queue.cancelled",
            cancelled_pending=cancelled_pending,
            signalled_running=signalled_running,
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
            log.info("jobs.retried", job_ids=retried_jobs)
            self._refresh_image_job_summaries()
            self._update_action_buttons()
            self.runner.schedule(self.config.max_concurrent_jobs)

    @Slot(int, str)
    def _job_progress(self, job_id: int, message: str) -> None:
        job = self._find_job(job_id)
        if message != job.message:
            log.debug(
                "job.progress",
                job_id=job.id,
                input=str(job.input_path),
                model=job.model,
                output=str(job.output_path),
                text=message,
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
            summary = job_status_summary(statuses_by_input[path])
            item = self.image_table.item(row, 2)
            item.setText(summary)
            item.setToolTip(summary)
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


def _item(text: str, *, tooltip: str | None = None) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    if tooltip:
        item.setToolTip(tooltip)
    return item


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


def _reveal_in_file_browser(path: Path) -> bool:
    """Reveal a path in the OS file browser, returning whether it succeeded.

    The macOS path and the cross-platform fallback report a real success/failure
    signal. Windows ``explorer.exe`` returns a non-zero exit code even when it
    succeeds, so its return code can't signal failure; there, launching without
    an ``OSError`` is the best signal available. Callers handle ``OSError`` (e.g.
    the helper binary is missing) and log accordingly.
    """
    target = path if path.exists() else path.parent
    if sys.platform == "darwin":
        return subprocess.run(["open", "-R", str(target)], check=False).returncode == 0
    if sys.platform == "win32":
        subprocess.run(["explorer", f"/select,{target}"], check=False)
        return True
    if target.is_file():
        target = target.parent
    return bool(QDesktopServices.openUrl(QUrl.fromLocalFile(str(target))))


def build_app(
    argv: list[str],
    *,
    log_file: Path | None = None,
    runtime_dirs: RuntimeDirs | None = None,
) -> tuple[QApplication, MainWindow]:
    """Build the fully-wired application and main window, ready to run.

    Everything ``main`` does except the blocking ``app.exec()`` lives here so it is exercisable
    headlessly: session logging, the runtime dirs, the QApplication and its style/icon, the window,
    and opening any image paths passed on the command line. ``log_file`` and ``runtime_dirs`` are
    injectable so a test can point them at a temp location; both default to the real resolution.
    """
    register_image_plugins()
    resolved_log_file = configure_session_logging(log_file)
    resolved_runtime_dirs = runtime_dirs if runtime_dirs is not None else resolve_runtime_dirs()
    app = QApplication.instance() or QApplication(argv)
    if "Fusion" in QStyleFactory.keys():
        app.setStyle("Fusion")
    # PixelUp follows the OS light/dark theme; it owns no colours, only two
    # corrections to what Fusion resolves (apply_palette_fixes).
    #
    # Applied once, at launch, and NOT re-applied when the OS theme changes mid-run.
    # That is deliberate, not an oversight: flip the theme with PixelUp open and the
    # window follows, but these corrections revert until relaunch. Accepted — this is
    # an upscaler you open, run and close. A colorSchemeChanged handler was tried and
    # removed: Qt re-derives the palette after the signal and silently undid the
    # ButtonText fix, so it only looked like it worked.
    apply_palette_fixes(app)
    # The scroll-bar QSS is palette-based so it stays consistent with whatever
    # theme the OS resolves, while replacing Fusion's thick square bar.
    apply_scrollbar_style(app)
    app.setApplicationName("PixelUp")
    app.setApplicationDisplayName("PixelUp")
    icon_path = Path(__file__).parent / "resources" / "icon.png"
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))
    log.info(
        "app.started",
        version=__version__,
        python=sys.version.split()[0],
        platform=sys.platform,
        log_file=str(resolved_log_file),
        runtime_dirs={
            "models_dir": str(resolved_runtime_dirs.models_dir),
            "temp_dir": str(resolved_runtime_dirs.temp_dir),
        },
        argv=argv[1:],
    )

    window = MainWindow(log_file=resolved_log_file)
    window.show()
    paths = [Path(arg) for arg in argv[1:] if not arg.startswith("-")]
    if paths:
        window.open_paths(paths)
    return app, window


def _selftest() -> int:
    """Import the full runtime stack and exit — proof that a build resolved every
    dependency, without downloading models or running inference.

    The inference libraries (torch, realesrgan, basicsr, gfpgan, facexlib, cv2) are
    imported lazily deep inside functions, so a frozen PyInstaller bundle can build
    and even launch while silently missing one of them — the gap only surfaces when a
    user first upscales. Running the packaged binary with PIXELUP_SELFTEST=1 forces
    every such import up front, so CI/packaging catches a missing hidden-import here
    rather than in the user's hands. Import-only: no window, no network, no weights.
    """
    import importlib

    for module in (
        "torch",
        "torchvision",
        "cv2",
        "numpy",
        "PIL",
        "pillow_heif",
        "filelock",
        "realesrgan",
        "realesrgan.archs.srvgg_arch",
        "basicsr",
        "basicsr.archs.rrdbnet_arch",
        "gfpgan",
        "facexlib",
        "PySide6.QtWidgets",
    ):
        importlib.import_module(module)
    return 0


def main() -> int:
    if os.environ.get("PIXELUP_SELFTEST") == "1":
        return _selftest()
    app, _window = build_app(sys.argv)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
