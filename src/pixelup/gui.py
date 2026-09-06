from __future__ import annotations

import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from itertools import count
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QSize, Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import (
    QAccessible,
    QAccessibleEvent,
    QCloseEvent,
    QDesktopServices,
    QDragEnterEvent,
    QDragLeaveEvent,
    QDragMoveEvent,
    QDropEvent,
    QGuiApplication,
    QIcon,
    QKeySequence,
    QPalette,
    QPixmap,
    QResizeEvent,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QStyleFactory,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pixelup import __version__
from pixelup.about_dialog import AboutDialog
from pixelup.app_config import (
    AppConfig,
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
from pixelup.managed_models_dialog import ManagedModelsDialog
from pixelup.message_dialogs import (
    show_startup_failure,
    warn_config_reset,
    warn_jobs_stopping,
)
from pixelup.model_management import (
    UPSCALE_MODELS,
    required_artifact_names,
)
from pixelup.model_manager import ModelManager
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
from pixelup.paths import absolute_user_path
from pixelup.quit_dialog import QuitConfirmDialog
from pixelup.runner import JobRunner
from pixelup.session_log import configure_session_logging, log
from pixelup.settings_dialog import SettingsDialog
from pixelup.shortcuts_dialog import ShortcutsDialog
from pixelup.ui_common import (
    apply_palette_fixes,
    apply_scrollbar_style,
    use_regular_spacing,
)
from pixelup.widgets import (
    EmptyStateTableWidget,
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
    OperationResult,
    device_combo,
    output_format_combo,
)

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
_REVEAL_TIMEOUT_SECONDS = 5


def _managed_models_warning_style(palette: QPalette) -> str:
    """Return a complete warning-button treatment for the current theme."""
    dark = palette.color(QPalette.ColorRole.Window).lightness() < 128
    if dark:
        background, hover, pressed, border, foreground = (
            "#8a5a00",
            "#a86d00",
            "#704900",
            "#d89a28",
            "#ffffff",
        )
    else:
        background, hover, pressed, border, foreground = (
            "#f2c94c",
            "#ffd86b",
            "#dbae30",
            "#a66b00",
            "#2a1d00",
        )
    return (
        "QPushButton {"
        f" background-color: {background}; color: {foreground};"
        f" border: 1px solid {border}; border-radius: 3px;"
        " padding: 4px 10px; font-weight: 600;"
        "}"
        f"QPushButton:hover {{ background-color: {hover}; }}"
        f"QPushButton:pressed {{ background-color: {pressed}; }}"
    )


@dataclass(frozen=True, slots=True)
class PendingEnqueue:
    input_paths: tuple[Path, ...]
    models: tuple[str, ...]
    settings: JobSettings
    required_artifacts: tuple[str, ...]

    @property
    def job_count(self) -> int:
        return len(self.input_paths) * len(self.models)


@dataclass(frozen=True, slots=True)
class PendingRetry:
    job_ids: frozenset[int]
    required_artifacts: tuple[str, ...]

    @property
    def job_count(self) -> int:
        return len(self.job_ids)


PendingModelWork = PendingEnqueue | PendingRetry
_WINDOW_TARGET_EXTRA_WIDTH = 160
_WINDOW_TARGET_EXTRA_HEIGHT = 120


def bounded_initial_window_size(minimum: QSize, work_area: QSize) -> QSize:
    """Roomy launch target capped to the current desktop work area.

    The content-derived minimum remains authoritative: on a work area smaller
    than that floor Qt/OS policy decides placement, but this function never
    weakens the floor merely to make a preferred startup target fit.
    """
    return QSize(
        max(minimum.width(), min(minimum.width() + _WINDOW_TARGET_EXTRA_WIDTH, work_area.width())),
        max(
            minimum.height(),
            min(minimum.height() + _WINDOW_TARGET_EXTRA_HEIGHT, work_area.height()),
        ),
    )


def announce_accessible_alert(widget: QWidget) -> None:
    """Announce a newly changed actionable result through Qt accessibility."""
    QAccessible.updateAccessibility(QAccessibleEvent(widget, QAccessible.Event.Alert))


def local_drop_paths(urls: Iterable[QUrl]) -> list[Path]:
    """Every literal local path delivered by an external drag."""
    return [absolute_user_path(Path(url.toLocalFile())) for url in urls if url.isLocalFile()]


def has_local_drop_offer(urls: Iterable[QUrl]) -> bool:
    """Whether hover metadata contains a local path, without touching the filesystem."""
    return any(url.isLocalFile() for url in urls)


class ImageDropTable(EmptyStateTableWidget):
    """The image collection's native external-drop boundary and visible destination cue."""

    paths_dropped = Signal(list)

    def __init__(self, rows: int, columns: int, *, empty_text: str) -> None:
        super().__init__(rows, columns, empty_text=empty_text)
        self.setObjectName("imageDropReceiver")
        self.setAcceptDrops(True)
        self.setProperty("dropActive", False)
        self.setStyleSheet(
            'QTableWidget#imageDropReceiver[dropActive="true"] {'
            " border: 3px solid palette(highlight);"
            "}"
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        self._update_drag(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        self._update_drag(event)

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        self._set_drop_active(False)
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:
        self._set_drop_active(False)
        paths = local_drop_paths(event.mimeData().urls())
        if not paths:
            event.ignore()
            return
        self.paths_dropped.emit(paths)
        event.acceptProposedAction()

    def hideEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._set_drop_active(False)
        super().hideEvent(event)

    def _update_drag(self, event: QDragEnterEvent | QDragMoveEvent) -> None:
        active = has_local_drop_offer(event.mimeData().urls())
        self._set_drop_active(active)
        if active:
            event.acceptProposedAction()
        else:
            event.ignore()

    def _set_drop_active(self, active: bool) -> None:
        if self.property("dropActive") != active:
            self.setProperty("dropActive", active)
            self.style().unpolish(self)
            self.style().polish(self)
            self.viewport().update()


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
    def __init__(self, *, log_file: Path, runtime_dirs: RuntimeDirs | None = None) -> None:
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
        self.runtime_dirs = runtime_dirs or resolve_runtime_dirs()
        self._job_ids = count(1)
        self._images_by_path: dict[Path, ImageEntry] = {}
        self._image_order: list[Path] = []
        self._image_rows: dict[Path, int] = {}
        self._queue_rows: dict[int, int] = {}
        self._queue_failure_count = 0
        self.model_manager = ModelManager(self.runtime_dirs.models_dir, self)
        self._active_models_dialog: ManagedModelsDialog | None = None
        self._pending_model_work: PendingModelWork | None = None
        self.jobs: list[Job] = []
        self.runner = JobRunner(self.jobs, self, runtime_dirs=self.runtime_dirs)
        self.runner.progress.connect(self._job_progress)
        self.runner.finished.connect(self._job_finished)
        self.runner.idle.connect(self._close_when_workers_stop)
        self._quit_when_workers_idle = False
        self._session_shutdown = False
        # Coalesces the Parameters panel's edits into one save (see
        # _PARAMETERS_SAVE_DELAY_MS). Built before the UI, because building the panel
        # connects the widget-change signals that start it.
        self._parameters_save_timer = QTimer(self)
        self._parameters_save_timer.setSingleShot(True)
        self._parameters_save_timer.setInterval(_PARAMETERS_SAVE_DELAY_MS)
        self._parameters_save_timer.timeout.connect(self._save_parameters)

        self.setWindowTitle("PixelUp")
        self._build_ui()
        self.model_manager.changed.connect(self._model_manager_changed)
        self.model_manager.cancelled.connect(self._model_install_cancelled)
        self.model_manager.idle.connect(self._close_when_workers_stop)
        self._bind_shortcuts()
        # Window minimum = the layout's content-based size hint: the central widget
        # sums the panes' needs, and each table's minimum is its measured column
        # widths plus a filename floor (see _fit_columns). Open a little roomier than
        # that floor so the stretch (filename) columns have slack on first launch.
        # The old fixed resize(1260, …) sat below this minimum, so it was dead.
        hint = self._refresh_layout_metrics()
        screen = self.screen() or QApplication.primaryScreen()
        work_area = screen.availableGeometry().size() if screen is not None else QSize(
            hint.width() + _WINDOW_TARGET_EXTRA_WIDTH,
            hint.height() + _WINDOW_TARGET_EXTRA_HEIGHT,
        )
        self.resize(bounded_initial_window_size(hint, work_area))
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
            warn_config_reset(self)

    def _on_commit_data_request(self, _manager: object) -> None:
        self._session_shutdown = True

    def closeEvent(self, event: QCloseEvent) -> None:
        app = QGuiApplication.instance()
        is_saving_session = app.isSavingSession() if app is not None else False
        session_shutdown = self._session_shutdown or is_saving_session
        if self._quit_when_workers_idle:
            if self._workers_clean_for_quit():
                event.accept()
            else:
                event.ignore()
            return

        # Land any edit still inside the debounce window before the app can go away.
        # A direct close stays open on failure so the user can retry; OS session
        # shutdown remains non-modal and records the failure in the session log.
        if not self._flush_parameters_save(surface_failure=not session_shutdown):
            if not session_shutdown:
                event.ignore()
                return
        if session_shutdown:
            log.info("quit.session_shutdown")
            self._finish_or_defer_close(event, surface_wait=False)
            return
        if not self._images_by_path:
            self._finish_or_defer_close(event, surface_wait=True)
            return
        active = sum(1 for job in self.jobs if job.status in {"pending", "running", "cancelling"})
        if QuitConfirmDialog(active, self).exec() == QDialog.DialogCode.Accepted:
            log.info("quit.confirmed", active_jobs=active)
            self._finish_or_defer_close(event, surface_wait=True)
        else:
            log.info("quit.cancelled")
            event.ignore()

    def _finish_or_defer_close(self, event: QCloseEvent, *, surface_wait: bool) -> None:
        self.runner.begin_shutdown()
        self.model_manager.begin_shutdown()
        if self._workers_clean_for_quit():
            event.accept()
            return
        self._quit_when_workers_idle = True
        event.ignore()
        if surface_wait:
            warn_jobs_stopping(self)

    def _close_when_workers_stop(self) -> None:
        if self._quit_when_workers_idle and self._workers_clean_for_quit():
            QTimer.singleShot(0, self.close)

    def _workers_clean_for_quit(self) -> bool:
        return self.runner.cleanup_for_quit() and self.model_manager.cleanup_for_quit()

    def open_paths(self, paths: list[Path]) -> None:
        log.info("open.requested", paths=[str(path) for path in paths])
        selected: Path | None = None
        added: list[Path] = []
        duplicates: list[Path] = []
        rejected: list[tuple[Path, str, bool]] = []
        for path in paths:
            try:
                input_path = absolute_user_path(path)
                resolved = input_path.resolve()
                if resolved.is_dir():
                    log.warning("open.ignored_directory", path=str(path), resolved=str(resolved))
                    rejected.append((input_path, "folders are not supported", False))
                    continue
                if not resolved.is_file():
                    log.warning("open.ignored_non_file", path=str(path), resolved=str(resolved))
                    rejected.append((input_path, "the file is unavailable", False))
                    continue
                image_size = _safe_image_size(resolved)
                if image_size is None:
                    log.warning("open.ignored_non_image", path=str(path), resolved=str(resolved))
                    rejected.append((input_path, "not a readable supported image", False))
                    continue
                if input_path not in self._images_by_path:
                    entry = ImageEntry(input_path, image_size)
                    self._images_by_path[input_path] = entry
                    self._image_order.append(input_path)
                    self._add_image_row(entry)
                    log.info("image.added", input=str(input_path), size=entry.input_size)
                    added.append(input_path)
                else:
                    log.info("image.focused_existing", input=str(input_path))
                    duplicates.append(input_path)
                selected = input_path
            except Exception:  # noqa: BLE001 - one bad offer must not escape the UI event loop.
                log.exception("open.failed", path=str(path))
                rejected.append((path, "could not be read", True))
        if selected is not None:
            self._select_image(selected)
        self._update_action_buttons()
        self._show_open_outcome(added, duplicates, rejected)

    def _show_open_outcome(
        self,
        added: list[Path],
        duplicates: list[Path],
        rejected: list[tuple[Path, str, bool]],
    ) -> None:
        if rejected:
            parts = []
            if added:
                parts.append(f"Added {len(added)} image{'s' if len(added) != 1 else ''}")
            if duplicates:
                parts.append(f"Already open: {', '.join(path.name for path in duplicates)}")
            parts.extend(f"{path.name or path}: {reason}" for path, reason, _error in rejected)
            self._set_open_result(
                "; ".join(parts) + ".",
                severity="error" if any(error for _path, _reason, error in rejected) else "warning",
                issue_paths=[*duplicates, *(path for path, _reason, _error in rejected)],
            )
        elif duplicates:
            self._set_open_result(
                "Already open: " + ", ".join(path.name for path in duplicates) + ".",
                severity="information",
                issue_paths=duplicates,
            )
        elif self._open_result_issue_paths and self._open_result_issue_paths.issubset(set(added)):
            self._dismiss_open_result()

    def _set_open_result(
        self,
        message: str,
        *,
        severity: Literal["information", "warning", "error"],
        issue_paths: list[Path],
    ) -> None:
        self._open_result_issue_paths = frozenset(issue_paths)
        self.open_result.show_result(message, severity=severity)

    def _dismiss_open_result(self) -> None:
        self._open_result_issue_paths = frozenset()
        self.open_result.clear_result()

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
        layout = QGridLayout(row)
        use_regular_spacing(layout, margins=False)

        self.logs_button = QPushButton("Reveal log")
        self.logs_button.clicked.connect(self._reveal_log_file)
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self._settings_dialog)
        settings_shortcut = QKeySequence("Ctrl+,")
        self.settings_button.setShortcut(settings_shortcut)
        self.settings_button.setToolTip(
            f"Settings ({settings_shortcut.toString(QKeySequence.SequenceFormat.NativeText)})"
        )
        self.shortcuts_button = QPushButton("Shortcuts")
        self.shortcuts_button.clicked.connect(self._shortcuts_dialog)
        self.about_button = QPushButton("About")
        self.about_button.clicked.connect(self._about_dialog)

        layout.addWidget(self.logs_button, 0, 0)
        layout.addWidget(self.settings_button, 0, 1)
        layout.addWidget(self.shortcuts_button, 1, 0)
        layout.addWidget(self.about_button, 1, 1)
        self.log_action_result = OperationResult(
            object_name="logActionResult",
            dismissible=True,
        )
        layout.addWidget(self.log_action_result, 2, 0, 1, 2)
        return row

    def _bind_shortcuts(self) -> None:
        self.shortcuts_shortcut = QShortcut(QKeySequence("Ctrl+/"), self)
        self.shortcuts_shortcut.activated.connect(self._shortcuts_dialog)
        self.shortcuts_question_shortcut = QShortcut(QKeySequence("Ctrl+?"), self)
        self.shortcuts_question_shortcut.activated.connect(self._shortcuts_dialog)

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

        self.image_table = ImageDropTable(
            0,
            3,
            empty_text="No images yet. Open images or drop them here.",
        )
        self.image_table.paths_dropped.connect(self.open_paths)
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
        self._fit_image_table_columns()
        self.image_table.setMinimumHeight(180)
        layout.addWidget(self.image_table, 1)

        self._open_result_issue_paths: frozenset[Path] = frozenset()
        self.open_result = OperationResult(object_name="openResult", dismissible=True)
        self.open_result.dismiss_button.clicked.connect(self._dismiss_open_result)
        layout.addWidget(self.open_result)

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
        self._remove_result_path: Path | None = None
        self.remove_result = OperationResult(
            object_name="removeImageResult",
            dismissible=True,
        )
        layout.addWidget(self.remove_result)
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
        layout.addWidget(self._build_actions_group())
        layout.addWidget(self._build_window_actions())
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
        self.manage_models_button = QPushButton("Managed models")
        self.manage_models_button.clicked.connect(self._managed_models_dialog)
        layout.addWidget(self.manage_models_button)
        self._refresh_model_rollup()
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
        self.parameters_result = OperationResult(
            object_name="parametersSaveResult",
            dismissible=True,
        )
        form.addRow(self.parameters_result)

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
        self._queue_action_issue: Literal["images", "models"] | None = None
        self.queue_action_result = OperationResult(
            object_name="queueActionResult",
            dismissible=True,
        )
        layout.addWidget(self.queue_action_result)
        layout.addStretch()
        return group

    def _build_queue_panel(self) -> QWidget:
        group = QGroupBox("Queue")
        layout = QVBoxLayout(group)
        use_regular_spacing(layout)
        self.queue_table = EmptyStateTableWidget(0, 5, empty_text="No jobs queued yet.")
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
        self._fit_queue_table_columns()
        self.queue_table.setMinimumHeight(180)

        self.queue_failure_result = QFrame()
        self.queue_failure_result.setObjectName("queueFailureResult")
        self.queue_failure_result.setStyleSheet(
            "QFrame#queueFailureResult {"
            " border: 1px solid #c0392b;"
            " border-radius: 5px;"
            " background: palette(base);"
            "}"
        )
        failure_layout = QHBoxLayout(self.queue_failure_result)
        failure_layout.setContentsMargins(10, 7, 10, 7)
        failure_layout.setSpacing(8)
        self.queue_failure_label = QLabel()
        self.queue_failure_label.setWordWrap(True)
        failure_layout.addWidget(self.queue_failure_label, 1)
        self.queue_failure_result.hide()
        layout.addWidget(self.queue_failure_result)
        layout.addWidget(self.queue_table)
        return group

    def _fit_image_table_columns(self) -> None:
        header = self.image_table.horizontalHeader()
        size_width = _fit_columns(self.image_table, "Size", "99999 x 99999", "Unavailable")
        jobs_width = _fit_columns(self.image_table, "Jobs", "12 done, 5 failed")
        header.resizeSection(1, size_width)
        header.resizeSection(2, jobs_width)
        self.image_table.setMinimumWidth(_NAME_MIN_WIDTH + size_width + jobs_width)

    def _fit_queue_table_columns(self) -> None:
        header = self.queue_table.horizontalHeader()
        model_width = _fit_columns(self.queue_table, "Model", max(UPSCALE_MODELS, key=len))
        scale_width = _fit_columns(self.queue_table, "Scale", "4x")
        status_width = _fit_columns(self.queue_table, "Status", "Cancelling...")
        header.resizeSection(0, _NAME_MIN_WIDTH)
        header.resizeSection(1, model_width)
        header.resizeSection(2, scale_width)
        header.resizeSection(4, status_width)
        self.queue_table.setMinimumWidth(
            _NAME_MIN_WIDTH + model_width + scale_width + _NAME_MIN_WIDTH + status_width
        )

    def _refresh_layout_metrics(self) -> QSize:
        """Re-measure font-dependent chrome and publish the resulting floor."""
        self._fit_image_table_columns()
        self._fit_queue_table_columns()
        central = self.centralWidget()
        central.updateGeometry()
        if central.layout() is not None:
            central.layout().invalidate()
            central.layout().activate()
        hint = central.sizeHint()
        self.setMinimumSize(hint)
        return hint

    def _open_dialog(self) -> None:
        try:
            files, _ = QFileDialog.getOpenFileNames(self, "Open images")
        except Exception:  # noqa: BLE001 - native picker failures belong to this action.
            log.exception("open.picker_failed")
            self.open_result.show_result(
                "Could not open the image picker. Try again.",
                severity="error",
            )
            return
        log.info("open.dialog_returned", count=len(files))
        self.open_paths([Path(file) for file in files])

    def _settings_dialog(self) -> None:
        # Land any pending panel edit first, so the config the dialog opens on — and
        # carries the unshown settings through from — is the current one. Otherwise a
        # debounce firing behind the modal would be undone on OK.
        if not self._flush_parameters_save():
            return
        log.info("settings.dialog_opened")
        previous_config = self.config
        dialog = SettingsDialog(
            self.config,
            self,
            try_save=lambda candidate: self._save_config_candidate(
                candidate,
                surface_failure=False,
            ),
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Re-apply the UI font so a changed family takes effect immediately,
            # no restart needed.
            app = QApplication.instance()
            apply_ui_font(app, self.config.font_family)
            self.setFont(app.font())
            self._refresh_layout_metrics()
            log.info(
                "settings.saved",
                path=str(config_path()),
                previous=config_log_payload(previous_config),
                current=config_log_payload(self.config),
            )
            self.runner.schedule(self.config.max_concurrent_jobs)
            return
        log.info("settings.dialog_cancelled")

    def _managed_models_dialog(self) -> None:
        self.model_manager.refresh_readiness()
        pending = self._pending_model_work
        self._open_models_dialog(
            required_artifacts=pending.required_artifacts if pending is not None else (),
            pending_job_count=pending.job_count if pending is not None else 0,
        )

    def _open_models_dialog(
        self,
        *,
        required_artifacts: tuple[str, ...] = (),
        pending_job_count: int = 0,
    ) -> None:
        if self._active_models_dialog is not None:
            self._active_models_dialog.raise_()
            self._active_models_dialog.activateWindow()
            return

        log.info("models.dialog_opened")
        dialog = ManagedModelsDialog(
            self.model_manager,
            self,
            required_artifacts=required_artifacts,
            pending_job_count=pending_job_count,
        )
        self._active_models_dialog = dialog
        dialog.finished.connect(lambda _result: self._models_dialog_finished(dialog))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _models_dialog_finished(self, dialog: ManagedModelsDialog) -> None:
        if self._active_models_dialog is dialog:
            self._active_models_dialog = None
        if (
            self._pending_model_work is not None
            and not self.model_manager.active_for(
                self._pending_model_work.required_artifacts
            )
        ):
            log.info("models.pending_work_abandoned")
            self._pending_model_work = None
        self._refresh_model_rollup()
        log.info("models.dialog_closed")
        dialog.deleteLater()

    def _model_manager_changed(self) -> None:
        self._refresh_model_rollup()
        pending = self._pending_model_work
        if pending is None:
            return
        if self.model_manager.active_for(
            pending.required_artifacts
        ) or self.model_manager.missing(pending.required_artifacts):
            return

        self._pending_model_work = None
        dialog = self._active_models_dialog
        if dialog is not None:
            dialog.accept()
        if isinstance(pending, PendingEnqueue):
            self._materialize_jobs(
                list(pending.input_paths),
                list(pending.models),
                pending.settings,
            )
        else:
            self._retry_failed_snapshot(pending.job_ids)

    @Slot(object)
    def _model_install_cancelled(self, artifact_names: object) -> None:
        pending = self._pending_model_work
        if not isinstance(artifact_names, tuple) or pending is None:
            return
        if not set(pending.required_artifacts).intersection(artifact_names):
            return
        self._pending_model_work = None
        dialog = self._active_models_dialog
        if dialog is not None:
            dialog.reject()
        log.info("models.pending_work_cancelled")

    def _refresh_model_rollup(self) -> None:
        ready, total = self.model_manager.ready_count()
        missing = ready < total
        if missing:
            self.manage_models_button.setStyleSheet(
                _managed_models_warning_style(self.palette())
            )
        else:
            self.manage_models_button.setStyleSheet("")
        if self.model_manager.active_operations:
            completed, operation_total = self.model_manager.aggregate_progress()
            progress = (
                0
                if operation_total <= 0
                else min(100, completed * 100 // operation_total)
            )
            self.manage_models_button.setText(f"Installing models — {progress}%")
            self.manage_models_button.setAccessibleName(
                f"Installing models, {progress} percent"
            )
            self.manage_models_button.setToolTip("Model installation is in progress.")
            return
        if missing:
            self.manage_models_button.setText("Models are missing")
            self.manage_models_button.setAccessibleName("Models are missing")
            self.manage_models_button.setToolTip(
                "Some models are not installed. Open Managed models to install them."
            )
        else:
            self.manage_models_button.setText("Managed models")
            self.manage_models_button.setAccessibleName(
                "Managed models, all models installed"
            )
            self.manage_models_button.setToolTip("All models are installed.")

    def _parameters_help_dialog(self) -> None:
        log.info("parameters_help.dialog_opened")
        ParametersHelpDialog(self).exec()

    def _shortcuts_dialog(self) -> None:
        log.info("shortcuts.dialog_opened")
        ShortcutsDialog(self).exec()

    def _about_dialog(self) -> None:
        log.info("about.dialog_opened")
        AboutDialog(self).exec()

    def _reveal_log_file(self) -> None:
        try:
            revealed = _reveal_in_file_browser(self.log_file)
        except (OSError, subprocess.TimeoutExpired):
            log.exception("log.reveal_failed", log_file=str(self.log_file))
            self.log_action_result.show_result(
                "Could not reveal the log. Try again from this button.",
                severity="error",
            )
            return
        if revealed:
            log.info("log.revealed", log_file=str(self.log_file))
            self.log_action_result.clear_result()
        else:
            log.warning("log.reveal_failed", log_file=str(self.log_file))
            self.log_action_result.show_result(
                "Could not reveal the log. Try again from this button.",
                severity="error",
            )

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
        if self._remove_result_path is not None and (
            path != self._remove_result_path
            or not self._has_active_jobs(self._remove_result_path)
        ):
            self._remove_result_path = None
            self.remove_result.clear_result()
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
            log.info("image.remove_blocked", input=str(path), reason="active_jobs")
            self._remove_result_path = path
            self.remove_result.show_result(
                "This image cannot be removed while pending or running jobs still use it.",
                severity="warning",
            )
            return
        self._remove_result_path = None
        self.remove_result.clear_result()
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

    def _flush_parameters_save(self, *, surface_failure: bool = True) -> bool:
        """Save any pending panel edit now, cancelling the debounce."""
        self._parameters_save_timer.stop()
        return self._save_parameters(surface_failure=surface_failure)

    def _save_parameters(self, *, surface_failure: bool = True) -> bool:
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
            self.parameters_result.clear_result()
            return True
        candidate = replace(self.config, parameters=parameters)
        if not self._save_config_candidate(candidate, surface_failure=surface_failure):
            return False
        log.info(
            "parameters.saved",
            path=str(config_path()),
            values=job_settings_log_payload(parameters),
        )
        self.parameters_result.clear_result()
        return True

    def _save_config_candidate(
        self,
        candidate: AppConfig,
        *,
        surface_failure: bool = True,
    ) -> bool:
        try:
            save_app_config(candidate)
        except Exception as exc:  # noqa: BLE001 - persistence failure must remain in the UI.
            log.warning(
                "config.save_failed",
                path=str(config_path()),
                reason=str(exc),
            )
            if surface_failure:
                self.parameters_result.show_result(
                    "PixelUp could not save these parameters. Your changes are still shown; "
                    "try again.",
                    severity="error",
                )
            return False
        self.config = candidate
        return True

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
            log.info("enqueue.rejected", reason="no_images")
            self._queue_action_issue = "images"
            self.queue_action_result.show_result(
                "Open or select at least one image before queueing.",
                severity="warning",
            )
            return
        if not models:
            log.info("enqueue.rejected", reason="no_models")
            self._queue_action_issue = "models"
            self.queue_action_result.show_result(
                "Select at least one model before queueing.",
                severity="warning",
            )
            return
        self._queue_action_issue = None
        self.queue_action_result.clear_result()
        # The enqueue snapshot: the panel captured whole, scale included, at the moment
        # the user pressed the button. Later panel edits do not reach these jobs.
        settings = self.current_job_settings()
        required = required_artifact_names(
            models,
            face_enhance=settings.face_enhance,
            denoise_strength=settings.denoise_strength,
        )
        self.model_manager.refresh_readiness()
        missing = self.model_manager.missing(required)
        if missing:
            pending = PendingEnqueue(
                input_paths=tuple(input_paths),
                models=tuple(models),
                settings=settings,
                required_artifacts=required,
            )
            self._pending_model_work = pending
            self._open_models_dialog(
                required_artifacts=pending.required_artifacts,
                pending_job_count=pending.job_count,
            )
            return
        self._materialize_jobs(input_paths, models, settings)

    def _materialize_jobs(
        self,
        input_paths: list[Path],
        models: list[str],
        settings: JobSettings,
    ) -> None:
        new_jobs = create_jobs(
            input_paths=input_paths,
            models=models,
            settings=settings,
            existing_jobs=self.jobs,
            job_ids=self._job_ids,
        )
        log.info(
            "enqueue.requested",
            inputs=[str(path) for path in input_paths],
            models=models,
            settings=job_settings_log_payload(settings),
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
        failed_jobs = [job for job in self.jobs if job.status == "failed"]
        failed_job_ids = frozenset(job.id for job in failed_jobs)
        required = tuple(
            dict.fromkeys(
                artifact
                for job in failed_jobs
                for artifact in required_artifact_names(
                    (job.model,),
                    face_enhance=job.settings.face_enhance,
                    denoise_strength=job.settings.denoise_strength,
                )
            )
        )
        self.model_manager.refresh_readiness()
        missing = self.model_manager.missing(required)
        if missing:
            pending = PendingRetry(
                job_ids=failed_job_ids,
                required_artifacts=required,
            )
            self._pending_model_work = pending
            self._open_models_dialog(
                required_artifacts=pending.required_artifacts,
                pending_job_count=pending.job_count,
            )
            return
        self._retry_failed_snapshot(failed_job_ids)

    def _retry_failed_snapshot(self, failed_job_ids: frozenset[int]) -> None:
        retried_jobs = retry_failed_jobs(self.jobs, only_job_ids=failed_job_ids)
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
        if (
            (self._queue_action_issue == "images" and has_selected_image)
            or (self._queue_action_issue == "models" and has_selected_models)
        ):
            self._queue_action_issue = None
            self.queue_action_result.clear_result()
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
        self._refresh_queue_failure_result()

    def _refresh_queue_failure_result(self) -> None:
        failed_count = sum(job.status == "failed" for job in self.jobs)
        if failed_count == 0:
            self._queue_failure_count = 0
            self.queue_failure_result.hide()
            self.queue_failure_label.clear()
            self.queue_failure_result.setAccessibleName("")
            return

        noun = "job has" if failed_count == 1 else "jobs have"
        message = f"{failed_count} queue {noun} failed. Review the failed rows or retry them."
        self.queue_failure_label.setText(message)
        self.queue_failure_result.setAccessibleName(message)
        self.queue_failure_result.show()
        if failed_count != self._queue_failure_count:
            self._queue_failure_count = failed_count
            announce_accessible_alert(self.queue_failure_result)


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
        return "Unavailable"
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
        return (
            subprocess.run(
                ["open", "-R", str(target)],
                check=False,
                timeout=_REVEAL_TIMEOUT_SECONDS,
            ).returncode
            == 0
        )
    if sys.platform == "win32":
        subprocess.run(
            ["explorer", f"/select,{target}"],
            check=False,
            timeout=_REVEAL_TIMEOUT_SECONDS,
        )
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
    # On macOS, Qt maps the application icon to NSApp.applicationIconImage and
    # overrides the Liquid Glass/classic icon selected from the app bundle.
    if sys.platform == "win32":
        icon_path = Path(__file__).parent / "resources" / "icon-win.png"
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

    window = MainWindow(log_file=resolved_log_file, runtime_dirs=resolved_runtime_dirs)
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
    import codecs
    import importlib

    for module in (
        "torch",
        "torchvision",
        "cv2",
        "numpy",
        "PIL",
        "pillow_heif",
        "filelock",
        "ncnn",
        "realesrgan",
        "realesrgan.archs.srvgg_arch",
        "basicsr",
        "basicsr.archs.rrdbnet_arch",
        "gfpgan",
        "facexlib",
        "PySide6.QtWidgets",
    ):
        importlib.import_module(module)
    # Importing the binding alone does not load its Vulkan/MoltenVK edge. The
    # count may legitimately be zero, but calling it proves the frozen native
    # runtime can initialize before a user's first GPU job.
    importlib.import_module("ncnn").get_gpu_count()
    # urllib resolves this codec dynamically at the first HTTPS hostname. The
    # packaged smoke must exercise the registry lookup, not merely import urllib.
    codecs.lookup("idna")
    return 0


def main() -> int:
    if os.environ.get("PIXELUP_SELFTEST") == "1":
        return _selftest()
    try:
        app, _window = build_app(sys.argv)
    except Exception as exc:  # noqa: BLE001 - windowed startup needs an owned visible failure.
        log.exception("app.startup_failed")
        app = QApplication.instance() or QApplication(sys.argv)
        app.setApplicationName("PixelUp")
        app.setApplicationDisplayName("PixelUp")
        detail = (
            exc.user_message
            if isinstance(exc, PixelupError)
            else "PixelUp could not read its storage or application state."
        )
        hint = exc.user_hint if isinstance(exc, PixelupError) else None
        show_startup_failure(
            detail,
            hint,
        )
        return 1
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
