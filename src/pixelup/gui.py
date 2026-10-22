from __future__ import annotations

import itertools
import logging
import os
import sys
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields, replace
from itertools import count
from pathlib import Path
from typing import Literal

from PySide6.QtCore import (
    QEvent,
    QLocale,
    QObject,
    QSize,
    Qt,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAccessible,
    QAccessibleEvent,
    QCloseEvent,
    QDragEnterEvent,
    QDragLeaveEvent,
    QDragMoveEvent,
    QDropEvent,
    QFontMetrics,
    QGuiApplication,
    QIcon,
    QImage,
    QImageIOHandler,
    QImageReader,
    QKeySequence,
    QPixmap,
    QResizeEvent,
    QShortcut,
    QTransform,
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
    QScrollArea,
    QStyleFactory,
    QTableWidget,
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
    load_app_config_result,
    save_app_config,
)
from pixelup.backup_store import drain_backups
from pixelup.config import MODELS_ENV, RuntimeDirs, resolve_runtime_dirs, saved_models_dir
from pixelup.config_save import ConfigSave, ConfigSaveOutcome
from pixelup.devices import DEVICE_CHOICES, resolve_device
from pixelup.errors import ErrorCode, PixelupError, user_text
from pixelup.fonts import apply_ui_font
from pixelup.i18n import localizer
from pixelup.i18n.bootstrap import settle_language
from pixelup.i18n.localized import localize, unlocalize
from pixelup.i18n.message import Message, join
from pixelup.imaging import read_image_orientation, read_image_size, register_image_plugins
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
    warn_config_newer,
    warn_config_rejected,
    warn_config_reset,
    warn_jobs_stopping,
)
from pixelup.model_management import (
    UPSCALE_MODELS,
    required_artifact_names,
)
from pixelup.model_manager import ModelManager
from pixelup.output_cleanup import output_cleanup
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
from pixelup.quit_dialog import QuitConfirmDialog, QuitSaveFailedDialog
from pixelup.read_threads import ReadThreads
from pixelup.records_window import RecordsReads, RecordsWindow, records_reads, stored_signal
from pixelup.runner import JobRunner
from pixelup.session_end import os_session_ending
from pixelup.session_log import configure_session_logging, log, set_stored_listener
from pixelup.settings_dialog import SettingsDialog
from pixelup.shortcuts_dialog import ShortcutsDialog
from pixelup.startup_admission import admit_data_root
from pixelup.theme import PixelupStyle, apply_theme
from pixelup.time_zones import set_display_zone
from pixelup.ui_common import (
    NAMED_IN_MESSAGE,
    REGULAR_SPACING,
    apply_palette_fixes,
    bind_close_shortcut,
    identifying_name,
    named_values,
    use_regular_spacing,
)
from pixelup.widgets import (
    EmptyStateTableWidget,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
    OperationResult,
    choice_combo,
    device_combo,
    output_format_combo,
    retranslate_choices,
)
from pixelup.window_placement import (
    restore_window_geometry,
    save_window_geometry,
    window_state,
)

# Horizontal slack added to a measured string so cell text never touches the
# column edges (covers Qt's default cell margins plus a little breathing room).
_CELL_PADDING = 24
# Minimum readable width for a filename column: a fixed filename column elides to
# this (with a tooltip), and a stretch filename column uses it as its floor.
_NAME_MIN_WIDTH = 180

# How long the Parameters panel waits after the last edit before writing config.json.
# Coalescing absorbs the quality and denoise spin boxes' per-digit changes before
# each atomic rewrite and backup submission. A crash or force-quit can lose edits
# whose save has not landed yet. Ordinary quit flushes the latest edit within its
# existing bound; OS exit never waits beyond that bound to guarantee a save.
_PARAMETERS_SAVE_DELAY_MS = 500

# How long a quit may take once it is under way, under the five or so seconds an OS
# gives an app to end at logout or shutdown (unsaved-edits-conventions, Quitting).
QUIT_BUDGET_S = 4.0
# How long an ordinary quit waits for pending backup writes, inside the quit budget.
BACKUP_DRAIN_S = 1.0


def _start_quit_watchdog(seconds: float, expire: Callable[[], None]) -> None:
    """Call ``expire`` after ``seconds`` on a thread of its own, so the bound holds
    even while the window's thread is held in a wait."""
    timer = threading.Timer(seconds, expire)
    timer.daemon = True
    timer.start()


def _exit_now() -> None:
    """End the process at once. A worker thread still running cannot be joined, and
    an ordinary exit with one alive aborts."""
    os._exit(0)


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


@dataclass(frozen=True, slots=True)
class _JobPlanning:
    work: PendingModelWork
    images: tuple[tuple[Path, ImageEntry | None], ...]
    runtime_dirs: RuntimeDirs


_WINDOW_TARGET_EXTRA_WIDTH = 160
_WINDOW_TARGET_EXTRA_HEIGHT = 120


def bounded_initial_window_size(minimum: QSize, work_area: QSize) -> QSize:
    """Roomy launch target capped to the current desktop work area.

    The layout owner supplies the native minimum, already capped to the work
    area while the inner scroll viewport retains the complete content floor.
    """
    return QSize(
        max(minimum.width(), min(minimum.width() + _WINDOW_TARGET_EXTRA_WIDTH, work_area.width())),
        max(
            minimum.height(),
            min(minimum.height() + _WINDOW_TARGET_EXTRA_HEIGHT, work_area.height()),
        ),
    )


def is_reopen(previous: Qt.ApplicationState, current: Qt.ApplicationState) -> bool:
    """Whether an application-state change is a reopen: on macOS, a Dock click or a
    second launch, which Qt reports as becoming active while already active."""
    active = Qt.ApplicationState.ApplicationActive
    return previous == active and current == active


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

    def __init__(self, rows: int, columns: int, *, empty_message: Message) -> None:
        super().__init__(rows, columns, empty_message=empty_message)
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


class _Loaded(QObject):
    """Carries a result from a background thread to the window thread: a signal
    emitted on a plain thread is queued to this object's thread, the GUI's."""

    ready = Signal(int, object)


class _Planned(QObject):
    ready = Signal(object, object)


class ImagePreview(QLabel):
    """The selected image, decoded on a background thread.

    Only a change of image starts a decode, so job events that re-select the same
    image never read the file again; a decode that finishes after another image was
    selected is dropped.
    """

    def __init__(self) -> None:
        super().__init__()
        localize(self, text="preview.none")
        self._pixmap: QPixmap | None = None
        self._path: Path | None = None
        self._request = 0
        self._loaded = _Loaded(self)
        self._loaded.ready.connect(self._show_loaded)
        self.reads = ReadThreads(self)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # A 320x240 (4:3 QVGA) floor keeps the preview genuinely useful: a
        # scaled-down image is still large enough to judge upscale quality,
        # unlike the previous 160x120 which crushed to a thumbnail.
        self.setMinimumSize(320, 240)

    @property
    def loading(self) -> bool:
        return self._request != 0

    def set_image(self, path: Path | None) -> None:
        if path == self._path or self.reads.closing:
            return
        self._path = path
        self._pixmap = None
        if path is None:
            self._request = 0
            self.clear()
            localize(self, text="preview.none")
            return
        self._request = next(_preview_requests)
        request = self._request
        def read() -> None:
            reader = QImageReader(str(path))
            # Match the displayed orientation used by size probes and processing.
            reader.setAutoTransform(True)
            image = reader.read()
            if (
                not image.isNull()
                and reader.transformation() == QImageIOHandler.Transformation.TransformationNone
            ):
                # Qt omits some PNG EXIF/XMP transforms. The shared header probe
                # returns only pending orientation; HEIF's applied transform is 1.
                try:
                    orientation = read_image_orientation(path)
                except Exception:  # noqa: BLE001 - optional metadata cannot hide a decoded image.
                    log.warning("preview.orientation_failed", input=str(path), exc_info=True)
                else:
                    rotation = {3: 180, 5: 90, 6: 90, 7: 90, 8: 270}.get(orientation)
                    if rotation is not None:
                        image = image.transformed(QTransform().rotate(rotation))
                    if orientation in {2, 5}:
                        image = image.flipped(Qt.Orientation.Horizontal)
                    elif orientation in {4, 7}:
                        image = image.flipped(Qt.Orientation.Vertical)
            try:
                self._loaded.ready.emit(request, image)
            except RuntimeError:
                pass  # A disposed preview cannot receive the obsolete result.

        self.reads.start("pixelup-preview", read)

    @Slot(int, object)
    def _show_loaded(self, request: int, image: QImage) -> None:
        if request != self._request or self.reads.closing:
            return
        self._request = 0
        self._pixmap = None if image.isNull() else QPixmap.fromImage(image)
        if self._pixmap is None:
            self.clear()
            localize(self, text="preview.unavailable")
            return
        # The picture replaces the words, and a language change must not write
        # them back over it.
        unlocalize(self, "text")
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
    _GEOMETRY_KEY = "mainWindow/geometry"

    def __init__(self, *, log_file: Path, runtime_dirs: RuntimeDirs | None = None) -> None:
        super().__init__()
        # A corrupt config.json is quarantined, a newer PixelUp's is left alone, and
        # a set that fails validation keeps its stored value while it runs on its
        # built-in; either way the app starts rather than failing (store-recovery-
        # conventions), and the notices below tell the user. They are surfaced only
        # after the window is built, so the message box has a real parent.
        load_result = load_app_config_result()
        self.config = load_result.config
        set_display_zone(self.config.display_zone)
        # What config.json holds; every save writes from it and replaces it.
        self._settings_file = load_result.file
        self._config_save: ConfigSave | None = None
        self._config_save_callback: Callable[[Message | None], None] | None = None
        self._config_save_failure: Message | None = None
        self._config_save_pending: Callable[[], None] | None = None
        # Submitted edits belong to the app until saved, even if their editor
        # closes after a timeout or failure. Each editor replaces only its sets.
        self._unsaved_config_changes: dict[str, object] = {}
        self._parameter_edit_generation = 0
        self._saving_parameter_generation = 0
        self._config_quarantined_to = load_result.quarantined_to
        self._config_newer_format = load_result.file.newer_format
        self._config_rejected = load_result.rejected
        self._opens_pending = 0
        self._open_results = _Loaded(self)
        self._open_results.ready.connect(self._apply_opened)
        self._open_reads = ReadThreads(self)
        self._open_reads.idle.connect(self._close_when_workers_stop)
        # What Auto resolves to on this computer, once the background probe knows.
        self._auto_device: str | None = None
        self._device_probed = _Loaded(self)
        self._device_probed.ready.connect(self._auto_device_known)
        self._device_reads = ReadThreads(self)
        self._device_reads.idle.connect(self._close_when_workers_stop)
        # Apply the configured UI font (family-only; the explicit size lives in
        # fonts.py) before building the UI so every widget inherits it. The blank
        # built-in resolves to the system UI font. setFont propagates app-wide, so
        # this is the single place the UI font is established.
        apply_ui_font(QApplication.instance(), self.config.font_family)
        # The window itself was made before that, so it takes the font too, as the
        # Settings path does; otherwise what inherits from it keeps the platform font.
        self.setFont(QApplication.instance().font())
        self.log_file = log_file
        # The Records window and the one thread its reads run on, made when the
        # window is first opened; the thread lasts until PixelUp quits.
        self._records_window: RecordsWindow | None = None
        self._records_reads: RecordsReads | None = None
        base_dirs = runtime_dirs or resolve_runtime_dirs()
        self._models_folder_locked = runtime_dirs is not None or bool(os.environ.get(MODELS_ENV))
        self._default_models_dir = base_dirs.models_dir
        self.runtime_dirs = base_dirs if self._models_folder_locked else RuntimeDirs(
            saved_models_dir(self.config.models_folder, base_dirs.models_dir),
            base_dirs.temp_dir,
        )
        self._job_ids = count(1)
        self._images_by_path: dict[Path, ImageEntry] = {}
        self._image_order: list[Path] = []
        self._image_rows: dict[Path, int] = {}
        self._queue_rows: dict[int, int] = {}
        self._queue_failure_count = 0
        self.model_manager = ModelManager(self.runtime_dirs.models_dir, self)
        self._active_models_dialog: ManagedModelsDialog | None = None
        self._pending_model_work: list[PendingModelWork] = []
        # Planning is serial: each request sees output names accepted by all
        # earlier plans. Logical cancellation never releases a physical read.
        self._pending_job_plans: list[_JobPlanning] = []
        self._active_job_plan: _JobPlanning | None = None
        self._job_plan_abandoned = False
        self._removed_plan_inputs: set[Path] = set()
        self._plan_results = _Planned(self)
        self._plan_results.ready.connect(self._apply_job_plan)
        self._planning_reads = ReadThreads(self)
        self._planning_reads.idle.connect(self._close_when_workers_stop)
        self._show_pending_models_after_scan = False
        self.jobs: list[Job] = []
        # Jobs are only ever appended, never removed, so these two indexes only need
        # updating where jobs are adopted (_adopt_new_jobs); every other job-state
        # change mutates an existing Job in place. They turn _find_job and an image
        # row's status summary from an O(jobs) scan into an O(1) lookup, which matters
        # because both run on every per-tile progress tick and job completion.
        self._jobs_by_id: dict[int, Job] = {}
        self._jobs_by_input_path: dict[Path, list[Job]] = defaultdict(list)
        self.runner = JobRunner(self.jobs, self, runtime_dirs=self.runtime_dirs)
        self.runner.progress.connect(self._job_progress)
        self.runner.finished.connect(self._job_finished)
        self.runner.idle.connect(self._close_when_workers_stop)
        output_cleanup.finished.connect(self._close_when_workers_stop)
        self._closing = False
        self._quit_ready = False
        self._quit_dialog = None
        self._os_close_accepted = False
        self._quit_when_workers_idle = False
        self._quit_bounded = False
        self._session_shutdown = False
        # Set once Qt announces that the event loop is ending (see _app_about_to_quit).
        self._app_quitting = False
        # Coalesces the Parameters panel's edits into one save (see
        # _PARAMETERS_SAVE_DELAY_MS). Built before the UI, because building the panel
        # connects the widget-change signals that start it.
        self._parameters_save_timer = QTimer(self)
        self._parameters_save_timer.setSingleShot(True)
        self._parameters_save_timer.setInterval(_PARAMETERS_SAVE_DELAY_MS)
        self._parameters_save_timer.timeout.connect(self._save_parameters)

        self.setWindowTitle("PixelUp")
        self._build_ui()
        # Everything a binding does not reach — rows, choices, headers, measured
        # widths — is re-derived here when the language changes.
        localizer.changed.connect(self._retranslate)
        self.model_manager.changed.connect(self._model_manager_changed)
        self.model_manager.cancelled.connect(self._model_install_cancelled)
        self.model_manager.idle.connect(self._close_when_workers_stop)
        self._bind_shortcuts()
        # Window minimum = the layout's content-based size hint: the central widget
        # sums the panes' needs, and each table's minimum is its measured column
        # widths plus a filename floor (see _fit_columns). Open a little roomier than
        # that floor so the stretch (filename) columns have slack on first launch.
        # The old fixed resize(1260, …) sat below this minimum, so it was dead.
        # Create the native frame while hidden so work-area sizing includes its
        # actual margins even on the first normal landing after unmaximizing.
        self.winId()
        hint = self._refresh_layout_metrics()
        work_area = self._available_client_size() or QSize(
            hint.width() + _WINDOW_TARGET_EXTRA_WIDTH,
            hint.height() + _WINDOW_TARGET_EXTRA_HEIGHT,
        )
        self.resize(bounded_initial_window_size(self.minimumSize(), work_area))
        self._window_state = window_state()
        restore_window_geometry(self, self._window_state, self._GEOMETRY_KEY)
        # The panel opens on what the user last left it at, not on a defaults layer:
        # config.parameters is the persisted panel, and with no stored set it is
        # JobSettings() — the built-ins.
        self._apply_job_settings(self.config.parameters)
        self._update_selected_image()
        self._update_action_buttons()

        app = QApplication.instance()
        if app is not None:
            commit = getattr(app, "commitDataRequest", None)
            if commit is not None:
                commit.connect(self._on_commit_data_request)
            app.aboutToQuit.connect(self._app_about_to_quit)
            self._application_state = app.applicationState()
            app.applicationStateChanged.connect(self._application_state_changed)
        # Each record the database stores tells an open Records window to read again.
        set_stored_listener(stored_signal().emit)
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
            # already running on the built-ins; this only tells the user.
            QTimer.singleShot(0, self._notify_config_reset)
        if self._config_newer_format is not None:
            log.warning(
                "config.newer_format",
                path=str(config_path()),
                format_version=self._config_newer_format,
            )
            QTimer.singleShot(0, self._notify_config_newer)
        if self._config_rejected:
            QTimer.singleShot(0, self._notify_config_rejected)

    def _retranslate(self) -> None:
        """Rewrite, after a language change, what the widget bindings do not reach."""
        _set_headers(self.image_table, _IMAGE_COLUMNS)
        _set_headers(self.queue_table, _QUEUE_COLUMNS)
        for combo, choices in (
            (self.alpha_mode, ALPHA_MODE_CHOICES),
            (self.tile, TILE_CHOICES),
            (self.device, self._device_choices()),
            (self.target_profile, TARGET_PROFILE_CHOICES),
        ):
            retranslate_choices(combo, choices)
        for path, row in self._image_rows.items():
            size = self._images_by_path[path].input_size
            self.image_table.item(row, 1).setText(localizer.of(_image_size_text(size)))
        self._refresh_image_job_summaries_for(list(self._image_rows))
        for job in self.jobs:
            self._update_job(job)
        # The numbers Qt formats itself, in the denoise and quality boxes, follow
        # the language's locale; a window passes its locale down to its children.
        self.setLocale(QLocale())
        self._refresh_layout_metrics()

    def _notify_config_reset(self) -> None:
        if self._config_quarantined_to is not None:
            warn_config_reset(self, self._config_quarantined_to)

    def _notify_config_newer(self) -> None:
        if self._config_newer_format is not None:
            warn_config_newer(self, config_path())

    def _notify_config_rejected(self) -> None:
        warn_config_rejected(
            self, tuple(_SET_LABELS[key] for key in self._config_rejected), config_path()
        )

    def _on_commit_data_request(self, _manager: object) -> None:
        # Asked again for every quit on macOS, the user's own included, so each
        # request decides afresh (see session_end).
        self._session_shutdown = os_session_ending()
        if (self._session_shutdown and self._config_save is None
                and self._parameters_save_timer.isActive()):
            # Windows ends the process soon after the session ends, so the
            # debounced Parameters save starts now rather than at its timer.
            self._flush_parameters_save(surface_failure=False)
        if self._session_shutdown and self._closing:
            self._bound_quit()
            self._dismiss_session_question()
            self._quit_ready = True
            QTimer.singleShot(0, self.close)

    @Slot()
    def _app_about_to_quit(self) -> None:
        # Windows emits this at WM_ENDSESSION, before Qt closes the windows; the
        # session is still ending, but isSavingSession() was true only during the
        # commit request. Elsewhere it follows the last close, so nothing changes.
        # Either way no user interaction is possible any more (QCoreApplication).
        self._app_quitting = True

    def _question_parent(self) -> QWidget | None:
        """The shown window a question attaches to: the main window, else Records.
        With neither showing, the question stands alone rather than attaching to a
        hidden or minimized window and bringing it back (window-conventions)."""
        for window in (self, self._records_window):
            if window is not None and window.isVisible() and not window.isMinimized():
                return window
        return None

    def _dismiss_session_question(self) -> None:
        if self._quit_dialog is not None:
            self._quit_dialog.reject()
        # Every modal is PixelUp's own, whether attached to a window or standing alone.
        modal = QApplication.activeModalWidget()
        if isinstance(modal, QDialog):
            modal.reject()

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.WindowActivate:
            # Model files copied into the folder by hand are found when PixelUp
            # comes back to the front.
            self.model_manager.rescan()
        return super().event(event)

    @Slot(Qt.ApplicationState)
    def _application_state_changed(self, state: Qt.ApplicationState) -> None:
        # Reopening PixelUp brings the main window back, even while the Records
        # window is the one on screen: Cocoa restores a minimized window by itself
        # only when no other window is visible.
        reopened = is_reopen(self._application_state, state)
        self._application_state = state
        if reopened and self.isVisible():
            if self.isMinimized():
                self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized)
            self.raise_()
            self.activateWindow()

    def _accept_close(self, event: QCloseEvent) -> None:
        save_window_geometry(self, self._window_state, self._GEOMETRY_KEY)
        event.accept()

    def show_prepared(self) -> None:
        self.show()
        handle = self.windowHandle()
        if handle is not None:
            handle.screenChanged.connect(self._update_native_minimum)
        for screen in QApplication.screens():
            self._watch_layout_screen(screen)
        QApplication.instance().screenAdded.connect(self._watch_layout_screen)
        self._update_native_minimum()
        self._probe_auto_device()

    def _probe_auto_device(self) -> None:
        """Learn what Auto resolves to, off the window thread and after it shows:
        the answer needs torch, which window startup never imports."""

        def read() -> None:
            try:
                device = _resolve_auto_device()
            except Exception:  # noqa: BLE001 - Auto then names nothing, as before.
                log.exception("device.auto_probe_failed")
                return
            try:
                self._device_probed.ready.emit(0, device)
            except RuntimeError:
                pass  # An explicitly disposed window has abandoned the probe.

        self._device_reads.start("pixelup-device-probe", read)

    @Slot(int, object)
    def _auto_device_known(self, _request: int, device: str) -> None:
        self._auto_device = device
        retranslate_choices(self.device, self._device_choices())

    def _device_choices(self) -> tuple[tuple[Message | str, str], ...]:
        """The device choices, with Auto naming what it resolves to once known
        (localization-conventions)."""
        if self._auto_device is None:
            return DEVICE_CHOICES
        names = dict((value, label) for label, value in DEVICE_CHOICES)
        auto = Message.of("parameters.deviceAutoResolved", device=names[self._auto_device])
        return tuple((auto if value == "auto" else label, value) for label, value in DEVICE_CHOICES)

    def _is_session_shutdown(self) -> bool:
        if self._app_quitting:
            return True
        app = QGuiApplication.instance()
        if sys.platform == "darwin":
            # Cocoa also starts Qt session handling for ordinary Cmd+Q.
            return self._session_shutdown
        return app is not None and app.isSavingSession()

    def closeEvent(self, event: QCloseEvent) -> None:
        # Qt accepts a close whose handler raised, which would end PixelUp with its
        # workers still running; each path below accepts only on purpose.
        event.ignore()
        if self._is_session_shutdown():
            try:
                self._closing = True
                self._quit_ready = True
                self._os_close_accepted = True
                self._bound_quit()
                self._dismiss_session_question()
                log.info("quit.session_shutdown")
                if self._config_save is None:
                    self._flush_parameters_save(surface_failure=False)
                self._finish_or_defer_close(event, session_shutdown=True)
            except Exception:
                # The OS is ending the session: a failure here never refuses it.
                log.exception("quit.session_shutdown_failed")
                event.accept()
            return
        if self._quit_when_workers_idle or self._quit_ready:
            self._finish_or_defer_close(event, session_shutdown=False)
            return
        if self._closing:
            return
        self._closing = True
        try:
            self._flush_parameters_save(
                finished=self._user_quit_save_finished,
                waiting=lambda: self._user_quit_save_finished(False),
            )
        except Exception:
            self._abandon_quit()
            return
        if self._quit_ready:
            self._finish_or_defer_close(event, session_shutdown=False)

    def _abandon_quit(self) -> None:
        """A failure before the quit was confirmed leaves PixelUp open and able to
        quit again, rather than stuck believing a quit is under way."""
        log.exception("quit.failed")
        self._closing = False
        self._quit_dialog = None

    def _user_quit_save_finished(self, saved: bool) -> None:
        try:
            self._continue_user_quit(saved)
        except Exception:
            self._abandon_quit()

    def _continue_user_quit(self, saved: bool) -> None:
        if self._quit_ready or not self._closing:
            return
        if self._is_session_shutdown():
            self._quit_ready = True
            self.close()
            return
        if not saved:
            dialog = QuitSaveFailedDialog(self._question_parent())
            self._quit_dialog = dialog
            choice = dialog.choose()
            self._quit_dialog = None
            if self._is_session_shutdown():
                self._quit_ready = True
                self.close()
                return
            if choice == "cancel":
                self._closing = False
                log.info("quit.cancelled")
                return
            if choice == "retry":
                self._flush_parameters_save(
                    finished=self._user_quit_save_finished,
                    waiting=lambda: self._user_quit_save_finished(False),
                )
                return
            log.warning(
                "quit.parameters_unsaved",
                values=job_settings_log_payload(self.current_job_settings()),
            )
        installs = len(self.model_manager.in_progress_operations)
        if self._images_by_path or installs:
            active = sum(
                1 for job in self.jobs if job.status in {"pending", "running", "cancelling"}
            )
            dialog = QuitConfirmDialog(
                active,
                self._question_parent(),
                installs=installs,
                images_open=bool(self._images_by_path),
            )
            self._quit_dialog = dialog
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
            self._quit_dialog = None
            if not accepted and not self._is_session_shutdown():
                self._closing = False
                log.info("quit.cancelled")
                return
            log.info("quit.confirmed", active_jobs=active, model_installs=installs)
        self._quit_ready = True
        QTimer.singleShot(0, self.close)

    def _finish_or_defer_close(self, event: QCloseEvent, *, session_shutdown: bool) -> None:
        self._bound_quit()
        self.runner.begin_shutdown()
        self.model_manager.begin_shutdown()
        output_cleanup.begin_shutdown()
        if self._records_window is not None:
            self._records_window.close()
        if self._workers_clean_for_quit():
            self._accept_close(event)
            return
        event.ignore()
        if session_shutdown:
            # Accept the OS close while the process retains its owners until settlement
            # or the one hard deadline. Hiding the last window must not destroy workers.
            QApplication.instance().setQuitOnLastWindowClosed(False)
            self._os_close_accepted = True
            self._accept_close(event)
            return
        if not self._quit_when_workers_idle:
            self._quit_when_workers_idle = True
            warn_jobs_stopping(self._question_parent())

    def _bound_quit(self) -> None:
        """Start the quit's one bound, once; past it PixelUp exits regardless."""
        if self._quit_bounded:
            return
        self._quit_bounded = True
        self._open_reads.closing = True
        self._device_reads.closing = True
        self._planning_reads.closing = True
        self._pending_job_plans.clear()
        self._job_plan_abandoned = True
        self.preview.reads.closing = True
        _start_quit_watchdog(QUIT_BUDGET_S, self._force_exit)

    def _force_exit(self) -> None:
        # Diagnostics may wait on a handler or volume; they cannot own the hard exit.
        try:
            threading.Thread(
                target=self._log_forced_exit, name="pixelup-quit-log", daemon=True
            ).start()
        finally:
            _exit_now()

    def _log_forced_exit(self) -> None:
        try:
            log.warning(
                "quit.forced_exit",
                budget_seconds=QUIT_BUDGET_S,
                running_jobs=[
                    {"job_id": job.id, "output": str(job.output_path)}
                    for job in list(self.jobs)
                    if job.status in {"running", "cancelling"}
                ],
                model_installs=[
                    list(operation.artifact_names)
                    for operation in list(self.model_manager.in_progress_operations)
                ],
            )
        except Exception:
            # Forced exit is already under way; another diagnostic sink could stall too.
            pass

    def holds_work_after_os_close(self) -> bool:
        """Whether an accepted OS close still owns work the event loop must settle."""
        return self._os_close_accepted and not self._workers_clean_for_quit()

    def _close_when_workers_stop(self) -> None:
        if self._os_close_accepted and self._workers_clean_for_quit():
            QApplication.instance().quit()
        elif self._quit_when_workers_idle and self._workers_clean_for_quit():
            QTimer.singleShot(0, self.close)

    def _workers_clean_for_quit(self) -> bool:
        return (
            self._config_save is None
            and self._open_reads.settled
            and self._device_reads.settled
            and self._planning_reads.settled
            and self.preview.reads.settled
            and output_cleanup.settled
            and self.runner.cleanup_for_quit()
            and self.model_manager.cleanup_for_quit()
            and (self._records_reads is None or self._records_reads.stop())
        )

    def open_paths(self, paths: list[Path]) -> None:
        """Offer ``paths`` as images. The filesystem checks and the header read run on
        a background thread, so a slow or stalled volume never freezes the window;
        the results are applied here, in the order the offer listed them."""
        log.info("open.requested", paths=[str(path) for path in paths])
        if not paths or self._quit_bounded:
            return
        request = next(_open_requests)
        self._opens_pending += 1
        def read() -> None:
            probes = _probe_open_paths(paths)
            try:
                self._open_results.ready.emit(request, probes)
            except RuntimeError:
                pass  # An explicitly disposed window has abandoned this offer.

        self._open_reads.start("pixelup-open", read)

    @property
    def opening(self) -> bool:
        """Whether an offer of images is still being checked."""
        return self._opens_pending > 0

    @Slot(int, object)
    def _apply_opened(self, _request: int, probes: list[_OpenProbe]) -> None:
        self._opens_pending -= 1
        if self._closing:
            return
        selected: Path | None = None
        added: list[Path] = []
        duplicates: list[Path] = []
        rejected: list[tuple[Path, Message, bool]] = []
        for probe in probes:
            if probe.rejected is not None:
                rejected.append((probe.input_path, probe.rejected, probe.failed))
                continue
            input_path = probe.input_path
            if input_path not in self._images_by_path:
                entry = ImageEntry(input_path, probe.size)
                self._images_by_path[input_path] = entry
                self._image_order.append(input_path)
                self._add_image_row(entry)
                log.info("image.added", input=str(input_path), size=entry.input_size)
                added.append(input_path)
            else:
                log.info("image.focused_existing", input=str(input_path))
                duplicates.append(input_path)
            selected = input_path
        if selected is not None:
            self._select_image(selected)
        self._update_action_buttons()
        self._show_open_outcome(added, duplicates, rejected)

    def _show_open_outcome(
        self,
        added: list[Path],
        duplicates: list[Path],
        rejected: list[tuple[Path, Message, bool]],
    ) -> None:
        # The banner names a few files and counts the rest; Records keeps every
        # rejection and the image table every open image.
        metrics = self.open_result.message_label.fontMetrics()
        if rejected:
            parts: list[Message] = []
            if added:
                parts.append(Message.of("images.added", count=len(added)))
            if duplicates:
                parts.append(_already_open(duplicates, metrics))
            parts.extend(
                Message.of(
                    "images.rejected",
                    name=identifying_name(path.name or str(path), metrics),
                    reason=reason,
                )
                for path, reason, _error in rejected[:NAMED_IN_MESSAGE]
            )
            if len(rejected) > NAMED_IN_MESSAGE:
                parts.append(
                    Message.of("images.rejectedMore", count=len(rejected) - NAMED_IN_MESSAGE)
                )
            self._set_open_result(
                join("common.sentences", parts),
                severity="error" if any(error for _path, _reason, error in rejected) else "warning",
                issue_paths=[*duplicates, *(path for path, _reason, _error in rejected)],
            )
        elif duplicates:
            self._set_open_result(
                _already_open(duplicates, metrics),
                severity="information",
                issue_paths=duplicates,
            )
        elif self._open_result_issue_paths and self._open_result_issue_paths.issubset(set(added)):
            self._dismiss_open_result()

    def _set_open_result(
        self,
        message: Message,
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
        self._content_root = root
        viewport = QScrollArea()
        viewport.setFrameShape(QFrame.Shape.NoFrame)
        viewport.setWidgetResizable(True)
        viewport.setWidget(root)
        self.setCentralWidget(viewport)

    def _build_window_actions(self) -> QWidget:
        # PixelUp's own windows and information, set apart from the queue buttons
        # above by a panel of their own; a heading would only restate the buttons.
        group = QFrame()
        group.setProperty("surface", "panel")
        layout = QGridLayout(group)
        use_regular_spacing(layout)

        self.records_button = localize(QPushButton(), text="main.records")
        self.records_button.clicked.connect(self._open_records_window)
        self.settings_button = localize(QPushButton(), text="main.settings")
        self.settings_button.clicked.connect(self._settings_dialog)
        settings_shortcut = QKeySequence("Ctrl+,")
        self.settings_button.setShortcut(settings_shortcut)
        localize(
            self.settings_button,
            tooltip=Message.of(
                "main.settingsTooltip",
                shortcut=settings_shortcut.toString(QKeySequence.SequenceFormat.NativeText),
            ),
        )
        self.shortcuts_button = localize(QPushButton(), text="main.shortcuts")
        self.shortcuts_button.clicked.connect(self._shortcuts_dialog)
        self.about_button = localize(QPushButton(), text="main.about")
        self.about_button.clicked.connect(self._about_dialog)

        start = Qt.AlignmentFlag.AlignLeft
        layout.addWidget(self.settings_button, 0, 0, start)
        layout.addWidget(self.records_button, 0, 1, start)
        layout.addWidget(self.shortcuts_button, 1, 0, start)
        layout.addWidget(self.about_button, 1, 1, start)
        return group

    def _bind_shortcuts(self) -> None:
        self.shortcuts_shortcut = QShortcut(QKeySequence("Ctrl+/"), self)
        self.shortcuts_shortcut.activated.connect(self._shortcuts_dialog)
        self.shortcuts_question_shortcut = QShortcut(QKeySequence("Ctrl+?"), self)
        self.shortcuts_question_shortcut.activated.connect(self._shortcuts_dialog)
        # Closing the main window by keyboard runs its ordinary close, the quit.
        self.close_shortcut = bind_close_shortcut(self)
        # Everyday commands press their own buttons, so a command that is off stays
        # off from the keyboard too.
        self.open_shortcut = _pressing(
            self.open_images_button, self, QKeySequence(QKeySequence.StandardKey.Open)
        )
        self.queue_shortcut = _pressing(
            self.queue_selected_button,
            self,
            QKeySequence("Ctrl+Return"),
            QKeySequence("Ctrl+Enter"),
        )
        # Only while the Images list has focus, where Backspace has no other meaning.
        self.remove_shortcut = _pressing(
            self.remove_image_button,
            self.image_table,
            QKeySequence(Qt.Key.Key_Backspace),
            QKeySequence(Qt.Key.Key_Delete),
        )
        self.remove_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)

    def _build_image_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        use_regular_spacing(layout, margins=False)

        layout.addWidget(self._build_images_group(), 1)
        layout.addWidget(self._build_selected_image_group(), 1)
        return panel

    def _build_images_group(self) -> QWidget:
        group = localize(QGroupBox(), title="images.title")
        layout = QVBoxLayout(group)
        use_regular_spacing(layout)

        self.image_table = ImageDropTable(0, 3, empty_message=Message("images.empty"))
        self.image_table.paths_dropped.connect(self.open_paths)
        _set_headers(self.image_table, _IMAGE_COLUMNS)
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
        self.open_images_button = localize(QPushButton(), text="images.open")
        self.open_images_button.clicked.connect(self._open_dialog)
        self.remove_image_button = localize(QPushButton(), text="images.remove")
        # Taking images out of the collection is the destructive path's trigger, so
        # it is outlined rather than filled (theme.py carries the two roles).
        self.remove_image_button.setProperty("role", "danger")
        self.remove_image_button.clicked.connect(self._remove_selected_image)
        button_layout.addWidget(self.open_images_button)
        button_layout.addWidget(self.remove_image_button)
        # A button is as wide as its label, not as wide as the space it sits in.
        button_layout.addStretch()
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
        # A sibling of the sections, so it starts where their headings start, and
        # only as wide as its buttons: the queue buttons above are always wider. The
        # extra space keeps it apart from the queue buttons it is not part of.
        layout.addSpacing(REGULAR_SPACING)
        layout.addWidget(self._build_window_actions(), 0, Qt.AlignmentFlag.AlignLeft)
        return column

    def _build_selected_image_group(self) -> QWidget:
        group = localize(QGroupBox(), title="preview.title")
        layout = QVBoxLayout(group)
        use_regular_spacing(layout)
        self.preview = ImagePreview()
        self.preview.reads.idle.connect(self._close_when_workers_stop)
        layout.addWidget(self.preview, 1)
        return group

    def _build_models_group(self) -> QWidget:
        group = localize(QGroupBox(), title="models.title")
        layout = QVBoxLayout(group)
        use_regular_spacing(layout)
        self.model_checks: dict[str, QCheckBox] = {}
        for model in UPSCALE_MODELS:
            checkbox = QCheckBox(model)
            checkbox.toggled.connect(self._update_action_buttons)
            self.model_checks[model] = checkbox
            layout.addWidget(checkbox)
        self.manage_models_button = localize(QPushButton(), text="models.manage")
        self.manage_models_button.clicked.connect(self._managed_models_dialog)
        layout.addWidget(self.manage_models_button, 0, Qt.AlignmentFlag.AlignLeft)
        # The models' state is a line of its own under the button that opens them:
        # drawn as a filled warning button, it read as a second, different action.
        self.model_status = QLabel()
        self.model_status.setWordWrap(True)
        layout.addWidget(self.model_status)
        self._refresh_model_rollup()
        layout.addStretch()
        return group

    def _build_parameters_group(self) -> QWidget:
        group = localize(QGroupBox(), title="parameters.title")
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

        self.denoise_strength = NoWheelDoubleSpinBox()
        self.denoise_strength.setRange(MIN_DENOISE_STRENGTH, MAX_DENOISE_STRENGTH)
        self.denoise_strength.setSingleStep(DENOISE_STRENGTH_STEP)
        self.denoise_strength.setDecimals(2)

        self.alpha_mode = choice_combo(ALPHA_MODE_CHOICES)

        self.output_format = output_format_combo()

        self.quality = NoWheelSpinBox()
        self.quality.setRange(MIN_QUALITY, MAX_QUALITY)

        self.tile = choice_combo(TILE_CHOICES)

        self.device = device_combo()

        self.strip_metadata = localize(QCheckBox(), text="parameters.stripMetadata")

        self.target_profile = choice_combo(TARGET_PROFILE_CHOICES)

        self.reset_parameters_button = localize(QPushButton(), text="parameters.reset")
        self.reset_parameters_button.clicked.connect(self._reset_parameters_to_defaults)

        # Per-control captions live in the Parameters Help dialog, not the panel:
        # always-visible help text was the main driver of the window's minimum
        # width, which had outgrown small screens.
        help_button = localize(QPushButton(), text="parameters.help")
        help_button.clicked.connect(self._parameters_help_dialog)

        form.addRow(_form_label("parameters.scale"), scale_row)
        form.addRow(_form_label("parameters.denoise"), self.denoise_strength)
        form.addRow(_form_label("parameters.alphaMode"), self.alpha_mode)
        form.addRow(_form_label("parameters.outputFormat"), self.output_format)
        form.addRow(_form_label("parameters.quality"), self.quality)
        form.addRow(_form_label("parameters.tileSize"), self.tile)
        form.addRow(_form_label("parameters.device"), self.device)
        form.addRow("", self.strip_metadata)
        form.addRow(_form_label("parameters.targetProfile"), self.target_profile)
        # Reset and Help stack as two rows: side by side they would be the widest
        # field in the form and re-widen the group the Help dialog exists to slim.
        form.addRow("", self.reset_parameters_button)
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
        group = localize(QGroupBox(), title="actions.title")
        layout = QVBoxLayout(group)
        use_regular_spacing(layout)
        self.queue_selected_button = localize(QPushButton(), text="actions.queueSelected")
        # The one action this window is for: queue what is selected, with the
        # parameters beside it. It is set apart by its label's weight, not an accent
        # fill: the three queue buttons under it do the same job at another scope,
        # and it is off at launch until a model is checked, so a fill would mostly
        # have shown as a faded block. The accent is kept for a dialog's commit.
        self.queue_selected_button.setProperty("role", "main")
        self.queue_selected_button.clicked.connect(self._queue_selected_image)
        self.queue_selected_all_models_button = localize(
            QPushButton(), text="actions.queueSelectedAllModels"
        )
        self.queue_selected_all_models_button.clicked.connect(self._queue_selected_image_all_models)
        self.queue_all_selected_models_button = localize(
            QPushButton(), text="actions.queueAllSelectedModels"
        )
        self.queue_all_selected_models_button.clicked.connect(
            self._queue_all_images_selected_models
        )
        self.queue_all_all_models_button = localize(QPushButton(), text="actions.queueAllAllModels")
        self.queue_all_all_models_button.clicked.connect(self._queue_all_images_all_models)
        self.retry_button = localize(QPushButton(), text="actions.retryFailed")
        self.retry_button.clicked.connect(self._retry_failed)
        self.cancel_button = localize(QPushButton(), text="actions.cancelQueue")
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
        group = localize(QGroupBox(), title="queue.title")
        layout = QVBoxLayout(group)
        use_regular_spacing(layout)
        self.queue_table = EmptyStateTableWidget(0, 5, empty_message=Message("queue.empty"))
        _set_headers(self.queue_table, _QUEUE_COLUMNS)
        self.queue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.queue_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        queue_header = self.queue_table.horizontalHeader()
        queue_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        queue_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        queue_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        queue_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        queue_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        # Widths measured from worst-case content in the current UI font and
        # language: Model shows the longest model name in full; Scale fits its
        # header; Status holds the longest status word. "Image" elides a long
        # filename to a readable floor (with a tooltip) and "Output" stretches.
        self._fit_queue_table_columns()
        self.queue_table.setMinimumHeight(180)

        self.queue_failure_result = QFrame()
        self.queue_failure_result.setObjectName("queueFailureResult")
        # An error banner: the app sheet draws it as a panel with a red edge.
        self.queue_failure_result.setProperty("surface", "panel")
        self.queue_failure_result.setProperty("severity", "error")
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
        size_width = _fit_columns(
            self.image_table,
            localizer.t(_IMAGE_COLUMNS[1]),
            localizer.of(_image_size_text((99999, 99999))),
            localizer.t("images.sizeUnavailable"),
        )
        jobs_width = _fit_columns(
            self.image_table,
            localizer.t(_IMAGE_COLUMNS[2]),
            localizer.of(job_status_summary(["succeeded"] * 12 + ["failed"] * 5)),
        )
        header.resizeSection(1, size_width)
        header.resizeSection(2, jobs_width)
        self.image_table.setMinimumWidth(_NAME_MIN_WIDTH + size_width + jobs_width)

    def _fit_queue_table_columns(self) -> None:
        header = self.queue_table.horizontalHeader()
        model_width = _fit_columns(
            self.queue_table, localizer.t(_QUEUE_COLUMNS[1]), max(UPSCALE_MODELS, key=len)
        )
        scale_width = _fit_columns(self.queue_table, localizer.t(_QUEUE_COLUMNS[2]), "4x")
        status_width = _fit_columns(
            self.queue_table,
            localizer.t(_QUEUE_COLUMNS[4]),
            *(localizer.t(key) for key in _STATUS_KEYS.values()),
        )
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
        central = self._content_root
        central.updateGeometry()
        if central.layout() is not None:
            central.layout().invalidate()
            central.layout().activate()
        hint = central.sizeHint()
        central.setMinimumSize(hint)
        self._update_native_minimum()
        return hint

    def _watch_layout_screen(self, screen) -> None:
        screen.availableGeometryChanged.connect(self._update_native_minimum)
        screen.logicalDotsPerInchChanged.connect(self._update_native_minimum)

    def _update_native_minimum(self, *_args) -> None:
        minimum = self._content_root.minimumSize()
        available = self._available_client_size()
        if available is not None:
            minimum = minimum.boundedTo(available)
        self.setMinimumSize(minimum)

    def _available_client_size(self) -> QSize | None:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return None
        available = screen.availableGeometry().size()
        handle = self.windowHandle()
        if handle is not None:
            margins = handle.frameMargins()
            available -= QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return available.expandedTo(QSize(1, 1))

    def _open_dialog(self) -> None:
        try:
            files, _ = QFileDialog.getOpenFileNames(self, localizer.t("images.openDialogTitle"))
        except Exception:  # noqa: BLE001 - native picker failures belong to this action.
            log.exception("open.picker_failed")
            self.open_result.show_result(
                Message("images.pickerFailed"),
                severity="error",
            )
            return
        log.info("open.dialog_returned", count=len(files))
        self.open_paths([Path(file) for file in files])

    def _settings_dialog(self) -> None:
        # Land any pending panel edit first, so the config the dialog opens on — and
        # carries the unshown settings through from — is the current one. Otherwise a
        # debounce firing behind the modal would be undone on OK.
        self._flush_parameters_save(finished=self._open_settings_after_save)

    def _open_settings_after_save(self, saved: bool) -> None:
        if not saved or self._quit_bounded:
            return
        log.info("settings.dialog_opened")
        dialog = SettingsDialog(
            self.config,
            self,
            try_save=lambda candidate, finished, waiting: self._save_config_candidate(
                candidate, Message("settings.saveFailed"), finished, waiting,
                edited_sets=("font_family", "language", "max_concurrent_jobs"),
            ),
            session_shutdown=self._is_session_shutdown,
        )
        # A saved change takes effect when its save lands (_apply_saved_settings),
        # even one that lands after the dialog was cancelled while it was slow.
        if dialog.exec() != QDialog.DialogCode.Accepted:
            log.info("settings.dialog_cancelled")

    def _apply_saved_settings(self, previous: AppConfig, current: AppConfig) -> None:
        """Make the settings a landed save changed take effect, whoever submitted it."""
        if previous.models_folder != current.models_folder and not self._models_folder_locked:
            self.runtime_dirs = RuntimeDirs(
                saved_models_dir(current.models_folder, self._default_models_dir),
                self.runtime_dirs.temp_dir,
            )
            self.runner.set_runtime_dirs(self.runtime_dirs)
            self.model_manager.set_models_dir(self.runtime_dirs.models_dir)
        if previous.display_zone != current.display_zone:
            set_display_zone(current.display_zone)
            if self._records_window is not None:
                self._records_window.show_times_again()
        if (
            previous.font_family,
            previous.language,
            previous.display_zone,
            previous.max_concurrent_jobs,
        ) == (
            current.font_family,
            current.language,
            current.display_zone,
            current.max_concurrent_jobs,
        ):
            return
        if previous.font_family != current.font_family:
            # Re-apply the UI font so a changed family takes effect immediately,
            # no restart needed.
            app = QApplication.instance()
            apply_ui_font(app, current.font_family)
            self.setFont(app.font())
        if previous.language != current.language:
            # A changed language applies on Save, like its neighbours, and rewrites
            # what is already on screen (see _retranslate), which re-measures.
            localizer.use(current.language)
        self._refresh_layout_metrics()
        log.info(
            "settings.saved",
            path=str(self._settings_file.path),
            previous=config_log_payload(previous),
            current=config_log_payload(current),
        )
        self.runner.schedule(current.max_concurrent_jobs)

    def _managed_models_dialog(self) -> None:
        self.model_manager.refresh_readiness()
        self._open_pending_models_dialog()

    def _pending_model_summary(self) -> tuple[tuple[str, ...], int]:
        required = tuple(dict.fromkeys(
            name for pending in self._pending_model_work for name in pending.required_artifacts
        ))
        return required, sum(pending.job_count for pending in self._pending_model_work)

    def _open_pending_models_dialog(self) -> None:
        required, count = self._pending_model_summary()
        self._open_models_dialog(required_artifacts=required, pending_job_count=count)

    def _open_models_dialog(
        self,
        *,
        required_artifacts: tuple[str, ...] = (),
        pending_job_count: int = 0,
    ) -> None:
        if self._active_models_dialog is not None:
            self._active_models_dialog.set_pending_work(required_artifacts, pending_job_count)
            self._active_models_dialog.raise_()
            self._active_models_dialog.activateWindow()
            return

        log.info("models.dialog_opened")
        dialog = ManagedModelsDialog(
            self.model_manager,
            self,
            required_artifacts=required_artifacts,
            pending_job_count=pending_job_count,
            models_folder=self.config.models_folder,
            session_shutdown=self._is_session_shutdown,
            default_models_dir=self._default_models_dir,
            folder_locked=self._models_folder_locked,
            save_folder=self._save_models_folder,
        )
        self._active_models_dialog = dialog
        dialog.finished.connect(lambda _result: self._models_dialog_finished(dialog))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _save_models_folder(
        self, value: str, finished: Callable[[Message | None], None]
    ) -> None:
        manager = self.model_manager
        if "\x00" in value:
            finished(Message("error.modelsDirCreateFailed"))
            return
        if self._models_folder_locked or (
            value == self.config.models_folder
            and "models_folder" not in self._unsaved_config_changes
        ):
            finished(None)
            return
        if manager.in_progress_operations or manager.folder_change_pending:
            finished(Message("managedModels.folderBusy"))
            return
        manager.folder_change_pending = True
        manager.changed.emit()

        def saved(error: Message | None) -> None:
            manager.folder_change_pending = False
            finished(error)
            manager.changed.emit()

        def flushed(ok: bool) -> None:
            if not ok or self._closing:
                saved(Message("settings.saveFailed"))
                return
            self._save_config_candidate(
                replace(self.config, models_folder=value),
                Message("settings.saveFailed"), saved,
                edited_sets=("models_folder",),
            )

        self._flush_parameters_save(finished=flushed)

    def _models_dialog_finished(self, dialog: ManagedModelsDialog) -> None:
        if self._active_models_dialog is dialog:
            self._active_models_dialog = None
        # Closing ends only what the user never authorized. A queue action that
        # needs missing models waits here as an offer to "Install and queue";
        # dismissing that offer declines it, so it creates no jobs. A request whose
        # every missing model is already installing needs nothing more from the
        # user: it keeps waiting and queues once those installs succeed.
        retained = [
            pending for pending in self._pending_model_work
            if self._installs_cover(pending.required_artifacts)
        ]
        if len(retained) != len(self._pending_model_work):
            log.info("models.pending_work_declined")
        self._pending_model_work = retained
        self._refresh_model_rollup()
        log.info("models.dialog_closed")
        dialog.deleteLater()

    def _installs_cover(self, artifact_names: tuple[str, ...]) -> bool:
        missing = self.model_manager.missing(artifact_names)
        installing = {
            name
            for operation in self.model_manager.in_progress_for(missing)
            for name in operation.artifact_names
        }
        return set(missing) <= installing

    def _model_manager_changed(self) -> None:
        if self._quit_bounded:
            return
        self._refresh_model_rollup()
        if not self.model_manager.readiness_known or self.model_manager.folder_change_pending:
            return
        ready = [
            pending for pending in self._pending_model_work
            if not self.model_manager.in_progress_for(pending.required_artifacts)
            and not self.model_manager.missing(pending.required_artifacts)
        ]
        if not ready:
            self._present_pending_models_after_scan()
            return
        for pending in ready:
            # Each request transfers to its planning owner before retirement;
            # one failed plan cannot discard other accepted requests.
            if isinstance(pending, PendingEnqueue):
                paths = pending.input_paths
            else:
                paths = tuple(self._find_job(job_id).input_path for job_id in pending.job_ids)
            self._pending_job_plans.append(_JobPlanning(
                work=pending,
                images=tuple((path, self._images_by_path.get(path)) for path in paths),
                runtime_dirs=RuntimeDirs(self.runtime_dirs.models_dir, self.runtime_dirs.temp_dir),
            ))
            self._pending_model_work.remove(pending)
        self._refresh_pending_models_dialog(accepted=True)
        self._start_next_job_plan()
        self._present_pending_models_after_scan()

    def _retained_plan_paths(self, request: _JobPlanning) -> set[Path]:
        # Retry can explicitly include historical rows with no open image. A
        # captured live image must retain its object identity after removal/reopen.
        return {
            path for path, image in request.images
            if (
                (image is None and isinstance(request.work, PendingRetry))
                or (image is not None and self._images_by_path.get(path) is image)
            )
            and (request is not self._active_job_plan or path not in self._removed_plan_inputs)
        }

    def _start_next_job_plan(self) -> None:
        while self._active_job_plan is None and self._pending_job_plans and not self._quit_bounded:
            request = self._pending_job_plans.pop(0)
            retained = self._retained_plan_paths(request)
            work = request.work
            if isinstance(work, PendingEnqueue):
                input_paths = [path for path in work.input_paths if path in retained]
                retry_ids = frozenset()
                if not input_paths:
                    continue
            else:
                input_paths = []
                retry_ids = frozenset(
                    job_id for job_id in work.job_ids
                    if self._find_job(job_id).status == "failed"
                    and self._find_job(job_id).input_path in retained
                )
                if not retry_ids:
                    continue
            # The worker may mutate these copies for retry, never the GUI's jobs.
            existing = [replace(job, warnings=list(job.warnings)) for job in self.jobs]
            results, job_ids = self._plan_results, self._job_ids
            self._active_job_plan = request
            self._job_plan_abandoned = False
            self._removed_plan_inputs.clear()

            def read(
                request=request, work=work, input_paths=input_paths,
                retry_ids=retry_ids, existing=existing, retained=retained,
                results=results, job_ids=job_ids,
            ) -> None:
                try:
                    if isinstance(work, PendingEnqueue):
                        planned: list[Job] | Message = create_jobs(
                            input_paths=input_paths, models=list(work.models),
                            settings=work.settings,
                            existing_jobs=existing, job_ids=job_ids,
                        )
                    else:
                        retried = set(retry_failed_jobs(existing, only_job_ids=retry_ids))
                        planned = [job for job in existing if job.id in retried]
                except Exception as exc:
                    log.exception(
                        "enqueue.planning_failed", inputs=[str(path) for path in retained],
                    )
                    planned = (
                        user_text(exc, internal_fallback=Message("actions.queueFailed"))
                        if isinstance(exc, PixelupError) else Message("actions.queueFailed")
                    )
                try:
                    results.ready.emit(request, planned)
                except RuntimeError:
                    pass  # An explicitly disposed window has abandoned this plan.

            try:
                self._planning_reads.start("pixelup-plan", read)
            except Exception:
                log.exception("enqueue.planning_failed", inputs=[str(path) for path in retained])
                self._active_job_plan = None
                self.queue_action_result.show_result(
                    Message("actions.queueFailed"), severity="error",
                )
                continue
        self._update_action_buttons()

    @Slot(object, object)
    def _apply_job_plan(self, request: _JobPlanning, planned: list[Job] | Message) -> None:
        if self._active_job_plan is not request:
            return
        retained = self._retained_plan_paths(request)
        self._active_job_plan = None
        self._removed_plan_inputs.clear()
        if self._quit_bounded:
            return
        if not self._job_plan_abandoned and retained:
            if isinstance(planned, Message):
                self.queue_action_result.show_result(planned, severity="error")
            else:
                planned = [job for job in planned if job.input_path in retained]
                try:
                    if isinstance(request.work, PendingEnqueue):
                        self._adopt_new_jobs(planned, request.runtime_dirs)
                    else:
                        self._adopt_retried_jobs(planned, request.runtime_dirs)
                except Exception:
                    log.exception("enqueue.adopt_failed")
                    self.queue_action_result.show_result(
                        Message("actions.queueFailed"), severity="error",
                    )
        self._start_next_job_plan()

    def _present_pending_models_after_scan(self) -> None:
        if self._show_pending_models_after_scan:
            self._show_pending_models_after_scan = False
            if self._pending_model_work:
                self._open_pending_models_dialog()

    def _refresh_pending_models_dialog(self, *, accepted: bool) -> None:
        dialog = self._active_models_dialog
        if dialog is None:
            return
        if self._pending_model_work:
            dialog.set_pending_work(*self._pending_model_summary())
        elif accepted:
            dialog.accept()
        else:
            dialog.reject()

    @Slot(object)
    def _model_install_cancelled(self, artifact_names: object) -> None:
        if not isinstance(artifact_names, tuple):
            return
        retained = [
            pending for pending in self._pending_model_work
            if not set(pending.required_artifacts).intersection(artifact_names)
        ]
        if len(retained) == len(self._pending_model_work):
            return
        self._pending_model_work = retained
        self._refresh_pending_models_dialog(accepted=False)
        log.info("models.pending_work_cancelled")

    def _refresh_model_rollup(self) -> None:
        ready, total = self.model_manager.ready_count()
        missing = ready < total
        if self.model_manager.in_progress_operations:
            completed, operation_total = self.model_manager.aggregate_progress()
            progress = 0 if operation_total <= 0 else min(100, completed * 100 // operation_total)
            self._show_model_status(Message.of("models.installing", progress=progress), severity="")
            localize(
                self.manage_models_button,
                accessible_name=Message.of("models.installingAccessible", progress=progress),
                tooltip="models.installingTooltip",
            )
            return
        if missing:
            self._show_model_status(Message("models.missing"), severity="warning")
            localize(
                self.manage_models_button,
                accessible_name="models.missingAccessible",
                tooltip="models.missingTooltip",
            )
        else:
            self._show_model_status(None, severity="")
            localize(
                self.manage_models_button,
                accessible_name="models.readyAccessible",
                tooltip="models.readyTooltip",
            )

    def _show_model_status(self, message: Message | None, *, severity: str) -> None:
        if message is None:
            unlocalize(self.model_status, "text")
            self.model_status.clear()
        else:
            localize(self.model_status, text=message)
        self.model_status.setVisible(message is not None)
        self.model_status.setProperty("severity", severity)
        # A dynamic property only restyles once the style is told to look again.
        self.model_status.style().unpolish(self.model_status)
        self.model_status.style().polish(self.model_status)

    def _parameters_help_dialog(self) -> None:
        log.info("parameters_help.dialog_opened")
        ParametersHelpDialog(self).exec()

    def _shortcuts_dialog(self) -> None:
        log.info("shortcuts.dialog_opened")
        ShortcutsDialog(self).exec()

    def _about_dialog(self) -> None:
        log.info("about.dialog_opened")
        AboutDialog(self).exec()

    def _open_records_window(self) -> None:
        if self._quit_when_workers_idle:
            return
        if self._records_window is not None:
            self._records_window.bring_forward()
            return
        if self._records_reads is None:
            self._records_reads = records_reads(self.log_file, self)
            # A quit deferred behind a read still in flight retries once it ends.
            self._records_reads.stopped.connect(self._close_when_workers_stop)
        log.info("records.window_opened")
        window = RecordsWindow(self._records_reads)
        window.closed.connect(lambda: self._records_window_closed(window))
        self._records_window = window
        # Opened at the user's request, it becomes the key window in front.
        window.bring_forward()

    def _records_window_closed(self, window: RecordsWindow) -> None:
        if self._records_window is window:
            self._records_window = None
        # Closing it as the key window hands the keyboard back to the main window.
        if window.isActiveWindow() and not self._closing and self.isVisible():
            self.activateWindow()
        log.info("records.window_closed")

    def _add_image_row(self, entry: ImageEntry) -> None:
        row = self.image_table.rowCount()
        self.image_table.insertRow(row)
        self._image_rows[entry.input_path] = row
        self.image_table.setItem(
            row,
            0,
            _item(entry.input_path.name, tooltip=str(entry.input_path)),
        )
        self.image_table.setItem(row, 1, _item(localizer.of(_image_size_text(entry.input_size))))
        self.image_table.setItem(row, 2, _item(localizer.t("images.noJobs")))

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
            path != self._remove_result_path or not self._has_active_jobs(self._remove_result_path)
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
                Message("images.removeBlocked"),
                severity="warning",
            )
            return
        self._remove_result_path = None
        self.remove_result.clear_result()
        row = self._image_rows.pop(path)
        self.image_table.removeRow(row)
        self._images_by_path.pop(path, None)
        # Removal also ends accepted planning intent, including a historical
        # retry captured while this image was absent before an open/remove cycle.
        if self._active_job_plan is not None:
            self._removed_plan_inputs.add(path)
        self._pending_job_plans = [
            replace(request, images=tuple(item for item in request.images if item[0] != path))
            for request in self._pending_job_plans
            if any(item[0] != path for item in request.images)
        ]
        # Removal ends captured intent even if the same file is reopened later.
        retained: list[PendingModelWork] = []
        for pending in self._pending_model_work:
            if isinstance(pending, PendingEnqueue):
                pending = replace(
                    pending, input_paths=tuple(item for item in pending.input_paths if item != path)
                )
                if not pending.input_paths:
                    continue
            else:
                ids = frozenset(
                    job_id for job_id in pending.job_ids
                    if self._find_job(job_id).input_path != path
                )
                if not ids:
                    continue
                required = tuple(dict.fromkeys(
                    artifact for job_id in ids
                    for artifact in required_artifact_names(
                        (self._find_job(job_id).model,),
                        denoise_strength=self._find_job(job_id).settings.denoise_strength,
                    )
                ))
                pending = PendingRetry(job_ids=ids, required_artifacts=required)
            retained.append(pending)
        had_pending = bool(self._pending_model_work)
        self._pending_model_work = retained
        if had_pending:
            self._refresh_pending_models_dialog(accepted=False)
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
        # rather than raising: the loader already rejects a scale outside SCALE_VALUES,
        # so this only guards a programmatic caller.
        self.scale_buttons.get(settings.scale, self.scale_buttons[DEFAULT_SCALE]).setChecked(True)
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
        self._update_reset_button()

    def _parameters_edited(self) -> None:
        self._parameter_edit_generation += 1
        self._parameters_save_timer.start()
        self._update_reset_button()

    def _update_reset_button(self) -> None:
        """Reset is off while the panel already holds the built-in parameters."""
        self.reset_parameters_button.setEnabled(self.current_job_settings() != JobSettings())

    def _flush_parameters_save(
        self,
        *,
        surface_failure: bool = True,
        finished: Callable[[bool], None] | None = None,
        waiting: Callable[[], None] | None = None,
    ) -> None:
        self._parameters_save_timer.stop()
        self._save_parameters(surface_failure=surface_failure, finished=finished, waiting=waiting)

    def _save_parameters(
        self,
        *,
        surface_failure: bool = True,
        finished: Callable[[bool], None] | None = None,
        waiting: Callable[[], None] | None = None,
    ) -> None:
        parameters = self.current_job_settings()
        if (parameters == self.config.parameters and self._config_save is None
                and not self._unsaved_config_changes):
            self.parameters_result.clear_result()
            if finished is not None:
                finished(True)
            return

        def saved(failure: Message | None) -> None:
            if failure is not None:
                if surface_failure:
                    self.parameters_result.show_result(failure, severity="error")
            else:
                log.info(
                    "parameters.saved",
                    path=str(self._settings_file.path),
                    values=job_settings_log_payload(parameters),
                )
                self.parameters_result.clear_result()
            if finished is not None:
                if failure is None and self.current_job_settings() != self.config.parameters:
                    # Quit and Settings opening need the visible panel's latest basis,
                    # rather than authorizing from an older save's successful result.
                    self._flush_parameters_save(
                        surface_failure=surface_failure, finished=finished, waiting=waiting
                    )
                else:
                    finished(failure is None)

        def pending() -> None:
            if surface_failure:
                self.parameters_result.show_result(
                    Message("notice.configSaving"), severity="warning"
                )
            if waiting is not None:
                waiting()

        self._save_config_candidate(
            replace(self.config, parameters=parameters),
            Message("parameters.saveFailed"),
            saved,
            pending,
            edited_sets=("parameters",),
        )

    def _save_config_candidate(
        self,
        candidate: AppConfig,
        failed: Message,
        finished: Callable[[Message | None], None],
        waiting: Callable[[], None] | None = None,
        *,
        edited_sets: tuple[str, ...] | None = None,
    ) -> None:
        if self._config_save is not None:
            finished(Message("notice.configSaving"))
            return
        if edited_sets is None:
            edited_sets = tuple(
                item.name for item in fields(AppConfig)
                if getattr(candidate, item.name) != getattr(self.config, item.name)
            )
        changes = {
            **self._unsaved_config_changes,
            **{name: getattr(candidate, name) for name in edited_sets},
        }
        candidate = replace(self.config, **changes)
        if candidate.models_folder != self.config.models_folder:
            if self.model_manager.in_progress_operations:
                finished(Message("managedModels.folderBusy"))
                return
            self.model_manager.folder_change_pending = True
        self._unsaved_config_changes = changes
        file = self._settings_file
        operation = ConfigSave(
            candidate,
            self.config,
            lambda candidate, previous: save_app_config(candidate, previous, file),
            self,
        )
        self._config_save = operation
        self._config_save_callback = finished
        self._config_save_pending = waiting
        self._config_save_failure = failed
        self._saving_parameter_generation = self._parameter_edit_generation
        operation.completed.connect(self._config_saved)
        operation.timed_out.connect(self._config_save_timed_out)
        operation.start()

    @Slot()
    def _config_save_timed_out(self) -> None:
        pending = self._config_save_pending
        self._config_save_pending = None
        if pending is not None:
            pending()

    @Slot(object)
    def _config_saved(self, outcome: ConfigSaveOutcome) -> None:
        operation = self._config_save
        if operation is None:
            return
        callback = self._config_save_callback
        failed = self._config_save_failure
        self._config_save = None
        self._config_save_callback = None
        self._config_save_pending = None
        failure = None
        if outcome.error is not None:
            exc = outcome.error
            log.warning(
                "config.save_failed",
                path=str(self._settings_file.path),
                reason=str(exc),
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            failure = (
                user_text(exc, internal_fallback=failed)
                if isinstance(exc, PixelupError)
                else failed
            )
        else:
            # One save runs at a time and only a landed save changes self.config, so
            # the landed result is the current settings, whether or not the editor
            # that submitted it is still open.
            previous = self.config
            self._settings_file = outcome.result.file
            self.config = outcome.result.config
            self._unsaved_config_changes.clear()
            if not self._quit_bounded:
                try:
                    self._apply_saved_settings(previous, self.config)
                except Exception:
                    # The file landed; a failure to apply it here must not leave its
                    # editor waiting or a quit behind it stalled.
                    log.exception("settings.apply_failed")
        if operation.candidate.models_folder != operation.previous.models_folder:
            self.model_manager.folder_change_pending = False
        if self._parameter_edit_generation != self._saving_parameter_generation:
            if self._os_close_accepted:
                # Capture the latest visible edit before the idle check can end
                # the event loop. A debounce cannot own work during OS takeover.
                self._flush_parameters_save(surface_failure=False)
            else:
                self._parameters_save_timer.start()
        elif callback is None and failure is None:
            self.parameters_result.clear_result()
        if callback is not None:
            callback(failure)
        if operation.candidate.models_folder != operation.previous.models_folder:
            self.model_manager.changed.emit()
        operation.deleteLater()
        self._close_when_workers_stop()

    def _reset_parameters_to_defaults(self) -> None:
        """Restore the panel to PixelUp's built-in parameters and save it like any edit.

        ``JobSettings()`` is the built-ins, and the only source of them — not the
        user's persisted config, which is what the reset exists to get *away* from.
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
                Message("actions.needImages"),
                severity="warning",
            )
            return
        if not models:
            log.info("enqueue.rejected", reason="no_models")
            self._queue_action_issue = "models"
            self.queue_action_result.show_result(
                Message("actions.needModels"),
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
            denoise_strength=settings.denoise_strength,
        )
        self._preflight_models(PendingEnqueue(
            input_paths=tuple(input_paths), models=tuple(models),
            settings=settings, required_artifacts=required,
        ))

    def _preflight_models(self, pending: PendingModelWork) -> None:
        self._pending_model_work.append(pending)
        self._show_pending_models_after_scan = True
        self.model_manager.refresh_readiness()


    def _adopt_new_jobs(self, new_jobs: list[Job], runtime_dirs: RuntimeDirs) -> None:
        if not new_jobs:
            return
        log.info(
            "enqueue.requested",
            inputs=list(dict.fromkeys(str(job.input_path) for job in new_jobs)),
            models=list(dict.fromkeys(job.model for job in new_jobs)),
            settings=job_settings_log_payload(new_jobs[0].settings),
        )
        for job in new_jobs:
            log.debug("job.queued", job_id=job.id, details=job_log_payload(job))
        self.jobs.extend(new_jobs)
        for job in new_jobs:
            self._jobs_by_id[job.id] = job
            self._jobs_by_input_path[job.input_path].append(job)
        self._add_queue_rows(new_jobs)
        self._refresh_image_job_summaries_for(job.input_path for job in new_jobs)
        self._update_action_buttons()
        self.runner.retain_runtime_dirs((job.id for job in new_jobs), runtime_dirs)
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
            self.queue_table.setItem(row, 2, _item(_scale_label(job.settings.scale)))
            self.queue_table.setItem(
                row,
                3,
                _item(job.output_path.name, tooltip=str(job.output_path)),
            )
            self.queue_table.setItem(row, 4, _item(localizer.of(_status_text(job.status))))

    def _cancel_queue(self) -> None:
        self._pending_job_plans.clear()
        self._job_plan_abandoned = True
        cancelled_pending: list[int] = []
        signalled_running: list[int] = []
        touched_paths: list[Path] = []
        for job in self.jobs:
            if job.status == "pending":
                job.status = "cancelled"
                job.message = Message("queue.statusCancelled")
                self._update_job(job)
                cancelled_pending.append(job.id)
                touched_paths.append(job.input_path)
            elif job.status == "running":
                self.runner.request_cancel(job.id)
                job.status = "cancelling"
                self._update_job(job)
                signalled_running.append(job.id)
                touched_paths.append(job.input_path)
        log.info(
            "queue.cancelled",
            cancelled_pending=cancelled_pending,
            signalled_running=signalled_running,
        )
        self._refresh_image_job_summaries_for(touched_paths)
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
                    denoise_strength=job.settings.denoise_strength,
                )
            )
        )
        if failed_job_ids:
            self._preflight_models(PendingRetry(
                job_ids=failed_job_ids, required_artifacts=required,
            ))

    def _adopt_retried_jobs(self, planned: list[Job], runtime_dirs: RuntimeDirs) -> None:
        retried_jobs: list[int] = []
        touched_paths: list[Path] = []
        for result in planned:
            job = self._find_job(result.id)
            if job.status != "failed":
                continue
            job.output_path = result.output_path
            job.status = "pending"
            job.message = None
            job.warnings = []
            retried_jobs.append(job.id)
            self._update_job(job)
            touched_paths.append(job.input_path)
        if retried_jobs:
            log.info("jobs.retried", job_ids=retried_jobs)
            self._refresh_image_job_summaries_for(touched_paths)
            self._update_action_buttons()
            self.runner.retain_runtime_dirs(retried_jobs, runtime_dirs)
            self.runner.schedule(self.config.max_concurrent_jobs)

    @Slot(int, object)
    def _job_progress(self, job_id: int, message: Message) -> None:
        job = self._find_job(job_id)
        if message != job.message:
            log.debug(
                "job.progress",
                job_id=job.id,
                input=str(job.input_path),
                model=job.model,
                output=str(job.output_path),
                text=localizer.english().of(message),
            )
        job.message = message
        self._update_job(job)

    @Slot(int, bool, object, object, object)
    def _job_finished(
        self,
        job_id: int,
        ok: bool,
        message: Message,
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
        self._refresh_image_job_summaries_for((job.input_path,))
        self._update_action_buttons()

    def _find_job(self, job_id: int) -> Job:
        try:
            return self._jobs_by_id[job_id]
        except KeyError as exc:
            raise RuntimeError(f"Unknown job id: {job_id}") from exc

    def _update_job(self, job: Job) -> None:
        row = self._queue_rows[job.id]
        self.queue_table.item(row, 3).setText(job.output_path.name)
        self.queue_table.item(row, 3).setToolTip(str(job.output_path))
        status = job.message or _status_text(job.status)
        if job.status == "cancelling":
            status = _status_text("cancelling")
        status_text = localizer.of(status)
        self.queue_table.item(row, 4).setText(status_text)
        # The cell elides a long message; its tooltip carries it whole, with warnings.
        tooltip = "\n".join((status_text, *(localizer.of(warning) for warning in job.warnings)))
        self.queue_table.item(row, 4).setToolTip(tooltip)

    def _refresh_image_job_summaries_for(self, paths: Iterable[Path]) -> None:
        """Recompute only the given images' summaries.

        A job status change never affects an image other than its own, so redoing
        every row on every job event costs more the larger the batch gets. Duplicate
        paths collapse to one recompute each (``self._jobs_by_input_path`` already
        holds every job for a path, so repeating it would be redundant, not wrong).
        """
        for path in dict.fromkeys(paths):
            row = self._image_rows.get(path)
            if row is None:
                continue
            statuses = [job.status for job in self._jobs_by_input_path.get(path, ())]
            summary = localizer.of(job_status_summary(statuses))
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
        if (self._queue_action_issue == "images" and has_selected_image) or (
            self._queue_action_issue == "models" and has_selected_models
        ):
            self._queue_action_issue = None
            self.queue_action_result.clear_result()
        has_failed = any(job.status == "failed" for job in self.jobs)
        has_cancellable = (
            bool(self._pending_job_plans)
            or (self._active_job_plan is not None and not self._job_plan_abandoned)
            or any(job.status in {"pending", "running"} for job in self.jobs)
        )
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
            unlocalize(self.queue_failure_label, "text")
            unlocalize(self.queue_failure_result, "accessible_name")
            self.queue_failure_label.clear()
            self.queue_failure_result.setAccessibleName("")
            return

        message = Message.of("queue.failed", count=failed_count)
        localize(self.queue_failure_label, text=message)
        localize(self.queue_failure_result, accessible_name=message)
        self.queue_failure_result.show()
        if failed_count != self._queue_failure_count:
            self._queue_failure_count = failed_count
            announce_accessible_alert(self.queue_failure_result)


def _item(text: str, *, tooltip: str | None = None) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    if tooltip:
        item.setToolTip(tooltip)
    return item


_STATUS_KEYS = {
    "pending": "queue.statusPending",
    "running": "queue.statusRunning",
    "succeeded": "queue.statusDone",
    "failed": "queue.statusFailed",
    "cancelling": "queue.statusCancelling",
    "cancelled": "queue.statusCancelled",
}
# Each stored settings set by the name the interface gives it, for the notice that
# names the sets PixelUp could not use.
_SET_LABELS = {
    "font_family": Message("settings.uiFont"),
    "language": Message("settings.language"),
    "display_zone": Message("settings.timeZone"),
    "max_concurrent_jobs": Message("settings.concurrentJobs"),
    "parameters": Message("parameters.title"),
    "models_folder": Message("managedModels.folder"),
}
_IMAGE_COLUMNS = ("images.columnImage", "images.columnSize", "images.columnJobs")
_QUEUE_COLUMNS = (
    "queue.columnImage",
    "queue.columnModel",
    "queue.columnScale",
    "queue.columnOutput",
    "queue.columnStatus",
)


def _status_text(status: str) -> Message:
    # A job's status is a stored word, shown as catalogue text (localization-conventions).
    return Message(_STATUS_KEYS[status])


def _set_headers(table: QTableWidget, keys: tuple[str, ...]) -> None:
    table.setHorizontalHeaderLabels([localizer.t(key) for key in keys])


def _scale_label(scale: int) -> str:
    # The queue shows a job's scale as the panel offered it ("4x").
    return next(localizer.display(label) for label, value in SCALE_CHOICES if value == scale)


def _form_label(key: str) -> QLabel:
    return localize(QLabel(), text=key)


def _already_open(paths: list[Path], metrics: QFontMetrics) -> Message:
    return Message.of("images.alreadyOpen", names=named_values([p.name for p in paths], metrics))


@dataclass(frozen=True, slots=True)
class _OpenProbe:
    """What the filesystem said about one offered path."""

    input_path: Path
    size: tuple[int, int] | None = None
    rejected: Message | None = None
    failed: bool = False


def _pressing(button: QPushButton, parent: QWidget, *keys: QKeySequence) -> QShortcut:
    """A shortcut that clicks ``button``, which does nothing while it is disabled."""
    shortcut = QShortcut(parent)
    shortcut.setKeys(list(keys))
    shortcut.activated.connect(button.click)
    return shortcut


def _resolve_auto_device() -> str:
    """The backend Auto runs on here; imports torch, so only off the window thread."""
    return resolve_device("auto")


def _probe_open_paths(paths: list[Path]) -> list[_OpenProbe]:
    """Check each offered path on the calling thread (a background one): folders,
    missing files and files that are not readable images are rejected with the
    reason the window shows."""
    probes: list[_OpenProbe] = []
    for path in paths:
        try:
            input_path = absolute_user_path(path)
            resolved = input_path.resolve()
            if resolved.is_dir():
                log.warning("open.ignored_directory", path=str(path), resolved=str(resolved))
                probes.append(_OpenProbe(input_path, rejected=Message("images.reasonFolder")))
                continue
            if not resolved.is_file():
                log.warning("open.ignored_non_file", path=str(path), resolved=str(resolved))
                probes.append(
                    _OpenProbe(input_path, rejected=Message("images.reasonUnavailable"))
                )
                continue
            image_size = _safe_image_size(resolved)
            if image_size is None:
                log.warning("open.ignored_non_image", path=str(path), resolved=str(resolved))
                probes.append(_OpenProbe(input_path, rejected=Message("images.reasonNotImage")))
                continue
            probes.append(_OpenProbe(input_path, size=image_size))
        except Exception:  # noqa: BLE001 - one bad offer must not escape the probe.
            log.exception("open.failed", path=str(path))
            probes.append(
                _OpenProbe(path, rejected=Message("images.reasonUnreadable"), failed=True)
            )
    return probes


_open_requests = itertools.count(1)
_preview_requests = itertools.count(1)


def _safe_image_size(path: Path) -> tuple[int, int] | None:
    try:
        return read_image_size(path) if path.exists() else None
    except PixelupError:
        return None


def _image_size_text(size: tuple[int, int] | None) -> Message:
    if size is None:
        return Message("images.sizeUnavailable")
    width, height = size
    # Pixel dimensions are passed as text: a width is a measurement read digit by
    # digit, not a quantity to group ("4000 x 3000", never "4,000 x 3,000").
    return Message.of("images.size", width=str(width), height=str(height))


def build_app(
    argv: list[str],
    *,
    runtime_dirs: RuntimeDirs | None = None,
) -> tuple[QApplication, MainWindow]:
    """Build the fully-wired application and main window, ready to run.

    Everything ``main`` does except the blocking ``app.exec()`` lives here so it is exercisable
    headlessly: root admission, session logging, the runtime dirs, the QApplication and its
    style/icon, the window, and opening any image paths passed on the command line.
    ``runtime_dirs`` is injectable so a test can point it at a temp location; it defaults to the
    real resolution.
    """
    # Before the application object exists, so AppKit's own menu items and the
    # first frame are both already in the reader's language.
    settle_language()
    # The language lookup reads settings and prepares the root idempotently.
    # Admission precedes every mutable store and stays held until process exit,
    # beyond Qt/worker/log shutdown.
    admit_data_root()
    register_image_plugins()
    log_file = configure_session_logging()
    app = QApplication.instance() or QApplication(argv)
    localizer.sync_qt_translation()
    if "Fusion" in QStyleFactory.keys():
        app.setStyle(PixelupStyle())
    # PixelUp follows the OS light/dark theme; it owns no colours, only two
    # corrections to what Fusion resolves (apply_palette_fixes).
    #
    # There is deliberately no in-app System/Light/Dark choice. This is the
    # developer's approved exception to the fleet theme contract: do not propose one,
    # and if asked to add one, explain this first. Forcing an appearance with
    # QStyleHints.setColorScheme does switch Fusion's palette, but the roles
    # apply_palette_fixes sets keep the previous theme's values (inactive button text
    # stays white on a light window) until the fixes are re-applied after Qt
    # re-derives the palette, the same timing that defeated the handler described
    # below. For an upscaler that owns no colours, an override is not worth carrying
    # that switch path.
    #
    # Applied once, at launch, and NOT re-applied when the OS theme changes mid-run.
    # That is deliberate, not an oversight: flip the theme with PixelUp open and the
    # window follows, but these corrections revert until relaunch. Accepted — this is
    # an upscaler you open, run and close. A colorSchemeChanged handler was tried and
    # removed: Qt re-derives the palette after the signal and silently undid the
    # ButtonText fix, so it only looked like it worked.
    apply_palette_fixes(app)
    # The app's own appearance, installed once the palette is settled: one set of
    # control sizes and a state for every control, drawn from the palette so the
    # OS theme still decides the colours (see theme.py).
    apply_theme(app)
    app.setApplicationName("PixelUp")
    app.setApplicationDisplayName("PixelUp")
    # No custom macOS menu bar: PixelUp keeps Qt's default, which has no Edit menu, so the
    # Edit menu's Emoji & Symbols and Start Dictation items are absent. Its one free-text
    # fields in Settings and Managed models do not justify what a working Edit menu costs
    # under Qt: Qt disables the shared bar while a modal dialog is open, so every dialog
    # would need its own copy. A developer-approved exception to the fleet menu bar
    # contract; do not propose adding one.
    # On macOS, Qt maps the application icon to NSApp.applicationIconImage and
    # overrides the Liquid Glass/classic icon selected from the app bundle.
    if sys.platform == "win32":
        icon_path = Path(__file__).parent / "resources" / "icon-win.png"
        if icon_path.is_file():
            app.setWindowIcon(QIcon(str(icon_path)))
    window = MainWindow(log_file=log_file, runtime_dirs=runtime_dirs)
    log.info(
        "app.started",
        version=__version__,
        python=sys.version.split()[0],
        platform=sys.platform,
        log_file=str(log_file),
        runtime_dirs={
            "models_dir": str(window.runtime_dirs.models_dir),
            "temp_dir": str(window.runtime_dirs.temp_dir),
        },
        argv=argv[1:],
    )

    window.show_prepared()
    paths = [Path(arg) for arg in argv[1:] if not arg.startswith("-")]
    if paths:
        window.open_paths(paths)
    return app, window


def _selftest() -> int:
    """Import the full runtime stack and exit — proof that a build resolved every
    dependency, without downloading models or running inference.

    Running the packaged binary with PIXELUP_SELFTEST=1 forces every inference
    import up front, so packaging catches a missing component before release.
    Import-only: no window, no network, no weights.
    """
    import codecs
    import importlib

    for module in (
        "torch",
        "cv2",
        "numpy",
        "PIL",
        "pillow_heif",
        "pixelup.realesrgan_models",
        "pixelup.realesrgan_runtime",
        "PySide6.QtWidgets",
    ):
        importlib.import_module(module)
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
        # Contention is an expected startup result. Before admission there is no
        # shared log writer; unexpected failures use the console fallback.
        if not isinstance(exc, PixelupError) or exc.code != ErrorCode.APP_ALREADY_RUNNING:
            log.exception("app.startup_failed")
        app = QApplication.instance() or QApplication(sys.argv)
        app.setApplicationName("PixelUp")
        app.setApplicationDisplayName("PixelUp")
        localizer.sync_qt_translation()
        if isinstance(exc, PixelupError) and isinstance(exc.message, Message):
            detail, hint = exc.message, exc.hint
        else:
            detail, hint = Message("startup.storageUnreadable"), None
        show_startup_failure(detail, hint)
        return 1
    try:
        code = app.exec()
        # An OS quit (Cocoa's terminate, Windows' WM_ENDSESSION) delivers Qt's Quit
        # event, which ends the event loop as soon as the windows accept closing,
        # while the window may still own jobs, installs and reads. Run on until they
        # settle; the quit watchdog bounds this to QUIT_BUDGET_S.
        while _window.holds_work_after_os_close():
            app.exec()
        return code
    finally:
        # An ordinary quit gives pending backup writes a short bound; an OS session
        # end skips them (data-backup-conventions). The quit watchdog still owns the
        # hard exit.
        if not getattr(_window, "_os_close_accepted", False):
            drain_backups(BACKUP_DRAIN_S)
        logging.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
