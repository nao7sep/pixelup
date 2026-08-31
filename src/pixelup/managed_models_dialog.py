from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QAccessible, QAccessibleEvent, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pixelup.errors import ErrorCode, PixelupError
from pixelup.model_management import (
    MANAGED_MODEL_BUNDLES,
    ManagedModelBundle,
    artifact_size_bytes,
    bundle_ready_count,
    bundle_size_bytes,
    missing_artifact_names,
)
from pixelup.models import download_model
from pixelup.session_log import log
from pixelup.ui_common import use_regular_spacing

_DOWNLOAD_TIMEOUT_SECONDS = 600
_LOCK_TIMEOUT_SECONDS = 600
_DIALOG_TARGET_WIDTH = 760
_DIALOG_TARGET_HEIGHT = 520


class ModelInstallSignals(QObject):
    progress = Signal(str, int, int)
    waiting = Signal(str)
    finished = Signal(bool, bool, str)


class ModelInstallWorker(QObject):
    def __init__(self, models_dir: Path, artifact_names: tuple[str, ...], *, force: bool) -> None:
        super().__init__()
        self.signals = ModelInstallSignals()
        self._models_dir = models_dir
        self._artifact_names = artifact_names
        self._force = force
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return self._cancel_requested

    @Slot()
    def run(self) -> None:
        try:
            total_bytes = artifact_size_bytes(self._artifact_names)
            completed_bytes = 0
            for name in self._artifact_names:
                if self._is_cancelled():
                    raise PixelupError(ErrorCode.JOB_CANCELLED, "Model installation cancelled.")

                def report_progress(
                    _model: str,
                    done: int,
                    _total: int | None,
                    *,
                    artifact_name: str = name,
                    completed: int = completed_bytes,
                ) -> None:
                    self.signals.progress.emit(
                        artifact_name,
                        min(total_bytes, completed + done),
                        total_bytes,
                    )

                download_model(
                    self._models_dir,
                    name,
                    download_timeout=_DOWNLOAD_TIMEOUT_SECONDS,
                    lock_timeout=_LOCK_TIMEOUT_SECONDS,
                    on_download=report_progress,
                    on_waiting=lambda model, _elapsed: self.signals.waiting.emit(model),
                    should_cancel=self._is_cancelled,
                    force=self._force,
                )
                completed_bytes += artifact_size_bytes((name,))
                self.signals.progress.emit(name, completed_bytes, total_bytes)
        except PixelupError as exc:
            cancelled = exc.code == ErrorCode.JOB_CANCELLED
            message = f"{exc.message} {exc.hint}" if exc.hint else exc.message
            self.signals.finished.emit(False, cancelled, message)
        except Exception as exc:  # noqa: BLE001 - the modal must settle every worker failure.
            log.exception("models.install_failed_unexpectedly")
            self.signals.finished.emit(False, False, f"Unexpected error: {exc}")
        else:
            self.signals.finished.emit(True, False, "")


class ManagedModelsDialog(QDialog):
    """The sole GUI lifecycle for PixelUp-owned model artifacts."""

    def __init__(
        self,
        models_dir: Path,
        parent: QWidget | None = None,
        *,
        required_artifacts: tuple[str, ...] = (),
        pending_job_count: int = 0,
    ) -> None:
        super().__init__(parent)
        self._models_dir = models_dir
        self._required_artifacts = tuple(dict.fromkeys(required_artifacts))
        self._pending_job_count = pending_job_count
        self._worker: ModelInstallWorker | None = None
        self._thread: QThread | None = None
        self._install_result: tuple[bool, bool, str] | None = None
        self._close_after_cancel = False

        self.setWindowTitle("Managed models")
        self.setModal(True)

        layout = QVBoxLayout(self)
        use_regular_spacing(layout)

        self.summary_label = QLabel(self._summary_text())
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.table = QTableWidget(len(MANAGED_MODEL_BUNDLES), 4)
        self.table.setHorizontalHeaderLabels(["Model", "Use", "Download", "Status"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._update_install_action)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)

        self.progress_frame = QFrame()
        progress_layout = QVBoxLayout(self.progress_frame)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(4)
        self.progress_label = QLabel()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        self.progress_frame.hide()
        layout.addWidget(self.progress_frame)

        self.result_frame = QFrame()
        self.result_frame.setObjectName("modelInstallResult")
        self.result_frame.setStyleSheet(
            "QFrame#modelInstallResult {"
            " border: 1px solid #c0392b;"
            " border-radius: 5px;"
            " background: palette(base);"
            "}"
            "QLabel#modelInstallSeverity { color: #c0392b; font-weight: 600; }"
        )
        result_layout = QHBoxLayout(self.result_frame)
        result_layout.setContentsMargins(10, 7, 10, 7)
        result_layout.setSpacing(8)
        severity = QLabel("Error")
        severity.setObjectName("modelInstallSeverity")
        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        result_layout.addWidget(severity, 0, Qt.AlignmentFlag.AlignTop)
        result_layout.addWidget(self.result_label, 1)
        self.result_frame.hide()
        layout.addWidget(self.result_frame)

        footer = QHBoxLayout()
        footer.addStretch()
        self.reveal_button = QPushButton("Reveal models folder")
        self.reveal_button.clicked.connect(self._reveal_models_folder)
        self.dismiss_button = QPushButton("Cancel" if self._required_artifacts else "Close")
        self.dismiss_button.clicked.connect(self.reject)
        self.install_button = QPushButton()
        self.install_button.clicked.connect(self._install)
        footer.addWidget(self.reveal_button)
        footer.addWidget(self.dismiss_button)
        footer.addWidget(self.install_button)
        layout.addLayout(footer)

        self._refresh_rows()
        self._select_initial_row()
        self._update_install_action()
        if self._required_artifacts:
            self.install_button.setFocus(Qt.FocusReason.OtherFocusReason)
        else:
            self.table.setFocus(Qt.FocusReason.OtherFocusReason)
        self.adjustSize()
        minimum = self.sizeHint()
        self.setMinimumSize(minimum)
        self.resize(
            max(_DIALOG_TARGET_WIDTH, minimum.width()),
            max(_DIALOG_TARGET_HEIGHT, minimum.height()),
        )

    def _summary_text(self) -> str:
        if not self._required_artifacts:
            return (
                "Install models before you need them. PixelUp verifies every download and "
                "keeps the installed files for later jobs."
            )
        missing = missing_artifact_names(self._models_dir, self._required_artifacts)
        return (
            f"This batch needs {len(missing)} model file{'' if len(missing) == 1 else 's'} "
            f"({_format_bytes(artifact_size_bytes(missing))}) before "
            f"{self._pending_job_count} job{'' if self._pending_job_count == 1 else 's'} can be "
            "queued. No jobs will be created until installation succeeds."
        )

    def _refresh_rows(self) -> None:
        required = set(self._required_artifacts)
        for row, bundle in enumerate(MANAGED_MODEL_BUNDLES):
            ready = bundle_ready_count(self._models_dir, bundle)
            total = len(bundle.artifact_names)
            status = (
                "Ready"
                if ready == total
                else "Not installed"
                if ready == 0
                else f"{ready} of {total} ready"
            )
            if required.intersection(bundle.artifact_names):
                status = f"Required — {status}"
            for column, text in enumerate(
                (bundle.label, bundle.purpose, _format_bytes(bundle_size_bytes(bundle)), status)
            ):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.table.setItem(row, column, item)

    def _select_initial_row(self) -> None:
        required = set(self._required_artifacts)
        row = next(
            (
                index
                for index, bundle in enumerate(MANAGED_MODEL_BUNDLES)
                if required.intersection(bundle.artifact_names)
            ),
            0,
        )
        self.table.selectRow(row)

    def _selected_bundle(self) -> ManagedModelBundle | None:
        row = self.table.currentRow()
        return MANAGED_MODEL_BUNDLES[row] if 0 <= row < len(MANAGED_MODEL_BUNDLES) else None

    def _update_install_action(self) -> None:
        if self._worker is not None:
            return
        if self._required_artifacts:
            count = self._pending_job_count
            self.install_button.setText(f"Install and queue {count} job{'' if count == 1 else 's'}")
            self.install_button.setEnabled(
                bool(missing_artifact_names(self._models_dir, self._required_artifacts))
            )
            return
        bundle = self._selected_bundle()
        if bundle is None:
            self.install_button.setText("Install selected")
            self.install_button.setEnabled(False)
            return
        missing = missing_artifact_names(self._models_dir, bundle.artifact_names)
        self.install_button.setText("Install selected" if missing else "Reinstall selected")
        self.install_button.setEnabled(True)

    def _install(self) -> None:
        if self._worker is not None:
            return
        if self._required_artifacts:
            artifact_names = missing_artifact_names(self._models_dir, self._required_artifacts)
            force = False
        else:
            bundle = self._selected_bundle()
            if bundle is None:
                return
            missing = missing_artifact_names(self._models_dir, bundle.artifact_names)
            artifact_names = missing or bundle.artifact_names
            force = not missing
        if not artifact_names:
            if self._required_artifacts:
                self.accept()
            return

        log.info(
            "models.install_requested",
            artifacts=list(artifact_names),
            force=force,
            pending_job_count=self._pending_job_count,
        )
        self.result_frame.hide()
        self.result_frame.setAccessibleName("")
        self.progress_label.setText("Starting model installation…")
        self.progress_bar.setValue(0)
        self.progress_frame.show()
        self.table.setEnabled(False)
        self.reveal_button.setEnabled(False)
        self.install_button.setEnabled(False)
        self.dismiss_button.setText("Cancel installation")

        thread = QThread(self)
        worker = ModelInstallWorker(self._models_dir, artifact_names, force=force)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.signals.progress.connect(self._show_progress)
        worker.signals.waiting.connect(self._show_waiting)
        worker.signals.finished.connect(self._worker_finished)
        worker.signals.finished.connect(worker.deleteLater)
        worker.signals.finished.connect(thread.quit)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    @Slot(str, int, int)
    def _show_progress(self, name: str, done: int, total: int) -> None:
        self.progress_label.setText(
            f"Installing {name} — {_format_bytes(done)} of {_format_bytes(total)}"
        )
        self.progress_bar.setValue(1000 if total <= 0 else min(1000, done * 1000 // total))

    @Slot(str)
    def _show_waiting(self, name: str) -> None:
        self.progress_label.setText(f"Waiting to install {name}…")

    @Slot(bool, bool, str)
    def _worker_finished(self, succeeded: bool, cancelled: bool, message: str) -> None:
        self._install_result = (succeeded, cancelled, message)
        log.info(
            "models.install_finished",
            succeeded=succeeded,
            cancelled=cancelled,
            reason=message,
        )

    @Slot()
    def _thread_finished(self) -> None:
        result = self._install_result
        self._install_result = None
        self._worker = None
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.deleteLater()
        if result is None:
            return
        succeeded, cancelled, message = result
        self.progress_frame.hide()
        self.table.setEnabled(True)
        self.reveal_button.setEnabled(True)
        self.dismiss_button.setText("Cancel" if self._required_artifacts else "Close")
        self._refresh_rows()
        self._update_install_action()

        if succeeded and self._required_artifacts:
            self.accept()
            return
        if succeeded:
            return
        if self._close_after_cancel and cancelled:
            super().reject()
            return
        if cancelled:
            return
        self._show_error(message)

    def reject(self) -> None:
        if self._worker is None:
            super().reject()
            return
        self._close_after_cancel = True
        self._worker.request_cancel()
        self.dismiss_button.setEnabled(False)
        self.dismiss_button.setText("Cancelling…")
        self.progress_label.setText("Cancelling model installation…")

    def _reveal_models_folder(self) -> None:
        try:
            self._models_dir.mkdir(parents=True, exist_ok=True)
            opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._models_dir)))
        except OSError as exc:
            log.warning(
                "models.reveal_failed",
                models_dir=str(self._models_dir),
                reason=str(exc),
            )
            self._show_error("Could not open the models folder.")
            return
        if not opened:
            log.warning("models.reveal_failed", models_dir=str(self._models_dir))
            self._show_error("Could not open the models folder.")

    def _show_error(self, message: str) -> None:
        self.result_label.setText(message)
        self.result_frame.setAccessibleName(f"Error: {message}")
        self.result_frame.show()
        QAccessible.updateAccessibility(
            QAccessibleEvent(self.result_frame, QAccessible.Event.Alert)
        )


def _format_bytes(size: int) -> str:
    if size < 1_000_000:
        return f"{size / 1_000:.0f} KB"
    return f"{size / 1_000_000:.1f} MB"
