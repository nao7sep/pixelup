from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAccessible, QAccessibleEvent, QDesktopServices
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from pixelup.model_management import (
    MANAGED_MODEL_BUNDLES,
    ManagedModelBundle,
    artifact_size_bytes,
    bundle_size_bytes,
)
from pixelup.model_manager import ModelManager
from pixelup.session_log import log
from pixelup.ui_common import use_regular_spacing

_DIALOG_TARGET_WIDTH = 760


class ManagedModelsDialog(QDialog):
    """Presentation and commands for application-owned model state."""

    def __init__(
        self,
        manager: ModelManager,
        parent: QWidget | None = None,
        *,
        required_artifacts: tuple[str, ...] = (),
        pending_job_count: int = 0,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._required_artifacts = tuple(dict.fromkeys(required_artifacts))
        self._pending_job_count = pending_job_count
        self._announced_error = ""

        self.setWindowTitle("Managed models")
        self.setModal(True)

        layout = QVBoxLayout(self)
        use_regular_spacing(layout)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.models_panel = QWidget()
        models_layout = QGridLayout(self.models_panel)
        models_layout.setContentsMargins(0, 0, 0, 0)
        models_layout.setHorizontalSpacing(14)
        models_layout.setVerticalSpacing(7)
        for column, title in enumerate(("Model", "Use", "Download", "Status")):
            heading = QLabel(title)
            heading.setStyleSheet("font-weight: 600;")
            models_layout.addWidget(heading, 0, column)
        models_layout.setColumnStretch(0, 1)

        self.bundle_group = QButtonGroup(self)
        self.bundle_buttons: list[QRadioButton] = []
        self.status_labels: list[QLabel] = []
        for index, bundle in enumerate(MANAGED_MODEL_BUNDLES):
            row = index + 1
            button = QRadioButton(bundle.label)
            button.toggled.connect(self._render)
            self.bundle_group.addButton(button, index)
            self.bundle_buttons.append(button)
            models_layout.addWidget(button, row, 0)
            models_layout.addWidget(QLabel(bundle.purpose), row, 1)
            models_layout.addWidget(QLabel(_format_bytes(bundle_size_bytes(bundle))), row, 2)
            status_label = QLabel()
            self.status_labels.append(status_label)
            models_layout.addWidget(status_label, row, 3)
        layout.addWidget(self.models_panel)

        self.progress_frame = QFrame()
        progress_layout = QVBoxLayout(self.progress_frame)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(4)
        self.progress_label = QLabel()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
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
        layout.addWidget(self.result_frame)

        footer = QHBoxLayout()
        footer.addStretch()
        self.reveal_button = QPushButton("Reveal models folder")
        self.reveal_button.clicked.connect(self._reveal_models_folder)
        self.dismiss_button = QPushButton("Cancel" if self._required_artifacts else "Close")
        self.dismiss_button.clicked.connect(self.reject)
        self.install_button = QPushButton()
        self.install_button.clicked.connect(self._install_or_cancel)
        footer.addWidget(self.reveal_button)
        footer.addWidget(self.dismiss_button)
        footer.addWidget(self.install_button)
        layout.addLayout(footer)

        self._select_initial_row()
        self._manager.changed.connect(self._render)
        self._render()
        if self._required_artifacts:
            self.install_button.setFocus(Qt.FocusReason.OtherFocusReason)
        else:
            selected = self.bundle_group.checkedButton()
            if selected is not None:
                selected.setFocus(Qt.FocusReason.OtherFocusReason)
        self.adjustSize()
        minimum = self.sizeHint()
        self.setMinimumSize(minimum)
        self.resize(
            max(_DIALOG_TARGET_WIDTH, minimum.width()),
            minimum.height(),
        )

    def _summary_text(self) -> str:
        if not self._required_artifacts:
            return (
                "Install models before you need them. PixelUp verifies every download and "
                "keeps the installed files for later jobs. Closing this window does not stop "
                "an installation."
            )
        missing = self._manager.missing(self._required_artifacts)
        return (
            f"This batch needs {len(missing)} model file{'' if len(missing) == 1 else 's'} "
            f"({_format_bytes(artifact_size_bytes(missing))}) before "
            f"{self._pending_job_count} job{'' if self._pending_job_count == 1 else 's'} can be "
            "queued. No jobs will be created unless installation succeeds."
        )

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
        self.bundle_buttons[row].setChecked(True)

    def _selected_bundle(self) -> ManagedModelBundle | None:
        row = self.bundle_group.checkedId()
        return MANAGED_MODEL_BUNDLES[row] if 0 <= row < len(MANAGED_MODEL_BUNDLES) else None

    def _render(self) -> None:
        operation = self._manager.operation
        ready_names = self._manager.ready_names
        required = set(self._required_artifacts)
        self.summary_label.setText(self._summary_text())

        for row, bundle in enumerate(MANAGED_MODEL_BUNDLES):
            ready = len(ready_names.intersection(bundle.artifact_names))
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
            self.status_labels[row].setText(status)
            self.status_labels[row].setToolTip(status)

        running = operation.kind == "running"
        self.models_panel.setEnabled(not running)
        self.reveal_button.setEnabled(not running)
        self.progress_frame.setVisible(running)
        if running:
            if operation.cancelling:
                self.progress_label.setText("Cancelling model installation…")
                self.install_button.setText("Cancelling…")
                self.install_button.setEnabled(False)
            else:
                name = operation.current_artifact
                self.progress_label.setText(
                    f"Installing {name}…" if name else "Starting model installation…"
                )
                self.install_button.setText("Cancel installation")
                self.install_button.setEnabled(True)
            total = operation.total_bytes
            self.progress_bar.setValue(
                1000
                if total <= 0
                else min(1000, operation.completed_bytes * 1000 // total)
            )
        else:
            self._render_install_action()

        if operation.kind == "failed" and operation.error:
            self._show_error(operation.error)
        else:
            self.result_frame.hide()
            self.result_frame.setAccessibleName("")
            self._announced_error = ""

    def _render_install_action(self) -> None:
        if self._required_artifacts:
            count = self._pending_job_count
            self.install_button.setText(
                f"Install and queue {count} job{'' if count == 1 else 's'}"
            )
            self.install_button.setEnabled(bool(self._manager.missing(self._required_artifacts)))
            return
        bundle = self._selected_bundle()
        if bundle is None:
            self.install_button.setText("Install selected")
            self.install_button.setEnabled(False)
            return
        missing = self._manager.missing(bundle.artifact_names)
        self.install_button.setText("Install selected" if missing else "Reinstall selected")
        self.install_button.setEnabled(True)

    def _install_or_cancel(self) -> None:
        operation = self._manager.operation
        if operation.kind == "running":
            self._manager.cancel()
            return
        if self._required_artifacts:
            artifact_names = self._manager.missing(self._required_artifacts)
            force = False
        else:
            bundle = self._selected_bundle()
            if bundle is None:
                return
            missing = self._manager.missing(bundle.artifact_names)
            artifact_names = missing or bundle.artifact_names
            force = not missing
        if artifact_names:
            self._manager.install(artifact_names, force=force)

    def _reveal_models_folder(self) -> None:
        models_dir = self._manager.models_dir
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
            opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(models_dir)))
        except OSError as exc:
            log.warning("models.reveal_failed", models_dir=str(models_dir), reason=str(exc))
            self._show_error("Could not open the models folder.")
            return
        if not opened:
            log.warning("models.reveal_failed", models_dir=str(models_dir))
            self._show_error("Could not open the models folder.")

    def _show_error(self, message: str) -> None:
        self.result_label.setText(message)
        self.result_frame.setAccessibleName(f"Error: {message}")
        self.result_frame.show()
        if message == self._announced_error:
            return
        self._announced_error = message
        QAccessible.updateAccessibility(
            QAccessibleEvent(self.result_frame, QAccessible.Event.Alert)
        )


def _format_bytes(value: int) -> str:
    if value >= 1024**3:
        return f"{value / 1024**3:.1f} GB"
    if value >= 1024**2:
        return f"{value / 1024**2:.1f} MB"
    if value >= 1024:
        return f"{value / 1024:.1f} KB"
    return f"{value} B"
