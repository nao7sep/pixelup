from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAccessible, QAccessibleEvent, QDesktopServices, QFontMetrics
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pixelup.model_management import (
    MANAGED_ARTIFACT_NAMES,
    MANAGED_MODEL_BUNDLES,
    artifact_size_bytes,
    bundle_size_bytes,
)
from pixelup.model_manager import ModelManager, ModelOperation
from pixelup.session_log import log
from pixelup.ui_common import secondary_label, title_label, use_dialog_spacing

_MODEL_ROW_SPACING = 12
_MODEL_LIST_MAX_HEIGHT = 420
_COLUMN_TEXT_PADDING = 24


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
        # Dialog + show() produces an ordinary titled native window on macOS.
        # QDialog.open() chooses the sheet presentation instead, hiding the native
        # title bar and traffic-light controls even though this is a normal dialog.
        super().__init__(parent, Qt.WindowType.Dialog)
        self._manager = manager
        self._required_artifacts = tuple(dict.fromkeys(required_artifacts))
        self._pending_job_count = pending_job_count
        self._announced_error = ""

        self.setWindowTitle("Managed models")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        layout = QVBoxLayout(self)
        use_dialog_spacing(layout)
        layout.addWidget(title_label("Managed models"))

        self.summary_label = secondary_label("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.models_panel = QFrame()
        self.models_panel.setObjectName("managedModelsList")
        self.models_panel.setStyleSheet(
            "QFrame#managedModelsList {"
            " border: 1px solid palette(mid);"
            " border-radius: 7px;"
            " background: palette(base);"
            "}"
        )
        models_layout = QGridLayout(self.models_panel)
        models_layout.setContentsMargins(14, 12, 14, 12)
        models_layout.setHorizontalSpacing(16)
        models_layout.setVerticalSpacing(_MODEL_ROW_SPACING)
        for column, heading in enumerate(("Model", "Use", "Size", "Status", "Action")):
            column_heading = QLabel(heading)
            column_heading.setStyleSheet("font-weight: 600;")
            models_layout.addWidget(column_heading, 0, column)

        self.status_labels: list[QLabel] = []
        self.row_action_buttons: list[QPushButton] = []
        for index, bundle in enumerate(MANAGED_MODEL_BUNDLES):
            row = index + 1
            models_layout.addWidget(QLabel(bundle.label), row, 0)
            models_layout.addWidget(secondary_label(bundle.purpose), row, 1)
            models_layout.addWidget(
                secondary_label(_format_bytes(bundle_size_bytes(bundle))),
                row,
                2,
            )
            status_label = QLabel()
            self.status_labels.append(status_label)
            models_layout.addWidget(status_label, row, 3)
            action = QPushButton()
            action.clicked.connect(
                lambda _checked=False, bundle_index=index: self._install_bundle(bundle_index)
            )
            self.row_action_buttons.append(action)
            models_layout.addWidget(action, row, 4)

        self.column_minimum_widths = _model_column_widths(self.fontMetrics())
        for column, width in enumerate(self.column_minimum_widths):
            models_layout.setColumnMinimumWidth(column, width)

        models_scroll = QScrollArea()
        models_scroll.setWidgetResizable(True)
        models_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        models_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        models_scroll.setWidget(self.models_panel)
        models_scroll.setMinimumWidth(models_layout.sizeHint().width() + 4)
        models_scroll.setMinimumHeight(
            min(_MODEL_LIST_MAX_HEIGHT, models_layout.sizeHint().height() + 4)
        )
        layout.addWidget(models_scroll, 1)

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
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(10)
        footer.addStretch()
        self.dismiss_button = QPushButton("Close")
        self.dismiss_button.clicked.connect(self.reject)
        self.reveal_button = QPushButton("Reveal models folder")
        self.reveal_button.clicked.connect(self._reveal_models_folder)
        self.primary_button = QPushButton()
        self.primary_button.clicked.connect(self._install_all_or_cancel)
        footer.addWidget(self.dismiss_button)
        footer.addWidget(self.reveal_button)
        footer.addWidget(self.primary_button)
        layout.addLayout(footer)

        self._manager.changed.connect(self._render)
        self._render()
        self.primary_button.setFocus(Qt.FocusReason.OtherFocusReason)
        self.adjustSize()
        self.setMinimumSize(self.sizeHint())

    def _summary_text(self) -> str:
        if not self._required_artifacts:
            return (
                "Install any model you want to use, or install every missing model at once. "
                "PixelUp verifies each model and keeps it for later jobs. Closing this window "
                "does not stop an installation."
            )
        missing = self._manager.missing(self._required_artifacts)
        return (
            f"This batch needs {len(missing)} model file{'' if len(missing) == 1 else 's'} "
            f"({_format_bytes(artifact_size_bytes(missing))}) before "
            f"{self._pending_job_count} job{'' if self._pending_job_count == 1 else 's'} can be "
            "queued. No jobs will be created unless installation succeeds."
        )

    def _render(self) -> None:
        ready_names = self._manager.ready_names
        required = set(self._required_artifacts)
        self.summary_label.setText(self._summary_text())

        for row, bundle in enumerate(MANAGED_MODEL_BUNDLES):
            bundle_operations = self._manager.operations_for(bundle.artifact_names)
            active = next(
                (operation for operation in bundle_operations if operation.kind == "running"),
                None,
            )
            failed = next(
                (operation for operation in bundle_operations if operation.kind == "failed"),
                None,
            )
            ready = len(ready_names.intersection(bundle.artifact_names))
            total = len(bundle.artifact_names)
            status = _bundle_status(active, failed, ready, total)
            if required.intersection(bundle.artifact_names):
                status = f"Required — {status}"
            self.status_labels[row].setText(status)
            self.status_labels[row].setToolTip(status)

            action = self.row_action_buttons[row]
            if active is not None:
                action.setText("Cancelling…" if active.cancelling else "Cancel")
                action.setEnabled(not active.cancelling and not self._required_artifacts)
            else:
                action.setText("Reinstall" if ready == total else "Install")
                # A queue-preflight surface has one exact authorization action in
                # its footer. Its row actions remain visible only for orientation.
                action.setEnabled(not self._required_artifacts)
            action.setToolTip(
                "Use Install and queue below for this batch."
                if self._required_artifacts
                else ""
            )

        self._render_primary_action()
        errors = tuple(
            dict.fromkeys(
                operation.error
                for operation in self._manager.failed_operations
                if operation.error
            )
        )
        if errors:
            self._show_error(" ".join(errors))
        else:
            self.result_frame.hide()
            self.result_frame.setAccessibleName("")
            self._announced_error = ""

    def _render_primary_action(self) -> None:
        if self._required_artifacts:
            active = self._manager.active_for(self._required_artifacts)
            if active:
                self.primary_button.setText(
                    "Cancelling…"
                    if all(item.cancelling for item in active)
                    else "Cancel installation"
                )
                self.primary_button.setEnabled(not all(item.cancelling for item in active))
                return
            count = self._pending_job_count
            self.primary_button.setText(
                f"Install and queue {count} job{'' if count == 1 else 's'}"
            )
            missing = self._manager.missing(self._required_artifacts)
            self.primary_button.setEnabled(
                bool(self._manager.available_to_install(missing))
            )
            return

        self.primary_button.setText("Install all")
        missing = self._manager.missing(MANAGED_ARTIFACT_NAMES)
        self.primary_button.setEnabled(bool(self._manager.available_to_install(missing)))

    def _install_bundle(self, bundle_index: int) -> None:
        if self._required_artifacts or not 0 <= bundle_index < len(MANAGED_MODEL_BUNDLES):
            return
        bundle = MANAGED_MODEL_BUNDLES[bundle_index]
        active = self._manager.active_for(bundle.artifact_names)
        if active:
            self._manager.cancel(active[0].id)
            return
        missing = self._manager.missing(bundle.artifact_names)
        artifact_names = missing or bundle.artifact_names
        self._manager.install(artifact_names, force=not missing)

    def _install_all_or_cancel(self) -> None:
        targets = (
            self._required_artifacts
            if self._required_artifacts
            else MANAGED_ARTIFACT_NAMES
        )
        if self._required_artifacts and self._manager.active_for(targets):
            self._manager.cancel_for(targets)
            return
        missing = self._manager.missing(targets)
        self._install_artifact_groups(self._manager.available_to_install(missing))

    def _install_artifact_groups(self, artifact_names: tuple[str, ...]) -> None:
        requested = set(artifact_names)
        for bundle in MANAGED_MODEL_BUNDLES:
            group = tuple(name for name in bundle.artifact_names if name in requested)
            if group:
                self._manager.install(group, force=False)

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


def _bundle_status(
    active: ModelOperation | None,
    failed: ModelOperation | None,
    ready: int,
    total: int,
) -> str:
    if active is not None:
        if active.cancelling:
            return "Cancelling…"
        return f"Installing {_percentage(active.completed_bytes, active.total_bytes)}%"
    if failed is not None:
        return "Failed"
    if ready == total:
        return "Installed"
    if ready == 0:
        return "Not installed"
    return f"{ready} of {total} installed"


def _percentage(done: int, total: int) -> int:
    return 0 if total <= 0 else min(100, done * 100 // total)


def _model_column_widths(metrics: QFontMetrics) -> tuple[int, ...]:
    model_values = ("Model", *(bundle.label for bundle in MANAGED_MODEL_BUNDLES))
    purpose_values = ("Use", *(bundle.purpose for bundle in MANAGED_MODEL_BUNDLES))
    size_values = (
        "Size",
        *(_format_bytes(bundle_size_bytes(bundle)) for bundle in MANAGED_MODEL_BUNDLES),
    )
    status_values = (
        "Status",
        "Installed",
        "Not installed",
        "3 of 3 installed",
        "Required — 3 of 3 installed",
        "Required — Installing 100%",
        "Cancelling…",
        "Failed",
    )
    action_values = ("Action", "Install", "Reinstall", "Cancel", "Cancelling…")
    return tuple(
        _column_width(metrics, values)
        for values in (model_values, purpose_values, size_values, status_values, action_values)
    )


def _column_width(metrics: QFontMetrics, values: Iterable[str]) -> int:
    return max(metrics.horizontalAdvance(value) for value in values) + _COLUMN_TEXT_PADDING


def _format_bytes(value: int) -> str:
    if value >= 1024**3:
        return f"{value / 1024**3:.1f} GB"
    if value >= 1024**2:
        return f"{value / 1024**2:.1f} MB"
    if value >= 1024:
        return f"{value / 1024:.1f} KB"
    return f"{value} B"
