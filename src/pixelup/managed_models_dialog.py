from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from PySide6.QtCore import QEvent, QUrl
from PySide6.QtGui import QDesktopServices, QFontMetrics
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)
from shiboken6 import isValid

from pixelup.dialog_shell import TABLE_WIDTH, DialogShell
from pixelup.i18n import localizer
from pixelup.i18n.localized import localize
from pixelup.i18n.message import Message, join
from pixelup.model_management import (
    MANAGED_ARTIFACT_NAMES,
    MANAGED_MODEL_BUNDLES,
    artifact_size_bytes,
    bundle_size_bytes,
)
from pixelup.model_manager import ModelManager, ModelOperation
from pixelup.session_log import log
from pixelup.settings_dialog import DiscardChangesDialog
from pixelup.theme import COLLECTION_INSET
from pixelup.ui_common import secondary_label
from pixelup.widgets import OperationResult, repolish

_MODEL_ROW_SPACING = 12
_COLUMN_KEYS = (
    "managedModels.columnModel",
    "managedModels.columnUse",
    "managedModels.columnSize",
    "managedModels.columnStatus",
    "managedModels.columnAction",
)
_COLUMN_TEXT_PADDING = 24


class ManagedModelsDialog(DialogShell):
    """Presentation and commands for application-owned model state."""

    def __init__(
        self,
        manager: ModelManager,
        parent: QWidget | None = None,
        *,
        required_artifacts: tuple[str, ...] = (),
        pending_job_count: int = 0,
        models_folder: str = "",
        session_shutdown: Callable[[], bool] = lambda: False,
        default_models_dir: Path | None = None,
        folder_locked: bool = False,
        save_folder: Callable[[str, Callable[[Message | None], None]], None] | None = None,
    ) -> None:
        # The body holds this dialog's row buttons, so its scroll region takes no
        # focus of its own — the buttons are the keyboard owners.
        super().__init__("managedModels.title", parent, width=TABLE_WIDTH)
        self._manager = manager
        self._required_artifacts = tuple(dict.fromkeys(required_artifacts))
        self._pending_job_count = pending_job_count
        self._session_shutdown = session_shutdown
        self._folder_error: Message | None = None
        self._saved_folder = models_folder
        self._folder_locked = folder_locked
        self._save_folder = save_folder
        self._saving_folder = False
        self._dismissed = False
        self._after_scan: Callable[[], None] | None = None
        self.finished.connect(self._finished)

        folder_label = localize(QLabel(), text="managedModels.folder")
        self.folder_edit = QLineEdit(models_folder)
        self.folder_edit.setPlaceholderText(str(default_models_dir or manager.models_dir))
        folder_label.setBuddy(self.folder_edit)
        self.body_layout.addWidget(folder_label)
        folder_row = QHBoxLayout()
        folder_row.setSpacing(8)
        folder_row.addWidget(self.folder_edit, 1)
        self.browse_button = localize(QPushButton(), text="managedModels.chooseFolder")
        self.browse_button.clicked.connect(self._choose_folder)
        folder_row.addWidget(self.browse_button)
        self.apply_folder_button = localize(QPushButton(), text="managedModels.useFolder")
        self.apply_folder_button.clicked.connect(lambda: self._with_saved_folder(lambda: None))
        folder_row.addWidget(self.apply_folder_button)
        self.body_layout.addLayout(folder_row)
        self.folder_help = localize(secondary_label(""), text="managedModels.folderHelp")
        self.folder_help.setWordWrap(True)
        self.body_layout.addWidget(self.folder_help)
        self.folder_edit.textChanged.connect(self._folder_draft_changed)

        self.summary_label = secondary_label("")
        self.summary_label.setWordWrap(True)
        self.body_layout.addWidget(self.summary_label)

        self.models_panel = QFrame()
        self.models_panel.setObjectName("managedModelsList")
        # The app sheet draws a panel on the tables' own surface and edge.
        self.models_panel.setProperty("surface", "panel")
        # The columns sit between two empty gutter columns, so the column gap
        # also pads the panel's sides and the header line can span the gutters —
        # edge to edge at the tables' own inset, as their header line is.
        models_layout = QGridLayout(self.models_panel)
        models_layout.setContentsMargins(0, 12, 0, 12)
        models_layout.setHorizontalSpacing(16)
        models_layout.setVerticalSpacing(_MODEL_ROW_SPACING)
        first, last = 1, len(_COLUMN_KEYS)
        for column, heading in enumerate(_COLUMN_KEYS, start=first):
            column_heading = localize(QLabel(), text=heading)
            column_heading.setProperty("emphasis", "strong")
            models_layout.addWidget(column_heading, 0, column)
        models_layout.addWidget(_header_line(), 1, 0, 1, last + 2)

        self.status_labels: list[QLabel] = []
        self.row_action_buttons: list[QPushButton] = []
        for index, bundle in enumerate(MANAGED_MODEL_BUNDLES):
            row = index + 2
            models_layout.addWidget(QLabel(localizer.display(bundle.label)), row, first)
            models_layout.addWidget(
                localize(secondary_label(""), text=bundle.purpose), row, first + 1
            )
            models_layout.addWidget(
                localize(secondary_label(""), text=format_bytes(bundle_size_bytes(bundle))),
                row,
                first + 2,
            )
            status_label = QLabel()
            self.status_labels.append(status_label)
            models_layout.addWidget(status_label, row, first + 3)
            action = QPushButton()
            action.clicked.connect(
                lambda _checked=False, bundle_index=index: self._install_bundle(bundle_index)
            )
            self.row_action_buttons.append(action)
            models_layout.addWidget(action, row, first + 4)

        self.column_minimum_widths = _model_column_widths(self.fontMetrics())
        for column, width in enumerate(self.column_minimum_widths, start=first):
            models_layout.setColumnMinimumWidth(column, width)
            # Spare width goes to the columns, never to the empty gutters the
            # header line spans.
            models_layout.setColumnStretch(column, 1)

        # The panel goes straight into the body: the shell's body is the sole
        # scroll region, so the list no longer nests a scroll area of its own — and
        # the bar it used to draw over the Action column now has the body's own
        # padding to sit in.
        self.body_layout.addWidget(self.models_panel)
        self.body_layout.addStretch()

        self.result_view = OperationResult(object_name="modelInstallResult")
        self.body_layout.addWidget(self.result_view)

        self.dismiss_button = localize(QPushButton(), text="managedModels.close")
        self.dismiss_button.clicked.connect(self.reject)
        self.reveal_button = localize(QPushButton(), text="managedModels.revealFolder")
        self.reveal_button.clicked.connect(self._reveal_models_folder)
        self.primary_button = QPushButton()
        self.primary_button.clicked.connect(self._install_all_or_cancel)
        self.add_footer_widget(self.dismiss_button)
        self.add_footer_widget(self.reveal_button)
        self.add_footer_widget(self.primary_button)

        self._manager.folder_available.connect(self._open_models_folder)
        self._manager.changed.connect(self._render)
        self._render()
        self.set_initial_focus(self.primary_button)

    def set_pending_work(self, required_artifacts: tuple[str, ...], job_count: int) -> None:
        """Reflect outstanding queue requests without owning their lifetime."""
        self._required_artifacts = required_artifacts
        self._pending_job_count = job_count
        self._render()

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.WindowActivate:
            # The folder may have gained files placed there by hand while the
            # dialog was behind another window.
            self._manager.rescan()
        return super().event(event)

    def _summary_text(self) -> Message:
        if not self._required_artifacts:
            return Message("managedModels.summary")
        missing = self._manager.missing(self._required_artifacts)
        # Two counts in one sentence: each is its own counted message, so each
        # keeps its own plural form (localization-conventions).
        return Message.of(
            "managedModels.batchSummary",
            files=Message.of("managedModels.modelFiles", count=len(missing)),
            size=format_bytes(artifact_size_bytes(missing)),
            jobs=Message.of("managedModels.jobs", count=self._pending_job_count),
        )

    def _render(self) -> None:
        self._update_folder_controls()
        ready_names = self._manager.ready_names
        required = set(self._required_artifacts)
        localize(self.summary_label, text=self._summary_text())

        for row, bundle in enumerate(MANAGED_MODEL_BUNDLES):
            bundle_operations = self._manager.operations_for(bundle.artifact_names)
            in_progress = next(
                (
                    operation
                    for operation in bundle_operations
                    if operation.kind in {"running", "queued"}
                ),
                None,
            )
            failed = next(
                (operation for operation in bundle_operations if operation.kind == "failed"),
                None,
            )
            ready = len(ready_names.intersection(bundle.artifact_names))
            total = len(bundle.artifact_names)
            status = (Message("managedModels.checking")
                      if not self._manager.readiness_known and in_progress is None
                      else _bundle_status(in_progress, failed, ready, total))
            localize(self.status_labels[row], text=status, tooltip=status)
            missing_required = bool(required.intersection(bundle.artifact_names)) and ready < total
            # Attention styling from the app sheet, without changing the factual
            # status the label states.
            attention = missing_required and in_progress is None and failed is None
            self.status_labels[row].setProperty("severity", "warning" if attention else None)
            self.status_labels[row].setProperty("emphasis", "strong" if attention else None)
            repolish(self.status_labels[row])

            action = self.row_action_buttons[row]
            if in_progress is not None:
                localize(
                    action,
                    text="managedModels.cancelling"
                    if in_progress.cancelling
                    else "managedModels.cancel",
                )
                action.setEnabled(not in_progress.cancelling and not self._required_artifacts)
            else:
                localize(
                    action,
                    text="managedModels.reinstall" if ready == total else "managedModels.install",
                )
                # A queue-preflight surface has one exact authorization action in
                # its footer. Its row actions remain visible only for orientation.
                action.setEnabled(not self._required_artifacts)
            if self._required_artifacts:
                localize(action, tooltip="managedModels.rowTooltipBatch")

        self._render_primary_action()
        if not self._manager.readiness_known or self._saving_folder:
            if not self._manager.in_progress_operations:
                self.primary_button.setEnabled(False)
                for action in self.row_action_buttons:
                    action.setEnabled(False)
        errors = tuple(
            dict.fromkeys(
                operation.error
                for operation in self._manager.failed_operations
                if operation.error
            )
        )
        if self._folder_error is not None:
            self._show_error(self._folder_error)
        elif errors:
            self._show_error(join("common.sentences", errors))
        else:
            self.result_view.clear_result()

        # Every render can change the body's height — the summary swaps between two
        # lengths, the result banner comes and goes — and the bound is arithmetic
        # over that height, so it is re-taken here rather than once at build time.
        self.fit()
        if self._after_scan is not None and self._manager.readiness_known:
            action, self._after_scan = self._after_scan, None
            if not self._dismissed and isValid(self):
                action()

    def _render_primary_action(self) -> None:
        if self._required_artifacts:
            active = self._manager.in_progress_for(self._required_artifacts)
            if active:
                localize(
                    self.primary_button,
                    text="managedModels.cancelling"
                    if all(item.cancelling for item in active)
                    else "managedModels.cancelInstallation",
                )
                self.primary_button.setEnabled(not all(item.cancelling for item in active))
                return
            localize(
                self.primary_button,
                text=Message.of("managedModels.installAndQueue", count=self._pending_job_count),
            )
            missing = self._manager.missing(self._required_artifacts)
            self.primary_button.setEnabled(
                bool(self._manager.available_to_install(missing))
            )
            return

        localize(self.primary_button, text="managedModels.installAll")
        missing = self._manager.missing(MANAGED_ARTIFACT_NAMES)
        self.primary_button.setEnabled(bool(self._manager.available_to_install(missing)))

    def _install_bundle(self, bundle_index: int) -> None:
        if self._required_artifacts or not 0 <= bundle_index < len(MANAGED_MODEL_BUNDLES):
            return
        bundle = MANAGED_MODEL_BUNDLES[bundle_index]
        in_progress = self._manager.in_progress_for(bundle.artifact_names)
        if in_progress:
            self._manager.cancel(in_progress[0].id)
            return
        self._with_saved_folder(lambda: self._install_bundle_in_current_folder(bundle_index))

    def _install_all_or_cancel(self) -> None:
        targets = (
            self._required_artifacts
            if self._required_artifacts
            else MANAGED_ARTIFACT_NAMES
        )
        if self._required_artifacts and self._manager.in_progress_for(targets):
            self._manager.cancel_for(targets)
            return
        self._with_saved_folder(
            lambda: self._install_artifact_groups(
                self._manager.available_to_install(self._manager.missing(targets))
            )
        )

    def _install_bundle_in_current_folder(self, bundle_index: int) -> None:
        names = MANAGED_MODEL_BUNDLES[bundle_index].artifact_names
        missing = self._manager.missing(names)
        self._manager.install(missing or names, force=not missing)

    def _folder_draft_changed(self) -> None:
        # Editing again cancels an install that has not yet begun. It must never
        # acquire files into a folder the user has just replaced in the draft.
        self._after_scan = None
        self._update_folder_controls()

    def _update_folder_controls(self) -> None:
        busy = bool(self._manager.in_progress_operations) or self._saving_folder
        editable = not busy and not self._folder_locked
        self.folder_edit.setEnabled(editable)
        self.browse_button.setEnabled(editable)
        self.apply_folder_button.setEnabled(
            editable and self.folder_edit.text() != self._saved_folder
        )
        localize(self.folder_help, text=(
            "managedModels.folderOverride" if self._folder_locked else
            "managedModels.folderBusy" if busy else "managedModels.folderHelp"
        ))
        if self._folder_locked and self.folder_edit.text() != str(self._manager.models_dir):
            self.folder_edit.setText(str(self._manager.models_dir))

    def _choose_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, localizer.t("managedModels.chooseFolder"),
            self.folder_edit.text() or str(self._manager.models_dir),
        )
        if chosen:
            self.folder_edit.setText(chosen)

    def _with_saved_folder(self, action: Callable[[], None]) -> None:
        if self._saving_folder or self._manager.folder_change_pending:
            return
        value = self.folder_edit.text()
        if self._folder_locked or value == self._saved_folder or self._save_folder is None:
            if self._manager.readiness_known:
                action()
            return
        self._folder_error = None
        self._saving_folder = True
        self._update_folder_controls()

        def saved(error: Message | None) -> None:
            if self._dismissed or not isValid(self):
                return
            self._saving_folder = False
            self._folder_error = error
            if error is not None:
                self._show_error(error)
            else:
                self._saved_folder = value
                self.result_view.clear_result()
                if self._manager.readiness_known:
                    action()
                else:
                    self._after_scan = action
            self._update_folder_controls()

        self._save_folder(value, saved)

    def _finished(self, _result: int) -> None:
        self._dismissed = True
        self._after_scan = None

    def reject(self) -> None:
        if (
            not self._session_shutdown()
            and not self._saving_folder and not self._folder_locked
            and self.folder_edit.text() != self._saved_folder
            and DiscardChangesDialog(self).exec() != QDialog.DialogCode.Accepted
        ):
            return
        super().reject()

    def _install_artifact_groups(self, artifact_names: tuple[str, ...]) -> None:
        requested = set(artifact_names)
        for bundle in MANAGED_MODEL_BUNDLES:
            group = tuple(name for name in bundle.artifact_names if name in requested)
            if group:
                self._manager.install(group, force=False)

    def _reveal_models_folder(self) -> None:
        self.reveal_button.setEnabled(False)
        self._manager.prepare_folder_reveal()

    def _open_models_folder(self, models_dir: Path, available: bool) -> None:
        if self._dismissed:
            return
        self.reveal_button.setEnabled(True)
        if not available or not QDesktopServices.openUrl(QUrl.fromLocalFile(str(models_dir))):
            log.warning("models.reveal_failed", models_dir=str(models_dir))
            self._folder_error = Message("managedModels.revealFailed")
            self._show_error(self._folder_error)

    def _show_error(self, message: Message) -> None:
        self.result_view.show_result(message, severity="error")
        # The banner is body content: the body just grew, so the bound is re-taken
        # over it. Reached both from _render and directly, hence here rather than
        # only at the end of a render.
        self.fit()


def _header_line() -> QWidget:
    """The line under the column headings: the sheet's separator, in the theme's
    hairline, kept in from the panel's rounded edge by the collections' inset."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    holder = QWidget()
    layout = QHBoxLayout(holder)
    layout.setContentsMargins(COLLECTION_INSET, 0, COLLECTION_INSET, 0)
    layout.addWidget(line)
    return holder


def _bundle_status(
    in_progress: ModelOperation | None,
    failed: ModelOperation | None,
    ready: int,
    total: int,
) -> Message:
    if in_progress is not None:
        if in_progress.kind == "queued":
            return Message("managedModels.statusQueued")
        if in_progress.cancelling:
            return Message("managedModels.cancelling")
        return Message.of(
            "managedModels.statusInstalling",
            progress=_percentage(in_progress.completed_bytes, in_progress.total_bytes),
        )
    if failed is not None:
        return Message("managedModels.statusFailed")
    if ready == total:
        return Message("managedModels.statusInstalled")
    if ready == 0:
        return Message("managedModels.statusNotInstalled")
    return Message.of("managedModels.statusPartial", ready=ready, count=total)


def _percentage(done: int, total: int) -> int:
    return 0 if total <= 0 else min(100, done * 100 // total)


def _model_column_widths(metrics: QFontMetrics) -> tuple[int, ...]:
    """Each column as wide as the widest thing it shows in the current language."""
    of = localizer.of
    model_values = (
        localizer.t(_COLUMN_KEYS[0]),
        *(localizer.display(bundle.label) for bundle in MANAGED_MODEL_BUNDLES),
    )
    purpose_values = (
        localizer.t(_COLUMN_KEYS[1]),
        *(of(bundle.purpose) for bundle in MANAGED_MODEL_BUNDLES),
    )
    size_values = (
        localizer.t(_COLUMN_KEYS[2]),
        *(of(format_bytes(bundle_size_bytes(bundle))) for bundle in MANAGED_MODEL_BUNDLES),
    )
    status_values = (
        localizer.t(_COLUMN_KEYS[3]),
        *(
            localizer.t(key)
            for key in (
                "managedModels.statusQueued",
                "managedModels.statusFailed",
                "managedModels.statusInstalled",
                "managedModels.statusNotInstalled",
                "managedModels.cancelling",
            )
        ),
        localizer.t("managedModels.statusInstalling", progress=100),
        localizer.t("managedModels.statusPartial", ready=3, count=3),
    )
    action_values = (
        localizer.t(_COLUMN_KEYS[4]),
        *(
            localizer.t(key)
            for key in (
                "managedModels.install",
                "managedModels.reinstall",
                "managedModels.cancel",
                "managedModels.cancelling",
            )
        ),
    )
    return tuple(
        _column_width(metrics, values)
        for values in (model_values, purpose_values, size_values, status_values, action_values)
    )


def _column_width(metrics: QFontMetrics, values: Iterable[str]) -> int:
    return max(metrics.horizontalAdvance(value) for value in values) + _COLUMN_TEXT_PADDING


def format_bytes(value: int) -> Message:
    """A size in the largest binary unit that keeps it at or above 1, to one place."""
    if value >= 1024**3:
        return Message.of("units.gigabytes", value=value / 1024**3)
    if value >= 1024**2:
        return Message.of("units.megabytes", value=value / 1024**2)
    if value >= 1024:
        return Message.of("units.kilobytes", value=value / 1024)
    return Message.of("units.bytes", value=value)
