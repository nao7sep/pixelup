from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pixelup.app_config import MAX_CONCURRENT_JOBS, MIN_CONCURRENT_JOBS, AppConfig
from pixelup.dialog_shell import FORM_WIDTH, NOTICE_WIDTH, DialogShell
from pixelup.fonts import system_ui_font_family
from pixelup.i18n import localizer
from pixelup.i18n.languages import LANGUAGES, SYSTEM
from pixelup.i18n.localized import localize
from pixelup.i18n.message import Message
from pixelup.text_cleanup import single_line
from pixelup.ui_common import secondary_label, use_regular_spacing
from pixelup.widgets import NoWheelComboBox, NoWheelSpinBox

# A settled column width for the value field so the control and its wrapped
# caption share one edge, and the caption wraps predictably instead of stretching
# the dialog to the widest single line.
_FIELD_WIDTH = 320


def _captioned(control: QWidget, caption_key: str) -> QWidget:
    """Group a control with a muted caption directly beneath it.

    The caption reads as sub-text of its control (a tight 2px gap) rather than a
    peer form row a full row-gap away. The control spans the column; the parameters
    live in the main window's panel rather than this dialog.
    """
    container = QWidget()
    container.setFixedWidth(_FIELD_WIDTH)
    box = QVBoxLayout(container)
    box.setContentsMargins(0, 0, 0, 0)
    box.setSpacing(2)
    box.addWidget(control)
    cap = localize(secondary_label(""), text=caption_key)
    cap.setWordWrap(True)
    box.addWidget(cap)
    return container


class DiscardChangesDialog(DialogShell):
    """Ask before Settings throws away its draft (modal-dialog-conventions).

    ``exec()`` returns ``Accepted`` to discard, ``Rejected`` (Keep editing, Escape)
    to stay in Settings.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("settings.unsavedTitle", parent, width=NOTICE_WIDTH)

        message = localize(QLabel(), text="settings.unsavedMessage")
        message.setWordWrap(True)
        self.body_layout.addWidget(message)

        keep_button = localize(QPushButton(), text="settings.keepEditing")
        keep_button.setDefault(True)
        keep_button.clicked.connect(self.reject)

        discard_button = localize(QPushButton(), text="settings.discard")
        discard_button.setProperty("role", "danger-confirm")
        discard_button.clicked.connect(self.accept)

        self.add_footer_widget(keep_button)
        self.add_footer_widget(discard_button)
        self.set_initial_focus(keep_button)
        self.fit()


class SettingsDialog(DialogShell):
    """Modal settings editor: everything PixelUp persists that the main window does not show.

    One home per thing. The image-processing parameters live in the main window's
    Parameters panel, which persists its own edits and resets to its own built-ins,
    so this dialog deliberately holds only the leftovers: the interface language, the
    UI font and the concurrent job count. It carries no reset button — no setting here
    is a stale-able built-in or a tuned, interacting set worth returning to, so there
    is no default worth a control (config-sets-conventions).

    The widgets hold a draft; the incoming config is never mutated. The commit
    (OK) button stays disabled until the draft differs from the config the
    dialog was opened with, so accepting always means "apply a real change".
    """

    def __init__(
        self,
        config: AppConfig,
        parent: QWidget | None = None,
        *,
        try_save: Callable[[AppConfig, Callable[[Message | None], None], Callable[[], None]], None]
        | None = None,
        session_shutdown: Callable[[], bool] = lambda: False,
    ) -> None:
        super().__init__("settings.title", parent, width=FORM_WIDTH)
        self._saving = False
        self._save_generation = 0
        self._initial = config
        self._try_save = try_save
        self._session_shutdown = session_shutdown

        form_widget = QWidget()
        form = QGridLayout(form_widget)
        use_regular_spacing(form)

        self.concurrent = NoWheelSpinBox()
        self.concurrent.setRange(MIN_CONCURRENT_JOBS, MAX_CONCURRENT_JOBS)
        self.concurrent.setValue(config.max_concurrent_jobs)

        self.font_family = QLineEdit()
        self.font_family.setText(config.font_family)
        # A built-in derived at runtime shows as the placeholder (config-sets-conventions).
        self.font_family.setPlaceholderText(system_ui_font_family())
        self.font_family.setMinimumWidth(260)

        # System first, then each language in its own name, so a reader finds
        # theirs whatever language is showing (localization-conventions). The
        # names are the languages' own and are never translated. A modal dialog
        # renders its choices once: the language cannot change while it is up.
        self.language = NoWheelComboBox()
        self.language.addItem(localizer.t("settings.languageSystem"), SYSTEM)
        for tag, name in LANGUAGES:
            self.language.addItem(name, tag)
        self.language.setCurrentIndex(max(0, self.language.findData(config.language)))

        # Each caption groups under its own control (see _captioned) rather than
        # floating a full row-gap away, so the label reads as sub-text of the
        # field it explains. Row labels top-align to sit beside the control, not
        # the caption.
        label_align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        row = 0
        form.addWidget(localize(QLabel(), text="settings.language"), row, 0)
        form.addWidget(self.language, row, 1, Qt.AlignmentFlag.AlignLeft)
        row += 1
        form.addWidget(localize(QLabel(), text="settings.uiFont"), row, 0, label_align)
        form.addWidget(_captioned(self.font_family, "settings.uiFontCaption"), row, 1)
        row += 1
        form.addWidget(localize(QLabel(), text="settings.concurrentJobs"), row, 0)
        form.addWidget(self.concurrent, row, 1, Qt.AlignmentFlag.AlignLeft)
        form.setColumnStretch(0, 0)
        form.setColumnStretch(1, 0)
        form.setColumnStretch(2, 1)

        # Built by hand rather than with QDialogButtonBox, whose platform layout puts
        # OK first on Windows (modal-dialog-conventions).
        self.cancel_button = localize(QPushButton(), text="settings.cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.ok_button = localize(QPushButton(), text="settings.ok")
        # OK is what saves, so it is the dialog's primary action.
        self.ok_button.setProperty("role", "primary")
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self._save)
        self.error_message = QLabel()
        self.error_message.setWordWrap(True)
        dark = self.palette().color(QPalette.ColorRole.Window).lightness() < 128
        self.error_message.setStyleSheet(
            f"color: {'#ff766a' if dark else '#b3261e'}; font-weight: 600;"
        )
        self.error_message.hide()

        self.body_layout.addWidget(form_widget, 0, Qt.AlignmentFlag.AlignLeft)
        self.body_layout.addWidget(self.error_message)
        self.body_layout.addStretch()
        self.add_footer_widget(self.cancel_button)
        self.add_footer_widget(self.ok_button)
        self.set_initial_focus(self.font_family)
        self.fit()

        for changed in (
            self.concurrent.valueChanged,
            self.font_family.textChanged,
            self.language.currentIndexChanged,
        ):
            changed.connect(self._update_commit_enabled)
        self._update_commit_enabled()

    def config(self) -> AppConfig:
        """The draft config: the opened one with exactly this dialog's fields replaced.

        Built by ``replace`` rather than a fresh ``AppConfig(...)`` so the settings this
        dialog does not show — today the Parameters panel — pass through untouched.
        Constructing one here would silently reset the panel to its built-ins every time
        the user pressed OK on a font change.
        """
        return replace(
            self._initial,
            max_concurrent_jobs=self.concurrent.value(),
            font_family=single_line(self.font_family.text()),
            language=self.language.currentData(),
        )

    def is_dirty(self) -> bool:
        return self.config() != self._initial

    def reject(self) -> None:
        """The one close path for Cancel, Escape and the title bar's close button."""
        if self._saving and not self._session_shutdown():
            return
        if (
            self.is_dirty()
            and not self._session_shutdown()
            and DiscardChangesDialog(self).exec() != QDialog.DialogCode.Accepted
        ):
            return
        self._save_generation += 1
        super().reject()

    def _update_commit_enabled(self) -> None:
        self.ok_button.setEnabled(self.is_dirty() and not self._saving)

    def _save(self) -> None:
        if self._saving:
            return
        candidate = self.config()
        self.error_message.clear()
        self.error_message.hide()
        if self._try_save is None:
            self.accept()
            return
        self._saving = True
        self._save_generation += 1
        generation = self._save_generation
        self._update_commit_enabled()
        self.cancel_button.setEnabled(False)
        self.concurrent.setEnabled(False)
        self.font_family.setEnabled(False)
        self.language.setEnabled(False)
        self._try_save(
            candidate,
            lambda failure: self._saved(generation, candidate, failure),
            lambda: self._saved(generation, candidate, Message("notice.configSaving")),
        )

    def _saved(self, generation: int, candidate: AppConfig, failure: Message | None) -> None:
        if generation != self._save_generation:
            return
        self._saving = False
        self.cancel_button.setEnabled(True)
        self.concurrent.setEnabled(True)
        self.font_family.setEnabled(True)
        self.language.setEnabled(True)
        self._update_commit_enabled()
        if failure is not None:
            localize(self.error_message, text=failure, accessible_name=failure)
            self.error_message.show()
            self.fit()
            return
        self.error_message.clear()
        self.error_message.hide()
        if candidate == self.config():
            self.accept()
