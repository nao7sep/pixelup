"""The interface in every language: no key reaches the screen, no English is left
over, a language change rewrites what is already drawn, and labels fit the
surfaces that cannot grow (localization-conventions).

Each surface is populated so that every part of it holds words — results
showing, rows in every state, a batch waiting on models — because a key or an
English leftover hides in exactly the part a bare surface does not draw.
"""

from __future__ import annotations

import gc
import json
import re
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from event_wait import process_until
from localization_support import speaking
from PIL import Image
from PySide6.QtCore import QCoreApplication, QEvent, Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QTableView,
    QWidget,
)

from pixelup.about_dialog import AboutDialog
from pixelup.app_config import AppConfig, ConfigLoadResult
from pixelup.fonts import system_ui_font_family
from pixelup.formats import RefusedFile
from pixelup.gui import MainWindow
from pixelup.i18n import localizer
from pixelup.i18n.catalogue import locale_file
from pixelup.i18n.languages import LANGUAGES, TAGS
from pixelup.i18n.message import Message
from pixelup.launch_files import refused_files_message
from pixelup.managed_models_dialog import ManagedModelsDialog
from pixelup.message_dialogs import MessageDialog, StartupFailureDialog
from pixelup.model_management import GENERAL_DENOISE_MODEL, UPSCALE_MODELS
from pixelup.model_manager import ModelManager
from pixelup.model_registry import ALL_MODELS
from pixelup.parameters_help_dialog import ParametersHelpDialog
from pixelup.queue_table import QueueTableView
from pixelup.quit_dialog import QuitConfirmDialog, QuitSaveFailedDialog
from pixelup.runner import JobRunner
from pixelup.session_log import configure_session_logging
from pixelup.settings_dialog import DiscardChangesDialog, SettingsDialog
from pixelup.shortcuts_dialog import ShortcutsDialog
from pixelup.time_zones import system_zone_id, zone_ids
from pixelup.widgets import EmptyStateTableWidget

KEYS = set(json.loads(locale_file("en").read_text(encoding="utf-8")))
_KEY_SHAPE = re.compile(
    r"\b(?:" + "|".join(sorted({key.split(".")[0] for key in KEYS})) + r")\.[a-zA-Z]+\b"
)

# Words that read the same in every language: names, standards, file formats,
# key tokens, and the languages' own names in the picker.
# A path is shown as it is: the files a launch stop or a store notice names.
_REFUSED_CONFIG = "/data/config.json"
_BACKUPS_FILE = "/data/backups.sqlite3"
_LITERAL_WORDS = (
    _REFUSED_CONFIG,
    _BACKUPS_FILE,
    *sorted(UPSCALE_MODELS, key=len, reverse=True),
    "PIXELUP_DATA_DIR",
    "PixelUp",
    "GitHub",
    "Real-ESRGAN",
    # Chinese interfaces keep the technical term, as in Photoshop's "Alpha 通道".
    "Alpha",
    "General",
    "sRGB",
    "Display",
    "Adobe",
    "RGB",
    "MPS",
    "CUDA",
    "CPU",
    "GPU",
    "PNG",
    "JPG",
    "WEBP",
    "WebP",
    "EXIF",
    "HTTPS",
    "GNU",
    "GPL",
    "Yoshinao",
    "Inoguchi",
    "Cmd",
    "Ctrl",
    "Comma",
    "Slash",
    "Question",
    "Enter",
    "Backspace",
    "Delete",
    "pth",
    "png",
    "jpg",
    "webp",
    *(name for _tag, name in LANGUAGES),
)
_ENGLISH_WORD = re.compile(r"[A-Za-z]{3,}")



def _open(window, paths) -> None:
    """Offer ``paths`` and wait until the window has checked and applied them, and
    any preview they selected has loaded."""
    window.open_paths(paths)
    process_until(lambda: not (window.opening or window.preview.loading), what="Opening")

def _png(directory: Path, name: str) -> Path:
    path = directory / name
    Image.new("RGB", (8, 6), "white").save(path)
    return path


@pytest.fixture
def window(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until
) -> Iterator[MainWindow]:
    monkeypatch.setenv("PIXELUP_DATA_DIR", str(tmp_path / "home"))
    monkeypatch.setattr(JobRunner, "schedule", lambda self: None)
    monkeypatch.setattr(
        "pixelup.gui.load_app_config_result",
        lambda: ConfigLoadResult(AppConfig()),
    )
    configure_session_logging()
    models_dir = tmp_path / "home" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    # Every model but the denoise companion, which no queued job here needs, so
    # the models line says some are missing.
    for info in ALL_MODELS:
        if info.name != GENERAL_DENOISE_MODEL:
            (models_dir / info.filename).write_bytes(b"ready")
    runtime_dirs = SimpleNamespace(models_dir=models_dir)
    main = MainWindow(runtime_dirs=runtime_dirs)
    try:
        _populate(main, tmp_path)
        yield main
    finally:
        # Setup can fail before yield. Retain the window and its test patches
        # until every background owner has returned, including abandoned plans.
        main._is_session_shutdown = lambda: True
        main.close()
        process_until(
            lambda: main._workers_clean_for_quit() and main._active_job_plan is None,
            what="Interface fixture workers",
        )
        main.deleteLater()
        qapp.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _settle_models(window):
    process_until(
        lambda: window.model_manager.readiness_known
        and window._active_job_plan is None and not window._pending_job_plans,
        what="Models and plans settle",
    )


def _populate(main: MainWindow, directory: Path) -> None:
    images = [_png(directory, name) for name in ("a.png", "b.png")]
    folder = directory / "d"
    folder.mkdir()
    _open(main, [*images, folder])
    for model in UPSCALE_MODELS[1:3]:
        main.model_checks[model].setChecked(True)
    main._queue_all_images_selected_models()
    _settle_models(main)
    done, failed, cancelled, running = main.jobs
    main._job_finished(done.id, True, Message("queue.statusDone"), {"ok": True}, [])
    main._job_finished(
        failed.id,
        False,
        Message.of("error.modelMissing", model=failed.model),
        {},
        [Message.of("warning.scaleMismatch", model=failed.model, native="2", scale="4")],
    )
    main._job_finished(
        cancelled.id, False, Message("queue.statusCancelled"), {"cancelled": True}, []
    )
    running.status = "running"
    main._job_progress(running.id, Message.of("progress.tiles", done=3, count=8))
    main._select_image(running.input_path)
    main._remove_selected_image()
    main.queue_action_result.show_result(Message("actions.needModels"), severity="warning")
    main.parameters_result.show_result(Message("parameters.saveFailed"), severity="error")


def test_population_failure_closes_window_after_planning_returns(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_until,
) -> None:
    from shiboken6 import isValid

    from pixelup import gui

    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    created: list[MainWindow] = []
    disposed: list[tuple[bool, bool]] = []
    real_create_jobs = gui.create_jobs

    def held_plan(*args, **kwargs):
        entered.set()
        assert release.wait(), "held interface plan was not released"
        try:
            return real_create_jobs(*args, **kwargs)
        finally:
            returned.set()

    def failed_population(main: MainWindow, directory: Path) -> None:
        created.append(main)
        main.destroyed.connect(lambda *_: disposed.append(
            (main._active_job_plan is None, not main.runner.running)
        ))
        _settle_models(main)
        _open(main, [_png(directory, "setup.png")])
        main.model_checks[UPSCALE_MODELS[1]].setChecked(True)
        main._queue_all_images_selected_models()
        process_until(entered.is_set, what="Held interface plan")
        QTimer.singleShot(0, release.set)
        raise RuntimeError("injected interface population failure")

    monkeypatch.setattr(gui, "create_jobs", held_plan)
    monkeypatch.setitem(globals(), "_populate", failed_population)
    setup = window.__wrapped__(qapp, tmp_path, monkeypatch, process_until)
    try:
        with pytest.raises(RuntimeError, match="injected interface population failure"):
            next(setup)
        assert returned.is_set()
        assert disposed == [(True, True)]
    finally:
        # Keep a failing regression safe too: release and close any owner that
        # the fixture under test failed to retire before its patches disappear.
        release.set()
        for main in created:
            if isValid(main):
                main._is_session_shutdown = lambda: True
                main.close()
                process_until(
                    main._workers_clean_for_quit, what="Failed interface regression workers",
                )
                main.deleteLater()
                qapp.processEvents()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _dialogs(manager: ModelManager) -> list[QDialog]:
    failed = Message("settings.saveFailed")
    settings = SettingsDialog(
        AppConfig(), try_save=lambda _candidate, finished: finished(failed)
    )
    settings.ok_button.setEnabled(True)
    settings._save()
    about = AboutDialog(opener=lambda _url: (_ for _ in ()).throw(OSError("offline")))
    about._open_external("https://example.com", Message("about.openIssuesFailed"))
    batch = ManagedModelsDialog(
        manager, required_artifacts=(UPSCALE_MODELS[0],), pending_job_count=3
    )
    library = ManagedModelsDialog(manager)
    library._show_error(Message("managedModels.revealFailed"))
    return [
        settings,
        about,
        ShortcutsDialog(),
        ParametersHelpDialog(),
        QuitConfirmDialog(0),
        QuitConfirmDialog(3),
        QuitSaveFailedDialog(),
        DiscardChangesDialog(),
        MessageDialog("app.name", Message.of("notice.backupsOff", path=_BACKUPS_FILE)),
        StartupFailureDialog(
            refused_files_message(
                (
                    RefusedFile(Path(_REFUSED_CONFIG), newer=True),
                    RefusedFile(Path(_REFUSED_CONFIG), newer=False),
                )
            ),
            Message("startup.refusedHint"),
        ),
        StartupFailureDialog(
            Message("error.storageCreateFailed"),
            Message.of("error.hintHomeWritableLocation", variable="PIXELUP_DATA_DIR"),
        ),
        batch,
        library,
    ]


def _texts(root: QWidget, *, include_cells: bool = True) -> list[str]:
    """Every string ``root`` shows or announces."""
    texts: list[str] = []
    for widget in [root, *root.findChildren(QWidget)]:
        if widget.isWindow():
            texts.append(widget.windowTitle())
        texts += [widget.toolTip(), widget.accessibleName(), widget.accessibleDescription()]
        if isinstance(widget, QLabel) and widget.pixmap().isNull():
            texts.append(widget.text())
        elif isinstance(widget, QAbstractButton):
            texts.append(widget.text())
        elif isinstance(widget, QGroupBox):
            texts.append(widget.title())
        elif isinstance(widget, QLineEdit):
            # The folder placeholder is a literal filesystem path, not UI copy.
            if not (isinstance(root, ManagedModelsDialog) and widget is root.folder_edit):
                texts.append(widget.placeholderText())
        elif isinstance(widget, QComboBox):
            texts += [widget.itemText(index) for index in range(widget.count())]
        if isinstance(widget, (EmptyStateTableWidget, QueueTableView)):
            texts.append(widget.empty_text)
        if isinstance(widget, QTableView):
            model = widget.model()
            texts += [
                model.headerData(column, Qt.Orientation.Horizontal)
                for column in range(model.columnCount())
            ]
            if include_cells:
                for row in range(model.rowCount()):
                    for column in range(model.columnCount()):
                        index = model.index(row, column)
                        texts += [index.data(), index.data(Qt.ItemDataRole.ToolTipRole)]
    return [text for text in texts if text]


def _english_left_over(text: str) -> list[str]:
    # IANA zone IDs are names, listed as they are in every language.
    if text in zone_ids():
        return []
    # The system font's family is a name, shown as the UI font field's placeholder.
    for word in (*_LITERAL_WORDS, system_ui_font_family(), system_zone_id()):
        text = text.replace(word, " ")
    return _ENGLISH_WORD.findall(text)


def _status_cells(main: MainWindow) -> list[str]:
    table = main.queue_table
    model = table.model()
    return [model.index(row, 4).data() for row in range(model.rowCount())] + [
        main.image_table.item(row, 2).text() for row in range(main.image_table.rowCount())
    ]


@pytest.mark.parametrize("tag", TAGS)
def test_no_key_and_no_english_reaches_any_surface(
    tag: str, window: MainWindow, qapp: QApplication, tmp_path: Path, process_until
) -> None:
    manager = ModelManager(tmp_path / "models")
    # The Records window, with its launches read so the launch filter holds this
    # launch's label; the records themselves are stored text, not interface text.
    window._open_records_window()
    records = window._records_window
    assert records is not None
    process_until(lambda: records.launch_filter.count() > 1, what="the Records window")
    with speaking(tag):
        qapp.processEvents()
        dialogs = _dialogs(manager)
        try:
            surfaces: list[QWidget] = [window, records, *dialogs]
            for surface in surfaces:
                for text in _texts(surface):
                    assert not _KEY_SHAPE.search(text), f"{tag}: a key reached the screen: {text!r}"
            if tag in {"ja", "ko", "zh-Hans", "ru"}:
                # These scripts share no letters with English, so a Latin word
                # that is not a name is a string that was never translated.
                shown = [text for s in (records, *dialogs) for text in _texts(s)]
                shown += _texts(window, include_cells=False) + _status_cells(window)
                leftovers = {text: _english_left_over(text) for text in shown}
                assert {text: words for text, words in leftovers.items() if words} == {}
        finally:
            for dialog in dialogs:
                dialog.deleteLater()


def test_a_language_change_rewrites_what_is_already_on_screen(
    window: MainWindow, qapp: QApplication
) -> None:
    gc.collect()
    assert window.open_images_button.text() == "Open"
    with speaking("de"):
        qapp.processEvents()
        assert window.open_images_button.text() == "Öffnen"
        assert window.findChild(QGroupBox).title() in {"Bilder", "Vorschau", "Modelle"}
        assert window.image_table.horizontalHeaderItem(0).text() == "Bild"
        assert window.target_profile.itemText(0) == "Standard"
        assert window.device.itemText(0) == "Automatisch"
        assert "Fertig" in _status_cells(window)
        assert any(cell.startswith("1 fertig") for cell in _status_cells(window))
        assert window.queue_action_result.message_label.text() == (
            "Wählen Sie mindestens ein Modell aus, bevor Sie Aufträge einreihen."
        )
        assert window.remove_result.message_label.text().startswith("Dieses Bild")
        assert window.model_status.text() == "Einige Modelle sind nicht installiert."
        assert "fehlgeschlagen" in window.queue_failure_label.text()
        # Qt's own words follow too, through its own translation.
        dialog = ShortcutsDialog()
        try:
            close = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Close)
            assert close.text() == "Schließen"
        finally:
            dialog.deleteLater()
    qapp.processEvents()
    assert window.open_images_button.text() == "Open"
    assert window.image_table.horizontalHeaderItem(0).text() == "Image"
    assert any(cell.startswith("1 done") for cell in _status_cells(window))


def test_saving_a_language_in_settings_applies_it_live(
    window: MainWindow, qapp: QApplication, monkeypatch: pytest.MonkeyPatch, process_until
) -> None:
    def accept_japanese(dialog: SettingsDialog) -> QDialog.DialogCode:
        dialog.language.setCurrentIndex(dialog.language.findData("ja"))
        dialog._save()
        process_until(lambda: not dialog._saving, what="language save")
        return dialog.result()

    monkeypatch.setattr(SettingsDialog, "exec", accept_japanese)
    window._settings_dialog()
    process_until(lambda: window.config.language == "ja", what="saved language")
    qapp.processEvents()
    assert window.config.language == "ja"
    assert localizer.preference() == "ja"
    assert window.open_images_button.text() == "開く"


def test_a_closed_dialog_is_left_alone_by_a_language_change(qapp: QApplication) -> None:
    dialog = AboutDialog()
    dialog.show()
    dialog.close()
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    kept = AboutDialog()
    kept.show()
    kept.close()
    with speaking("ru"):
        qapp.processEvents()
        assert kept.windowTitle() == "О программе PixelUp"
    kept.deleteLater()


def test_the_settings_picker_offers_system_then_each_language_in_its_own_name(
    qapp: QApplication,
) -> None:
    with speaking("ja"):
        dialog = SettingsDialog(AppConfig(language="ko"))
        try:
            items = [
                (dialog.language.itemText(index), dialog.language.itemData(index))
                for index in range(dialog.language.count())
            ]
            # System names the language it resolves to on this computer (pinned to
            # English for tests), in the interface language.
            assert items[0] == ("システム（英語）", "system")
            assert items[1:] == list((name, tag) for tag, name in LANGUAGES)
            assert dialog.language.currentData() == "ko"
        finally:
            dialog.deleteLater()


@pytest.mark.parametrize("tag", TAGS)
def test_labels_fit_the_dialogs_that_cannot_grow(
    tag: str, qapp: QApplication, tmp_path: Path
) -> None:
    # Every dialog has one chosen width and wraps its prose; what cannot wrap —
    # a row label, a column of statuses, a footer of buttons — must fit it.
    manager = ModelManager(tmp_path / "models")
    with speaking(tag):
        dialogs = _dialogs(manager)
        try:
            for dialog in dialogs:
                dialog.show()
                qapp.processEvents()
                viewport = dialog.body_scroll.viewport().width()
                body = dialog.body.minimumSizeHint().width()
                footer = dialog.footer_layout.minimumSize().width()
                name = type(dialog).__name__
                assert body <= viewport, f"{tag}: {name} body needs {body}px of {viewport}px"
                assert footer <= dialog.width(), (
                    f"{tag}: {name} footer needs {footer}px of {dialog.width()}px"
                )
                dialog.close()
        finally:
            for dialog in dialogs:
                dialog.deleteLater()


@pytest.mark.parametrize(
    ("tag", "computer", "language", "zone"),
    [
        ("en", "ja", "System (Japanese)", "System ({zone})"),
        ("ja", "de", "システム（ドイツ語）", "システム（{zone}）"),
        ("ru", "zh-Hans", "Системный (китайский)", "Системный ({zone})"),
    ],
)
def test_system_entries_name_what_they_resolve_to_in_the_interface_language(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tag: str,
    computer: str,
    language: str,
    zone: str,
) -> None:
    monkeypatch.setattr(localizer, "system_language", lambda: computer)
    with speaking(tag):
        dialog = SettingsDialog(AppConfig(language=tag))
        try:
            assert dialog.language.itemText(0) == language
            assert dialog.display_zone.itemText(0) == zone.format(zone=system_zone_id())
        finally:
            dialog.deleteLater()


def test_every_language_names_every_language(qapp: QApplication) -> None:
    from pixelup.i18n.languages import name_key

    for tag in TAGS:
        with speaking(tag):
            names = [localizer.t(name_key(other)) for other in TAGS]
        assert len(set(names)) == len(TAGS), tag
