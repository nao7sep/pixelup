from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from functools import cache

from PySide6.QtCore import (
    QCoreApplication,
    QEvent,
    QLibraryInfo,
    QLocale,
    QObject,
    QTranslator,
    Signal,
)

from pixelup.i18n import languages
from pixelup.i18n.message import Message
from pixelup.i18n.translator import Translator

# The language the app is speaking now, and the one translator everything reads
# through. It is settled once before the application object exists
# (bootstrap.settle_language) and again whenever Settings is saved. A change
# emits ``changed``, which every surface holding words already on screen answers:
# the bindings in ``localized``, the main window's rows and choices, and the
# empty-state tables. Nothing restarts.


class _Notifier(QObject):
    changed = Signal()


_notifier = _Notifier()
changed = _notifier.changed
"""Emitted after the language has changed, on the thread that changed it."""

_translator = Translator(languages.ENGLISH, QLocale(languages.ENGLISH))
_preference = languages.SYSTEM
_computer_languages: tuple[str, ...] = (languages.ENGLISH,)
_qt_translator: QTranslator | None = None


def current() -> Translator:
    """The translator for the current language. Never None, from the first line of the app."""
    return _translator


def language() -> str:
    """The current language's tag."""
    return _translator.tag


def preference() -> str:
    """The preference as saved: a tag, or languages.SYSTEM."""
    return _preference


def t(key: str, /, **values: object) -> str:
    return _translator.t(key, **values)


def of(message: Message) -> str:
    return _translator.of(message)


def display(label: Message | str) -> str:
    """A choice label: a Message is interface text, a plain string a literal name (MPS, sRGB)."""
    return _translator.of(label) if isinstance(label, Message) else label


@cache
def english() -> Translator:
    """English, for what the app writes down rather than shows: the log and the sidecar."""
    return Translator(languages.ENGLISH, QLocale(languages.ENGLISH))


def use(preference: object, computer_languages: Iterable[str] | None = None) -> None:
    """Speak ``preference`` from now on.

    ``computer_languages`` is the computer's own list, in order, which decides what
    System means. It is read once at launch and remembered, so a later change in
    Settings resolves System the same way and it cannot drift mid-session.
    """
    global _preference, _computer_languages
    if computer_languages is not None:
        _computer_languages = tuple(computer_languages)
    _preference = languages.normalize_preference(preference)
    tag = languages.resolve(_preference, _computer_languages)
    if tag != _translator.tag:
        _switch(tag)


@contextmanager
def speaking(tag: str) -> Iterator[None]:
    """Speak ``tag`` for the length of one test, then restore the previous language.

    Only tests use this; the app changes its language through ``use``. The
    previous language comes back even if a handler raised.
    """
    global _preference
    previous_tag = _translator.tag
    previous_preference = _preference
    _switch(tag)
    try:
        yield
    finally:
        _preference = previous_preference
        if _translator.tag != previous_tag:
            _switch(previous_tag)


def sync_qt_translation() -> None:
    """Load Qt's own translation for the current language into the application.

    Qt draws some words itself — a dialog button box's OK, Cancel and Close, and
    on macOS the application menu's About, Services, Hide and Quit — and takes
    them from its own catalogues, which ship with PySide6 but load only when
    asked. Called when the application object is created and on every change.
    Installing or removing a translator makes Qt post a LanguageChange event to
    every window, which is what retranslates those buttons in place.
    """
    global _qt_translator
    app = QCoreApplication.instance()
    if app is None:
        return
    removed = False
    if _qt_translator is not None:
        removed = QCoreApplication.removeTranslator(_qt_translator)
        _qt_translator = None
    installed = False
    name = languages.qt_translation_name(_translator.tag)
    if name is not None:
        translator = QTranslator(app)
        directory = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        if translator.load(f"qtbase_{name}", directory):
            installed = QCoreApplication.installTranslator(translator)
            _qt_translator = translator
    if not removed and not installed:
        # Nothing Qt-side moved, so Qt posts nothing; say it ourselves so a widget
        # that answers LanguageChange still hears about the switch.
        QCoreApplication.postEvent(app, QEvent(QEvent.Type.LanguageChange))


def _switch(tag: str) -> None:
    global _translator
    locale = languages.formatting_locale(tag, QLocale.system())
    _translator = Translator(tag, locale)
    # The default locale is what a spin box, a date or a number Qt formats itself
    # reads, so it moves with the language.
    QLocale.setDefault(locale)
    sync_qt_translation()
    changed.emit()
