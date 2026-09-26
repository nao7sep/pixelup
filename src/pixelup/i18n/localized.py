from __future__ import annotations

import weakref
from collections.abc import Callable

import shiboken6
from PySide6.QtWidgets import QAbstractButton, QGroupBox, QLabel, QLineEdit, QWidget

from pixelup.i18n import localizer
from pixelup.i18n.message import Message

# A control holds a key, not a sentence. ``localize`` writes the current
# language's words into a widget and remembers the key, and every binding is
# written again when the language changes, so a label built once in a
# constructor follows a change without its owner doing anything.
#
# The bindings are weak: a widget that has been deleted, or whose dialog has
# closed and been collected, simply drops out, and a widget whose C++ side is
# already gone is skipped rather than written into.

type Text = Message | str
"""A catalogue key, or a key with its values."""

_bindings: weakref.WeakKeyDictionary[QWidget, dict[str, Message]] = weakref.WeakKeyDictionary()


def localize[W: QWidget](
    widget: W,
    *,
    text: Text | None = None,
    title: Text | None = None,
    tooltip: Text | None = None,
    accessible_name: Text | None = None,
    accessible_description: Text | None = None,
    placeholder: Text | None = None,
) -> W:
    """Bind the given roles of ``widget`` to catalogue text and write them now.

    ``title`` is a group box's title or a window's title. A role left as None
    keeps whatever binding it had.
    """
    roles = _bindings.setdefault(widget, {})
    for role, value in (
        ("text", text),
        ("title", title),
        ("tooltip", tooltip),
        ("accessible_name", accessible_name),
        ("accessible_description", accessible_description),
        ("placeholder", placeholder),
    ):
        if value is None:
            continue
        message = value if isinstance(value, Message) else Message(value)
        roles[role] = message
        _apply(widget, role, message)
    return widget


def unlocalize(widget: QWidget, *roles: str) -> None:
    """Stop rewriting ``roles`` of ``widget``; what they show now stays as it is."""
    bound = _bindings.get(widget)
    if bound is None:
        return
    for role in roles:
        bound.pop(role, None)


def bound_message(widget: QWidget, role: str) -> Message | None:
    """The message a role is bound to, for tests and owners that re-derive text."""
    bound = _bindings.get(widget)
    return None if bound is None else bound.get(role)


def _setter(widget: QWidget, role: str) -> Callable[[str], None]:
    match role:
        case "text":
            if isinstance(widget, QLabel | QAbstractButton | QLineEdit):
                return widget.setText
        case "title":
            if isinstance(widget, QGroupBox):
                return widget.setTitle
            return widget.setWindowTitle
        case "tooltip":
            return widget.setToolTip
        case "accessible_name":
            return widget.setAccessibleName
        case "accessible_description":
            return widget.setAccessibleDescription
        case "placeholder":
            if isinstance(widget, QLineEdit):
                return widget.setPlaceholderText
    raise TypeError(f"{type(widget).__name__} has no {role} to localize")


def _apply(widget: QWidget, role: str, message: Message) -> None:
    _setter(widget, role)(localizer.of(message))


def _reapply() -> None:
    for widget, roles in list(_bindings.items()):
        if not shiboken6.isValid(widget):
            continue
        for role, message in roles.items():
            _apply(widget, role, message)


localizer.changed.connect(_reapply)
