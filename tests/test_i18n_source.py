"""The hard-coded-text gate (localization-conventions).

No sentence a reader sees is written in the shipped source, every key the source
names is in the catalogue, and every catalogue key is used. This reads the source
as Python syntax trees, which is what a test of files rather than of behaviour
does (tests-folder-conventions): a string handed to a call that puts words on
screen must be a key or one of the few listed literals.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from pixelup.i18n.catalogue import locale_file

SOURCE = Path(__file__).resolve().parents[1] / "src" / "pixelup"

# Calls whose string arguments a person reads or hears. A widget constructor
# takes its text first; the setters take it as their only argument.
TEXT_CALLS = frozenset(
    {
        "QLabel",
        "QPushButton",
        "QCheckBox",
        "QRadioButton",
        "QGroupBox",
        "QToolButton",
        "QAction",
        "QTableWidgetItem",
        "setText",
        "setToolTip",
        "setWindowTitle",
        "setTitle",
        "setPlaceholderText",
        "setAccessibleName",
        "setAccessibleDescription",
        "setStatusTip",
        "setWhatsThis",
        "addItem",
        "addItems",
        "insertItem",
        "setItemText",
        "setHorizontalHeaderLabels",
        "addRow",
        "getOpenFileNames",
        "secondary_label",
        "title_label",
        "_item",
    }
)

# Literals allowed in a text position: the product's own name, and the empty
# string a control starts with before a binding writes into it.
ALLOWED_LITERALS = frozenset({"", "PixelUp"})

_KEY = re.compile(r"[a-z][a-zA-Z]*(\.[a-zA-Z0-9]+)+")


def _english_keys() -> set[str]:
    return set(json.loads(locale_file("en").read_text(encoding="utf-8")))


def _namespaces() -> set[str]:
    return {key.split(".", 1)[0] for key in _english_keys()}


def _modules() -> list[tuple[Path, ast.Module]]:
    return [
        (path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        for path in sorted(SOURCE.rglob("*.py"))
    ]


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _text_arguments(node: ast.Call) -> list[ast.expr]:
    arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
    flattened: list[ast.expr] = []
    for argument in arguments:
        if isinstance(argument, ast.List | ast.Tuple):
            flattened.extend(argument.elts)
        else:
            flattened.append(argument)
    return flattened


def _is_log_call(node: ast.Call) -> bool:
    # log.info("event.name", ...): an event name is a key-shaped identifier for
    # the log, never interface text.
    return (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "log"
    )


def test_no_source_gives_a_sentence_to_a_call_that_shows_it() -> None:
    offenders: list[str] = []
    for path, module in _modules():
        for node in ast.walk(module):
            if not isinstance(node, ast.Call) or _call_name(node) not in TEXT_CALLS:
                continue
            for argument in _text_arguments(node):
                if isinstance(argument, ast.JoinedStr):
                    literal = "".join(
                        part.value
                        for part in argument.values
                        if isinstance(part, ast.Constant) and isinstance(part.value, str)
                    )
                elif isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    literal = argument.value
                else:
                    continue
                if literal in ALLOWED_LITERALS or not any(ch.isalpha() for ch in literal):
                    continue
                offenders.append(f"{path.name}:{node.lineno} {_call_name(node)}({literal!r})")
    assert offenders == []


def test_every_key_the_source_names_is_in_the_catalogue() -> None:
    keys = _english_keys()
    namespaces = _namespaces()
    missing: list[str] = []
    for path, module in _modules():
        log_names = {
            id(node.args[0])
            for node in ast.walk(module)
            if isinstance(node, ast.Call) and _is_log_call(node) and node.args
        }
        for node in ast.walk(module):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and _KEY.fullmatch(node.value)
                and node.value.split(".", 1)[0] in namespaces
                and id(node) not in log_names
                and node.value not in keys
            ):
                missing.append(f"{path.name}:{node.lineno} {node.value}")
    assert missing == []


def test_every_key_in_the_catalogue_is_used() -> None:
    # A key nothing names is dead weight every translator still has to translate.
    used = {
        node.value
        for _path, module in _modules()
        for node in ast.walk(module)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert sorted(_english_keys() - used) == []
