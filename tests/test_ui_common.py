from __future__ import annotations

from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import QApplication

from pixelup.i18n.message import Message
from pixelup.ui_common import NAMED_IN_MESSAGE, identifying_name, named_values


def test_identifying_name_keeps_the_beginning_of_a_long_name(qapp: QApplication) -> None:
    metrics = QFontMetrics(QFont())
    assert identifying_name("short.png", metrics) == "short.png"
    name = "holiday-" + "x" * 200 + ".png"
    shown = identifying_name(name, metrics)
    assert shown.startswith("holiday-")
    assert shown.endswith("\u2026")
    assert metrics.horizontalAdvance(shown) < metrics.horizontalAdvance(name) / 4


def test_named_values_name_a_few_and_count_the_rest(qapp: QApplication) -> None:
    metrics = QFontMetrics(QFont())
    names = [f"{index}.png" for index in range(NAMED_IN_MESSAGE + 4)]
    assert named_values(names, metrics) == (
        *names[:NAMED_IN_MESSAGE],
        Message.of("common.more", count=4),
    )
    assert named_values(names[:2], metrics) == tuple(names[:2])
