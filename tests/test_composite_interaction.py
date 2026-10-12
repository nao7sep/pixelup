"""PixelUp's native choice/list owners keep hover separate from keyboard use."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QAccessible, QColor, QImage, QMouseEvent, QPainter, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QRadioButton,
    QStyle,
    QStyleOptionButton,
    QStyleOptionViewItem,
    QTableWidgetItem,
)

from pixelup import theme
from pixelup.records import RecordSummary
from pixelup.records_window import RecordsListModel, _RecordDelegate, _RecordList
from pixelup.widgets import EmptyStateTableWidget, NoWheelComboBox


def _pointer(view, row: int, kind=QEvent.Type.MouseMove) -> None:
    """Deliver a real Qt event at a measured row without offscreen cursor routing."""
    point = view.visualRect(view.model().index(row, 0)).center()
    button = (
        Qt.MouseButton.NoButton if kind == QEvent.Type.MouseMove else Qt.MouseButton.LeftButton
    )
    buttons = (
        Qt.MouseButton.LeftButton if kind == QEvent.Type.MouseButtonPress
        else Qt.MouseButton.NoButton
    )
    QApplication.sendEvent(view.viewport(), QMouseEvent(
        kind, QPointF(point), QPointF(view.viewport().mapToGlobal(point)),
        button, buttons, Qt.KeyboardModifier.NoModifier,
    ))


@pytest.fixture
def combo(qapp, tmp_path):
    marks = theme.write_control_marks(qapp.palette(), tmp_path / "marks")
    qapp.setStyleSheet(theme.build_stylesheet(qapp.palette(), marks))
    widget = NoWheelComboBox()
    widget.addItems(["A", "B", "C", "D", "E", "F"])
    widget.setCurrentIndex(2)
    widget.resize(180, theme.CONTROL_HEIGHT)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.hidePopup()
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def _open(combo, opening: str) -> None:
    if opening == "pointer":
        QTest.mouseClick(
            combo, Qt.MouseButton.LeftButton,
            pos=QPoint(combo.width() - 12, combo.height() // 2),
        )
    else:
        QTest.keyClick(combo, Qt.Key.Key_Space)
    QApplication.processEvents()
    assert combo.view().isVisible()
    assert combo.view().currentIndex().row() == 2


@pytest.mark.parametrize("opening", ["pointer", "keyboard"])
def test_popup_first_arrow_and_enter_keep_the_keyboard_target_after_hover(combo, opening) -> None:
    committed = []
    combo.activated.connect(committed.append)
    _open(combo, opening)
    view = combo.view()
    QTest.keyClick(view, Qt.Key.Key_Down)
    assert view.currentIndex().row() == 3  # The first arrow moves from the choice.
    QTest.keyClick(view, Qt.Key.Key_Down)
    assert view.currentIndex().row() == 4
    assert combo.currentIndex() == 2
    assert [index.row() for index in view.selectionModel().selectedRows()] == [4]
    accessible = QAccessible.queryAccessibleInterface(view)
    assert accessible.child(4).state().selected
    assert accessible.child(4).state().focused
    assert QAccessible.queryAccessibleInterface(combo).text(QAccessible.Text.Value) == "C"
    focus = QApplication.focusWidget()
    _pointer(view, 0)
    assert view.currentIndex().row() == 4
    assert combo.currentIndex() == 2
    assert QApplication.focusWidget() is focus
    QTest.keyClick(view, Qt.Key.Key_Down)
    assert view.currentIndex().row() == 5
    QTest.keyClick(view, Qt.Key.Key_Return)
    assert combo.currentIndex() == 5
    assert committed == [5]
    assert not view.isVisible()


@pytest.mark.parametrize("opening", ["pointer", "keyboard"])
def test_popup_click_acts_on_the_clicked_item_after_keyboard_navigation(combo, opening) -> None:
    _open(combo, opening)
    view = combo.view()
    QTest.keyClick(view, Qt.Key.Key_Down)
    QTest.keyClick(view, Qt.Key.Key_Down)
    _pointer(view, 0)
    assert view.currentIndex().row() == 4
    _pointer(view, 0, QEvent.Type.MouseButtonPress)
    _pointer(view, 0, QEvent.Type.MouseButtonRelease)
    assert combo.currentIndex() == 0
    assert not view.isVisible()


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("kind", ["combo", "table", "records"])
def test_choice_and_keyboard_focus_remain_distinct_when_hovered(qapp, tmp_path, dark, kind) -> None:
    palette = QPalette(qapp.palette())
    palette.setColor(QPalette.ColorRole.Window, QColor("#323232" if dark else "#efefef"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#f0f0f0" if dark else "#000000"))
    qapp.setPalette(palette)
    marks = theme.write_control_marks(palette, tmp_path / "marks")
    qapp.setStyleSheet(theme.build_stylesheet(palette, marks))
    if kind == "combo":
        widget = NoWheelComboBox()
        widget.addItems(["Choice", "Choice"])
        widget.setCurrentIndex(0)
        view = widget.view()
    elif kind == "table":
        widget = view = EmptyStateTableWidget(2, 1, empty_message="images.empty")
        for row in range(2):
            view.setItem(row, 0, QTableWidgetItem("Choice"))
    else:
        widget = view = _RecordList()
        model = RecordsListModel(view)
        model.replace([
            RecordSummary(row, "session", "2026-10-12T00:00:00.000Z", "error", "Choice", "{}")
            for row in range(2)
        ])
        view.setModel(model)
        view.setItemDelegate(_RecordDelegate(view))
    widget.ensurePolished()
    view.ensurePolished()

    def paint(row, flags):
        height = max(1, view.sizeHintForRow(row))
        image = QImage(180, height, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor(theme.surfaces(palette)["field"]))
        option = QStyleOptionViewItem()
        option.initFrom(view)
        option.rect = QRect(0, 0, 180, height)
        option.widget = view
        if kind != "combo" and row == 0:
            flags |= QStyle.StateFlag.State_Selected
        option.state = QStyle.StateFlag.State_Enabled | QStyle.StateFlag.State_Active | flags
        painter = QPainter(image)
        view.itemDelegate().paint(painter, option, view.model().index(row, 0))
        painter.end()
        return image

    try:
        none = QStyle.StateFlag.State_None
        hover = QStyle.StateFlag.State_MouseOver
        focus = QStyle.StateFlag.State_HasFocus | QStyle.StateFlag.State_KeyboardFocusChange
        selected = paint(0, none)
        ordinary = paint(1, none)
        assert selected != ordinary
        # Native highlighting still represents navigation and agrees with
        # accessibility; the persistent tick separately marks the combo's value.
        if kind == "combo":
            assert paint(1, QStyle.StateFlag.State_Selected) != ordinary
            current = QStyle.StateFlag.State_Selected
            for flags in (current, current | hover, current | hover | focus):
                assert paint(0, flags) != paint(1, flags)
            assert paint(0, current | focus) != paint(0, current)
            assert paint(0, current | hover | focus) != paint(0, current | hover)
        for row in (0, 1):
            assert paint(row, focus) != paint(row, none)
            assert paint(row, hover | focus) != paint(row, hover)
            assert paint(row, hover) != paint(row, none)
        assert paint(0, hover) != paint(1, hover)
    finally:
        widget.deleteLater()


@pytest.mark.parametrize("kind", ["table", "queue", "records"])
def test_list_hover_keeps_keyboard_selection_and_click_selects_its_row(qapp, kind) -> None:
    if kind in ("table", "queue"):
        view = EmptyStateTableWidget(6, 1, empty_message="images.empty")
        for row in range(6):
            view.setItem(row, 0, QTableWidgetItem(str(row)))
        view.setSelectionBehavior(view.SelectionBehavior.SelectRows)
        view.setSelectionMode(
            view.SelectionMode.SingleSelection if kind == "table"
            else view.SelectionMode.ExtendedSelection
        )
    else:
        view = _RecordList()
        model = RecordsListModel(view)
        model.replace([
            RecordSummary(row, "session", "2026-10-12T00:00:00.000Z", "info", str(row), "{}")
            for row in range(6)
        ])
        view.setModel(model)
    view.resize(240, 300)
    view.show()
    view.setFocus()
    qapp.processEvents()
    try:
        view.setCurrentIndex(view.model().index(2, 0))
        modifiers = (
            Qt.KeyboardModifier.ShiftModifier if kind == "queue"
            else Qt.KeyboardModifier.NoModifier
        )
        QTest.keyClick(view, Qt.Key.Key_Down, modifiers)
        QTest.keyClick(view, Qt.Key.Key_Down, modifiers)
        selected = view.selectionModel().selectedIndexes()
        focus = QApplication.focusWidget()
        _pointer(view, 0)
        assert view.currentIndex().row() == 4
        assert view.selectionModel().selectedIndexes() == selected
        assert QApplication.focusWidget() is focus
        QTest.keyClick(view, Qt.Key.Key_Down, modifiers)
        assert view.currentIndex().row() == 5
        if kind == "queue":
            assert [index.row() for index in view.selectionModel().selectedRows()] == [2, 3, 4, 5]
        _pointer(view, 0, QEvent.Type.MouseButtonPress)
        _pointer(view, 0, QEvent.Type.MouseButtonRelease)
        assert view.currentIndex().row() == 0
        assert view.selectionModel().isSelected(view.model().index(0, 0))
    finally:
        view.close()
        view.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("control", [QRadioButton, QCheckBox])
def test_checked_indicator_retains_hover_and_keyboard_focus(qapp, tmp_path, dark, control) -> None:
    palette = QPalette(qapp.palette())
    palette.setColor(QPalette.ColorRole.Window, QColor("#323232" if dark else "#efefef"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#f0f0f0" if dark else "#000000"))
    qapp.setPalette(palette)
    marks = theme.write_control_marks(palette, tmp_path / "marks")
    qapp.setStyleSheet(theme.build_stylesheet(palette, marks))
    button = control("Choice")
    button.setChecked(True)
    button.ensurePolished()
    element = (
        QStyle.ControlElement.CE_RadioButton if control is QRadioButton
        else QStyle.ControlElement.CE_CheckBox
    )

    def paint(flags):
        image = QImage(180, 30, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(palette.color(QPalette.ColorRole.Window))
        option = QStyleOptionButton()
        option.initFrom(button)
        option.text = button.text()
        option.rect = QRect(0, 0, 180, 30)
        option.state = (
            QStyle.StateFlag.State_Enabled | QStyle.StateFlag.State_Active
            | QStyle.StateFlag.State_On | flags
        )
        painter = QPainter(image)
        button.style().drawControl(element, option, painter, button)
        painter.end()
        return image

    try:
        hover = QStyle.StateFlag.State_MouseOver
        focus = QStyle.StateFlag.State_HasFocus | QStyle.StateFlag.State_KeyboardFocusChange
        assert paint(focus) != paint(QStyle.StateFlag.State_None)
        assert paint(hover | focus) != paint(hover)
        assert paint(hover) != paint(QStyle.StateFlag.State_None)
    finally:
        button.deleteLater()
