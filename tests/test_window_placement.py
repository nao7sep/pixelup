from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QWidget

from pixelup import window_placement
from pixelup.app_state import (
    AppState,
    WindowBounds,
    WindowPlacement,
    load_app_state,
    save_app_state,
)
from pixelup.window_placement import capture_window_placement, restore_window_placement


def test_missing_and_malformed_placement_self_heal_without_corrupting_state(tmp_path) -> None:
    path = tmp_path / "state.json"
    assert load_app_state(path) == AppState()
    path.write_text(json.dumps({"window_placements": {"main": {
        "normal_bounds": {"x": 10, "y": 20, "width": "wide", "height": 700},
        "mode": "fullscreen",
    }}}), encoding="utf-8")
    assert load_app_state(path) == AppState(
        main_window=WindowPlacement(normal_bounds=None, mode="normal")
    )


def test_state_round_trip_is_atomic_managed_text(tmp_path) -> None:
    path = tmp_path / "state.json"
    state = AppState(WindowPlacement(
        WindowBounds(10, 20, 1200, 800), "maximized", "encoded-native-geometry"
    ))
    save_app_state(state, path)
    assert load_app_state(path) == state


@pytest.mark.parametrize("bounds", [
    WindowBounds(9000, 9000, 200, 150),
    WindowBounds(10, 10, 1, 1),
    WindowBounds(10, 10, 10000, 10000),
])
def test_qt_adjusts_legacy_bounds_to_minimum_and_available_screen(qapp, bounds) -> None:
    window = QWidget()
    window.setMinimumSize(200, 150)
    try:
        assert restore_window_placement(window, WindowPlacement(bounds, "normal")) == "normal"
        assert window.width() >= 200
        assert window.height() >= 150
        assert qapp.primaryScreen().availableGeometry().contains(window.geometry())
    finally:
        window.close()


@pytest.mark.parametrize("mode", ["normal", "maximized"])
def test_bad_qt_geometry_keeps_mode_and_designed_defaults(qapp, mode) -> None:
    window = QWidget()
    window.setGeometry(20, 30, 300, 250)
    try:
        assert restore_window_placement(window, None) == "maximized"
        assert restore_window_placement(window, WindowPlacement(None, mode, "bad!")) == mode
        assert window.geometry() == QRect(20, 30, 300, 250)
    finally:
        window.close()


@pytest.mark.parametrize("mode", ["normal", "maximized"])
@pytest.mark.parametrize("transient", ["showMinimized", "showFullScreen"])
def test_qt_blob_restores_normal_landing_bounds_without_transient_mode(qapp, mode, transient):
    original = QWidget()
    restored = QWidget()
    original.setGeometry(40, 50, 300, 250)
    original.show()
    try:
        if mode == "maximized":
            original.showMaximized()
        getattr(original, transient)()
        placement = capture_window_placement(original, mode)
        assert placement.mode == mode
        assert restore_window_placement(restored, placement) == mode
        assert restored.windowState() == Qt.WindowState.WindowNoState
        assert restored.geometry() == QRect(40, 50, 300, 250)
    finally:
        original.close()
        restored.close()


def test_qt_geometry_round_trip_does_not_drift(qapp) -> None:
    window = QWidget()
    window.setGeometry(40, 50, 300, 250)
    window.show()
    placement = capture_window_placement(window, "normal")
    window.close()
    for _ in range(3):
        window = QWidget()
        restore_window_placement(window, placement)
        window.show()
        placement = capture_window_placement(window, "normal")
        window.close()
        assert placement.normal_bounds == WindowBounds(40, 50, 300, 250)


def test_native_unzoom_uses_cocoa_state_instead_of_dimensions(monkeypatch) -> None:
    monkeypatch.setattr(window_placement.sys, "platform", "darwin")
    monkeypatch.setattr(window_placement, "_cocoa_messages", lambda: (
        lambda view, selector: 456 if (view, selector) == (123, b"window") else None,
        lambda window, selector: False if (window, selector) == (456, b"isZoomed") else None,
        lambda name: name,
    ))
    window = SimpleNamespace(
        isMaximized=lambda: True,
        isMinimized=lambda: False,
        isFullScreen=lambda: False,
        winId=lambda: 123,
        geometry=lambda: QRect(80, 90, 300, 250),
    )
    assert capture_window_placement(window, "maximized") == WindowPlacement(
        WindowBounds(80, 90, 300, 250), "normal"
    )
