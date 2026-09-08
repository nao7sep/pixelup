from __future__ import annotations

import json

from PySide6.QtCore import QRect, QSize

from pixelup.app_state import (
    AppState,
    WindowBounds,
    WindowPlacement,
    load_app_state,
    save_app_state,
)
from pixelup.window_placement import resolve_window_restoration, usable_window_bounds


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
    state = AppState(WindowPlacement(WindowBounds(10, 20, 1200, 800), "maximized"))
    save_app_state(state, path)
    assert load_app_state(path) == state


def test_bounds_must_meet_minimum_and_fit_wholly_in_one_work_area() -> None:
    areas = [QRect(0, 0, 1920, 1080), QRect(-1280, 0, 1280, 1024)]
    minimum = QSize(900, 600)
    valid = WindowBounds(-1200, 20, 1000, 700)
    assert usable_window_bounds(valid, minimum, areas)
    assert not usable_window_bounds(WindowBounds(10, 10, 899, 700), minimum, areas)
    assert not usable_window_bounds(WindowBounds(1200, 10, 900, 700), minimum, areas)


def test_restoration_keeps_mode_when_bounds_fall_back_and_defaults_maximized() -> None:
    minimum = QSize(900, 600)
    areas = [QRect(0, 0, 1920, 1080)]
    assert resolve_window_restoration(None, minimum, areas) == WindowPlacement(None, "maximized")
    saved = WindowPlacement(WindowBounds(9000, 9000, 1200, 800), "normal")
    assert resolve_window_restoration(saved, minimum, areas) == WindowPlacement(None, "normal")
