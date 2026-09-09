from __future__ import annotations

import json

from pixelup.app_state import AppState, load_app_state, save_app_state


def test_missing_and_legacy_placement_are_ignored(tmp_path) -> None:
    path = tmp_path / "state.json"
    assert load_app_state(path) == AppState()

    path.write_text(
        json.dumps(
            {
                "window_placements": {
                    "main": {
                        "normal_bounds": {"x": 10, "y": 20, "width": 1200, "height": 800},
                        "mode": "maximized",
                        "qt_geometry": "disposable-legacy-value",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert load_app_state(path) == AppState()


def test_geometry_blob_round_trips(tmp_path) -> None:
    path = tmp_path / "state.json"
    state = AppState(main_window_geometry="encoded-native-geometry")

    save_app_state(state, path)

    assert load_app_state(path) == state
