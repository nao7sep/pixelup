from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pixelup.config import quarantine_corrupt_file, resolve_state_dir, write_managed_text
from pixelup.session_log import log

WindowMode = Literal["normal", "maximized"]


@dataclass(frozen=True, slots=True)
class WindowBounds:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class WindowPlacement:
    normal_bounds: WindowBounds | None
    mode: WindowMode
    qt_geometry: str | None = None


@dataclass(frozen=True, slots=True)
class AppState:
    main_window: WindowPlacement | None = None


def state_path() -> Path:
    return resolve_state_dir() / "state.json"


def load_app_state(path: Path | None = None) -> AppState:
    target = path or state_path()
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return AppState()
    except OSError:
        raise
    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("state root must be an object")
    except (ValueError, json.JSONDecodeError) as exc:
        quarantined = quarantine_corrupt_file(target)
        log.warning(
            "state.corrupt_reset",
            path=str(target),
            quarantined_to=str(quarantined),
            reason=str(exc),
        )
        return AppState()
    placements = value.get("window_placements")
    main = placements.get("main") if isinstance(placements, dict) else None
    return AppState(main_window=_decode_placement(main))


def save_app_state(state: AppState, path: Path | None = None) -> None:
    target = path or state_path()
    placement = state.main_window
    value = {
        "window_placements": {
            "main": None
            if placement is None
            else {
                "normal_bounds": (
                    None if placement.normal_bounds is None else asdict(placement.normal_bounds)
                ),
                "mode": placement.mode,
                "qt_geometry": placement.qt_geometry,
            }
        }
    }
    write_managed_text(target, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _decode_placement(value: object) -> WindowPlacement | None:
    if not isinstance(value, dict):
        return None
    mode: WindowMode = "maximized" if value.get("mode") == "maximized" else "normal"
    geometry = value.get("qt_geometry")
    return WindowPlacement(
        normal_bounds=_decode_bounds(value.get("normal_bounds")),
        mode=mode,
        qt_geometry=geometry if isinstance(geometry, str) else None,
    )


def _decode_bounds(value: object) -> WindowBounds | None:
    if not isinstance(value, dict):
        return None
    fields = [value.get(name) for name in ("x", "y", "width", "height")]
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in fields):
        return None
    if not all(-(2**31) <= item < 2**31 for item in fields):
        return None
    if fields[2] <= 0 or fields[3] <= 0:
        return None
    return WindowBounds(*fields)
