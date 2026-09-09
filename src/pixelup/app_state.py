from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pixelup.config import quarantine_corrupt_file, resolve_state_dir, write_managed_text
from pixelup.session_log import log


@dataclass(frozen=True, slots=True)
class AppState:
    main_window_geometry: str | None = None


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
    geometry = value.get("main_window_geometry")
    return AppState(main_window_geometry=geometry if isinstance(geometry, str) else None)


def save_app_state(state: AppState, path: Path | None = None) -> None:
    target = path or state_path()
    value = {"main_window_geometry": state.main_window_geometry}
    write_managed_text(target, json.dumps(value, indent=2, sort_keys=True) + "\n")
