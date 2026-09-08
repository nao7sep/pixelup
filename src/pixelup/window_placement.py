from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QRect, QSize

from pixelup.app_state import WindowBounds, WindowPlacement


def usable_window_bounds(
    bounds: WindowBounds,
    minimum: QSize,
    work_areas: Iterable[QRect],
) -> bool:
    if bounds.width < minimum.width() or bounds.height < minimum.height():
        return False
    rectangle = QRect(bounds.x, bounds.y, bounds.width, bounds.height)
    return any(area.contains(rectangle) for area in work_areas)


def resolve_window_restoration(
    saved: WindowPlacement | None,
    minimum: QSize,
    work_areas: Iterable[QRect],
) -> WindowPlacement:
    bounds = (
        saved.normal_bounds
        if saved is not None and saved.normal_bounds is not None
        and usable_window_bounds(saved.normal_bounds, minimum, work_areas)
        else None
    )
    mode = saved.mode if saved is not None else "maximized"
    return WindowPlacement(normal_bounds=bounds, mode=mode)
