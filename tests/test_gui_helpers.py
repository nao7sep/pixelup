from __future__ import annotations

from pathlib import Path

from PIL import Image

from pixelup.gui import _image_size_text, _plural, _safe_image_size, _status_text


def test_status_text_maps_known_statuses() -> None:
    assert _status_text("pending") == "Pending"
    assert _status_text("succeeded") == "Done"
    assert _status_text("cancelling") == "Cancelling..."


def test_status_text_humanizes_unknown_status() -> None:
    assert _status_text("weird-state") == "Weird state"


def test_image_size_text() -> None:
    assert _image_size_text((640, 480)) == "640 x 480"
    assert _image_size_text(None) == "unavailable"


def test_plural() -> None:
    assert _plural(1, "job") == "job"
    assert _plural(2, "job") == "jobs"
    assert _plural(0, "job") == "jobs"
    assert _plural(2, "child", "children") == "children"


def test_safe_image_size_reads_existing_image(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    Image.new("RGB", (12, 7), "white").save(path)
    assert _safe_image_size(path) == (12, 7)


def test_safe_image_size_returns_none_for_missing_path(tmp_path: Path) -> None:
    assert _safe_image_size(tmp_path / "missing.png") is None


def test_safe_image_size_returns_none_for_non_image(tmp_path: Path) -> None:
    path = tmp_path / "bad.png"
    path.write_text("not an image", encoding="utf-8")
    assert _safe_image_size(path) is None
