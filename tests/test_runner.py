from __future__ import annotations

from pixelup.runner import _download_text, _progress_text, _tile_progress_text


def test_download_text_with_total_shows_percentage() -> None:
    assert _download_text("RealESRGAN_x4plus", 1, 4) == "25% - downloading RealESRGAN_x4plus"


def test_download_text_without_total_shows_plain_message() -> None:
    assert _download_text("modelx", 0, None) == "Downloading modelx"
    assert _download_text("modelx", 0, 0) == "Downloading modelx"


def test_progress_text_maps_known_phases() -> None:
    assert _progress_text("upscale") == "Upscaling"
    assert _progress_text("encode") == "Saving"


def test_progress_text_humanizes_unknown_phase() -> None:
    assert _progress_text("post-process_step") == "Post process step"


def test_tile_progress_text() -> None:
    assert _tile_progress_text(3, 8) == "3/8 tiles processed"
