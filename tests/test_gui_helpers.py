from __future__ import annotations

from pixelup.gui import AdvancedSettings, _advanced_log_payload, _coerce_output_format
from pixelup.paths import OutputFormat


def test_coerce_output_format_accepts_string_values() -> None:
    assert _coerce_output_format("png") == OutputFormat.PNG
    assert _coerce_output_format("jpg") == OutputFormat.JPG


def test_advanced_log_payload_accepts_string_backed_output_format() -> None:
    payload = _advanced_log_payload(AdvancedSettings(output_format="webp"))  # type: ignore[arg-type]

    assert payload["output_format"] == "webp"
