from __future__ import annotations

from PySide6.QtWidgets import (
    QDialogButtonBox,
    QLabel,
    QWidget,
)

from pixelup.dialog_shell import FORM_WIDTH, DialogShell
from pixelup.ui_common import REGULAR_SPACING, secondary_label

# This dialog is one long manual, and opening at the height of the whole thing
# makes a reference surface as tall as the display allows. It takes a reading
# height instead and scrolls, which is what the entries were always going to do
# on a smaller screen anyway.
_BODY_HEIGHT = 480

# One entry per Parameters-panel control, in the panel's own order. This dialog is
# the single home for parameter explanations — the panel itself carries none, so
# it stays narrow enough for the window to fit small screens.
_ENTRIES: tuple[tuple[str, str], ...] = (
    (
        "Scale",
        "How much the image is enlarged: 2x or 4x. The available models are trained "
        "for 4x (the x2 model is the one 2x-native exception); a scale/model "
        "mismatch is surfaced as a queue warning, not an error.",
    ),
    (
        "Denoise",
        "Denoising strength from 0.0 (strongest denoise) to 1.0 (none). Only "
        "realesr-general-x4v3 supports it; every other model ignores the value.",
    ),
    (
        "Alpha mode",
        "How a transparent image's alpha channel is upscaled: through Real-ESRGAN "
        "itself, or with plain bicubic scaling (faster, slightly softer edges).",
    ),
    (
        "Output format",
        "PNG, JPG, or WebP. JPG has no transparency, so alpha is flattened onto a "
        "background color.",
    ),
    (
        "Quality",
        "Compression quality (0-100) used for JPG and WebP. Ignored for PNG.",
    ),
    (
        "Tile size",
        "Images are processed in tiles; peak memory grows with the tile's area, so "
        "smaller tiles use less memory. \"Whole image\" disables tiling — the "
        "fastest path, but it can exhaust GPU memory on large inputs.",
    ),
    (
        "Device",
        "Where inference runs. Auto prefers MPS, then CUDA, then CPU; an "
        "explicitly chosen backend is validated as actually available.",
    ),
    (
        "Strip metadata",
        "Removes the source image's metadata (EXIF and similar) from the output. "
        "If the source carried a color profile, colors are converted to sRGB "
        "before the profile is dropped, so they still display correctly.",
    ),
    (
        "Target profile",
        "Converts the output to the chosen color profile (sRGB, Display P3, or "
        "Adobe RGB). Default keeps the source's own profile untouched.",
    ),
)


class ParametersHelpDialog(DialogShell):
    """Read-only reference for every control in the Parameters panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            "Parameters help",
            parent,
            width=FORM_WIDTH,
            body_height_limit=_BODY_HEIGHT,
        )

        # The shell's body is the sole scroll region and the entries are all of it,
        # so they go straight in. Tighter than the body's default rhythm, because a
        # term and its description are one entry rather than two sections.
        self.body_layout.setSpacing(REGULAR_SPACING)
        for index, (name, text) in enumerate(_ENTRIES):
            if index > 0:
                self.body_layout.addSpacing(8)
            term = QLabel(name)
            font = term.font()
            font.setBold(True)
            term.setFont(font)
            description = secondary_label(text)
            description.setWordWrap(True)
            self.body_layout.addWidget(term)
            self.body_layout.addWidget(description)
        self.body_layout.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        # Close has RejectRole, so `rejected` covers both the button click and
        # Escape with a single, unambiguous close path.
        buttons.rejected.connect(self.reject)
        self.add_footer_widget(buttons)
        self.set_initial_focus(buttons.button(QDialogButtonBox.StandardButton.Close))
        self.fit()
