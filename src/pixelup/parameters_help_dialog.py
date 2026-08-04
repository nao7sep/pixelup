from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pixelup.ui_common import secondary_label, use_regular_spacing

# One entry per Parameters-panel control, in the panel's own order. This dialog is
# the single home for parameter explanations — the panel itself carries none, so
# it stays narrow enough for the window to fit small screens.
_ENTRIES: tuple[tuple[str, str], ...] = (
    (
        "Scale",
        "How much the image is enlarged: 2x or 4x. The bundled models are trained "
        "for 4x (the x2 model is the one 2x-native exception); a scale/model "
        "mismatch is surfaced as a queue warning, not an error.",
    ),
    (
        "Face enhancement",
        "Restores faces with GFPGAN on top of the upscale. Its model weights are "
        "downloaded on first use, like the upscale models.",
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


class ParametersHelpDialog(QDialog):
    """Read-only reference for every control in the Parameters panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Parameters Help")
        self.setModal(True)

        layout = QVBoxLayout(self)
        use_regular_spacing(layout)

        # Title/footer fixed, body the sole scroll region (modal-dialog
        # conventions): the entries live in a scroll area with a bounded height,
        # so the Close path can never be pushed out of reach.
        body = QWidget()
        body_layout = QVBoxLayout(body)
        use_regular_spacing(body_layout)
        for index, (name, text) in enumerate(_ENTRIES):
            if index > 0:
                body_layout.addSpacing(8)
            term = QLabel(name)
            font = term.font()
            font.setBold(True)
            term.setFont(font)
            description = secondary_label(text)
            description.setWordWrap(True)
            body_layout.addWidget(term)
            body_layout.addWidget(description)
        body_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        # Close has RejectRole, so `rejected` covers both the button click and
        # Escape with a single, unambiguous close path.
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Wide enough that descriptions wrap to a few readable lines; tall enough
        # to show most entries while leaving the rest to the scroll body.
        self.resize(520, 560)
        self.setMinimumSize(420, 320)
