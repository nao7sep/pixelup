from __future__ import annotations

from PySide6.QtWidgets import (
    QDialogButtonBox,
    QLabel,
    QWidget,
)

from pixelup.dialog_shell import FORM_WIDTH, DialogShell
from pixelup.i18n.localized import localize
from pixelup.ui_common import REGULAR_SPACING, secondary_label

# This dialog is one long manual, and opening at the height of the whole thing
# makes a reference surface as tall as the display allows. It takes a reading
# height instead and scrolls, which is what the entries were always going to do
# on a smaller screen anyway.
_BODY_HEIGHT = 480

# One entry per Parameters-panel control, in the panel's own order: the control's
# own label and its explanation. This dialog is the single home for parameter
# explanations — the panel itself carries none, so it stays narrow enough for the
# window to fit small screens.
_ENTRIES: tuple[tuple[str, str], ...] = (
    ("parameters.scale", "help.scale"),
    ("parameters.denoise", "help.denoise"),
    ("parameters.alphaMode", "help.alphaMode"),
    ("parameters.outputFormat", "help.outputFormat"),
    ("parameters.quality", "help.quality"),
    ("parameters.tileSize", "help.tileSize"),
    ("parameters.device", "help.device"),
    ("parameters.stripMetadata", "help.stripMetadata"),
    ("parameters.targetProfile", "help.targetProfile"),
)


class ParametersHelpDialog(DialogShell):
    """Read-only reference for every control in the Parameters panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            "help.title",
            parent,
            width=FORM_WIDTH,
            body_height_limit=_BODY_HEIGHT,
        )

        # The shell's body is the sole scroll region and the entries are all of it,
        # so they go straight in. Tighter than the body's default rhythm, because a
        # term and its description are one entry rather than two sections.
        self.body_layout.setSpacing(REGULAR_SPACING)
        for index, (name_key, text_key) in enumerate(_ENTRIES):
            if index > 0:
                self.body_layout.addSpacing(8)
            term = localize(QLabel(), text=name_key)
            font = term.font()
            font.setBold(True)
            term.setFont(font)
            description = localize(secondary_label(""), text=text_key)
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
