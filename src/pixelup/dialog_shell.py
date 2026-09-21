from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pixelup.ui_common import DIALOG_MARGIN, DIALOG_SPACING, REGULAR_SPACING
from pixelup.widgets import PassiveScrollArea

# The share of the screen's working area a whole dialog may take — the title bar
# the OS draws above it included, which is what the allowance below stands in for
# (modal-dialog-conventions).
DIALOG_HEIGHT_FRACTION = 0.85
_DECORATION_ALLOWANCE = 48

# Below this a bound would be doing more harm than the overflow it prevents.
MIN_BODY_HEIGHT = 160

_FOOTER_MARGIN_Y = 16

# One chosen width per surface rather than one that follows its content, so the
# same dialog keeps its shape whatever it happens to be saying
# (modal-dialog-conventions). Three roles, three widths: a column of prose, a
# form or reference, and the one table.
NOTICE_WIDTH = 440
FORM_WIDTH = 560
TABLE_WIDTH = 820


class DialogShell(QDialog):
    """The two bands a dialog takes when the OS draws its title bar.

    That title bar is the header: it already carries the name and the close
    control, so nothing inside repeats the window's title and there is no line
    above the body. Below it come the body — the sole scroll region, carrying its
    own padding so its scroll bar sits at the dialog's edge rather than against a
    control — then the hairline that opens the fixed footer
    (modal-dialog-conventions).

    The surface is fixed in both directions. None of PixelUp's dialogs is one
    anyone settles into and works in, so none of them earns an edge to drag; what
    replaces the handle is a chosen width and a body bounded to a share of the
    screen, both applied before the window is shown. A body that outgrows the
    bound scrolls, and the footer never moves.
    """

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        *,
        width: int,
        passive_body_name: str | None = None,
    ) -> None:
        # Dialog + exec() produces an ordinary titled native window on macOS.
        # QDialog.open() chooses the sheet presentation instead, hiding the native
        # title bar and traffic-light controls — and that title bar is this
        # layout's header.
        super().__init__(parent, Qt.WindowType.Dialog)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._width = width

        outer = QVBoxLayout(self)
        # No margins out here: each band carries its own padding, so the body's
        # scroll region reaches the dialog's edge and the footer's line spans it.
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._body = QWidget()
        self.body_layout = QVBoxLayout(self._body)
        self.body_layout.setContentsMargins(
            DIALOG_MARGIN, DIALOG_MARGIN, DIALOG_MARGIN, DIALOG_MARGIN
        )
        self.body_layout.setSpacing(DIALOG_SPACING)

        # An informational body opts into the shared keyboard-scroll owner; a form
        # keeps its fields as the focus and keyboard owners, so its scroll region
        # takes no focus of its own (app-chrome-conventions).
        if passive_body_name is not None:
            self.body_scroll: QScrollArea = PassiveScrollArea(accessible_name=passive_body_name)
        else:
            self.body_scroll = QScrollArea()
            self.body_scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.body_scroll.setFrameShape(QFrame.Shape.NoFrame)
            self.body_scroll.setWidgetResizable(True)
            self.body_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body_scroll.setWidget(self._body)
        outer.addWidget(self.body_scroll)

        self._footer_line = QFrame()
        self._footer_line.setObjectName("dialogFooterLine")
        self._footer_line.setFixedHeight(1)
        self._footer_line.setStyleSheet(
            "QFrame#dialogFooterLine { background: palette(mid); }"
        )
        outer.addWidget(self._footer_line)

        self._footer = QWidget()
        self.footer_layout = QHBoxLayout(self._footer)
        self.footer_layout.setContentsMargins(
            DIALOG_MARGIN, _FOOTER_MARGIN_Y, DIALOG_MARGIN, _FOOTER_MARGIN_Y
        )
        self.footer_layout.setSpacing(REGULAR_SPACING)
        # Everything a caller adds lands after this, so the action row is right
        # aligned however many buttons it holds.
        self.footer_layout.addStretch()
        outer.addWidget(self._footer)

    @property
    def body(self) -> QWidget:
        """The widget inside the scroll region: everything but the footer band."""
        return self._body

    def add_footer_widget(self, widget: QWidget) -> None:
        """Append an action to the footer band, left to right."""
        self.footer_layout.addWidget(widget)

    def fit(self) -> None:
        """Settle the chosen width and the body's bound. Call once the bands are filled.

        Also call it again whenever the body's content changes height on its own —
        a result banner appearing, a validation message — because the bound is
        arithmetic over the content, and Qt's ``adjustSize`` applies no screen cap
        of its own.
        """
        screen = self.screen() or QApplication.primaryScreen()
        work_height = screen.availableGeometry().height() if screen is not None else 720
        # The footer band, its line, and the scroll region's own painted border are
        # all fixed, so what is left of the share is the body's to grow into.
        frame = 2 * self.body_scroll.frameWidth()
        chrome = self._footer_line.height() + self._footer.sizeHint().height() + frame
        limit = int(work_height * DIALOG_HEIGHT_FRACTION) - _DECORATION_ALLOWANCE
        cap = max(MIN_BODY_HEIGHT, limit - chrome)

        # A wrapped label answers a different height at every width, and its
        # ``sizeHint`` answers for none of them — it reports the shape Qt would
        # pick if it were free to choose. The height that will actually be drawn
        # is the one the layout computes for the viewport's own width, which is
        # also what QScrollArea itself asks for once the widget is resizable.
        natural = self._body_height_for(self._width - frame)
        body_height = min(natural, cap)
        self.body_scroll.setMinimumHeight(body_height + frame)
        self.body_scroll.setMaximumHeight(cap + frame)
        # Fixed, and added up here rather than asked of the layout: the outer
        # layout answers its own natural width, not the chosen one, and a column
        # of bands with no margins or spacing between them is exactly their sum.
        self.setFixedSize(self._width, body_height + chrome)

    def _body_height_for(self, width: int) -> int:
        layout = self._body.layout()
        layout.activate()
        if layout.hasHeightForWidth():
            height = layout.totalHeightForWidth(width)
            if height > 0:
                return height
        return layout.totalSizeHint().height()
