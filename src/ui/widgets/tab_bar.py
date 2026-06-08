"""
Top navigation tab bar widget.

TabBar renders two pill-shaped toggle buttons — "Video" and "Audio" — at the
top of the main window and emits a tab_changed signal whenever the active tab
switches.  The MainWindow uses this signal to swap its central content widget
between the video-extraction panel and the audio-trim panel.

Usage example:
    bar = TabBar(parent=self)
    bar.tab_changed.connect(self._on_tab_changed)

    # In the slot:
    def _on_tab_changed(self, index: int) -> None:
        # 0 = Video tab, 1 = Audio tab
        self._content_stack.setCurrentIndex(index)
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from src.ui.theme import AppTheme


# ---------------------------------------------------------------------------
# Tab constants
# ---------------------------------------------------------------------------

TAB_VIDEO = 0
TAB_AUDIO = 1

_LABELS = {
    TAB_VIDEO: "🎬  Video",
    TAB_AUDIO: "🎵  Audio",
}


# ---------------------------------------------------------------------------
# TabBar widget
# ---------------------------------------------------------------------------

class TabBar(QWidget):
    """
    Horizontal two-tab navigation bar.

    The active tab button uses objectName "tabButtonActive" (accent fill);
    the inactive tab uses "tabButton" (transparent background).  Stylesheet
    rules for both names are defined in AppTheme.build_stylesheet().

    Signals:
        tab_changed(int): emitted on tab switch; value is TAB_VIDEO or TAB_AUDIO.

    Example:
        bar = TabBar()
        bar.tab_changed.connect(lambda idx: stack.setCurrentIndex(idx))
    """

    tab_changed = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("tabBar")
        self.setFixedHeight(46)

        # Track active index so we can skip redundant switches.
        self._active: int = TAB_VIDEO
        self._buttons: dict[int, QPushButton] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the horizontal pill-button row."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(4)

        # Create one button per tab.
        # We use a closure capture via default argument to avoid the classic
        # Python loop-closure trap (all lambdas would capture the same `idx`).
        for idx, label in _LABELS.items():
            btn = QPushButton(label, self)
            btn.setCheckable(False)   # we manage visual state manually
            btn.clicked.connect(lambda checked, i=idx: self._on_tab_clicked(i))
            self._buttons[idx] = btn
            layout.addWidget(btn)

        # Push everything to the left; right side is blank window chrome.
        layout.addStretch()

        # Apply initial active styling.
        self._apply_styles()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_active_tab(self, index: int) -> None:
        """
        Programmatically switch the active tab without emitting tab_changed.

        Use this when the main window needs to sync the tab bar to an
        externally driven state change.

        Args:
            index: TAB_VIDEO (0) or TAB_AUDIO (1).
        """
        if index == self._active:
            return
        self._active = index
        self._apply_styles()

    @property
    def active_tab(self) -> int:
        """Return the currently active tab index."""
        return self._active

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _on_tab_clicked(self, index: int) -> None:
        """
        Handle a tab button click.

        If the clicked tab is already active we do nothing (no redundant
        signal emission). Otherwise we update the active state, repaint
        the buttons, and emit tab_changed.

        Args:
            index: The index of the clicked tab.
        """
        if index == self._active:
            return
        self._active = index
        self._apply_styles()
        self.tab_changed.emit(index)

    def _apply_styles(self) -> None:
        """
        Refresh button objectNames so Qt re-applies stylesheet rules.

        We toggle between "tabButton" (inactive) and "tabButtonActive"
        (active) so the global stylesheet can differentiate them.  After
        renaming we must call style().unpolish/polish to force Qt to
        re-evaluate the stylesheet for each button.
        """
        for idx, btn in self._buttons.items():
            if idx == self._active:
                btn.setObjectName("tabButtonActive")
            else:
                btn.setObjectName("tabButton")

            # Force Qt to re-evaluate the stylesheet after objectName change.
            btn.style().unpolish(btn)
            btn.style().polish(btn)
