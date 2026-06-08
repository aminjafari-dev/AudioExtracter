"""
Empty-state widget shown in the main content area when no videos are queued.

EmptyStateWidget is a purely presentational widget. It shows a large, bold
"Drag your video" prompt centered in the available space. It contains no
drag-and-drop logic — that responsibility now lives in MainWindow, which makes
the entire window a drop target.

Usage example:
    from src.ui.widgets.drop_zone import EmptyStateWidget

    empty_state = EmptyStateWidget(parent=self)
    stack.addWidget(empty_state)   # index 0 in QStackedWidget
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from src.ui.theme import AppTheme


class EmptyStateWidget(QWidget):
    """
    Full-area placeholder rendered when no video files are in the queue.

    Displays three stacked, center-aligned elements:
      1. A large 🎬 emoji acting as a lightweight icon.
      2. Bold hero text: "Drag your video"  (FONT_SIZE_HERO — 36 px).
      3. A secondary hint line listing accepted formats.

    The widget is completely passive — it never calls setAcceptDrops() or
    overrides any drag event. MainWindow is responsible for intercepting
    file drops anywhere in the window and routing them to _on_files_received.

    Example of expected appearance (no files loaded):

        [large empty area]

              🎬

        Drag your video

        MP4, MKV, AVI, MOV and more — or click Browse below

        [toolbar at bottom]
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("emptyState")
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """
        Build the vertically-centered label stack.

        Three labels are stacked with generous spacing so the prompt
        reads cleanly at any window height ≥ 400 px.
        """
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(14)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Large emoji icon — no external SVG/asset dependency required.
        icon_label = QLabel("🎬", self)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("font-size: 56px; background: transparent;")

        # Hero instruction — large and bold as the focal point.
        # FONT_SIZE_HERO (36 px) and font-weight 700 make this prominent.
        hero_label = QLabel("Drag your video", self)
        hero_label.setObjectName("emptyStateHero")
        hero_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hero_label.setStyleSheet(
            f"font-size: {AppTheme.FONT_SIZE_HERO}px; "
            f"font-weight: 700; "
            f"color: {AppTheme.TEXT_PRIMARY}; "
            f"background: transparent;"
        )

        # Secondary hint — smaller, muted colour, lists accepted formats.
        hint_label = QLabel(
            "MP4, MKV, AVI, MOV and more — or click Browse below", self
        )
        hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint_label.setStyleSheet(
            f"font-size: {AppTheme.FONT_SIZE_BASE}px; "
            f"color: {AppTheme.TEXT_SECONDARY}; "
            f"background: transparent;"
        )

        layout.addWidget(icon_label)
        layout.addWidget(hero_label)
        layout.addWidget(hint_label)
