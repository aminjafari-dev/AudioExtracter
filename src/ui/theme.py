"""
UI theme constants for the AudioExtracter application.

All colors, fonts, sizes, and the global Qt stylesheet are defined here.
No widget ever hardcodes a hex color or pixel size — it imports from this
module instead so the whole visual language can be updated in one place.

Usage example:
    from src.ui.theme import AppTheme

    label.setStyleSheet(f"color: {AppTheme.TEXT_PRIMARY};")
    widget.setMinimumHeight(AppTheme.TOOLBAR_HEIGHT)
"""


class AppTheme:
    """
    Central registry for all design tokens used in the application.

    Colors follow the dark-theme palette. Typography and spacing values
    are expressed in pixels and are scaled once in build_stylesheet() to
    allow easy density adjustments.
    """

    # ------------------------------------------------------------------
    # Color palette
    # ------------------------------------------------------------------

    # Background layers (darkest → lightest)
    BG_WINDOW     = "#1C1C1E"   # Window / app background
    BG_SURFACE    = "#2C2C2E"   # Cards, panels
    BG_ELEVATED   = "#3A3A3C"   # Hover states, selected rows
    BG_DROP_ZONE  = "#252527"   # Drop zone idle background

    # Accent
    ACCENT        = "#6C63FF"   # Primary purple accent (buttons, focus ring)
    ACCENT_HOVER  = "#8178FF"   # Lighter shade for hover
    ACCENT_PRESS  = "#5549DD"   # Darker shade for pressed state

    # Status colors
    STATUS_DONE    = "#30D158"  # Green — extraction succeeded
    STATUS_FAILED  = "#FF453A"  # Red   — extraction failed
    STATUS_RUNNING = "#FFD60A"  # Yellow — currently processing
    STATUS_PENDING = "#8E8E93"  # Gray  — waiting in queue

    # Text
    TEXT_PRIMARY   = "#FFFFFF"
    TEXT_SECONDARY = "#AEAEB2"
    TEXT_DISABLED  = "#636366"

    # Borders & dividers
    BORDER         = "#3A3A3C"
    BORDER_DROP    = "#6C63FF"   # Drop zone active border

    # ------------------------------------------------------------------
    # Typography
    # ------------------------------------------------------------------

    FONT_FAMILY    = "Inter, -apple-system, Helvetica Neue, Arial, sans-serif"
    FONT_SIZE_SM   = 11   # px — small labels, badges
    FONT_SIZE_BASE = 13   # px — body text, list items
    FONT_SIZE_LG   = 15   # px — section headings
    FONT_SIZE_XL   = 20   # px — secondary hero text
    FONT_SIZE_HERO = 36   # px — empty-state hero text ("Drag your video")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    BORDER_RADIUS  = 10   # px — cards and panels
    BUTTON_RADIUS  = 8    # px — buttons
    TOOLBAR_HEIGHT = 60   # px — bottom toolbar

    # ------------------------------------------------------------------
    # Global stylesheet
    # ------------------------------------------------------------------

    @staticmethod
    def build_stylesheet() -> str:
        """
        Return the application-wide Qt stylesheet as a single string.

        Applied once via QApplication.setStyleSheet(). Individual widgets
        can override specific properties locally, but should stay within
        the palette defined above.

        Example:
            app = QApplication(sys.argv)
            app.setStyleSheet(AppTheme.build_stylesheet())
        """
        t = AppTheme  # shorter alias

        return f"""
        /* ── Global ─────────────────────────────────────────────── */
        QWidget {{
            background-color: {t.BG_WINDOW};
            color: {t.TEXT_PRIMARY};
            font-family: {t.FONT_FAMILY};
            font-size: {t.FONT_SIZE_BASE}px;
        }}

        /* ── Main window ─────────────────────────────────────────── */
        QMainWindow {{
            background-color: {t.BG_WINDOW};
        }}

        /* ── Scroll area & viewport ──────────────────────────────── */
        QScrollArea, QScrollArea > QWidget > QWidget {{
            background-color: {t.BG_WINDOW};
            border: none;
        }}
        QScrollBar:vertical {{
            background: {t.BG_SURFACE};
            width: 6px;
            border-radius: 3px;
        }}
        QScrollBar::handle:vertical {{
            background: {t.BORDER};
            border-radius: 3px;
            min-height: 20px;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}

        /* ── Toolbar / bottom bar ────────────────────────────────── */
        #toolbar {{
            background-color: {t.BG_SURFACE};
            border-top: 1px solid {t.BORDER};
        }}

        /* ── Primary button ─────────────────────────────────────── */
        QPushButton#primaryButton {{
            background-color: {t.ACCENT};
            color: {t.TEXT_PRIMARY};
            border: none;
            border-radius: {t.BUTTON_RADIUS}px;
            padding: 8px 20px;
            font-size: {t.FONT_SIZE_BASE}px;
            font-weight: 600;
        }}
        QPushButton#primaryButton:hover {{
            background-color: {t.ACCENT_HOVER};
        }}
        QPushButton#primaryButton:pressed {{
            background-color: {t.ACCENT_PRESS};
        }}
        QPushButton#primaryButton:disabled {{
            background-color: {t.BG_ELEVATED};
            color: {t.TEXT_DISABLED};
        }}

        /* ── Secondary / ghost button ───────────────────────────── */
        QPushButton#secondaryButton {{
            background-color: {t.BG_ELEVATED};
            color: {t.TEXT_PRIMARY};
            border: 1px solid {t.BORDER};
            border-radius: {t.BUTTON_RADIUS}px;
            padding: 8px 16px;
            font-size: {t.FONT_SIZE_BASE}px;
        }}
        QPushButton#secondaryButton:hover {{
            background-color: #48484A;
        }}
        QPushButton#secondaryButton:pressed {{
            background-color: {t.BG_SURFACE};
        }}

        /* ── Combo box (format selector) ────────────────────────── */
        QComboBox {{
            background-color: {t.BG_ELEVATED};
            color: {t.TEXT_PRIMARY};
            border: 1px solid {t.BORDER};
            border-radius: {t.BUTTON_RADIUS}px;
            padding: 6px 12px;
            font-size: {t.FONT_SIZE_BASE}px;
            min-width: 180px;
        }}
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {t.BG_SURFACE};
            color: {t.TEXT_PRIMARY};
            border: 1px solid {t.BORDER};
            selection-background-color: {t.ACCENT};
            outline: none;
        }}

        /* ── Empty state (no files loaded) ─────────────────────── */
        /* Shown in the center of the content area as a passive prompt.
           No border needed — the whole window is the drop target.    */
        #emptyState {{
            background-color: transparent;
        }}

        /* ── Full-window drag overlay ───────────────────────────── */
        /* Translucent widget rendered on top of all content while a
           drag is hovering over the window. Mouse/drag events pass
           through it (WA_TransparentForMouseEvents) so MainWindow
           still receives the actual dropEvent.                       */
        #dragOverlay {{
            background-color: rgba(44, 44, 72, 180);
            border: 3px solid {t.BORDER_DROP};
            border-radius: {t.BORDER_RADIUS}px;
        }}

        /* ── File list rows ─────────────────────────────────────── */
        #fileRow {{
            background-color: {t.BG_SURFACE};
            border-radius: 8px;
        }}
        #fileRow:hover {{
            background-color: {t.BG_ELEVATED};
        }}

        /* ── Status badge labels ────────────────────────────────── */
        #badgePending  {{ color: {t.STATUS_PENDING}; }}
        #badgeRunning  {{ color: {t.STATUS_RUNNING}; }}
        #badgeDone     {{ color: {t.STATUS_DONE};    }}
        #badgeFailed   {{ color: {t.STATUS_FAILED};  }}

        /* ── Section label ──────────────────────────────────────── */
        #sectionLabel {{
            color: {t.TEXT_SECONDARY};
            font-size: {t.FONT_SIZE_SM}px;
            font-weight: 600;
            letter-spacing: 1px;
        }}
        """
