"""
Interactive waveform visualization widget.

WaveformWidget renders an audio waveform from a numpy peak array and
lets the user drag to select a trim region.  It is purely presentational:
it receives data through load_peaks() and emits region_changed when the
selection changes; it never calls ffmpeg or file I/O directly.

Visual design:
  ┌─────────────────────────────────────────────────────┐
  │  [dim bars] [█████ ACCENT BARS █████] [dim bars]   │  ← waveform
  │                ▎                    ▎               │  ← handles
  │  0:00.000    [█ selection overlay █]    3:24.000   │  ← time ruler
  └─────────────────────────────────────────────────────┘

  • Bars outside the selection are rendered in WAVEFORM_BAR (dim purple).
  • Bars inside the selection are rendered in WAVEFORM_SELECTED (accent).
  • A semi-transparent purple rectangle covers the selected region.
  • Two bright vertical handle lines mark the selection edges.
  • A time ruler at the bottom shows 0:00, the selection boundaries, and
    the total duration.

Mouse interaction:
  • Click anywhere         → moves the nearest handle to that position
  • Drag from empty area   → creates a new selection
  • Drag on/near a handle  → moves that handle

Signals:
    region_changed(float, float): emitted on every selection change;
        values are (start_seconds, end_seconds).

Usage example:
    waveform = WaveformWidget(parent=self)
    waveform.region_changed.connect(self._on_region_changed)

    # After loading peaks from AudioLoader:
    waveform.load_peaks(peaks, duration=210.5)

    # Read current selection:
    start, end = waveform.selection
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QLinearGradient,
    QPainter,
    QPen,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

from src.ui.theme import AppTheme


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

_RULER_H     = 22   # px — height of the time ruler strip at the bottom
_HANDLE_W    = 2    # px — width of selection handle lines
_HANDLE_HIT  = 10   # px — horizontal hit-test radius around a handle
_MIN_SEL     = 0.001  # seconds — minimum selection width


# ---------------------------------------------------------------------------
# WaveformWidget
# ---------------------------------------------------------------------------

class WaveformWidget(QWidget):
    """
    Interactive audio waveform with drag-to-select trim region.

    The widget is driven entirely by data pushed in via load_peaks() and
    by the selection state managed through mouse events.  It has no
    internal file I/O.

    Attributes:
        region_changed: signal emitted as (start_sec, end_sec) whenever
            the selection region changes.

    Example:
        w = WaveformWidget()
        w.load_peaks(np.array([...], dtype=np.float32), duration=90.0)
        w.region_changed.connect(lambda s, e: print(f"{s:.3f} → {e:.3f}"))
    """

    region_changed = pyqtSignal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("waveformCanvas")
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)    # receive mouseMoveEvent even without button

        # Waveform data.
        self._peaks: np.ndarray | None = None   # shape (N,), values in [0, 1]
        self._duration: float = 0.0

        # Selection region in seconds.
        self._sel_start: float = 0.0
        self._sel_end:   float = 0.0

        # Drag state.
        # _drag_mode: "none" | "new" | "left" | "right"
        self._drag_mode: str = "none"
        self._drag_anchor: float = 0.0    # fixed side during a "new" drag

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_peaks(self, peaks: np.ndarray, duration: float) -> None:
        """
        Load waveform peak data and reset the selection to cover the full file.

        Call this from the main thread after AudioLoader.load_peaks() finishes
        on a background thread.

        Args:
            peaks:    1-D float32 numpy array with values in [0, 1].
                      Length determines horizontal resolution.
            duration: Total audio duration in seconds.
        """
        self._peaks = peaks.astype(np.float32)
        self._duration = max(duration, 0.001)
        # Default selection: entire file.
        self._sel_start = 0.0
        self._sel_end   = self._duration
        self.update()
        self.region_changed.emit(self._sel_start, self._sel_end)

    def clear(self) -> None:
        """
        Remove the current waveform and reset to empty state.

        Called when the user loads a new file or clears the trimmer panel.
        """
        self._peaks = None
        self._duration = 0.0
        self._sel_start = 0.0
        self._sel_end = 0.0
        self.update()

    def set_selection(self, start: float, end: float) -> None:
        """
        Programmatically set the selection region (e.g. from the spinboxes).

        Clamps both values to [0, duration] and ensures start < end.

        Args:
            start: Selection start in seconds.
            end:   Selection end in seconds.
        """
        start = max(0.0, min(start, self._duration))
        end   = max(0.0, min(end,   self._duration))
        if start > end:
            start, end = end, start
        self._sel_start = start
        self._sel_end   = max(end, start + _MIN_SEL)
        self.update()

    @property
    def selection(self) -> tuple[float, float]:
        """Return the current (start_sec, end_sec) selection."""
        return self._sel_start, self._sel_end

    # ------------------------------------------------------------------
    # Qt paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: ANN001
        """
        Render the waveform, selection region, handles, and time ruler.

        Drawing order (back → front):
          1. Canvas background
          2. Waveform bars (dim outside, accent inside selection)
          3. Selection region semi-transparent overlay
          4. Selection handle vertical lines + triangle ears
          5. Time ruler strip at the bottom
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        w = self.width()
        h = self.height()
        wave_h = h - _RULER_H   # height of the waveform area

        # ----------------------------------------------------------
        # 1. Background
        # ----------------------------------------------------------
        painter.fillRect(0, 0, w, h, QColor(AppTheme.WAVEFORM_BG))

        # ----------------------------------------------------------
        # 2. Waveform bars
        # ----------------------------------------------------------
        if self._peaks is not None and self._duration > 0:
            self._draw_waveform(painter, w, wave_h)

        # ----------------------------------------------------------
        # 3. Selection overlay (skip if no file loaded)
        # ----------------------------------------------------------
        if self._peaks is not None and self._duration > 0:
            self._draw_selection_overlay(painter, w, wave_h)

        # ----------------------------------------------------------
        # 4. Selection handles
        # ----------------------------------------------------------
        if self._peaks is not None and self._duration > 0:
            self._draw_handles(painter, w, wave_h)

        # ----------------------------------------------------------
        # 5. Time ruler
        # ----------------------------------------------------------
        self._draw_ruler(painter, w, h, wave_h)

        # ----------------------------------------------------------
        # Placeholder when no file is loaded
        # ----------------------------------------------------------
        if self._peaks is None:
            self._draw_placeholder(painter, w, wave_h)

        painter.end()

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        """
        Begin a drag.

        If the press is within _HANDLE_HIT pixels of a handle, drag that
        handle.  Otherwise start a brand-new selection from this point.
        """
        if self._duration == 0:
            return

        x = event.position().x()
        t = self._x_to_time(x)

        left_x  = self._time_to_x(self._sel_start)
        right_x = self._time_to_x(self._sel_end)

        # Determine drag mode based on proximity to handles.
        if abs(x - left_x) <= _HANDLE_HIT:
            # Dragging the left (start) handle.
            self._drag_mode = "left"
        elif abs(x - right_x) <= _HANDLE_HIT:
            # Dragging the right (end) handle.
            self._drag_mode = "right"
        else:
            # Starting a new selection from scratch.
            self._drag_mode = "new"
            self._drag_anchor = t
            self._sel_start = t
            self._sel_end   = t

        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        """
        Update selection or handle position during a drag, and set cursor shape.
        """
        if self._duration == 0:
            return

        x = event.position().x()
        t = self._x_to_time(x)

        # Update cursor to indicate draggable handles even without a press.
        if self._drag_mode == "none":
            self._update_cursor(x)

        if self._drag_mode == "new":
            # Extend selection from the anchor to the current position.
            # Start is always the smaller of the two.
            if t < self._drag_anchor:
                self._sel_start = max(0.0, t)
                self._sel_end   = self._drag_anchor
            else:
                self._sel_start = self._drag_anchor
                self._sel_end   = min(self._duration, t)

        elif self._drag_mode == "left":
            # Move the left handle (cannot cross the right handle).
            self._sel_start = max(0.0, min(t, self._sel_end - _MIN_SEL))

        elif self._drag_mode == "right":
            # Move the right handle (cannot cross the left handle).
            self._sel_end = min(self._duration, max(t, self._sel_start + _MIN_SEL))

        if self._drag_mode != "none":
            self.region_changed.emit(self._sel_start, self._sel_end)
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        """Finish the drag and emit a final region_changed signal."""
        if self._drag_mode != "none":
            self._drag_mode = "none"
            self.region_changed.emit(self._sel_start, self._sel_end)
            self._update_cursor(event.position().x())

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _draw_waveform(self, painter: QPainter, w: int, wave_h: int) -> None:
        """
        Draw one vertical bar per horizontal pixel.

        Bars inside the selection are painted in WAVEFORM_SELECTED (accent
        purple); bars outside are painted in WAVEFORM_BAR (dim).

        Using per-pixel bar drawing instead of a polygon gives the classic
        DAW waveform look.
        """
        assert self._peaks is not None
        n = len(self._peaks)
        mid = wave_h // 2

        col_dim     = QColor(AppTheme.WAVEFORM_BAR)
        col_sel     = QColor(AppTheme.WAVEFORM_SELECTED)

        sel_x0 = self._time_to_x(self._sel_start)
        sel_x1 = self._time_to_x(self._sel_end)

        for x in range(w):
            # Map pixel x → peak index.
            peak_idx = int(x / w * n)
            peak_idx = min(peak_idx, n - 1)
            amp = float(self._peaks[peak_idx])

            # Bar height: leave 4px padding from centre to edge.
            bar_h = max(1, int(amp * (mid - 4)))

            # Choose colour based on whether this pixel is inside the selection.
            color = col_sel if sel_x0 <= x <= sel_x1 else col_dim

            painter.setPen(QPen(color, 1))
            painter.drawLine(x, mid - bar_h, x, mid + bar_h)

    def _draw_selection_overlay(self, painter: QPainter, w: int, wave_h: int) -> None:
        """
        Fill the selected region with a semi-transparent accent tint.

        This overlay sits on top of the waveform bars to give a visible
        "selected" region even at low zoom levels.
        """
        x0 = self._time_to_x(self._sel_start)
        x1 = self._time_to_x(self._sel_end)
        if x1 <= x0:
            return

        overlay_color = QColor(108, 99, 255, 45)   # #6C63FF at ~18% alpha
        painter.fillRect(x0, 0, x1 - x0, wave_h, overlay_color)

    def _draw_handles(self, painter: QPainter, w: int, wave_h: int) -> None:
        """
        Draw two bright vertical lines at the selection boundaries.

        Each handle also has a small downward triangle "ear" at the top and
        an upward triangle at the bottom to make it obvious they are
        draggable markers (like handles in a professional DAW).
        """
        handle_color = QColor(AppTheme.WAVEFORM_HANDLE)

        for t in (self._sel_start, self._sel_end):
            x = self._time_to_x(t)
            painter.setPen(QPen(handle_color, _HANDLE_W))
            painter.drawLine(x, 0, x, wave_h)

            # Draw small triangle ears at top and bottom of the handle.
            _draw_triangle_ear(painter, x, 0, pointing_down=True, color=handle_color)
            _draw_triangle_ear(painter, x, wave_h, pointing_down=False, color=handle_color)

    def _draw_ruler(
        self, painter: QPainter, w: int, h: int, wave_h: int
    ) -> None:
        """
        Draw the time ruler strip below the waveform.

        Shows: 0:00.000, selection start, selection end, total duration.
        Each label is clamped so it does not overflow the widget edge.
        """
        ruler_y = wave_h
        ruler_color = QColor(AppTheme.WAVEFORM_RULER_BG)
        painter.fillRect(0, ruler_y, w, _RULER_H, ruler_color)

        painter.setPen(QColor(AppTheme.WAVEFORM_RULER_FG))
        font = QFont(AppTheme.FONT_FAMILY.split(",")[0].strip(), 9)
        painter.setFont(font)

        def draw_time_label(t: float, label_x: int, align: Qt.AlignmentFlag) -> None:
            """Draw a formatted time string centered/left/right at label_x."""
            text = _format_time(t)
            fm = painter.fontMetrics()
            text_w = fm.horizontalAdvance(text)
            text_h = fm.height()

            # Clamp to widget bounds.
            if align == Qt.AlignmentFlag.AlignLeft:
                tx = max(2, min(label_x, w - text_w - 2))
            elif align == Qt.AlignmentFlag.AlignRight:
                tx = max(2, min(label_x - text_w, w - text_w - 2))
            else:
                tx = max(2, min(label_x - text_w // 2, w - text_w - 2))

            ty = ruler_y + (_RULER_H + text_h) // 2 - 2
            painter.drawText(int(tx), int(ty), text)

        # Always show 0:00 at the far left.
        draw_time_label(0.0, 2, Qt.AlignmentFlag.AlignLeft)

        # Show total duration at the far right.
        if self._duration > 0:
            draw_time_label(self._duration, w - 2, Qt.AlignmentFlag.AlignRight)

        # Show selection start and end (accent color).
        if self._peaks is not None and self._duration > 0:
            painter.setPen(QColor(AppTheme.WAVEFORM_HANDLE))
            sx = self._time_to_x(self._sel_start)
            ex = self._time_to_x(self._sel_end)
            # Only draw selection labels if they are not at 0 or duration.
            if self._sel_start > 0.1:
                draw_time_label(self._sel_start, sx, Qt.AlignmentFlag.AlignHCenter)
            if self._sel_end < self._duration - 0.1:
                draw_time_label(self._sel_end, ex, Qt.AlignmentFlag.AlignHCenter)

    def _draw_placeholder(self, painter: QPainter, w: int, wave_h: int) -> None:
        """
        Draw a centered hint text when no audio is loaded.

        This is purely decorative — the drop zone above the widget handles
        the actual file loading interaction.
        """
        painter.setPen(QColor(AppTheme.TEXT_DISABLED))
        font = QFont(AppTheme.FONT_FAMILY.split(",")[0].strip(), 12)
        painter.setFont(font)
        from PyQt6.QtCore import QRect
        painter.drawText(
            QRect(0, 0, w, wave_h),
            Qt.AlignmentFlag.AlignCenter,
            "Load an audio file to see its waveform",
        )

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _x_to_time(self, x: float) -> float:
        """
        Convert a pixel x-coordinate to a time in seconds.

        Clamps the result to [0, duration].
        """
        if self.width() == 0 or self._duration == 0:
            return 0.0
        t = (x / self.width()) * self._duration
        return max(0.0, min(t, self._duration))

    def _time_to_x(self, t: float) -> int:
        """
        Convert a time in seconds to a pixel x-coordinate.

        Returns an integer pixel position clamped to [0, width].
        """
        if self._duration == 0:
            return 0
        x = (t / self._duration) * self.width()
        return int(max(0, min(x, self.width())))

    def _update_cursor(self, x: float) -> None:
        """
        Change the mouse cursor based on whether it is hovering over a handle.

        Near a handle → SizeHorCursor (resize arrows).
        Elsewhere     → ArrowCursor.
        """
        if self._duration == 0:
            return
        left_x  = self._time_to_x(self._sel_start)
        right_x = self._time_to_x(self._sel_end)
        if abs(x - left_x) <= _HANDLE_HIT or abs(x - right_x) <= _HANDLE_HIT:
            self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))


# ---------------------------------------------------------------------------
# Drawing utility
# ---------------------------------------------------------------------------

def _draw_triangle_ear(
    painter: QPainter,
    cx: int,
    y: int,
    pointing_down: bool,
    color: QColor,
    size: int = 6,
) -> None:
    """
    Draw a small filled triangle "ear" on a selection handle.

    The ear is centered horizontally on `cx` and sits at `y`.  When
    pointing_down=True (top ear) the triangle points downward into the
    waveform; when pointing_down=False (bottom ear) it points upward.

    Args:
        painter:      Active QPainter.
        cx:           Center x of the handle line.
        y:            y-coordinate where the tip or base sits.
        pointing_down: Direction of the triangle.
        color:        Fill and outline color.
        size:         Half-width / height of the triangle in pixels.
    """
    from PyQt6.QtGui import QPolygon
    from PyQt6.QtCore import QPoint

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)

    # Build the three triangle vertices.
    if pointing_down:
        # Base at y, tip at y + size.
        pts = QPolygon([
            QPoint(cx - size, y),
            QPoint(cx + size, y),
            QPoint(cx, y + size),
        ])
    else:
        # Base at y, tip at y - size.
        pts = QPolygon([
            QPoint(cx - size, y),
            QPoint(cx + size, y),
            QPoint(cx, y - size),
        ])

    painter.drawPolygon(pts)


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------

def _format_time(seconds: float) -> str:
    """
    Format a duration in seconds as M:SS.mmm.

    Examples:
        _format_time(0.0)    → "0:00.000"
        _format_time(90.5)   → "1:30.500"
        _format_time(3600.0) → "60:00.000"
    """
    total_ms = int(round(seconds * 1000))
    minutes  = total_ms // 60000
    secs_ms  = (total_ms % 60000) / 1000.0
    return f"{minutes}:{secs_ms:06.3f}"
