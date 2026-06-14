"""
Video compression panel widget.

This panel fills the entire "Compress" tab. It lets the user:
  1. Drop (or browse) a video file onto the panel.
  2. Choose the output codec (H.264 / H.265), CRF quality level, and
     encoding speed preset.
  3. Optionally pick a custom output folder.
  4. Click "Compress Video" to run ffmpeg in a background thread.
  5. Watch a live progress bar + elapsed / estimated remaining time.
  6. See the before/after file sizes and the percentage saved.

Layout overview:
    VideoCompressorPanel (QWidget)
    ├─ _inner_stack (QStackedWidget)
    │   ├─ [0] _drop_zone       — shown when no file is loaded
    │   └─ [1] _file_bar        — shown when a file is loaded
    ├─ _settings_section        — codec / CRF / preset controls
    ├─ _progress_section        — progress bar + time labels (hidden at rest)
    └─ _action_bar              — output folder + Compress button + status label

Progress tracking:
    ffmpeg is launched with ``-progress pipe:1`` so it writes structured
    key=value progress data to stdout every ~0.5 s. The worker reads
    ``out_time_us`` (microseconds of video encoded so far) and divides by
    the total duration (obtained via ffprobe beforehand) to compute a
    0–100 % value that is emitted as a Qt signal to the UI thread.

    A QTimer fires every second to refresh the elapsed / remaining display
    independently of the ffmpeg output rate.

Drag-and-drop:
    The panel itself accepts drops (setAcceptDrops(True)) so it works
    independently of the window-level DnD used by the Video tab.

Usage (internal — created by MainWindow._build_ui):
    panel = VideoCompressorPanel(parent=page_stack)
    page_stack.addWidget(panel)
"""

import threading
import time
from pathlib import Path

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.core.compressor import VideoCompressor, _parse_ffmpeg_error
from src.core.models import (
    CompressionJob,
    CompressionPreset,
    CompressionResult,
    VideoCodec,
)
from src.ui.theme import AppTheme
from src.utils.file_utils import (
    build_video_output_path,
    human_readable_size,
)
from src.utils.validators import VIDEO_FILE_FILTER, filter_supported_videos

import subprocess


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _CompressSignals(QObject):
    """
    Qt signal carrier for _CompressWorker.

    QRunnable is not a QObject, so signals must live in a separate QObject.
    The worker holds a reference to this instance and emits from the
    background thread; Qt automatically queues the delivery to the UI thread.

    Signals:
        progress_updated (float):
            Emitted periodically during encoding.
            Value is 0.0–99.0 for deterministic mode (duration known), or
            -1.0 to signal indeterminate mode (duration unavailable).
        finished (CompressionResult):
            Emitted exactly once when encoding ends (success or failure).
    """

    progress_updated = pyqtSignal(float)
    finished         = pyqtSignal(CompressionResult)


class _CompressWorker(QRunnable):
    """
    Runs one CompressionJob on a background thread from QThreadPool.

    Strategy:
      1. Use ffprobe (via compressor.get_duration) to get the total duration.
         If that fails we fall back to indeterminate progress (emit -1.0).
      2. Launch ffmpeg with ``-progress pipe:1 -nostats`` so structured
         progress data is written to stdout instead of stderr.
      3. Drain stderr on a daemon thread to avoid pipe-buffer deadlock.
      4. Parse ``out_time_us=<microseconds>`` lines from stdout and emit
         progress_updated with the percentage encoded so far (0–99 %).
      5. After ffmpeg exits, emit finished with the CompressionResult.

    Args:
        job:        The CompressionJob to process.
        compressor: Shared VideoCompressor instance (stateless, thread-safe).
        signals:    _CompressSignals instance for cross-thread communication.

    Usage (internal — created by VideoCompressorPanel._on_compress):
        worker = _CompressWorker(job, compressor, signals)
        QThreadPool.globalInstance().start(worker)
    """

    def __init__(
        self,
        job: CompressionJob,
        compressor: VideoCompressor,
        signals: _CompressSignals,
    ) -> None:
        super().__init__()
        self._job        = job
        self._compressor = compressor
        self._signals    = signals

    @pyqtSlot()
    def run(self) -> None:
        """
        Execute compression with live progress reporting.

        This method is called on a pooled background thread.  It blocks
        until ffmpeg finishes and must not touch any Qt widgets directly.
        """
        job = self._job
        job.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Record the original size before we start encoding.
        try:
            original_size = job.input_path.stat().st_size
        except OSError:
            original_size = 0

        # Probe total duration so we can express progress as a percentage.
        # Returns 0.0 on failure — we switch to indeterminate mode in that case.
        total_duration = self._compressor.get_duration(job.input_path)

        # Signal the UI immediately: either deterministic (0 %) or indeterminate.
        if total_duration <= 0:
            self._signals.progress_updated.emit(-1.0)
        else:
            self._signals.progress_updated.emit(0.0)

        # Build the ffmpeg command.
        # -progress pipe:1  → write key=value progress data to stdout every ~0.5 s
        # -nostats          → suppress the interleaved stderr progress lines
        command = [
            self._compressor.ffmpeg_path,
            "-y",
            "-i", str(job.input_path),
            "-vcodec", job.codec.value,
            "-crf",    str(job.crf),
            "-preset", job.preset.value,
            "-acodec", "aac",
            "-b:a",    "192k",
            "-progress", "pipe:1",
            "-nostats",
            str(job.output_path),
        ]

        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Drain stderr on a daemon thread so the pipe buffer never fills
            # and causes a deadlock during a long encode.
            stderr_lines: list[str] = []

            def _drain_stderr() -> None:
                """Read all stderr lines into stderr_lines."""
                assert proc.stderr is not None
                for line in proc.stderr:
                    stderr_lines.append(line)

            stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
            stderr_thread.start()

            # Read progress from stdout line by line.
            # ffmpeg writes blocks like:
            #   frame=120
            #   out_time_us=4000000      ← microseconds encoded so far
            #   out_time=00:00:04.000000
            #   progress=continue        ← "end" on the final block
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()

                # Only process timing lines when we have a known duration.
                if line.startswith("out_time_us=") and total_duration > 0:
                    try:
                        us = int(line.split("=", 1)[1])
                        if us >= 0:
                            elapsed_s = us / 1_000_000
                            # Cap at 99 % — the finished signal will snap to 100 %.
                            pct = min(elapsed_s / total_duration * 100.0, 99.0)
                            self._signals.progress_updated.emit(pct)
                    except (ValueError, IndexError):
                        pass

            proc.wait()
            stderr_thread.join(timeout=5.0)

            if proc.returncode != 0:
                stderr_text = "".join(stderr_lines)
                error_detail = _parse_ffmpeg_error(stderr_text)
                self._signals.finished.emit(CompressionResult(
                    success=False,
                    error_message=f"ffmpeg error: {error_detail}",
                    original_size=original_size,
                ))
                return

            try:
                compressed_size = job.output_path.stat().st_size
            except OSError:
                compressed_size = 0

            self._signals.finished.emit(CompressionResult(
                success=True,
                output_path=job.output_path,
                original_size=original_size,
                compressed_size=compressed_size,
            ))

        except FileNotFoundError:
            self._signals.finished.emit(CompressionResult(
                success=False,
                error_message="ffmpeg executable not found.",
                original_size=original_size,
            ))
        except Exception as exc:  # noqa: BLE001
            self._signals.finished.emit(CompressionResult(
                success=False,
                error_message=str(exc),
                original_size=original_size,
            ))


# ---------------------------------------------------------------------------
# VideoCompressorPanel
# ---------------------------------------------------------------------------

class VideoCompressorPanel(QWidget):
    """
    Full-tab panel for video compression.

    Drop a video file onto the panel (or use Browse), adjust the quality
    settings, and click "Compress Video".  The result is placed next to the
    original file (or in a custom output folder) with the suffix "_compressed".

    Signals:
        (none — all interactions are self-contained within this panel)

    Example:
        panel = VideoCompressorPanel(parent=stack)
        stack.addWidget(panel)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("compressorPanel")
        # Accept file drops at the panel level (independent of the window).
        self.setAcceptDrops(True)

        # Currently loaded source video path; None while the drop zone is shown.
        self._input_path: Path | None = None
        # User-chosen output directory; None means "same folder as input".
        self._output_dir: Path | None = None

        # Progress tracking state.
        self._compress_start_time: float = 0.0
        self._current_progress: float    = 0.0  # 0–100 or -1 for indeterminate

        try:
            self._compressor = VideoCompressor()
        except EnvironmentError as exc:
            self._compressor        = None  # type: ignore[assignment]
            self._ffmpeg_error_msg  = str(exc)
        else:
            self._ffmpeg_error_msg = ""

        # Worker signals.
        self._signals = _CompressSignals()
        self._signals.progress_updated.connect(self._on_progress_updated)
        self._signals.finished.connect(self._on_compress_finished)

        # QTimer fires every second to refresh elapsed / remaining display.
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(1000)
        self._tick_timer.timeout.connect(self._update_time_display)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """
        Build the widget tree for the panel.

        Structure:
            QVBoxLayout (root)
            ├─ _inner_stack         — drop zone (0) or file info bar (1)
            ├─ _settings_section    — codec / CRF / preset
            ├─ _progress_section    — progress bar + time labels (hidden at rest)
            └─ _action_bar          — output folder, Compress button, status
        """
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 0)
        root.setSpacing(16)

        # ── Inner stack: drop zone ↔ file bar ─────────────────────────
        self._inner_stack = QStackedWidget(self)
        self._drop_zone   = self._make_drop_zone()
        self._file_bar    = self._make_file_bar()
        self._inner_stack.addWidget(self._drop_zone)   # index 0
        self._inner_stack.addWidget(self._file_bar)    # index 1
        self._inner_stack.setCurrentIndex(0)
        root.addWidget(self._inner_stack)

        # ── Settings section ──────────────────────────────────────────
        settings_widget = self._make_settings_section()
        root.addWidget(settings_widget)

        # ── Progress section (hidden until compression starts) ────────
        self._progress_section = self._make_progress_section()
        self._progress_section.hide()
        root.addWidget(self._progress_section)

        # Push the action bar to the bottom of the panel.
        root.addStretch()

        # ── Action bar ────────────────────────────────────────────────
        action_widget = self._make_action_bar()
        root.addWidget(action_widget)

    # ------------------------------------------------------------------
    # Sub-widget factories
    # ------------------------------------------------------------------

    def _make_drop_zone(self) -> QWidget:
        """
        Build the passive drop-zone placeholder widget.

        This widget is shown when no file has been loaded yet.  The panel
        itself handles drag-and-drop, so the drop zone is display-only.
        """
        zone = QWidget(self)
        zone.setObjectName("compressDropZone")
        zone.setMinimumHeight(160)
        zone.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(zone)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(8)

        icon_lbl = QLabel("📦", zone)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_lbl.setStyleSheet(
            "font-size: 40px; background: transparent; border: none;"
        )

        title_lbl = QLabel("Drop your video here", zone)
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_lbl.setStyleSheet(
            f"font-size: {AppTheme.FONT_SIZE_XL}px; "
            f"font-weight: 700; "
            f"color: {AppTheme.TEXT_PRIMARY}; "
            "background: transparent; border: none;"
        )

        sub_lbl = QLabel("or click Browse to pick a file", zone)
        sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub_lbl.setStyleSheet(
            f"font-size: {AppTheme.FONT_SIZE_BASE}px; "
            f"color: {AppTheme.TEXT_SECONDARY}; "
            "background: transparent; border: none;"
        )

        layout.addWidget(icon_lbl)
        layout.addWidget(title_lbl)
        layout.addWidget(sub_lbl)

        return zone

    def _make_file_bar(self) -> QWidget:
        """
        Build the file information bar widget.

        This widget replaces the drop zone once a video file is loaded.
        It shows the filename, file size, and a button to clear the file.
        """
        bar = QWidget(self)
        bar.setObjectName("compressFileBar")
        bar.setFixedHeight(52)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(10)

        self._file_icon_lbl = QLabel("🎬", bar)
        self._file_icon_lbl.setStyleSheet(
            "font-size: 18px; background: transparent; border: none;"
        )

        self._file_name_lbl = QLabel("—", bar)
        self._file_name_lbl.setStyleSheet(
            f"font-size: {AppTheme.FONT_SIZE_BASE}px; "
            f"font-weight: 600; color: {AppTheme.TEXT_PRIMARY}; "
            "background: transparent; border: none;"
        )

        self._file_size_lbl = QLabel("", bar)
        self._file_size_lbl.setStyleSheet(
            f"font-size: {AppTheme.FONT_SIZE_SM}px; "
            f"color: {AppTheme.TEXT_SECONDARY}; "
            "background: transparent; border: none;"
        )

        browse_btn = QPushButton("Browse…", bar)
        browse_btn.setObjectName("secondaryButton")
        browse_btn.clicked.connect(self._on_browse)

        clear_btn = QPushButton("✕", bar)
        clear_btn.setObjectName("secondaryButton")
        clear_btn.setFixedWidth(36)
        clear_btn.setToolTip("Remove file")
        clear_btn.clicked.connect(self._on_clear_file)

        layout.addWidget(self._file_icon_lbl)
        layout.addWidget(self._file_name_lbl)
        layout.addWidget(self._file_size_lbl)
        layout.addStretch()
        layout.addWidget(browse_btn)
        layout.addWidget(clear_btn)

        return bar

    def _make_settings_section(self) -> QWidget:
        """
        Build the codec / quality / preset settings panel.

        Returns a QWidget with:
          - Codec combo (H.264 / H.265)
          - CRF quality slider (15–35, default 23) with live value label
          - Preset combo (ultrafast → veryslow)

        Why CRF 15–35?  The full 0–51 range is valid but values outside
        this range are rarely useful for distribution:
          < 15 → near-lossless, huge files
          > 35 → visible quality loss for most content
        """
        section = QWidget(self)
        section.setObjectName("compressSettings")

        root_layout = QVBoxLayout(section)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(12)

        # ── Section heading ────────────────────────────────────────────
        heading = QLabel("COMPRESSION SETTINGS", section)
        heading.setObjectName("sectionLabel")

        root_layout.addWidget(heading)

        # ── Controls grid ──────────────────────────────────────────────
        controls = QWidget(section)
        controls.setObjectName("compressSettingsCard")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(16, 12, 16, 12)
        controls_layout.setSpacing(12)

        # — Codec row —
        codec_row = QHBoxLayout()
        codec_row.setSpacing(12)

        codec_lbl = QLabel("Codec:", controls)
        codec_lbl.setFixedWidth(100)
        codec_lbl.setStyleSheet(
            f"color: {AppTheme.TEXT_SECONDARY}; "
            f"font-size: {AppTheme.FONT_SIZE_BASE}px; "
            "background: transparent;"
        )

        self._codec_combo = QComboBox(controls)
        # Store the VideoCodec enum member as item data — never parse strings.
        for codec in VideoCodec:
            self._codec_combo.addItem(codec.label, userData=codec)

        codec_row.addWidget(codec_lbl)
        codec_row.addWidget(self._codec_combo)
        codec_row.addStretch()

        # — CRF quality row —
        # CRF 0 = lossless, CRF 51 = worst. Lower = better quality + larger file.
        # We expose 15–35, a practical range for most use cases.
        crf_row = QHBoxLayout()
        crf_row.setSpacing(12)

        crf_lbl = QLabel("Quality:", controls)
        crf_lbl.setFixedWidth(100)
        crf_lbl.setStyleSheet(
            f"color: {AppTheme.TEXT_SECONDARY}; "
            f"font-size: {AppTheme.FONT_SIZE_BASE}px; "
            "background: transparent;"
        )

        self._crf_slider = QSlider(Qt.Orientation.Horizontal, controls)
        self._crf_slider.setMinimum(15)
        self._crf_slider.setMaximum(35)
        self._crf_slider.setValue(23)
        self._crf_slider.setTickInterval(5)

        # Live CRF value label — updates as the slider moves.
        self._crf_value_lbl = QLabel("CRF 23", controls)
        self._crf_value_lbl.setFixedWidth(58)
        self._crf_value_lbl.setStyleSheet(
            f"color: {AppTheme.ACCENT}; "
            f"font-size: {AppTheme.FONT_SIZE_BASE}px; "
            "font-weight: 600; "
            "background: transparent;"
        )

        # Helper label explaining the scale direction.
        crf_hint = QLabel("← better quality / larger file", controls)
        crf_hint.setStyleSheet(
            f"color: {AppTheme.TEXT_DISABLED}; "
            f"font-size: {AppTheme.FONT_SIZE_SM}px; "
            "background: transparent;"
        )

        self._crf_slider.valueChanged.connect(self._on_crf_changed)

        crf_row.addWidget(crf_lbl)
        crf_row.addWidget(self._crf_slider, stretch=1)
        crf_row.addWidget(self._crf_value_lbl)
        crf_row.addWidget(crf_hint)

        # — Preset row —
        preset_row = QHBoxLayout()
        preset_row.setSpacing(12)

        preset_lbl = QLabel("Preset:", controls)
        preset_lbl.setFixedWidth(100)
        preset_lbl.setStyleSheet(
            f"color: {AppTheme.TEXT_SECONDARY}; "
            f"font-size: {AppTheme.FONT_SIZE_BASE}px; "
            "background: transparent;"
        )

        self._preset_combo = QComboBox(controls)
        for preset in CompressionPreset:
            self._preset_combo.addItem(preset.label, userData=preset)
        # Select "medium" (index 2) as the default.
        self._preset_combo.setCurrentIndex(2)

        preset_row.addWidget(preset_lbl)
        preset_row.addWidget(self._preset_combo)
        preset_row.addStretch()

        controls_layout.addLayout(codec_row)
        controls_layout.addLayout(crf_row)
        controls_layout.addLayout(preset_row)

        root_layout.addWidget(controls)

        return section

    def _make_progress_section(self) -> QWidget:
        """
        Build the progress bar + time display widget.

        This section is hidden while idle and revealed when compression
        starts.  It contains:
          - A QProgressBar that shows 0–100 % (or animated when indeterminate)
          - A percentage label on the right of the bar
          - A time label below showing elapsed time and estimated remaining

        The widget follows the same card style as the settings section.
        """
        section = QWidget(self)
        section.setObjectName("compressProgressSection")

        root_layout = QVBoxLayout(section)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(12)

        # ── Section heading ────────────────────────────────────────────
        heading = QLabel("PROGRESS", section)
        heading.setObjectName("sectionLabel")
        root_layout.addWidget(heading)

        # ── Card with bar + labels ─────────────────────────────────────
        card = QWidget(section)
        card.setObjectName("compressSettingsCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(10)

        # — Bar row: [progress bar ─────────────────] [  64%] —
        bar_row = QHBoxLayout()
        bar_row.setSpacing(12)

        self._progress_bar = QProgressBar(card)
        self._progress_bar.setObjectName("compressProgressBar")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)   # we show our own label
        self._progress_bar.setFixedHeight(10)

        # Percentage label — right-aligned, fixed width so the bar doesn't jump.
        self._pct_lbl = QLabel("0%", card)
        self._pct_lbl.setFixedWidth(42)
        self._pct_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._pct_lbl.setStyleSheet(
            f"color: {AppTheme.ACCENT}; "
            f"font-size: {AppTheme.FONT_SIZE_BASE}px; "
            "font-weight: 600; background: transparent;"
        )

        bar_row.addWidget(self._progress_bar, stretch=1)
        bar_row.addWidget(self._pct_lbl)

        # — Time row: "Elapsed: 0:05  ·  Remaining: ~1:30" —
        self._time_lbl = QLabel("Elapsed: 0:00  ·  Remaining: calculating…", card)
        self._time_lbl.setObjectName("timeLabel")
        self._time_lbl.setStyleSheet(
            f"color: {AppTheme.TEXT_SECONDARY}; "
            f"font-size: {AppTheme.FONT_SIZE_SM}px; "
            "background: transparent;"
        )

        card_layout.addLayout(bar_row)
        card_layout.addWidget(self._time_lbl)

        root_layout.addWidget(card)

        return section

    def _make_action_bar(self) -> QWidget:
        """
        Build the bottom action bar.

        Contains:
          - "Output: same folder" button  (choose output directory)
          - Stretch
          - Status label (shown after compression)
          - "Compress Video" primary button
        """
        bar = QWidget(self)
        bar.setObjectName("compressActionBar")
        bar.setFixedHeight(AppTheme.TOOLBAR_HEIGHT)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # — Output folder button —
        self._output_btn = QPushButton("Output: same folder", bar)
        self._output_btn.setObjectName("secondaryButton")
        self._output_btn.clicked.connect(self._on_choose_output_folder)

        # — Browse button (also accessible from action bar) —
        browse_btn = QPushButton("Browse…", bar)
        browse_btn.setObjectName("secondaryButton")
        browse_btn.clicked.connect(self._on_browse)

        # — Status label (hidden until first compression) —
        self._status_lbl = QLabel("", bar)
        self._status_lbl.setObjectName("compressStatus")
        self._status_lbl.hide()

        # — Compress button (primary CTA) —
        self._compress_btn = QPushButton("▶  Compress Video", bar)
        self._compress_btn.setObjectName("primaryButton")
        self._compress_btn.setEnabled(False)   # enabled only when a file is loaded
        self._compress_btn.clicked.connect(self._on_compress)

        layout.addWidget(browse_btn)
        layout.addWidget(self._output_btn)
        layout.addStretch()
        layout.addWidget(self._status_lbl)
        layout.addWidget(self._compress_btn)

        return bar

    # ------------------------------------------------------------------
    # Drag-and-drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """
        Accept the drag if it contains at least one supported video file.

        We check the extension of each dropped URL; if any passes
        is_supported_video() we highlight the drop zone and accept.
        """
        mime = event.mimeData()
        if mime.hasUrls():
            paths = [Path(u.toLocalFile()) for u in mime.urls() if u.isLocalFile()]
            valid, _ = filter_supported_videos(paths)
            if valid:
                event.acceptProposedAction()
                self._set_drop_highlight(True)
                return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        """Keep accepting the drag as the cursor moves within the panel."""
        mime = event.mimeData()
        if mime.hasUrls() and any(u.isLocalFile() for u in mime.urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        """Remove drop-zone highlight when the drag leaves the panel."""
        self._set_drop_highlight(False)

    def dropEvent(self, event: QDropEvent) -> None:
        """
        Load the first valid video file from the drop payload.

        Only the first valid video is used (single-file workflow).
        """
        self._set_drop_highlight(False)
        paths = [
            Path(u.toLocalFile())
            for u in event.mimeData().urls()
            if u.isLocalFile()
        ]
        valid, _ = filter_supported_videos(paths)
        if valid:
            event.acceptProposedAction()
            self._load_file(valid[0])
        else:
            event.ignore()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_file(self, path: Path) -> None:
        """
        Update the panel state to reflect a newly loaded video file.

        Args:
            path: Path to the video file to load.
        """
        self._input_path = path

        # Populate file bar labels.
        self._file_name_lbl.setText(path.name)
        size_str = human_readable_size(path)
        self._file_size_lbl.setText(size_str)

        # Switch from drop zone to file bar.
        self._inner_stack.setCurrentIndex(1)

        # Allow compression now that a file is loaded.
        self._compress_btn.setEnabled(True)

        # Reset any previous status / progress display.
        self._status_lbl.hide()
        self._status_lbl.setText("")
        self._progress_section.hide()
        self._progress_bar.setValue(0)
        self._pct_lbl.setText("0%")

    def _set_drop_highlight(self, active: bool) -> None:
        """
        Toggle the visual drop-zone highlight on or off.

        We swap the objectName so the stylesheet can apply a different
        border/background for the "active drag" state.

        Args:
            active: True while a drag hovers over the panel, False otherwise.
        """
        # When the drop zone is visible (no file loaded), highlight it.
        # When a file is loaded (file bar visible), highlight the whole panel.
        if self._inner_stack.currentIndex() == 0:
            obj_name = "compressDropZoneActive" if active else "compressDropZone"
            self._drop_zone.setObjectName(obj_name)
            self._drop_zone.style().unpolish(self._drop_zone)
            self._drop_zone.style().polish(self._drop_zone)
        else:
            obj_name = "compressorPanelActive" if active else "compressorPanel"
            self.setObjectName(obj_name)
            self.style().unpolish(self)
            self.style().polish(self)

    # ------------------------------------------------------------------
    # Slots — file input
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_browse(self) -> None:
        """Open the native file browser and load the selected video."""
        selected, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video File",
            str(Path.home()),
            VIDEO_FILE_FILTER,
        )
        if not selected:
            return
        paths = [Path(p) for p in selected]
        valid, _ = filter_supported_videos(paths)
        if valid:
            self._load_file(valid[0])

    @pyqtSlot()
    def _on_clear_file(self) -> None:
        """Remove the loaded file and return to the drop zone."""
        self._input_path = None
        self._compress_btn.setEnabled(False)
        self._status_lbl.hide()
        self._progress_section.hide()
        self._inner_stack.setCurrentIndex(0)

    @pyqtSlot()
    def _on_choose_output_folder(self) -> None:
        """Open a directory picker and update the output folder button label."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose Output Folder",
            str(Path.home()),
        )
        if not folder:
            return
        self._output_dir = Path(folder)
        short_name = self._output_dir.name or str(self._output_dir)
        self._output_btn.setText(f"Output: {short_name}")

    # ------------------------------------------------------------------
    # Slots — settings
    # ------------------------------------------------------------------

    @pyqtSlot(int)
    def _on_crf_changed(self, value: int) -> None:
        """
        Update the live CRF value label as the slider moves.

        The label shows the numeric CRF and a qualitative hint so users
        understand the tradeoff without needing to memorise the scale.

        Args:
            value: Current slider position (15–35).
        """
        self._crf_value_lbl.setText(f"CRF {value}")

    # ------------------------------------------------------------------
    # Slots — compression
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_compress(self) -> None:
        """
        Start compression in a background thread.

        Reads the current settings from the UI, builds a CompressionJob,
        creates a _CompressWorker, and dispatches it to QThreadPool.
        The UI is put into a "busy" state (progress section shown, button
        disabled) until the worker reports back.
        """
        if self._input_path is None:
            return

        if self._compressor is None:
            self._show_status(success=False, message=self._ffmpeg_error_msg)
            return

        # Read selected codec, CRF, and preset from the UI controls.
        codec:  VideoCodec        = self._codec_combo.currentData()
        crf:    int               = self._crf_slider.value()
        preset: CompressionPreset = self._preset_combo.currentData()

        # Build the output path with "_compressed" suffix.
        output_path = build_video_output_path(
            input_path=self._input_path,
            output_dir=self._output_dir,
        )

        job = CompressionJob(
            input_path=self._input_path,
            output_path=output_path,
            codec=codec,
            crf=crf,
            preset=preset,
        )

        # Reset progress state.
        self._current_progress = 0.0
        self._compress_start_time = time.monotonic()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._pct_lbl.setText("0%")
        self._time_lbl.setText("Elapsed: 0:00  ·  Remaining: calculating…")

        # Show progress section; hide any previous status result.
        self._progress_section.show()
        self._status_lbl.hide()

        # Disable the Compress button while running.
        self._compress_btn.setText("Compressing…")
        self._compress_btn.setEnabled(False)

        # Start the 1-second tick timer to update the elapsed/remaining labels.
        self._tick_timer.start()

        worker = _CompressWorker(job, self._compressor, self._signals)
        QThreadPool.globalInstance().start(worker)

    @pyqtSlot(float)
    def _on_progress_updated(self, pct: float) -> None:
        """
        Receive a progress value from the background worker and update the UI.

        When pct < 0, the video duration was unavailable, so we switch the
        progress bar to indeterminate (animated) mode.  Otherwise we update
        the bar value and percentage label, then refresh the time display.

        Args:
            pct: 0.0–99.0 for a known percentage, or -1.0 for indeterminate.
        """
        if pct < 0:
            # Indeterminate mode: setRange(0, 0) makes Qt show an animation.
            self._progress_bar.setRange(0, 0)
            self._pct_lbl.setText("—")
            self._current_progress = 0.0
        else:
            # Ensure bar is in determinate mode (may have been indeterminate).
            if self._progress_bar.maximum() == 0:
                self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(int(pct))
            self._pct_lbl.setText(f"{int(pct)}%")
            self._current_progress = pct
            self._update_time_display()

    @pyqtSlot(CompressionResult)
    def _on_compress_finished(self, result: CompressionResult) -> None:
        """
        Handle the result emitted by _CompressWorker.

        Stops the tick timer, snaps the progress bar to 100 %, then briefly
        shows the result before hiding the progress section.

        Args:
            result: CompressionResult from the worker.
        """
        # Stop the elapsed-time ticker.
        self._tick_timer.stop()

        # Snap the bar to 100 % (or hide indeterminate animation) on success.
        if result.success:
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(100)
            self._pct_lbl.setText("100%")

        # A short delay lets the user see "100%" before the bar disappears.
        # We use a single-shot QTimer so we don't need to track a flag.
        QTimer.singleShot(800, self._progress_section.hide)

        # Restore the Compress button.
        self._compress_btn.setText("▶  Compress Video")
        self._compress_btn.setEnabled(self._input_path is not None)

        if result.success:
            before = _format_size(result.original_size)
            after  = _format_size(result.compressed_size)

            if result.original_size > 0:
                saved_pct = 100 * (1 - result.compressed_size / result.original_size)
                summary = f"✅  {before} → {after}  (saved {saved_pct:.0f}%)"
            else:
                summary = f"✅  Compressed → {after}"

            self._show_status(success=True, message=summary)
        else:
            self._progress_section.hide()
            self._show_status(success=False, message=f"❌  {result.error_message}")

    @pyqtSlot()
    def _update_time_display(self) -> None:
        """
        Refresh the elapsed / estimated-remaining time labels.

        Called every second by _tick_timer AND after each progress signal
        so the display stays current between timer ticks.

        Remaining time is estimated by extrapolating from elapsed time and
        current progress percentage.  We guard against division-by-zero and
        negative estimates from timing jitter.
        """
        elapsed_s = time.monotonic() - self._compress_start_time
        elapsed_str = _format_duration(elapsed_s)

        # We need at least 2 % progress to give a meaningful estimate
        # (very early estimates would be wildly inaccurate).
        if self._current_progress >= 2.0:
            total_estimated = elapsed_s / (self._current_progress / 100.0)
            remaining_s = max(0.0, total_estimated - elapsed_s)
            remaining_str = f"~{_format_duration(remaining_s)}"
        else:
            remaining_str = "calculating…"

        self._time_lbl.setText(
            f"Elapsed: {elapsed_str}  ·  Remaining: {remaining_str}"
        )

    # ------------------------------------------------------------------
    # Private display helpers
    # ------------------------------------------------------------------

    def _show_status(self, *, success: bool, message: str) -> None:
        """
        Update and reveal the status label.

        The label uses objectName "compressStatusOk" (green) on success or
        "compressStatusErr" (red) on failure so the stylesheet can colour it.

        Args:
            success: True for a success state, False for an error state.
            message: Human-readable message to display.
        """
        self._status_lbl.setText(message)
        if success:
            self._status_lbl.setObjectName("compressStatusOk")
        else:
            self._status_lbl.setObjectName("compressStatusErr")
        self._status_lbl.style().unpolish(self._status_lbl)
        self._status_lbl.style().polish(self._status_lbl)
        self._status_lbl.show()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _format_size(size_bytes: int) -> str:
    """
    Return a human-readable file size string.

    Used to format before/after sizes in the compression result summary.

    Args:
        size_bytes: File size in bytes.

    Returns:
        A string like "45.2 MB" or "820 KB".
    """
    if size_bytes <= 0:
        return "—"
    for unit, threshold in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if size_bytes >= threshold:
            return f"{size_bytes / threshold:.1f} {unit}"
    return f"{size_bytes} B"


def _format_duration(seconds: float) -> str:
    """
    Format a duration in seconds as a M:SS string.

    Used for the elapsed and remaining time labels in the progress section.

    Args:
        seconds: Duration in seconds (may be fractional).

    Returns:
        A string like "1:05" or "0:42".

    Example:
        >>> _format_duration(65.3)
        '1:05'
        >>> _format_duration(7.0)
        '0:07'
    """
    if seconds < 0:
        return "—"
    total = int(seconds)
    mins = total // 60
    secs = total % 60
    return f"{mins}:{secs:02d}"

