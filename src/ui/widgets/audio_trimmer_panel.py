"""
Audio Trimmer Panel.

This widget is the entire content area shown when the "Audio" tab is active.
It lets the user:
  1. Load an audio file (drag-and-drop anywhere on the panel, or Browse button)
  2. See the audio waveform and drag to select a trim region
  3. Fine-tune start/end times with spinboxes
  4. Choose output format and output folder
  5. Click "Trim" to export the selected region

Layout (top → bottom):
  ┌───────────────────────────────────────────────┐
  │  File info bar  (name + Browse button)        │
  ├───────────────────────────────────────────────┤
  │  WaveformWidget  (interactive, fills space)   │
  ├───────────────────────────────────────────────┤
  │  Time controls: [Start] [End] [Duration]      │
  ├───────────────────────────────────────────────┤
  │  Trim toolbar: [Format▼] [Output dir] [Trim]  │
  └───────────────────────────────────────────────┘

Drag-and-drop is handled at the panel level: the panel overrides the Qt
drag events so that any audio file dropped anywhere on it is loaded.

Threading:
  Audio loading (ffmpeg decode) is done on a QRunnable worker so the UI
  stays responsive.  The worker emits a signal with the peaks array back
  to the main thread which then calls WaveformWidget.load_peaks().

  Trimming is also done on a background QRunnable; the result signal
  drives a status banner update in the panel.

Usage:
    panel = AudioTrimmerPanel(parent=self)
    # No configuration needed — the panel is self-contained.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import (
    QObject,
    QRunnable,
    QThreadPool,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.core.audio_loader import AUDIO_FILE_FILTER, AudioLoader, is_supported_audio
from src.core.models import AudioFileInfo, AudioFormat, SilenceRemovalJob, TrimJob
from src.core.silence_remover import SilenceRemover
from src.core.trimmer import AudioTrimmer
from src.ui.theme import AppTheme
from src.ui.widgets.waveform_widget import WaveformWidget
from src.utils.file_utils import build_output_path


# ---------------------------------------------------------------------------
# Background worker signals
# ---------------------------------------------------------------------------

class _LoadSignals(QObject):
    """
    Signals for the background audio-load worker.

    Qt signals must live on a QObject, so this helper is owned by the
    worker and passed back to the panel.
    """
    # Emitted on success: (peaks array, AudioFileInfo)
    finished = pyqtSignal(object, object)
    # Emitted on failure: error message string
    failed   = pyqtSignal(str)


class _TrimSignals(QObject):
    """Signals for the background trim worker."""
    finished = pyqtSignal(Path)   # output path on success
    failed   = pyqtSignal(str)    # error message on failure


class _SilenceSignals(QObject):
    """
    Signals for the background silence-removal worker.

    ``finished`` carries both the output path and the number of silence
    segments that were detected and removed so the status bar can show
    a meaningful summary to the user.
    """
    # (output_path, segments_removed)
    finished = pyqtSignal(Path, int)
    failed   = pyqtSignal(str)


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class _LoadWorker(QRunnable):
    """
    Background worker: decode an audio file into waveform peak data.

    Uses AudioLoader to probe metadata and decode PCM at a low rate,
    then emits the resulting peaks + info via _LoadSignals.

    The worker is submitted to QThreadPool.globalInstance() by
    AudioTrimmerPanel._start_load().

    Example (internal):
        worker = _LoadWorker(path, signals)
        QThreadPool.globalInstance().start(worker)
    """

    def __init__(self, path: Path, signals: _LoadSignals) -> None:
        super().__init__()
        self._path    = path
        self._signals = signals

    @pyqtSlot()
    def run(self) -> None:
        """Run in a thread pool thread — never call directly."""
        try:
            loader = AudioLoader()
            info   = loader.get_info(self._path)
            peaks  = loader.load_peaks(self._path, num_bins=1200)
            self._signals.finished.emit(peaks, info)
        except Exception as exc:  # noqa: BLE001
            self._signals.failed.emit(str(exc))


class _TrimWorker(QRunnable):
    """
    Background worker: run AudioTrimmer.trim() for a TrimJob.

    Emits _TrimSignals.finished(output_path) on success or
    _TrimSignals.failed(message) on error.

    Example (internal):
        worker = _TrimWorker(job, signals)
        QThreadPool.globalInstance().start(worker)
    """

    def __init__(self, job: TrimJob, signals: _TrimSignals) -> None:
        super().__init__()
        self._job     = job
        self._signals = signals

    @pyqtSlot()
    def run(self) -> None:
        """Run in a thread pool thread — never call directly."""
        try:
            trimmer = AudioTrimmer()
            result  = trimmer.trim(self._job)
            if result.success:
                self._signals.finished.emit(result.output_path)
            else:
                self._signals.failed.emit(result.error_message)
        except Exception as exc:  # noqa: BLE001
            self._signals.failed.emit(str(exc))


class _SilenceWorker(QRunnable):
    """
    Background worker: run SilenceRemover.remove_silence() for a job.

    Emits ``_SilenceSignals.finished(output_path, segments_removed)`` on
    success, or ``_SilenceSignals.failed(message)`` on any error.

    Always submit to QThreadPool.globalInstance() — never call run() directly.

    Example (internal):
        worker = _SilenceWorker(job, signals)
        QThreadPool.globalInstance().start(worker)
    """

    def __init__(self, job: SilenceRemovalJob, signals: _SilenceSignals) -> None:
        super().__init__()
        self._job     = job
        self._signals = signals

    @pyqtSlot()
    def run(self) -> None:
        """Run in a thread pool thread — never call directly."""
        try:
            remover = SilenceRemover()
            result  = remover.remove_silence(self._job)
            if result.success:
                self._signals.finished.emit(
                    result.output_path, result.segments_removed
                )
            else:
                self._signals.failed.emit(result.error_message)
        except Exception as exc:  # noqa: BLE001
            self._signals.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# AudioTrimmerPanel
# ---------------------------------------------------------------------------

class AudioTrimmerPanel(QWidget):
    """
    Self-contained audio trim panel shown under the "Audio" tab.

    Responsibilities:
      - Accept drag-and-drop of audio files onto the panel
      - Load audio metadata + waveform peaks in the background
      - Display the waveform via WaveformWidget
      - Keep spinbox values in sync with the waveform selection
      - Run AudioTrimmer on a background thread when "Trim" is clicked
      - Show a status label with success / error feedback
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("trimmerPanel")
        self.setAcceptDrops(True)

        # Currently loaded file information.
        self._info: AudioFileInfo | None = None

        # Output directory (None = same folder as input).
        self._output_dir: Path | None = None

        # Whether a load or trim operation is in progress.
        self._busy: bool = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """
        Assemble all child widgets into the panel layout.

        Structure:
          VBoxLayout
            ├─ File info bar  (HBox: icon + filename label + browse btn)
            ├─ WaveformWidget (stretch=1)
            ├─ Time controls  (HBox: start spin, end spin, duration label)
            ├─ Trim toolbar   (HBox: format combo, output btn, trim btn)
            └─ Status label
        """
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 0)
        root.setSpacing(10)

        # ── File info bar ─────────────────────────────────────────────
        file_bar = QHBoxLayout()
        file_bar.setSpacing(10)

        self._file_icon = QLabel("🎵", self)
        self._file_icon.setStyleSheet("font-size: 22px; background: transparent;")

        self._filename_label = QLabel("No file loaded  —  drop an audio file here", self)
        self._filename_label.setStyleSheet(
            f"color: {AppTheme.TEXT_SECONDARY}; "
            f"font-size: {AppTheme.FONT_SIZE_BASE}px; "
            "background: transparent;"
        )
        self._filename_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        browse_btn = QPushButton("Browse Audio…", self)
        browse_btn.setObjectName("secondaryButton")
        browse_btn.clicked.connect(self._on_browse)

        file_bar.addWidget(self._file_icon)
        file_bar.addWidget(self._filename_label)
        file_bar.addWidget(browse_btn)
        root.addLayout(file_bar)

        # ── Waveform ──────────────────────────────────────────────────
        self._waveform = WaveformWidget(self)
        self._waveform.region_changed.connect(self._on_region_changed)
        root.addWidget(self._waveform, stretch=1)

        # ── Loading indicator (hidden by default) ─────────────────────
        self._loading_label = QLabel("Loading waveform…", self)
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_label.setStyleSheet(
            f"color: {AppTheme.TEXT_SECONDARY}; "
            f"font-size: {AppTheme.FONT_SIZE_BASE}px; "
            "background: transparent;"
        )
        self._loading_label.hide()
        root.addWidget(self._loading_label)

        # ── Time controls ─────────────────────────────────────────────
        time_bar = QHBoxLayout()
        time_bar.setSpacing(8)

        # Helper: create a labeled spinbox group.
        def _labeled_spin(label_text: str) -> tuple[QLabel, QDoubleSpinBox]:
            lbl  = QLabel(label_text, self)
            lbl.setObjectName("timeLabel")
            spin = QDoubleSpinBox(self)
            spin.setRange(0.0, 999999.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            spin.setSuffix(" s")
            return lbl, spin

        start_lbl, self._start_spin = _labeled_spin("Start:")
        end_lbl,   self._end_spin   = _labeled_spin("End:")

        self._dur_label = QLabel("Duration: —", self)
        self._dur_label.setObjectName("durationLabel")
        self._dur_label.setStyleSheet(
            f"color: {AppTheme.ACCENT}; "
            f"font-size: {AppTheme.FONT_SIZE_SM}px; "
            "font-weight: 600; background: transparent;"
        )

        self._start_spin.valueChanged.connect(self._on_start_spin_changed)
        self._end_spin.valueChanged.connect(self._on_end_spin_changed)

        time_bar.addWidget(start_lbl)
        time_bar.addWidget(self._start_spin)
        time_bar.addSpacing(16)
        time_bar.addWidget(end_lbl)
        time_bar.addWidget(self._end_spin)
        time_bar.addSpacing(16)
        time_bar.addWidget(self._dur_label)
        time_bar.addStretch()
        root.addLayout(time_bar)

        # ── Trim toolbar ──────────────────────────────────────────────
        trim_bar_widget = QWidget(self)
        trim_bar_widget.setObjectName("trimToolbar")
        trim_bar_widget.setFixedHeight(AppTheme.TOOLBAR_HEIGHT)

        trim_bar = QHBoxLayout(trim_bar_widget)
        trim_bar.setContentsMargins(0, 0, 0, 0)
        trim_bar.setSpacing(10)

        # Format selector.
        fmt_label = QLabel("Format:", self)
        fmt_label.setStyleSheet(
            f"color: {AppTheme.TEXT_SECONDARY}; "
            f"font-size: {AppTheme.FONT_SIZE_BASE}px; background: transparent;"
        )
        self._format_combo = QComboBox(self)
        for fmt in AudioFormat:
            self._format_combo.addItem(fmt.label, userData=fmt)

        # Output folder selector.
        self._output_btn = QPushButton("Output: same folder", self)
        self._output_btn.setObjectName("secondaryButton")
        self._output_btn.clicked.connect(self._on_choose_output)

        trim_bar.addWidget(fmt_label)
        trim_bar.addWidget(self._format_combo)
        trim_bar.addWidget(self._output_btn)
        trim_bar.addStretch()

        # Trim button (primary CTA).
        self._trim_btn = QPushButton("✂  Trim", self)
        self._trim_btn.setObjectName("primaryButton")
        self._trim_btn.setEnabled(False)
        self._trim_btn.clicked.connect(self._on_trim_clicked)
        trim_bar.addWidget(self._trim_btn)

        root.addWidget(trim_bar_widget)

        # ── Silence removal toolbar ───────────────────────────────────
        # A second toolbar-style row that groups all silence-removal
        # controls together. Sharing the same Format / Output controls
        # from the trim toolbar above keeps the UI compact.
        #
        # Layout (left → right):
        #   "Threshold:" [-30.0 dB spin]  "Min Silence:" [0.50 s spin]
        #   "Padding:" [0.10 s spin]  <stretch>  [🔇 Remove Silence]
        silence_bar_widget = QWidget(self)
        silence_bar_widget.setObjectName("silenceToolbar")
        silence_bar_widget.setFixedHeight(AppTheme.TOOLBAR_HEIGHT)

        silence_bar = QHBoxLayout(silence_bar_widget)
        silence_bar.setContentsMargins(12, 0, 0, 0)
        silence_bar.setSpacing(8)

        # Helper: create a small right-aligned label for the silence row.
        def _silence_label(text: str) -> QLabel:
            lbl = QLabel(text, self)
            lbl.setObjectName("timeLabel")
            return lbl

        # Threshold spinbox — loudness floor below which audio is silence.
        # Range: -80 dB (very quiet) to -10 dB (aggressive).
        # Default -30 dB is a good balance for speech recordings.
        thresh_lbl = _silence_label("Threshold:")
        self._threshold_spin = QDoubleSpinBox(self)
        self._threshold_spin.setRange(-80.0, -10.0)
        self._threshold_spin.setValue(-30.0)
        self._threshold_spin.setDecimals(1)
        self._threshold_spin.setSingleStep(1.0)
        self._threshold_spin.setSuffix(" dB")
        self._threshold_spin.setToolTip(
            "Audio quieter than this level is treated as silence.\n"
            "Lower values (e.g. -40 dB) are more aggressive."
        )

        # Min silence duration spinbox — gaps shorter than this are kept.
        # Prevents removing short natural pauses between words.
        min_lbl = _silence_label("Min Silence:")
        self._min_silence_spin = QDoubleSpinBox(self)
        self._min_silence_spin.setRange(0.05, 10.0)
        self._min_silence_spin.setValue(0.5)
        self._min_silence_spin.setDecimals(2)
        self._min_silence_spin.setSingleStep(0.1)
        self._min_silence_spin.setSuffix(" s")
        self._min_silence_spin.setToolTip(
            "Only gaps longer than this are removed.\n"
            "Increase to keep short natural pauses between words."
        )

        # Padding spinbox — buffer preserved at each edge of a silence boundary.
        # Prevents clipping the first/last syllable of speech at a cut point.
        pad_lbl = _silence_label("Padding:")
        self._padding_spin = QDoubleSpinBox(self)
        self._padding_spin.setRange(0.0, 2.0)
        self._padding_spin.setValue(0.1)
        self._padding_spin.setDecimals(2)
        self._padding_spin.setSingleStep(0.05)
        self._padding_spin.setSuffix(" s")
        self._padding_spin.setToolTip(
            "Extra audio kept on each side of a silence boundary.\n"
            "Prevents cutting off the beginning or end of speech."
        )

        silence_bar.addWidget(thresh_lbl)
        silence_bar.addWidget(self._threshold_spin)
        silence_bar.addSpacing(12)
        silence_bar.addWidget(min_lbl)
        silence_bar.addWidget(self._min_silence_spin)
        silence_bar.addSpacing(12)
        silence_bar.addWidget(pad_lbl)
        silence_bar.addWidget(self._padding_spin)
        silence_bar.addStretch()

        # Primary CTA for silence removal — green button to distinguish it
        # visually from the purple "Trim" button above.
        self._silence_btn = QPushButton("🔇  Remove Silence", self)
        self._silence_btn.setObjectName("silenceButton")
        self._silence_btn.setEnabled(False)
        self._silence_btn.clicked.connect(self._on_remove_silence_clicked)
        silence_bar.addWidget(self._silence_btn)

        root.addWidget(silence_bar_widget)

        # ── Status bar ────────────────────────────────────────────────
        self._status_label = QLabel("", self)
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet(
            f"font-size: {AppTheme.FONT_SIZE_SM}px; "
            "background: transparent; padding: 4px 0;"
        )
        root.addWidget(self._status_label)

    # ------------------------------------------------------------------
    # Drag-and-drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept drag if it contains at least one supported audio file URL."""
        mime = event.mimeData()
        if mime.hasUrls():
            paths = [Path(u.toLocalFile()) for u in mime.urls() if u.isLocalFile()]
            if any(is_supported_audio(p) for p in paths):
                event.acceptProposedAction()
                self._set_drop_highlight(True)
                return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        """Keep accepting the drag as the cursor moves."""
        mime = event.mimeData()
        if mime.hasUrls() and any(
            is_supported_audio(Path(u.toLocalFile()))
            for u in mime.urls()
            if u.isLocalFile()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        """Remove drop highlight when the drag leaves the panel."""
        self._set_drop_highlight(False)

    def dropEvent(self, event: QDropEvent) -> None:
        """Load the first supported audio file in the drop payload."""
        self._set_drop_highlight(False)
        paths = [
            Path(u.toLocalFile())
            for u in event.mimeData().urls()
            if u.isLocalFile()
        ]
        audio = [p for p in paths if is_supported_audio(p)]
        if audio:
            event.acceptProposedAction()
            self._load_file(audio[0])
        else:
            event.ignore()

    # ------------------------------------------------------------------
    # Slots — file input
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_browse(self) -> None:
        """Open a native file browser filtered to audio files."""
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            str(Path.home()),
            AUDIO_FILE_FILTER,
        )
        if selected:
            self._load_file(Path(selected))

    # ------------------------------------------------------------------
    # File loading (background thread)
    # ------------------------------------------------------------------

    def _load_file(self, path: Path) -> None:
        """
        Kick off a background load for `path`.

        Shows the loading indicator and dispatches a _LoadWorker to
        QThreadPool.  Results come back via _on_load_finished / _on_load_failed.

        Args:
            path: Validated audio file path.
        """
        if self._busy:
            return

        self._busy = True
        self._trim_btn.setEnabled(False)
        self._waveform.clear()
        self._filename_label.setText(f"Loading  {path.name} …")
        self._loading_label.show()
        self._status_label.setText("")

        signals = _LoadSignals()
        signals.finished.connect(self._on_load_finished)
        signals.failed.connect(self._on_load_failed)

        worker = _LoadWorker(path, signals)
        QThreadPool.globalInstance().start(worker)

    @pyqtSlot(object, object)
    def _on_load_finished(
        self, peaks: np.ndarray, info: AudioFileInfo
    ) -> None:
        """
        Receive waveform data from the background worker and update the UI.

        Called on the main thread via Qt's auto-connection.

        Args:
            peaks: 1-D float32 peak array ready for WaveformWidget.
            info:  AudioFileInfo with duration, sample rate, etc.
        """
        self._busy = False
        self._info = info
        self._loading_label.hide()
        self._filename_label.setText(
            f"{info.path.name}   ·   {info.duration_str}   ·   "
            f"{info.sample_rate // 1000} kHz   ·   "
            f"{'Stereo' if info.channels == 2 else 'Mono'}"
        )
        self._filename_label.setStyleSheet(
            f"color: {AppTheme.TEXT_PRIMARY}; "
            f"font-size: {AppTheme.FONT_SIZE_BASE}px; "
            "background: transparent;"
        )

        # Push data into the waveform widget (always done on main thread).
        self._waveform.load_peaks(peaks, info.duration)

        # Sync spinboxes to full file range.
        self._sync_spinboxes_from_waveform()
        # Enable both the trim and silence-removal buttons now that a file is ready.
        self._trim_btn.setEnabled(True)
        self._silence_btn.setEnabled(True)

    @pyqtSlot(str)
    def _on_load_failed(self, message: str) -> None:
        """
        Show an error message when audio loading fails.

        Args:
            message: Human-readable description of the failure.
        """
        self._busy = False
        self._loading_label.hide()
        self._filename_label.setText("Failed to load file")
        self._filename_label.setStyleSheet(
            f"color: {AppTheme.STATUS_FAILED}; "
            f"font-size: {AppTheme.FONT_SIZE_BASE}px; "
            "background: transparent;"
        )
        # Keep silence button disabled when load failed — no valid file is open.
        self._silence_btn.setEnabled(False)
        self._set_status(f"Error: {message}", error=True)

    # ------------------------------------------------------------------
    # Slots — region / spinbox sync
    # ------------------------------------------------------------------

    @pyqtSlot(float, float)
    def _on_region_changed(self, start: float, end: float) -> None:
        """
        Propagate waveform selection changes to the spinboxes.

        The spinboxes' valueChanged signals are temporarily blocked to
        prevent a feedback loop (spinbox → waveform → spinbox → …).

        Args:
            start: Selection start in seconds.
            end:   Selection end in seconds.
        """
        self._start_spin.blockSignals(True)
        self._end_spin.blockSignals(True)

        self._start_spin.setValue(start)
        self._end_spin.setValue(end)

        self._start_spin.blockSignals(False)
        self._end_spin.blockSignals(False)

        self._update_duration_label(start, end)

    def _on_start_spin_changed(self, value: float) -> None:
        """
        Push spinbox start change into the waveform widget.

        Clamps value to [0, end - minimum] to keep the selection valid.

        Args:
            value: New start value in seconds from the spinbox.
        """
        if self._info is None:
            return
        end = self._end_spin.value()
        # Ensure start does not exceed end.
        value = min(value, end - 0.001)
        self._waveform.set_selection(value, end)
        self._update_duration_label(value, end)

    def _on_end_spin_changed(self, value: float) -> None:
        """
        Push spinbox end change into the waveform widget.

        Args:
            value: New end value in seconds from the spinbox.
        """
        if self._info is None:
            return
        start = self._start_spin.value()
        # Ensure end does not go below start.
        value = max(value, start + 0.001)
        self._waveform.set_selection(start, value)
        self._update_duration_label(start, value)

    # ------------------------------------------------------------------
    # Slots — trim
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_trim_clicked(self) -> None:
        """
        Validate the selection and dispatch a background trim worker.

        Guards:
          - File must be loaded
          - Selection must be at least 0.1 s long
          - Not already busy
        """
        if self._info is None or self._busy:
            return

        start, end = self._waveform.selection
        if end - start < 0.1:
            QMessageBox.warning(
                self,
                "Selection too short",
                "Please select at least 0.1 seconds to trim.",
            )
            return

        output_format: AudioFormat = self._format_combo.currentData()
        output_path = build_output_path(
            input_path=self._info.path,
            output_format=output_format,
            output_dir=self._output_dir,
            suffix="_trimmed",
        )

        job = TrimJob(
            input_path=self._info.path,
            output_path=output_path,
            output_format=output_format,
            start_time=start,
            end_time=end,
        )

        self._busy = True
        self._trim_btn.setEnabled(False)
        self._silence_btn.setEnabled(False)
        self._trim_btn.setText("Trimming…")
        self._set_status("Trimming…")

        signals = _TrimSignals()
        signals.finished.connect(self._on_trim_finished)
        signals.failed.connect(self._on_trim_failed)

        worker = _TrimWorker(job, signals)
        QThreadPool.globalInstance().start(worker)

    @pyqtSlot(Path)
    def _on_trim_finished(self, output_path: Path) -> None:
        """
        Handle a successful trim — show a success status and re-enable buttons.

        Args:
            output_path: Path of the newly created trimmed file.
        """
        self._busy = False
        self._trim_btn.setEnabled(True)
        self._silence_btn.setEnabled(True)
        self._trim_btn.setText("✂  Trim")
        self._set_status(f"Saved: {output_path.name}", error=False)

    @pyqtSlot(str)
    def _on_trim_failed(self, message: str) -> None:
        """
        Handle a failed trim — show the error message and re-enable buttons.

        Args:
            message: Error description from AudioTrimmer.
        """
        self._busy = False
        self._trim_btn.setEnabled(True)
        self._silence_btn.setEnabled(True)
        self._trim_btn.setText("✂  Trim")
        self._set_status(f"Error: {message}", error=True)

    # ------------------------------------------------------------------
    # Slots — silence removal
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_remove_silence_clicked(self) -> None:
        """
        Read the silence parameters from the spinboxes, build a
        SilenceRemovalJob, and dispatch a _SilenceWorker to the thread pool.

        Guards (same pattern as _on_trim_clicked):
          - A file must be loaded (_info is not None).
          - No other operation must be in progress (_busy flag).

        The output file is written next to the source with the suffix
        ``_no_silence``, using the same format and output directory
        selected in the trim toolbar above.
        """
        if self._info is None or self._busy:
            return

        output_format: AudioFormat = self._format_combo.currentData()
        output_path = build_output_path(
            input_path=self._info.path,
            output_format=output_format,
            output_dir=self._output_dir,
            suffix="_no_silence",
        )

        job = SilenceRemovalJob(
            input_path=self._info.path,
            output_path=output_path,
            output_format=output_format,
            threshold_db=self._threshold_spin.value(),
            min_silence_duration=self._min_silence_spin.value(),
            padding=self._padding_spin.value(),
        )

        self._busy = True
        self._trim_btn.setEnabled(False)
        self._silence_btn.setEnabled(False)
        self._silence_btn.setText("Removing silence…")
        self._set_status("Scanning for silence…")

        signals = _SilenceSignals()
        signals.finished.connect(self._on_silence_finished)
        signals.failed.connect(self._on_silence_failed)

        worker = _SilenceWorker(job, signals)
        QThreadPool.globalInstance().start(worker)

    @pyqtSlot(Path, int)
    def _on_silence_finished(self, output_path: Path, segments_removed: int) -> None:
        """
        Handle a completed silence-removal operation.

        Shows how many silence segments were removed in the status bar so
        the user gets meaningful feedback even when the result sounds
        identical (e.g. 0 silences found in a dense music track).

        Args:
            output_path:     Path of the newly written file.
            segments_removed: Number of distinct silence intervals removed.
        """
        self._busy = False
        self._trim_btn.setEnabled(True)
        self._silence_btn.setEnabled(True)
        self._silence_btn.setText("🔇  Remove Silence")

        # Decide on a human-friendly summary depending on the result count.
        if segments_removed == 0:
            summary = f"Saved: {output_path.name}  (no silence detected)"
        elif segments_removed == 1:
            summary = f"Saved: {output_path.name}  (1 silence removed)"
        else:
            summary = f"Saved: {output_path.name}  ({segments_removed} silences removed)"

        self._set_status(summary, error=False)

    @pyqtSlot(str)
    def _on_silence_failed(self, message: str) -> None:
        """
        Handle a failed silence-removal operation — show error and re-enable.

        Args:
            message: Error description from SilenceRemover.
        """
        self._busy = False
        self._trim_btn.setEnabled(True)
        self._silence_btn.setEnabled(True)
        self._silence_btn.setText("🔇  Remove Silence")
        self._set_status(f"Error: {message}", error=True)

    # ------------------------------------------------------------------
    # Slots — output folder
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_choose_output(self) -> None:
        """Open a native directory picker to select the output folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Choose Output Folder", str(Path.home())
        )
        if folder:
            self._output_dir = Path(folder)
            short = self._output_dir.name or str(self._output_dir)
            self._output_btn.setText(f"Output: {short}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sync_spinboxes_from_waveform(self) -> None:
        """
        Initialise the spinboxes to match the waveform's current selection.

        Called once after a successful load to set both spinboxes to
        [0, duration] and configure their ranges.
        """
        if self._info is None:
            return
        dur = self._info.duration
        start, end = self._waveform.selection

        # Update allowed range first so setValue doesn't clamp prematurely.
        self._start_spin.setMaximum(dur)
        self._end_spin.setMaximum(dur)

        self._start_spin.blockSignals(True)
        self._end_spin.blockSignals(True)
        self._start_spin.setValue(start)
        self._end_spin.setValue(end)
        self._start_spin.blockSignals(False)
        self._end_spin.blockSignals(False)

        self._update_duration_label(start, end)

    def _update_duration_label(self, start: float, end: float) -> None:
        """
        Refresh the "Duration: X.XXX s" label from the current selection.

        Args:
            start: Selection start in seconds.
            end:   Selection end in seconds.
        """
        duration = max(0.0, end - start)
        self._dur_label.setText(f"Duration:  {duration:.3f} s")

    def _set_status(self, message: str, error: bool = False) -> None:
        """
        Display a status message below the toolbar.

        Args:
            message: Text to show. Pass "" to clear.
            error:   If True the text is shown in red, otherwise in green.
        """
        color = AppTheme.STATUS_FAILED if error else AppTheme.STATUS_DONE
        self._status_label.setText(message)
        self._status_label.setStyleSheet(
            f"color: {color}; "
            f"font-size: {AppTheme.FONT_SIZE_SM}px; "
            "background: transparent; padding: 4px 0;"
        )

    def _set_drop_highlight(self, active: bool) -> None:
        """
        Toggle the drop-hover visual style on the panel border.

        When a drag is hovering we switch the panel objectName so the
        stylesheet rule #audioDropZoneActive applies.

        Args:
            active: True while a drag hovers, False otherwise.
        """
        name = "audioDropZoneActive" if active else "trimmerPanel"
        self.setObjectName(name)
        self.style().unpolish(self)
        self.style().polish(self)
