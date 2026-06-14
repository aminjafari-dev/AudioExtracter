"""
Main application window.

MainWindow owns all top-level widgets and coordinates between them:
  1. Hosts a top TabBar (Video / Audio / Compress tabs)
  2. Switches a QStackedWidget between the Video panel, the Audio panel,
     and the Video Compress panel
  3. Video panel: full-window drag-and-drop, job queue, threaded extraction
  4. Audio panel: waveform visualisation and audio trimming (AudioTrimmerPanel)
  5. Compress panel: single-file drop, quality settings, threaded compression

Tab layout:
    MainWindow
    ├─ TabBar                  (top navigation — Video / Audio / Compress)
    └─ QStackedWidget (pages)
        ├─ [0] Video page
        │   ├─ _DragOverlay    (transparent overlay during drag)
        │   ├─ QStackedWidget  (empty state ↔ file list)
        │   │   ├─ [0] EmptyStateWidget
        │   │   └─ [1] FileList
        │   └─ Toolbar
        ├─ [1] AudioTrimmerPanel
        └─ [2] VideoCompressorPanel

Drag-and-drop:
    Only the Video page accepts window-level drops. The Audio and Compress
    panels each manage their own drag-and-drop internally.

Usage:
    window = MainWindow()
    window.show()
"""

from pathlib import Path

from PyQt6.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal, pyqtSlot, Qt
from PyQt6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent
from PyQt6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.core.extractor import AudioExtractor
from src.core.models import AudioFormat, ExtractionJob, JobStatus
from src.ui.theme import AppTheme
from src.ui.widgets.audio_trimmer_panel import AudioTrimmerPanel
from src.ui.widgets.drop_zone import EmptyStateWidget
from src.ui.widgets.file_list import FileList
from src.ui.widgets.tab_bar import TAB_AUDIO, TAB_COMPRESS, TAB_VIDEO, TabBar
from src.ui.widgets.toolbar import Toolbar
from src.ui.widgets.video_compressor_panel import VideoCompressorPanel
from src.utils.file_utils import build_output_path
from src.utils.validators import VIDEO_FILE_FILTER, filter_supported_videos


# ---------------------------------------------------------------------------
# Worker signals
# ---------------------------------------------------------------------------

class _WorkerSignals(QObject):
    """
    Carrier for signals emitted by ExtractionWorker back to the main thread.

    Qt signals can only be emitted from a QObject, so ExtractionWorker
    (a QRunnable) delegates signal ownership to this helper class.
    """
    job_finished = pyqtSignal(ExtractionJob)
    all_done     = pyqtSignal()


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class ExtractionWorker(QRunnable):
    """
    Runs one ExtractionJob on a background thread from QThreadPool.

    After the job completes (success or failure) the worker updates
    job.status in place and emits signals so the main thread can refresh
    the UI without any shared-state issues.

    Args:
        job:       The ExtractionJob to process.
        extractor: The shared AudioExtractor instance (stateless, thread-safe).
        signals:   _WorkerSignals instance for cross-thread communication.

    Usage (internal — created by MainWindow._on_extract_all):
        worker = ExtractionWorker(job, extractor, signals)
        QThreadPool.globalInstance().start(worker)
    """

    def __init__(
        self,
        job: ExtractionJob,
        extractor: AudioExtractor,
        signals: _WorkerSignals,
    ) -> None:
        super().__init__()
        self.job       = job
        self.extractor = extractor
        self.signals   = signals

    @pyqtSlot()
    def run(self) -> None:
        """Execute the extraction and emit job_finished when complete."""
        self.job.mark_running()
        result = self.extractor.extract(self.job)
        if result.success:
            self.job.mark_done()
        else:
            self.job.mark_failed(result.error_message)
        self.signals.job_finished.emit(self.job)


# ---------------------------------------------------------------------------
# Drag overlay (Video page only)
# ---------------------------------------------------------------------------

class _DragOverlay(QWidget):
    """
    Full-page translucent overlay rendered while a drag hovers over the
    Video page.  Parented to the video page widget but NOT in any layout —
    geometry is set in resizeEvent.

    WA_TransparentForMouseEvents ensures drag events pass through to MainWindow.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("dragOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """
    Top-level application window.

    Responsibilities:
      - Render the top TabBar and switch content pages on tab change
      - (Video tab) Accept drag-and-drop, queue extraction jobs, run workers
      - (Audio tab) Delegate entirely to AudioTrimmerPanel
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AudioExtracter")
        self.setMinimumSize(820, 580)
        self.resize(960, 660)
        self.setAcceptDrops(True)

        # ── Video-panel state ──────────────────────────────────────────
        self._jobs: dict[Path, ExtractionJob] = {}
        self._output_format: AudioFormat = AudioFormat.MP3
        self._output_dir: Path | None = None

        try:
            self._extractor = AudioExtractor()
        except EnvironmentError as exc:
            self._extractor = None  # type: ignore[assignment]
            self._ffmpeg_missing_message = str(exc)
        else:
            self._ffmpeg_missing_message = ""

        self._signals = _WorkerSignals()
        self._signals.job_finished.connect(self._on_job_finished)
        self._running_count = 0

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """
        Build the top-level widget tree.

        Structure inside the central widget:
            VBoxLayout
              ├─ TabBar
              └─ QStackedWidget (page_stack)
                  ├─ [0] _video_page  (VBox: inner_stack + Toolbar)
                  └─ [1] AudioTrimmerPanel
        """
        central = QWidget(self)
        central.setObjectName("centralContent")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Tab bar ───────────────────────────────────────────────────
        self._tab_bar = TabBar(parent=central)
        root.addWidget(self._tab_bar)

        # ── Page stack ────────────────────────────────────────────────
        self._page_stack = QStackedWidget(central)
        root.addWidget(self._page_stack, stretch=1)

        # ── Video page ────────────────────────────────────────────────
        self._video_page = QWidget(self._page_stack)
        video_layout = QVBoxLayout(self._video_page)
        video_layout.setContentsMargins(16, 16, 16, 0)
        video_layout.setSpacing(12)

        # Inner stacked widget: empty state (0) ↔ file list (1)
        self._inner_stack = QStackedWidget(self._video_page)
        self._empty_state = EmptyStateWidget()
        self._file_list   = FileList()
        self._inner_stack.addWidget(self._empty_state)   # index 0
        self._inner_stack.addWidget(self._file_list)     # index 1
        self._inner_stack.setCurrentIndex(0)

        self._toolbar = Toolbar(parent=self._video_page)

        video_layout.addWidget(self._inner_stack, stretch=1)
        video_layout.addWidget(self._toolbar)

        # Drag overlay — covers the video page content during a drag hover.
        self._drag_overlay = _DragOverlay(self._video_page)
        self._drag_overlay.setGeometry(self._video_page.rect())
        self._drag_overlay.raise_()

        self._page_stack.addWidget(self._video_page)     # page index 0

        # ── Audio page ────────────────────────────────────────────────
        self._audio_panel = AudioTrimmerPanel(parent=self._page_stack)
        self._page_stack.addWidget(self._audio_panel)    # page index 1

        # ── Compress page ─────────────────────────────────────────────
        # VideoCompressorPanel handles its own drag-and-drop internally,
        # just like AudioTrimmerPanel does for the audio tab.
        self._compress_panel = VideoCompressorPanel(parent=self._page_stack)
        self._page_stack.addWidget(self._compress_panel) # page index 2

        # Show the Video page initially.
        self._page_stack.setCurrentIndex(TAB_VIDEO)

    def _connect_signals(self) -> None:
        """Wire all widget signals to their handler slots."""
        self._tab_bar.tab_changed.connect(self._on_tab_changed)
        self._toolbar.browse_clicked.connect(self._on_browse)
        self._toolbar.extract_clicked.connect(self._on_extract_all)
        self._toolbar.clear_clicked.connect(self._on_clear)
        self._toolbar.format_changed.connect(self._on_format_changed)
        self._toolbar.output_folder_changed.connect(self._on_output_folder_changed)

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: ANN001
        """Show ffmpeg-missing warning after the window appears."""
        super().showEvent(event)
        if self._ffmpeg_missing_message:
            QMessageBox.warning(self, "ffmpeg not found", self._ffmpeg_missing_message)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        """
        Keep the drag overlay sized to cover the video page content area.

        The overlay is outside any layout so we must resize it manually
        whenever the window changes size.
        """
        super().resizeEvent(event)
        if hasattr(self, "_drag_overlay"):
            self._drag_overlay.setGeometry(self._video_page.rect())

    # ------------------------------------------------------------------
    # Tab switching
    # ------------------------------------------------------------------

    @pyqtSlot(int)
    def _on_tab_changed(self, index: int) -> None:
        """
        Switch the page stack to the selected tab.

        Window-level drag-and-drop is only meaningful for the Video tab
        (TAB_VIDEO = 0).  The Audio and Compress panels each manage their
        own drops internally, so we disable window-level DnD for those tabs.

        Args:
            index: TAB_VIDEO (0), TAB_AUDIO (1), or TAB_COMPRESS (2).
        """
        self._page_stack.setCurrentIndex(index)
        # Only accept window-level drops when the Video page is active.
        # The Audio and Compress panels handle their own DnD independently.
        self.setAcceptDrops(index == TAB_VIDEO)

    # ------------------------------------------------------------------
    # Drag-and-drop (Video tab only)
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept the drag if it contains at least one local file URL."""
        mime = event.mimeData()
        if mime.hasUrls() and any(url.isLocalFile() for url in mime.urls()):
            event.acceptProposedAction()
            self._set_drag_active(True)
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        """Keep accepting the drag as the cursor moves across the window."""
        mime = event.mimeData()
        if mime.hasUrls() and any(url.isLocalFile() for url in mime.urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        """Remove visual feedback when the drag leaves the window."""
        self._set_drag_active(False)

    def dropEvent(self, event: QDropEvent) -> None:
        """Validate dropped files and route to _on_files_received."""
        self._set_drag_active(False)
        paths = [
            Path(url.toLocalFile())
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        valid, _ = filter_supported_videos(paths)
        if valid:
            event.acceptProposedAction()
            self._on_files_received(valid)
        else:
            event.ignore()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _set_drag_active(self, active: bool) -> None:
        """
        Show or hide the drag overlay on the video page.

        Args:
            active: True while a drag hovers, False otherwise.
        """
        if active:
            self._drag_overlay.setGeometry(self._video_page.rect())
            self._drag_overlay.raise_()
            self._drag_overlay.show()
        else:
            self._drag_overlay.hide()

    def _update_content_view(self) -> None:
        """Switch the inner stack between empty-state (0) and file-list (1)."""
        target = 1 if self._jobs else 0
        if self._inner_stack.currentIndex() != target:
            self._inner_stack.setCurrentIndex(target)

    # ------------------------------------------------------------------
    # Slots — file input (Video tab)
    # ------------------------------------------------------------------

    @pyqtSlot(list)
    def _on_files_received(self, paths: list[Path]) -> None:
        """
        Handle new video paths from either drag-and-drop or the file browser.

        For each path we build an ExtractionJob and add it to the queue.
        Already-queued files are skipped silently.

        Args:
            paths: List of validated video file Paths.
        """
        new_jobs: list[ExtractionJob] = []
        for path in paths:
            if path in self._jobs:
                continue
            output_path = build_output_path(
                input_path=path,
                output_format=self._output_format,
                output_dir=self._output_dir,
            )
            job = ExtractionJob(
                input_path=path,
                output_path=output_path,
                output_format=self._output_format,
            )
            self._jobs[path] = job
            new_jobs.append(job)

        if new_jobs:
            self._file_list.add_jobs(new_jobs)
            self._toolbar.set_extract_enabled(True)
            self._update_content_view()

    @pyqtSlot()
    def _on_browse(self) -> None:
        """Open the native file browser for video files."""
        selected, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            str(Path.home()),
            VIDEO_FILE_FILTER,
        )
        if not selected:
            return
        paths = [Path(p) for p in selected]
        valid, _ = filter_supported_videos(paths)
        if valid:
            self._on_files_received(valid)

    # ------------------------------------------------------------------
    # Slots — extraction (Video tab)
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_extract_all(self) -> None:
        """Start extraction for every PENDING job in the queue."""
        if self._extractor is None:
            QMessageBox.critical(self, "ffmpeg not found", self._ffmpeg_missing_message)
            return

        pending_jobs = [j for j in self._jobs.values() if j.status == JobStatus.PENDING]
        if not pending_jobs:
            return

        self._running_count = len(pending_jobs)
        self._toolbar.set_busy(True)
        pool = QThreadPool.globalInstance()

        for job in pending_jobs:
            job.mark_running()
            self._file_list.refresh_job(job)
            worker = ExtractionWorker(job, self._extractor, self._signals)
            pool.start(worker)

    @pyqtSlot(ExtractionJob)
    def _on_job_finished(self, job: ExtractionJob) -> None:
        """
        Receive a completed job from a background worker and update the UI.

        Args:
            job: The ExtractionJob with updated status (DONE or FAILED).
        """
        self._file_list.refresh_job(job)
        self._running_count -= 1
        if self._running_count <= 0:
            self._toolbar.set_busy(False)
            self._toolbar.set_extract_enabled(False)
            self._running_count = 0

    # ------------------------------------------------------------------
    # Slots — queue management (Video tab)
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_clear(self) -> None:
        """Remove all jobs from the queue and reset the UI."""
        self._jobs.clear()
        self._file_list.clear()
        self._toolbar.set_extract_enabled(False)
        self._update_content_view()

    @pyqtSlot(AudioFormat)
    def _on_format_changed(self, fmt: AudioFormat) -> None:
        """Update the output format for future jobs."""
        self._output_format = fmt

    @pyqtSlot(Path)
    def _on_output_folder_changed(self, folder: Path) -> None:
        """Update the output directory for future jobs."""
        self._output_dir = folder
