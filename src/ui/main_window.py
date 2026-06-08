"""
Main application window.

MainWindow owns all top-level widgets and coordinates between them:
  1. Receives file paths via full-window drag-and-drop or the file-browse dialog
  2. Builds ExtractionJob objects and hands them to FileList
  3. Runs extraction on a QThreadPool via ExtractionWorker tasks
  4. Propagates job-status updates back to FileList rows

Drag-and-drop architecture (key design decision):
  The ENTIRE window is the drop target. MainWindow overrides Qt's drag events
  directly (dragEnterEvent, dragLeaveEvent, dragMoveEvent, dropEvent). None of
  the child widgets call setAcceptDrops(True), so Qt's event propagation
  naturally delivers unhandled drag events up to MainWindow.

  While a drag is hovering, _DragOverlay (a transparent child widget with
  WA_TransparentForMouseEvents) is raised to the top of the central widget to
  render the purple tint + border without interrupting event delivery.

Content area uses a QStackedWidget to switch between two states:
  - Index 0  →  EmptyStateWidget  ("Drag your video" prompt, no files loaded)
  - Index 1  →  FileList          (queue of extraction jobs)

Architecture:
    MainWindow
    ├── _DragOverlay       (overlay, not in layout, transparent to events)
    ├── QStackedWidget
    │   ├── [0] EmptyStateWidget   (no files)
    │   └── [1] FileList           (files loaded)
    └── Toolbar            (signals: browse_clicked, extract_clicked,
                                    clear_clicked, format_changed,
                                    output_folder_changed)

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
from src.ui.widgets.drop_zone import EmptyStateWidget
from src.ui.widgets.file_list import FileList
from src.ui.widgets.toolbar import Toolbar
from src.utils.file_utils import build_output_path
from src.utils.validators import VIDEO_FILE_FILTER, filter_supported_videos


# ---------------------------------------------------------------------------
# Worker signals (must live on a QObject to cross thread boundaries)
# ---------------------------------------------------------------------------

class _WorkerSignals(QObject):
    """
    Carrier for signals emitted by ExtractionWorker back to the main thread.

    Qt signals can only be emitted from a QObject, so ExtractionWorker
    (a QRunnable) delegates signal ownership to this helper class.
    """
    # Emitted when a single job finishes (success or failure).
    job_finished = pyqtSignal(ExtractionJob)
    # Emitted when all queued jobs have been processed.
    all_done = pyqtSignal()


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
        self.job = job
        self.extractor = extractor
        self.signals = signals

    @pyqtSlot()
    def run(self) -> None:
        """
        Execute the extraction and emit job_finished when complete.

        Called by QThreadPool on a worker thread — never call directly.
        """
        self.job.mark_running()

        result = self.extractor.extract(self.job)

        # Mutate the job in place before signalling so the UI gets the
        # latest state in a single signal delivery.
        if result.success:
            self.job.mark_done()
        else:
            self.job.mark_failed(result.error_message)

        self.signals.job_finished.emit(self.job)


# ---------------------------------------------------------------------------
# Drag overlay
# ---------------------------------------------------------------------------

class _DragOverlay(QWidget):
    """
    Full-window translucent overlay rendered while a drag is hovering.

    Parented to the central widget but NOT placed in any layout — its
    geometry is managed manually in MainWindow.resizeEvent() so it always
    covers the entire central widget area.

    WA_TransparentForMouseEvents ensures that drag events are NOT consumed
    by this widget; they pass through to MainWindow's override handlers.

    Visual effect:
        • Semi-transparent purple tint (rgba(44, 44, 72, 180))
        • 3 px solid accent border (#6C63FF)
        • Matching border-radius to the panel style

    Example usage (internal — managed by MainWindow):
        overlay = _DragOverlay(central_widget)
        overlay.setGeometry(central_widget.rect())
        overlay.show()   # called in _set_drag_active(True)
        overlay.hide()   # called in _set_drag_active(False)
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("dragOverlay")
        # Allow drag events to pass through — MainWindow receives them instead.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """
    Top-level application window.

    Responsibilities:
      - Accept drag-and-drop on the entire window surface
      - Show EmptyStateWidget when queue is empty, FileList when it has jobs
      - Translate file paths into ExtractionJob objects
      - Schedule jobs on the thread pool
      - Route job-status signals back to the FileList
      - Show error/warning dialogs when needed
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AudioExtracter")
        self.setMinimumSize(760, 540)
        self.resize(900, 620)

        # Enable drops on the window itself. Child widgets do NOT call
        # setAcceptDrops(True), so unhandled drags bubble up here naturally.
        self.setAcceptDrops(True)

        # Active extraction jobs keyed by input_path for O(1) lookup.
        self._jobs: dict[Path, ExtractionJob] = {}

        # Selected output format (kept in sync with the toolbar combo).
        self._output_format: AudioFormat = AudioFormat.MP3

        # Output directory (None = same folder as each input file).
        self._output_dir: Path | None = None

        # Shared extractor (stateless, safe to reuse across threads).
        try:
            self._extractor = AudioExtractor()
        except EnvironmentError as exc:
            # ffmpeg not found — we still let the window open but warn the user.
            self._extractor = None  # type: ignore[assignment]
            self._ffmpeg_missing_message = str(exc)
        else:
            self._ffmpeg_missing_message = ""

        # Signals shared across all workers.
        self._signals = _WorkerSignals()
        self._signals.job_finished.connect(self._on_job_finished)

        # Counter to track when the whole batch is done.
        self._running_count = 0

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """
        Create and arrange all child widgets.

        Layout (top → bottom inside central widget):
            QStackedWidget  [stretch=1]
              ├─ EmptyStateWidget  (index 0 — no files queued)
              └─ FileList          (index 1 — files present)
            Toolbar

        _DragOverlay sits outside the layout but is parented to the
        central widget so it can be sized to cover everything.
        """
        central = QWidget(self)
        central.setObjectName("centralContent")
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 0)
        layout.setSpacing(12)

        # QStackedWidget switches between empty-state and file-list views.
        # Index 0: nothing queued — show the hero "Drag your video" prompt.
        # Index 1: one or more jobs queued — show the scrollable file list.
        self._stack = QStackedWidget(central)
        self._empty_state = EmptyStateWidget()
        self._file_list = FileList()
        self._stack.addWidget(self._empty_state)   # index 0
        self._stack.addWidget(self._file_list)      # index 1
        self._stack.setCurrentIndex(0)

        self._toolbar = Toolbar(parent=central)

        layout.addWidget(self._stack, stretch=1)
        layout.addWidget(self._toolbar)

        # Drag overlay — parented to central widget but NOT in the layout.
        # Positioned and sized in resizeEvent() to always cover the full area.
        # Must be created AFTER layout children so raise_() puts it on top.
        self._drag_overlay = _DragOverlay(central)
        self._drag_overlay.setGeometry(central.rect())
        self._drag_overlay.raise_()

    def _connect_signals(self) -> None:
        """Wire every widget signal to the appropriate handler."""
        # Drop zone is now MainWindow itself — no separate DropZone signal.
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
            QMessageBox.warning(
                self,
                "ffmpeg not found",
                self._ffmpeg_missing_message,
            )

    def resizeEvent(self, event) -> None:
        """
        Keep the drag overlay sized to match the central widget at all times.

        The overlay lives outside the layout so Qt doesn't resize it
        automatically — we must do it here whenever the window changes size.
        """
        super().resizeEvent(event)
        if hasattr(self, "_drag_overlay"):
            # centralWidget().rect() is in the central widget's local
            # coordinate space (0, 0, w, h) — exactly what we need since
            # _drag_overlay is parented to the central widget.
            self._drag_overlay.setGeometry(self.centralWidget().rect())

    # ------------------------------------------------------------------
    # Full-window drag-and-drop events
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """
        Accept the incoming drag if it contains at least one local file URL.

        Called when the cursor first enters the window with a drag payload.
        Accepting here enables dragMoveEvent and dropEvent to fire.
        The overlay is shown to give the user visual confirmation that the
        whole window is receptive to the drop.
        """
        mime = event.mimeData()
        # Only files dragged from the OS file manager have local URLs.
        if mime.hasUrls() and any(url.isLocalFile() for url in mime.urls()):
            event.acceptProposedAction()
            self._set_drag_active(True)
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        """
        Keep accepting the drag as the cursor moves across the window.

        Qt re-checks acceptance at every move; without this override the
        drop cursor would revert to a "not allowed" icon mid-drag.
        """
        mime = event.mimeData()
        if mime.hasUrls() and any(url.isLocalFile() for url in mime.urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        """
        Revert visual state when the drag leaves the window bounds.

        Called when the user drags the payload outside the window without
        dropping, or cancels with Escape.
        """
        self._set_drag_active(False)

    def dropEvent(self, event: QDropEvent) -> None:
        """
        Handle the final drop, validate files, and route to _on_files_received.

        The overlay is hidden before processing so the UI snaps back to its
        normal appearance immediately on drop. Unsupported file types are
        silently discarded; at least one valid file must be present for
        the event to be accepted.

        Args:
            event: Qt drop event carrying MIME data with file URLs.
        """
        self._set_drag_active(False)

        paths = [
            Path(url.toLocalFile())
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]

        # Filter to only supported video files; discard everything else.
        valid, _ = filter_supported_videos(paths)

        # Only accept and process the event if at least one video was found.
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
        Show or hide the full-window drag overlay.

        When active=True the _DragOverlay widget is raised to the top of
        the Z-order and made visible, giving a purple tint and border over
        the entire content area. When active=False it is hidden.

        Args:
            active: True while a drag is hovering, False otherwise.
        """
        if active:
            # Ensure the overlay covers the current central widget area.
            self._drag_overlay.setGeometry(self.centralWidget().rect())
            self._drag_overlay.raise_()
            self._drag_overlay.show()
        else:
            self._drag_overlay.hide()

    def _update_content_view(self) -> None:
        """
        Switch the stacked widget between the empty-state and file-list views.

        Rule:
          - No jobs in queue  →  index 0 (EmptyStateWidget)
          - At least one job  →  index 1 (FileList)

        Called whenever the job set changes (files received or queue cleared).
        """
        # Switch to FileList when there are jobs; back to empty state when not.
        target_index = 1 if self._jobs else 0
        if self._stack.currentIndex() != target_index:
            self._stack.setCurrentIndex(target_index)

    # ------------------------------------------------------------------
    # Slots — file input
    # ------------------------------------------------------------------

    @pyqtSlot(list)
    def _on_files_received(self, paths: list[Path]) -> None:
        """
        Handle new video paths from either the drop event or file browser.

        For each path we build an ExtractionJob with a unique output path
        and add it to the queue. Already-queued files are skipped silently.
        After adding, the content view is updated to show the file list.

        Args:
            paths: List of validated video file Paths.
        """
        new_jobs: list[ExtractionJob] = []

        for path in paths:
            # Skip files already in the queue — prevents duplicates.
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
            # New files mean we should be showing the file list, not the prompt.
            self._update_content_view()

    @pyqtSlot()
    def _on_browse(self) -> None:
        """
        Open the native file browser and hand selected files to _on_files_received.

        The dialog filter is built from SUPPORTED_VIDEO_EXTENSIONS so only
        valid video files appear in the browser by default.
        """
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
    # Slots — extraction
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_extract_all(self) -> None:
        """
        Start extraction for every PENDING job in the queue.

        Jobs that are already DONE or RUNNING are skipped. Each job is
        submitted to QThreadPool as an ExtractionWorker. The thread pool
        manages parallelism automatically (defaults to CPU-core count).
        """
        if self._extractor is None:
            QMessageBox.critical(
                self, "ffmpeg not found", self._ffmpeg_missing_message
            )
            return

        pending_jobs = [
            job for job in self._jobs.values()
            if job.status == JobStatus.PENDING
        ]

        if not pending_jobs:
            return

        self._running_count = len(pending_jobs)
        self._toolbar.set_busy(True)

        pool = QThreadPool.globalInstance()

        for job in pending_jobs:
            # Mark as running optimistically before the thread picks it up.
            job.mark_running()
            self._file_list.refresh_job(job)

            worker = ExtractionWorker(job, self._extractor, self._signals)
            pool.start(worker)

    @pyqtSlot(ExtractionJob)
    def _on_job_finished(self, job: ExtractionJob) -> None:
        """
        Receive a completed job from a background worker and update the UI.

        This slot is always called on the main thread because _signals is
        connected via Qt's auto-connection mechanism.

        Args:
            job: The ExtractionJob with updated status (DONE or FAILED).
        """
        self._file_list.refresh_job(job)

        self._running_count -= 1
        if self._running_count <= 0:
            # All jobs finished — restore toolbar state.
            self._toolbar.set_busy(False)
            self._toolbar.set_extract_enabled(False)
            self._running_count = 0

    # ------------------------------------------------------------------
    # Slots — queue management
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_clear(self) -> None:
        """
        Remove all jobs from the queue and reset the UI.

        After clearing, _update_content_view switches back to the
        EmptyStateWidget so the "Drag your video" prompt reappears.
        """
        self._jobs.clear()
        self._file_list.clear()
        self._toolbar.set_extract_enabled(False)
        # No jobs remain — show the empty-state prompt again.
        self._update_content_view()

    @pyqtSlot(AudioFormat)
    def _on_format_changed(self, fmt: AudioFormat) -> None:
        """
        Update the output format for any future jobs added to the queue.

        Already-created jobs keep their original format; only new jobs
        added after this change use the new format. This avoids
        silently renaming output paths of pending jobs mid-session.

        Args:
            fmt: The newly selected AudioFormat.
        """
        self._output_format = fmt

    @pyqtSlot(Path)
    def _on_output_folder_changed(self, folder: Path) -> None:
        """
        Update the output directory for future jobs.

        As with format changes, existing jobs are not re-pathed —
        only new additions pick up the new folder.

        Args:
            folder: The selected output directory Path.
        """
        self._output_dir = folder
