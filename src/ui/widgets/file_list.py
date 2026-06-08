"""
File queue widget — shows added video files with per-file status.

FileList owns the display of ExtractionJob objects. It never modifies
them; the main window passes updated jobs back via refresh_job() and
the widget re-renders the matching row.

Two classes live here:
  • FileRowWidget  — one row per file (name, size, arrow, status badge)
  • FileList       — scrollable container that manages a list of rows

Usage example:
    file_list = FileList(parent=self)
    file_list.add_jobs(jobs)           # append new rows
    file_list.refresh_job(updated_job) # re-render a single row's badge
    file_list.clear()                  # remove all rows
"""

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.core.models import ExtractionJob, JobStatus
from src.ui.theme import AppTheme
from src.utils.file_utils import human_readable_size


# ---------------------------------------------------------------------------
# Status badge helpers
# ---------------------------------------------------------------------------

# Maps each JobStatus to (label text, Qt object-name for CSS colouring).
# The object name matches the #badge* rules in theme.py.
_STATUS_CONFIG: dict[JobStatus, tuple[str, str]] = {
    JobStatus.PENDING: ("● Pending",  "badgePending"),
    JobStatus.RUNNING: ("⟳ Running…", "badgeRunning"),
    JobStatus.DONE:    ("✓ Done",     "badgeDone"),
    JobStatus.FAILED:  ("✗ Failed",   "badgeFailed"),
}


# ---------------------------------------------------------------------------
# FileRowWidget
# ---------------------------------------------------------------------------

class FileRowWidget(QWidget):
    """
    A single row in the file queue.

    Layout (left → right):
        [icon]  [filename + size]  [→ output name]  [status badge]

    The row is identified by the job's input_path so the parent list can
    look it up by path when calling refresh_job().

    Example:
        row = FileRowWidget(job)
        layout.addWidget(row)
    """

    def __init__(self, job: ExtractionJob, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("fileRow")
        self.job = job
        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the horizontal row layout."""
        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(12)

        # File icon emoji.
        icon = QLabel("🎞️", self)
        icon.setStyleSheet("font-size: 18px; background: transparent;")
        icon.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Left section: file name + size stacked vertically.
        left_col = QVBoxLayout()
        left_col.setSpacing(2)

        name_label = QLabel(self.job.filename, self)
        name_label.setStyleSheet(
            f"color: {AppTheme.TEXT_PRIMARY}; "
            f"font-size: {AppTheme.FONT_SIZE_BASE}px; "
            f"font-weight: 600; background: transparent;"
        )

        size_text = human_readable_size(self.job.input_path)
        size_label = QLabel(size_text, self)
        size_label.setStyleSheet(
            f"color: {AppTheme.TEXT_SECONDARY}; "
            f"font-size: {AppTheme.FONT_SIZE_SM}px; background: transparent;"
        )

        left_col.addWidget(name_label)
        left_col.addWidget(size_label)

        # Arrow + output filename.
        arrow_label = QLabel("→", self)
        arrow_label.setStyleSheet(
            f"color: {AppTheme.TEXT_DISABLED}; "
            f"font-size: {AppTheme.FONT_SIZE_BASE}px; background: transparent;"
        )

        output_label = QLabel(self.job.output_path.name, self)
        output_label.setStyleSheet(
            f"color: {AppTheme.TEXT_SECONDARY}; "
            f"font-size: {AppTheme.FONT_SIZE_BASE}px; background: transparent;"
        )

        # Status badge (right-aligned).
        self._status_label = QLabel(self)
        self._status_label.setStyleSheet(
            f"font-size: {AppTheme.FONT_SIZE_SM}px; "
            f"font-weight: 600; background: transparent;"
        )
        self._update_status_label(self.job.status)

        # Assemble the row.
        outer.addWidget(icon)
        outer.addLayout(left_col, stretch=2)
        outer.addWidget(arrow_label)
        outer.addWidget(output_label, stretch=2)
        outer.addStretch()
        outer.addWidget(self._status_label)

    def refresh(self, job: ExtractionJob) -> None:
        """
        Update the displayed status badge to reflect the latest job state.

        Call this whenever the main window receives a progress update for
        the job that corresponds to this row.

        Args:
            job: The updated ExtractionJob instance.
        """
        self.job = job
        self._update_status_label(job.status)

        # If the job failed, show the error in the tooltip for details.
        if job.status == JobStatus.FAILED and job.error:
            self._status_label.setToolTip(job.error)

    def _update_status_label(self, status: JobStatus) -> None:
        """
        Set the text and CSS class on the status badge label.

        The objectName is used by the global stylesheet (#badge* rules)
        to colour each status differently without any inline style logic.
        """
        text, obj_name = _STATUS_CONFIG[status]
        self._status_label.setText(text)
        self._status_label.setObjectName(obj_name)
        # Re-polish is needed for dynamic object-name changes to take effect.
        self._status_label.style().unpolish(self._status_label)
        self._status_label.style().polish(self._status_label)


# ---------------------------------------------------------------------------
# FileList
# ---------------------------------------------------------------------------

class FileList(QWidget):
    """
    A scrollable list of FileRowWidget rows, one per queued video file.

    The parent widget communicates with FileList exclusively through:
      - add_jobs(jobs)         — append new rows
      - refresh_job(job)       — re-render a single row's status badge
      - clear()                — remove all rows
      - job_count property     — check how many jobs are queued

    FileList stores a mapping from input_path → FileRowWidget for O(1)
    row lookup during refresh.

    Example:
        file_list = FileList(parent=self)
        file_list.add_jobs([job1, job2])
        file_list.refresh_job(updated_job1)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Path → row widget index for fast lookup.
        self._rows: dict[Path, FileRowWidget] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the outer scroll area and the inner container."""
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # Section header label.
        header = QLabel("QUEUE", self)
        header.setObjectName("sectionLabel")
        header.setContentsMargins(4, 0, 0, 8)
        outer_layout.addWidget(header)

        # Scroll area wraps the rows container.
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setFrameShape(scroll.Shape.NoFrame)

        # Inner container holds all FileRowWidget instances.
        self._container = QWidget()
        self._container.setObjectName("fileListContainer")

        self._inner_layout = QVBoxLayout(self._container)
        self._inner_layout.setContentsMargins(0, 0, 0, 0)
        self._inner_layout.setSpacing(6)
        self._inner_layout.addStretch()   # pushes rows to the top

        scroll.setWidget(self._container)
        outer_layout.addWidget(scroll)

        # Empty-state label, hidden once jobs are added.
        self._empty_label = QLabel("No files added yet.", self._container)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(
            f"color: {AppTheme.TEXT_DISABLED}; "
            f"font-size: {AppTheme.FONT_SIZE_BASE}px;"
        )
        self._inner_layout.insertWidget(0, self._empty_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def job_count(self) -> int:
        """Number of jobs currently displayed in the list."""
        return len(self._rows)

    def add_jobs(self, jobs: list[ExtractionJob]) -> None:
        """
        Append new rows to the list for each job in *jobs*.

        Jobs whose input_path is already present are skipped (no duplicates).

        Args:
            jobs: List of ExtractionJob objects to add.
        """
        for job in jobs:
            # Prevent duplicates — same file added twice.
            if job.input_path in self._rows:
                continue

            row = FileRowWidget(job)
            # Insert before the trailing stretch (last item in layout).
            insert_pos = self._inner_layout.count() - 1
            self._inner_layout.insertWidget(insert_pos, row)
            self._rows[job.input_path] = row

        self._sync_empty_label()

    def refresh_job(self, job: ExtractionJob) -> None:
        """
        Update the status badge on the row matching *job.input_path*.

        If no matching row exists the call is a no-op (safe to call
        from the worker thread's completion callback).

        Args:
            job: The updated ExtractionJob with new status/error fields.
        """
        row = self._rows.get(job.input_path)
        if row is not None:
            row.refresh(job)

    def clear(self) -> None:
        """Remove all rows and reset the list to the empty state."""
        for row in self._rows.values():
            self._inner_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()
        self._sync_empty_label()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sync_empty_label(self) -> None:
        """Show or hide the 'No files added yet' placeholder."""
        self._empty_label.setVisible(len(self._rows) == 0)
