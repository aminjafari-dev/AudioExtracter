"""
Bottom toolbar widget.

The toolbar sits at the bottom of the main window and contains:
  - A format selector ComboBox (which audio format to extract)
  - A "Browse" button (open file dialog)
  - A "Choose Output Folder" button
  - An "Extract All" primary action button
  - A "Clear" secondary action button

The toolbar emits signals for every user action; it does not call any
extraction logic itself. The main window connects to these signals.

Usage example:
    toolbar = Toolbar(parent=self)
    toolbar.browse_clicked.connect(self._on_browse)
    toolbar.extract_clicked.connect(self._on_extract)
    toolbar.clear_clicked.connect(self._on_clear)
    toolbar.output_folder_changed.connect(self._on_output_folder_changed)
"""

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from src.core.models import AudioFormat
from src.ui.theme import AppTheme


class Toolbar(QWidget):
    """
    Bottom action bar with format selection and operation buttons.

    Signals:
        browse_clicked:          User clicked the Browse button.
        extract_clicked:         User clicked the Extract All button.
        clear_clicked:           User clicked the Clear button.
        format_changed (AudioFormat):    User selected a different output format.
        output_folder_changed (Path):    User chose a different output directory.
    """

    browse_clicked          = pyqtSignal()
    extract_clicked         = pyqtSignal()
    clear_clicked           = pyqtSignal()
    format_changed          = pyqtSignal(AudioFormat)
    output_folder_changed   = pyqtSignal(Path)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("toolbar")
        self.setFixedHeight(AppTheme.TOOLBAR_HEIGHT)

        # Track the currently selected output directory (None = same as input).
        self._output_dir: Path | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Lay out the toolbar controls horizontally."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(10)

        # ── Format selector ────────────────────────────────────────
        format_label = QLabel("Format:", self)
        format_label.setStyleSheet(
            f"color: {AppTheme.TEXT_SECONDARY}; "
            f"font-size: {AppTheme.FONT_SIZE_BASE}px; background: transparent;"
        )

        self._format_combo = QComboBox(self)
        for fmt in AudioFormat:
            # Store the enum member as item data so we never parse strings.
            self._format_combo.addItem(fmt.label, userData=fmt)

        # Emit format_changed with the AudioFormat enum whenever selection changes.
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)

        # ── Browse button ──────────────────────────────────────────
        browse_btn = QPushButton("Browse…", self)
        browse_btn.setObjectName("secondaryButton")
        browse_btn.clicked.connect(self.browse_clicked.emit)

        # ── Output folder button ──────────────────────────────────
        self._output_btn = QPushButton("Output: same folder", self)
        self._output_btn.setObjectName("secondaryButton")
        self._output_btn.clicked.connect(self._on_choose_output_folder)

        # Spacer pushes Extract/Clear to the far right.
        layout.addWidget(format_label)
        layout.addWidget(self._format_combo)
        layout.addWidget(browse_btn)
        layout.addWidget(self._output_btn)
        layout.addStretch()

        # ── Clear button ───────────────────────────────────────────
        clear_btn = QPushButton("Clear", self)
        clear_btn.setObjectName("secondaryButton")
        clear_btn.clicked.connect(self.clear_clicked.emit)

        # ── Extract All button (primary CTA) ───────────────────────
        self._extract_btn = QPushButton("Extract All", self)
        self._extract_btn.setObjectName("primaryButton")
        self._extract_btn.setEnabled(False)   # enabled only when queue has items
        self._extract_btn.clicked.connect(self.extract_clicked.emit)

        layout.addWidget(clear_btn)
        layout.addWidget(self._extract_btn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def selected_format(self) -> AudioFormat:
        """Return the AudioFormat enum currently selected in the combo box."""
        return self._format_combo.currentData()

    @property
    def output_dir(self) -> Path | None:
        """
        Return the output directory chosen by the user, or None to use the
        same directory as each input file.
        """
        return self._output_dir

    def set_extract_enabled(self, enabled: bool) -> None:
        """
        Enable or disable the Extract All button.

        The main window calls this to disable the button when the queue
        is empty or when an extraction is already running.

        Args:
            enabled: True to enable, False to disable.
        """
        self._extract_btn.setEnabled(enabled)

    def set_busy(self, busy: bool) -> None:
        """
        Switch the Extract All button between its normal and in-progress state.

        While extraction is running we change the button label and disable
        it so the user cannot start a second batch.

        Args:
            busy: True while extraction is in progress.
        """
        if busy:
            self._extract_btn.setText("Extracting…")
            self._extract_btn.setEnabled(False)
        else:
            self._extract_btn.setText("Extract All")
            # Re-enable only if there are still pending jobs; the main window
            # will call set_extract_enabled(True/False) after this.

    # ------------------------------------------------------------------
    # Private slots
    # ------------------------------------------------------------------

    def _on_format_changed(self, _index: int) -> None:
        """
        Called whenever the combo box selection changes.

        We emit format_changed with the AudioFormat enum value so the main
        window can rebuild output paths before extraction starts.
        """
        self.format_changed.emit(self._format_combo.currentData())

    def _on_choose_output_folder(self) -> None:
        """
        Open a native directory picker.

        If the user selects a folder we update the button label to show the
        short folder name and emit output_folder_changed with the Path.
        Clicking Cancel leaves the existing selection unchanged.
        """
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose Output Folder",
            str(Path.home()),
        )

        if not folder:
            # User cancelled — leave the existing selection unchanged.
            return

        self._output_dir = Path(folder)
        # Show a trimmed path on the button to avoid overflow.
        short_name = self._output_dir.name or str(self._output_dir)
        self._output_btn.setText(f"Output: {short_name}")
        self.output_folder_changed.emit(self._output_dir)
