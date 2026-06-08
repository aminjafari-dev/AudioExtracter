"""
AudioExtracter — entry point.

Run this file to launch the application:

    python main.py

The module does nothing but bootstrap the Qt application, apply the
global stylesheet, create the main window, and start the event loop.
All business logic lives in src/.
"""

import sys

from PyQt6.QtWidgets import QApplication

from src.ui.main_window import MainWindow
from src.ui.theme import AppTheme


def main() -> None:
    """
    Bootstrap and run the AudioExtracter application.

    Creates the QApplication singleton, applies the dark theme stylesheet,
    shows the main window, and enters the Qt event loop. The process exits
    with the event-loop's return code so the OS can detect abnormal exits.
    """
    app = QApplication(sys.argv)
    app.setApplicationName("AudioExtracter")
    app.setApplicationDisplayName("AudioExtracter")

    # Apply the dark theme stylesheet defined in ui/theme.py to every widget.
    app.setStyleSheet(AppTheme.build_stylesheet())

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
