"""
AudioExtracter — entry point.

Run this file to launch the application:

    python main.py

The module does nothing but bootstrap the Qt application, apply the
global stylesheet, create the main window, and start the event loop.
All business logic lives in src/.
"""

import os
import sys

# Ensure Qt can locate its platform plugins (e.g. "cocoa" on macOS).
# When running inside a conda environment the plugins folder is not on
# Qt's default search path, so we set it explicitly before any Qt import.
_pyqt6_dir = os.path.dirname(os.path.abspath(__file__))
_qt_plugin_path = os.path.join(
    os.path.dirname(sys.executable),
    "..",
    "lib",
    f"python{sys.version_info.major}.{sys.version_info.minor}",
    "site-packages",
    "PyQt6",
    "Qt6",
    "plugins",
    "platforms",
)
if os.path.isdir(_qt_plugin_path):
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", os.path.abspath(_qt_plugin_path))

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
