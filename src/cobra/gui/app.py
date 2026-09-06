import sys

from PySide6.QtWidgets import QApplication

from cobra.console import ensure_logging
from cobra.gui.main_window import MainWindow


def run_gui():
    # The CLI normally configures logging first; this covers `python -m cobra.gui.app`
    # and any embedding that reaches run_gui() directly.
    ensure_logging()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_gui()
