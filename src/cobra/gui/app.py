import sys

from PySide6.QtWidgets import QApplication

from cobra.console import ensure_logging
from cobra.gui import theme
from cobra.gui.main_window import MainWindow


def run_gui():
    # The CLI normally configures logging first; this covers `python -m cobra.gui.app`
    # and any embedding that reaches run_gui() directly.
    ensure_logging()
    app = QApplication(sys.argv)
    # Settings (appearance mode) and the cache (theme assets) are filed under these names.
    app.setOrganizationName("COBRA")
    app.setApplicationName("COBRA")
    # The stylesheet lives on the application so every window and dialog inherits it.
    theme.manager().apply()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_gui()
