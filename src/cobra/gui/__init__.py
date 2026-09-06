# The GUI modules are deliberately not re-exported: importing them pulls in PySide6,
# which is optional for headless runs.  Import the entry point explicitly via
# ``from cobra.gui.app import ...`` instead.
