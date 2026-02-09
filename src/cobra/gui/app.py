import sys
from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout

def run_gui():
    app = QApplication(sys.argv)

    # Show a simple dialog for demonstration purposes
    dialog = QDialog()
    dialog.setWindowTitle("COBRA GUI")
    dialog.resize(400, 300)
    # Add text 
    label = QLabel("Welcome to COBRA - A Circuit-Level Open-Source Based RFIC AI-Assisted Optimizer!", dialog)
    label.setAlignment(Qt.AlignCenter)
    layout = QVBoxLayout()
    layout.addWidget(label)
    dialog.setLayout(layout)
    dialog.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_gui()
