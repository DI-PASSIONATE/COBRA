THEME_STYLESHEET = """
QMainWindow {
    background-color: #F5F7FA;
}
QWidget {
    font-family: "Segoe UI", "Noto Sans", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    color: #263238;
}
QGroupBox {
    border: 1px solid #D9E0E6;
    border-radius: 10px;
    margin-top: 12px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #455A64;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #FFFFFF;
    border: 1px solid #CFD8DC;
    border-radius: 8px;
    padding: 6px 8px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #64B5F6;
}
QTableWidget {
    background-color: #FFFFFF;
    border: 1px solid #D9E0E6;
    border-radius: 8px;
}
QHeaderView::section {
    background-color: #ECEFF1;
    border: none;
    padding: 6px;
    font-weight: 600;
}
QProgressBar {
    border: 1px solid #CFD8DC;
    border-radius: 8px;
    text-align: center;
    background: #FFFFFF;
    height: 12px;
}
QProgressBar::chunk {
    background-color: #42A5F5;
    border-radius: 8px;
}
QPushButton {
    border-radius: 10px;
    padding: 8px 16px;
    border: 1px solid #CFD8DC;
    background-color: #ECEFF1;
    color: #263238;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #E3F2FD;
    border: 1px solid #90CAF9;
}
QPushButton:pressed {
    background-color: #BBDEFB;
}
QPushButton[tabButton="true"] {
    background-color: #E0E7EF;
}
QPushButton[tabButton="true"][tabActive="true"] {
    background-color: #FFFFFF;
    border: 2px solid #1E88E5;
    color: #1E88E5;
}
QPushButton[tabButton="true"]:disabled {
    background-color: #FFFFFF;
    border: 2px solid #1E88E5;
    color: #1E88E5;
}
QPushButton[primaryAction="true"] {
    border: none;
    background-color: #1E88E5;
    color: #FFFFFF;
}
QPushButton[primaryAction="true"][actionState="pause"] {
    background-color: #F9A825;
}
QPushButton[primaryAction="true"][actionState="resume"] {
    background-color: #43A047;
}
QPushButton[primaryAction="true"][actionState="stopping"] {
    background-color: #90A4AE;
}
QPushButton[primaryAction="true"]:disabled {
    background-color: #B0BEC5;
    color: #ECEFF1;
}
QPushButton[dangerAction="true"] {
    background-color: #E53935;
    color: #FFFFFF;
    border: none;
}
QPushButton[dangerAction="true"]:disabled {
    background-color: #EF9A9A;
    color: #FFFFFF;
}
"""


def apply_theme(widget) -> None:
    widget.setStyleSheet(THEME_STYLESHEET)