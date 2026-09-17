"""Appearance of the COBRA GUI: colour tokens, the two themes, and the stylesheet.

This module is the single owner of every colour used in ``cobra.gui``.  It must
stay importable without a display: no widget is created at import time, so the
token tables and :func:`build_stylesheet` can be unit-tested headless.
See ``.claude/GUI_DESIGN.md`` for the design rules the tokens implement.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QSettings, QSize, QStandardPaths, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPalette
from PySide6.QtWidgets import QApplication, QWidget

if TYPE_CHECKING:
    import pyqtgraph as pg
    from PySide6.QtWidgets import QAbstractButton

#: Values accepted by the persisted ``appearance/mode`` setting.
MODES: tuple[str, ...] = ("system", "light", "dark")


@dataclass(frozen=True)
class ThemeTokens:
    name: str
    dark: bool

    # Surfaces and text
    canvas: str
    surface: str
    surface_alt: str
    border: str
    border_strong: str
    text: str
    text_muted: str
    text_disabled: str
    selection: str

    # Semantic colours
    tide: str
    tide_hover: str
    tide_pressed: str
    tide_subtle: str
    on_tide: str
    moss: str
    moss_strong: str
    moss_subtle: str
    ochre: str
    ochre_strong: str
    ochre_subtle: str
    ember: str
    ember_strong: str
    ember_subtle: str
    slate: str

    #: Plot series, taken by index and cycled.
    series: tuple[str, ...]

    # Shape and spacing
    radius_sm: int = 6
    radius_md: int = 8
    radius_lg: int = 10
    space_1: int = 4
    space_2: int = 8
    space_3: int = 12
    space_4: int = 16
    space_5: int = 24
    control_h: int = 36
    control_h_lg: int = 48

    def series_color(self, index: int) -> str:
        return self.series[index % len(self.series)]


SANDBANK = ThemeTokens(
    name="Sandbank",
    dark=False,
    canvas="#F3F4F1",
    surface="#FBFBF9",
    surface_alt="#E8EAE5",
    border="#D3D7D0",
    border_strong="#B6BCB3",
    text="#2B3138",
    text_muted="#5F6870",
    text_disabled="#9AA2A8",
    selection="#D6E2EC",
    tide="#3C5A78",
    tide_hover="#34506B",
    tide_pressed="#2C4359",
    tide_subtle="#E4EBF2",
    on_tide="#FFFFFF",
    moss="#6B8F5E",
    moss_strong="#4E6E44",
    moss_subtle="#E6EEE1",
    ochre="#B08A3E",
    ochre_strong="#7F6226",
    ochre_subtle="#F3ECDC",
    ember="#A85A4E",
    ember_strong="#8A4438",
    ember_subtle="#F2E3E0",
    slate="#6B7580",
    series=("#3C5A78", "#6B8F5E", "#B08A3E", "#A85A4E", "#7A6A93", "#4A8A8A", "#6B7580", "#8A6B4F"),
)

DEEPWATER = ThemeTokens(
    name="Deepwater",
    dark=True,
    canvas="#1B1F24",
    surface="#23282E",
    surface_alt="#2C3239",
    border="#3A4149",
    border_strong="#4C5560",
    text="#E3E6E3",
    text_muted="#A3ABB2",
    text_disabled="#6C757D",
    selection="#34475A",
    tide="#7FA3C4",
    tide_hover="#91B1CE",
    tide_pressed="#6E92B3",
    tide_subtle="#2A3A4A",
    on_tide="#101820",
    moss="#8FAF82",
    moss_strong="#8FAF82",
    moss_subtle="#2E3A2C",
    ochre="#CBA45B",
    ochre_strong="#CBA45B",
    ochre_subtle="#3A3222",
    ember="#C77A6E",
    ember_strong="#C77A6E",
    ember_subtle="#3E2A27",
    slate="#9AA5B0",
    series=("#7FA3C4", "#8FAF82", "#CBA45B", "#C77A6E", "#A394BE", "#78B3B0", "#9AA5B0", "#B49072"),
)

THEMES: dict[bool, ThemeTokens] = {False: SANDBANK, True: DEEPWATER}


# One QSS template for both themes; ``$token`` placeholders are substituted from
# ThemeTokens.  Elevation is canvas < surface < surface_alt, separated by borders.
_STYLESHEET = Template("""
QMainWindow, QDialog, QMenu, QToolTip {
    background-color: $canvas;
}
QWidget {
    font-family: "Segoe UI", "Noto Sans", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    color: $text;
}
QWidget:disabled {
    color: $text_disabled;
}
QLabel[role="muted"] {
    color: $text_muted;
}
QLabel[role="caption"] {
    color: $text_muted;
    font-size: 11px;
}
QLabel[role="heading"] {
    font-size: 15px;
    font-weight: 600;
}
QLabel[role="error"] {
    color: $ember_strong;
}
QToolTip {
    color: $text;
    border: 1px solid $border_strong;
    padding: ${space_1}px ${space_2}px;
}
QMenu {
    border: 1px solid $border;
    padding: ${space_1}px;
}
QMenu::item {
    padding: ${space_2}px ${space_4}px;
    border-radius: ${radius_sm}px;
}
QMenu::item:selected {
    background-color: $selection;
}
QScrollArea, QStackedWidget {
    background-color: transparent;
    border: none;
}
QGroupBox {
    background-color: $surface;
    border: 1px solid $border;
    border-radius: ${radius_lg}px;
    margin-top: ${space_3}px;
    padding-top: ${space_2}px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: ${space_3}px;
    padding: 0 ${space_2}px;
    color: $text_muted;
}
QGroupBox QGroupBox {
    background-color: $canvas;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextBrowser, QListWidget {
    background-color: $surface;
    border: 1px solid $border;
    border-radius: ${radius_md}px;
    padding: 6px 8px;
    selection-background-color: $selection;
    selection-color: $text;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    min-height: ${input_min_h}px;
}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border: 1px solid $border_strong;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QListWidget:focus {
    border: 1px solid $tide;
}
QLineEdit[invalid="true"] {
    border: 1px solid $ember;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    background-color: $surface_alt;
}
QComboBox::drop-down {
    border: none;
    width: ${space_5}px;
}
QComboBox QAbstractItemView {
    background-color: $surface;
    border: 1px solid $border;
    selection-background-color: $selection;
    selection-color: $text;
    outline: none;
}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    border: none;
    width: ${space_4}px;
}
QListWidget::item {
    min-height: 28px;
    padding: 0 ${space_1}px;
}
QListWidget::item:selected, QTableWidget::item:selected {
    background-color: $selection;
    color: $text;
}
QTableWidget {
    background-color: $surface;
    alternate-background-color: $surface;
    border: 1px solid $border;
    border-radius: ${radius_md}px;
    gridline-color: $border;
    outline: none;
}
QHeaderView::section {
    background-color: $surface_alt;
    color: $text_muted;
    border: none;
    border-bottom: 1px solid $border;
    padding: 6px;
    font-weight: 600;
}
QTableCornerButton::section {
    background-color: $surface_alt;
    border: none;
}
QProgressBar {
    background-color: $surface;
    border: 1px solid $border;
    border-radius: ${radius_md}px;
    text-align: center;
    color: $text;
    min-height: ${space_4}px;
}
QProgressBar::chunk {
    background-color: $tide;
    border-radius: ${radius_sm}px;
}
QProgressBar[progressState="paused"]::chunk {
    background-color: $ochre_strong;
}
QProgressBar[progressState="finished"]::chunk {
    background-color: $moss_strong;
}
QCheckBox {
    spacing: ${space_2}px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid $border_strong;
    border-radius: ${radius_sm}px;
    background-color: $surface;
}
QCheckBox::indicator:hover {
    border: 1px solid $tide;
}
QCheckBox::indicator:checked {
    background-color: $tide;
    border: 1px solid $tide;
    $check_image
}
QCheckBox::indicator:disabled {
    background-color: $surface_alt;
    border: 1px solid $border;
}
QTabBar::tab {
    background-color: $surface_alt;
    color: $text_muted;
    border: 1px solid $border;
    border-bottom: none;
    border-top-left-radius: ${radius_md}px;
    border-top-right-radius: ${radius_md}px;
    padding: ${space_2}px ${space_4}px;
    margin-right: 2px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: $surface;
    color: $tide;
    border-color: $tide;
}
QPushButton {
    background-color: $surface_alt;
    color: $text;
    border: 1px solid $border;
    border-radius: ${radius_lg}px;
    padding: ${space_2}px ${space_4}px;
    min-height: ${button_min_h}px;
    font-weight: 600;
}
QPushButton:hover {
    border: 1px solid $border_strong;
}
QPushButton:pressed {
    background-color: $selection;
}
QPushButton:focus {
    border: 1px solid $tide;
}
QPushButton:disabled {
    background-color: $surface_alt;
    color: $text_disabled;
    border: 1px solid $border;
}
QPushButton[flat="true"] {
    background-color: transparent;
    color: $tide;
    border: 1px solid transparent;
}
QPushButton[flat="true"]:hover {
    background-color: $tide_subtle;
}
QPushButton[flat="true"]:pressed {
    background-color: $selection;
}
QPushButton[flat="true"]:disabled {
    background-color: transparent;
    color: $text_disabled;
}
QPushButton[tabButton="true"] {
    background-color: $surface_alt;
    color: $text_muted;
    min-height: ${button_lg_min_h}px;
}
QPushButton[tabButton="true"][tabActive="true"],
QPushButton[tabButton="true"]:disabled {
    background-color: $surface;
    border: 2px solid $tide;
    color: $tide;
}
QPushButton[primaryAction="true"] {
    background-color: $tide;
    color: $on_tide;
    border: none;
    min-height: ${button_lg_min_h}px;
}
QPushButton[primaryAction="true"]:hover {
    background-color: $tide_hover;
}
QPushButton[primaryAction="true"]:pressed {
    background-color: $tide_pressed;
}
QPushButton[primaryAction="true"][actionState="pause"] {
    background-color: $ochre_strong;
}
QPushButton[primaryAction="true"][actionState="resume"] {
    background-color: $moss_strong;
}
QPushButton[primaryAction="true"][actionState="stopping"] {
    background-color: $slate;
}
QPushButton[primaryAction="true"]:disabled {
    background-color: $surface_alt;
    color: $text_disabled;
    border: 1px solid $border;
}
QPushButton[dangerAction="true"] {
    background-color: $ember_strong;
    color: $on_tide;
    border: none;
}
QPushButton[dangerAction="true"]:hover {
    background-color: $ember;
}
QPushButton[dangerAction="true"]:disabled {
    background-color: $surface_alt;
    color: $text_disabled;
    border: 1px solid $border;
}
QDialogButtonBox QPushButton {
    min-width: 88px;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background-color: transparent;
    border: none;
    margin: 0;
}
QScrollBar:vertical {
    width: ${space_3}px;
}
QScrollBar:horizontal {
    height: ${space_3}px;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background-color: $border_strong;
    border-radius: ${radius_sm}px;
    min-height: ${space_5}px;
    min-width: ${space_5}px;
    margin: 2px;
}
QScrollBar::handle:hover {
    background-color: $text_disabled;
}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {
    background: none;
    border: none;
    height: 0;
    width: 0;
}
QSplitter::handle {
    background-color: $border;
}
QStatusBar {
    color: $text_muted;
}
QStatusBar::item {
    border: none;
}
""")

# Check mark drawn in ``on_tide``; QSS cannot colour the native indicator and
# ``url()`` needs a file, so ThemeManager.apply() writes it to the cache directory.
_CHECK_SVG = Template(
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
    "<path fill='none' stroke='$color' stroke-width='3' stroke-linecap='round' "
    "stroke-linejoin='round' d='M5 12.5l4.5 4.5L19 7.5'/></svg>"
)


def build_stylesheet(tokens: ThemeTokens, check_icon: Path | None = None) -> str:
    """The application stylesheet for *tokens*; raises ``KeyError`` on a missing token.

    *check_icon* is the SVG drawn inside a checked checkbox; without it the
    indicator is a plain ``tide`` square.
    """
    values = asdict(tokens)
    values["check_image"] = f"image: url({check_icon.as_posix()});" if check_icon else ""
    # QSS min-height is the content box: subtract the padding and 1 px border
    # declared in the template so controls end up at control_h / control_h_lg.
    values["input_min_h"] = tokens.control_h - 2 * 6 - 2
    values["button_min_h"] = tokens.control_h - 2 * tokens.space_2 - 2
    values["button_lg_min_h"] = tokens.control_h_lg - 2 * tokens.space_2 - 2
    return _STYLESHEET.substitute(values)


def build_palette(tokens: ThemeTokens) -> QPalette:
    """A palette for the parts Qt draws natively (dialogs, combo popups, indicators)."""
    palette = QPalette()
    role = QPalette.ColorRole
    for color_role, value in (
        (role.Window, tokens.canvas),
        (role.WindowText, tokens.text),
        (role.Base, tokens.surface),
        (role.AlternateBase, tokens.surface_alt),
        (role.Text, tokens.text),
        (role.Button, tokens.surface_alt),
        (role.ButtonText, tokens.text),
        (role.ToolTipBase, tokens.canvas),
        (role.ToolTipText, tokens.text),
        (role.PlaceholderText, tokens.text_disabled),
        (role.Highlight, tokens.selection),
        (role.HighlightedText, tokens.text),
        (role.Link, tokens.tide),
        (role.Light, tokens.surface),
        (role.Midlight, tokens.surface_alt),
        (role.Mid, tokens.border_strong),
        (role.Dark, tokens.border_strong),
        (role.Shadow, tokens.border_strong),
    ):
        palette.setColor(color_role, QColor(value))
    disabled = QPalette.ColorGroup.Disabled
    for color_role in (role.WindowText, role.Text, role.ButtonText):
        palette.setColor(disabled, color_role, QColor(tokens.text_disabled))
    return palette


def style_plot(plot: pg.PlotWidget, tokens: ThemeTokens) -> None:
    """Colour a pyqtgraph plot's chrome from *tokens* (series colours are the caller's job)."""
    import pyqtgraph as pg

    plot.setBackground(tokens.surface)
    axis_pen = pg.mkPen(tokens.text_muted)
    for axis_name in ("left", "bottom", "top", "right"):
        axis = plot.getAxis(axis_name)
        axis.setPen(axis_pen)
        axis.setTextPen(axis_pen)
    plot_item = plot.getPlotItem()
    plot_item.showGrid(x=True, y=True, alpha=0.15)
    if plot_item.titleLabel is not None:
        plot_item.titleLabel.setText(plot_item.titleLabel.text, color=tokens.text)
    if plot_item.legend is not None:
        plot_item.legend.setLabelTextColor(tokens.text)
        plot_item.legend.setBrush(QColor(tokens.surface))
        plot_item.legend.setPen(pg.mkPen(tokens.border))


def icon(name: str, tokens: ThemeTokens, *, on_fill: bool = False, tertiary: bool = False) -> QIcon:
    """A Material Design icon (``mdi6.<name>``) coloured for the current theme.

    Icons are ``text_muted`` on surfaces, ``on_tide`` on filled buttons and
    ``tide`` on tertiary (flat) buttons.
    """
    import qtawesome

    if on_fill:
        color = tokens.on_tide
    elif tertiary:
        color = tokens.tide
    else:
        color = tokens.text_muted
    return qtawesome.icon(f"mdi6.{name}", color=color, color_disabled=tokens.text_disabled)


def refresh_style(widget: QWidget) -> None:
    """Re-polish *widget* after one of its dynamic QSS properties changed."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def system_prefers_dark() -> bool:
    app = QGuiApplication.instance()
    if app is None:
        return False
    return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark


class ThemeManager(QObject):
    """Resolves the appearance mode, applies the stylesheet, and announces changes.

    The mode (``system`` / ``light`` / ``dark``) is persisted in
    ``QSettings("COBRA", "COBRA")`` under ``appearance/mode``.  ``system``
    follows the OS colour scheme live; the manual modes ignore it.
    """

    theme_changed = Signal(object)  # ThemeTokens

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._settings = QSettings("COBRA", "COBRA")
        mode = str(self._settings.value("appearance/mode", "system"))
        self._mode = mode if mode in MODES else "system"
        self._tokens = self._resolve()
        app = QGuiApplication.instance()
        if app is not None:
            QGuiApplication.styleHints().colorSchemeChanged.connect(self._on_system_scheme_changed)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def tokens(self) -> ThemeTokens:
        return self._tokens

    def _resolve(self) -> ThemeTokens:
        if self._mode == "system":
            return THEMES[system_prefers_dark()]
        return THEMES[self._mode == "dark"]

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"Unknown appearance mode '{mode}'; expected one of {', '.join(MODES)}")
        self._mode = mode
        self._settings.setValue("appearance/mode", mode)
        self.apply()

    def cycle_mode(self) -> str:
        """Advance system → light → dark → system and return the new mode."""
        self.set_mode(MODES[(MODES.index(self._mode) + 1) % len(MODES)])
        return self._mode

    def apply(self) -> None:
        """Apply the resolved theme to the application and notify listeners."""
        self._tokens = self._resolve()
        app = QApplication.instance()
        if isinstance(app, QApplication):
            app.setPalette(build_palette(self._tokens))
            app.setStyleSheet(build_stylesheet(self._tokens, self._check_icon()))
        self.theme_changed.emit(self._tokens)

    def _check_icon(self) -> Path | None:
        cache_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
        if not cache_dir:
            return None
        path = Path(cache_dir) / f"check-{self._tokens.on_tide.lstrip('#')}.svg"
        try:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(_CHECK_SVG.substitute(color=self._tokens.on_tide))
        except OSError:
            return None
        return path

    def bind_icon(
        self,
        button: QAbstractButton,
        name: str,
        *,
        on_fill: bool = False,
        tertiary: bool = False,
        size: int = 20,
    ) -> None:
        """Give *button* a themed icon now and again whenever the theme changes."""
        button.setIconSize(QSize(size, size))

        def refresh(tokens: ThemeTokens) -> None:
            button.setIcon(icon(name, tokens, on_fill=on_fill, tertiary=tertiary))

        refresh(self._tokens)
        self.theme_changed.connect(refresh)
        button.destroyed.connect(lambda: self._disconnect(refresh))

    def _disconnect(self, slot) -> None:
        # Already disconnected when the application is shutting down.
        with suppress(RuntimeError, TypeError):
            self.theme_changed.disconnect(slot)

    def _on_system_scheme_changed(self, _scheme) -> None:
        if self._mode == "system":
            self.apply()


@cache
def manager() -> ThemeManager:
    """The process-wide theme manager, created on first use."""
    return ThemeManager()
