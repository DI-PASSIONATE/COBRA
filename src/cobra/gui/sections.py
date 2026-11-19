"""Collapsible sections for the configuration panel.

A section is a header row (chevron, title, and a one-line summary shown while it
is collapsed) over a form. Rarely changed settings go into the section's own
*Advanced* fold, so the panel shows what a run will do without listing every knob.
Sections are flat header rows, not group boxes: the panel already sits in one, and
the analysis settings inside a section are group boxes themselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .theme import icon

if TYPE_CHECKING:
    from .theme import ThemeManager, ThemeTokens


class _Fold(QWidget):
    """A chevron button that shows or hides a body widget."""

    toggled = Signal(bool)

    def __init__(self, text: str, body: QWidget, theme: ThemeManager, role: str, expanded: bool):
        super().__init__()
        self._body = body
        self._theme = theme
        self.button = QPushButton(text)
        self.button.setProperty(role, True)
        self.button.setCheckable(True)
        self.button.setChecked(expanded)
        # A scroll area shrinks its content to the minimum size, which would clip the text.
        self.button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.button.toggled.connect(self._on_toggled)
        theme.theme_changed.connect(self._refresh_icon)
        self._body.setVisible(expanded)
        self._refresh_icon(theme.tokens)

    def _on_toggled(self, expanded: bool) -> None:
        self._body.setVisible(expanded)
        self._refresh_icon(self._theme.tokens)
        self.toggled.emit(expanded)

    def _refresh_icon(self, tokens: ThemeTokens) -> None:
        name = "chevron-down" if self.button.isChecked() else "chevron-right"
        self.button.setIcon(icon(name, tokens))


class CollapsibleSection(QWidget):
    """A titled, collapsible part of the configuration panel.

    Rows go into :attr:`form`; rarely changed ones into :attr:`advanced_form`, which
    the section shows behind its *Advanced* fold once it has a row (see
    :meth:`update_advanced`). :meth:`set_summary` sets the text the header shows
    while the section is collapsed.
    """

    toggled = Signal(bool)

    def __init__(self, title: str, theme: ThemeManager, expanded: bool = True):
        super().__init__()
        layout = QVBoxLayout(self)
        tokens = theme.tokens
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(tokens.space_2)

        self.header = QFrame()
        self.header.setProperty("sectionHeader", True)
        self.header.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(0, tokens.space_1, 0, tokens.space_1)
        header_layout.setSpacing(tokens.space_3)
        self.summary_label = QLabel()
        self.summary_label.setProperty("role", "muted")

        self.body = QWidget()
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(tokens.space_4, 0, 0, tokens.space_2)
        body_layout.setSpacing(tokens.space_2)
        self.form = QFormLayout()
        body_layout.addLayout(self.form)

        self.advanced_body = QWidget()
        self.advanced_form = QFormLayout(self.advanced_body)
        self.advanced_form.setContentsMargins(tokens.space_4, 0, 0, 0)
        self._advanced = _Fold("Advanced", self.advanced_body, theme, "advancedToggle", expanded=False)
        body_layout.addWidget(self._advanced.button, alignment=Qt.AlignmentFlag.AlignLeft)
        body_layout.addWidget(self.advanced_body)

        self._fold = _Fold(title, self.body, theme, "sectionToggle", expanded)
        self._fold.toggled.connect(self._on_toggled)
        header_layout.addWidget(self._fold.button)
        self._header_widgets = QHBoxLayout()
        self._header_widgets.setSpacing(tokens.space_2)
        header_layout.addLayout(self._header_widgets)
        header_layout.addWidget(self.summary_label)
        header_layout.addStretch(1)  # keeps the header's items left when the summary is hidden

        layout.addWidget(self.header)
        layout.addWidget(self.body)
        self.update_advanced()
        self._on_toggled(expanded)

    @property
    def expanded(self) -> bool:
        return self._fold.button.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self._fold.button.setChecked(expanded)

    @property
    def advanced_expanded(self) -> bool:
        return self._advanced.button.isChecked()

    def set_advanced_expanded(self, expanded: bool) -> None:
        self._advanced.button.setChecked(expanded)

    def add_header_widget(self, widget: QWidget) -> None:
        """Put *widget* (e.g. an on/off checkbox) in the header, visible while collapsed."""
        self._header_widgets.addWidget(widget)

    def set_summary(self, text: str) -> None:
        self.summary_label.setText(text)
        self.summary_label.setToolTip(text)

    def update_advanced(self) -> None:
        """Offer the *Advanced* fold only when it holds a row."""
        self._advanced.button.setVisible(self.advanced_form.rowCount() > 0)
        self.advanced_body.setVisible(self.advanced_form.rowCount() > 0 and self.advanced_expanded)

    def forms(self) -> tuple[QFormLayout, QFormLayout]:
        return self.form, self.advanced_form

    def _on_toggled(self, expanded: bool) -> None:
        self.summary_label.setVisible(not expanded)
        self.toggled.emit(expanded)
