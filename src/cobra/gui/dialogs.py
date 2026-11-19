import json
import logging
import re
from typing import TYPE_CHECKING

from PySide6.QtCore import QLocale
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QTabBar,
)

from cobra.configuration.configuration import AnalysisVariantConfig, ConfigurationError
from cobra.optimizers.base_optimizer import OptimizationProperty, OptimizationType
from cobra.optimizers.design_goal import (
    DesignGoal,
    DesignParameter,
    ExpressionParameter,
    GoalInputs,
)
from cobra.optimizers.design_goal_collection import make_expression_parameter
from cobra.optimizers.goal_expression import FUNCTIONS, GoalExpression

from .help_texts import tooltip
from .theme import refresh_style

if TYPE_CHECKING:
    from collections.abc import Collection

    from cobra.spice_sim.simulation_type import Analysis, SimulationType

logger = logging.getLogger(__name__)


_FREQUENCY_RE = re.compile(
    r"^\s*(?P<min>\d*\.?\d+)(?:-(?P<max>\d*\.?\d+))?\s*(?P<unit>[a-zA-Z]*)\s*$"
)
_FREQUENCY_UNITS = {"hz": "Hz", "khz": "kHz", "mhz": "MHz", "ghz": "GHz"}

# The goal dialog's tab for an expression over other goals.
_EXPRESSION = "Expression"


def _variable_names(taken: set[str]):
    """Short variable names a, b, ..., z, a1, b1, ... that are not in *taken* or functions."""
    index = 0
    while True:
        name = chr(ord("a") + index % 26) + (str(index // 26) if index >= 26 else "")
        index += 1
        if name not in taken and name not in FUNCTIONS:
            yield name

# Frequency modes of the goal dialog: how the frequency_range string is built.
_FREQ_ANY = "Whole sweep / spectrum"
_FREQ_POINT = "Single frequency"
_FREQ_RANGE = "Frequency range"


def _number_validator(bottom: float, top: float, parent: QDialog,
                      notation: QDoubleValidator.Notation) -> QDoubleValidator:
    """A validator for dot-decimal numbers, independent of the system locale.

    The default locale would accept ``-10,5`` (which ``float()`` cannot parse)
    and reject ``-10.0`` (which ``str(float)`` writes back on a comma-decimal
    system such as ``de_DE``).
    """
    locale = QLocale.c()
    locale.setNumberOptions(QLocale.NumberOption.RejectGroupSeparator)
    validator = QDoubleValidator(bottom, top, 15, parent)
    validator.setNotation(notation)
    validator.setLocale(locale)
    return validator


def _ok_cancel_buttons(dialog: QDialog) -> QDialogButtonBox:
    """OK (primary) and Cancel buttons wired to *dialog*."""
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dialog)
    buttons.button(QDialogButtonBox.StandardButton.Ok).setProperty("primaryAction", True)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    return buttons


class DesignGoalDialog(QDialog):
    """Create or edit a :class:`DesignGoal`.

    The parameters are grouped by the analysis they need, one tab per analysis
    (an analysis variant gets its own), so the parameter list stays short for
    netlists with many ports and nodes. The form adapts to the selected
    parameter's :class:`GoalInputs`: bounds the goal cannot take are hidden, and
    the frequency mode is limited to what the goal supports (a point, a range, or
    the whole sweep). With *term_parameters* (the parameters of the other goals)
    an *Expression* tab offers a goal on a formula over them, e.g. ``a - b``
    (:class:`ExpressionParameter`); *taken_names* are the parameter names an
    expression may not take.
    """

    def __init__(self, parent=None, goal: "DesignGoal | None" = None,
                 available_parameters: "list[DesignParameter] | None" = None,
                 initial_simulation_type: "SimulationType | None" = None,
                 term_parameters: "list[DesignParameter] | None" = None,
                 taken_names: "Collection[str]" = ()):
        super().__init__(parent)
        self.setWindowTitle("Design goal")
        self.setMinimumWidth(400)
        self.form_layout = QFormLayout(self)

        # Parameters per analysis, in order of first appearance.
        self._parameters_by_type: dict[Analysis | str, list[DesignParameter]] = {}
        for dp in available_parameters or []:
            self._parameters_by_type.setdefault(dp.analysis, []).append(dp)
        if (
            goal
            and not isinstance(goal.parameter, ExpressionParameter)
            and goal.parameter not in self._parameters_by_type.get(goal.parameter.analysis, [])
        ):
            # Not offered any more (netlist changed): keep it editable at the top of its tab.
            self._parameters_by_type.setdefault(goal.parameter.analysis, []).insert(0, goal.parameter)

        # Terms an expression can use: the goal's own under their variables, then the
        # parameters of the other goals under new ones.
        self._taken_names = set(taken_names)
        own = goal.parameter.terms if goal and isinstance(goal.parameter, ExpressionParameter) else {}
        self._terms: dict[str, DesignParameter] = dict(own)
        names = _variable_names(set(own))
        for parameter in term_parameters or []:
            if parameter not in self._terms.values():
                self._terms[next(names)] = parameter
        if self._terms:
            self._parameters_by_type[_EXPRESSION] = []

        self.sim_type_tabs = QTabBar()
        self.sim_type_tabs.setToolTip(tooltip("sim_type_tabs"))
        for analysis in self._parameters_by_type:
            self.sim_type_tabs.addTab(analysis if isinstance(analysis, str) else analysis.display_name)
        self.param_combo = QComboBox()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Compression_dB")
        self.name_edit.setToolTip(tooltip("expression_name_edit"))
        self.terms_list = QListWidget()
        self.terms_list.setToolTip(tooltip("expression_terms"))
        self.terms_list.setMaximumHeight(96)
        for variable, term in self._terms.items():
            item = QListWidgetItem(f"{variable} = {term.name}")
            item.setData(256, variable)  # Qt.UserRole
            self.terms_list.addItem(item)
        self.terms_list.itemDoubleClicked.connect(self._insert_variable)
        self.expression_edit = QLineEdit()
        self.expression_edit.setPlaceholderText("e.g. a - b")
        self.expression_edit.setToolTip(tooltip("expression_edit"))

        self.weight_edit = QLineEdit()
        self.weight_edit.setPlaceholderText("Default: 1.0")
        self.weight_edit.setText("1.0")
        # Scientific notation, as str(float) writes small values like 1e-05.
        self.weight_edit.setValidator(
            _number_validator(0.0, 1e15, self, QDoubleValidator.Notation.ScientificNotation)
        )

        self.scale_edit = QLineEdit()
        self.scale_edit.setPlaceholderText("Default: relative to the limit")
        self.scale_edit.setToolTip(tooltip("goal_scale_edit"))
        self.scale_edit.setValidator(
            _number_validator(0.0, 1e15, self, QDoubleValidator.Notation.ScientificNotation)
        )

        self.min_edit = QLineEdit()
        self.max_edit = QLineEdit()
        self.min_edit.setPlaceholderText("Optional")
        self.max_edit.setPlaceholderText("Optional")
        value_validator = _number_validator(-1e15, 1e15, self, QDoubleValidator.Notation.ScientificNotation)
        self.min_edit.setValidator(value_validator)
        self.max_edit.setValidator(value_validator)

        self.freq_mode_combo = QComboBox()
        self.freq_mode_combo.setToolTip(tooltip("freq_mode_combo"))
        self.freq_min_label = QLabel("Frequency")
        self.freq_min_edit = QLineEdit()
        self.freq_max_edit = QLineEdit()
        # Standard notation: _FREQUENCY_RE does not parse exponents.
        freq_validator = _number_validator(0.0, 1e15, self, QDoubleValidator.Notation.StandardNotation)
        self.freq_min_edit.setValidator(freq_validator)
        self.freq_max_edit.setValidator(freq_validator)
        self.freq_unit_combo = QComboBox()
        self.freq_unit_combo.addItems(list(_FREQUENCY_UNITS.values()))
        self.freq_unit_combo.setCurrentText("GHz")
        self.freq_unit_combo.setToolTip(tooltip("freq_unit_combo"))

        # Validation problems are shown here, next to the field they concern.
        self.error_label = QLabel()
        self.error_label.setProperty("role", "error")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)

        self.form_layout.addRow(self.sim_type_tabs)
        self.form_layout.addRow("Parameter", self.param_combo)
        self.form_layout.addRow("Name", self.name_edit)
        self.form_layout.addRow("Terms", self.terms_list)
        self.form_layout.addRow("Expression", self.expression_edit)
        self.form_layout.addRow("Weight", self.weight_edit)
        self.form_layout.addRow("Scale", self.scale_edit)
        self.form_layout.addRow("Min value", self.min_edit)
        self.form_layout.addRow("Max value", self.max_edit)
        self.form_layout.addRow("Frequency", self.freq_mode_combo)
        self.form_layout.addRow(self.freq_min_label, self.freq_min_edit)
        self.form_layout.addRow("Max frequency", self.freq_max_edit)
        self.form_layout.addRow("Frequency unit", self.freq_unit_combo)
        self.form_layout.addRow(self.error_label)

        self.buttons = _ok_cancel_buttons(self)
        self.form_layout.addRow(self.buttons)
        for edit in self._edits:
            edit.textChanged.connect(self._clear_error)

        self.sim_type_tabs.currentChanged.connect(self._on_sim_type_changed)
        self.param_combo.currentIndexChanged.connect(self._on_parameter_changed)
        self.freq_mode_combo.currentIndexChanged.connect(self._on_frequency_mode_changed)

        sim_types = list(self._parameters_by_type)
        selected: Analysis | str | None = initial_simulation_type
        if goal and isinstance(goal.parameter, ExpressionParameter):
            selected = _EXPRESSION
        elif goal:
            selected = goal.parameter.analysis
        self.sim_type_tabs.setCurrentIndex(sim_types.index(selected) if selected in sim_types else 0)
        self._on_sim_type_changed()
        if goal and isinstance(goal.parameter, ExpressionParameter):
            self.name_edit.setText(goal.parameter.name)
            if goal.parameter.expression is not None:
                self.expression_edit.setText(goal.parameter.expression.text)
        elif goal:
            self.param_combo.setCurrentIndex(self.param_combo.findText(goal.parameter.name))

        if goal:
            if goal.min_value is not None:
                self.min_edit.setText(str(goal.min_value))
            if goal.max_value is not None:
                self.max_edit.setText(str(goal.max_value))
            if goal.weight is not None:
                self.weight_edit.setText(str(goal.weight))
            if goal.scale is not None:
                self.scale_edit.setText(str(goal.scale))
            self._set_frequency(goal.frequency_range)

    def _is_expression(self) -> bool:
        """Whether the *Expression* tab is selected."""
        sim_types = list(self._parameters_by_type)
        index = self.sim_type_tabs.currentIndex()
        return 0 <= index < len(sim_types) and sim_types[index] == _EXPRESSION

    def _insert_variable(self, item: QListWidgetItem) -> None:
        """Put the double-clicked term's variable into the expression at the cursor."""
        self.expression_edit.insert(str(item.data(256)))
        self.expression_edit.setFocus()

    def _on_sim_type_changed(self):
        """Offer the parameters of the analysis selected in the tab bar."""
        sim_types = list(self._parameters_by_type)
        index = self.sim_type_tabs.currentIndex()
        parameters = self._parameters_by_type[sim_types[index]] if 0 <= index < len(sim_types) else []
        expression = self._is_expression()
        self.form_layout.setRowVisible(self.param_combo, not expression)
        for widget in (self.name_edit, self.terms_list, self.expression_edit):
            self.form_layout.setRowVisible(widget, expression)
        self.param_combo.blockSignals(True)
        self.param_combo.clear()
        for dp in parameters:
            self.param_combo.addItem(dp.name, dp)
            if dp.description:
                self.param_combo.setItemData(self.param_combo.count() - 1, dp.description, 3)  # Qt.ToolTipRole = 3
        self.param_combo.blockSignals(False)
        self._on_parameter_changed()

    def _current_parameter(self) -> "DesignParameter | None":
        param = self.param_combo.currentData()
        return param if isinstance(param, DesignParameter) else None

    def _current_inputs(self) -> GoalInputs:
        """The inputs a goal on the current selection takes; an expression takes any."""
        param = self._current_parameter()
        return param.inputs if param and not self._is_expression() else GoalInputs()

    def _expression_parameter(self) -> ExpressionParameter:
        """The expression typed in over the terms it uses; raises ValueError if it is not valid."""
        expression = GoalExpression(self.expression_edit.text())
        unknown = [variable for variable in expression.variables if variable not in self._terms]
        if unknown:
            raise ValueError(f"{', '.join(unknown)} is not a term; use the variables listed above.")
        terms = {variable: self._terms[variable] for variable in expression.variables}
        return make_expression_parameter(self.name_edit.text().strip(), expression, terms)

    def _on_parameter_changed(self):
        """Show the inputs the selected parameter's goals take."""
        inputs = self._current_inputs()

        self.form_layout.setRowVisible(self.max_edit, inputs.max_value)
        self.min_edit.setPlaceholderText("Required" if not inputs.max_value else "Optional")

        modes = [] if inputs.frequency_required else [_FREQ_ANY]
        modes.append(_FREQ_POINT)
        if not inputs.single_frequency:
            modes.append(_FREQ_RANGE)
        current = self.freq_mode_combo.currentText()
        self.freq_mode_combo.blockSignals(True)
        self.freq_mode_combo.clear()
        self.freq_mode_combo.addItems(modes)
        self.freq_mode_combo.blockSignals(False)
        self.freq_mode_combo.setCurrentText(current if current in modes else modes[0])
        self._on_frequency_mode_changed()

    def _on_frequency_mode_changed(self):
        mode = self.freq_mode_combo.currentText()
        self.form_layout.setRowVisible(self.freq_min_edit, mode != _FREQ_ANY)
        self.form_layout.setRowVisible(self.freq_max_edit, mode == _FREQ_RANGE)
        self.form_layout.setRowVisible(self.freq_unit_combo, mode != _FREQ_ANY)
        self.freq_min_label.setText("Frequency" if mode == _FREQ_POINT else "Min frequency")

    def _set_frequency(self, frequency_range: str | None):
        """Fill the frequency inputs from a goal's ``frequency_range`` string."""
        match = _FREQUENCY_RE.match(frequency_range or "")
        if not match:
            self.freq_mode_combo.setCurrentText(_FREQ_ANY)
            return
        self.freq_min_edit.setText(match.group("min"))
        if match.group("max"):
            self.freq_mode_combo.setCurrentText(_FREQ_RANGE)
            self.freq_max_edit.setText(match.group("max"))
        else:
            self.freq_mode_combo.setCurrentText(_FREQ_POINT)
        unit = _FREQUENCY_UNITS.get(match.group("unit").lower())
        if unit is not None:
            self.freq_unit_combo.setCurrentText(unit)

    @property
    def _edits(self) -> tuple[QLineEdit, ...]:
        return (
            self.name_edit, self.expression_edit, self.weight_edit, self.scale_edit,
            self.min_edit, self.max_edit, self.freq_min_edit, self.freq_max_edit,
        )

    def _validate(self) -> tuple[str, QLineEdit | None] | None:
        """The first problem with the current inputs and the field it concerns, or ``None``."""
        expression = None
        if self._is_expression():
            name = self.name_edit.text().strip()
            if not name:
                return "Give the expression a name.", self.name_edit
            if name in self._taken_names:
                return f"Another goal reads {name}; choose another name.", self.name_edit
            try:
                expression = self._expression_parameter()
            except ValueError as exc:
                return str(exc), self.expression_edit
        elif self._current_parameter() is None:
            return "No valid parameter selected.", None
        inputs = self._current_inputs()
        min_text = self.min_edit.text().strip()
        max_text = self.max_edit.text().strip() if inputs.max_value else ""
        if min_text and not self.min_edit.hasAcceptableInput():
            return "Min value must be numeric.", self.min_edit
        if max_text and not self.max_edit.hasAcceptableInput():
            return "Max value must be numeric.", self.max_edit
        if self.weight_edit.text().strip() and not self.weight_edit.hasAcceptableInput():
            return "Weight must be numeric.", self.weight_edit
        scale_text = self.scale_edit.text().strip()
        if scale_text and not (self.scale_edit.hasAcceptableInput() and float(scale_text) > 0):
            return "Scale must be a positive number.", self.scale_edit
        if not min_text and not max_text:
            if inputs.max_value:
                return "At least one of min value or max value must be set.", self.min_edit
            return "Min value must be set.", self.min_edit
        if min_text and max_text and float(min_text) > float(max_text):
            return "Min value must not exceed max value.", self.max_edit

        mode = self.freq_mode_combo.currentText()
        freq_min = self.freq_min_edit.text().strip()
        freq_max = self.freq_max_edit.text().strip()
        if mode == _FREQ_POINT and not freq_min:
            return "A frequency is required.", self.freq_min_edit
        if mode == _FREQ_RANGE:
            if not freq_min:
                return "Both min and max frequency are required for a range.", self.freq_min_edit
            if not freq_max:
                return "Both min and max frequency are required for a range.", self.freq_max_edit
            if float(freq_min) >= float(freq_max):
                return "Min frequency must be below max frequency.", self.freq_max_edit
        if expression is not None and len(expression.required_analyses) > 1 and mode != _FREQ_POINT:
            return "Terms from several analyses need a single frequency.", self.freq_min_edit
        return None

    def _show_error(self, message: str, field: QLineEdit | None) -> None:
        self.error_label.setText(message)
        self.error_label.setVisible(True)
        if field is not None:
            field.setProperty("invalid", True)
            refresh_style(field)
            field.setFocus()

    def _clear_error(self) -> None:
        self.error_label.setVisible(False)
        for edit in self._edits:
            if edit.property("invalid"):
                edit.setProperty("invalid", False)
                refresh_style(edit)

    def _frequency_range(self) -> str | None:
        mode = self.freq_mode_combo.currentText()
        if mode == _FREQ_ANY:
            return None
        unit = self.freq_unit_combo.currentText().lower()
        if mode == _FREQ_POINT:
            return f"{self.freq_min_edit.text().strip()}{unit}"
        return f"{self.freq_min_edit.text().strip()}-{self.freq_max_edit.text().strip()}{unit}"

    def get_data(self) -> DesignGoal:
        problem = self._validate()
        if problem:
            raise ValueError(problem[0])
        param = self._expression_parameter() if self._is_expression() else self._current_parameter()

        min_val = float(self.min_edit.text()) if self.min_edit.text().strip() else None
        max_val = (
            float(self.max_edit.text())
            if self._current_inputs().max_value and self.max_edit.text().strip()
            else None
        )
        try:
            weight = float(self.weight_edit.text())
        except ValueError:
            weight = 1.0

        scale = float(self.scale_edit.text()) if self.scale_edit.text().strip() else None

        if param is None:
            raise ValueError("No valid parameter selected.")
        return DesignGoal(parameter=param, frequency_range=self._frequency_range(),
                          min_value=min_val, max_value=max_val, weight=weight, scale=scale)

    def accept(self):
        problem = self._validate()
        if problem:
            self._show_error(*problem)
            return
        super().accept()


class AnalysisVariantDialog(QDialog):
    """Create or edit an :class:`AnalysisVariantConfig`: an analysis on the netlist with values changed.

    *analyses* are the analysis types the simulator runs, *netlist_names* the
    elements and instances a value may name, *taken_names* the names of the other
    variants. A variant goals read (*in_use*) keeps its name and analysis.
    """

    def __init__(
        self,
        parent=None,
        variant: AnalysisVariantConfig | None = None,
        analyses: "Collection[str]" = (),
        netlist_names: "Collection[str]" = (),
        taken_names: "Collection[str]" = (),
        in_use: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle("Analysis variant")
        self.setMinimumWidth(400)
        self._netlist_names = {name.casefold() for name in netlist_names}
        self._taken_names = set(taken_names)
        self.form_layout = QFormLayout(self)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. HB_m40")
        self.name_edit.setToolTip(tooltip("variant_name_edit"))
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItems(list(analyses))
        self.analysis_combo.setToolTip(tooltip("variant_analysis_combo"))
        self.values_edit = QLineEdit()
        self.values_edit.setPlaceholderText("e.g. vp_rf:ampl=0.00632, R1=60")
        self.values_edit.setToolTip(tooltip("variant_values_edit"))
        if variant is not None:
            self.name_edit.setText(variant.name)
            self.analysis_combo.setCurrentText(variant.analysis.upper())
            self.values_edit.setText(
                ", ".join(f"{key}={value}" for key, value in variant.netlist_values.items())
            )
        if in_use:
            # Goals read the variant by name and parameters of its analysis.
            self.name_edit.setEnabled(False)
            self.analysis_combo.setEnabled(False)

        self.error_label = QLabel()
        self.error_label.setProperty("role", "error")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)

        self.form_layout.addRow("Name", self.name_edit)
        self.form_layout.addRow("Analysis", self.analysis_combo)
        self.form_layout.addRow("Netlist values", self.values_edit)
        self.form_layout.addRow(self.error_label)
        self.buttons = _ok_cancel_buttons(self)
        self.form_layout.addRow(self.buttons)
        for edit in (self.name_edit, self.values_edit):
            edit.textChanged.connect(self._clear_error)

    def _values(self) -> dict[str, float | str]:
        """The netlist values typed in, numbers as numbers; raises ValueError on a malformed entry."""
        values: dict[str, float | str] = {}
        for entry in filter(None, (item.strip() for item in self.values_edit.text().split(","))):
            name, separator, value = (part.strip() for part in entry.partition("="))
            if not separator or not name or not value:
                raise ValueError(f"'{entry}' is not of the form name=value.")
            try:
                values[name] = float(value)
            except ValueError:
                values[name] = value
        return values

    def _validate(self) -> tuple[str, QLineEdit | None] | None:
        """The first problem with the current inputs and the field it concerns, or ``None``."""
        try:
            config = self._config()
        except ValueError as exc:
            return str(exc), self.values_edit
        if config.name in self._taken_names:
            return f"A variant named {config.name} exists already.", self.name_edit
        try:
            config.validate()
        except ConfigurationError as exc:
            message = str(exc)
            return message, self.name_edit if message.startswith("Analysis variant name") else self.values_edit
        for name in config.netlist_values:
            element = name.split(":", 1)[0]
            if element.casefold() not in self._netlist_names:
                return f"The netlist has no element or instance '{element}'.", self.values_edit
        return None

    def _config(self) -> AnalysisVariantConfig:
        return AnalysisVariantConfig(
            name=self.name_edit.text().strip(),
            analysis=self.analysis_combo.currentText(),
            netlist_values=self._values(),
        )

    def _show_error(self, message: str, field: QLineEdit | None) -> None:
        self.error_label.setText(message)
        self.error_label.setVisible(True)
        if field is not None:
            field.setProperty("invalid", True)
            refresh_style(field)
            field.setFocus()

    def _clear_error(self) -> None:
        self.error_label.setVisible(False)
        for edit in (self.name_edit, self.values_edit):
            if edit.property("invalid"):
                edit.setProperty("invalid", False)
                refresh_style(edit)

    def get_data(self) -> AnalysisVariantConfig:
        problem = self._validate()
        if problem:
            raise ValueError(problem[0])
        return self._config()

    def accept(self):
        problem = self._validate()
        if problem:
            self._show_error(*problem)
            return
        super().accept()


class OptimizationParamDialog(QDialog):
    def __init__(
        self,
        from_source=None,
        source_data=None,
        parent=None,
        param: OptimizationProperty | None = None,
        metadata: dict | None = None,
        link_candidates: list[str] | None = None,
    ):
        super().__init__(parent)
        self.metadata = metadata or {}
        self.link_candidates = link_candidates or []
        self.setWindowTitle("Optimization parameter")
        self.form_layout = QFormLayout(self)

        self.name_edit = QLineEdit()
        self.type_combo = QComboBox()
        self.type_combo.addItems([t.value for t in OptimizationType])

        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(-1e15, 1e15)
        self.min_spin.setDecimals(15)

        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(-1e15, 1e15)
        self.max_spin.setDecimals(15)

        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0, 1e15)
        self.step_spin.setDecimals(15)

        self.unit_edit = QLineEdit()
        self.unit_edit.setPlaceholderText("Optional (e.g. F (=femto)) for Xyce")
        # Its minimum stands for "no start value": the optimizer chooses the first point.
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(-1e15, 1e15)
        self.start_spin.setDecimals(15)
        self.start_spin.setSpecialValueText("None")
        self.start_spin.setValue(self.start_spin.minimum())
        self.start_spin.setToolTip(tooltip("param_start_spin"))
        self.link_to_combo = QComboBox()
        self.link_to_combo.addItem("None", None)

        if param:
            self.name_edit.setText(param.name)
            self.type_combo.setCurrentText(param.type.value)
            self.min_spin.setValue(param.min_value)
            self.max_spin.setValue(param.max_value)
            if param.step:
                self.step_spin.setValue(param.step)
            if param.unit:
                self.unit_edit.setText(param.unit)
            if param.initial_value is not None:
                self.start_spin.setValue(param.initial_value)
            if param.linked_to:
                self.link_to_combo.addItem(param.linked_to, param.linked_to)
                self.link_to_combo.setCurrentIndex(self.link_to_combo.count() - 1)

        if from_source == "ONNX" and source_data:
            self.name_combo = QComboBox()
            self.name_combo.addItems(source_data)
            self.form_layout.addRow("Name", self.name_combo)
            self.type_combo.setCurrentText(OptimizationType.MODEL_INPUT.value)
            self.type_combo.setEnabled(False)
            self.use_combo_name = True

            self.step_spin.setValue(0.1)
            self.name_combo.currentTextChanged.connect(self._update_onnx_metadata)
            self._update_onnx_metadata(self.name_combo.currentText())
            self.name_combo.currentTextChanged.connect(self._refresh_link_targets)
        elif from_source == "NETLIST" and source_data:
            self.name_combo = QComboBox()
            self.name_combo.addItems(source_data)
            self.form_layout.addRow("Name", self.name_combo)
            self.type_combo.setCurrentText(OptimizationType.NETLIST_VARIABLE.value)
            self.type_combo.setEnabled(False)
            self.step_spin.setValue(1.0)
            self.use_combo_name = True
            self.name_combo.currentTextChanged.connect(self._refresh_link_targets)
        else:
            self.form_layout.addRow("Name", self.name_edit)
            self.use_combo_name = False
            self.name_edit.textChanged.connect(self._refresh_link_targets)

        self.form_layout.addRow("Type", self.type_combo)
        self.form_layout.addRow("Min", self.min_spin)
        self.form_layout.addRow("Max", self.max_spin)
        self.form_layout.addRow("Step", self.step_spin)
        self.form_layout.addRow("Unit", self.unit_edit)
        self.form_layout.addRow("Start value", self.start_spin)
        self.form_layout.addRow("Link to", self.link_to_combo)

        self._refresh_link_targets(self.name_combo.currentText() if self.use_combo_name else self.name_edit.text())
        if param and param.linked_to:
            idx = self.link_to_combo.findData(param.linked_to)
            if idx >= 0:
                self.link_to_combo.setCurrentIndex(idx)

        self.link_to_combo.currentIndexChanged.connect(self._on_link_target_changed)
        self._on_link_target_changed()

        self.buttons = _ok_cancel_buttons(self)
        self.form_layout.addRow(self.buttons)

    def _update_onnx_metadata(self, name):
        # Allow checking metadata natively and without prefix (e.g. 'W' from 'X1:W')
        base_name = name.split(":", 1)[1] if ":" in name else name

        if "input_parameter_ranges" in self.metadata:
            try:
                meta_data = json.loads(self.metadata["input_parameter_ranges"])
                params = None
                if name in meta_data:
                    params = meta_data[name]
                elif base_name in meta_data:
                    params = meta_data[base_name]

                if params:
                    if "min" in params:
                        self.min_spin.setValue(float(params["min"]))
                    if "max" in params:
                        self.max_spin.setValue(float(params["max"]))
                    if "step" in params:
                        self.step_spin.setValue(float(params["step"]))
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning("Could not read ONNX metadata: %s", e)

    def _refresh_link_targets(self, current_name):
        current_name = (current_name or "").strip()
        selected = self.link_to_combo.currentData()
        self.link_to_combo.blockSignals(True)
        self.link_to_combo.clear()
        self.link_to_combo.addItem("None", None)
        for candidate in self.link_candidates:
            if candidate != current_name:
                self.link_to_combo.addItem(candidate, candidate)
        if selected is not None:
            idx = self.link_to_combo.findData(selected)
            if idx >= 0:
                self.link_to_combo.setCurrentIndex(idx)
        self.link_to_combo.blockSignals(False)
        self._on_link_target_changed()

    def _on_link_target_changed(self):
        linked = self.link_to_combo.currentData() is not None
        self.min_spin.setEnabled(not linked)
        self.max_spin.setEnabled(not linked)
        self.step_spin.setEnabled(not linked)
        self.unit_edit.setEnabled(not linked)
        self.start_spin.setEnabled(not linked)

    def get_data(self):
        name = self.name_combo.currentText() if self.use_combo_name else self.name_edit.text()
        t_str = self.type_combo.currentText()
        t = OptimizationType(t_str)
        unit = self.unit_edit.text().strip() or None
        linked_to = self.link_to_combo.currentData()
        start = self.start_spin.value()
        return OptimizationProperty(
            name,
            t,
            self.min_spin.value(),
            self.max_spin.value(),
            self.step_spin.value() or None,
            unit=unit,
            linked_to=linked_to,
            initial_value=None if linked_to or start == self.start_spin.minimum() else start,
        )

