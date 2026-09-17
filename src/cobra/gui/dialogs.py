import json
import logging
import re
from typing import TYPE_CHECKING

from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTabBar,
)

from cobra.optimizers.base_optimizer import OptimizationProperty, OptimizationType
from cobra.optimizers.design_goal import DesignGoal, DesignParameter, GoalInputs

from .help_texts import tooltip

if TYPE_CHECKING:
    from cobra.spice_sim.simulation_type import SimulationType

logger = logging.getLogger(__name__)


_FREQUENCY_RE = re.compile(
    r"^\s*(?P<min>\d*\.?\d+)(?:-(?P<max>\d*\.?\d+))?\s*(?P<unit>[a-zA-Z]*)\s*$"
)
_FREQUENCY_UNITS = {"hz": "Hz", "khz": "kHz", "mhz": "MHz", "ghz": "GHz"}

# Frequency modes of the goal dialog: how the frequency_range string is built.
_FREQ_ANY = "Whole sweep / spectrum"
_FREQ_POINT = "Single frequency"
_FREQ_RANGE = "Frequency range"


class DesignGoalDialog(QDialog):
    """Create or edit a :class:`DesignGoal`.

    The parameters are grouped by the analysis they need, one tab per analysis,
    so the parameter list stays short for netlists with many ports and nodes.
    The form adapts to the selected parameter's :class:`GoalInputs`: bounds the
    goal cannot take are hidden, and the frequency mode is limited to what the
    goal supports (a point, a range, or the whole sweep).
    """

    def __init__(self, parent=None, goal: "DesignGoal | None" = None,
                 available_parameters: "list[DesignParameter] | None" = None,
                 initial_simulation_type: "SimulationType | None" = None):
        super().__init__(parent)
        self.setWindowTitle("Design Goal")
        self.setMinimumWidth(400)
        self.form_layout = QFormLayout(self)

        # Parameters per analysis, in order of first appearance.
        self._parameters_by_type: dict[SimulationType, list[DesignParameter]] = {}
        for dp in available_parameters or []:
            self._parameters_by_type.setdefault(dp.simulation_type, []).append(dp)
        if goal and goal.parameter not in self._parameters_by_type.get(goal.parameter.simulation_type, []):
            # Not offered any more (netlist changed): keep it editable at the top of its tab.
            self._parameters_by_type.setdefault(goal.parameter.simulation_type, []).insert(0, goal.parameter)

        self.sim_type_tabs = QTabBar()
        self.sim_type_tabs.setToolTip(tooltip("sim_type_tabs"))
        for sim_type in self._parameters_by_type:
            self.sim_type_tabs.addTab(sim_type.value)
        self.param_combo = QComboBox()

        self.weight_edit = QLineEdit()
        self.weight_edit.setPlaceholderText("Default: 1.0")
        self.weight_edit.setText("1.0")
        weight_validator = QDoubleValidator(0.0, 1e15, 15, self)
        weight_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.weight_edit.setValidator(weight_validator)

        self.min_edit = QLineEdit()
        self.max_edit = QLineEdit()
        self.min_edit.setPlaceholderText("Optional")
        self.max_edit.setPlaceholderText("Optional")
        value_validator = QDoubleValidator(-1e15, 1e15, 15, self)
        value_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.min_edit.setValidator(value_validator)
        self.max_edit.setValidator(value_validator)

        self.freq_mode_combo = QComboBox()
        self.freq_mode_combo.setToolTip(tooltip("freq_mode_combo"))
        self.freq_min_label = QLabel("Frequency:")
        self.freq_min_edit = QLineEdit()
        self.freq_max_edit = QLineEdit()
        freq_validator = QDoubleValidator(0.0, 1e15, 15, self)
        freq_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.freq_min_edit.setValidator(freq_validator)
        self.freq_max_edit.setValidator(freq_validator)
        self.freq_unit_combo = QComboBox()
        self.freq_unit_combo.addItems(list(_FREQUENCY_UNITS.values()))
        self.freq_unit_combo.setCurrentText("GHz")
        self.freq_unit_combo.setToolTip(tooltip("freq_unit_combo"))

        self.form_layout.addRow(self.sim_type_tabs)
        self.form_layout.addRow("Parameter:", self.param_combo)
        self.form_layout.addRow("Weight:", self.weight_edit)
        self.form_layout.addRow("Min Value:", self.min_edit)
        self.form_layout.addRow("Max Value:", self.max_edit)
        self.form_layout.addRow("Frequency:", self.freq_mode_combo)
        self.form_layout.addRow(self.freq_min_label, self.freq_min_edit)
        self.form_layout.addRow("Max Frequency:", self.freq_max_edit)
        self.form_layout.addRow("Frequency Unit:", self.freq_unit_combo)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.form_layout.addRow(self.buttons)

        self.sim_type_tabs.currentChanged.connect(self._on_sim_type_changed)
        self.param_combo.currentIndexChanged.connect(self._on_parameter_changed)
        self.freq_mode_combo.currentIndexChanged.connect(self._on_frequency_mode_changed)

        sim_types = list(self._parameters_by_type)
        selected = goal.parameter.simulation_type if goal else initial_simulation_type
        self.sim_type_tabs.setCurrentIndex(sim_types.index(selected) if selected in sim_types else 0)
        self._on_sim_type_changed()
        if goal:
            self.param_combo.setCurrentIndex(self.param_combo.findText(goal.parameter.name))

        if goal:
            if goal.min_value is not None:
                self.min_edit.setText(str(goal.min_value))
            if goal.max_value is not None:
                self.max_edit.setText(str(goal.max_value))
            if goal.weight is not None:
                self.weight_edit.setText(str(goal.weight))
            self._set_frequency(goal.frequency_range)

    def _on_sim_type_changed(self):
        """Offer the parameters of the analysis selected in the tab bar."""
        sim_types = list(self._parameters_by_type)
        index = self.sim_type_tabs.currentIndex()
        parameters = self._parameters_by_type[sim_types[index]] if 0 <= index < len(sim_types) else []
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

    def _on_parameter_changed(self):
        """Show the inputs the selected parameter's goals take."""
        param = self._current_parameter()
        inputs = param.inputs if param else GoalInputs()

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
        self.freq_min_label.setText("Frequency:" if mode == _FREQ_POINT else "Min Frequency:")

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

    def _validate(self) -> str | None:
        """The first problem with the current inputs, or ``None`` when they are valid."""
        param = self._current_parameter()
        if param is None:
            return "No valid parameter selected."
        min_text = self.min_edit.text().strip()
        max_text = self.max_edit.text().strip() if param.inputs.max_value else ""
        if min_text and not self.min_edit.hasAcceptableInput():
            return "Min Value must be numeric."
        if max_text and not self.max_edit.hasAcceptableInput():
            return "Max Value must be numeric."
        if self.weight_edit.text().strip() and not self.weight_edit.hasAcceptableInput():
            return "Weight must be numeric."
        if not min_text and not max_text:
            if param.inputs.max_value:
                return "At least one of Min Value or Max Value must be set."
            return "Min Value must be set."
        if min_text and max_text and float(min_text) > float(max_text):
            return "Min Value must not exceed Max Value."

        mode = self.freq_mode_combo.currentText()
        freq_min = self.freq_min_edit.text().strip()
        freq_max = self.freq_max_edit.text().strip()
        if mode == _FREQ_POINT and not freq_min:
            return "A frequency is required."
        if mode == _FREQ_RANGE:
            if not (freq_min and freq_max):
                return "Both Min Frequency and Max Frequency are required for a range."
            if float(freq_min) >= float(freq_max):
                return "Min Frequency must be below Max Frequency."
        return None

    def _frequency_range(self) -> str | None:
        mode = self.freq_mode_combo.currentText()
        if mode == _FREQ_ANY:
            return None
        unit = self.freq_unit_combo.currentText().lower()
        if mode == _FREQ_POINT:
            return f"{self.freq_min_edit.text().strip()}{unit}"
        return f"{self.freq_min_edit.text().strip()}-{self.freq_max_edit.text().strip()}{unit}"

    def get_data(self) -> DesignGoal:
        param = self._current_parameter()
        error = self._validate()
        if error or param is None:
            raise ValueError(error or "No valid parameter selected.")

        min_val = float(self.min_edit.text()) if self.min_edit.text().strip() else None
        max_val = (
            float(self.max_edit.text())
            if param.inputs.max_value and self.max_edit.text().strip()
            else None
        )
        try:
            weight = float(self.weight_edit.text())
        except ValueError:
            weight = 1.0

        return DesignGoal(parameter=param, frequency_range=self._frequency_range(),
                          min_value=min_val, max_value=max_val, weight=weight)

    def accept(self):
        error = self._validate()
        if error:
            QMessageBox.warning(self, "Invalid Design Goal", error)
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
        self.setWindowTitle("Optimization Parameter")
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
            if param.linked_to:
                self.link_to_combo.addItem(param.linked_to, param.linked_to)
                self.link_to_combo.setCurrentIndex(self.link_to_combo.count() - 1)

        if from_source == "ONNX" and source_data:
            self.name_combo = QComboBox()
            self.name_combo.addItems(source_data)
            self.form_layout.addRow("Name:", self.name_combo)
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
            self.form_layout.addRow("Name:", self.name_combo)
            self.type_combo.setCurrentText(OptimizationType.NETLIST_VARIABLE.value)
            self.type_combo.setEnabled(False)
            self.step_spin.setValue(1.0)
            self.use_combo_name = True
            self.name_combo.currentTextChanged.connect(self._refresh_link_targets)
        else:
            self.form_layout.addRow("Name:", self.name_edit)
            self.use_combo_name = False
            self.name_edit.textChanged.connect(self._refresh_link_targets)

        self.form_layout.addRow("Type:", self.type_combo)
        self.form_layout.addRow("Min:", self.min_spin)
        self.form_layout.addRow("Max:", self.max_spin)
        self.form_layout.addRow("Step:", self.step_spin)
        self.form_layout.addRow("Unit:", self.unit_edit)
        self.form_layout.addRow("Link To:", self.link_to_combo)

        self._refresh_link_targets(self.name_combo.currentText() if self.use_combo_name else self.name_edit.text())
        if param and param.linked_to:
            idx = self.link_to_combo.findData(param.linked_to)
            if idx >= 0:
                self.link_to_combo.setCurrentIndex(idx)

        self.link_to_combo.currentIndexChanged.connect(self._on_link_target_changed)
        self._on_link_target_changed()

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
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

    def get_data(self):
        name = self.name_combo.currentText() if self.use_combo_name else self.name_edit.text()
        t_str = self.type_combo.currentText()
        t = OptimizationType(t_str)
        unit = self.unit_edit.text().strip() or None
        linked_to = self.link_to_combo.currentData()
        return OptimizationProperty(
            name,
            t,
            self.min_spin.value(),
            self.max_spin.value(),
            self.step_spin.value() or None,
            unit=unit,
            linked_to=linked_to,
        )

